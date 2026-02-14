"""E10: Best Tree Ensemble with OOF for Stacking

Produces the tree ensemble OOF predictions needed for the meta-learner.
Uses the best feature config from ablation: core+fft+tab+tgt+flight (105 feats)
with LGB+XGB+CB triple ensemble and optimized weights.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

N_CLASSES = len(CLASSES)
N_FOLDS = 5

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── Main ──────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Best feature config from ablation
FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print(f"Extracting features: {FEATURE_SETS}", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
print(f"  Feature count: {train_feats.shape[1]}", flush=True)

print("Extracting test features...", flush=True)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

class_counts = np.bincount(y, minlength=N_CLASSES)
class_weights = len(y) / (N_CLASSES * class_counts)
sample_weights = np.array([class_weights[yi] for yi in y])

# Model params — ALL ON GPU
lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "is_unbalance": True,
    "device": "gpu",
}

xgb_params = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), N_CLASSES))
oof_xgb = np.zeros((len(X), N_CLASSES))
oof_cb = np.zeros((len(X), N_CLASSES))
test_lgb = np.zeros((len(X_test), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))
test_cb = np.zeros((len(X_test), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold} ---", flush=True)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(X_va)
    test_lgb += mdl.predict(X_test) / N_FOLDS

    # XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                    evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = mdl.predict(dval_xgb)
    test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

    # CatBoost (GPU)
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        auto_class_weights="Balanced",
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X_va)
    test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Per-fold ensemble check
    oof_ens = 0.25 * oof_lgb[va_idx] + 0.30 * oof_xgb[va_idx] + 0.45 * oof_cb[va_idx]
    fold_map, _ = compute_map(y_va, oof_ens)
    print(f"  Fold {fold} ensemble mAP: {fold_map:.4f}", flush=True)

# ── Optimize ensemble weights on OOF ─────────────────────────────
print("\nOptimizing ensemble weights...", flush=True)
best_map = 0
best_w = None
for w1 in np.arange(0.10, 0.50, 0.05):
    for w2 in np.arange(0.10, 0.50, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.10:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        m, _ = compute_map(y, oof_ens)
        if m > best_map:
            best_map = m
            best_w = (w1, w2, w3)

print(f"Best weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}",
      flush=True)

oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E10 Tree Ensemble (best feats)")

np.save("oof_e10.npy", oof_final)
np.save("test_e10.npy", test_final)
print("Saved oof_e10.npy and test_e10.npy for stacking", flush=True)

save_submission(test_final, "e10_tree_ensemble", cv_map=final_map)
