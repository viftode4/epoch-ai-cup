"""E05: Wavelet + Flight Mode Features — Tabular Ensemble

Replaces weak FFT features with CWT wavelet features (Zaugg et al. 2008)
and adds flight mode segmentation for species discrimination.
Uses core + wavelet + flight_mode + tabular (drops rcs_fft).
Same LGB+XGB+CB ensemble as E02.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

# ── Load & extract features ───────────────────────────────────────
print("Loading data...", flush=True)
train = load_train()
test = load_test()

print("Extracting train features...", flush=True)
train_feats = build_features(train, ["core", "wavelet", "flight_mode", "tabular"])
print("Extracting test features...", flush=True)
test_feats = build_features(test, ["core", "wavelet", "flight_mode", "tabular"])

print(f"Feature count: {train_feats.shape[1]}", flush=True)

# ── Prepare target ────────────────────────────────────────────────
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train["bird_group"])
n_classes = len(CLASSES)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

class_counts = np.bincount(y, minlength=n_classes)
class_weights = len(y) / (n_classes * class_counts)
sample_weights = np.array([class_weights[yi] for yi in y])

# ── 5-fold CV with LGB + XGB + CB ─────────────────────────────────
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), n_classes))
oof_xgb = np.zeros((len(X), n_classes))
oof_cb = np.zeros((len(X), n_classes))
test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_cb = np.zeros((len(X_test), n_classes))

lgb_params = {
    "objective": "multiclass", "num_class": n_classes,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "is_unbalance": True,
}

xgb_params = {
    "objective": "multi:softprob", "num_class": n_classes,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
}

# ── LightGBM ──────────────────────────────────────────────────────
print("\nTraining LightGBM...", flush=True)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx],
                         feature_name=feature_names)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=feature_names,
                       reference=dtrain)
    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(500)])
    oof_lgb[va_idx] = model.predict(X[va_idx])
    test_lgb += model.predict(X_test) / N_FOLDS
    fold_map, _ = compute_map(y[va_idx], oof_lgb[va_idx])
    print(f"  Fold {fold}: mAP = {fold_map:.4f}", flush=True)

lgb_map, lgb_per = compute_map(y, oof_lgb)
print_results(lgb_map, lgb_per, "LightGBM")

# Print top-20 LGB feature importances
imp = model.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
imp_df = imp_df.sort_values("importance", ascending=False).head(20)
print("\nTop-20 LGB features (gain):", flush=True)
for _, row in imp_df.iterrows():
    print(f"  {row.feature:30s}: {row.importance:.0f}", flush=True)

# ── XGBoost ───────────────────────────────────────────────────────
print("\nTraining XGBoost...", flush=True)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx],
                         feature_names=feature_names)
    dval = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=feature_names)
    model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                      evals=[(dval, "val")], early_stopping_rounds=80, verbose_eval=500)
    oof_xgb[va_idx] = model.predict(dval)
    test_xgb += model.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS
    fold_map, _ = compute_map(y[va_idx], oof_xgb[va_idx])
    print(f"  Fold {fold}: mAP = {fold_map:.4f}", flush=True)

xgb_map, xgb_per = compute_map(y, oof_xgb)
print_results(xgb_map, xgb_per, "XGBoost")

# ── CatBoost ──────────────────────────────────────────────────────
print("\nTraining CatBoost...", flush=True)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    cb_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        auto_class_weights="Balanced",
    )
    cb_model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb_model.predict_proba(X[va_idx])
    test_cb += cb_model.predict_proba(X_test) / N_FOLDS
    fold_map, _ = compute_map(y[va_idx], oof_cb[va_idx])
    print(f"  Fold {fold}: mAP = {fold_map:.4f}", flush=True)

cb_map, cb_per = compute_map(y, oof_cb)
print_results(cb_map, cb_per, "CatBoost")

# ── Ensemble weight optimization ──────────────────────────────────
print("\nOptimizing ensemble weights...", flush=True)
best_map = 0
best_w = (1/3, 1/3, 1/3)
for w1 in np.arange(0.15, 0.65, 0.05):
    for w2 in np.arange(0.15, 0.65, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        ens_map, _ = compute_map(y, oof_ens)
        if ens_map > best_map:
            best_map = ens_map
            best_w = (w1, w2, w3)

print(f"Best weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}")

oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E05 Ensemble (wavelet + flight_mode)")

# ── Save OOF for later blending + submission ──────────────────────
np.save("oof_e05.npy", oof_final)
np.save("test_e05.npy", test_final)
print("Saved oof_e05.npy and test_e05.npy for blending", flush=True)

save_submission(test_final, "e05_wavelet_flightmode", cv_map=final_map)
