"""E25D: Generate submission for Config D (no temporal + weakclass).
Quick re-run of just config D from E25.
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

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
BETA = 0.999

TEMPORAL_OVERFIT = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring",
    "hour_bin_3h",
]

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

print("Building features (no temporal + weakclass)...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in TEMPORAL_OVERFIT]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
print(f"  Features: {train_feats.shape[1]}", flush=True)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
sample_weights = np.array([class_w[yi] for yi in y])

lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
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
folds = list(skf.split(np.zeros(len(y)), y))

oof_lgb = np.zeros((len(X), N_CLASSES))
oof_xgb = np.zeros((len(X), N_CLASSES))
oof_cb = np.zeros((len(X), N_CLASSES))
test_lgb = np.zeros((len(X_test), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))
test_cb = np.zeros((len(X_test), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(X_va)
    test_lgb += mdl.predict(X_test) / N_FOLDS

    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
    mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                    evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = mdl.predict(dval_xgb)
    test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=fn)) / N_FOLDS

    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb[va_idx] = cb.predict_proba(X_va)
    test_cb += cb.predict_proba(X_test) / N_FOLDS

    print(f"  Fold {fold} done", flush=True)

# Optimize weights
best_map = 0
best_w = None
for w1 in np.arange(0.05, 0.50, 0.05):
    for w2 in np.arange(0.05, 0.50, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        m, _ = compute_map(y, oof_ens)
        if m > best_map:
            best_map = m
            best_w = (w1, w2, w3)

oof = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_pred = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

# Logit adjustment
priors = counts / counts.sum()
tau = np.zeros(N_CLASSES)
best_map_adj = best_map
for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_t = tau[c]
        best_m = best_map_adj
        for t in np.arange(-0.5, 1.51, 0.02):
            tau[c] = t
            adj = priors ** (-tau)
            a = oof * adj[None, :]
            a = a / a.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, a)
            if m > best_m:
                best_m = m
                best_t = t
        tau[c] = best_t
        if best_m > best_map_adj:
            best_map_adj = best_m
            improved = True
    print(f"  Logit adj round {iteration+1}: {best_map_adj:.4f}", flush=True)
    if not improved:
        break

adj = priors ** (-tau)
oof_adj = oof * adj[None, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
test_adj = test_pred * adj[None, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

final_map, final_per = compute_map(y, oof_adj)
print_results(final_map, final_per, "E25D: No temporal + weakclass + logit adj")

# Test distribution
preds = test_adj.argmax(axis=1)
dist = np.bincount(preds, minlength=N_CLASSES)
print(f"\nTest predictions: {dict(zip(CLASSES, dist))}", flush=True)

# Save
np.save(ROOT / "oof_e25d.npy", oof_adj)
np.save(ROOT / "test_e25d.npy", test_adj)
save_submission(test_adj, "e25d_no_temporal_weakclass", cv_map=final_map)
print("\nDone!", flush=True)
