"""E32: Honest Baseline With Error Bars

Clean baseline with NO tricks:
- ALL 23 temporal features removed (from src.features.ALL_TEMPORAL)
- Fixed ensemble weights (0.33/0.33/0.34) -- NO optimization
- NO logit adjustment or post-processing
- Effective number class weights (beta=0.999)
- RepeatedStratifiedKFold (5x5) for variance estimation
- LOMO (4 folds) as lower bound
- Bootstrap 95% CI on primary SKF OOF

This is the REAL number. Expected ~0.68-0.69 SKF CV.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map, print_results, bootstrap_map_ci
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

# Fixed ensemble weights -- NO optimization
W_LGB = 0.33
W_XGB = 0.33
W_CB = 0.34

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}


def train_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, fold_label):
    """Train LGB+XGB+CB on a single fold. Returns (oof_pred, test_pred)."""
    # LGB
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    mdl_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb = mdl_lgb.predict(X_va)
    test_lgb = mdl_lgb.predict(X_test) if X_test is not None else None

    # XGB
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
    mdl_xgb = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=2000,
                         evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb = mdl_xgb.predict(dval_xgb)
    test_xgb = mdl_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    # CatBoost
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    # Fixed-weight ensemble
    oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    if X_test is not None:
        test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb
    else:
        test_ens = None

    fold_map, _ = compute_map(y_va, oof_ens)
    print(f"  {fold_label}: mAP={fold_map:.4f} (n={len(y_va)})", flush=True)
    return oof_ens, test_ens


# ======================================================================
# Data loading + feature building
# ======================================================================
print("=" * 60, flush=True)
print("E32 HONEST BASELINE", flush=True)
print("=" * 60, flush=True)

print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Effective number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# Build features -- all feature sets, then drop ALL_TEMPORAL
print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)

print(f"  Features: {len(fn)} ({len(ALL_TEMPORAL)} temporal removed)", flush=True)
print(f"  Weights: LGB={W_LGB} XGB={W_XGB} CB={W_CB} (FIXED, no optimization)", flush=True)

# ======================================================================
# 1. RepeatedStratifiedKFold (5 folds x 5 repeats)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("RepeatedStratifiedKFold (5 folds x 5 repeats = 25 splits)", flush=True)
print("=" * 60, flush=True)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
all_folds = list(rskf.split(np.zeros(len(y)), y))

# Group folds by repeat
repeat_maps = []
oof_primary = np.zeros((len(y), N_CLASSES))
test_primary = np.zeros((len(X_test), N_CLASSES))

for repeat_idx in range(5):
    start = repeat_idx * 5
    end = start + 5
    repeat_folds = all_folds[start:end]

    oof_repeat = np.zeros((len(y), N_CLASSES))
    test_repeat = np.zeros((len(X_test), N_CLASSES))

    print(f"\n--- Repeat {repeat_idx} (seed offset) ---", flush=True)
    for fold_idx, (tr_idx, va_idx) in enumerate(repeat_folds):
        # Only generate test predictions for repeat 0
        produce_test = (repeat_idx == 0)
        oof_fold, test_fold = train_fold(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            sample_weights[tr_idx],
            X_test if produce_test else None,
            fn,
            f"Rep{repeat_idx} Fold{fold_idx}",
        )
        oof_repeat[va_idx] = oof_fold
        if produce_test and test_fold is not None:
            test_repeat += test_fold / 5

    repeat_map, _ = compute_map(y, oof_repeat)
    repeat_maps.append(repeat_map)
    print(f"  Repeat {repeat_idx} mAP: {repeat_map:.4f}", flush=True)

    if repeat_idx == 0:
        oof_primary = oof_repeat.copy()
        test_primary = test_repeat.copy()

rskf_mean = np.mean(repeat_maps)
rskf_std = np.std(repeat_maps)
primary_map, primary_per = compute_map(y, oof_primary)

# Bootstrap CI on primary OOF
print("\nComputing bootstrap 95% CI on primary OOF...", flush=True)
bs = bootstrap_map_ci(y, oof_primary, n_bootstrap=2000, ci=0.95, seed=42)

# ======================================================================
# 2. LOMO (Leave-One-Month-Out, 4 folds)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("LOMO -- Leave-One-Month-Out (4 folds)", flush=True)
print("=" * 60, flush=True)

ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = ts.dt.month.values
unique_months = sorted(np.unique(train_months))
print(f"  Train months: {unique_months}", flush=True)

oof_lomo = np.zeros((len(y), N_CLASSES))
test_lomo = np.zeros((len(X_test), N_CLASSES))
n_lomo = len(unique_months)

for i, m in enumerate(unique_months):
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    oof_fold, test_fold = train_fold(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test, fn,
        f"LOMO Month {m}",
    )
    oof_lomo[va_idx] = oof_fold
    test_lomo += test_fold / n_lomo

lomo_map, lomo_per = compute_map(y, oof_lomo)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E32 HONEST BASELINE -- SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Features: {len(fn)} ({len(ALL_TEMPORAL)} temporal removed)", flush=True)
print(f"  Weights: LGB={W_LGB} XGB={W_XGB} CB={W_CB} (FIXED, no optimization)", flush=True)
print(f"  Post-processing: NONE", flush=True)
print(flush=True)
print(f"  RepeatedSKF (5x5):   {rskf_mean:.4f} +/- {rskf_std:.4f}", flush=True)
print(f"  Primary SKF (rep 0): {primary_map:.4f}  [95% CI: {bs['ci_lo']:.4f} - {bs['ci_hi']:.4f}]", flush=True)
print(f"  Bootstrap mean/std:  {bs['mean']:.4f} +/- {bs['std']:.4f}", flush=True)
print(f"  LOMO (lower bound):  {lomo_map:.4f}", flush=True)
print(flush=True)

print("  Per-class APs with 95% CIs:", flush=True)
print(f"  {'Class':<15s} {'AP':>7s} {'CI_lo':>7s} {'CI_hi':>7s}", flush=True)
for cls in CLASSES:
    ap = primary_per[cls]
    mean_bs, lo, hi = bs["per_class"][cls]
    marker = " <-- weak" if ap < 0.6 else ""
    print(f"  {cls:<15s} {ap:>7.4f} {lo:>7.4f} {hi:>7.4f}{marker}", flush=True)

print(flush=True)
print("  LOMO per-class:", flush=True)
print(f"  {'Class':<15s} {'SKF':>7s} {'LOMO':>7s} {'Delta':>7s}", flush=True)
for cls in CLASSES:
    s = primary_per.get(cls, 0)
    l = lomo_per.get(cls, 0)
    print(f"  {cls:<15s} {s:>7.4f} {l:>7.4f} {l - s:>+7.4f}", flush=True)

# Test prediction distribution
print("\n  Test prediction class distribution:", flush=True)
pred_classes = test_primary.argmax(axis=1)
dist = np.bincount(pred_classes, minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {dist[i]}", flush=True)

# ======================================================================
# Save
# ======================================================================
np.save(ROOT / "oof_e32.npy", oof_primary)
np.save(ROOT / "test_e32.npy", test_primary)
print(f"\n  Saved: oof_e32.npy, test_e32.npy", flush=True)

save_submission(test_primary, "e32_honest", cv_map=primary_map)
print("\nDone!", flush=True)
