"""Adversarial Validation: Which features distinguish train from test?

Train a binary classifier (train=0, test=1).
- High AUC = large distribution shift between train and test.
- Feature importances reveal WHICH features cause the shift.
- Features with high adversarial importance should be removed or downweighted.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features

ROOT = Path(__file__).resolve().parent.parent

TEMPORAL_OVERFIT = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring",
    "hour_bin_3h",
]

# Also the 5 weakclass temporal features we found leaking
WEAKCLASS_TEMPORAL = [
    "is_oct_nov", "migration_alt", "migration_speed",
    "is_night", "night_high_alt",
]

ALL_TEMPORAL = TEMPORAL_OVERFIT + WEAKCLASS_TEMPORAL

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

# Build same feature set as E25D
print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# === Test 1: With temporal features still present (to see their impact) ===
print("\n" + "=" * 60, flush=True)
print("TEST 1: Adversarial validation WITH temporal features", flush=True)
print("=" * 60, flush=True)

# Keep only the OLD temporal overfit list (not the new 5)
keep_old = [c for c in train_feats.columns if c not in TEMPORAL_OVERFIT]
X_train_old = train_feats[keep_old].values.astype(np.float32)
X_test_old = test_feats[keep_old].values.astype(np.float32)
fn_old = list(train_feats[keep_old].columns)

# Combine
X_adv = np.vstack([X_train_old, X_test_old])
y_adv = np.array([0] * len(X_train_old) + [1] * len(X_test_old))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_adv))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv, y_adv)):
    dtrain = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx])
    dval = lgb.Dataset(X_adv[va_idx], label=y_adv[va_idx])
    mdl = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
         "device": "gpu"},
        dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    oof[va_idx] = mdl.predict(X_adv[va_idx])

auc1 = roc_auc_score(y_adv, oof)
print(f"  AUC (old filter, 5 temporal leaks remain): {auc1:.4f}", flush=True)
print(f"  (0.5 = no shift, 1.0 = complete shift)", flush=True)

# Feature importances from a full model
dtrain_full = lgb.Dataset(X_adv, label=y_adv)
mdl_full = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
     "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
     "device": "gpu", "n_jobs": -1},
    dtrain_full, num_boost_round=300,
)
imp = mdl_full.feature_importance(importance_type="gain")
feat_imp = sorted(zip(fn_old, imp), key=lambda x: -x[1])

print("\n  Top 30 features distinguishing train from test:")
for i, (name, importance) in enumerate(feat_imp[:30]):
    is_temporal = " *** TEMPORAL LEAK ***" if name in WEAKCLASS_TEMPORAL else ""
    print(f"  {i+1:3d}. {name:30s}: {importance:8.1f}{is_temporal}", flush=True)


# === Test 2: With ALL temporal features removed ===
print("\n" + "=" * 60, flush=True)
print("TEST 2: Adversarial validation WITHOUT temporal features", flush=True)
print("=" * 60, flush=True)

keep_all = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
X_train_clean = train_feats[keep_all].values.astype(np.float32)
X_test_clean = test_feats[keep_all].values.astype(np.float32)
fn_clean = list(train_feats[keep_all].columns)

X_adv2 = np.vstack([X_train_clean, X_test_clean])
y_adv2 = np.array([0] * len(X_train_clean) + [1] * len(X_test_clean))

oof2 = np.zeros(len(y_adv2))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv2, y_adv2)):
    dtrain = lgb.Dataset(X_adv2[tr_idx], label=y_adv2[tr_idx])
    dval = lgb.Dataset(X_adv2[va_idx], label=y_adv2[va_idx])
    mdl = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
         "device": "gpu"},
        dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    oof2[va_idx] = mdl.predict(X_adv2[va_idx])

auc2 = roc_auc_score(y_adv2, oof2)
print(f"  AUC (all temporal removed): {auc2:.4f}", flush=True)

mdl_full2 = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
     "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
     "device": "gpu", "n_jobs": -1},
    lgb.Dataset(X_adv2, label=y_adv2), num_boost_round=300,
)
imp2 = mdl_full2.feature_importance(importance_type="gain")
feat_imp2 = sorted(zip(fn_clean, imp2), key=lambda x: -x[1])

print("\n  Top 30 features distinguishing train from test (after cleanup):")
for i, (name, importance) in enumerate(feat_imp2[:30]):
    print(f"  {i+1:3d}. {name:30s}: {importance:8.1f}", flush=True)


# === Test 3: Feature-level distribution comparison ===
print("\n" + "=" * 60, flush=True)
print("TEST 3: Feature distribution shift (KS test)", flush=True)
print("=" * 60, flush=True)

from scipy.stats import ks_2samp

ks_results = []
for i, name in enumerate(fn_clean):
    stat, pval = ks_2samp(X_train_clean[:, i], X_test_clean[:, i])
    ks_results.append((name, stat, pval))

ks_results.sort(key=lambda x: -x[1])
print("\n  Top 20 features by KS statistic (distribution shift):")
for i, (name, stat, pval) in enumerate(ks_results[:20]):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {i+1:3d}. {name:30s}: KS={stat:.4f}  p={pval:.2e} {sig}", flush=True)


# === Test 4: Adversarial validation with ONLY core trajectory features ===
print("\n" + "=" * 60, flush=True)
print("TEST 4: Adversarial validation with ONLY core trajectory features", flush=True)
print("=" * 60, flush=True)

core_feats_train = build_features(train_df, feature_sets=["core", "rcs_fft"])
core_feats_test = build_features(test_df, feature_sets=["core", "rcs_fft"])

X_core_train = core_feats_train.values.astype(np.float32)
X_core_test = core_feats_test.values.astype(np.float32)
fn_core = list(core_feats_train.columns)

X_adv3 = np.vstack([X_core_train, X_core_test])
y_adv3 = np.array([0] * len(X_core_train) + [1] * len(X_core_test))

oof3 = np.zeros(len(y_adv3))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv3, y_adv3)):
    dtrain = lgb.Dataset(X_adv3[tr_idx], label=y_adv3[tr_idx])
    dval = lgb.Dataset(X_adv3[va_idx], label=y_adv3[va_idx])
    mdl = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
         "device": "gpu"},
        dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    oof3[va_idx] = mdl.predict(X_adv3[va_idx])

auc3 = roc_auc_score(y_adv3, oof3)
print(f"  AUC (core + rcs_fft only): {auc3:.4f}", flush=True)

mdl_full3 = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
     "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
     "device": "gpu", "n_jobs": -1},
    lgb.Dataset(X_adv3, label=y_adv3), num_boost_round=300,
)
imp3 = mdl_full3.feature_importance(importance_type="gain")
feat_imp3 = sorted(zip(fn_core, imp3), key=lambda x: -x[1])

print("\n  Top 15 core features distinguishing train from test:")
for i, (name, importance) in enumerate(feat_imp3[:15]):
    print(f"  {i+1:3d}. {name:30s}: {importance:8.1f}", flush=True)


# === Summary ===
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  AUC with 5 temporal leaks:     {auc1:.4f}", flush=True)
print(f"  AUC all temporal removed:       {auc2:.4f}", flush=True)
print(f"  AUC core+rcs_fft only:          {auc3:.4f}", flush=True)
print(f"  Features in clean set:          {len(fn_clean)}", flush=True)
print(f"  Features in core-only set:      {len(fn_core)}", flush=True)
print(flush=True)

if auc3 > 0.6:
    print("  WARNING: Even core trajectory features show distribution shift!", flush=True)
    print("  This suggests inherent seasonal differences in bird flight patterns.", flush=True)
    print("  Features to consider removing: those with KS > 0.1 AND high adversarial importance.", flush=True)
elif auc3 < 0.55:
    print("  GOOD: Core trajectory features are mostly stable across train/test.", flush=True)
    print("  The shift is primarily from tabular/temporal features.", flush=True)

print("\nDone!", flush=True)
