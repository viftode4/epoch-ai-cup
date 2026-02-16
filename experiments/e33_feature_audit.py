"""E33: Feature Audit -- Which Features Are Still Leaking?

1. Build clean features (same as E32), run adversarial validation
2. Rank all features by adversarial importance (gain) AND KS statistic
3. Get classification importance from LGB model
4. Cross-reference: high-adversarial + low-classification = leak candidates
5. Iterative pruning: remove top-N adversarial features, retrain LGB baseline
6. Print recommendation table: keep/prune/flag
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

# ======================================================================
# Data + Features
# ======================================================================
print("=" * 60, flush=True)
print("E33 FEATURE AUDIT", flush=True)
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

print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove ALL_TEMPORAL
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

X_train = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"  Clean features: {len(fn)}", flush=True)

# ======================================================================
# Step 1: Adversarial Validation
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("STEP 1: Adversarial Validation (train=0, test=1)", flush=True)
print("=" * 60, flush=True)

X_adv = np.vstack([X_train, X_test])
y_adv = np.array([0] * len(X_train) + [1] * len(X_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_adv = np.zeros(len(y_adv))

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
    oof_adv[va_idx] = mdl.predict(X_adv[va_idx])

adv_auc = roc_auc_score(y_adv, oof_adv)
print(f"  Adversarial AUC: {adv_auc:.4f}  (0.5=no shift, 1.0=complete shift)", flush=True)

# Full adversarial model for importances
mdl_adv = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
     "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
     "device": "gpu", "n_jobs": -1},
    lgb.Dataset(X_adv, label=y_adv), num_boost_round=300,
)
adv_imp = mdl_adv.feature_importance(importance_type="gain")
adv_rank = {fn[i]: i + 1 for i, _ in enumerate(
    sorted(range(len(adv_imp)), key=lambda x: -adv_imp[x])
)}
adv_score = {fn[i]: adv_imp[i] for i in range(len(fn))}

# ======================================================================
# Step 2: KS Statistics
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("STEP 2: KS Statistics (feature distribution shift)", flush=True)
print("=" * 60, flush=True)

ks_results = {}
for i, name in enumerate(fn):
    stat, pval = ks_2samp(X_train[:, i], X_test[:, i])
    ks_results[name] = (stat, pval)

ks_sorted = sorted(ks_results.items(), key=lambda x: -x[1][0])
print("\n  Top 20 features by KS statistic:", flush=True)
for i, (name, (stat, pval)) in enumerate(ks_sorted[:20]):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {i+1:3d}. {name:30s}: KS={stat:.4f}  p={pval:.2e} {sig}", flush=True)

# ======================================================================
# Step 3: Classification Importance (LGB only, for speed)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("STEP 3: Classification Importance (LGB multiclass)", flush=True)
print("=" * 60, flush=True)

# Train a single LGB model on full training data for importances
dtrain_cls = lgb.Dataset(X_train, label=y, weight=sample_weights, feature_name=fn)
mdl_cls = lgb.train(LGB_PARAMS, dtrain_cls, num_boost_round=500)
cls_imp = mdl_cls.feature_importance(importance_type="gain")
cls_rank = {fn[i]: i + 1 for i, _ in enumerate(
    sorted(range(len(cls_imp)), key=lambda x: -cls_imp[x])
)}
cls_score = {fn[i]: cls_imp[i] for i in range(len(fn))}

# ======================================================================
# Step 4: Cross-Reference Table
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("STEP 4: Feature Recommendation Table", flush=True)
print("=" * 60, flush=True)

# Decision logic
recommendations = {}
prune_list = []
flag_list = []

for name in fn:
    ar = adv_rank[name]
    cr = cls_rank[name]
    ks_stat = ks_results[name][0]

    if ar <= 10 and cr > 50:
        rec = "PRUNE"
        prune_list.append(name)
    elif ar <= 10 and cr <= 20:
        rec = "FLAG"
        flag_list.append(name)
    elif ar <= 20 and cr > 80:
        rec = "PRUNE"
        prune_list.append(name)
    else:
        rec = "keep"
    recommendations[name] = rec

# Print sorted by adversarial rank
print(f"\n  {'Feature':<30s} {'AdvRank':>8s} {'ClsRank':>8s} {'KS':>7s} {'Rec':>8s}", flush=True)
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*8}", flush=True)
for name in sorted(fn, key=lambda x: adv_rank[x]):
    ar = adv_rank[name]
    cr = cls_rank[name]
    ks_stat = ks_results[name][0]
    rec = recommendations[name]
    if ar <= 30 or rec != "keep":
        marker = " ***" if rec == "PRUNE" else " !!" if rec == "FLAG" else ""
        print(f"  {name:<30s} {ar:>8d} {cr:>8d} {ks_stat:>7.4f} {rec:>8s}{marker}", flush=True)

print(f"\n  PRUNE ({len(prune_list)}): {prune_list}", flush=True)
print(f"  FLAG  ({len(flag_list)}): {flag_list}", flush=True)

# ======================================================================
# Step 5: Iterative Pruning Test (LGB only, for speed)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("STEP 5: Iterative Pruning Test", flush=True)
print("=" * 60, flush=True)

# Get features sorted by adversarial importance (most suspicious first)
adv_sorted = sorted(fn, key=lambda x: -adv_score.get(x, 0))

skf_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds_cls = list(skf_cls.split(np.zeros(len(y)), y))


def quick_lgb_cv(X_in, y_in, weights, feature_names, folds):
    """Quick LGB-only CV for pruning test."""
    oof = np.zeros((len(y_in), N_CLASSES))
    for tr_idx, va_idx in folds:
        dtrain = lgb.Dataset(X_in[tr_idx], label=y_in[tr_idx],
                             weight=weights[tr_idx], feature_name=feature_names)
        dval = lgb.Dataset(X_in[va_idx], label=y_in[va_idx],
                           feature_name=feature_names, reference=dtrain)
        mdl = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va_idx] = mdl.predict(X_in[va_idx])
    m, _ = compute_map(y_in, oof)
    return m


# Also compute adversarial AUC for each pruning level
def quick_adv_auc(X_tr, X_te):
    """Quick adversarial AUC (single model, no CV)."""
    X_c = np.vstack([X_tr, X_te])
    y_c = np.array([0] * len(X_tr) + [1] * len(X_te))
    dtrain = lgb.Dataset(X_c, label=y_c)
    mdl = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "max_depth": 5, "verbose": -1, "seed": 42,
         "device": "gpu", "n_jobs": -1, "bagging_fraction": 0.8,
         "bagging_freq": 1},
        dtrain, num_boost_round=200,
    )
    preds = mdl.predict(X_c)
    return roc_auc_score(y_c, preds)


# Baseline (no pruning)
baseline_map = quick_lgb_cv(X_train, y, sample_weights, fn, folds_cls)
baseline_auc = quick_adv_auc(X_train, X_test)
print(f"  Baseline: {len(fn)} features, mAP={baseline_map:.4f}, adv_AUC={baseline_auc:.4f}", flush=True)

# Prune top-N adversarial features
for n_prune in [5, 10, 15, 20]:
    to_remove = set(adv_sorted[:n_prune])
    keep_idx = [i for i, f in enumerate(fn) if f not in to_remove]
    fn_pruned = [fn[i] for i in keep_idx]

    X_train_pruned = X_train[:, keep_idx]
    X_test_pruned = X_test[:, keep_idx]

    prune_map = quick_lgb_cv(X_train_pruned, y, sample_weights, fn_pruned, folds_cls)
    prune_auc = quick_adv_auc(X_train_pruned, X_test_pruned)

    delta = prune_map - baseline_map
    print(f"  Prune top-{n_prune:2d}: {len(fn_pruned)} features, "
          f"mAP={prune_map:.4f} ({delta:+.4f}), adv_AUC={prune_auc:.4f}", flush=True)

# Also test pruning only PRUNE-recommended features
if prune_list:
    to_remove = set(prune_list)
    keep_idx = [i for i, f in enumerate(fn) if f not in to_remove]
    fn_pruned = [fn[i] for i in keep_idx]
    X_train_pruned = X_train[:, keep_idx]
    X_test_pruned = X_test[:, keep_idx]

    prune_map = quick_lgb_cv(X_train_pruned, y, sample_weights, fn_pruned, folds_cls)
    prune_auc = quick_adv_auc(X_train_pruned, X_test_pruned)
    delta = prune_map - baseline_map
    print(f"  Prune REC'd ({len(prune_list)}): {len(fn_pruned)} features, "
          f"mAP={prune_map:.4f} ({delta:+.4f}), adv_AUC={prune_auc:.4f}", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E33 FEATURE AUDIT -- SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Clean features: {len(fn)}", flush=True)
print(f"  Adversarial AUC (5-fold): {adv_auc:.4f}", flush=True)
print(f"  PRUNE candidates: {len(prune_list)}", flush=True)
print(f"  FLAG candidates: {len(flag_list)}", flush=True)
print(f"  Baseline LGB mAP: {baseline_map:.4f}", flush=True)
print(flush=True)
print("  Recommended feature list saved as analysis only.", flush=True)
print("  Use pruning results to decide if further removal helps.", flush=True)
print("\nDone!", flush=True)
