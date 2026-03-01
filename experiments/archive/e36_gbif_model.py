"""E36: GBIF Data-Driven Model Improvement

Two-pronged approach using 96K real GBIF bird observations from Eemshaven:

Part A: GBIF-informed Bayesian post-processing on E32 predictions
  - Replace E35's hand-crafted ecological priors with real GBIF seasonal indices
  - P_radar_eco(c|m) = P_train(c) * SI_gbif(c, m), then normalize
  - Shared months (Sep, Oct): use training distribution directly
  - Unseen months (Feb, May, Dec): use GBIF seasonal index to adjust

Part B: Retrain model with 10 GBIF seasonal features
  - 9 features: gbif_si_{class} = GBIF seasonal index for sample's month
  - 1 feature: month_gbif_diversity = Shannon entropy of GBIF proportions
  - These encode EXTERNAL knowledge (not training label patterns)
  - Total: 114 base + 10 GBIF = 124 features, 23 temporal still removed

Part A+B: Combined post-processing on retrained model predictions
"""
import sys
import numpy as np
import pandas as pd
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
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

# Fixed ensemble weights (same as E32)
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


def bayesian_adjust(preds, p_eco, p_train, alpha):
    """Apply Bayesian prior adjustment: adjusted[c] = pred[c] * (p_eco[c] / p_train[c])^alpha."""
    if alpha == 0:
        return preds.copy()
    p_train_safe = np.maximum(p_train, 1e-10)
    ratio = (p_eco / p_train_safe) ** alpha
    adjusted = preds * ratio[np.newaxis, :]
    row_sums = np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-10)
    return adjusted / row_sums


# ======================================================================
# Step 1: Load GBIF data and compute seasonal indices
# ======================================================================
print("=" * 60, flush=True)
print("E36 GBIF DATA-DRIVEN MODEL", flush=True)
print("=" * 60, flush=True)

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
print(f"\n  GBIF data: {gbif['total_gbif'].sum():,} total observations, 12 months", flush=True)

# Compute Seasonal Index: SI(class, month) = count(class, month) / mean(count(class, all_months))
# For Clutter: SI = 1.0 (no GBIF data, non-biological)
gbif_si = {}  # {month: np.array of shape (N_CLASSES,)}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0  # no seasonal adjustment for Clutter
        else:
            class_counts = gbif[cls].values  # all 12 months
            class_mean = class_counts.mean()
            if class_mean > 0:
                si[i] = row[cls] / class_mean
            else:
                si[i] = 1.0
    gbif_si[m] = si

# Print seasonal index table
print("\n  GBIF SEASONAL INDICES (SI = month_count / yearly_mean):", flush=True)
# Header: show key months
key_months = [1, 2, 5, 9, 10, 12]
header = f"  {'Class':<15s}"
for m in key_months:
    header += f" {m:>5d}"
print(header, flush=True)
for i, cls in enumerate(CLASSES):
    line = f"  {cls:<15s}"
    for m in key_months:
        line += f" {gbif_si[m][i]:>5.2f}"
    print(line, flush=True)

# ======================================================================
# Step 2: Load data + E32 predictions
# ======================================================================
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Training class prior
p_train = counts.astype(float) / counts.sum()

# Effective number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# Timestamps / months
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_train_months = sorted(np.unique(train_months))
unique_test_months = sorted(np.unique(test_months))

print(f"  Train months: {unique_train_months}", flush=True)
print(f"  Test months:  {unique_test_months}", flush=True)
unseen_months = [m for m in unique_test_months if m not in unique_train_months]
print(f"  Unseen months: {unseen_months}", flush=True)
n_unseen = sum(test_months == m for m in unseen_months).sum() if unseen_months else 0
print(f"  Unseen test samples: {n_unseen}/{len(test_months)} ({100*n_unseen/len(test_months):.1f}%)", flush=True)

# Load E32 OOF/test predictions
oof_e32 = np.load(ROOT / "oof_e32.npy")
test_e32 = np.load(ROOT / "test_e32.npy")
print(f"  Loaded E32: oof {oof_e32.shape}, test {test_e32.shape}", flush=True)

# Per-month training priors (for shared months)
train_month_priors = {}
for m in unique_train_months:
    mask = train_months == m
    m_counts = np.bincount(y[mask], minlength=N_CLASSES).astype(float)
    m_total = m_counts.sum()
    train_month_priors[m] = (m_counts + 0.5) / (m_total + 0.5 * N_CLASSES)

# ======================================================================
# Part A: GBIF Post-Processing on E32 Predictions
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART A: GBIF POST-PROCESSING (on E32 predictions)", flush=True)
print("=" * 60, flush=True)

# Build GBIF-informed priors: P_radar_eco(c|m) = P_train(c) * SI_gbif(c, m), then normalize
gbif_priors = {}  # {month: normalized prior vector}
print("\n  GBIF-informed priors per month:", flush=True)
header = f"  {'Month':>5s} {'Source':>8s}"
for cls in CLASSES:
    header += f" {cls[:5]:>6s}"
print(header, flush=True)

for m in sorted(set(unique_train_months + unique_test_months)):
    if m in unique_train_months:
        # Shared month: use training distribution directly (no GBIF adjustment needed)
        gbif_priors[m] = train_month_priors[m]
        source = "train"
    else:
        # Unseen month: use P_train * SI_gbif, then normalize
        raw = p_train * gbif_si[m]
        raw = np.maximum(raw, 1e-6)  # floor to avoid zeros
        gbif_priors[m] = raw / raw.sum()
        source = "GBIF"

    line = f"  {m:5d} {source:>8s}"
    for i in range(N_CLASSES):
        line += f" {gbif_priors[m][i]:>6.3f}"
    print(line, flush=True)

# Alpha sweep on OOF (shared months Sep=9, Oct=10)
ALPHAS = [0, 0.1, 0.25, 0.5, 0.75, 1.0]

shared_mask = np.isin(train_months, [9, 10])
y_shared = y[shared_mask]
oof_shared = oof_e32[shared_mask]
months_shared = train_months[shared_mask]

baseline_shared_map, _ = compute_map(y_shared, oof_shared)
baseline_full_map, baseline_full_per = compute_map(y, oof_e32)

print(f"\n  OOF shared months (Sep, Oct) -- N={shared_mask.sum()}:", flush=True)
print(f"  alpha=0.00 (baseline): mAP={baseline_shared_map:.4f}", flush=True)

best_alpha_a = 0
best_map_a = baseline_shared_map

for alpha in ALPHAS:
    if alpha == 0:
        continue
    adjusted = oof_shared.copy()
    for m in [9, 10]:
        m_mask = months_shared == m
        if m_mask.sum() > 0:
            adjusted[m_mask] = bayesian_adjust(oof_shared[m_mask], gbif_priors[m], p_train, alpha)
    adj_map, _ = compute_map(y_shared, adjusted)
    delta = adj_map - baseline_shared_map
    print(f"  alpha={alpha:.2f}: mAP={adj_map:.4f} (delta: {delta:+.4f})", flush=True)
    if adj_map > best_map_a:
        best_map_a = adj_map
        best_alpha_a = alpha

print(f"\n  Best alpha (shared months): {best_alpha_a:.2f} -> mAP={best_map_a:.4f}", flush=True)

# Full OOF validation (all training months)
print(f"\n  Full OOF (all training months):", flush=True)
print(f"  alpha=0.00 (baseline): mAP={baseline_full_map:.4f}", flush=True)

for alpha in ALPHAS:
    if alpha == 0:
        continue
    adjusted = oof_e32.copy()
    for m in unique_train_months:
        m_mask = train_months == m
        if m_mask.sum() > 0:
            adjusted[m_mask] = bayesian_adjust(oof_e32[m_mask], gbif_priors[m], p_train, alpha)
    adj_map, _ = compute_map(y, adjusted)
    delta = adj_map - baseline_full_map
    print(f"  alpha={alpha:.2f}: mAP={adj_map:.4f} (delta: {delta:+.4f})", flush=True)

# Apply Part A to test predictions
print(f"\n  Applying GBIF post-processing to test (alpha={best_alpha_a:.2f}):", flush=True)
test_a = test_e32.copy()
if best_alpha_a > 0:
    for m in unique_test_months:
        m_mask = test_months == m
        if m_mask.sum() > 0:
            test_a[m_mask] = bayesian_adjust(test_e32[m_mask], gbif_priors[m], p_train, best_alpha_a)

# Show test prediction changes per month
print(f"\n  {'Month':>5s} {'N':>5s} {'Unseen':>6s}  {'Top-3 Before':30s}  {'Top-3 After':30s}", flush=True)
for m in sorted(unique_test_months):
    m_mask = test_months == m
    n_m = m_mask.sum()
    unseen = "YES" if m not in unique_train_months else "no"

    def top3_str(preds):
        mean_p = preds.mean(axis=0)
        order = np.argsort(mean_p)[::-1]
        return " ".join(f"{CLASSES[i][:5]}:{mean_p[i]:.3f}" for i in order[:3])

    before = top3_str(test_e32[m_mask])
    after = top3_str(test_a[m_mask])
    print(f"  {m:5d} {n_m:5d} {unseen:>6s}  {before:30s}  {after:30s}", flush=True)

# ======================================================================
# Part B: Retrain Model with GBIF Seasonal Features
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART B: RETRAIN MODEL WITH GBIF FEATURES", flush=True)
print("=" * 60, flush=True)

# Build base features (same as E32)
print("\nBuilding base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

print(f"  Base features: {len(train_feats.columns)} ({len(ALL_TEMPORAL)} temporal removed)", flush=True)

# Add GBIF seasonal features
print("\nAdding GBIF seasonal features...", flush=True)

# For each sample, look up its month and get the GBIF SI values
for i, cls in enumerate(CLASSES):
    col_name = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col_name] = [gbif_si[m][i] for m in train_months]
    test_feats[col_name] = [gbif_si[m][i] for m in test_months]

# Shannon entropy of GBIF proportions for each month
# Load GBIF priors (normalized proportions per month)
gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.array([row[cls] for cls in CLASSES])
    probs = np.maximum(probs, 1e-10)  # avoid log(0)
    entropy = -np.sum(probs * np.log(probs))
    month_entropy[m] = entropy

train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

gbif_feat_names = [c for c in train_feats.columns if c.startswith("gbif_si_") or c == "month_gbif_diversity"]
print(f"  Added {len(gbif_feat_names)} GBIF features: {gbif_feat_names}", flush=True)
print(f"  Total features: {len(train_feats.columns)}", flush=True)

# Prepare arrays
X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)

# Print GBIF feature values for key months
print("\n  GBIF feature values by month:", flush=True)
for m in key_months:
    mask = train_months == m
    n_m = mask.sum()
    if n_m == 0:
        # Check test
        mask_t = test_months == m
        n_m = mask_t.sum()
        if n_m == 0:
            continue
        ent = month_entropy[m]
        print(f"  Month {m:2d} (test, N={n_m}): entropy={ent:.3f}", end="", flush=True)
    else:
        ent = month_entropy[m]
        print(f"  Month {m:2d} (train, N={n_m}): entropy={ent:.3f}", end="", flush=True)
    # Show a few SI values
    for cls_name in ["Gulls", "Ducks", "Geese", "Songbirds"]:
        ci = CLASSES.index(cls_name)
        print(f"  SI_{cls_name[:4]}={gbif_si[m][ci]:.2f}", end="")
    print(flush=True)

# 5-fold SKF training
print(f"\n  Training LGB+XGB+CB with {len(fn)} features, 5-fold SKF...", flush=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_b = np.zeros((len(y), N_CLASSES))
test_b = np.zeros((len(X_test), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    oof_fold, test_fold = train_fold(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test, fn,
        f"Fold {fold_idx}",
    )
    oof_b[va_idx] = oof_fold
    test_b += test_fold / 5

map_b, per_b = compute_map(y, oof_b)

print(f"\n  Part B: SKF CV mAP = {map_b:.4f} (E32 baseline: {baseline_full_map:.4f}, delta: {map_b - baseline_full_map:+.4f})", flush=True)
print(f"\n  Per-class comparison (E32 vs E36-B):", flush=True)
print(f"  {'Class':<15s} {'E32':>7s} {'E36-B':>7s} {'Delta':>7s}", flush=True)
for cls in CLASSES:
    ap_e32 = baseline_full_per.get(cls, 0)
    ap_b = per_b.get(cls, 0)
    marker = " *" if abs(ap_b - ap_e32) > 0.01 else ""
    print(f"  {cls:<15s} {ap_e32:>7.4f} {ap_b:>7.4f} {ap_b - ap_e32:>+7.4f}{marker}", flush=True)

# ======================================================================
# Part A+B: Post-process E36-B predictions with GBIF priors
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART A+B: COMBINED (GBIF features + GBIF post-processing)", flush=True)
print("=" * 60, flush=True)

# Alpha sweep on E36-B OOF (shared months)
oof_b_shared = oof_b[shared_mask]

baseline_b_shared, _ = compute_map(y_shared, oof_b_shared)
print(f"\n  OOF shared months (Sep, Oct) for E36-B:", flush=True)
print(f"  alpha=0.00 (baseline): mAP={baseline_b_shared:.4f}", flush=True)

best_alpha_ab = 0
best_map_ab = baseline_b_shared

for alpha in ALPHAS:
    if alpha == 0:
        continue
    adjusted = oof_b_shared.copy()
    for m in [9, 10]:
        m_mask = months_shared == m
        if m_mask.sum() > 0:
            adjusted[m_mask] = bayesian_adjust(oof_b_shared[m_mask], gbif_priors[m], p_train, alpha)
    adj_map, _ = compute_map(y_shared, adjusted)
    delta = adj_map - baseline_b_shared
    print(f"  alpha={alpha:.2f}: mAP={adj_map:.4f} (delta: {delta:+.4f})", flush=True)
    if adj_map > best_map_ab:
        best_map_ab = adj_map
        best_alpha_ab = alpha

print(f"\n  Best alpha for A+B: {best_alpha_ab:.2f}", flush=True)

# Apply to test
test_ab = test_b.copy()
if best_alpha_ab > 0:
    for m in unique_test_months:
        m_mask = test_months == m
        if m_mask.sum() > 0:
            test_ab[m_mask] = bayesian_adjust(test_b[m_mask], gbif_priors[m], p_train, best_alpha_ab)

# Full OOF for combined
oof_ab = oof_b.copy()
if best_alpha_ab > 0:
    for m in unique_train_months:
        m_mask = train_months == m
        if m_mask.sum() > 0:
            oof_ab[m_mask] = bayesian_adjust(oof_b[m_mask], gbif_priors[m], p_train, best_alpha_ab)
map_ab, per_ab = compute_map(y, oof_ab)

# ======================================================================
# Step 5: Summary and Save
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E36 SUMMARY", flush=True)
print("=" * 60, flush=True)

print(f"\n  {'Variant':<25s} {'CV mAP':>8s} {'Delta vs E32':>12s}", flush=True)
print(f"  {'-'*45}", flush=True)
print(f"  {'E32 baseline':<25s} {baseline_full_map:>8.4f} {'---':>12s}", flush=True)
print(f"  {'E36-A (post-proc only)':<25s} {'N/A':>8s} {'alpha='+str(best_alpha_a):>12s}", flush=True)
print(f"  {'E36-B (GBIF features)':<25s} {map_b:>8.4f} {map_b - baseline_full_map:>+12.4f}", flush=True)
print(f"  {'E36-AB (combined)':<25s} {map_ab:>8.4f} {map_ab - baseline_full_map:>+12.4f}", flush=True)

print(f"\n  Per-class breakdown:", flush=True)
print(f"  {'Class':<15s} {'E32':>7s} {'E36-B':>7s} {'E36-AB':>7s}", flush=True)
for cls in CLASSES:
    ap_32 = baseline_full_per.get(cls, 0)
    ap_b_cls = per_b.get(cls, 0)
    ap_ab_cls = per_ab.get(cls, 0)
    print(f"  {cls:<15s} {ap_32:>7.4f} {ap_b_cls:>7.4f} {ap_ab_cls:>7.4f}", flush=True)

# Test prediction distributions
print(f"\n  Test class distributions (argmax):", flush=True)
print(f"  {'Class':<15s} {'E32':>6s} {'E36-A':>6s} {'E36-B':>6s} {'E36-AB':>6s}", flush=True)
dist_32 = np.bincount(test_e32.argmax(axis=1), minlength=N_CLASSES)
dist_a = np.bincount(test_a.argmax(axis=1), minlength=N_CLASSES)
dist_b = np.bincount(test_b.argmax(axis=1), minlength=N_CLASSES)
dist_ab = np.bincount(test_ab.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<15s} {dist_32[i]:>6d} {dist_a[i]:>6d} {dist_b[i]:>6d} {dist_ab[i]:>6d}", flush=True)

# Per-month test distributions for E36-B
print(f"\n  E36-B test predictions per month:", flush=True)
print(f"  {'Month':>5s} {'N':>5s} {'Unseen':>6s}  Top-3", flush=True)
for m in sorted(unique_test_months):
    m_mask = test_months == m
    n_m = m_mask.sum()
    unseen = "YES" if m not in unique_train_months else "no"
    t3 = top3_str(test_b[m_mask])
    print(f"  {m:5d} {n_m:5d} {unseen:>6s}  {t3}", flush=True)

# Save predictions
np.save(ROOT / "oof_e36b.npy", oof_b)
np.save(ROOT / "test_e36b.npy", test_b)
print(f"\n  Saved: oof_e36b.npy, test_e36b.npy", flush=True)

# Save submissions
print("\n  Saving submissions...", flush=True)

# E36-A: E32 + GBIF post-processing only
save_submission(test_a, f"e36a_gbif_postproc_a{best_alpha_a:.2f}", cv_map=baseline_full_map)

# E36-B: retrained model with GBIF features (no post-proc)
save_submission(test_b, "e36b_gbif_features", cv_map=map_b)

# E36-AB: retrained + post-processing
save_submission(test_ab, f"e36ab_gbif_combined_a{best_alpha_ab:.2f}", cv_map=map_ab)

# Also save alpha=1.0 variants for A and AB if not already best
if best_alpha_a != 1.0:
    test_a_1 = test_e32.copy()
    for m in unique_test_months:
        m_mask = test_months == m
        if m_mask.sum() > 0:
            test_a_1[m_mask] = bayesian_adjust(test_e32[m_mask], gbif_priors[m], p_train, 1.0)
    save_submission(test_a_1, "e36a_gbif_postproc_a1.00", cv_map=baseline_full_map)

if best_alpha_ab != 1.0:
    test_ab_1 = test_b.copy()
    for m in unique_test_months:
        m_mask = test_months == m
        if m_mask.sum() > 0:
            test_ab_1[m_mask] = bayesian_adjust(test_b[m_mask], gbif_priors[m], p_train, 1.0)
    save_submission(test_ab_1, "e36ab_gbif_combined_a1.00", cv_map=map_ab)

print("\nDone!", flush=True)
