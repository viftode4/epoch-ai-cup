"""Feature Importance & Selection Analysis

Comprehensive analysis of feature importance, stability, and selection
using CatBoost with LOMO (leave-one-month-out) cross-validation.

PART 1: Global feature importance (average across LOMO folds)
PART 2: Feature importance stability across months (rank correlation)
PART 3: Greedy backward elimination on LOMO (find optimal feature count)
PART 4: Per-class feature importance (SHAP or binary classifiers)
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "G:/projects/epoch-ai-cup")
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map

ROOT = Path("G:/projects/epoch-ai-cup")
N_CLASSES = len(CLASSES)

# ======================================================================
# Load data + features (E38 config, same as e42)
# ======================================================================
print("=" * 70, flush=True)
print("  FEATURE IMPORTANCE & SELECTION ANALYSIS", flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Timestamps
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# Base features (temporal-free)
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar + GBIF (E38 config)
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_counts = gbif[cls].values
            class_mean = class_counts.mean()
            si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    gbif_si[m] = si

for i, cls in enumerate(CLASSES):
    col = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col] = [gbif_si[m][i] for m in train_months]
    test_feats[col] = [gbif_si[m][i] for m in test_months]

gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

fn = list(train_feats.columns)
X = train_feats.values.astype(np.float32)
print(f"  Total features: {len(fn)}", flush=True)
print(f"  Train samples: {len(y)}", flush=True)
print(f"  Train months: {unique_months}", flush=True)

# Effective number class weights
BETA = 0.999
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])


# ======================================================================
# Helper: train CatBoost LOMO and return OOF + per-fold importance
# ======================================================================
def train_lomo_catboost(X_data, y_data, feature_names, train_months_arr, unique_months_arr,
                        sample_w, return_importance=True, verbose_folds=True):
    """Train CatBoost with LOMO CV. Returns oof_preds, lomo_map, per_class, fold_importances."""
    n_samples = len(y_data)
    oof = np.zeros((n_samples, N_CLASSES))
    fold_importances = []

    for m in unique_months_arr:
        va_idx = np.where(train_months_arr == m)[0]
        tr_idx = np.where(train_months_arr != m)[0]

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_data[tr_idx], y_data[tr_idx],
               eval_set=(X_data[va_idx], y_data[va_idx]),
               verbose=0, sample_weight=sample_w[tr_idx])
        oof[va_idx] = cb.predict_proba(X_data[va_idx])

        if return_importance:
            imp = cb.get_feature_importance()
            fold_importances.append(imp)

        if verbose_folds:
            m_val, _ = compute_map(y_data[va_idx], oof[va_idx])
            print(f"    Month {m}: mAP={m_val:.4f} (n={len(va_idx)})", flush=True)

    overall_map, per_class = compute_map(y_data, oof)
    return oof, overall_map, per_class, fold_importances


# ######################################################################
# PART 1: GLOBAL FEATURE IMPORTANCE
# ######################################################################
print("\n" + "=" * 70, flush=True)
print("  PART 1: GLOBAL FEATURE IMPORTANCE (LOMO)", flush=True)
print("=" * 70, flush=True)

t0 = time.time()
oof_preds, baseline_map, baseline_per, fold_imps = train_lomo_catboost(
    X, y, fn, train_months, unique_months, sample_weights,
    return_importance=True, verbose_folds=True
)
t1 = time.time()
print(f"\n  Baseline LOMO mAP: {baseline_map:.4f} ({t1-t0:.0f}s)", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {baseline_per[cls]:.4f}", flush=True)

# Average importance across folds
avg_imp = np.mean(fold_imps, axis=0)
imp_pct = avg_imp / avg_imp.sum() * 100
imp_df = pd.DataFrame({
    "feature": fn,
    "importance": avg_imp,
    "pct": imp_pct,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n  --- TOP 30 FEATURES ---", flush=True)
print(f"  {'Rank':<5s} {'Feature':<40s} {'Importance':>10s} {'%':>6s}", flush=True)
print(f"  {'-'*63}", flush=True)
for i, row in imp_df.head(30).iterrows():
    print(f"  {i+1:<5d} {row['feature']:<40s} {row['importance']:>10.2f} {row['pct']:>5.2f}%", flush=True)

print(f"\n  --- BOTTOM 30 FEATURES ---", flush=True)
print(f"  {'Rank':<5s} {'Feature':<40s} {'Importance':>10s} {'%':>6s}", flush=True)
print(f"  {'-'*63}", flush=True)
for i, row in imp_df.tail(30).iterrows():
    print(f"  {i+1:<5d} {row['feature']:<40s} {row['importance']:>10.2f} {row['pct']:>5.2f}%", flush=True)

# Near-zero features
near_zero = imp_df[imp_df["pct"] < 0.1]
print(f"\n  Features with <0.1% importance: {len(near_zero)} / {len(fn)}", flush=True)
if len(near_zero) > 0:
    print(f"  These features contribute <0.1% each:", flush=True)
    for _, row in near_zero.iterrows():
        print(f"    {row['feature']:<40s} {row['pct']:.3f}%", flush=True)

zero_imp = imp_df[imp_df["importance"] == 0]
print(f"\n  Features with ZERO importance: {len(zero_imp)} / {len(fn)}", flush=True)
if len(zero_imp) > 0:
    for _, row in zero_imp.iterrows():
        print(f"    {row['feature']}", flush=True)


# ######################################################################
# PART 2: FEATURE IMPORTANCE STABILITY ACROSS MONTHS
# ######################################################################
print("\n" + "=" * 70, flush=True)
print("  PART 2: FEATURE IMPORTANCE STABILITY ACROSS MONTHS", flush=True)
print("=" * 70, flush=True)

# Rank features within each fold
n_feats = len(fn)
fold_ranks = []
for fi, imp in enumerate(fold_imps):
    # Higher importance -> lower rank (rank 1 = most important)
    rank = np.zeros(n_feats, dtype=int)
    sorted_idx = np.argsort(-imp)
    for r, idx in enumerate(sorted_idx):
        rank[idx] = r + 1
    fold_ranks.append(rank)

fold_ranks = np.array(fold_ranks)  # shape: (4, n_feats)

# Spearman rank correlation between all pairs
print(f"\n  Spearman rank correlation between LOMO folds:", flush=True)
print(f"  {'':>12s}", end="", flush=True)
for m in unique_months:
    print(f"  Month {m:>2d}", end="", flush=True)
print(flush=True)

pair_corrs = []
for i in range(len(unique_months)):
    print(f"  Month {unique_months[i]:>2d}  ", end="", flush=True)
    for j in range(len(unique_months)):
        rho, _ = spearmanr(fold_ranks[i], fold_ranks[j])
        pair_corrs.append(rho)
        print(f"  {rho:>7.3f} ", end="", flush=True)
    print(flush=True)

avg_rho = np.mean([c for c in pair_corrs if c < 1.0])
print(f"\n  Average pairwise Spearman rho (excl. self): {avg_rho:.4f}", flush=True)

# Identify unstable features: top-20 in some folds, bottom-50 in others
print(f"\n  --- UNSTABLE FEATURES (top-20 in some folds, bottom-50 in others) ---", flush=True)
unstable_features = []
for fi in range(n_feats):
    ranks_across_folds = fold_ranks[:, fi]
    best_rank = ranks_across_folds.min()
    worst_rank = ranks_across_folds.max()
    if best_rank <= 20 and worst_rank > (n_feats - 50):
        unstable_features.append({
            "feature": fn[fi],
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "rank_range": worst_rank - best_rank,
            "ranks": list(ranks_across_folds),
            "avg_imp_pct": imp_pct[fi],
        })

unstable_features.sort(key=lambda x: -x["rank_range"])
if len(unstable_features) == 0:
    print(f"  None found (good -- features are stable).", flush=True)
else:
    print(f"  Found {len(unstable_features)} unstable features:", flush=True)
    print(f"  {'Feature':<40s} {'Best':>5s} {'Worst':>6s} {'Range':>6s}  Ranks by month", flush=True)
    print(f"  {'-'*80}", flush=True)
    for uf in unstable_features:
        ranks_str = ", ".join([f"M{m}:{r}" for m, r in zip(unique_months, uf["ranks"])])
        print(f"  {uf['feature']:<40s} {uf['best_rank']:>5d} {uf['worst_rank']:>6d} {uf['rank_range']:>6d}  [{ranks_str}]", flush=True)

# Also show features with high rank variance (top 20 most variable)
rank_std = np.std(fold_ranks, axis=0)
rank_var_order = np.argsort(-rank_std)
print(f"\n  --- TOP 20 FEATURES BY RANK VARIANCE ---", flush=True)
print(f"  {'Feature':<40s} {'Mean Rank':>10s} {'Rank Std':>9s} {'Avg %':>6s}  Ranks by month", flush=True)
print(f"  {'-'*90}", flush=True)
for idx in rank_var_order[:20]:
    ranks_str = ", ".join([f"M{m}:{fold_ranks[fi, idx]}" for fi, m in enumerate(unique_months)])
    mean_rank = np.mean(fold_ranks[:, idx])
    print(f"  {fn[idx]:<40s} {mean_rank:>10.1f} {rank_std[idx]:>9.1f} {imp_pct[idx]:>5.2f}%  [{ranks_str}]", flush=True)


# ######################################################################
# PART 3: GREEDY BACKWARD ELIMINATION ON LOMO
# ######################################################################
print("\n" + "=" * 70, flush=True)
print("  PART 3: GREEDY BACKWARD ELIMINATION ON LOMO", flush=True)
print("=" * 70, flush=True)
print(f"  Starting with {len(fn)} features, baseline LOMO mAP = {baseline_map:.4f}", flush=True)
print(f"  Strategy: remove batches of 10, then 5, then 1 near optimum", flush=True)

# Sort features by importance (least important first for removal)
imp_order = imp_df["feature"].tolist()  # most important first
removal_order = list(reversed(imp_order))  # least important first

# Track results
elimination_log = []
elimination_log.append({
    "n_features": len(fn),
    "lomo_map": baseline_map,
    "removed": "none",
    "per_class": dict(baseline_per),
})

current_features = list(fn)
current_X = X.copy()
best_map = baseline_map
best_n_features = len(fn)
best_features = list(fn)

# Batch sizes: remove 10 at a time until 50 features left,
# then 5 at a time until 20 features left, then 1 at a time
def get_batch_size(n_remaining):
    if n_remaining > 80:
        return 10
    elif n_remaining > 40:
        return 5
    elif n_remaining > 15:
        return 3
    else:
        return 1

step = 0
while len(current_features) > 10:
    batch_size = get_batch_size(len(current_features))
    # Find least important features in current set
    # Re-rank using the last fold importances or use the global rank
    # For efficiency, use the global importance ranking
    to_remove = []
    for feat in removal_order:
        if feat in current_features and feat not in to_remove:
            to_remove.append(feat)
            if len(to_remove) >= batch_size:
                break

    if len(to_remove) == 0:
        break

    # Remove features
    for feat in to_remove:
        current_features.remove(feat)
        removal_order.remove(feat)

    # Build reduced X
    feat_idx = [fn.index(f) for f in current_features]
    X_reduced = X[:, feat_idx]
    sw_reduced = sample_weights.copy()

    step += 1
    print(f"\n  Step {step}: {len(current_features)} features (removed {len(to_remove)}: {', '.join(to_remove[:3])}{'...' if len(to_remove) > 3 else ''})", flush=True)

    _, lomo_map, per_class, _ = train_lomo_catboost(
        X_reduced, y, current_features, train_months, unique_months,
        sw_reduced, return_importance=False, verbose_folds=False
    )

    elimination_log.append({
        "n_features": len(current_features),
        "lomo_map": lomo_map,
        "removed": ", ".join(to_remove),
        "per_class": dict(per_class),
    })

    delta = lomo_map - baseline_map
    marker = " *** NEW BEST ***" if lomo_map > best_map else ""
    print(f"    LOMO mAP = {lomo_map:.4f} (delta={delta:+.4f}){marker}", flush=True)

    if lomo_map > best_map:
        best_map = lomo_map
        best_n_features = len(current_features)
        best_features = list(current_features)

# Summary
print(f"\n  {'='*60}", flush=True)
print(f"  BACKWARD ELIMINATION SUMMARY", flush=True)
print(f"  {'='*60}", flush=True)
print(f"\n  {'N Features':>12s} {'LOMO mAP':>10s} {'Delta':>8s}  Removed", flush=True)
print(f"  {'-'*70}", flush=True)
for entry in elimination_log:
    delta = entry["lomo_map"] - baseline_map
    is_best = " <-- BEST" if entry["lomo_map"] == best_map and entry["n_features"] == best_n_features else ""
    removed_str = entry["removed"]
    if len(removed_str) > 35:
        removed_str = removed_str[:35] + "..."
    print(f"  {entry['n_features']:>12d} {entry['lomo_map']:>10.4f} {delta:>+8.4f}  {removed_str}{is_best}", flush=True)

print(f"\n  Best: {best_n_features} features, LOMO mAP = {best_map:.4f} (baseline: {baseline_map:.4f}, delta: {best_map - baseline_map:+.4f})", flush=True)

# Show per-class comparison at best point
best_entry = [e for e in elimination_log if e["n_features"] == best_n_features and e["lomo_map"] == best_map][0]
print(f"\n  Per-class comparison (baseline vs best):", flush=True)
print(f"  {'Class':<18s} {'Baseline':>10s} {'Best':>10s} {'Delta':>8s}", flush=True)
print(f"  {'-'*48}", flush=True)
for cls in CLASSES:
    b = baseline_per[cls]
    bst = best_entry["per_class"][cls]
    print(f"  {cls:<18s} {b:>10.4f} {bst:>10.4f} {bst-b:>+8.4f}", flush=True)

# List the features that were removed to reach the best point
if best_n_features < len(fn):
    removed_features = [f for f in fn if f not in best_features]
    print(f"\n  Features removed to reach best ({len(removed_features)}):", flush=True)
    for f in removed_features:
        orig_rank = imp_df[imp_df["feature"] == f].index[0] + 1
        print(f"    {f:<40s} (original rank: {orig_rank})", flush=True)

print(f"\n  Recommended feature set ({best_n_features} features):", flush=True)
# Sort by importance
best_feat_imp = [(f, imp_pct[fn.index(f)]) for f in best_features]
best_feat_imp.sort(key=lambda x: -x[1])
for f, pct in best_feat_imp[:20]:
    print(f"    {f:<40s} {pct:.2f}%", flush=True)
if best_n_features > 20:
    print(f"    ... and {best_n_features - 20} more", flush=True)


# ######################################################################
# PART 4: PER-CLASS FEATURE IMPORTANCE
# ######################################################################
print("\n" + "=" * 70, flush=True)
print("  PART 4: PER-CLASS FEATURE IMPORTANCE", flush=True)
print("=" * 70, flush=True)

# Train one-vs-rest binary CatBoost per class, collect feature importance
# Use full training data (no LOMO) to get stable importance estimates
print(f"\n  Training 9 binary (one-vs-rest) classifiers on full data...", flush=True)

class_top_features = {}
class_importances = {}

for ci, cls in enumerate(CLASSES):
    y_bin = (y == ci).astype(int)
    pos_count = y_bin.sum()
    neg_count = len(y_bin) - pos_count

    # Scale factor for positive class
    scale = neg_count / max(pos_count, 1)

    cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=0,
        scale_pos_weight=min(scale, 20),  # cap to avoid instability
        task_type="GPU",
    )
    cb.fit(X, y_bin, verbose=0)
    imp = cb.get_feature_importance()
    imp_pct_cls = imp / imp.sum() * 100

    class_importances[cls] = imp_pct_cls

    # Top 10 features for this class
    top_idx = np.argsort(-imp)[:10]
    top_feats = [(fn[i], imp_pct_cls[i]) for i in top_idx]
    class_top_features[cls] = top_feats

    print(f"\n  {cls} (n={pos_count}):", flush=True)
    print(f"  {'Rank':<5s} {'Feature':<40s} {'Importance %':>12s}", flush=True)
    print(f"  {'-'*59}", flush=True)
    for rank, (feat, pct) in enumerate(top_feats, 1):
        print(f"  {rank:<5d} {feat:<40s} {pct:>11.2f}%", flush=True)

# Cross-class analysis: features critical for minority but not majority
print(f"\n  {'='*60}", flush=True)
print(f"  MINORITY-CRITICAL FEATURES", flush=True)
print(f"  (Important for minority classes but NOT for Gulls/Songbirds)", flush=True)
print(f"  {'='*60}", flush=True)

majority = ["Gulls", "Songbirds"]
minority = ["Cormorants", "Ducks", "Pigeons", "Waders", "Birds of Prey", "Geese", "Clutter"]

# For each feature, compute average importance in minority vs majority
feat_minority_imp = np.zeros(n_feats)
feat_majority_imp = np.zeros(n_feats)

for cls in majority:
    feat_majority_imp += class_importances[cls]
feat_majority_imp /= len(majority)

for cls in minority:
    feat_minority_imp += class_importances[cls]
feat_minority_imp /= len(minority)

# Features where minority importance >> majority importance
ratio = (feat_minority_imp + 0.01) / (feat_majority_imp + 0.01)
ratio_order = np.argsort(-ratio)

print(f"\n  Top 20 features with highest minority/majority importance ratio:", flush=True)
print(f"  {'Feature':<40s} {'Minority %':>11s} {'Majority %':>11s} {'Ratio':>7s}", flush=True)
print(f"  {'-'*72}", flush=True)
for idx in ratio_order[:20]:
    print(f"  {fn[idx]:<40s} {feat_minority_imp[idx]:>10.2f}% {feat_majority_imp[idx]:>10.2f}% {ratio[idx]:>6.2f}x", flush=True)

# Features important ONLY for majority (candidates for removal if hurting minority)
ratio_order_rev = np.argsort(ratio)
print(f"\n  Top 20 features with highest majority/minority importance ratio:", flush=True)
print(f"  (These features help majority but not minority -- potential noise for minority)", flush=True)
print(f"  {'Feature':<40s} {'Minority %':>11s} {'Majority %':>11s} {'Ratio':>7s}", flush=True)
print(f"  {'-'*72}", flush=True)
for idx in ratio_order_rev[:20]:
    print(f"  {fn[idx]:<40s} {feat_minority_imp[idx]:>10.2f}% {feat_majority_imp[idx]:>10.2f}% {ratio[idx]:>6.2f}x", flush=True)

# Per-class unique features (top-5 for a class that are NOT in top-20 for any other class)
print(f"\n  {'='*60}", flush=True)
print(f"  CLASS-UNIQUE FEATURES (top-5 for class, not top-20 for others)", flush=True)
print(f"  {'='*60}", flush=True)

# Build top-20 sets per class
top20_per_class = {}
for cls in CLASSES:
    top_idx = np.argsort(-class_importances[cls])[:20]
    top20_per_class[cls] = set(fn[i] for i in top_idx)

for cls in CLASSES:
    top5_idx = np.argsort(-class_importances[cls])[:5]
    unique_feats = []
    for idx in top5_idx:
        feat = fn[idx]
        # Check if in top-20 of any other class
        in_other = [c for c in CLASSES if c != cls and feat in top20_per_class[c]]
        if len(in_other) == 0:
            unique_feats.append((feat, class_importances[cls][idx]))

    if unique_feats:
        print(f"\n  {cls}:", flush=True)
        for feat, pct in unique_feats:
            print(f"    {feat:<40s} {pct:.2f}%", flush=True)
    else:
        print(f"\n  {cls}: (no unique features in top-5)", flush=True)


# ######################################################################
# FINAL SUMMARY
# ######################################################################
print("\n" + "=" * 70, flush=True)
print("  FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)

print(f"\n  Total features analyzed: {len(fn)}", flush=True)
print(f"  Features with <0.1% importance: {len(near_zero)}", flush=True)
print(f"  Features with zero importance: {len(zero_imp)}", flush=True)
print(f"  Unstable features (rank varies wildly): {len(unstable_features)}", flush=True)
print(f"  Avg rank correlation across months: {avg_rho:.4f}", flush=True)
print(f"\n  Backward elimination:", flush=True)
print(f"    Baseline ({len(fn)} features): LOMO mAP = {baseline_map:.4f}", flush=True)
print(f"    Best ({best_n_features} features): LOMO mAP = {best_map:.4f}", flush=True)
print(f"    Delta: {best_map - baseline_map:+.4f}", flush=True)

if best_map > baseline_map + 0.002:
    print(f"\n  RECOMMENDATION: Use {best_n_features} features instead of {len(fn)}.", flush=True)
    print(f"  Feature reduction improves LOMO by {best_map - baseline_map:+.4f}.", flush=True)
elif best_map >= baseline_map - 0.002:
    print(f"\n  RECOMMENDATION: Feature count has minimal impact on LOMO.", flush=True)
    print(f"  Can safely reduce to {best_n_features} for faster training with no loss.", flush=True)
else:
    print(f"\n  RECOMMENDATION: Keep all {len(fn)} features.", flush=True)
    print(f"  Removing features consistently hurts LOMO.", flush=True)

# Save results for future use
results_path = ROOT / "data" / "feature_importance_analysis.csv"
imp_df.to_csv(results_path, index=False)
print(f"\n  Feature importance saved to: {results_path}", flush=True)

# Save best feature list
best_feat_path = ROOT / "data" / "best_features.txt"
with open(best_feat_path, "w") as f:
    for feat in best_features:
        f.write(feat + "\n")
print(f"  Best feature list saved to: {best_feat_path}", flush=True)

# Save elimination log
elim_path = ROOT / "data" / "elimination_log.csv"
elim_rows = []
for entry in elimination_log:
    row = {"n_features": entry["n_features"], "lomo_map": entry["lomo_map"], "removed": entry["removed"]}
    for cls in CLASSES:
        row[f"ap_{cls}"] = entry["per_class"][cls]
    elim_rows.append(row)
pd.DataFrame(elim_rows).to_csv(elim_path, index=False)
print(f"  Elimination log saved to: {elim_path}", flush=True)

print(f"\n  DONE.", flush=True)
