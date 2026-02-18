"""Deep-dive analysis: month distributions, feature shift, LOMO breakdown, test estimation.

Comprehensive investigation into WHY LOMO September = 0.26 (worst month despite being
"shared" with test), and what the optimal LOMO weighting should be.

Five parts:
  1. Month distribution analysis (class counts, entropy, JS divergence)
  2. Feature distribution shift per month
  3. Per-class per-month LOMO breakdown with CatBoost
  4. Test month estimation and predicted class distribution
  5. Optimal LOMO weighting for test-like evaluation
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ======================================================================
# DATA LOADING
# ======================================================================
P("=" * 70)
P("DEEP DIVE ANALYSIS: Month Distributions, Feature Shift, LOMO Breakdown")
P("=" * 70)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

P(f"\nTrain: {len(train_df)} samples, months {unique_months}")
P(f"Test:  {len(test_df)} samples, months {sorted(np.unique(test_months))}")

# ======================================================================
# PART 1: MONTH DISTRIBUTION ANALYSIS
# ======================================================================
P("\n" + "=" * 70)
P("PART 1: MONTH DISTRIBUTION ANALYSIS")
P("=" * 70)

month_names = {1: "Jan", 2: "Feb", 4: "Apr", 5: "May", 9: "Sep", 10: "Oct", 12: "Dec"}

# 1a: Per-month class counts and percentages
P("\n--- 1a: Class distribution per training month ---\n")

month_distributions = {}
for m in unique_months:
    mask = train_months == m
    m_y = y[mask]
    counts = np.bincount(m_y, minlength=N_CLASSES)
    total = counts.sum()
    pct = counts / total * 100
    month_distributions[m] = counts / total  # normalized for later use

    P(f"Month {m} ({month_names.get(m, '?')}): {total} samples")
    P(f"  {'Class':>15s}  {'Count':>5s}  {'Pct':>6s}")
    P(f"  {'-'*15}  {'-'*5}  {'-'*6}")
    for i, cls in enumerate(CLASSES):
        marker = " *** RARE ***" if counts[i] < 5 else ""
        P(f"  {cls:>15s}  {counts[i]:5d}  {pct[i]:5.1f}%{marker}")

    # Entropy
    dist = counts / total
    h = entropy(dist + 1e-10)
    P(f"  Class entropy: {h:.3f} (max = {np.log(N_CLASSES):.3f})")
    P()

# 1b: Jensen-Shannon divergence between months
P("\n--- 1b: Jensen-Shannon divergence between training months ---\n")

P(f"  {'':>6s}", end="")
for m2 in unique_months:
    P(f"  M{m2:02d}", end="")
P()
P(f"  {'':>6s}", end="")
for _ in unique_months:
    P(f"  ----", end="")
P()

for m1 in unique_months:
    P(f"  M{m1:02d}  ", end="")
    for m2 in unique_months:
        d1 = month_distributions[m1]
        d2 = month_distributions[m2]
        js = jensenshannon(d1, d2)
        P(f"  {js:.3f}" if m1 != m2 else "  ----", end="")
    P()

# 1c: Absent or very rare classes per month
P("\n--- 1c: Classes with <5 samples per month ---\n")
for m in unique_months:
    mask = train_months == m
    m_y = y[mask]
    counts = np.bincount(m_y, minlength=N_CLASSES)
    rare = [(CLASSES[i], counts[i]) for i in range(N_CLASSES) if counts[i] < 5]
    if rare:
        rare_str = ", ".join([f"{cls}({n})" for cls, n in rare])
        P(f"  Month {m:2d} ({month_names.get(m, '?')}): {rare_str}")
    else:
        P(f"  Month {m:2d} ({month_names.get(m, '?')}): all classes have >=5 samples")

# ======================================================================
# PART 2: FEATURE DISTRIBUTION SHIFT PER MONTH
# ======================================================================
P("\n" + "=" * 70)
P("PART 2: FEATURE DISTRIBUTION SHIFT PER MONTH")
P("=" * 70)

P("\nBuilding features (E38 config)...", end=" ")
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
month_entropy_map = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy_map[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy_map[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy_map[m] for m in test_months]

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
P(f"done. {len(fn)} features.")

# 2a: Top 20 features by variance
P("\n--- 2a: Top 20 features by variance, mean/std per month ---\n")

variances = np.var(X, axis=0)
top20_idx = np.argsort(variances)[::-1][:20]

P(f"  {'Feature':>30s}", end="")
for m in unique_months:
    P(f"  M{m:02d}_mean   M{m:02d}_std", end="")
P()
P(f"  {'-'*30}", end="")
for m in unique_months:
    P(f"  {'-'*9}  {'-'*8}", end="")
P()

for idx in top20_idx:
    fname = fn[idx][:30]
    P(f"  {fname:>30s}", end="")
    for m in unique_months:
        mask = train_months == m
        vals = X[mask, idx]
        P(f"  {np.mean(vals):9.2f}  {np.std(vals):8.2f}", end="")
    P()

# 2b: Feature shift: for each month, KL-like distance from "rest"
P("\n--- 2b: Features with largest shift for September vs other months ---")
P("        (measured by absolute difference in standardized mean)\n")

# For each feature, compute mean/std on "not-month-9", then see how far month-9 is
global_mean = np.mean(X, axis=0)
global_std = np.std(X, axis=0) + 1e-10

sep_mask = train_months == 9
sep_means = np.mean(X[sep_mask], axis=0)
rest_means = np.mean(X[~sep_mask], axis=0)
rest_stds = np.std(X[~sep_mask], axis=0) + 1e-10

# Z-score of September mean relative to non-September distribution
sep_shift = np.abs(sep_means - rest_means) / rest_stds
top_shift_idx = np.argsort(sep_shift)[::-1][:30]

P(f"  {'Rank':>4s}  {'Feature':>35s}  {'Sep_mean':>9s}  {'Rest_mean':>9s}  {'Z-shift':>8s}")
P(f"  {'-'*4}  {'-'*35}  {'-'*9}  {'-'*9}  {'-'*8}")
for rank, idx in enumerate(top_shift_idx, 1):
    fname = fn[idx][:35]
    P(f"  {rank:4d}  {fname:>35s}  {sep_means[idx]:9.3f}  {rest_means[idx]:9.3f}  {sep_shift[idx]:8.3f}")

# 2c: Same analysis but Oct vs rest (Oct is best LOMO month)
P("\n--- 2c: Features with largest shift for October vs other months ---\n")

oct_mask = train_months == 10
oct_means = np.mean(X[oct_mask], axis=0)
rest_oct_means = np.mean(X[~oct_mask], axis=0)
rest_oct_stds = np.std(X[~oct_mask], axis=0) + 1e-10

oct_shift = np.abs(oct_means - rest_oct_means) / rest_oct_stds
top_oct_shift_idx = np.argsort(oct_shift)[::-1][:20]

P(f"  {'Rank':>4s}  {'Feature':>35s}  {'Oct_mean':>9s}  {'Rest_mean':>9s}  {'Z-shift':>8s}")
P(f"  {'-'*4}  {'-'*35}  {'-'*9}  {'-'*9}  {'-'*8}")
for rank, idx in enumerate(top_oct_shift_idx, 1):
    fname = fn[idx][:35]
    P(f"  {rank:4d}  {fname:>35s}  {oct_means[idx]:9.3f}  {rest_oct_means[idx]:9.3f}  {oct_shift[idx]:8.3f}")

# 2d: Compare Sep shift to Oct shift -- which features differentiate them?
P("\n--- 2d: Features where Sep and Oct shift in OPPOSITE directions ---")
P("        (relative to global mean, sign of deviation differs)\n")

sep_dev = sep_means - global_mean
oct_dev = oct_means - global_mean
# Features where Sep and Oct deviate in opposite directions
opposite = (sep_dev * oct_dev) < 0
opp_magnitude = np.abs(sep_dev - oct_dev) / global_std
opp_idx = np.where(opposite)[0]
opp_sorted = opp_idx[np.argsort(opp_magnitude[opp_idx])[::-1]][:20]

P(f"  {'Feature':>35s}  {'Sep_dev':>8s}  {'Oct_dev':>8s}  {'Magnitude':>10s}")
P(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*10}")
for idx in opp_sorted:
    fname = fn[idx][:35]
    P(f"  {fname:>35s}  {sep_dev[idx]:8.3f}  {oct_dev[idx]:8.3f}  {opp_magnitude[idx]:10.3f}")

# ======================================================================
# PART 3: PER-CLASS PER-MONTH LOMO BREAKDOWN
# ======================================================================
P("\n" + "=" * 70)
P("PART 3: PER-CLASS PER-MONTH LOMO BREAKDOWN (CatBoost)")
P("=" * 70)

# Effective number weights
counts_all = np.bincount(y, minlength=N_CLASSES)
BETA = 0.999
effective_n = 1.0 - np.power(BETA, counts_all)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

P("\nTraining CatBoost LOMO (4 folds, one per month)...\n")

oof_preds = np.zeros((len(y), N_CLASSES))
fold_results = {}

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU"
    )
    cb.fit(
        X[tr_idx], y[tr_idx],
        eval_set=(X[va_idx], y[va_idx]),
        verbose=0,
        sample_weight=sample_weights[tr_idx]
    )

    preds = cb.predict_proba(X[va_idx])
    oof_preds[va_idx] = preds

    # Per-class AP
    y_val = y[va_idx]
    y_onehot = np.eye(N_CLASSES)[y_val]

    per_class_ap = {}
    per_class_counts = {}
    for c in range(N_CLASSES):
        n_pos = int(y_onehot[:, c].sum())
        per_class_counts[CLASSES[c]] = n_pos
        if n_pos > 0:
            per_class_ap[CLASSES[c]] = average_precision_score(y_onehot[:, c], preds[:, c])
        else:
            per_class_ap[CLASSES[c]] = 0.0

    macro_map = np.mean(list(per_class_ap.values()))
    fold_results[m] = {
        "n_val": len(va_idx),
        "n_train": len(tr_idx),
        "macro_map": macro_map,
        "per_class_ap": per_class_ap,
        "per_class_counts": per_class_counts,
        "preds": preds,
        "y_val": y_val,
    }

    P(f"  Month {m:2d} ({month_names.get(m, '?')}): val={len(va_idx):4d}, "
      f"train={len(tr_idx):4d}, mAP={macro_map:.4f}")

# 3a: Detailed per-class per-month table
P("\n--- 3a: Per-class AP by validation month ---\n")

P(f"  {'Class':>15s}", end="")
for m in unique_months:
    P(f"  M{m:02d}(AP)", end="")
P(f"  {'Mean':>8s}")
P(f"  {'-'*15}", end="")
for m in unique_months:
    P(f"  {'-'*8}", end="")
P(f"  {'-'*8}")

for cls in CLASSES:
    P(f"  {cls:>15s}", end="")
    aps = []
    for m in unique_months:
        ap = fold_results[m]["per_class_ap"][cls]
        aps.append(ap)
        P(f"  {ap:8.4f}", end="")
    P(f"  {np.mean(aps):8.4f}")

P(f"\n  {'MACRO mAP':>15s}", end="")
for m in unique_months:
    P(f"  {fold_results[m]['macro_map']:8.4f}", end="")
overall_map = np.mean([fold_results[m]["macro_map"] for m in unique_months])
P(f"  {overall_map:8.4f}")

# 3b: True counts per class per month
P("\n--- 3b: True class counts in each validation fold ---\n")

P(f"  {'Class':>15s}", end="")
for m in unique_months:
    P(f"  M{m:02d}", end="")
P(f"  {'Total':>6s}")
P(f"  {'-'*15}", end="")
for m in unique_months:
    P(f"  {'-'*5}", end="")
P(f"  {'-'*6}")

for cls in CLASSES:
    P(f"  {cls:>15s}", end="")
    total = 0
    for m in unique_months:
        n = fold_results[m]["per_class_counts"][cls]
        total += n
        P(f"  {n:5d}", end="")
    P(f"  {total:6d}")

# 3c: Predicted class distribution (argmax) vs true
P("\n--- 3c: Predicted (argmax) vs true class distribution per month ---\n")

for m in unique_months:
    fr = fold_results[m]
    pred_classes = np.argmax(fr["preds"], axis=1)
    true_counts = np.bincount(fr["y_val"], minlength=N_CLASSES)
    pred_counts = np.bincount(pred_classes, minlength=N_CLASSES)

    P(f"  Month {m} ({month_names.get(m, '?')}): {fr['n_val']} samples")
    P(f"    {'Class':>15s}  {'True':>5s}  {'Pred':>5s}  {'Diff':>5s}  {'Note':>15s}")
    P(f"    {'-'*15}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*15}")
    for i, cls in enumerate(CLASSES):
        diff = pred_counts[i] - true_counts[i]
        note = ""
        if true_counts[i] > 0 and pred_counts[i] == 0:
            note = "ZERO predicted!"
        elif true_counts[i] > 0 and pred_counts[i] < true_counts[i] * 0.3:
            note = "under-predicted"
        elif pred_counts[i] > true_counts[i] * 2 and true_counts[i] > 0:
            note = "over-predicted"
        P(f"    {cls:>15s}  {true_counts[i]:5d}  {pred_counts[i]:5d}  {diff:+5d}  {note:>15s}")
    P()

# 3d: September deep dive -- which classes collapse?
P("\n--- 3d: SEPTEMBER DEEP DIVE ---\n")

m9 = fold_results[9]
P(f"  September val: {m9['n_val']} samples, mAP = {m9['macro_map']:.4f}")
P(f"\n  Classes that COLLAPSE in September (AP < 0.20):")
for cls in CLASSES:
    ap = m9["per_class_ap"][cls]
    n = m9["per_class_counts"][cls]
    if ap < 0.20:
        # What do these get predicted as?
        cls_idx = CLASSES.index(cls)
        true_mask = m9["y_val"] == cls_idx
        if true_mask.sum() > 0:
            pred_probs = m9["preds"][true_mask]
            pred_argmax = np.argmax(pred_probs, axis=1)
            confused_with = Counter(pred_argmax)
            confused_str = ", ".join([f"{CLASSES[k]}({v})" for k, v in confused_with.most_common(3)])
            avg_true_prob = np.mean(pred_probs[:, cls_idx])
            P(f"\n    {cls}: AP={ap:.4f}, N={n}")
            P(f"      Mean predicted prob for true class: {avg_true_prob:.4f}")
            P(f"      Predicted as: {confused_str}")

# 3e: What's unique about September's training context?
P("\n\n--- 3e: Training set composition when September is held out ---")
P("        (i.e., training on months 1, 4, 10)\n")

sep_tr_idx = np.where(train_months != 9)[0]
sep_tr_y = y[sep_tr_idx]
sep_tr_counts = np.bincount(sep_tr_y, minlength=N_CLASSES)
sep_val_counts = np.bincount(y[train_months == 9], minlength=N_CLASSES)

P(f"  {'Class':>15s}  {'Train(1,4,10)':>13s}  {'Val(Sep)':>8s}  {'Train_pct':>9s}  {'Val_pct':>8s}  {'Ratio':>7s}")
P(f"  {'-'*15}  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*7}")
tr_total = sep_tr_counts.sum()
val_total = sep_val_counts.sum()
for i, cls in enumerate(CLASSES):
    tr_pct = sep_tr_counts[i] / tr_total * 100
    val_pct = sep_val_counts[i] / val_total * 100
    ratio = val_pct / tr_pct if tr_pct > 0 else 0
    P(f"  {cls:>15s}  {sep_tr_counts[i]:13d}  {sep_val_counts[i]:8d}  {tr_pct:8.1f}%  {val_pct:7.1f}%  {ratio:7.2f}")

# ======================================================================
# PART 4: TEST MONTH ESTIMATION
# ======================================================================
P("\n" + "=" * 70)
P("PART 4: TEST MONTH ESTIMATION AND PREDICTED DISTRIBUTION")
P("=" * 70)

# 4a: Use daylight hours to identify test months
P("\n--- 4a: Test month identification via daylight hours ---\n")

test_month_counts = Counter(test_months)
P(f"  {'Month':>5s}  {'Count':>5s}  {'Pct':>6s}  {'Shared?':>7s}")
P(f"  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}")
for m in sorted(test_month_counts.keys()):
    pct = test_month_counts[m] / len(test_df) * 100
    shared = "YES" if m in unique_months else "NO"
    P(f"  {m:5d}  {test_month_counts[m]:5d}  {pct:5.1f}%  {shared:>7s}")

shared_count = sum(test_month_counts[m] for m in test_month_counts if m in unique_months)
unseen_count = sum(test_month_counts[m] for m in test_month_counts if m not in unique_months)
P(f"\n  Shared months (in train):   {shared_count} ({shared_count/len(test_df)*100:.1f}%)")
P(f"  Unseen months (not train): {unseen_count} ({unseen_count/len(test_df)*100:.1f}%)")

# 4b: Load E38 submission and analyze predicted class distribution per test month
P("\n--- 4b: Predicted class distribution per test month (E38 submission) ---\n")

sub = pd.read_csv(ROOT / "submission.csv")
sub_probs = sub[CLASSES].values if all(c in sub.columns for c in CLASSES) else None

# The submission columns might be in different order
if sub_probs is None:
    # Try to find the columns
    sub_cols = [c for c in sub.columns if c != "track_id"]
    P(f"  Submission columns: {sub_cols}")
    sub_probs = np.zeros((len(sub), N_CLASSES))
    for i, cls in enumerate(CLASSES):
        if cls in sub.columns:
            sub_probs[:, i] = sub[cls].values

pred_argmax_test = np.argmax(sub_probs, axis=1)

P(f"  {'Class':>15s}", end="")
for m in sorted(test_month_counts.keys()):
    P(f"  M{m:02d}(N)", end="")
    P(f"  M{m:02d}(%)", end="")
P(f"  {'Total':>6s}")
P(f"  {'-'*15}", end="")
for m in sorted(test_month_counts.keys()):
    P(f"  {'-'*6}  {'-'*6}", end="")
P(f"  {'-'*6}")

for i, cls in enumerate(CLASSES):
    P(f"  {cls:>15s}", end="")
    total = 0
    for m in sorted(test_month_counts.keys()):
        m_mask = test_months == m
        m_preds = pred_argmax_test[m_mask]
        n = (m_preds == i).sum()
        pct = n / m_mask.sum() * 100 if m_mask.sum() > 0 else 0
        total += n
        P(f"  {n:6d}  {pct:5.1f}%", end="")
    P(f"  {total:6d}")

# 4c: Compare train month distributions vs test month (predicted) distributions
P("\n--- 4c: Train class dist vs Test predicted class dist per shared month ---\n")

for m in [9, 10]:
    P(f"  Month {m} ({month_names.get(m, '?')}):")
    train_mask = train_months == m
    test_mask = test_months == m
    train_y_m = y[train_mask]
    test_pred_m = pred_argmax_test[test_mask]
    train_counts_m = np.bincount(train_y_m, minlength=N_CLASSES)
    test_counts_m = np.bincount(test_pred_m, minlength=N_CLASSES)
    train_pct = train_counts_m / train_counts_m.sum() * 100
    test_pct = test_counts_m / test_counts_m.sum() * 100

    P(f"    {'Class':>15s}  {'Train%':>7s}  {'TestPred%':>9s}  {'Diff':>6s}")
    P(f"    {'-'*15}  {'-'*7}  {'-'*9}  {'-'*6}")
    for i, cls in enumerate(CLASSES):
        diff = test_pct[i] - train_pct[i]
        P(f"    {cls:>15s}  {train_pct[i]:6.1f}%  {test_pct[i]:8.1f}%  {diff:+5.1f}%")
    P()

# 4d: Mean predicted probabilities per test month (not argmax, but soft probs)
P("\n--- 4d: Mean predicted probabilities per test month ---\n")

P(f"  {'Class':>15s}", end="")
for m in sorted(test_month_counts.keys()):
    P(f"  M{m:02d}", end="")
P()
P(f"  {'-'*15}", end="")
for m in sorted(test_month_counts.keys()):
    P(f"  ------", end="")
P()

for i, cls in enumerate(CLASSES):
    P(f"  {cls:>15s}", end="")
    for m in sorted(test_month_counts.keys()):
        m_mask = test_months == m
        mean_prob = np.mean(sub_probs[m_mask, i])
        P(f"  {mean_prob:.4f}", end="")
    P()

# ======================================================================
# PART 5: OPTIMAL LOMO WEIGHTING
# ======================================================================
P("\n" + "=" * 70)
P("PART 5: OPTIMAL LOMO WEIGHTING FOR TEST-LIKE EVALUATION")
P("=" * 70)

# 5a: Current equal-weight LOMO
P("\n--- 5a: Equal-weight LOMO (current method) ---\n")

equal_lomo_maps = {}
for m in unique_months:
    equal_lomo_maps[m] = fold_results[m]["macro_map"]
    P(f"  Month {m:2d}: mAP = {fold_results[m]['macro_map']:.4f}")

equal_lomo = np.mean(list(equal_lomo_maps.values()))
P(f"\n  Equal-weight LOMO = {equal_lomo:.4f}")

# 5b: Test month proportions
P("\n--- 5b: Test month proportions ---\n")

# Shared months: Sep and Oct. Unseen months: Feb, May, Dec
# For LOMO, we only have folds for months 1, 4, 9, 10
# Shared with test: 9, 10
# Not in test: 1, 4
# In test but not train: 2, 5, 12

test_total = len(test_df)
P(f"  Test composition:")
P(f"    Sep: {test_month_counts.get(9, 0)} ({test_month_counts.get(9, 0)/test_total*100:.1f}%)")
P(f"    Oct: {test_month_counts.get(10, 0)} ({test_month_counts.get(10, 0)/test_total*100:.1f}%)")
P(f"    Feb: {test_month_counts.get(2, 0)} ({test_month_counts.get(2, 0)/test_total*100:.1f}%)")
P(f"    May: {test_month_counts.get(5, 0)} ({test_month_counts.get(5, 0)/test_total*100:.1f}%)")
P(f"    Dec: {test_month_counts.get(12, 0)} ({test_month_counts.get(12, 0)/test_total*100:.1f}%)")

# 5c: Weighted LOMO using only shared months (Sep/Oct weight by test proportion)
P("\n--- 5c: Weighted LOMO variants ---\n")

# Approach 1: Only weight shared months (Sep, Oct) by their test proportions
sep_weight = test_month_counts.get(9, 0) / test_total
oct_weight = test_month_counts.get(10, 0) / test_total
unseen_weight = 1 - sep_weight - oct_weight

P(f"  Approach 1: Weight shared months by test proportions")
P(f"    Sep weight: {sep_weight:.3f}, Oct weight: {oct_weight:.3f}")
P(f"    Unseen months weight: {unseen_weight:.3f}")

# For unseen months, use average of Jan and Apr as proxy (also "unseen" from perspective of 9,10)
jan_map = fold_results[1]["macro_map"]
apr_map = fold_results[4]["macro_map"]
sep_map = fold_results[9]["macro_map"]
oct_map = fold_results[10]["macro_map"]

# Weighted: shared months weighted by test proportion, unseen proxied by Jan/Apr
# Normalize: Sep/(Sep+Oct) for shared portion, Jan+Apr for unseen portion
shared_total_weight = sep_weight + oct_weight
if shared_total_weight > 0:
    shared_lomo = (sep_weight * sep_map + oct_weight * oct_map) / shared_total_weight
else:
    shared_lomo = (sep_map + oct_map) / 2

unseen_lomo = (jan_map + apr_map) / 2

# Various blend weights
P(f"\n  Shared months (Sep+Oct) LOMO = {shared_lomo:.4f}")
P(f"  Unseen proxy (Jan+Apr)  LOMO = {unseen_lomo:.4f}")

for shared_frac in [0.67, 0.75, 0.80, 1.0]:
    unseen_frac = 1 - shared_frac
    weighted = shared_frac * shared_lomo + unseen_frac * unseen_lomo
    P(f"    {shared_frac*100:.0f}% shared + {unseen_frac*100:.0f}% unseen -> weighted LOMO = {weighted:.4f}")

# Approach 2: Weight all 4 folds individually
P(f"\n  Approach 2: Weight each fold individually")
P(f"    Scenario: Test-proportional weighting")

# Month weights reflecting test importance
# Sep and Oct are in test, Jan and Apr are not
# Weight Sep/Oct proportional to test size, Jan/Apr as unseen proxies
scenarios = {
    "Equal (current)": {1: 0.25, 4: 0.25, 9: 0.25, 10: 0.25},
    "Test-proportional (shared only)": {
        1: 0.0, 4: 0.0,
        9: sep_weight / shared_total_weight if shared_total_weight > 0 else 0.5,
        10: oct_weight / shared_total_weight if shared_total_weight > 0 else 0.5
    },
    "67% shared + 33% unseen proxy": {
        1: 0.33 * 0.5, 4: 0.33 * 0.5,
        9: 0.67 * (sep_weight / shared_total_weight) if shared_total_weight > 0 else 0.335,
        10: 0.67 * (oct_weight / shared_total_weight) if shared_total_weight > 0 else 0.335
    },
    "Sep-Oct only (optimistic)": {
        1: 0.0, 4: 0.0, 9: 0.5, 10: 0.5
    },
    "Oct-heavy (best month)": {
        1: 0.1, 4: 0.1, 9: 0.2, 10: 0.6
    },
}

P(f"\n  {'Scenario':>40s}  {'M01':>6s}  {'M04':>6s}  {'M09':>6s}  {'M10':>6s}  {'LOMO':>7s}")
P(f"  {'-'*40}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")

for name, weights in scenarios.items():
    weighted_map = sum(weights[m] * fold_results[m]["macro_map"] for m in unique_months)
    w1 = weights.get(1, 0)
    w4 = weights.get(4, 0)
    w9 = weights.get(9, 0)
    w10 = weights.get(10, 0)
    P(f"  {name:>40s}  {w1:.3f}  {w4:.3f}  {w9:.3f}  {w10:.3f}  {weighted_map:.4f}")

# 5d: Per-class analysis -- which classes are most affected by weighting?
P("\n--- 5d: Per-class AP under different LOMO weightings ---\n")

P(f"  {'Class':>15s}  {'Equal':>7s}  {'SharedOnly':>10s}  {'67/33':>7s}  {'OctHeavy':>8s}")
P(f"  {'-'*15}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*8}")

weight_sets = {
    "Equal": {1: 0.25, 4: 0.25, 9: 0.25, 10: 0.25},
    "SharedOnly": scenarios["Test-proportional (shared only)"],
    "67/33": scenarios["67% shared + 33% unseen proxy"],
    "OctHeavy": scenarios["Oct-heavy (best month)"],
}

for cls in CLASSES:
    P(f"  {cls:>15s}", end="")
    for wname, weights in weight_sets.items():
        weighted_ap = sum(weights[m] * fold_results[m]["per_class_ap"][cls] for m in unique_months)
        P(f"  {weighted_ap:7.4f}" if wname == "Equal" else f"  {weighted_ap:{'10' if wname == 'SharedOnly' else '7' if wname == '67/33' else '8'}.4f}", end="")
    P()

# ======================================================================
# SUMMARY & INSIGHTS
# ======================================================================
P("\n" + "=" * 70)
P("SUMMARY OF KEY FINDINGS")
P("=" * 70)

P("\n1. SEPTEMBER IS WORST BECAUSE:")
P(f"   - Sep mAP = {fold_results[9]['macro_map']:.4f} vs Oct mAP = {fold_results[10]['macro_map']:.4f}")

# Find which classes collapse in Sep
for cls in CLASSES:
    sep_ap = fold_results[9]["per_class_ap"][cls]
    oct_ap = fold_results[10]["per_class_ap"][cls]
    if sep_ap < 0.20 and oct_ap > sep_ap + 0.15:
        P(f"   - {cls}: Sep AP={sep_ap:.3f}, Oct AP={oct_ap:.3f} (collapses in Sep)")

P(f"\n2. CLASS DISTRIBUTION VARIES HUGELY BETWEEN MONTHS:")
for m in unique_months:
    mask = train_months == m
    m_counts = np.bincount(y[mask], minlength=N_CLASSES)
    gull_pct = m_counts[CLASSES.index("Gulls")] / m_counts.sum() * 100
    P(f"   - Month {m:2d}: Gulls = {gull_pct:.1f}%, total = {m_counts.sum()}")

P(f"\n3. TEST COMPOSITION:")
P(f"   - Shared (Sep+Oct): {shared_count}/{test_total} = {shared_count/test_total*100:.1f}%")
P(f"   - Unseen (Feb+May+Dec): {unseen_count}/{test_total} = {unseen_count/test_total*100:.1f}%")
P(f"   - Oct dominates test: {test_month_counts.get(10, 0)}/{test_total} = {test_month_counts.get(10, 0)/test_total*100:.1f}%")

P(f"\n4. RECOMMENDED LOMO WEIGHTING:")
# Best estimate: weight by test month proportions, use Jan/Apr as unseen proxy
best_weighted = 0.67 * shared_lomo + 0.33 * unseen_lomo
P(f"   - 67% shared + 33% unseen proxy = {best_weighted:.4f}")
P(f"   - vs current equal LOMO = {equal_lomo:.4f}")
P(f"   - Difference: {best_weighted - equal_lomo:+.4f}")

P(f"\n5. TOP FEATURES SHIFTING IN SEPTEMBER:")
for rank, idx in enumerate(top_shift_idx[:5], 1):
    P(f"   {rank}. {fn[idx]} (Z-shift = {sep_shift[idx]:.2f})")

P("\n" + "=" * 70)
P("ANALYSIS COMPLETE")
P("=" * 70)
