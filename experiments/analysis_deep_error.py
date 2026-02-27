"""Deep error analysis on E79 OOF + test predictions.

Goal: understand WHERE and WHY we're wrong, not optimize.
Outputs: confusion patterns, outliers, per-month breakdowns,
         confidence analysis, trajectory-level patterns.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, load_test, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

print("=" * 70, flush=True)
print("DEEP ERROR ANALYSIS".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load everything --
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

oof_e79 = np.load(ROOT / "oof_e79.npy", allow_pickle=True).astype(float)
test_e79 = np.load(ROOT / "test_e79.npy", allow_pickle=True).astype(float)

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# =====================================================================
# PART 1: CONFUSION MATRIX -- WHO GETS CONFUSED WITH WHOM?
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 1: CONFUSION MATRIX".center(70), flush=True)
print("=" * 70, flush=True)

y_pred_cls = oof_e79.argmax(1)
cm = confusion_matrix(y, y_pred_cls)

print("\nPredicted -->", flush=True)
header = "True \\ Pred  " + "".join(f"{CLASSES[i][:6]:>8s}" for i in range(N_CLASSES))
print(header, flush=True)
for i in range(N_CLASSES):
    row = f"{CLASSES[i]:14s}" + "".join(f"{cm[i, j]:8d}" for j in range(N_CLASSES))
    print(row, flush=True)

# Percentage confusion (row-normalized)
print("\n--- Row-normalized (where do true samples go?) ---", flush=True)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
header = "True \\ Pred  " + "".join(f"{CLASSES[i][:6]:>8s}" for i in range(N_CLASSES))
print(header, flush=True)
for i in range(N_CLASSES):
    row = f"{CLASSES[i]:14s}" + "".join(f"{cm_pct[i, j]:7.1f}%" for j in range(N_CLASSES))
    print(row, flush=True)

# Top confusions
print("\n--- Top 15 confusions (off-diagonal) ---", flush=True)
confusions = []
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j and cm[i, j] > 0:
            confusions.append((cm[i, j], CLASSES[i], CLASSES[j], cm_pct[i, j]))
confusions.sort(reverse=True)
for count, true_cls, pred_cls, pct in confusions[:15]:
    print(f"  {true_cls:15s} -> {pred_cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)

# =====================================================================
# PART 2: PER-CLASS ERROR ANALYSIS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 2: PER-CLASS AP AND ERROR BREAKDOWN".center(70), flush=True)
print("=" * 70, flush=True)

overall_map, per_class_ap = compute_map(y, oof_e79)
print(f"\nOverall mAP: {overall_map:.4f}", flush=True)
print(f"\n{'Class':15s} {'AP':>6s} {'N':>5s} {'Acc':>6s} {'Top confusor':>15s} {'Lost to':>8s}", flush=True)
print("-" * 65, flush=True)
for i, cls in enumerate(CLASSES):
    mask = y == i
    n = int(mask.sum())
    correct = int((y_pred_cls[mask] == i).sum())
    acc = correct / n if n > 0 else 0
    # Top confusor
    wrong_mask = mask & (y_pred_cls != i)
    if wrong_mask.sum() > 0:
        wrong_preds = y_pred_cls[wrong_mask]
        top_conf_idx = Counter(wrong_preds).most_common(1)[0]
        top_conf = CLASSES[top_conf_idx[0]]
        top_conf_n = top_conf_idx[1]
    else:
        top_conf, top_conf_n = "-", 0
    print(f"  {cls:15s} {per_class_ap[cls]:.4f} {n:5d} {acc:5.1%} {top_conf:>15s} {top_conf_n:8d}", flush=True)

# =====================================================================
# PART 3: PER-MONTH BREAKDOWN (LOMO perspective)
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 3: PER-MONTH OOF PERFORMANCE".center(70), flush=True)
print("=" * 70, flush=True)

for month in sorted(np.unique(train_months)):
    mask = train_months == month
    n = int(mask.sum())
    if n == 0:
        continue
    m, per = compute_map(y[mask], oof_e79[mask])
    classes_present = sorted(set(y[mask]))
    classes_absent = [CLASSES[c] for c in range(N_CLASSES) if c not in classes_present]
    print(f"\n  Month {month:2d}: n={n:4d}, mAP={m:.4f} (classes absent: {classes_absent})", flush=True)
    # Show per-class for this month
    for c in range(N_CLASSES):
        c_mask = y[mask] == c
        nc = int(c_mask.sum())
        if nc == 0:
            print(f"    {CLASSES[c]:15s}: n=0 (absent)", flush=True)
        else:
            ap = average_precision_score((y[mask] == c).astype(int), oof_e79[mask, c])
            pred_top1 = oof_e79[mask].argmax(1)
            correct = int((pred_top1[c_mask] == c).sum())
            print(f"    {CLASSES[c]:15s}: n={nc:4d} AP={ap:.4f} acc={correct}/{nc}", flush=True)

# =====================================================================
# PART 4: CONFIDENCE AND MARGIN ANALYSIS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 4: CONFIDENCE ANALYSIS".center(70), flush=True)
print("=" * 70, flush=True)

top1_prob = oof_e79.max(axis=1)
top1_cls = oof_e79.argmax(axis=1)
sorted_probs = np.sort(oof_e79, axis=1)[:, ::-1]
margin = sorted_probs[:, 0] - sorted_probs[:, 1]

correct = (top1_cls == y)
print(f"\n  Overall top-1 accuracy: {correct.mean():.4f} ({int(correct.sum())}/{len(y)})", flush=True)

# Confidence bins
print("\n  --- Accuracy by confidence bin ---", flush=True)
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
    mask = (top1_prob >= lo) & (top1_prob < hi)
    n = int(mask.sum())
    if n > 0:
        acc = correct[mask].mean()
        print(f"    conf [{lo:.1f}, {hi:.1f}): n={n:5d} acc={acc:.4f}", flush=True)

# Margin bins
print("\n  --- Accuracy by margin (top1-top2) ---", flush=True)
for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]:
    mask = (margin >= lo) & (margin < hi)
    n = int(mask.sum())
    if n > 0:
        acc = correct[mask].mean()
        print(f"    margin [{lo:.2f}, {hi:.2f}): n={n:5d} acc={acc:.4f}", flush=True)

# =====================================================================
# PART 5: THE HARDEST SAMPLES -- CONFIDENT AND WRONG
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 5: HARDEST SAMPLES (confident but wrong)".center(70), flush=True)
print("=" * 70, flush=True)

wrong = ~correct
conf_wrong = wrong & (top1_prob > 0.5)
print(f"\n  Confident (p>0.5) and WRONG: {int(conf_wrong.sum())} samples", flush=True)

if conf_wrong.sum() > 0:
    idx_cw = np.where(conf_wrong)[0]
    # Sort by confidence (most confident wrong first)
    sort_order = np.argsort(-top1_prob[idx_cw])
    idx_cw = idx_cw[sort_order]

    print(f"\n  Top 30 most confidently wrong:", flush=True)
    print(f"  {'#':>3s} {'TrueClass':>15s} {'PredClass':>15s} {'Conf':>6s} {'Margin':>7s} {'Month':>5s} {'Speed':>7s} {'Size':>12s} {'MinZ':>6s} {'MaxZ':>6s}", flush=True)
    print("  " + "-" * 95, flush=True)
    for rank, idx in enumerate(idx_cw[:30]):
        true_cls = CLASSES[y[idx]]
        pred_cls = CLASSES[top1_cls[idx]]
        conf = top1_prob[idx]
        marg = margin[idx]
        month = train_months[idx]
        speed = train_df.iloc[idx]["airspeed"]
        size = train_df.iloc[idx]["radar_bird_size"]
        min_z = train_df.iloc[idx]["min_z"]
        max_z = train_df.iloc[idx]["max_z"]
        print(f"  {rank+1:3d} {true_cls:>15s} {pred_cls:>15s} {conf:6.3f} {marg:7.3f} {month:5d} {str(speed):>7s} {str(size):>12s} {str(min_z):>6s} {str(max_z):>6s}", flush=True)

# =====================================================================
# PART 6: PER-CLASS DEEP DIVE -- WHAT FEATURES DISTINGUISH ERRORS?
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 6: ERROR FEATURE ANALYSIS (per class)".center(70), flush=True)
print("=" * 70, flush=True)

# For each weak class, compare correctly vs incorrectly classified samples
weak_classes = [i for i, cls in enumerate(CLASSES) if per_class_ap[cls] < 0.75]
feature_cols = ["airspeed", "min_z", "max_z", "radar_bird_size"]

for cls_idx in weak_classes:
    cls_name = CLASSES[cls_idx]
    mask_true = y == cls_idx
    mask_correct = mask_true & correct
    mask_wrong = mask_true & wrong

    n_true = int(mask_true.sum())
    n_correct = int(mask_correct.sum())
    n_wrong = int(mask_wrong.sum())

    print(f"\n  === {cls_name} (n={n_true}, correct={n_correct}, wrong={n_wrong}) ===", flush=True)

    if n_wrong == 0:
        print(f"    No errors!", flush=True)
        continue

    # Where do the wrong ones go?
    wrong_preds = Counter(y_pred_cls[mask_wrong].tolist())
    print(f"    Wrong predictions go to:", flush=True)
    for pred_cls_idx, count in wrong_preds.most_common():
        print(f"      -> {CLASSES[pred_cls_idx]:15s}: {count:3d}", flush=True)

    # Feature comparison: correct vs wrong
    print(f"    Feature comparison (correct vs wrong):", flush=True)
    for col in feature_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce")
        v_correct = vals[mask_correct].dropna()
        v_wrong = vals[mask_wrong].dropna()
        if len(v_correct) > 0 and len(v_wrong) > 0:
            print(f"      {col:15s}: correct={v_correct.mean():.2f}+/-{v_correct.std():.2f} "
                  f"wrong={v_wrong.mean():.2f}+/-{v_wrong.std():.2f}", flush=True)

    # Size distribution
    size_correct = train_df.loc[mask_correct, "radar_bird_size"].value_counts()
    size_wrong = train_df.loc[mask_wrong, "radar_bird_size"].value_counts()
    print(f"    Size distribution:", flush=True)
    print(f"      Correct: {dict(size_correct)}", flush=True)
    print(f"      Wrong:   {dict(size_wrong)}", flush=True)

    # Month distribution of errors
    month_correct = Counter(train_months[mask_correct])
    month_wrong = Counter(train_months[mask_wrong])
    print(f"    Month distribution:", flush=True)
    print(f"      Correct: {dict(sorted(month_correct.items()))}", flush=True)
    print(f"      Wrong:   {dict(sorted(month_wrong.items()))}", flush=True)

# =====================================================================
# PART 7: TRAJECTORY-LEVEL ANALYSIS -- RCS PATTERNS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 7: TRAJECTORY RCS PATTERNS (per class)".center(70), flush=True)
print("=" * 70, flush=True)

# For each class, extract RCS time series stats that we DON'T currently use
rcs_patterns = {cls: {"autocorr_lag1": [], "autocorr_lag2": [], "n_peaks": [],
                       "rcs_oscillation_freq": [], "rcs_trend": [], "n_pts": [],
                       "speed_variability": [], "alt_trend": []}
                for cls in CLASSES}

print("\nExtracting trajectory patterns (2601 tracks)...", flush=True)
for idx in range(len(train_df)):
    if idx % 500 == 0:
        print(f"  {idx}/{len(train_df)}...", flush=True)

    row = train_df.iloc[idx]
    cls = row["bird_group"]
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])
    except Exception:
        continue

    rcs = np.array([p[3] for p in pts])
    alts = np.array([p[2] for p in pts])
    n = len(rcs)
    rcs_patterns[cls]["n_pts"].append(n)

    if n < 5:
        continue

    # RCS autocorrelation (proxy for wingbeat regularity)
    rcs_centered = rcs - rcs.mean()
    var = np.var(rcs_centered)
    if var > 1e-8:
        ac1 = np.correlate(rcs_centered[:-1], rcs_centered[1:])[0] / (var * (n - 1))
        ac2 = np.correlate(rcs_centered[:-2], rcs_centered[2:])[0] / (var * (n - 2)) if n > 4 else 0
        rcs_patterns[cls]["autocorr_lag1"].append(ac1)
        rcs_patterns[cls]["autocorr_lag2"].append(ac2)

    # RCS oscillation: count zero-crossings of centered RCS
    crossings = np.sum(np.diff(np.sign(rcs_centered)) != 0)
    duration = times[-1] - times[0] if times[-1] > times[0] else 1.0
    rcs_patterns[cls]["rcs_oscillation_freq"].append(crossings / (2.0 * duration))

    # RCS trend (linear slope)
    if n > 2:
        t = np.array(times) - times[0]
        if t[-1] > 0:
            slope = np.polyfit(t, rcs, 1)[0]
            rcs_patterns[cls]["rcs_trend"].append(slope)

    # Altitude trend
    if n > 2:
        t = np.array(times) - times[0]
        if t[-1] > 0:
            alt_slope = np.polyfit(t, alts, 1)[0]
            rcs_patterns[cls]["alt_trend"].append(alt_slope)

    # Speed variability (coefficient of variation of segment speeds)
    if n > 2:
        dt = np.diff(times)
        dalt = np.diff(alts)
        # Use altitude rate as proxy for vertical speed variability
        valid = dt > 0
        if valid.sum() > 2:
            vert_speed = dalt[valid] / dt[valid]
            cv = np.std(vert_speed) / (np.abs(np.mean(vert_speed)) + 0.01)
            rcs_patterns[cls]["speed_variability"].append(cv)

    # Count RCS peaks (local maxima)
    if n > 3:
        peaks = 0
        for i in range(1, n - 1):
            if rcs[i] > rcs[i - 1] and rcs[i] > rcs[i + 1]:
                peaks += 1
        rcs_patterns[cls]["n_peaks"].append(peaks)

print("\n--- RCS and trajectory pattern summary per class ---", flush=True)
print(f"\n{'Class':15s} {'N_pts':>6s} {'AC_lag1':>8s} {'AC_lag2':>8s} {'OscFreq':>8s} "
      f"{'RCS_trend':>10s} {'Alt_trend':>10s} {'SpeedVar':>9s} {'N_peaks':>8s}", flush=True)
print("-" * 95, flush=True)

for cls in CLASSES:
    p = rcs_patterns[cls]
    def safe_mean(arr):
        return f"{np.mean(arr):.4f}" if len(arr) > 0 else "  N/A"
    print(f"  {cls:15s} {safe_mean(p['n_pts']):>6s} {safe_mean(p['autocorr_lag1']):>8s} "
          f"{safe_mean(p['autocorr_lag2']):>8s} {safe_mean(p['rcs_oscillation_freq']):>8s} "
          f"{safe_mean(p['rcs_trend']):>10s} {safe_mean(p['alt_trend']):>10s} "
          f"{safe_mean(p['speed_variability']):>9s} {safe_mean(p['n_peaks']):>8s}", flush=True)

# Also show std to see if features are discriminative
print(f"\n{'Class':15s} {'AC1_std':>8s} {'OscF_std':>8s} {'AltTr_std':>10s} {'SpVar_std':>9s}", flush=True)
print("-" * 55, flush=True)
for cls in CLASSES:
    p = rcs_patterns[cls]
    def safe_std(arr):
        return f"{np.std(arr):.4f}" if len(arr) > 1 else "  N/A"
    print(f"  {cls:15s} {safe_std(p['autocorr_lag1']):>8s} {safe_std(p['rcs_oscillation_freq']):>8s} "
          f"{safe_std(p['alt_trend']):>10s} {safe_std(p['speed_variability']):>9s}", flush=True)

# =====================================================================
# PART 8: TEST PREDICTION ANALYSIS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 8: TEST PREDICTION STRUCTURE".center(70), flush=True)
print("=" * 70, flush=True)

test_top1 = test_e79.argmax(1)
test_top1_prob = test_e79.max(1)
test_margin = np.sort(test_e79, axis=1)[:, -1] - np.sort(test_e79, axis=1)[:, -2]

# Per-month test prediction distribution
print("\n--- Test top-1 prediction distribution per month ---", flush=True)
for month in sorted(np.unique(test_months)):
    mask = test_months == month
    n = int(mask.sum())
    dist = Counter(test_top1[mask].tolist())
    print(f"\n  Month {month:2d} (n={n:4d}, {'SHARED' if month in (9, 10) else 'UNSEEN'}):", flush=True)
    for c in range(N_CLASSES):
        count = dist.get(c, 0)
        pct = count / n * 100
        avg_prob = test_e79[mask, c].mean()
        bar = "#" * int(pct / 2)
        print(f"    {CLASSES[c]:15s}: {count:4d} ({pct:5.1f}%) avg_p={avg_prob:.3f} {bar}", flush=True)

# Confidence per month
print("\n--- Test confidence per month ---", flush=True)
for month in sorted(np.unique(test_months)):
    mask = test_months == month
    n = int(mask.sum())
    mean_conf = test_top1_prob[mask].mean()
    mean_margin = test_margin[mask].mean()
    low_conf = (test_top1_prob[mask] < 0.5).sum()
    print(f"  Month {month:2d}: n={n:4d} avg_conf={mean_conf:.3f} avg_margin={mean_margin:.3f} "
          f"low_conf(<0.5)={low_conf:3d} ({low_conf/n*100:.1f}%)", flush=True)

# =====================================================================
# PART 9: TRAIN vs TEST FEATURE DISTRIBUTION COMPARISON
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 9: TRAIN vs TEST DISTRIBUTION BY MONTH".center(70), flush=True)
print("=" * 70, flush=True)

numeric_cols = ["airspeed", "min_z", "max_z"]
for col in numeric_cols:
    train_vals = pd.to_numeric(train_df[col], errors="coerce")
    test_vals = pd.to_numeric(test_df[col], errors="coerce")
    print(f"\n  {col}:", flush=True)
    print(f"    Train overall: mean={train_vals.mean():.2f} std={train_vals.std():.2f} "
          f"median={train_vals.median():.2f}", flush=True)
    print(f"    Test  overall: mean={test_vals.mean():.2f} std={test_vals.std():.2f} "
          f"median={test_vals.median():.2f}", flush=True)
    for month in sorted(np.unique(test_months)):
        t_mask = test_months == month
        v = test_vals[t_mask].dropna()
        # Compare to closest train month
        tr_month_vals = train_vals[train_months == month].dropna() if month in train_months else None
        tr_str = ""
        if tr_month_vals is not None and len(tr_month_vals) > 0:
            tr_str = f" (train m{month}: mean={tr_month_vals.mean():.2f})"
        print(f"    Test month {month:2d}: mean={v.mean():.2f} std={v.std():.2f} n={len(v)}{tr_str}", flush=True)

# Size distribution
print("\n  radar_bird_size per month:", flush=True)
for month in sorted(np.unique(test_months)):
    t_mask = test_months == month
    dist = test_df.loc[t_mask, "radar_bird_size"].value_counts().to_dict()
    print(f"    Test month {month:2d}: {dist}", flush=True)

# =====================================================================
# PART 10: BLEND ANALYSIS -- E75 vs E79 test prediction differences
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("PART 10: E75 vs E79 TEST PREDICTION COMPARISON".center(70), flush=True)
print("=" * 70, flush=True)

# Load E75 submission to compare
e75_file = ROOT / "submissions" / "e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv"
if e75_file.exists():
    e75_df = pd.read_csv(e75_file)
    e75_probs = e75_df[CLASSES].values
    e79_sub_file = ROOT / "submissions" / "e79_pruned_tuned_base_0.7736_20260225_0201.csv"
    if e79_sub_file.exists():
        e79_df = pd.read_csv(e79_sub_file)
        e79_probs = e79_df[CLASSES].values

        top_e75 = e75_probs.argmax(1)
        top_e79 = e79_probs.argmax(1)
        differ = top_e75 != top_e79
        n_diff = int(differ.sum())
        print(f"\n  E75 vs E79 top-1 differences: {n_diff}/{len(top_e75)} ({n_diff/len(top_e75)*100:.1f}%)", flush=True)

        # Per-month differences
        for month in sorted(np.unique(test_months)):
            mask = test_months == month
            diff_m = int(differ[mask].sum())
            n_m = int(mask.sum())
            print(f"    Month {month:2d}: {diff_m}/{n_m} differ ({diff_m/n_m*100:.1f}%)", flush=True)

        # What classes change?
        print(f"\n  What changes (E75 -> E79):", flush=True)
        changes = Counter()
        for idx in np.where(differ)[0]:
            changes[(CLASSES[top_e75[idx]], CLASSES[top_e79[idx]])] += 1
        for (from_cls, to_cls), count in changes.most_common(15):
            print(f"    {from_cls:15s} -> {to_cls:15s}: {count:3d}", flush=True)

        # Blend test
        print(f"\n  --- Quick blend test (E75 and E79 raw test preds) ---", flush=True)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blend = alpha * e75_probs + (1 - alpha) * e79_probs
            top_blend = blend.argmax(1)
            n_diff_from_e75 = int((top_blend != top_e75).sum())
            n_diff_from_e79 = int((top_blend != top_e79).sum())
            # Distribution
            dist = Counter(top_blend.tolist())
            gulls = dist.get(5, 0)
            print(f"    alpha={alpha:.2f}: "
                  f"diff_from_e75={n_diff_from_e75:3d} diff_from_e79={n_diff_from_e79:3d} "
                  f"gulls={gulls}", flush=True)
    else:
        print("  E79 submission not found, skipping blend analysis", flush=True)
else:
    print("  E75 submission not found, skipping comparison", flush=True)

print("\n\nDone.", flush=True)
