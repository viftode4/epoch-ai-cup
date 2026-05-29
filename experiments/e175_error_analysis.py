"""E175 Error Analysis — What does the model get wrong and why?"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
NC = len(CLASSES)

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

# Load best OOF predictions
oof = np.load(ROOT / "oof_e175_best.npy")
# Also load individual models
oof_lgb = np.load(ROOT / "oof_e175_lgb.npy")
oof_dro = np.load(ROOT / "oof_e175_dro.npy")

print("=" * 80)
print("  ERROR ANALYSIS — What goes wrong?")
print("=" * 80)

# ═══ 1. CONFUSION MATRIX ═══
print("\n[1] CONFUSION MATRIX (OOF predictions)")
pred_classes = np.argmax(oof, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, pred_classes)
print(f"\n  {'True \\ Pred':>15s}", end="")
for c in CLASSES:
    print(f" {c[:5]:>6s}", end="")
print(f" {'Acc':>6s}")

for i, cls in enumerate(CLASSES):
    print(f"  {cls:>15s}", end="")
    for j in range(NC):
        val = cm[i, j]
        marker = "*" if i != j and val > 5 else " "
        print(f" {val:>5d}{marker}", end="")
    acc = cm[i, i] / max(cm[i].sum(), 1) * 100
    print(f" {acc:>5.1f}%")

# ═══ 2. PER-CLASS AP AND WORST SAMPLES ═══
print("\n[2] PER-CLASS ANALYSIS")
from sklearn.metrics import average_precision_score

for cls_idx, cls in enumerate(CLASSES):
    y_bin = (y == cls_idx).astype(int)
    ap = average_precision_score(y_bin, oof[:, cls_idx])
    n_true = y_bin.sum()

    # Find false negatives (true class but low prediction)
    true_mask = y == cls_idx
    pred_scores = oof[true_mask, cls_idx]
    pred_top1 = np.argmax(oof[true_mask], axis=1)

    # What do FNs get classified as?
    fn_mask = pred_top1 != cls_idx
    fn_count = fn_mask.sum()
    if fn_count > 0:
        fn_predicted = pred_top1[fn_mask]
        fn_classes = [CLASSES[c] for c in fn_predicted]
        from collections import Counter
        fn_dist = Counter(fn_classes).most_common(5)
        fn_str = ", ".join(f"{c}={n}" for c, n in fn_dist)
    else:
        fn_str = "none"

    # Per-month breakdown
    month_aps = {}
    for m in sorted(set(months)):
        mask = (months == m) & (y == cls_idx)
        if mask.sum() >= 2:
            lm = average_precision_score(y_bin[months == m], oof[months == m, cls_idx])
            month_aps[m] = (lm, int(mask.sum()))

    month_str = "  ".join(f"M{m}:{ap:.2f}(n={n})" for m, (ap, n) in sorted(month_aps.items()))

    print(f"\n  {cls} (n={n_true}, AP={ap:.4f}):")
    print(f"    Misclassified as: {fn_str}")
    print(f"    Per-month: {month_str}")
    print(f"    Pred score on true samples: mean={pred_scores.mean():.3f}, "
          f"min={pred_scores.min():.3f}, median={np.median(pred_scores):.3f}")

# ═══ 3. WORST MONTH × CLASS COMBINATIONS ═══
print("\n\n[3] WORST MONTH × CLASS COMBINATIONS")
results = []
for m in sorted(set(months)):
    for cls_idx, cls in enumerate(CLASSES):
        mask = (months == m) & (y == cls_idx)
        n = mask.sum()
        if n < 2:
            continue
        y_bin = (y[months == m] == cls_idx).astype(int)
        ap = average_precision_score(y_bin, oof[months == m, cls_idx])
        results.append((ap, m, cls, n))

results.sort()
print(f"  Bottom 15 (worst AP):")
print(f"  {'Month':>6s} {'Class':>15s} {'N':>5s} {'AP':>7s}")
for ap, m, cls, n in results[:15]:
    name = {1: "Jan", 4: "Apr", 9: "Sep", 10: "Oct"}.get(m, f"M{m}")
    print(f"  {name:>6s} {cls:>15s} {n:>5d} {ap:>7.4f}")

# ═══ 4. HARDEST INDIVIDUAL SAMPLES ═══
print("\n[4] HARDEST SAMPLES (true class prob < 0.1)")
hard_mask = np.array([oof[i, y[i]] < 0.1 for i in range(len(y))])
hard_idx = np.where(hard_mask)[0]
print(f"  Total hard samples: {hard_idx.shape[0]} / {len(y)} ({100*hard_idx.shape[0]/len(y):.1f}%)")

hard_by_class = {}
for i in hard_idx:
    cls = CLASSES[y[i]]
    hard_by_class.setdefault(cls, []).append(i)

print(f"  By class:")
for cls in CLASSES:
    n_hard = len(hard_by_class.get(cls, []))
    n_total = (y == CLASSES.index(cls)).sum()
    pct = 100 * n_hard / max(n_total, 1)
    print(f"    {cls:15s}: {n_hard:4d} / {n_total:4d} ({pct:5.1f}%)")

# ═══ 5. MODEL DISAGREEMENT ═══
print("\n[5] MODEL DISAGREEMENT (LGB DART vs CB DRO)")
pred_lgb = np.argmax(oof_lgb, axis=1)
pred_dro = np.argmax(oof_dro, axis=1)
disagree = pred_lgb != pred_dro
print(f"  Disagreements: {disagree.sum()} / {len(y)} ({100*disagree.sum()/len(y):.1f}%)")

# When they disagree, which is right?
lgb_right = (pred_lgb[disagree] == y[disagree]).sum()
dro_right = (pred_dro[disagree] == y[disagree]).sum()
neither = disagree.sum() - lgb_right - dro_right
print(f"  LGB right: {lgb_right}, DRO right: {dro_right}, Neither: {neither}")

# Disagreement by class
print(f"  Disagreement by true class:")
for cls_idx, cls in enumerate(CLASSES):
    mask = (y == cls_idx) & disagree
    n = mask.sum()
    n_total = (y == cls_idx).sum()
    print(f"    {cls:15s}: {n:4d} / {n_total:4d} ({100*n/max(n_total,1):5.1f}%)")

# ═══ 6. FEATURE ANALYSIS ON HARD SAMPLES ═══
print("\n[6] WHAT MAKES HARD SAMPLES DIFFERENT?")
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")

# Key features to check
key_feats = ["airspeed", "alt_mean", "rcs_mean_dB", "straightness", "heading_std",
             "speed_median", "radar_bird_size", "duration", "n_points"]
key_feats = [f for f in key_feats if f in train_v3.columns]

for cls_idx, cls in enumerate(CLASSES):
    hard_ids = hard_by_class.get(cls, [])
    if len(hard_ids) < 3:
        continue
    all_ids = np.where(y == cls_idx)[0]
    easy_ids = [i for i in all_ids if i not in set(hard_ids)]

    print(f"\n  {cls} (hard={len(hard_ids)}, easy={len(easy_ids)}):")
    for feat in key_feats:
        hard_vals = train_v3.iloc[hard_ids][feat].values
        easy_vals = train_v3.iloc[easy_ids][feat].values
        h_mean = np.mean(hard_vals)
        e_mean = np.mean(easy_vals)
        diff = h_mean - e_mean
        if abs(diff) > 0.01 * max(abs(e_mean), 1):
            print(f"    {feat:20s}: hard={h_mean:8.2f}, easy={e_mean:8.2f}, diff={diff:+8.2f}")

# ═══ 7. MONTH SHIFT ANALYSIS ═══
print("\n\n[7] MONTH SHIFT — What changes between months?")
for cls_idx, cls in enumerate(CLASSES):
    cls_mask = y == cls_idx
    if cls_mask.sum() < 10:
        continue
    month_means = {}
    for m in sorted(set(months)):
        mask = cls_mask & (months == m)
        if mask.sum() >= 3:
            month_means[m] = {}
            for feat in ["airspeed", "alt_mean", "rcs_mean_dB"]:
                if feat in train_v3.columns:
                    month_means[m][feat] = train_v3.iloc[np.where(mask)[0]][feat].mean()

    if len(month_means) >= 2:
        print(f"\n  {cls}:")
        for feat in ["airspeed", "alt_mean", "rcs_mean_dB"]:
            if feat in train_v3.columns:
                vals = {m: d.get(feat, 0) for m, d in month_means.items()}
                spread = max(vals.values()) - min(vals.values())
                vals_str = "  ".join(f"M{m}={v:.1f}" for m, v in sorted(vals.items()))
                flag = " !! HIGH SHIFT" if spread > 0.3 * abs(np.mean(list(vals.values()))) else ""
                print(f"    {feat:20s}: {vals_str}{flag}")

print(f"\n{'='*80}")
print("  ERROR ANALYSIS COMPLETE")
print(f"{'='*80}")
