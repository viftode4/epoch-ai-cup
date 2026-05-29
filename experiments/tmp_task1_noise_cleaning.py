"""Task 1: Validate noise cleaning impact (train-only removal).

Uses cleanlab with existing TabPFN + E175 OOF predictions to identify noisy labels.
Tests 3 variants with LightGBM (fast iteration):
  A) Remove 133 agreed noisy labels from TRAINING ONLY, keep in validation
  B) Relabel noise to TabPFN's predicted class
  C) Remove only extreme noise (quality < 0.05)

Reports per-class AP + LOMO for each variant.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train, load_test
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
MONTHS = [1, 4, 9, 10]

t0 = time.time()
print("=" * 90)
print("  TASK 1: Noise Cleaning Validation")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)


# ══════════════════════════════════════════════════════════════════════
# 1. Load data + identify noisy labels
# ══════════════════════════════════════════════════════════════════════

print("\n[1] Loading data and predictions...", flush=True)

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
n_train = len(y)

# Load cached v3 features + E175 selected features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(selected)}")

# Load OOF predictions from two different models
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")

print(f"  E175 OOF shape: {oof_e175.shape}")
print(f"  TabPFN OOF shape: {oof_tabpfn.shape}")


# ══════════════════════════════════════════════════════════════════════
# 2. Identify noisy labels with cleanlab
# ══════════════════════════════════════════════════════════════════════

print("\n[2] Running cleanlab noise detection...", flush=True)
from cleanlab.rank import get_label_quality_scores

# Get quality scores from both models
quality_e175 = get_label_quality_scores(labels=y, pred_probs=oof_e175, method="self_confidence")
quality_tabpfn = get_label_quality_scores(labels=y, pred_probs=oof_tabpfn, method="self_confidence")

# Agreed noisy: both models flag as noisy
# Use a threshold: quality < 0.5 means model disagrees with label
noisy_e175 = set(np.where(quality_e175 < 0.5)[0])
noisy_tabpfn = set(np.where(quality_tabpfn < 0.5)[0])
agreed_noisy = sorted(noisy_e175 & noisy_tabpfn)

# Extreme noise: quality < 0.05 in both
extreme_e175 = set(np.where(quality_e175 < 0.05)[0])
extreme_tabpfn = set(np.where(quality_tabpfn < 0.05)[0])
agreed_extreme = sorted(extreme_e175 & extreme_tabpfn)

print(f"  E175 noisy (q<0.5):  {len(noisy_e175)}")
print(f"  TabPFN noisy (q<0.5): {len(noisy_tabpfn)}")
print(f"  Agreed noisy:         {len(agreed_noisy)}")
print(f"  Extreme (q<0.05):     {len(agreed_extreme)}")

# Show per-class breakdown
print("\n  Per-class noise breakdown:")
print(f"  {'Class':15s}  {'Total':>6s}  {'Agreed':>7s}  {'%Noisy':>7s}  {'Extreme':>8s}  {'TabPFN pred':>12s}")
for c in range(N_CLASSES):
    class_mask = y == c
    n_class = class_mask.sum()
    n_noisy = sum(1 for i in agreed_noisy if y[i] == c)
    n_extreme = sum(1 for i in agreed_extreme if y[i] == c)
    # What does TabPFN think they should be?
    noisy_in_class = [i for i in agreed_noisy if y[i] == c]
    if noisy_in_class:
        tabpfn_preds = oof_tabpfn[noisy_in_class].argmax(axis=1)
        pred_counts = np.bincount(tabpfn_preds, minlength=N_CLASSES)
        top_pred = CLASSES[pred_counts.argmax()]
        top_n = pred_counts.max()
    else:
        top_pred = "-"
        top_n = 0
    pct = 100 * n_noisy / max(n_class, 1)
    print(f"  {CLASSES[c]:15s}  {n_class:6d}  {n_noisy:7d}  {pct:6.1f}%  {n_extreme:8d}  {top_pred}({top_n})")


# ══════════════════════════════════════════════════════════════════════
# 3. Helper: LGB training with noise handling
# ══════════════════════════════════════════════════════════════════════

def lomo_eval(oof, y_eval, months, label=""):
    """TRUE LOMO evaluation."""
    skf, per_class = compute_map(y_eval, oof)
    lomo_maps = {}
    for held in MONTHS:
        mask = months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y_eval[mask], oof[mask])
            lomo_maps[held] = lm
    lomo_avg = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  {label:40s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]", flush=True)
    return skf, lomo_avg, per_class


def train_lgb_with_noise_handling(
    X_train_full, y_full, groups_full, months_full,
    X_test, noise_indices, noise_mode="remove",
    relabel_probs=None, n_seeds=5, label="",
):
    """Train LGB DART with noise handling.

    noise_mode:
      - "remove": drop noise_indices from training folds only
      - "relabel": change labels to argmax of relabel_probs
      - "baseline": no noise handling (for comparison)

    Always keeps ALL samples in validation for honest evaluation.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_full = len(y_full)
    noise_set = set(noise_indices)

    # Prepare modified labels if relabeling
    y_modified = y_full.copy()
    if noise_mode == "relabel" and relabel_probs is not None:
        for idx in noise_indices:
            y_modified[idx] = relabel_probs[idx].argmax()
        n_changed = sum(1 for idx in noise_indices if y_modified[idx] != y_full[idx])
        print(f"    Relabeled {n_changed}/{len(noise_indices)} samples", flush=True)

    oof_all = np.zeros((n_seeds, n_full, N_CLASSES))
    test_all = np.zeros((n_seeds, X_test.shape[0], N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((n_full, N_CLASSES))
        test_s = np.zeros((X_test.shape[0], N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train_full, y_full, groups_full)):
            # CRITICAL: validation always uses ALL samples with ORIGINAL labels
            X_va = X_train_full[va_idx]
            y_va = y_full[va_idx]

            # Training: handle noise
            if noise_mode == "remove":
                # Remove noisy samples from training fold only
                clean_tr = np.array([i for i in tr_idx if i not in noise_set])
                X_tr = X_train_full[clean_tr]
                y_tr = y_full[clean_tr]
            elif noise_mode == "relabel":
                X_tr = X_train_full[tr_idx]
                y_tr = y_modified[tr_idx]
            else:  # baseline
                X_tr = X_train_full[tr_idx]
                y_tr = y_full[tr_idx]

            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

            oof_s[va_idx] = m.predict_proba(X_va)
            test_s += m.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)

    skf, lomo, per_class = lomo_eval(oof_mean, y_full, months_full, label)
    return oof_mean, test_mean, skf, lomo, per_class


# ══════════════════════════════════════════════════════════════════════
# 4. Run experiments
# ══════════════════════════════════════════════════════════════════════

N_SEEDS = 5
print(f"\n[3] Training LGB variants ({N_SEEDS} seeds each)...", flush=True)

# Baseline: no noise cleaning
print("\n  --- Variant 0: Baseline (no noise cleaning) ---", flush=True)
oof_base, test_base, skf_base, lomo_base, pc_base = train_lgb_with_noise_handling(
    X_train, y, groups, train_months, X_test,
    noise_indices=[], noise_mode="baseline",
    n_seeds=N_SEEDS, label="Baseline (no cleaning)"
)

# Variant A: Remove 133 agreed noisy from training
print(f"\n  --- Variant A: Remove {len(agreed_noisy)} agreed noisy ---", flush=True)
oof_a, test_a, skf_a, lomo_a, pc_a = train_lgb_with_noise_handling(
    X_train, y, groups, train_months, X_test,
    noise_indices=agreed_noisy, noise_mode="remove",
    n_seeds=N_SEEDS, label=f"Remove {len(agreed_noisy)} agreed noisy"
)

# Variant B: Relabel noise to TabPFN's predicted class
print(f"\n  --- Variant B: Relabel {len(agreed_noisy)} to TabPFN pred ---", flush=True)
oof_b, test_b, skf_b, lomo_b, pc_b = train_lgb_with_noise_handling(
    X_train, y, groups, train_months, X_test,
    noise_indices=agreed_noisy, noise_mode="relabel",
    relabel_probs=oof_tabpfn,
    n_seeds=N_SEEDS, label=f"Relabel {len(agreed_noisy)} to TabPFN"
)

# Variant C: Remove only extreme noise
print(f"\n  --- Variant C: Remove {len(agreed_extreme)} extreme noise (q<0.05) ---", flush=True)
oof_c, test_c, skf_c, lomo_c, pc_c = train_lgb_with_noise_handling(
    X_train, y, groups, train_months, X_test,
    noise_indices=agreed_extreme, noise_mode="remove",
    n_seeds=N_SEEDS, label=f"Remove {len(agreed_extreme)} extreme noise"
)


# ══════════════════════════════════════════════════════════════════════
# 5. Per-class comparison
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  PER-CLASS AP COMPARISON")
print("=" * 90)

variants = [
    ("Baseline", pc_base, skf_base, lomo_base),
    (f"Remove {len(agreed_noisy)}", pc_a, skf_a, lomo_a),
    (f"Relabel {len(agreed_noisy)}", pc_b, skf_b, lomo_b),
    (f"Extreme only ({len(agreed_extreme)})", pc_c, skf_c, lomo_c),
]

header = f"  {'Class':15s}"
for name, _, _, _ in variants:
    header += f"  {name:>14s}"
header += "  {'Best delta':>11s}"
print(header)
print(f"  {'-' * (15 + 16 * len(variants) + 13)}")

for cls in CLASSES:
    line = f"  {cls:15s}"
    base_val = pc_base[cls]
    best_delta = 0
    for name, pc, _, _ in variants:
        val = pc[cls]
        line += f"  {val:14.4f}"
        if name != "Baseline":
            delta = val - base_val
            if abs(delta) > abs(best_delta):
                best_delta = delta
    marker = "***" if best_delta > 0.01 else "  *" if best_delta > 0.005 else "   " if best_delta > 0 else " --"
    line += f"  {best_delta:+10.4f} {marker}"
    print(line)

print(f"\n  {'MACRO mAP':15s}", end="")
for name, _, skf, _ in variants:
    print(f"  {skf:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for name, _, _, lomo in variants:
    print(f"  {lomo:14.4f}", end="")
print()


# ══════════════════════════════════════════════════════════════════════
# 6. Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TASK 1 SUMMARY")
print("=" * 90)

print(f"\n  Noise detection:")
print(f"    E175 flagged:       {len(noisy_e175)} samples")
print(f"    TabPFN flagged:     {len(noisy_tabpfn)} samples")
print(f"    Agreed noisy:       {len(agreed_noisy)} samples")
print(f"    Extreme (q<0.05):   {len(agreed_extreme)} samples")

print(f"\n  Results (SKF / LOMO):")
print(f"    Baseline:           {skf_base:.4f} / {lomo_base:.4f}")
print(f"    A: Remove agreed:   {skf_a:.4f} / {lomo_a:.4f}  (delta: {skf_a-skf_base:+.4f} / {lomo_a-lomo_base:+.4f})")
print(f"    B: Relabel agreed:  {skf_b:.4f} / {lomo_b:.4f}  (delta: {skf_b-skf_base:+.4f} / {lomo_b-lomo_base:+.4f})")
print(f"    C: Remove extreme:  {skf_c:.4f} / {lomo_c:.4f}  (delta: {skf_c-skf_base:+.4f} / {lomo_c-lomo_base:+.4f})")

# Key classes
for cls in ["Cormorants", "Waders", "Ducks", "Pigeons"]:
    print(f"\n  {cls} AP breakdown:")
    for name, pc, _, _ in variants:
        delta = pc[cls] - pc_base[cls] if name != "Baseline" else 0
        d_str = f"({delta:+.4f})" if name != "Baseline" else ""
        print(f"    {name:25s}: {pc[cls]:.4f} {d_str}")

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed/60:.1f} min")
print("=" * 90)
