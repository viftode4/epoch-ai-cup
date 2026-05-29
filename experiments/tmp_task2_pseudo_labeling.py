"""Task 2: Validate pseudo-labeling with existing predictions.

Uses TabPFN test predictions for pseudo-labeling:
- Add test samples at various confidence thresholds to training
- Retrain LightGBM (fast), evaluate on original train CV
- Also test iterative pseudo-labeling (2 rounds)
- Report per-class AP + LOMO, especially Cormorant and month breakdown
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
print("  TASK 2: Pseudo-Labeling Validation")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)


# ══════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════

print("\n[1] Loading data...", flush=True)

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
n_train = len(y)
n_test = len(test_df)

# Load cached features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# Load test predictions from TabPFN
test_tabpfn = np.load(ROOT / "test_e183_tabpfn.npy")
# Also load E175 test predictions
test_e175 = np.load(ROOT / "test_e175_best.npy")

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  TabPFN test shape: {test_tabpfn.shape}")
print(f"  E175 test shape: {test_e175.shape}")

# Show test month distribution
print("\n  Test month distribution:")
for m in sorted(set(test_months)):
    mask = test_months == m
    print(f"    Month {m:2d}: {mask.sum()} samples")

# Confidence distribution of TabPFN test predictions
max_probs = test_tabpfn.max(axis=1)
print(f"\n  TabPFN confidence stats: mean={max_probs.mean():.3f}, median={np.median(max_probs):.3f}")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    n_above = (max_probs >= thresh).sum()
    print(f"    >= {thresh:.2f}: {n_above} samples ({100*n_above/n_test:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 2. Helper functions
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
    print(f"  {label:45s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]", flush=True)
    return skf, lomo_avg, per_class, lomo_maps


def train_lgb_pseudo(
    X_train_orig, y_orig, groups_orig, months_orig,
    X_test_full, test_months_full,
    pseudo_X, pseudo_y, pseudo_months,
    pseudo_weight=0.5, n_seeds=5, label="",
):
    """Train LGB with pseudo-labeled data added to training folds.

    Validation is ALWAYS on original train data only.
    Pseudo-labeled samples are added to training folds only.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_orig = len(y_orig)
    n_pseudo = len(pseudo_y)

    # Combine features
    X_combined = np.vstack([X_train_orig, pseudo_X])
    y_combined = np.concatenate([y_orig, pseudo_y])

    # Groups: pseudo samples get unique IDs
    max_group = groups_orig.max() + 1
    pseudo_groups = np.arange(max_group, max_group + n_pseudo)

    # Sample weights: original=1.0, pseudo=pseudo_weight
    w_orig = np.ones(n_orig)
    w_pseudo = np.full(n_pseudo, pseudo_weight)
    w_combined = np.concatenate([w_orig, w_pseudo])

    oof_all = np.zeros((n_seeds, n_orig, N_CLASSES))
    test_all = np.zeros((n_seeds, X_test_full.shape[0], N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((n_orig, N_CLASSES))
        test_s = np.zeros((X_test_full.shape[0], N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train_orig, y_orig, groups_orig)):
            # Training: original train fold + ALL pseudo samples
            pseudo_idx = np.arange(n_orig, n_orig + n_pseudo)
            combined_tr = np.concatenate([tr_idx, pseudo_idx])

            X_tr = X_combined[combined_tr]
            y_tr = y_combined[combined_tr]
            w_tr = w_combined[combined_tr]

            # Validation: original samples only, with original labels
            X_va = X_train_orig[va_idx]
            y_va = y_orig[va_idx]

            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

            oof_s[va_idx] = m.predict_proba(X_va)
            test_s += m.predict_proba(X_test_full) / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)

    skf, lomo, per_class, lomo_maps = lomo_eval(oof_mean, y_orig, months_orig, label)
    return oof_mean, test_mean, skf, lomo, per_class, lomo_maps


# ══════════════════════════════════════════════════════════════════════
# 3. Baseline: no pseudo-labeling
# ══════════════════════════════════════════════════════════════════════

N_SEEDS = 5
print(f"\n[2] Baseline (no pseudo-labeling, {N_SEEDS} seeds)...", flush=True)
oof_base, test_base, skf_base, lomo_base, pc_base, lomo_maps_base = train_lgb_pseudo(
    X_train, y, groups, train_months,
    X_test, test_months,
    pseudo_X=np.zeros((0, X_train.shape[1]), dtype=np.float32),
    pseudo_y=np.zeros(0, dtype=int),
    pseudo_months=np.zeros(0, dtype=int),
    n_seeds=N_SEEDS, label="Baseline (no pseudo)"
)


# ══════════════════════════════════════════════════════════════════════
# 4. Pseudo-labeling at different confidence thresholds
# ══════════════════════════════════════════════════════════════════════

# Use ensemble of TabPFN + E175 for pseudo-labels (more robust)
test_ensemble = 0.5 * test_tabpfn + 0.5 * test_e175
test_max_prob = test_ensemble.max(axis=1)
test_hard_labels = test_ensemble.argmax(axis=1)

results = {}
results["Baseline"] = (skf_base, lomo_base, pc_base, lomo_maps_base)

for thresh in [0.5, 0.6, 0.7, 0.8]:
    pseudo_mask = test_max_prob >= thresh
    n_pseudo = pseudo_mask.sum()
    pseudo_y_t = test_hard_labels[pseudo_mask]
    pseudo_X_t = X_test[pseudo_mask]
    pseudo_months_t = test_months[pseudo_mask]

    # Show class dist
    dist = np.bincount(pseudo_y_t, minlength=N_CLASSES)
    dist_str = " ".join(f"{CLASSES[c][:4]}={dist[c]}" for c in range(N_CLASSES))
    print(f"\n[3] Pseudo-label threshold={thresh:.1f} ({n_pseudo} samples): {dist_str}", flush=True)

    # Show month dist
    for m in sorted(set(pseudo_months_t)):
        mask_m = pseudo_months_t == m
        print(f"    Month {m:2d}: {mask_m.sum()} pseudo samples")

    oof_p, test_p, skf_p, lomo_p, pc_p, lomo_maps_p = train_lgb_pseudo(
        X_train, y, groups, train_months,
        X_test, test_months,
        pseudo_X=pseudo_X_t,
        pseudo_y=pseudo_y_t,
        pseudo_months=pseudo_months_t,
        pseudo_weight=0.5,
        n_seeds=N_SEEDS,
        label=f"Pseudo thresh={thresh:.1f} (n={n_pseudo})"
    )
    results[f"Pseudo {thresh:.1f}"] = (skf_p, lomo_p, pc_p, lomo_maps_p)


# ══════════════════════════════════════════════════════════════════════
# 5. Iterative pseudo-labeling (2 rounds) at threshold 0.7
# ══════════════════════════════════════════════════════════════════════

print(f"\n[4] Iterative pseudo-labeling (2 rounds, thresh=0.7)...", flush=True)

# Round 1: use existing test preds
pseudo_mask_r1 = test_max_prob >= 0.7
pseudo_X_r1 = X_test[pseudo_mask_r1]
pseudo_y_r1 = test_hard_labels[pseudo_mask_r1]
pseudo_months_r1 = test_months[pseudo_mask_r1]

print(f"  Round 1: {pseudo_mask_r1.sum()} pseudo-labeled samples")

import lightgbm as lgb_lib
from sklearn.model_selection import StratifiedGroupKFold

# Train round 1 model to get new test predictions
n_orig = len(y)
X_comb_r1 = np.vstack([X_train, pseudo_X_r1])
y_comb_r1 = np.concatenate([y, pseudo_y_r1])
max_g = groups.max() + 1
groups_pseudo_r1 = np.arange(max_g, max_g + len(pseudo_y_r1))
w_r1 = np.concatenate([np.ones(n_orig), np.full(len(pseudo_y_r1), 0.5)])

test_preds_r2 = np.zeros((n_test, N_CLASSES))
sgkf_r1 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(sgkf_r1.split(X_train, y, groups)):
    pseudo_idx = np.arange(n_orig, n_orig + len(pseudo_y_r1))
    combined_tr = np.concatenate([tr_idx, pseudo_idx])
    m = lgb_lib.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
        n_estimators=1500, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
        drop_rate=0.15, is_unbalance=True, verbosity=-1,
        random_state=42 + fold, n_jobs=-1,
    )
    m.fit(X_comb_r1[combined_tr], y_comb_r1[combined_tr],
          sample_weight=w_r1[combined_tr],
          eval_set=[(X_train[va_idx], y[va_idx])],
          callbacks=[lgb_lib.early_stopping(100, verbose=False)])
    test_preds_r2 += m.predict_proba(X_test) / N_FOLDS

# Round 2: use updated predictions
test_max_r2 = test_preds_r2.max(axis=1)
test_labels_r2 = test_preds_r2.argmax(axis=1)
pseudo_mask_r2 = test_max_r2 >= 0.7
n_pseudo_r2 = pseudo_mask_r2.sum()
print(f"  Round 2: {n_pseudo_r2} pseudo-labeled samples (was {pseudo_mask_r1.sum()})")

# How many labels changed?
common = pseudo_mask_r1 & pseudo_mask_r2
changed = (test_hard_labels[common] != test_labels_r2[common]).sum()
print(f"  Labels changed in round 2: {changed}/{common.sum()}")

oof_iter, test_iter, skf_iter, lomo_iter, pc_iter, lomo_maps_iter = train_lgb_pseudo(
    X_train, y, groups, train_months,
    X_test, test_months,
    pseudo_X=X_test[pseudo_mask_r2],
    pseudo_y=test_labels_r2[pseudo_mask_r2],
    pseudo_months=test_months[pseudo_mask_r2],
    pseudo_weight=0.5,
    n_seeds=N_SEEDS,
    label="Iterative pseudo (2 rounds)"
)
results["Iterative 2R"] = (skf_iter, lomo_iter, pc_iter, lomo_maps_iter)


# ══════════════════════════════════════════════════════════════════════
# 6. Per-class comparison
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  PER-CLASS AP COMPARISON")
print("=" * 90)

variant_names = list(results.keys())
header = f"  {'Class':15s}"
for name in variant_names:
    header += f"  {name:>14s}"
print(header)
print(f"  {'-' * (15 + 16 * len(variant_names))}")

for cls in CLASSES:
    line = f"  {cls:15s}"
    for name in variant_names:
        _, _, pc, _ = results[name]
        line += f"  {pc[cls]:14.4f}"
    # Delta vs baseline
    base_val = results["Baseline"][2][cls]
    best_delta = 0
    for name in variant_names[1:]:
        delta = results[name][2][cls] - base_val
        if abs(delta) > abs(best_delta):
            best_delta = delta
    marker = " ***" if best_delta > 0.01 else "   *" if best_delta > 0 else "  --"
    line += f"  {best_delta:+.4f}{marker}"
    print(line)

print(f"\n  {'SKF':15s}", end="")
for name in variant_names:
    print(f"  {results[name][0]:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for name in variant_names:
    print(f"  {results[name][1]:14.4f}", end="")
print()


# ══════════════════════════════════════════════════════════════════════
# 7. Month breakdown
# ══════════════════════════════════════════════════════════════════════

print(f"\n  LOMO Month Breakdown:")
header = f"  {'Variant':20s}"
for m in MONTHS:
    header += f"  M{m:02d}"
print(header)
for name in variant_names:
    lomo_maps = results[name][3]
    line = f"  {name:20s}"
    for m in MONTHS:
        line += f"  {lomo_maps.get(m, 0):.3f}"
    print(line)


# ══════════════════════════════════════════════════════════════════════
# 8. Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TASK 2 SUMMARY")
print("=" * 90)

for name in variant_names:
    skf, lomo, pc, _ = results[name]
    d_skf = skf - skf_base if name != "Baseline" else 0
    d_lomo = lomo - lomo_base if name != "Baseline" else 0
    d_str = f"  (dSKF={d_skf:+.4f}, dLOMO={d_lomo:+.4f})" if name != "Baseline" else ""
    corm = pc.get("Cormorants", 0)
    wader = pc.get("Waders", 0)
    print(f"  {name:20s}: SKF={skf:.4f} LOMO={lomo:.4f}  Corm={corm:.4f} Wader={wader:.4f}{d_str}")

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed/60:.1f} min")
print("=" * 90)
