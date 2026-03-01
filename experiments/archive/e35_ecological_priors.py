"""E35: Month-Aware Ecological Prior Adjustment

Post-processing on E32 predictions using Bayesian prior adjustment.
For each test sample in month m:
    adjusted[c] = pred[c] * (P_eco(c|m) / P_train(c))^alpha
    adjusted = adjusted / sum(adjusted)

- Shared months (Sep, Oct): P_eco from training data
- Unseen months (Feb, May, Dec): P_eco from ecological research (Eemshaven ornithology)
- Also tests "nearest training month" transfer priors

NO retraining, NO new features, NO temporal leakage.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ======================================================================
# Step 1: Define ecological priors
# ======================================================================

# Ecological presence scores for UNSEEN test months (Feb, May, Dec)
# Sources: waarneming.nl, birdingplaces.eu, Wadden Sea QSR, Dutch ornithology
# Order matches CLASSES (alphabetical): BoP, Clutter, Corm, Ducks, Geese,
#                                        Gulls, Pigeons, Songbirds, Waders
ECO_SCORES = {
    2:  [0.05, 0.05, 0.05, 0.08, 0.10, 0.50, 0.04, 0.05, 0.08],  # Feb
    5:  [0.10, 0.05, 0.04, 0.03, 0.01, 0.45, 0.04, 0.18, 0.10],  # May
    12: [0.05, 0.05, 0.04, 0.10, 0.12, 0.45, 0.04, 0.05, 0.10],  # Dec
}

# "Nearest training month" transfer priors (alternative to ecological)
NEAREST_MONTH = {
    2: 1,    # Feb -> Jan (closest winter month)
    5: 4,    # May -> Apr (closest spring month)
    12: 1,   # Dec -> Jan (closest winter month)
}

ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def normalize_prior(scores):
    """Normalize a score vector to sum to 1."""
    arr = np.array(scores, dtype=np.float64)
    return arr / arr.sum()


def bayesian_adjust(preds, p_eco, p_train, alpha):
    """Apply Bayesian prior adjustment to predictions.

    adjusted[c] = pred[c] * (p_eco[c] / p_train[c])^alpha
    Then renormalize each row.
    """
    if alpha == 0:
        return preds.copy()
    # Avoid division by zero
    p_train_safe = np.maximum(p_train, 1e-10)
    ratio = (p_eco / p_train_safe) ** alpha  # shape (9,)
    adjusted = preds * ratio[np.newaxis, :]  # broadcast
    # Renormalize
    row_sums = adjusted.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    adjusted = adjusted / row_sums
    return adjusted


def class_dist_str(preds, top_n=3):
    """Return string showing top-N predicted classes by mean probability."""
    mean_probs = preds.mean(axis=0)
    order = np.argsort(mean_probs)[::-1]
    parts = []
    for i in order[:top_n]:
        parts.append(f"{CLASSES[i][:4]}:{mean_probs[i]:.2f}")
    return " ".join(parts)


# ======================================================================
# Main
# ======================================================================
print("=" * 60, flush=True)
print("E35 ECOLOGICAL PRIOR ADJUSTMENT", flush=True)
print("=" * 60, flush=True)

# Load E32 predictions
oof = np.load(ROOT / "oof_e32.npy")
test_preds = np.load(ROOT / "test_e32.npy")
print(f"  Loaded: oof_e32.npy {oof.shape}, test_e32.npy {test_preds.shape}", flush=True)

# Load data for labels and timestamps
train_df = load_train()
test_df = load_test()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

# ======================================================================
# Step 1: Build full prior table
# ======================================================================
print("\n" + "-" * 60, flush=True)
print("PRIORS", flush=True)
print("-" * 60, flush=True)

# Overall training prior
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

print(f"\n  Training prior (overall, N={len(y)}):", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {p_train[i]:.4f} (N={int(counts[i])})", flush=True)

# Per-month training priors (for shared months Sep=9, Oct=10)
train_month_priors = {}
unique_train_months = sorted(np.unique(train_months))
print(f"\n  Training month priors:", flush=True)
for m in unique_train_months:
    mask = train_months == m
    m_counts = np.bincount(y[mask], minlength=N_CLASSES).astype(float)
    m_total = m_counts.sum()
    # Add smoothing (Laplace) to avoid zeros
    m_prior = (m_counts + 0.5) / (m_total + 0.5 * N_CLASSES)
    train_month_priors[m] = m_prior
    print(f"    Month {m:2d} (N={int(m_total)}):", end="", flush=True)
    for i in range(N_CLASSES):
        print(f" {m_prior[i]:.3f}", end="")
    print(flush=True)

# Ecological priors for unseen months
eco_priors = {}
print(f"\n  Ecological priors (unseen months):", flush=True)
for m, scores in ECO_SCORES.items():
    eco_priors[m] = normalize_prior(scores)
    print(f"    Month {m:2d}        :", end="", flush=True)
    for i in range(N_CLASSES):
        print(f" {eco_priors[m][i]:.3f}", end="")
    print(flush=True)

# "Nearest training month" priors for unseen months
nearest_priors = {}
print(f"\n  Nearest-month transfer priors:", flush=True)
for unseen_m, train_m in NEAREST_MONTH.items():
    nearest_priors[unseen_m] = train_month_priors[train_m]
    print(f"    Month {unseen_m:2d} <- Month {train_m:2d}:", end="", flush=True)
    for i in range(N_CLASSES):
        print(f" {nearest_priors[unseen_m][i]:.3f}", end="")
    print(flush=True)

# Build full prior lookup: shared months from training, unseen from ecological
full_priors_eco = {}
full_priors_nearest = {}
for m in sorted(np.unique(test_months)):
    if m in train_month_priors:
        # Shared month: use actual training distribution
        full_priors_eco[m] = train_month_priors[m]
        full_priors_nearest[m] = train_month_priors[m]
    else:
        # Unseen month: use ecological or nearest
        full_priors_eco[m] = eco_priors[m]
        full_priors_nearest[m] = nearest_priors[m]

# Print column header for reference
print(f"\n  {'':17s}", end="", flush=True)
for cls in CLASSES:
    print(f" {cls[:5]:>5s}", end="")
print(flush=True)

# ======================================================================
# Step 2: Validate on training OOF (shared months only: Sep=9, Oct=10)
# ======================================================================
print("\n" + "-" * 60, flush=True)
print("OOF VALIDATION (shared months 9, 10 only)", flush=True)
print("-" * 60, flush=True)

shared_mask = np.isin(train_months, [9, 10])
n_shared = shared_mask.sum()
print(f"  N samples in shared months: {n_shared}", flush=True)

y_shared = y[shared_mask]
oof_shared = oof[shared_mask]
months_shared = train_months[shared_mask]

baseline_map, baseline_per = compute_map(y_shared, oof_shared)
print(f"\n  alpha=0.00 (baseline): mAP={baseline_map:.4f}", flush=True)

best_alpha_eco = 0
best_map_eco = baseline_map

for alpha in ALPHAS:
    if alpha == 0:
        continue
    # Apply per-month adjustment on shared training samples
    adjusted = oof_shared.copy()
    for m in [9, 10]:
        m_mask = months_shared == m
        if m_mask.sum() == 0:
            continue
        adjusted[m_mask] = bayesian_adjust(
            oof_shared[m_mask], train_month_priors[m], p_train, alpha
        )
    adj_map, _ = compute_map(y_shared, adjusted)
    delta = adj_map - baseline_map
    print(f"  alpha={alpha:.2f}: mAP={adj_map:.4f} (delta: {delta:+.4f})", flush=True)
    if adj_map > best_map_eco:
        best_map_eco = adj_map
        best_alpha_eco = alpha

print(f"\n  Best alpha on shared months: {best_alpha_eco:.2f} (mAP={best_map_eco:.4f})", flush=True)

# Full OOF validation (all training months) -- use training month priors
print("\n  Full OOF validation (all months, using training priors):", flush=True)
full_baseline_map, _ = compute_map(y, oof)
print(f"  alpha=0.00 (baseline): mAP={full_baseline_map:.4f}", flush=True)

for alpha in ALPHAS:
    if alpha == 0:
        continue
    adjusted = oof.copy()
    for m in unique_train_months:
        m_mask = train_months == m
        if m_mask.sum() == 0:
            continue
        adjusted[m_mask] = bayesian_adjust(
            oof[m_mask], train_month_priors[m], p_train, alpha
        )
    adj_map, adj_per = compute_map(y, adjusted)
    delta = adj_map - full_baseline_map
    print(f"  alpha={alpha:.2f}: mAP={adj_map:.4f} (delta: {delta:+.4f})", flush=True)

# ======================================================================
# Step 3: Test "nearest training month" vs ecological priors
# ======================================================================
print("\n" + "-" * 60, flush=True)
print("PRIOR COMPARISON (ecological vs nearest-month)", flush=True)
print("-" * 60, flush=True)

# We can't directly evaluate on unseen months (no labels), so just compare
# test class distributions
for alpha in [0.5, 1.0]:
    print(f"\n  alpha={alpha:.2f}:", flush=True)

    # Ecological
    test_eco = test_preds.copy()
    for m in sorted(np.unique(test_months)):
        m_mask = test_months == m
        test_eco[m_mask] = bayesian_adjust(
            test_preds[m_mask], full_priors_eco[m], p_train, alpha
        )

    # Nearest month
    test_near = test_preds.copy()
    for m in sorted(np.unique(test_months)):
        m_mask = test_months == m
        test_near[m_mask] = bayesian_adjust(
            test_preds[m_mask], full_priors_nearest[m], p_train, alpha
        )

    print(f"    {'Month':>5s} {'N':>4s}  {'Baseline':20s}  {'Ecological':20s}  {'Nearest':20s}", flush=True)
    for m in sorted(np.unique(test_months)):
        m_mask = test_months == m
        n_m = m_mask.sum()
        base_str = class_dist_str(test_preds[m_mask])
        eco_str = class_dist_str(test_eco[m_mask])
        near_str = class_dist_str(test_near[m_mask])
        tag = " *unseen*" if m not in unique_train_months else ""
        print(f"    {m:5d} {n_m:4d}  {base_str:20s}  {eco_str:20s}  {near_str:20s}{tag}", flush=True)

# ======================================================================
# Step 4: Sensitivity analysis (3 prior variants)
# ======================================================================
print("\n" + "-" * 60, flush=True)
print("SENSITIVITY ANALYSIS (prior variants)", flush=True)
print("-" * 60, flush=True)

# Conservative: flatten ecological priors toward uniform (50% eco + 50% uniform)
uniform = np.ones(N_CLASSES) / N_CLASSES
eco_conservative = {}
for m, p in eco_priors.items():
    eco_conservative[m] = normalize_prior(0.5 * p + 0.5 * uniform)

# Aggressive: exaggerate differences from uniform (2x eco - 1x uniform, clipped)
eco_aggressive = {}
for m, p in eco_priors.items():
    raw = 2.0 * p - uniform
    raw = np.maximum(raw, 0.01)  # clip negatives
    eco_aggressive[m] = normalize_prior(raw)

print(f"\n  Prior variants for unseen months:", flush=True)
for m in sorted(eco_priors.keys()):
    print(f"    Month {m:2d}:", flush=True)
    print(f"      Conservative: {np.array2string(eco_conservative[m], precision=3, separator=', ')}", flush=True)
    print(f"      Moderate:     {np.array2string(eco_priors[m], precision=3, separator=', ')}", flush=True)
    print(f"      Aggressive:   {np.array2string(eco_aggressive[m], precision=3, separator=', ')}", flush=True)

# Test all variants at alpha=1.0
print(f"\n  Test class distributions at alpha=1.0 for each variant:", flush=True)
for variant_name, variant_priors in [("conservative", eco_conservative),
                                      ("moderate", eco_priors),
                                      ("aggressive", eco_aggressive)]:
    full_variant = {}
    for m in sorted(np.unique(test_months)):
        if m in train_month_priors:
            full_variant[m] = train_month_priors[m]
        else:
            full_variant[m] = variant_priors.get(m, normalize_prior(eco_priors.get(m, uniform)))

    test_adj = test_preds.copy()
    for m in sorted(np.unique(test_months)):
        m_mask = test_months == m
        test_adj[m_mask] = bayesian_adjust(
            test_preds[m_mask], full_variant[m], p_train, 1.0
        )

    pred_classes = test_adj.argmax(axis=1)
    dist = np.bincount(pred_classes, minlength=N_CLASSES)
    print(f"  {variant_name:12s}:", end="", flush=True)
    for i in range(N_CLASSES):
        print(f" {CLASSES[i][:4]}={dist[i]:4d}", end="")
    print(flush=True)

# Baseline distribution for comparison
pred_base = test_preds.argmax(axis=1)
dist_base = np.bincount(pred_base, minlength=N_CLASSES)
print(f"  {'baseline':12s}:", end="", flush=True)
for i in range(N_CLASSES):
    print(f" {CLASSES[i][:4]}={dist_base[i]:4d}", end="")
print(flush=True)

# ======================================================================
# Step 5: Apply to test and save submissions
# ======================================================================
print("\n" + "-" * 60, flush=True)
print("SAVING SUBMISSIONS", flush=True)
print("-" * 60, flush=True)

# Detailed per-month before/after table
print("\n  Test prediction changes (ecological, alpha=1.0):", flush=True)
print(f"  {'Month':>5s} {'N':>5s}  {'Unseen?':>7s}  {'Top-3 Before':25s}  {'Top-3 After':25s}", flush=True)

test_eco_1 = test_preds.copy()
for m in sorted(np.unique(test_months)):
    m_mask = test_months == m
    test_eco_1[m_mask] = bayesian_adjust(
        test_preds[m_mask], full_priors_eco[m], p_train, 1.0
    )

for m in sorted(np.unique(test_months)):
    m_mask = test_months == m
    n_m = m_mask.sum()
    unseen = "YES" if m not in unique_train_months else "no"
    before = class_dist_str(test_preds[m_mask])
    after = class_dist_str(test_eco_1[m_mask])
    print(f"  {m:5d} {n_m:5d}  {unseen:>7s}  {before:25s}  {after:25s}", flush=True)

# Save submissions for multiple alphas using ecological priors
saved_files = []
for alpha in [0, 0.5, 1.0, 1.5]:
    test_adj = test_preds.copy()
    if alpha > 0:
        for m in sorted(np.unique(test_months)):
            m_mask = test_months == m
            test_adj[m_mask] = bayesian_adjust(
                test_preds[m_mask], full_priors_eco[m], p_train, alpha
            )
    label = f"e35_eco_a{alpha:.1f}"
    path = save_submission(test_adj, label, cv_map=full_baseline_map)
    saved_files.append(path.name)

# Also save with nearest-month priors at alpha=1.0
test_near_1 = test_preds.copy()
for m in sorted(np.unique(test_months)):
    m_mask = test_months == m
    test_near_1[m_mask] = bayesian_adjust(
        test_preds[m_mask], full_priors_nearest[m], p_train, 1.0
    )
path = save_submission(test_near_1, "e35_nearest_a1.0", cv_map=full_baseline_map)
saved_files.append(path.name)

# Also save with best shared-month alpha using ecological
if best_alpha_eco > 0 and best_alpha_eco not in [0.5, 1.0, 1.5]:
    test_best = test_preds.copy()
    for m in sorted(np.unique(test_months)):
        m_mask = test_months == m
        test_best[m_mask] = bayesian_adjust(
            test_preds[m_mask], full_priors_eco[m], p_train, best_alpha_eco
        )
    path = save_submission(test_best, f"e35_eco_a{best_alpha_eco:.2f}", cv_map=full_baseline_map)
    saved_files.append(path.name)

print(f"\n  Saved {len(saved_files)} submission files:", flush=True)
for f in saved_files:
    print(f"    {f}", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E35 SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Base model: E32 (CV mAP={full_baseline_map:.4f})", flush=True)
print(f"  OOF shared-month validation:", flush=True)
print(f"    Baseline (alpha=0): {baseline_map:.4f}", flush=True)
print(f"    Best alpha:         {best_alpha_eco:.2f} -> {best_map_eco:.4f} (delta: {best_map_eco - baseline_map:+.4f})", flush=True)
print(f"  Test months: {sorted(np.unique(test_months))}", flush=True)
print(f"  Unseen months: {sorted([m for m in np.unique(test_months) if m not in unique_train_months])}", flush=True)
n_unseen = sum(test_months == m for m in np.unique(test_months) if m not in unique_train_months).sum() if any(m not in unique_train_months for m in np.unique(test_months)) else 0
print(f"  Unseen test samples: {n_unseen}/{len(test_months)} ({100*n_unseen/len(test_months):.1f}%)", flush=True)
print(f"\n  Key insight: prior adjustment changes predictions for {n_unseen} unseen-month samples.", flush=True)
print(f"  If OOF validation on shared months HURTS (best alpha=0), priors may not help.", flush=True)
print(f"  Submit alpha=0 (baseline) and alpha=1.0 (ecological) to compare on LB.", flush=True)
print("\nDone!", flush=True)
