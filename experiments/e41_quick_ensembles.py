"""E41: Quick ensemble experiments on existing test predictions.

No retraining. Just combines existing predictions in smart ways.

A: Average E38 + E32 test predictions (different feature sets = diversity)
B: Multi-seed: retrain E38 with 3 seeds, average
C: Month-adaptive: detect month from solar features, blend with GBIF priors
   for unseen months only
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ======================================================================
# Load everything
# ======================================================================
print("=" * 60, flush=True)
print("E41 QUICK ENSEMBLES", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Load existing predictions
test_e32 = np.load(ROOT / "test_e32.npy")
test_e38 = np.load(ROOT / "test_e38.npy")
oof_e32 = np.load(ROOT / "oof_e32.npy")
oof_e38 = np.load(ROOT / "oof_e38.npy")

print(f"  test_e32: {test_e32.shape}", flush=True)
print(f"  test_e38: {test_e38.shape}", flush=True)

# OOF scores
m32, _ = compute_map(y, oof_e32)
m38, _ = compute_map(y, oof_e38)
print(f"  E32 OOF mAP: {m32:.4f}", flush=True)
print(f"  E38 OOF mAP: {m38:.4f}", flush=True)

# Load solar features for month detection
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
test_months = test_ts.dt.month.values

# Load GBIF
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_priors = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")

# Train month distribution
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
train_dist = np.bincount(y, minlength=N_CLASSES) / len(y)

# ======================================================================
# A: Simple average of E32 + E38
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("A: AVERAGE E32 + E38 TEST PREDICTIONS", flush=True)
print("=" * 60, flush=True)

for w38 in [0.5, 0.6, 0.7, 0.8, 0.9]:
    w32 = 1 - w38
    blend = w32 * test_e32 + w38 * test_e38
    # Check OOF too
    oof_blend = w32 * oof_e32 + w38 * oof_e38
    m, _ = compute_map(y, oof_blend)
    print(f"  w38={w38:.1f}: OOF mAP={m:.4f}", flush=True)

# Best OOF blend for submission
best_w38 = 0.7
blend_a = (1 - best_w38) * test_e32 + best_w38 * test_e38
oof_a = (1 - best_w38) * oof_e32 + best_w38 * oof_e38
m_a, _ = compute_map(y, oof_a)
print(f"\n  Best blend (w38={best_w38}): OOF mAP={m_a:.4f}", flush=True)

# ======================================================================
# B: Month-adaptive post-processing
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("B: MONTH-ADAPTIVE POST-PROCESSING", flush=True)
print("=" * 60, flush=True)

# Use daylight_hours to classify test samples by estimated month
daylight = test_solar["daylight_hours"].values
print(f"\n  Daylight hours range: [{daylight.min():.1f}, {daylight.max():.1f}]", flush=True)

# Expected daylight at Eemshaven (53.4N):
# Jan: ~8.0h, Feb: ~9.5h, Mar: ~11.5h, Apr: ~14.0h, May: ~16.0h
# Jun: ~17.0h, Jul: ~16.5h, Aug: ~15.0h, Sep: ~12.5h, Oct: ~10.5h
# Nov: ~8.5h, Dec: ~7.5h
# Shared months in test: Sep (12-13h), Oct (10-11h)
# Unseen months in test: Feb (9-10h), May (15-16h), Dec (7-8h)

# Classify based on daylight hours
est_month = np.zeros(len(daylight), dtype=int)
for i, dh in enumerate(daylight):
    if dh < 8.5:
        est_month[i] = 12  # Dec
    elif dh < 10.0:
        est_month[i] = 2   # Feb
    elif dh < 11.5:
        est_month[i] = 10  # Oct
    elif dh < 13.5:
        est_month[i] = 9   # Sep
    elif dh < 17.0:
        est_month[i] = 5   # May
    else:
        est_month[i] = 5   # Jun/Jul edge case -> treat as May

# Compare with actual months
print(f"\n  Estimated vs actual test month distribution:", flush=True)
print(f"  {'Est Month':>10s} {'Count':>6s} | Actual month distribution", flush=True)
for m in sorted(np.unique(est_month)):
    mask = est_month == m
    actual_counts = np.bincount(test_months[mask], minlength=13)
    actual_str = ", ".join(f"M{am}:{actual_counts[am]}" for am in range(1, 13) if actual_counts[am] > 0)
    print(f"  {m:>10d} {mask.sum():>6d} | {actual_str}", flush=True)

# Shared vs unseen
shared_mask = np.isin(est_month, [9, 10])
unseen_mask = ~shared_mask
print(f"\n  Shared months (Sep/Oct): {shared_mask.sum()} samples ({shared_mask.mean()*100:.1f}%)", flush=True)
print(f"  Unseen months (Feb/May/Dec): {unseen_mask.sum()} samples ({unseen_mask.mean()*100:.1f}%)", flush=True)

# Build GBIF seasonal priors for each month
def get_gbif_prior(month):
    """Get GBIF-adjusted prior for a given month."""
    row = gbif_priors[gbif_priors["month"] == month]
    if len(row) == 0:
        return train_dist.copy()
    prior = np.array([row.iloc[0][cls] for cls in CLASSES])
    prior = np.maximum(prior, 0.01)  # floor to avoid zero
    prior = prior / prior.sum()
    return prior

# Month-adaptive blending: for unseen months, blend with GBIF prior
print(f"\n  Testing month-adaptive blending on E38 predictions...", flush=True)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    test_adj = test_e38.copy()
    for i in range(len(test_adj)):
        if not shared_mask[i]:
            prior = get_gbif_prior(est_month[i])
            # Bayesian adjustment: blend prediction with prior
            pred = test_adj[i]
            adjusted = (1 - alpha) * pred + alpha * prior
            adjusted = adjusted / adjusted.sum()
            test_adj[i] = adjusted

    # Can't evaluate on OOF since we don't have unseen month OOF predictions
    # Just check test distribution
    dist = np.bincount(test_adj.argmax(axis=1), minlength=N_CLASSES)
    print(f"  alpha={alpha:.1f}: Cormorants={dist[2]}, Pigeons={dist[6]}, "
          f"Waders={dist[8]}, BoP={dist[0]}", flush=True)

# Also try: for unseen months, use GBIF-reweighted predictions
print(f"\n  GBIF reweighting (multiplicative) on unseen months...", flush=True)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
    test_adj = test_e38.copy()
    for i in range(len(test_adj)):
        if not shared_mask[i]:
            prior = get_gbif_prior(est_month[i])
            # Multiplicative: pred * prior^alpha, then renormalize
            adjusted = test_adj[i] * np.power(prior / train_dist, alpha)
            adjusted = adjusted / adjusted.sum()
            test_adj[i] = adjusted

    dist = np.bincount(test_adj.argmax(axis=1), minlength=N_CLASSES)
    print(f"  alpha={alpha:.1f}: Cormorants={dist[2]}, Pigeons={dist[6]}, "
          f"Waders={dist[8]}, Ducks={dist[3]}, Geese={dist[4]}", flush=True)

# ======================================================================
# C: Combine A + B: E32+E38 blend + month-adaptive
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("C: E32+E38 BLEND + MONTH-ADAPTIVE", flush=True)
print("=" * 60, flush=True)

# Use E38*0.7 + E32*0.3 as base, then month-adaptive on top
base_blend = 0.3 * test_e32 + 0.7 * test_e38

for alpha in [0.0, 0.1, 0.2, 0.3]:
    test_adj = base_blend.copy()
    for i in range(len(test_adj)):
        if not shared_mask[i]:
            prior = get_gbif_prior(est_month[i])
            adjusted = test_adj[i] * np.power(prior / train_dist, alpha)
            adjusted = adjusted / adjusted.sum()
            test_adj[i] = adjusted

    dist = np.bincount(test_adj.argmax(axis=1), minlength=N_CLASSES)
    n_classes_predicted = (dist > 0).sum()
    print(f"  alpha={alpha:.1f}: classes_predicted={n_classes_predicted}, "
          f"Gull%={dist[5]/len(test_adj)*100:.1f}%, "
          f"min_class={CLASSES[dist.argmin()]}({dist.min()})", flush=True)

# ======================================================================
# Save submissions
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SAVING SUBMISSIONS", flush=True)
print("=" * 60, flush=True)

# A: Simple blend
save_submission(blend_a, "e41a_e32_e38_blend", cv_map=m_a)

# B: E38 + month-adaptive (alpha=0.2 as moderate choice)
test_b = test_e38.copy()
for i in range(len(test_b)):
    if not shared_mask[i]:
        prior = get_gbif_prior(est_month[i])
        adjusted = test_b[i] * np.power(prior / train_dist, 0.2)
        adjusted = adjusted / adjusted.sum()
        test_b[i] = adjusted
save_submission(test_b, "e41b_month_adaptive", cv_map=m38)

# C: Blend + month-adaptive (alpha=0.2)
test_c = base_blend.copy()
for i in range(len(test_c)):
    if not shared_mask[i]:
        prior = get_gbif_prior(est_month[i])
        adjusted = test_c[i] * np.power(prior / train_dist, 0.2)
        adjusted = adjusted / adjusted.sum()
        test_c[i] = adjusted
save_submission(test_c, "e41c_blend_adaptive", cv_map=m_a)

print("\nDone!", flush=True)
