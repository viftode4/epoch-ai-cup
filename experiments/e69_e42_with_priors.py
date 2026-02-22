"""E69: Apply E54 unseen-month GBIF priors to E42 predictions.

E42 is the best LOMO model (0.3799) but its LB score is unknown because
it was submitted before the E53/E54 prior-adjustment discovery.

E54 applied the same prior tilt to E50 (LOMO 0.3625) and got LB 0.56.
E42 has +0.017 better LOMO → if the prior tilt helps by the same margin,
E42+priors should match or beat E54's 0.56.

Strategy:
  1. Load E42 best submission (LOMO 0.3799, alpha=0.40 specialist blend)
  2. Apply E54 winter_tilt priors {m2=0.22, m5=0.12, m12=0.24} for unseen months
  3. Save multiple variants for Kaggle comparison

Also try:
  A. E42 + winter_tilt (same as E54 but on E42 base)
  B. E42 + slightly higher winter correction (E55-style sweep but on better base)
  C. E42 + E51 blend (E51 OOF blended E42+E50) + winter_tilt
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_test, CLASSES
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

print("=" * 60, flush=True)
print("E69: E42 + E54 PRIORS (better base + same prior tilt)", flush=True)
print("=" * 60, flush=True)

# ── Load E42 best test predictions ─────────────────────────────────
e42_csv = ROOT / "submissions" / "e42_blend40_0.3799_20260216_0020.csv"
print(f"\n  Loading E42 predictions: {e42_csv.name}", flush=True)
e42_df = pd.read_csv(e42_csv)
print(f"  Rows: {len(e42_df)}, Columns: {list(e42_df.columns)}", flush=True)
test_e42 = e42_df[CLASSES].values
print(f"  Shape: {test_e42.shape}, row-sum check: {test_e42.sum(axis=1)[:3]}", flush=True)

# ── Load test timestamps for month detection ────────────────────────
test_df = load_test()
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
test_months = test_ts.dt.month.values

# Distribution of test predictions by month
print(f"\n  Test month distribution:", flush=True)
for m in sorted(np.unique(test_months)):
    mask = test_months == m
    top1 = test_e42[mask].argmax(axis=1)
    dist = np.bincount(top1, minlength=N_CLASSES)
    top_cls = CLASSES[dist.argmax()]
    print(f"    Month {m:2d}: n={mask.sum():4d}, top class={top_cls} "
          f"({dist.max()}/{mask.sum()} = {dist.max()/mask.sum()*100:.1f}%)", flush=True)

# ── Load GBIF monthly priors ────────────────────────────────────────
gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
gbif_prior_map = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    prior = np.array([row[cls] for cls in CLASSES], dtype=float)
    prior = prior / prior.sum()
    gbif_prior_map[m] = prior

print(f"\n  GBIF priors loaded for months: {sorted(gbif_prior_map.keys())}", flush=True)
print(f"\n  GBIF prior for Feb (m=2):", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<18s}: {gbif_prior_map[2][i]:.4f}", flush=True)

# ── Helper: apply prior adjustment ─────────────────────────────────
def apply_unseen_priors(preds, months, alphas):
    """Bayesian prior tilt for unseen months only.

    adjusted = (1-alpha) * model_probs + alpha * gbif_prior
    Only applied to rows where month is in alphas dict.
    """
    out = preds.copy()
    n_adjusted = 0
    for idx in range(len(out)):
        m = months[idx]
        if m in alphas:
            alpha = alphas[m]
            prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
            adjusted = (1 - alpha) * out[idx] + alpha * prior
            out[idx] = adjusted / adjusted.sum()
            n_adjusted += 1
    print(f"    Adjusted {n_adjusted} rows (months {sorted(alphas.keys())})", flush=True)
    return out

# ── Check E42 base class distribution for unseen months ────────────
print(f"\n  E42 base predictions for unseen months (before adjustment):", flush=True)
for m in [2, 5, 12]:
    mask = test_months == m
    if mask.sum() == 0:
        print(f"    Month {m}: 0 rows", flush=True)
        continue
    top1_dist = np.bincount(test_e42[mask].argmax(axis=1), minlength=N_CLASSES)
    mean_probs = test_e42[mask].mean(axis=0)
    print(f"    Month {m} (n={mask.sum()}):", flush=True)
    for i in np.argsort(-mean_probs)[:4]:
        print(f"      {CLASSES[i]:<18s}: mean_p={mean_probs[i]:.3f}, top1_count={top1_dist[i]}", flush=True)

# ── Variant A: E42 + E54 winter_tilt (m2=0.22, m5=0.12, m12=0.24) ─
print(f"\n{'='*60}", flush=True)
print("Variant A: E42 + E54 winter_tilt alphas", flush=True)
alphas_winter = {2: 0.22, 5: 0.12, 12: 0.24}
test_A = apply_unseen_priors(test_e42, test_months, alphas_winter)
save_submission(test_A, "e69_e42_winter_tilt_m2_0.22_m5_0.12_m12_0.24", cv_map=0.3799)

# ── Variant B: Slightly stronger winter (E55 hypothesis on E42 base) ─
print(f"\nVariant B: E42 + stronger winter (m2=0.26, m5=0.12, m12=0.28)", flush=True)
alphas_stronger = {2: 0.26, 5: 0.12, 12: 0.28}
test_B = apply_unseen_priors(test_e42, test_months, alphas_stronger)
save_submission(test_B, "e69_e42_stronger_winter_m2_0.26_m5_0.12_m12_0.28", cv_map=0.3799)

# ── Variant C: E42 + spring_tilt (m2=0.15, m5=0.28, m12=0.15) ──────
# E54 spring_tilt got LB 0.55 on E50 — test on E42 base
print(f"\nVariant C: E42 + E54 spring_tilt alphas (reference)", flush=True)
alphas_spring = {2: 0.15, 5: 0.28, 12: 0.15}
test_C = apply_unseen_priors(test_e42, test_months, alphas_spring)
save_submission(test_C, "e69_e42_spring_tilt_m2_0.15_m5_0.28_m12_0.15", cv_map=0.3799)

# ── Variant D: E42 + moderate sweep ──────────────────────────────────
print(f"\nVariant D: E42 + moderate (m2=0.20, m5=0.10, m12=0.22)", flush=True)
alphas_moderate = {2: 0.20, 5: 0.10, 12: 0.22}
test_D = apply_unseen_priors(test_e42, test_months, alphas_moderate)
save_submission(test_D, "e69_e42_moderate_m2_0.20_m5_0.10_m12_0.22", cv_map=0.3799)

# ── Show how predictions shift for unseen months ────────────────────
print(f"\n  Post-adjustment distributions for Variant A (winter_tilt):", flush=True)
for m in [2, 5, 12]:
    mask = test_months == m
    if mask.sum() == 0:
        continue
    before = test_e42[mask].mean(axis=0)
    after = test_A[mask].mean(axis=0)
    print(f"  Month {m} (n={mask.sum()}) — mean prob changes:", flush=True)
    for i in np.argsort(-np.abs(after - before))[:5]:
        print(f"    {CLASSES[i]:<18s}: {before[i]:.3f} → {after[i]:.3f} ({after[i]-before[i]:+.3f})", flush=True)

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Base: E42 LOMO=0.3799  vs  E50 LOMO=0.3625 (E54 base)", flush=True)
print(f"  E54 (E50+winter_tilt) → LB 0.56", flush=True)
print(f"", flush=True)
print(f"  Variants saved:", flush=True)
print(f"    A: e69_e42_winter_tilt      → same alphas as E54, stronger base", flush=True)
print(f"    B: e69_e42_stronger_winter  → slightly more aggressive", flush=True)
print(f"    C: e69_e42_spring_tilt      → E54 spring variant on E42 (reference)", flush=True)
print(f"    D: e69_e42_moderate         → conservative", flush=True)
print(f"", flush=True)
print(f"  Submit A first — it is the direct apples-to-apples test.", flush=True)
print(f"  If A > 0.56: stronger base helps. Submit B next.", flush=True)
print(f"  If A = 0.56: base model doesn't matter for priors. Stop here.", flush=True)
print(f"  If A < 0.56: E50 flight-priors were doing work beyond base quality.", flush=True)
print("\nDone!", flush=True)
