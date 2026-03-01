"""E72: Per-group GBIF alpha post-processing on E50 base predictions.

Root cause of E71 failure (LB 0.50 vs 0.56 target):
  E71 dropped binary specialists → Waders/Pigeon probability collapsed:
    Waders top-1:  E50=249  vs  E71=19   (13x drop)
    Pigeon mean p: E50=0.125 vs  E71=0.022
  Without specialists, the 3-model ensemble assigns near-zero probability to
  Waders, so GBIF prior correction (even at α=0.50) can't recover them.

This experiment keeps E50 as the base (Waders=249 top-1, LB 0.56 with E54
uniform alphas) and tests whether per-group GBIF correction improves over
the uniform approach. No retraining required.

Per-group hypothesis:
  rare  classes (Waders, Ducks, Cormorants, Geese): need stronger correction
  common classes (Gulls, Songbirds, Birds of Prey, Pigeons, Clutter): minimal
  → differentiated alpha should beat uniform at same total correction budget

Alpha optimisation proxy: E71 LOMO OOF (best available without E50 OOF).
E71 OOF Waders distribution ≠ E50, so proxy is imperfect; use conservatively.

Submissions:
  e72_e50_e54alphas         — control (E54 uniform on E50, should ≈ LB 0.56)
  e72_e50_pergroup_tuned    — per-group from E71 OOF proxy optimisation
  e72_e50_strong_rare_0.35  — fixed alpha_rare=0.35, alpha_common=0.0
  e72_e50_strong_rare_0.40  — fixed alpha_rare=0.40, alpha_common=0.0
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
CLASS_IDX = {cls: i for i, cls in enumerate(CLASSES)}

RARE_CLASSES  = ["Waders", "Ducks", "Cormorants", "Geese"]
COMMON_CLASSES = ["Gulls", "Songbirds", "Birds of Prey", "Pigeons", "Clutter"]
RARE_IDX   = [CLASS_IDX[c] for c in RARE_CLASSES]
COMMON_IDX = [CLASS_IDX[c] for c in COMMON_CLASSES]

# Unseen test months → LOMO proxy month for alpha tuning
MONTH_ANALOG = {2: 1, 5: 4, 12: 10}

print("=" * 65, flush=True)
print("E72: PER-GROUP GBIF ALPHA ON E50 BASE PREDICTIONS", flush=True)
print("=" * 65, flush=True)

# ── Load E50 test predictions ──────────────────────────────────────
e50_path = ROOT / "submissions" / "e50_perclass_specialist_blend_0.3625_20260218_2146.csv"
e50_df = pd.read_csv(e50_path)
print(f"\nE50 test predictions loaded: {len(e50_df)} rows", flush=True)

# Reconstruct probability matrix in CLASSES order
test_preds_e50 = e50_df[CLASSES].values.astype(np.float64)

# Sanity check: top-1 distribution
top1 = e50_df[CLASSES].idxmax(axis=1).value_counts()
print("  E50 top-1 distribution:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {top1.get(cls, 0):4d}", flush=True)

# ── Load test timestamps for month assignment ──────────────────────
test_df = load_test()
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
test_months = test_ts.dt.month.values
print(f"\n  Test month distribution: { {m: int((test_months==m).sum()) for m in sorted(set(test_months))} }",
      flush=True)

# ── Load GBIF monthly priors ───────────────────────────────────────
gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
gbif_prior_map = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    prior = np.array([row[cls] for cls in CLASSES], dtype=float)
    gbif_prior_map[m] = prior / prior.sum()

# ── Load E71 LOMO OOF for alpha proxy optimisation ────────────────
from src.data import load_train
from sklearn.preprocessing import LabelEncoder

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

oof_e71 = np.load(ROOT / "oof_e71.npy")
print(f"\nE71 OOF loaded: {oof_e71.shape}", flush=True)
from src.metrics import compute_map as cmap
base_lomo, _ = cmap(y, oof_e71)
print(f"  E71 OOF LOMO: {base_lomo:.4f}", flush=True)

# ── Helper: apply per-group prior correction ───────────────────────
def apply_pergroup(preds, months, month_alphas):
    """Apply per-group GBIF correction.

    month_alphas: dict[month → (alpha_rare, alpha_common)]
    Only adjusts rows whose month is in month_alphas.
    """
    out = preds.copy()
    for i in range(len(out)):
        m = months[i]
        if m not in month_alphas:
            continue
        ar, ac = month_alphas[m]
        prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
        adj = out[i].copy()
        for j in RARE_IDX:
            adj[j] = (1 - ar) * out[i, j] + ar * prior[j]
        for j in COMMON_IDX:
            adj[j] = (1 - ac) * out[i, j] + ac * prior[j]
        adj = np.clip(adj, 1e-10, None)
        out[i] = adj / adj.sum()
    return out


def apply_uniform(preds, months, alphas):
    """Apply uniform GBIF correction (all classes same alpha per month)."""
    out = preds.copy()
    for i in range(len(out)):
        m = months[i]
        if m not in alphas:
            continue
        prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
        adj = (1 - alphas[m]) * out[i] + alphas[m] * prior
        out[i] = adj / adj.sum()
    return out


# ── Per-group alpha optimisation on E71 OOF (proxy) ───────────────
print("\n" + "=" * 65, flush=True)
print("PER-GROUP ALPHA GRID SEARCH (E71 OOF proxy)", flush=True)
print("=" * 65, flush=True)
print("  Note: E71 OOF Waders ≈ near-zero (no specialists); proxy is", flush=True)
print("  directional only. Results compared against E54 reference.", flush=True)

alpha_rare_grid   = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
alpha_common_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

best_alphas = {}
for unseen_m, proxy_m in MONTH_ANALOG.items():
    if proxy_m not in unique_months:
        proxy_m = min(unique_months, key=lambda m: abs(m - unseen_m))

    proxy_idx = np.where(train_months == proxy_m)[0]
    y_proxy   = y[proxy_idx]
    oof_proxy = oof_e71[proxy_idx]

    # Baseline (no correction)
    base_m, _ = cmap(y_proxy, oof_proxy)

    best_m_map, best_ar, best_ac = base_m, 0.0, 0.0
    prior = gbif_prior_map.get(proxy_m, np.ones(N_CLASSES) / N_CLASSES)

    for ar in alpha_rare_grid:
        for ac in alpha_common_grid:
            adj = oof_proxy.copy()
            for j in RARE_IDX:
                adj[:, j] = (1 - ar) * oof_proxy[:, j] + ar * prior[j]
            for j in COMMON_IDX:
                adj[:, j] = (1 - ac) * oof_proxy[:, j] + ac * prior[j]
            adj = np.clip(adj, 1e-10, None)
            adj /= adj.sum(axis=1, keepdims=True)
            m_map, _ = cmap(y_proxy, adj)
            if m_map > best_m_map:
                best_m_map = m_map
                best_ar, best_ac = ar, ac

    best_alphas[unseen_m] = (best_ar, best_ac)
    print(f"  Month {unseen_m} (proxy={proxy_m}): "
          f"no-corr={base_m:.4f}  "
          f"best alpha_rare={best_ar:.2f}, alpha_common={best_ac:.2f} "
          f"→ {best_m_map:.4f}  (delta={best_m_map-base_m:+.4f})", flush=True)

print(f"\n  Tuned per-group alphas: {best_alphas}", flush=True)

# ── E54 reference alphas (known LB 0.56) ─────────────────────────
alphas_e54 = {2: 0.22, 5: 0.12, 12: 0.24}

# ── Variant A: E54 uniform alphas on E50 (control) ────────────────
print("\n" + "=" * 65, flush=True)
print("VARIANT A — E54 UNIFORM ALPHAS ON E50 (CONTROL ≈ LB 0.56)", flush=True)
print("=" * 65, flush=True)

test_A = apply_uniform(test_preds_e50, test_months, alphas_e54)
n_adj_A = sum(1 for m in test_months if m in alphas_e54)
print(f"  Adjusted {n_adj_A} test rows (months {sorted(alphas_e54)})", flush=True)
top1_A = pd.Series([CLASSES[i] for i in np.argmax(test_A, axis=1)]).value_counts()
for cls in CLASSES:
    print(f"    {cls:<18s}: {top1_A.get(cls, 0):4d}", flush=True)
save_submission(test_A, "e72_e50_e54alphas_control", cv_map=0.3625)

# ── Variant B: per-group tuned alphas (from E71 OOF proxy) ────────
print("\n" + "=" * 65, flush=True)
print("VARIANT B — PER-GROUP TUNED ALPHAS (E71 OOF PROXY)", flush=True)
print("=" * 65, flush=True)

test_B = apply_pergroup(test_preds_e50, test_months, best_alphas)
n_adj_B = sum(1 for m in test_months if m in best_alphas)
print(f"  Adjusted {n_adj_B} test rows (months {sorted(best_alphas)})", flush=True)
print(f"  Alphas: {best_alphas}", flush=True)
top1_B = pd.Series([CLASSES[i] for i in np.argmax(test_B, axis=1)]).value_counts()
for cls in CLASSES:
    print(f"    {cls:<18s}: {top1_B.get(cls, 0):4d}", flush=True)
ar_str = "_".join(f"m{m}r{int(v[0]*100)}c{int(v[1]*100)}"
                  for m, v in sorted(best_alphas.items()))
save_submission(test_B, f"e72_e50_pergroup_{ar_str}", cv_map=0.3625)

# ── Variants C/D: fixed strong_rare (hypothesis test, proxy-free) ──
print("\n" + "=" * 65, flush=True)
print("VARIANTS C/D — FIXED STRONG RARE, ZERO COMMON", flush=True)
print("=" * 65, flush=True)

for alpha_rare_fixed in [0.35, 0.40]:
    alphas_fixed = {m: (alpha_rare_fixed, 0.0) for m in [2, 5, 12]}
    test_fixed = apply_pergroup(test_preds_e50, test_months, alphas_fixed)
    top1_f = pd.Series([CLASSES[i] for i in np.argmax(test_fixed, axis=1)]).value_counts()
    print(f"\n  alpha_rare={alpha_rare_fixed:.2f}, alpha_common=0.0:", flush=True)
    for cls in CLASSES:
        print(f"    {cls:<18s}: {top1_f.get(cls, 0):4d}", flush=True)
    save_submission(test_fixed,
                    f"e72_e50_strong_rare_{int(alpha_rare_fixed*100)}",
                    cv_map=0.3625)

# ── Variant E: keep E54 rare alpha, zero common ────────────────────
print("\n" + "=" * 65, flush=True)
print("VARIANT E — E54 RARE ALPHA, ZERO COMMON", flush=True)
print("=" * 65, flush=True)
# E54 uniform: Feb=0.22, May=0.12, Dec=0.24
# Here: same alpha for rare, 0 for common
alphas_e = {2: (0.22, 0.0), 5: (0.12, 0.0), 12: (0.24, 0.0)}
test_E = apply_pergroup(test_preds_e50, test_months, alphas_e)
top1_E = pd.Series([CLASSES[i] for i in np.argmax(test_E, axis=1)]).value_counts()
print(f"  Alphas: rare={{'Feb':0.22,'May':0.12,'Dec':0.24}}, common=0.0", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {top1_E.get(cls, 0):4d}", flush=True)
save_submission(test_E, "e72_e50_e54rare_nocommon", cv_map=0.3625)

# ── Variant F/G: correct per-group — high alpha GULLS, lower rest ──
# Analysis revealed the mechanism of E54: GBIF Gulls prior (0.062 Feb) <<
# E50 Gulls prediction (0.270), so uniform alpha suppresses Gulls and frees
# probability for everything else. Per-group boosting Waders was wrong because
# GBIF Waders prior (0.197) is also BELOW E50's predictions (0.293).
# Correct split: alpha_dominant (Gulls) >> alpha_rest (everything else).
# Also: Ducks GBIF prior (0.238 Feb/Dec) >> E50 prediction (~0.015) — big gain.
print("\n" + "=" * 65, flush=True)
print("VARIANTS F/G — HIGH ALPHA GULLS, LOWER REST (CORRECT DIRECTION)", flush=True)
print("=" * 65, flush=True)

GULLS_IDX = CLASS_IDX["Gulls"]
DUCKS_IDX = CLASS_IDX["Ducks"]

def apply_gull_dominant(preds, months, month_alphas_gull, month_alphas_rest):
    """High alpha for Gulls (suppress), lower for all others."""
    out = preds.copy()
    for i in range(len(out)):
        m = months[i]
        if m not in month_alphas_gull:
            continue
        ag = month_alphas_gull[m]
        ar = month_alphas_rest[m]
        prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
        adj = out[i].copy()
        for j in range(N_CLASSES):
            alpha = ag if j == GULLS_IDX else ar
            adj[j] = (1 - alpha) * out[i, j] + alpha * prior[j]
        adj = np.clip(adj, 1e-10, None)
        out[i] = adj / adj.sum()
    return out

# F: Gulls alpha = 2x E54, rest = E54 (concentrate correction on dominant class)
alphas_gull_F = {2: 0.44, 5: 0.24, 12: 0.48}
alphas_rest_F = {2: 0.22, 5: 0.12, 12: 0.24}
test_F = apply_gull_dominant(test_preds_e50, test_months,
                              alphas_gull_F, alphas_rest_F)
top1_F = pd.Series([CLASSES[i] for i in np.argmax(test_F, axis=1)]).value_counts()
print(f"\n  Variant F: Gulls alpha=2x E54, rest=E54", flush=True)
for cls in CLASSES:
    diff = top1_F.get(cls, 0) - top1.get(cls, 0)
    print(f"    {cls:<18s}: {top1_F.get(cls, 0):4d}  ({diff:+d})", flush=True)
save_submission(test_F, "e72_e50_gullhigh_2xe54", cv_map=0.3625)

# G: Gulls alpha = 3x E54, rest = E54/2 (really lean into suppressing Gulls)
alphas_gull_G = {2: 0.55, 5: 0.30, 12: 0.60}
alphas_rest_G = {2: 0.11, 5: 0.06, 12: 0.12}
test_G = apply_gull_dominant(test_preds_e50, test_months,
                              alphas_gull_G, alphas_rest_G)
top1_G = pd.Series([CLASSES[i] for i in np.argmax(test_G, axis=1)]).value_counts()
print(f"\n  Variant G: Gulls alpha=3x E54, rest=E54/2", flush=True)
for cls in CLASSES:
    diff = top1_G.get(cls, 0) - top1.get(cls, 0)
    print(f"    {cls:<18s}: {top1_G.get(cls, 0):4d}  ({diff:+d})", flush=True)
save_submission(test_G, "e72_e50_gullhigh_3xe54_resthalf", cv_map=0.3625)

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("E72 SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"  Base: E50 (LOMO 0.3625, LB 0.56 with E54 uniform priors)", flush=True)
print(f"  E54 control alphas:   {{Feb:0.22, May:0.12, Dec:0.24}}", flush=True)
print(f"  Per-group tuned:      {best_alphas}", flush=True)
print(f"\n  Top-1 Waders comparison:", flush=True)
print(f"    E50 base (no prior): {top1.get('Waders', 0)}", flush=True)
print(f"    Variant A (E54):     {top1_A.get('Waders', 0)}", flush=True)
print(f"    Variant B (pergroup):{top1_B.get('Waders', 0)}", flush=True)
print(f"    Variant E (e54rare): {top1_E.get('Waders', 0)}", flush=True)
print(f"\n  Submit order:", flush=True)
print(f"    1. e72_e50_e54alphas_control   (verify pipeline = LB 0.56)", flush=True)
print(f"    2. e72_e50_e54rare_nocommon    (core hypothesis, no proxy risk)", flush=True)
print(f"    3. e72_e50_pergroup_*          (proxy-tuned)", flush=True)
print(f"    4. e72_e50_strong_rare_35/40   — SKIP (wrong direction)", flush=True)
print(f"    5. e72_e50_gullhigh_2xe54      (key insight: suppress Gulls)", flush=True)
print(f"    6. e72_e50_gullhigh_3xe54_resthalf  (stronger Gull suppression)", flush=True)
print("\nDone.", flush=True)
