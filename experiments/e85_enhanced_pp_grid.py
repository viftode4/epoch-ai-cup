"""E85: Enhanced post-processing grid search on E79 base.

ANALYSIS FINDINGS driving this experiment:
  - Unseen mAP ~ 0.18 (only 23% of shared-month performance)
  - E79 raw = LB 0.59 (same as E50+PP = LB 0.59)
  - E80 (E79+RCS NB) = LB 0.57 (RCS NB HURTS)
  - Current gamma=0.10 is very conservative
  - Doubling unseen mAP from 0.18 to 0.37 would give LB 0.65

NEW IDEAS TO TEST:
  A. E79 + E75 NB (no RCS) with STRONGER gamma (0.15-0.50)
  B. E79 + NB WITHOUT uncertainty gate (correct ALL unseen samples)
  C. E79 + pure NB for unseen months (replace tree preds entirely)
  D. E79 + E83 TabPFN blend + PP
  E. E79 + GBIF only (no NB) to isolate GBIF contribution
  F. E79 + per-class gamma (strong for Cormorants/BoP, weak for others)

Generates multiple submission CSVs for Kaggle grid search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent

UNSEEN_MONTHS = (2, 5, 12)

# GBIF prior config (best-known from E54/E67)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

LAPLACE = 1.0
MIN_SIGMA = 0.50


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(len(CLASSES))
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals
    priors = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_gated_ratio_priors(preds, months, p_train, priors, alpha_map, tau):
    out = preds.copy()
    margin = top2_margin(out)
    changed = 0
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_nb_params(train_df: pd.DataFrame):
    """E75-style NB: size + speed + alt_mid + alt_range."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    size_idx = (
        train_df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}

    K, S = len(CLASSES), len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu, sig = {}, {}
    for feat, x in feats.items():
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        gm, gs = float(np.nanmean(x)), float(np.nanstd(x))
        if not np.isfinite(gs) or gs < MIN_SIGMA:
            gs = MIN_SIGMA
        for c in range(K):
            xc = x[y == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat], sig[feat] = mu_f, sig_f

    return size_levels, log_p_size, mu, sig, y, p_size


def compute_nb_factors(df, size_levels, log_p_size, mu, sig):
    """Compute NB likelihood factors."""
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
    loglik = log_p_size[:, size_idx].T
    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


def compute_pure_nb_probs(df, size_levels, log_p_size, mu, sig, p_train, gbif_priors, months):
    """Full NB classifier: P(class|features) using Bayes rule with GBIF priors per month."""
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)

    N = len(df)
    K = len(CLASSES)
    loglik = log_p_size[:, size_idx].T  # (N, K)

    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])

    # Add log-prior per month (GBIF-adjusted)
    log_prior = np.zeros((N, K), dtype=float)
    for i in range(N):
        m = months[i]
        if m in gbif_priors:
            log_prior[i] = np.log(np.clip(gbif_priors[m], 1e-12, None))
        else:
            log_prior[i] = np.log(np.clip(p_train, 1e-12, None))

    log_posterior = loglik + log_prior
    log_posterior = log_posterior - log_posterior.max(axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    return renorm_rows(posterior), ok


# ====================================================================
print("=" * 70, flush=True)
print("E85 ENHANCED POST-PROCESSING GRID ON E79".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data ---------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
shared_mask = ~unseen_mask

print(f"  Test: {len(test_df)} samples", flush=True)
print(f"  Unseen months ({UNSEEN_MONTHS}): {unseen_mask.sum()} ({100*unseen_mask.mean():.1f}%)", flush=True)
print(f"  Shared months: {shared_mask.sum()} ({100*shared_mask.mean():.1f}%)", flush=True)

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Build NB params
print("\nBuilding NB params...", flush=True)
size_levels, log_p_size, mu, sig, _, p_size = build_nb_params(train_df)
factors, ok_feats = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)
print(f"  Valid test samples for NB: {ok_feats.sum()}/{len(ok_feats)}", flush=True)

# -- Load base models --------------------------------------------------
print("\nLoading base models...", flush=True)
base_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))
print(f"  E79: {base_e79.shape}", flush=True)

e83_path = ROOT / "test_e83.npy"
has_e83 = e83_path.exists()
if has_e83:
    base_e83 = renorm_rows(np.load(e83_path).astype(float))
    print(f"  E83 (TabPFN): {base_e83.shape}", flush=True)


def per_month_counts(mask):
    parts = []
    for m in UNSEEN_MONTHS:
        mm = test_months == m
        parts.append(f"m{m}:{int((mask & mm).sum())}")
    return " ".join(parts)


def report(name, pred, base):
    """Report prediction change stats."""
    top_base = base.argmax(axis=1)
    top_pred = pred.argmax(axis=1)
    flips_unseen = int(((top_base != top_pred) & unseen_mask).sum())
    flips_shared = int(((top_base != top_pred) & shared_mask).sum())

    # Per-class prediction counts
    pred_counts = np.bincount(top_pred, minlength=len(CLASSES))
    base_counts = np.bincount(top_base, minlength=len(CLASSES))

    print(f"\n  >> {name}", flush=True)
    print(f"     Flips vs E79: unseen={flips_unseen}, shared={flips_shared}", flush=True)

    # Show big class count changes
    changes = []
    for i, cls in enumerate(CLASSES):
        diff = int(pred_counts[i]) - int(base_counts[i])
        if abs(diff) >= 5:
            changes.append(f"{cls}:{diff:+d}")
    if changes:
        print(f"     Class count changes: {', '.join(changes)}", flush=True)


# ====================================================================
#  A. E79 + E75 NB with STRONGER gamma
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  A. STRONGER GAMMA FOR E75-STYLE NB".center(70), flush=True)
print("=" * 70, flush=True)

# First apply GBIF priors
pred0, changed_prior = apply_gated_ratio_priors(
    base_e79, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
)
margin0 = top2_margin(pred0)
print(f"  GBIF prior: changed {changed_prior} samples", flush=True)

# Grid: gamma from gentle to aggressive
for gamma in [0.10, 0.15, 0.20, 0.30, 0.50]:
    for tau_nb in [0.30, 0.50]:
        gate = unseen_mask & ok_feats & (margin0 < tau_nb)
        out = pred0.copy()
        out[gate] = out[gate] * (factors[gate] ** gamma)
        out = renorm_rows(out)

        label = f"e85_A_e79_nb_g{gamma:.2f}_t{tau_nb:.2f}"
        report(label, out, base_e79)
        save_submission(out, label, cv_map=None)

# ====================================================================
#  B. E79 + NB WITHOUT gate (correct ALL unseen samples)
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  B. UNGATED NB (ALL UNSEEN SAMPLES)".center(70), flush=True)
print("=" * 70, flush=True)

for gamma in [0.10, 0.20, 0.30]:
    gate = unseen_mask & ok_feats  # No margin gate!
    out = pred0.copy()
    out[gate] = out[gate] * (factors[gate] ** gamma)
    out = renorm_rows(out)

    label = f"e85_B_e79_nb_ungated_g{gamma:.2f}"
    report(label, out, base_e79)
    save_submission(out, label, cv_map=None)

# ====================================================================
#  C. Pure NB for unseen, tree for shared
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  C. PURE NB FOR UNSEEN MONTHS".center(70), flush=True)
print("=" * 70, flush=True)

nb_probs, nb_ok = compute_pure_nb_probs(
    test_df, size_levels, log_p_size, mu, sig, p_train, priors, test_months
)
print(f"  NB valid: {nb_ok.sum()}/{len(nb_ok)}", flush=True)

# Show NB class distribution for unseen months
nb_top = nb_probs.argmax(axis=1)
print("  NB class distribution (unseen months):", flush=True)
for i, cls in enumerate(CLASSES):
    count = int((nb_top[unseen_mask] == i).sum())
    print(f"    {cls:15s}: {count}", flush=True)

# Blend NB with tree at various ratios for unseen months
for nb_weight in [0.3, 0.5, 0.7, 1.0]:
    out = base_e79.copy()
    mask = unseen_mask & nb_ok
    out[mask] = (1.0 - nb_weight) * base_e79[mask] + nb_weight * nb_probs[mask]
    out = renorm_rows(out)

    label = f"e85_C_e79_pureNB_w{nb_weight:.1f}"
    report(label, out, base_e79)
    save_submission(out, label, cv_map=None)

# ====================================================================
#  D. E79 + E83 TabPFN blend + PP
# ====================================================================
if has_e83:
    print("\n" + "=" * 70, flush=True)
    print("  D. E79 + E83 TABPFN BLEND".center(70), flush=True)
    print("=" * 70, flush=True)

    for tree_w in [0.5, 0.6, 0.7, 0.8]:
        # Blend base models
        blended = tree_w * base_e79 + (1.0 - tree_w) * base_e83
        blended = renorm_rows(blended)

        # Apply GBIF priors + NB PP
        pred_b, _ = apply_gated_ratio_priors(
            blended, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
        )
        margin_b = top2_margin(pred_b)
        gate = unseen_mask & ok_feats & (margin_b < 0.30)
        out = pred_b.copy()
        out[gate] = out[gate] * (factors[gate] ** 0.10)
        out = renorm_rows(out)

        label = f"e85_D_e79{int(tree_w*100)}_e83{int((1-tree_w)*100)}_pp"
        report(label, out, base_e79)
        save_submission(out, label, cv_map=None)

    # Per-month blend: TabPFN for shared, trees for unseen
    for tabpfn_shared_w in [0.3, 0.5, 0.7]:
        out = base_e79.copy()
        # Shared months: blend in TabPFN
        out[shared_mask] = (
            (1.0 - tabpfn_shared_w) * base_e79[shared_mask]
            + tabpfn_shared_w * base_e83[shared_mask]
        )
        # Unseen months: keep E79 + PP
        pred_unseen, _ = apply_gated_ratio_priors(
            out, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
        )
        margin_u = top2_margin(pred_unseen)
        gate = unseen_mask & ok_feats & (margin_u < 0.30)
        out_final = pred_unseen.copy()
        out_final[gate] = out_final[gate] * (factors[gate] ** 0.10)
        out_final = renorm_rows(out_final)

        label = f"e85_D_shared_tpfn{int(tabpfn_shared_w*100)}_unseen_e79pp"
        report(label, out_final, base_e79)
        save_submission(out_final, label, cv_map=None)

# ====================================================================
#  E. E79 + GBIF only (no NB) — isolate GBIF contribution
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  E. GBIF ONLY (NO NB)".center(70), flush=True)
print("=" * 70, flush=True)

# Stronger GBIF alphas
for alpha_scale in [1.0, 1.5, 2.0, 3.0]:
    scaled_alpha = {m: a * alpha_scale for m, a in BASE_ALPHA.items()}
    for tau in [0.15, 0.30, 0.50, 1.0]:
        out, changed = apply_gated_ratio_priors(
            base_e79, test_months, p_train, priors, scaled_alpha, tau=tau
        )
        if changed > 0:
            label = f"e85_E_e79_gbif_s{alpha_scale:.1f}_t{tau:.2f}"
            report(label, out, base_e79)
            save_submission(out, label, cv_map=None)

# ====================================================================
#  F. Per-class gamma NB
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  F. PER-CLASS GAMMA NB".center(70), flush=True)
print("=" * 70, flush=True)

# Cormorants (idx=2) and BoP (idx=0) have most headroom
# Gulls (idx=5) and Clutter (idx=1) need no correction
gamma_profiles = {
    "aggressive_weak": {
        0: 0.30,  # Birds of Prey (AP=0.61, headroom=0.39)
        1: 0.05,  # Clutter (AP=0.96, nearly perfect)
        2: 0.50,  # Cormorants (AP=0.34, most headroom)
        3: 0.20,  # Ducks (AP=0.77)
        4: 0.15,  # Geese (AP=0.82)
        5: 0.02,  # Gulls (AP=0.97, nearly perfect)
        6: 0.10,  # Pigeons (AP=0.90)
        7: 0.10,  # Songbirds (AP=0.88)
        8: 0.20,  # Waders (AP=0.71)
    },
    "moderate_weak": {
        0: 0.20, 1: 0.05, 2: 0.30, 3: 0.15, 4: 0.10,
        5: 0.02, 6: 0.10, 7: 0.05, 8: 0.15,
    },
}

for profile_name, gamma_per_class in gamma_profiles.items():
    for tau_nb in [0.30, 0.50]:
        gate = unseen_mask & ok_feats & (margin0 < tau_nb)
        out = pred0.copy()
        if gate.any():
            # Apply per-class gamma: factor^gamma_c for each class c
            gamma_vec = np.array([gamma_per_class[c] for c in range(len(CLASSES))])
            # factors[gate] is (N_gated, K), gamma_vec is (K,)
            out[gate] = out[gate] * (factors[gate] ** gamma_vec[None, :])
            out = renorm_rows(out)

        label = f"e85_F_e79_{profile_name}_t{tau_nb:.2f}"
        report(label, out, base_e79)
        save_submission(out, label, cv_map=None)

# ====================================================================
#  Summary: rank all submissions by expected value
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  SUMMARY: ALL GENERATED SUBMISSIONS".center(70), flush=True)
print("=" * 70, flush=True)

print("""
UPLOAD PRIORITY (based on analysis reasoning):

  TOP TIER (test fundamentally different approaches):
  1. e85_C_e79_pureNB_w0.5 -- 50/50 tree+NB for unseen months
     (tests: can full NB replacement help unseen months?)
  2. e85_B_e79_nb_ungated_g0.20 -- NB on ALL unseen, no gate
     (tests: does removing the uncertainty gate help?)
  3. e85_F_e79_aggressive_weak_t0.30 -- per-class gamma
     (tests: targeted correction for weak classes)

  SECOND TIER (incremental improvements):
  4. e85_A_e79_nb_g0.30_t0.30 -- 3x stronger gamma
     (tests: is current gamma too conservative?)
  5. e85_E_e79_gbif_s2.0_t0.50 -- stronger GBIF
     (tests: does more aggressive GBIF prior help?)

  TABPFN BLENDS (if TabPFN exists):
  6. e85_D_shared_tpfn50_unseen_e79pp -- TabPFN shared, tree unseen
     (tests: can TabPFN help on shared months?)
""", flush=True)

print("Done.", flush=True)
