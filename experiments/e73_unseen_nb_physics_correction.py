"""E73: Unseen-month Naive-Bayes physics correction on top of E67 priors.

Motivation
----------
E54/E67 plateau at LB=0.56 uses *month* priors to correct cross-month ranking on
unseen months (2/5/12), optionally uncertainty-gated. E70 and E71 show:
  - small unseen-month perturbations that don't change rankings won't move LB,
  - shared-month blending doesn't break the plateau.

Hypothesis
----------
Remaining LB headroom is in *within-month* confusions on unseen months, which
month priors cannot fully resolve. We add a second, biologically stable signal:
tabular "physics" cues (airspeed + radar_bird_size) via a simple Naive Bayes
likelihood model learned from train.

Method
------
1) Start from `test_e50.npy`.
2) Apply E67-style gated GBIF ratio priors (tau=0.15, E54 winter alphas).
3) For unseen months only (2/5/12) and moderately uncertain predictions
   (margin < TAU_NB), apply a product-of-experts correction:

     q_{i,c} ∝ p_{i,c} · exp( log P(size_i | c) + log N(speed_i | μ_c, σ_c) )^γ

   Then renormalize rows.

This correction is per-sample (can change ranking) but uses only stable cues.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

UNSEEN_MONTHS = (2, 5, 12)

# E54 winter alphas + E67 gate (known good)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# NB correction settings
TAU_NB = 0.25  # apply to moderately uncertain examples
GAMMA = 0.12  # strength of NB product-of-experts (small to avoid over-correction)
LAPLACE = 1.0  # size-prob smoothing
MIN_SIGMA = 0.50  # m/s


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


def apply_gated_ratio_priors(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
    tau: float,
) -> tuple[np.ndarray, int]:
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


def build_nb_factors(train_df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (size_levels, log_p_size[K,S], mu_speed[K], sigma_speed[K])."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    size_idx = train_df["radar_bird_size"].fillna("__UNK__").map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)

    K = len(CLASSES)
    S = len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)

    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            vc = np.bincount(size_idx[mask], minlength=S).astype(float)
            counts_cs[c] = vc

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu = np.zeros(K, dtype=float)
    sig = np.zeros(K, dtype=float)
    global_mu = float(np.nanmean(speed))
    global_sig = float(np.nanstd(speed)) if float(np.nanstd(speed)) > MIN_SIGMA else MIN_SIGMA
    for c in range(K):
        s = speed[y == c]
        if np.isfinite(s).sum() >= 5:
            mu[c] = float(np.nanmean(s))
            sig_c = float(np.nanstd(s))
            sig[c] = sig_c if sig_c > MIN_SIGMA else MIN_SIGMA
        else:
            mu[c] = global_mu
            sig[c] = global_sig
    return size_levels, log_p_size, mu, sig


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized log N(x | mu, sigma) for x shape (N,), mu/sigma shape (K,)."""
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


print("=" * 70, flush=True)
print("E73: UNSEEN NB PHYSICS CORRECTION (E67 + NB)".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)

# Base model predictions
base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

# Train priors for ratio-tilt
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Apply E67 priors first
pred, changed_prior = apply_gated_ratio_priors(base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
print(f"\nE67 stage: tau={TAU_PRIOR:.2f} changed_rows={changed_prior}", flush=True)

# Build NB factors from train
size_levels, log_p_size, mu_speed, sig_speed = build_nb_factors(train_df)
size_to_idx = {s: i for i, s in enumerate(size_levels)}

test_size_idx = (
    test_df["radar_bird_size"]
    .fillna("__UNK__")
    .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
    .values
)
test_speed = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)

loglik = log_p_size[:, test_size_idx].T  # (N,K)
speed_ok = np.isfinite(test_speed)
if speed_ok.any():
    loglik[speed_ok] = loglik[speed_ok] + log_gaussian(test_speed[speed_ok], mu_speed, sig_speed)

# Stabilize factors per row (subtract max); exp -> (0,1]
loglik = loglik - loglik.max(axis=1, keepdims=True)
factors = np.exp(loglik)

# Apply NB correction (only unseen months + moderate uncertainty + speed present)
margin = top2_margin(pred)
gate_nb = unseen_mask & (margin < TAU_NB) & speed_ok

pred2 = pred.copy()
pred2[gate_nb] = pred2[gate_nb] * (factors[gate_nb] ** GAMMA)
pred2 = renorm_rows(pred2)

print(f"\nNB stage: tau_nb={TAU_NB:.2f} gamma={GAMMA:.2f} gated_rows={int(gate_nb.sum())}", flush=True)
for m in UNSEEN_MONTHS:
    mm = test_months == m
    print(f"  month={m}: n={int(mm.sum())} gated={int((gate_nb & mm).sum())}", flush=True)

save_submission(
    pred2,
    f"e73_nbphys_unseen_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}",
    cv_map=None,
)

print("\nDone.", flush=True)

