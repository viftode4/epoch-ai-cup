"""E78: Feature-weighted NB-alt correction (refine E75 without new evidence).

E75 is the current best (LB 0.59) with a NB product-of-experts on unseen months:
  u = (radar_bird_size, airspeed, alt_mid, alt_range)
  q ∝ p^(m) * P(u|c)^gamma   for uncertain cases only

Hypothesis
----------
`alt_mid` is likely the most stable altitude cue; `alt_range` may be noisier and
partially redundant with `alt_mid`. NB multiplies both Gaussians, which can
double-count altitude evidence. A simple refinement is to **downweight** some
evidence components inside the likelihood (equivalently, apply per-feature
exponents).

We keep the same pipeline and generate two candidates:
  - A: w_alt_range=0.5 (reduce range influence)
  - B: w_alt_range=0.0 (alt_mid only; drop alt_range)
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

# Prior stage (fixed; best-known)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# NB stage
TAU_NB = 0.30
GAMMA = 0.10  # keep same as best E75

CANDIDATES: list[dict[str, float]] = [
    {"w_size": 1.0, "w_speed": 1.0, "w_alt_mid": 1.0, "w_alt_range": 0.5},  # A
    {"w_size": 1.0, "w_speed": 1.0, "w_alt_mid": 1.0, "w_alt_range": 0.0},  # B
]

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


def build_nb_params(train_df: pd.DataFrame) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (size_levels, log_p_size[K,S], mu[feat][K], sigma[feat][K])."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    size_idx = (
        train_df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
    )

    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    feats: dict[str, np.ndarray] = {
        "speed": speed,
        "alt_mid": alt_mid,
        "alt_range": alt_range,
    }

    K = len(CLASSES)
    S = len(size_levels)

    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu: dict[str, np.ndarray] = {}
    sig: dict[str, np.ndarray] = {}
    for feat, x in feats.items():
        mu_f = np.zeros(K, dtype=float)
        sig_f = np.zeros(K, dtype=float)
        global_mu = float(np.nanmean(x))
        global_sig = float(np.nanstd(x))
        if not np.isfinite(global_sig) or global_sig < MIN_SIGMA:
            global_sig = MIN_SIGMA
        for c in range(K):
            xc = x[y == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c] = global_mu
                sig_f[c] = global_sig
        mu[feat] = mu_f
        sig[feat] = sig_f

    return size_levels, log_p_size, mu, sig


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def compute_weighted_factors(
    df: pd.DataFrame,
    size_levels: list[str],
    log_p_size: np.ndarray,
    mu: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
    )

    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    # Weighted log-likelihood
    loglik = weights["w_size"] * (log_p_size[:, size_idx].T)
    loglik = loglik + weights["w_speed"] * log_gaussian(speed, mu["speed"], sig["speed"])
    loglik = loglik + weights["w_alt_mid"] * log_gaussian(alt_mid, mu["alt_mid"], sig["alt_mid"])
    loglik = loglik + weights["w_alt_range"] * log_gaussian(alt_range, mu["alt_range"], sig["alt_range"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik)


print("=" * 70, flush=True)
print("E78 FEATURE-WEIGHTED NB-ALT (E75 REFINEMENT)".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)

# Base model probabilities
base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

# Priors stage (E67)
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)
pred0, changed_prior = apply_gated_ratio_priors(base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
print(f"\nPrior stage: tau_prior={TAU_PRIOR:.2f} changed_rows={changed_prior}", flush=True)

margin0 = top2_margin(pred0)
gate = unseen_mask & (margin0 < TAU_NB)
print(f"NB gate: tau_nb={TAU_NB:.2f} gated_rows={int(gate.sum())}", flush=True)

# NB params (from full train)
size_levels, log_p_size, mu, sig = build_nb_params(train_df)

print("\nGenerating candidates...", flush=True)
for weights in CANDIDATES:
    factors = compute_weighted_factors(test_df, size_levels, log_p_size, mu, sig, weights)
    out = pred0.copy()
    out[gate] = out[gate] * (factors[gate] ** GAMMA)
    out = renorm_rows(out)

    tag = (
        f"ws{weights['w_size']:.2f}_wv{weights['w_speed']:.2f}"
        f"_wm{weights['w_alt_mid']:.2f}_wr{weights['w_alt_range']:.2f}"
    )
    print(f"  {tag}", flush=True)
    save_submission(
        out,
        f"e78_nbalt_weighted_{tag}_tau{TAU_NB:.2f}_g{GAMMA:.2f}",
        cv_map=None,
    )

print("\nDone.", flush=True)

