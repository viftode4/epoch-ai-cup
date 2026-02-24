"""E75: Extend E73 NB correction with altitude cues (min/max Z).

E73/E74 showed that a *per-sample* physics likelihood (airspeed + radar size),
applied only on unseen months, breaks the 0.56 plateau and reaches LB=0.58.

Hypothesis
----------
Add additional *stable* tabular evidence that should generalize across months:
absolute altitude and altitude variability. In train, `min_z`/`max_z` separate
low-altitude classes (Ducks/Songbirds) from high-altitude classes (Geese/Waders).

We implement a Naive-Bayes product-of-experts on unseen months:
  q_{i,c} ∝ p_{i,c} · L_c(size_i, speed_i, alt_mid_i, alt_range_i)^gamma

where:
  - size is categorical with Laplace smoothing
  - (speed, alt_mid, alt_range) are class-conditional Gaussians (diagonal NB)

Pipeline
--------
test_e50.npy  ->  E67 gated GBIF ratio priors (tau_prior=0.15)  ->  NB-alt correction
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

# Prior stage (fixed; best-known configuration)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# NB-alt stage candidates (tau_nb, gamma)
# Crossfit-on-train suggested tau_nb≈0.30 with smaller gamma (because we add 2 extra Gaussians).
CANDIDATES: list[tuple[float, float]] = [
    (0.30, 0.10),  # best crossfit in quick scan
    (0.30, 0.08),  # slightly more conservative strength
]

LAPLACE = 1.0
MIN_SIGMA = 0.50  # units: m or m/s, acts as a floor on std


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


def build_nb_params(
    train_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
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


def compute_nb_factors(
    df: pd.DataFrame,
    size_levels: list[str],
    log_p_size: np.ndarray,
    mu: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (factors[N,K], ok_mask[N]) where ok_mask means all continuous feats finite."""
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

    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)

    loglik = log_p_size[:, size_idx].T  # (N,K)
    if ok.any():
        loglik[ok] = loglik[ok] + log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] = loglik[ok] + log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] = loglik[ok] + log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


print("=" * 70, flush=True)
print("E75 NB-PHYS + ALTITUDE (E67 + NB-alt)".center(70), flush=True)
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

# NB-alt factors
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors, ok_feats = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)
margin0 = top2_margin(pred0)


def per_month_counts(mask: np.ndarray) -> str:
    parts = []
    for m in UNSEEN_MONTHS:
        mm = test_months == m
        parts.append(f"m{m}:{int((mask & mm).sum())}")
    return " ".join(parts)


print("\nGenerating candidates...", flush=True)
for tau_nb, gamma in CANDIDATES:
    gate = unseen_mask & ok_feats & (margin0 < tau_nb)
    out = pred0.copy()
    out[gate] = out[gate] * (factors[gate] ** gamma)
    out = renorm_rows(out)

    top_before = pred0.argmax(axis=1)
    top_after = out.argmax(axis=1)
    top_flip = int(((top_before != top_after) & unseen_mask).sum())

    print(
        f"\n  tau_nb={tau_nb:.2f} gamma={gamma:.2f} "
        f"gated_rows={int(gate.sum())} ({per_month_counts(gate)}) "
        f"top1_flips_unseen={top_flip}",
        flush=True,
    )

    save_submission(
        out,
        f"e75_nbalt_unseen_tau{tau_nb:.2f}_g{gamma:.2f}_priortau{TAU_PRIOR:.2f}",
        cv_map=None,
    )

print("\nDone.", flush=True)

