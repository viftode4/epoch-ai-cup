"""E103: Pair-gated Birds-of-Prey (BoP) soaring evidence on top of best NB-alt pipeline.

Why
---
E102 failed badly (LB 0.51) because high-dimensional diagonal-Gaussian evidence can
over-sharpen (log-likelihood sums scale with dimension) and because some trajectory-shape
features are not domain-invariant.

This experiment keeps the proven recipe (E96/E98/E100):
  base preds (test_e50) -> gated GBIF ratio priors (unseen only) -> gated NB evidence update
and adds a *targeted* trajectory expert that is:
  - low-dimensional (single scalar score)
  - trained as a 1D likelihood ratio: log p(score|BoP) - log p(score|not-BoP)
  - pair-gated: only applied when BoP is in the top-2 on unseen months and margin is small

If this helps, it should improve BoP ranking specifically without perturbing the rest.
If it fails, it tells us BoP-vs-others errors are not driven by circling/soaring cues in the
unseen months, or those cues are too noisy at 1Hz.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

UNSEEN_MONTHS = (2, 5, 12)

# Priors stage (best known)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Base NB evidence (same as E96-ish)
TAU_NB = 0.25
GAMMA_NB = 0.10
W_SPEED = 1.00
W_ALTMID = 1.00
W_ALTRANGE = 0.50
W_HEADING = 1.00
W_AC1 = 1.00

LAPLACE = 1.0
DEFAULT_MIN_SIGMA = 0.50
MIN_SIGMA = {
    "speed": 0.50,
    "alt_mid": 0.50,
    "alt_range": 0.50,
    "heading_R": 0.10,
    "rcs_ac1": 0.10,
}

# BoP expert gate + strength
TAU_BOP = 0.30
GAMMA_BOP_LIST = (0.10, 0.15)


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def top2_classes(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-pred, axis=1)
    return order[:, 0], order[:, 1]


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si: dict[int, np.ndarray] = {}
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

    priors: dict[int, np.ndarray] = {}
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


def extract_heading_ac1(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (heading_R, rcs_ac1, ok_mask) from raw trajectories."""
    n = len(df)
    heading_r = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction (heading/ac1): {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            npts = len(pts)
            if npts < 6:
                continue

            rcs = np.array([p[3] for p in pts], dtype=float)
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            rcs_c = rcs - float(np.mean(rcs))
            var_rcs = float(np.var(rcs_c))
            if var_rcs > 1e-12:
                ac1_val = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs)
                if np.isfinite(ac1_val):
                    ac1[i] = ac1_val

            times = parse_trajectory_time(row["trajectory_time"])
            _ = np.diff(times)
            dx = np.diff(lons) * 67000.0
            dy = np.diff(lats) * 111000.0
            headings = np.arctan2(dy, dx)
            if len(headings) > 1:
                R = float(np.sqrt(np.mean(np.sin(headings)) ** 2 + np.mean(np.cos(headings)) ** 2))
                if np.isfinite(R):
                    heading_r[i] = R
        except Exception:
            continue

    ok = np.isfinite(heading_r) & np.isfinite(ac1)
    heading_r = np.where(np.isfinite(heading_r), heading_r, 0.0)
    ac1 = np.where(np.isfinite(ac1), ac1, 0.0)
    print(f"  Evidence valid (heading/ac1): {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)
    return heading_r, ac1, ok


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_nb_params(
    df: pd.DataFrame,
    y: np.ndarray,
    heading_r: np.ndarray,
    ac1: np.ndarray,
    ok: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}

    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
        .astype(int)
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    feats: dict[str, np.ndarray] = {
        "speed": speed,
        "alt_mid": alt_mid,
        "alt_range": alt_range,
        "heading_R": heading_r,
        "rcs_ac1": ac1,
    }

    # P(size|class)
    K, S = N_CLASSES, len(size_levels)
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
    for feat_name, x in feats.items():
        min_s = MIN_SIGMA.get(feat_name, DEFAULT_MIN_SIGMA)
        x_use = np.where(ok, x, np.nan) if feat_name in ("heading_R", "rcs_ac1") else x

        gm = float(np.nanmean(x_use))
        gs = float(np.nanstd(x_use))
        if not np.isfinite(gs) or gs < min_s:
            gs = min_s

        mu_f = np.full(K, gm, dtype=float)
        sig_f = np.full(K, gs, dtype=float)
        for c in range(K):
            xc = x_use[y == c]
            ok_c = np.isfinite(xc)
            if ok_c.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > min_s else min_s

        mu[feat_name] = mu_f
        sig[feat_name] = sig_f

    return size_levels, log_p_size, mu, sig


def compute_log_p_u_given_c(
    df: pd.DataFrame,
    size_levels: list[str],
    log_p_size: np.ndarray,
    mu: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
    heading_r: np.ndarray,
    ac1: np.ndarray,
    ok: np.ndarray,
) -> np.ndarray:
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
        .astype(int)
    )
    loglike = log_p_size[:, size_idx].T  # (n, K)

    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    channels: list[tuple[str, np.ndarray, float, np.ndarray]] = [
        ("speed", speed, W_SPEED, np.isfinite(speed)),
        ("alt_mid", alt_mid, W_ALTMID, np.isfinite(alt_mid)),
        ("alt_range", alt_range, W_ALTRANGE, np.isfinite(alt_range)),
        ("heading_R", heading_r, W_HEADING, ok & np.isfinite(heading_r)),
        ("rcs_ac1", ac1, W_AC1, ok & np.isfinite(ac1)),
    ]
    for feat_name, x, w, valid in channels:
        if w == 0 or valid.sum() == 0:
            continue
        lg = log_gaussian(np.where(np.isfinite(x), x, 0.0), mu[feat_name], sig[feat_name])
        loglike[valid] += w * lg[valid]
    return loglike


def apply_nb_poe(base: np.ndarray, loglike: np.ndarray, gamma: float, gate: np.ndarray) -> np.ndarray:
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)
    ll = loglike[gate]
    ll = ll - ll.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def extract_bop_score(df: pd.DataFrame) -> np.ndarray:
    """Scalar circling/soaring score from trajectory (no training labels).

    Uses only stable-ish cues:
      - heading circular resultant length R (lower => more turning)
      - altitude gain rate (higher => soaring/climbing)
      - slow flight fraction proxy via ground speeds (lower mean speed tends to BoP soaring)

    Score is standardized later using train distributions.
    """
    n = len(df)
    score = np.full(n, np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction (BoP score): {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 8:
                continue
            times = parse_trajectory_time(row["trajectory_time"])
            if len(times) != len(pts) or times[-1] <= times[0]:
                continue
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)
            alts = np.array([p[2] for p in pts], dtype=float)

            dt = np.maximum(np.diff(times), 0.001)
            dx = np.diff(lons) * 67000.0
            dy = np.diff(lats) * 111000.0
            headings = np.arctan2(dy, dx)
            R = float(np.sqrt(np.mean(np.sin(headings)) ** 2 + np.mean(np.cos(headings)) ** 2))

            # ground speed proxy
            sp = np.sqrt(dx * dx + dy * dy) / dt
            sp_mean = float(np.mean(sp))
            slow_frac = float(np.mean(sp < 10.0))

            # altitude gain rate
            dur = float(times[-1] - times[0])
            alt_gain = float(np.sum(np.maximum(np.diff(alts), 0.0)))
            gain_rate = alt_gain / max(dur, 0.01)

            # Combine into a single score. BoP: low R (turning), higher gain_rate, higher slow_frac.
            s = (1.0 - R) + 0.50 * np.tanh(gain_rate / 1.5) + 0.50 * slow_frac + 0.10 * np.tanh((12.0 - sp_mean) / 4.0)
            if np.isfinite(s):
                score[i] = s
        except Exception:
            continue

    score = np.where(np.isfinite(score), score, 0.0)
    return score


def fit_1d_gauss(x: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(s) or s < 0.10:
        s = 0.10
    return m, s


def logpdf_1d(x: np.ndarray, m: float, s: float) -> np.ndarray:
    z = (x - m) / s
    return -0.5 * z * z - np.log(s)


def main() -> None:
    print("=" * 70, flush=True)
    print("E103 BoP PAIR-GATED SOARING LR".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

    # --- Stage 1: month priors (unseen only, gated) ---
    test_p0, changed = apply_gated_ratio_priors(test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
    print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

    # --- Stage 2: NB evidence (unseen only, gated) ---
    print("\nExtracting heading/ac1 on train...", flush=True)
    tr_heading, tr_ac1, tr_ok = extract_heading_ac1(train_df)
    print("\nExtracting heading/ac1 on test...", flush=True)
    te_heading, te_ac1, te_ok = extract_heading_ac1(test_df)

    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, tr_heading, tr_ac1, tr_ok)
    ll_test = compute_log_p_u_given_c(test_df, size_levels, log_p_size, mu, sig, te_heading, te_ac1, te_ok)

    margin0 = top2_margin(test_p0)
    gate_nb = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
    print(f"\nNB gate: unseen only, tau_nb={TAU_NB:.2f} rows={int(gate_nb.sum())}", flush=True)
    test_p1 = apply_nb_poe(test_p0, ll_test, gamma=GAMMA_NB, gate=gate_nb)

    # --- Stage 3: BoP pair-gated 1D LR expert ---
    bop_idx = CLASSES.index("Birds of Prey")

    print("\nExtracting BoP soaring score on train...", flush=True)
    tr_score = extract_bop_score(train_df)
    print("\nExtracting BoP soaring score on test...", flush=True)
    te_score = extract_bop_score(test_df)

    # Standardize using train mean/std for numerical stability
    m_all = float(np.mean(tr_score))
    s_all = float(np.std(tr_score))
    if not np.isfinite(s_all) or s_all < 1e-6:
        s_all = 1.0
    tr_z = (tr_score - m_all) / s_all
    te_z = (te_score - m_all) / s_all

    z_bop = tr_z[y == bop_idx]
    z_not = tr_z[y != bop_idx]
    mb, sb = fit_1d_gauss(z_bop)
    mn, sn = fit_1d_gauss(z_not)
    print(f"\nBoP score Gaussians: BoP mean={mb:.3f} sd={sb:.3f} | not-BoP mean={mn:.3f} sd={sn:.3f}", flush=True)

    loglr = logpdf_1d(te_z, mb, sb) - logpdf_1d(te_z, mn, sn)

    # Gate: unseen month, low margin, and BoP in top-2 (after NB stage)
    m1 = top2_margin(test_p1)
    t1, t2 = top2_classes(test_p1)
    gate_bop = np.isin(test_months, UNSEEN_MONTHS) & (m1 < TAU_BOP) & ((t1 == bop_idx) | (t2 == bop_idx))
    print(f"BoP gate: unseen & margin<{TAU_BOP:.2f} & BoP in top2 rows={int(gate_bop.sum())}", flush=True)

    for g_bop in GAMMA_BOP_LIST:
        out = test_p1.copy()
        if gate_bop.sum() > 0:
            fac = np.exp(np.clip(g_bop * loglr[gate_bop], -10.0, 10.0))
            out[gate_bop, bop_idx] *= fac
            out[gate_bop] = out[gate_bop] / np.clip(out[gate_bop].sum(axis=1, keepdims=True), 1e-12, None)
        out = renorm_rows(out)

        name = f"e103_boplr_tau{TAU_BOP:.2f}_g{g_bop:.2f}_nbtau{TAU_NB:.2f}_priortau{TAU_PRIOR:.2f}"
        save_submission(out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

