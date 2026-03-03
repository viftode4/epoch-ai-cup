"""E104: Apply the proven unseen-month PP to multiple base posteriors, then ensemble.

Motivation
----------
Public LB is computed on ~24% of test. Many materially different pipelines tie at 0.59.
We also observed that our "new experts" often touch only O(10) rows, so they won't move the
public slice unless those rows are included.

To break a public tie (and more importantly improve private), the most reliable lever is to
change the *starting ranking* (base posterior) while keeping the robust PP stages:
  - GBIF ratio priors on unseen months only (gated)
  - NB physics evidence (size + speed + alt + heading_R + rcs_ac1) on unseen months only (gated)

We run that PP on several saved base predictions (`test_e*.npy`) and then ensemble the
post-processed predictions.

Outputs
-------
Saves 4 candidates:
  - geo3:  geometric mean of PP(e11,e50,e52)
  - avg3:  arithmetic mean of PP(e11,e50,e52)
  - geoD:  geometric mean of PP(e11,e42,e52)  (more diverse, riskier)
  - avgD:  arithmetic mean of PP(e11,e42,e52)
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

# Evidence stage (same family as E96)
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
            print(f"  Evidence extraction: {i}/{n}", flush=True)
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
    print(f"  Evidence valid: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)
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
        .values.astype(int)
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
        .values.astype(int)
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


def pp_one_base(base: np.ndarray, months: np.ndarray, p_train: np.ndarray, priors: dict[int, np.ndarray], loglike: np.ndarray) -> np.ndarray:
    p0, _ = apply_gated_ratio_priors(base, months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
    gate_nb = np.isin(months, UNSEEN_MONTHS) & (top2_margin(p0) < TAU_NB)
    return apply_nb_poe(p0, loglike, gamma=GAMMA_NB, gate=gate_nb)


def ens_geo(stack: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(stack, 1e-15, 1.0))
    m = np.mean(logp, axis=0)
    p = np.exp(m)
    p = np.clip(p, 1e-15, None)
    return p / p.sum(axis=1, keepdims=True)


def ens_avg(stack: np.ndarray) -> np.ndarray:
    p = np.mean(stack, axis=0)
    p = np.clip(p, 1e-15, None)
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    print("=" * 70, flush=True)
    print("E104 MULTI-BASE PP ENSEMBLE".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    print("\nExtracting trajectory evidence (heading/ac1) on train...", flush=True)
    tr_heading, tr_ac1, tr_ok = extract_heading_ac1(train_df)
    print("\nExtracting trajectory evidence (heading/ac1) on test...", flush=True)
    te_heading, te_ac1, te_ok = extract_heading_ac1(test_df)

    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, tr_heading, tr_ac1, tr_ok)
    loglike_test = compute_log_p_u_given_c(test_df, size_levels, log_p_size, mu, sig, te_heading, te_ac1, te_ok)

    bases = {
        "e11": renorm_rows(np.load(ROOT / "test_e11.npy").astype(float)),
        "e42": renorm_rows(np.load(ROOT / "test_e42.npy").astype(float)),
        "e50": renorm_rows(np.load(ROOT / "test_e50.npy").astype(float)),
        "e52": renorm_rows(np.load(ROOT / "test_e52.npy").astype(float)),
    }
    for k, v in bases.items():
        print(f"Loaded base {k}: shape={v.shape}", flush=True)

    pp = {k: pp_one_base(v, test_months, p_train, priors, loglike_test) for k, v in bases.items()}

    stack3 = np.stack([pp["e11"], pp["e50"], pp["e52"]], axis=0)
    stackD = np.stack([pp["e11"], pp["e42"], pp["e52"]], axis=0)

    geo3 = ens_geo(stack3)
    avg3 = ens_avg(stack3)
    geoD = ens_geo(stackD)
    avgD = ens_avg(stackD)

    save_submission(geo3, "e104_ppens_geo3", cv_map=None)
    save_submission(avg3, "e104_ppens_avg3", cv_map=None)
    save_submission(geoD, "e104_ppens_geoD", cv_map=None)
    save_submission(avgD, "e104_ppens_avgD", cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

