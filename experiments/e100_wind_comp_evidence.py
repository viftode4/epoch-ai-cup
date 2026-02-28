"""E100: Wind-compensation physics evidence as a gated PoE stage (unseen months only).

Rationale
---------
We have plateaued at ~0.59 on the (noisy) public LB (~24% of test). Micro-tweaks to
priors and small NB evidence changes often tie on public, so we need a new *physics-based*
signal that is plausibly invariant and targets within-unseen-month ranking errors.

Deep-research suggests using wind-decoupled kinematics and drift compensation:
  v_air = v_ground - v_wind
and derived invariants like drift angle and headwind/crosswind components.

We already have wind components in data/{train,test}_weather.csv (wind_u, wind_v).
This experiment:
  1) Starts from base predictions (test_e50.npy).
  2) Applies the proven gated GBIF ratio priors (unseen only).
  3) Applies the proven baseline NB evidence update (size + speed + alt_mid + alt_range +
     heading_R + rcs_ac1) on unseen-only uncertain rows.
  4) Applies an *additional* wind-compensation evidence update on still-uncertain rows.

This isolates the new contribution and reduces risk of broad regressions.
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

# Baseline evidence stage (as in E96)
TAU_BASE = 0.25
GAMMA_BASE = 0.10
W_SPEED = 1.00
W_ALTMID = 1.00
W_ALTRANGE = 0.50
W_HEADING = 1.00
W_AC1 = 1.00

# Wind evidence stage (new; conservative weights)
W_AIR_WIND = 0.50
W_DRIFT = 0.35
W_HEADWIND = 0.25
W_CROSSWIND = 0.25

LAPLACE = 1.0

DEFAULT_MIN_SIGMA = 0.50
MIN_SIGMA = {
    # baseline
    "speed": 0.50,
    "alt_mid": 0.50,
    "alt_range": 0.50,
    "heading_R": 0.10,
    "rcs_ac1": 0.10,
    # wind
    "airspeed_wind": 0.50,
    "drift_angle": 0.10,
    "headwind": 0.50,
    "crosswind": 0.50,
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
    n = len(df)
    heading_r = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 6:
                continue
            rcs = np.array([p[3] for p in pts], dtype=float)
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            # rcs_ac1
            rcs_c = rcs - float(np.mean(rcs))
            var_rcs = float(np.var(rcs_c))
            if var_rcs > 1e-12:
                ac1_val = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs)
                if np.isfinite(ac1_val):
                    ac1[i] = ac1_val

            # heading_R
            times = parse_trajectory_time(row["trajectory_time"])
            _ = np.diff(times)
            mean_lat = float(np.mean(lats))
            lon_scale = 111000.0 * float(np.cos(np.deg2rad(mean_lat)))
            dx = np.diff(lons) * lon_scale
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


def extract_ground_velocity(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average ground-velocity vector (east,north) in m/s per track."""
    n = len(df)
    vg_e = np.full(n, np.nan)
    vg_n = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Ground velocity: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            times = parse_trajectory_time(row["trajectory_time"])
            if len(pts) < 3 or len(times) < 3:
                continue
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            dt = np.diff(times).astype(float)
            dt = np.maximum(dt, 1e-3)
            dur = float(times[-1] - times[0])
            if not np.isfinite(dur) or dur <= 1e-3:
                continue

            mean_lat = float(np.mean(lats))
            lon_scale = 111000.0 * float(np.cos(np.deg2rad(mean_lat)))
            dx = np.diff(lons) * lon_scale
            dy = np.diff(lats) * 111000.0

            # Weighted average velocity over segments
            ve = float(np.sum(dx) / np.sum(dt))
            vn = float(np.sum(dy) / np.sum(dt))

            if np.isfinite(ve) and np.isfinite(vn) and (ve * ve + vn * vn) > 0.25:
                vg_e[i] = ve
                vg_n[i] = vn
                ok[i] = True
        except Exception:
            continue

    vg_e = np.where(np.isfinite(vg_e), vg_e, 0.0)
    vg_n = np.where(np.isfinite(vg_n), vg_n, 0.0)
    print(f"  Ground vel valid: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)
    return vg_e, vg_n, ok


def build_size_logprob(train_df: pd.DataFrame, y: np.ndarray) -> tuple[list[str], np.ndarray]:
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        train_df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )

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
    return size_levels, log_p_size


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_gauss_params(x: np.ndarray, y: np.ndarray, min_sigma: float, ok_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = N_CLASSES
    x_use = np.where(ok_mask, x, np.nan)
    gm = float(np.nanmean(x_use))
    gs = float(np.nanstd(x_use))
    if not np.isfinite(gs) or gs < min_sigma:
        gs = min_sigma
    mu = np.full(K, gm, dtype=float)
    sig = np.full(K, gs, dtype=float)
    for c in range(K):
        xc = x_use[y == c]
        ok = np.isfinite(xc)
        if ok.sum() >= 5:
            mu[c] = float(np.nanmean(xc))
            sc = float(np.nanstd(xc))
            sig[c] = sc if sc > min_sigma else min_sigma
    return mu, sig


def apply_poe(base: np.ndarray, loglike: np.ndarray, gate: np.ndarray, gamma: float) -> np.ndarray:
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)
    ll = loglike[gate]
    ll = ll - ll.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def main() -> None:
    print("=" * 70, flush=True)
    print("E100 WIND-COMP EVIDENCE PoE (UNSEEN ONLY)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    # Weather (wind components)
    train_wx = pd.read_csv(ROOT / "data" / "train_weather.csv")
    test_wx = pd.read_csv(ROOT / "data" / "test_weather.csv")
    assert len(train_wx) == len(train_df) and len(test_wx) == len(test_df), "Weather rows mismatch"
    w_u_tr = pd.to_numeric(train_wx["wind_u"], errors="coerce").values.astype(float)
    w_v_tr = pd.to_numeric(train_wx["wind_v"], errors="coerce").values.astype(float)
    w_u_te = pd.to_numeric(test_wx["wind_u"], errors="coerce").values.astype(float)
    w_v_te = pd.to_numeric(test_wx["wind_v"], errors="coerce").values.astype(float)

    # Base predictions
    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
    print(f"\nBase preds: test_e50.npy shape={test_base.shape}", flush=True)

    # Baseline evidence extraction
    print("\nExtracting baseline trajectory evidence on train...", flush=True)
    tr_heading, tr_ac1, tr_ok0 = extract_heading_ac1(train_df)
    print("\nExtracting baseline trajectory evidence on test...", flush=True)
    te_heading, te_ac1, te_ok0 = extract_heading_ac1(test_df)

    # Wind evidence extraction (ground velocity vectors)
    print("\nExtracting ground velocity vectors on train...", flush=True)
    vg_e_tr, vg_n_tr, vg_ok_tr = extract_ground_velocity(train_df)
    print("\nExtracting ground velocity vectors on test...", flush=True)
    vg_e_te, vg_n_te, vg_ok_te = extract_ground_velocity(test_df)

    # Build wind-comp features
    def build_wind_features(vg_e, vg_n, w_u, w_v, ok):
        vg_norm = np.sqrt(vg_e * vg_e + vg_n * vg_n) + 1e-12
        vg_hat_e = vg_e / vg_norm
        vg_hat_n = vg_n / vg_norm
        # air velocity
        va_e = vg_e - w_u
        va_n = vg_n - w_v
        air_w = np.sqrt(va_e * va_e + va_n * va_n)
        # drift angle between va and vg (absolute)
        dot = vg_e * va_e + vg_n * va_n
        cross = vg_e * va_n - vg_n * va_e
        drift = np.abs(np.arctan2(cross, dot))
        # headwind component (positive = headwind)
        headwind = -(w_u * vg_hat_e + w_v * vg_hat_n)
        # crosswind magnitude
        proj = (w_u * vg_hat_e + w_v * vg_hat_n)
        cw_e = w_u - proj * vg_hat_e
        cw_n = w_v - proj * vg_hat_n
        crosswind = np.sqrt(cw_e * cw_e + cw_n * cw_n)
        ok2 = ok & np.isfinite(air_w) & np.isfinite(drift) & np.isfinite(headwind) & np.isfinite(crosswind)
        return air_w, drift, headwind, crosswind, ok2

    air_tr, drift_tr, head_tr, cross_tr, ok_w_tr = build_wind_features(vg_e_tr, vg_n_tr, w_u_tr, w_v_tr, vg_ok_tr)
    air_te, drift_te, head_te, cross_te, ok_w_te = build_wind_features(vg_e_te, vg_n_te, w_u_te, w_v_te, vg_ok_te)

    # Diagnose redundancy with provided airspeed (if highly correlated, wind-airspeed is redundant)
    a_obs = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    corr = np.corrcoef(np.where(np.isfinite(a_obs), a_obs, 0.0), np.where(np.isfinite(air_tr), air_tr, 0.0))[0, 1]
    print(f"\nDiag: corr(airspeed, airspeed_wind) on train = {corr:.3f}", flush=True)

    # Stage 1: priors
    test_p0, changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

    # Stage 2: baseline evidence (size + speed + altitude + heading/ac1)
    size_levels, log_p_size = build_size_logprob(train_df, y)
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx_te = (
        test_df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )
    loglike0 = log_p_size[:, size_idx_te].T  # (n_test, K)

    # baseline continuous channels
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    minz_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    maxz_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    altmid_tr = 0.5 * (minz_tr + maxz_tr)
    altrange_tr = (maxz_tr - minz_tr)

    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    minz_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    maxz_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    altmid_te = 0.5 * (minz_te + maxz_te)
    altrange_te = (maxz_te - minz_te)

    # Use intersection ok mask for baseline evidence
    ok_base_tr = tr_ok0 & np.isfinite(speed_tr) & np.isfinite(altmid_tr) & np.isfinite(altrange_tr)
    ok_base_te = te_ok0 & np.isfinite(speed_te) & np.isfinite(altmid_te) & np.isfinite(altrange_te)

    for name, x_tr, x_te, w in [
        ("speed", speed_tr, speed_te, W_SPEED),
        ("alt_mid", altmid_tr, altmid_te, W_ALTMID),
        ("alt_range", altrange_tr, altrange_te, W_ALTRANGE),
        ("heading_R", tr_heading, te_heading, W_HEADING),
        ("rcs_ac1", tr_ac1, te_ac1, W_AC1),
    ]:
        if w == 0:
            continue
        mu, sig = build_gauss_params(x_tr, y, MIN_SIGMA.get(name, DEFAULT_MIN_SIGMA), ok_base_tr)
        lg = log_gaussian(np.where(np.isfinite(x_te), x_te, 0.0), mu, sig)
        valid = ok_base_te & np.isfinite(x_te)
        loglike0[valid] += w * lg[valid]

    margin0 = top2_margin(test_p0)
    gate0 = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_BASE) & ok_base_te
    print(f"Baseline evidence gate: unseen-only, tau={TAU_BASE:.2f} rows={int(gate0.sum())}", flush=True)
    test_p1 = apply_poe(test_p0, loglike0, gate0, gamma=GAMMA_BASE)

    # Stage 3: wind-comp evidence (incremental; apply only if still uncertain)
    margin1 = top2_margin(test_p1)
    # two candidate gates / strengths
    candidates = [
        ("A", 0.25, 0.10),
        ("B", 0.20, 0.12),
    ]

    # Fit wind Gaussian params on train using ok_w_tr (and finite)
    ok_w_tr2 = ok_w_tr & np.isfinite(air_tr) & np.isfinite(drift_tr) & np.isfinite(head_tr) & np.isfinite(cross_tr)
    mu_air, sig_air = build_gauss_params(air_tr, y, MIN_SIGMA["airspeed_wind"], ok_w_tr2)
    mu_drift, sig_drift = build_gauss_params(drift_tr, y, MIN_SIGMA["drift_angle"], ok_w_tr2)
    mu_head, sig_head = build_gauss_params(head_tr, y, MIN_SIGMA["headwind"], ok_w_tr2)
    mu_cross, sig_cross = build_gauss_params(cross_tr, y, MIN_SIGMA["crosswind"], ok_w_tr2)

    lg_air = log_gaussian(np.where(np.isfinite(air_te), air_te, 0.0), mu_air, sig_air)
    lg_drift = log_gaussian(np.where(np.isfinite(drift_te), drift_te, 0.0), mu_drift, sig_drift)
    lg_head = log_gaussian(np.where(np.isfinite(head_te), head_te, 0.0), mu_head, sig_head)
    lg_cross = log_gaussian(np.where(np.isfinite(cross_te), cross_te, 0.0), mu_cross, sig_cross)

    loglike_wind = np.zeros((len(test_df), N_CLASSES), dtype=float)
    valid_w = ok_w_te & np.isfinite(air_te) & np.isfinite(drift_te) & np.isfinite(head_te) & np.isfinite(cross_te)
    loglike_wind[valid_w] = (
        W_AIR_WIND * lg_air[valid_w]
        + W_DRIFT * lg_drift[valid_w]
        + W_HEADWIND * lg_head[valid_w]
        + W_CROSSWIND * lg_cross[valid_w]
    )

    for tag, tau_w, gamma_w in candidates:
        gate_w = np.isin(test_months, UNSEEN_MONTHS) & (margin1 < tau_w) & valid_w
        print(
            f"Wind evidence gate {tag}: unseen-only, tau={tau_w:.2f}, gamma={gamma_w:.2f} rows={int(gate_w.sum())}",
            flush=True,
        )
        out = apply_poe(test_p1, loglike_wind, gate_w, gamma=gamma_w)
        name = f"e100_windcomp_{tag}_tau{tau_w:.2f}_gw{gamma_w:.2f}_priortau{TAU_PRIOR:.2f}"
        save_submission(out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

