"""E102: Trajectory-shape evidence PoE (unseen-month, low-margin gated).

Goal
----
Exploit the raw `trajectory` column more aggressively by injecting a *trajectory-shape*
generative expert on top of the strong E50 base predictions, while preserving robustness to
calendar shift via:
  - applying GBIF month-ratio priors only on unseen months (E67-style)
  - applying trajectory-shape evidence only on unseen months AND only when base is uncertain

Evidence
--------
We reuse existing trajectory feature extractors in `src/features.py` (no copy-paste parsing):
  - `weakclass` (straightness, turning stats, soaring index, RCS stability)
  - `flight_mode` (flap/glide segmentation, curvature, effective_speed_ratio)
  - `enhanced_bio_shape` (turn-direction consistency, loop fraction, burstiness)

External data integration (minimal but real)
-------------------------------------------
We use publicly downloaded Movebank/Zenodo GPS tracks to compute a *soft prior* for the
turning-angle tail (p90) for:
  - Birds of Prey: H_GRONINGEN (marsh harrier; Groningen)
  - Waders: O_BALGZAND (oystercatcher) + CURLEW_VLAANDEREN (curlew)

We then shrink the train-estimated class mean of `turn_angle_p90` toward these external priors.
This is intentionally conservative: it uses external data only to regularize one invariant cue.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.features import build_features, haversine  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

UNSEEN_MONTHS = (2, 5, 12)

# Priors stage (best known settings)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Evidence stage
TAU_EVID = 0.25
GAMMAS = (0.06, 0.10)  # produce two candidates

# External shrinkage (conservative)
LAMBDA_EXT = 0.25

# Numerical safety
EPS = 1e-12
MIN_SIGMA = 0.05


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


def _infer_cols(df: pd.DataFrame) -> tuple[str, str, str, str]:
    # DarwinCore-ish exports (as used in INBO movepub) typically include these.
    tcol = next((c for c in df.columns if c.lower() in ("eventdate", "timestamp", "time")), None)
    latcol = next((c for c in df.columns if "lat" in c.lower()), None)
    loncol = next((c for c in df.columns if "lon" in c.lower() or "long" in c.lower()), None)
    idcol = next((c for c in df.columns if "individual" in c.lower() and "identifier" in c.lower()), None)
    if tcol is None or latcol is None or loncol is None:
        raise ValueError(f"Could not infer time/lat/lon columns. cols={list(df.columns)[:30]}")
    if idcol is None:
        idcol = "__single__"
        df[idcol] = 0
    return tcol, latcol, loncol, idcol


def external_turn_p90_rad(paths: list[Path]) -> float:
    """Compute p90 turning angle (radians) from GPS tracks, filtering to in-flight steps."""
    all_angles_deg: list[np.ndarray] = []
    for p in paths:
        df = pd.read_csv(p, compression="gzip")
        tcol, latcol, loncol, idcol = _infer_cols(df)

        df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
        df = df.dropna(subset=[tcol, latcol, loncol])
        df = df.sort_values([idcol, tcol])

        lat = df[latcol].to_numpy(float)
        lon = df[loncol].to_numpy(float)
        t = df[tcol].astype("int64").to_numpy() / 1e9
        ids = df[idcol].to_numpy()

        # step speeds to filter out stationary/foraging noise
        same = ids[1:] == ids[:-1]
        dt = (t[1:] - t[:-1])[same]
        d = haversine(lon[:-1][same], lat[:-1][same], lon[1:][same], lat[1:][same])
        ok = (dt > 0) & np.isfinite(dt) & np.isfinite(d)
        dt = dt[ok]
        d = d[ok]
        v = d / dt
        in_flight = (v > 5.0) & (v < 45.0)  # m/s

        # turning angles (vector angle between consecutive displacements), per individual
        # Use a crude local projection for angle stability.
        latr = np.radians(lat)
        x = np.radians(lon) * 6371000.0 * np.cos(latr)
        y = np.radians(lat) * 6371000.0

        # We'll compute angles for each individual, and then apply the in_flight mask
        # approximately by requiring both adjacent step speeds to be in-flight.
        for uid, g in df.groupby(idcol, sort=False):
            if len(g) < 4:
                continue
            idx = g.index.to_numpy()
            pos = df.index.get_indexer(idx)
            xx = x[pos]
            yy = y[pos]
            tt = t[pos]

            dx1 = xx[1:-2] - xx[:-3]
            dy1 = yy[1:-2] - yy[:-3]
            dx2 = xx[2:-1] - xx[1:-2]
            dy2 = yy[2:-1] - yy[1:-2]
            dt1 = tt[1:-2] - tt[:-3]
            dt2 = tt[2:-1] - tt[1:-2]
            ok_t = (dt1 > 0) & (dt2 > 0)
            dx1, dy1, dx2, dy2 = dx1[ok_t], dy1[ok_t], dx2[ok_t], dy2[ok_t]

            n1 = np.hypot(dx1, dy1)
            n2 = np.hypot(dx2, dy2)
            ok_n = (n1 > 1e-3) & (n2 > 1e-3)
            dx1, dy1, dx2, dy2 = dx1[ok_n], dy1[ok_n], dx2[ok_n], dy2[ok_n]
            n1 = np.hypot(dx1, dy1)
            n2 = np.hypot(dx2, dy2)
            if len(n1) == 0:
                continue

            cosang = np.clip((dx1 * dx2 + dy1 * dy2) / (n1 * n2), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            ang = ang[np.isfinite(ang)]
            if len(ang) >= 20:
                all_angles_deg.append(ang)

    if not all_angles_deg:
        return float(np.radians(120.0))
    ang = np.concatenate(all_angles_deg)
    ang = ang[np.isfinite(ang)]
    if len(ang) == 0:
        return float(np.radians(120.0))
    return float(np.radians(np.quantile(ang, 0.90)))


def fit_gaussian_nb(
    X: np.ndarray, y: np.ndarray, ext_turn_p90: dict[int, float], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit diagonal Gaussian class conditionals with mild shrinkage."""
    n, d = X.shape
    mu = np.zeros((N_CLASSES, d), dtype=float)
    sig = np.zeros((N_CLASSES, d), dtype=float)

    gm = np.nanmean(X, axis=0)
    gs = np.nanstd(X, axis=0)
    gs = np.where(np.isfinite(gs) & (gs > MIN_SIGMA), gs, MIN_SIGMA)

    for c in range(N_CLASSES):
        xc = X[y == c]
        if xc.shape[0] >= 5:
            m = np.nanmean(xc, axis=0)
            s = np.nanstd(xc, axis=0)
        else:
            m = gm
            s = gs
        mu[c] = np.where(np.isfinite(m), m, gm)
        s = np.where(np.isfinite(s) & (s > MIN_SIGMA), s, MIN_SIGMA)
        sig[c] = s

    # External shrinkage for turning-angle p90 (if present)
    if "turn_angle_p90" in feature_names:
        j = feature_names.index("turn_angle_p90")
        for c, prior in ext_turn_p90.items():
            mu[c, j] = (1.0 - LAMBDA_EXT) * mu[c, j] + LAMBDA_EXT * prior
            sig[c, j] = max(sig[c, j], 0.10)  # don't over-trust the prior

    return mu, sig


def loglike_diag_gauss(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    # X: (n,d), mu/sig: (K,d) -> ll: (n,K)
    X = np.where(np.isfinite(X), X, 0.0)
    sig = np.clip(sig, MIN_SIGMA, None)
    z = (X[:, None, :] - mu[None, :, :]) / sig[None, :, :]
    ll = -0.5 * np.sum(z * z, axis=2) - np.sum(np.log(sig[None, :, :]), axis=2)
    return ll


def apply_poe(base: np.ndarray, ll: np.ndarray, gamma: float, gate: np.ndarray) -> np.ndarray:
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)
    l = ll[gate]
    l = l - l.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * l, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), EPS, None)
    return renorm_rows(out)


def main() -> None:
    print("=" * 70, flush=True)
    print("E102 TRAJECTORY-SHAPE EVIDENCE PoE (UNSEEN, GATED)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
    print(f"\nBase preds: test_e50.npy shape={test_base.shape}", flush=True)

    # Priors stage
    test_p0, changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    print(f"Applied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_EVID)
    print(f"Evidence gate: unseen & margin<{TAU_EVID:.2f} rows={int(gate.sum())}", flush=True)

    # Build trajectory-shape features (train/test)
    # NOTE: `weakclass` triggers `add_weakclass_tabular()` inside `build_features()`,
    # which expects core + tabular columns (e.g. `alt_mean`, `airspeed`).
    # We include them, but we do NOT use temporal tabular columns in the evidence update.
    feature_sets = ["core", "tabular", "weakclass", "flight_mode", "enhanced_bio_shape"]
    print("\nBuilding trajectory-shape features on train...", flush=True)
    Xtr_df = build_features(train_df, feature_sets=feature_sets)
    print("\nBuilding trajectory-shape features on test...", flush=True)
    Xte_df = build_features(test_df, feature_sets=feature_sets)

    evidence_cols = [
        # turning / loopiness
        "straightness",
        "turn_angle_var",
        "turn_angle_p90",
        "turn_dir_consistency",
        "max_sustained_turn_frac",
        "turn_reversal_rate",
        "path_loop_fraction",
        # flap/glide / curvature
        "flap_fraction",
        "glide_fraction",
        "curvature_mean",
        "curvature_max",
        "effective_speed_ratio",
        "alt_osc_freq",
        "alt_osc_amplitude",
        # RCS texture (invariant-ish)
        "rcs_cv",
        "rcs_autocorr_lag1",
        "rcs_stability",
        "rcs_dominant_ac_lag",
        "rcs_flap_regularity",
        "rcs_glide_flap_var_ratio",
        "rcs_burst_fraction",
        # flight style
        "soaring_index",
        "alt_gain_rate",
        "slow_flight_frac",
        "speed_cv",
    ]
    missing = [c for c in evidence_cols if c not in Xtr_df.columns]
    if missing:
        raise KeyError(f"Missing evidence cols: {missing}")

    Xtr = Xtr_df[evidence_cols].to_numpy(float)
    Xte = Xte_df[evidence_cols].to_numpy(float)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    # External priors for turn-angle tail (p90)
    ext = {}
    bop = CLASSES.index("Birds of Prey")
    wad = CLASSES.index("Waders")

    h_gron = ROOT / "data" / "other_datasets" / "zenodo_10053658_h_groningen_marsh_harrier" / "H_GRONINGEN-gps-2018.csv.gz"
    o_balg = ROOT / "data" / "other_datasets" / "zenodo_10053932_o_balgzand_oystercatcher" / "O_BALGZAND-gps-2012.csv.gz"
    curlew = ROOT / "data" / "other_datasets" / "zenodo_15696532_curlew_vlaanderen" / "CURLEW_VLAANDEREN-gps-2022.csv.gz"

    try:
        if h_gron.exists():
            ext[bop] = external_turn_p90_rad([h_gron])
        if o_balg.exists() and curlew.exists():
            ext[wad] = external_turn_p90_rad([o_balg, curlew])
        print(
            f"External turn p90 priors (rad): BoP={ext.get(bop, np.nan):.3f}  Waders={ext.get(wad, np.nan):.3f}",
            flush=True,
        )
    except Exception as e:
        print(f"External priors: failed ({e}); continuing without.", flush=True)
        ext = {}

    mu, sig = fit_gaussian_nb(Xtr, y, ext_turn_p90=ext, feature_names=evidence_cols)
    ll_te = loglike_diag_gauss(Xte, mu, sig)

    for gamma in GAMMAS:
        out = apply_poe(test_p0, ll_te, gamma=gamma, gate=gate)
        name = f"e102_trajshape_tau{TAU_EVID:.2f}_g{gamma:.2f}_priortau{TAU_PRIOR:.2f}"
        save_submission(out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

