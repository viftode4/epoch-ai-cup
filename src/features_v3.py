"""Feature extraction v3 — E175 validated architecture features.

Adds to v2 base:
  - 90 log-signature features (esig, depth 3, 3 windows, normalized channels)
  - 88 catch22 features (4 channels × 22, pure numpy/scipy implementation)
  - 10 new trajectory features (alt_curvature, alt_r2, rcs_ac_lag2-5, speed_cv, speed_ac1, speed_trend)
  - 8 physics score features (domain-specific compound indicators)
  - 1 flock size predictor (from n_birds_observed regression)

Total: ~264 trajectory + ~60 environment = ~324 candidate features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .data import CLASSES, parse_ewkb_4d, parse_trajectory_time
from .features_v2 import (
    SIZE_MAP,
    _haversine,
    _load_external_csv,
    build_features_v2,
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)


# ══════════════════════════════════════════════════════════════════════
# 1. NEW TRAJECTORY FEATURES (10 features)
# ══════════════════════════════════════════════════════════════════════

def extract_new_trajectory_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract 10 new trajectory features not in v2.

    - alt_curvature: 2nd derivative of altitude (BoP circling vs straight flight)
    - alt_r2: R² of linear fit to altitude (how linear is the altitude profile)
    - rcs_ac_lag2..5: RCS autocorrelation at lags 2-5 (wingbeat harmonics)
    - speed_cv: coefficient of variation of segment speeds
    - speed_ac1: autocorrelation lag-1 of speeds (speed regularity)
    - speed_trend: linear slope of speed over time (accelerating vs decelerating)
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    alts = np.array([p[2] for p in pts])
    rcs_dB = np.array([p[3] for p in pts])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    n = len(pts)

    # ── Altitude curvature (2nd derivative) ──
    if n >= 3:
        alt_dd = np.diff(alts, n=2)
        alt_curvature = float(np.mean(np.abs(alt_dd)))
    else:
        alt_curvature = 0.0

    # ── Altitude R² (linearity of altitude profile) ──
    if n >= 3:
        t_norm = np.linspace(0, 1, n)
        slope, intercept = np.polyfit(t_norm, alts, 1)
        predicted = slope * t_norm + intercept
        ss_res = np.sum((alts - predicted) ** 2)
        ss_tot = np.sum((alts - np.mean(alts)) ** 2)
        alt_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    else:
        alt_r2 = 1.0

    # ── RCS autocorrelation lags 2-5 ──
    rcs_centered = rcs_dB - np.mean(rcs_dB)
    rcs_var = float(np.var(rcs_dB))
    rcs_ac_lags = {}
    for lag in range(2, 6):
        if rcs_var > 1e-12 and n > lag:
            ac = float(np.mean(rcs_centered[:-lag] * rcs_centered[lag:]) / rcs_var)
            rcs_ac_lags[f"rcs_ac_lag{lag}"] = ac if np.isfinite(ac) else 0.0
        else:
            rcs_ac_lags[f"rcs_ac_lag{lag}"] = 0.0

    # ── Speed features ──
    if n > 1:
        dists = np.array([
            _haversine(lons[i], lats[i], lons[i + 1], lats[i + 1])
            for i in range(n - 1)
        ])
        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)
        speeds = dists / dt
        valid = raw_dt >= 0.5
        if valid.sum() >= 2:
            speeds_v = speeds[valid]
        else:
            speeds_v = speeds
    else:
        speeds_v = np.array([0.0])

    # Speed CV
    speed_mean = float(np.mean(speeds_v))
    speed_std = float(np.std(speeds_v))
    speed_cv = speed_std / max(speed_mean, 1e-6)

    # Speed autocorrelation lag-1
    if len(speeds_v) > 2:
        sp_c = speeds_v - np.mean(speeds_v)
        sp_var = float(np.var(speeds_v))
        if sp_var > 1e-12:
            speed_ac1 = float(np.mean(sp_c[:-1] * sp_c[1:]) / sp_var)
            speed_ac1 = speed_ac1 if np.isfinite(speed_ac1) else 0.0
        else:
            speed_ac1 = 0.0
    else:
        speed_ac1 = 0.0

    # Speed trend (linear slope over time)
    if len(speeds_v) >= 3:
        t_s = np.linspace(0, 1, len(speeds_v))
        speed_trend = float(np.polyfit(t_s, speeds_v, 1)[0])
        if not np.isfinite(speed_trend):
            speed_trend = 0.0
    else:
        speed_trend = 0.0

    return {
        "alt_curvature": alt_curvature,
        "alt_r2": alt_r2,
        **rcs_ac_lags,
        "speed_cv": speed_cv,
        "speed_ac1": speed_ac1,
        "speed_trend": speed_trend,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. LOG-SIGNATURES (90 features: 3 windows × 30 log-sig depth-3)
# ══════════════════════════════════════════════════════════════════════

def _normalize_path(path: np.ndarray) -> np.ndarray:
    """Normalize each channel to [0,1] for signature computation.

    Critical: raw channels have very different scales
    (lon ~6.8, lat ~53.4, alt ~0-500, RCS ~-40 to -10).
    Without normalization, signatures are dominated by the largest channel.
    """
    out = np.empty_like(path, dtype=np.float64)
    for j in range(path.shape[1]):
        col = path[:, j]
        mn, mx = col.min(), col.max()
        rng = mx - mn
        if rng > 1e-12:
            out[:, j] = (col - mn) / rng
        else:
            out[:, j] = 0.0
    return out


def _safe_logsig(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute log-signature with robust error handling.

    esig can fail on degenerate paths (constant channels, single point, etc.).
    Returns zeros on failure.
    """
    import esig

    n_features = 30  # depth-3, 4D path = 30 log-sig features
    if path.shape[0] < 2:
        return np.zeros(n_features)

    # esig requires at least 2 distinct points and no NaN/inf
    if not np.all(np.isfinite(path)):
        path = np.nan_to_num(path, nan=0.0, posinf=0.0, neginf=0.0)

    # Check if path is degenerate (all points identical)
    if np.allclose(path[0], path[-1]) and path.shape[0] <= 2:
        return np.zeros(n_features)

    try:
        ls = esig.stream2logsig(path, depth)
        ls = np.where(np.isfinite(ls), ls, 0.0)
        return ls
    except Exception:
        return np.zeros(n_features)


def extract_log_signature_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract 90 log-signature features from a 4D trajectory.

    Uses esig.stream2logsig at depth 3 on normalized (lon, lat, alt, RCS) path.
    Three temporal windows: full track, first half, second half.
    30 features per window × 3 windows = 90 features.
    """
    pts = parse_ewkb_4d(hex_str)
    n = len(pts)

    path = np.array(pts, dtype=np.float64)  # (n, 4): lon, lat, alt, RCS
    path_norm = _normalize_path(path)

    features = {}

    # Three windows: full, first half, second half
    mid = max(n // 2, 2)
    windows = {
        "full": path_norm,
        "h1": path_norm[:mid],
        "h2": path_norm[max(mid - 1, 0):],  # overlap by 1 point for continuity
    }

    for wname, wpath in windows.items():
        if wpath.shape[0] < 2:
            wpath = np.vstack([wpath, wpath[-1:] + 1e-8])  # tiny perturbation
        ls = _safe_logsig(wpath, 3)
        for i, val in enumerate(ls):
            features[f"lsig_{wname}_{i:02d}"] = float(val)

    return features


# ══════════════════════════════════════════════════════════════════════
# 3. CATCH22 FEATURES (22 per channel × 4 channels = 88 features)
#    Pure numpy/scipy implementation
# ══════════════════════════════════════════════════════════════════════

def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for lags 1..max_lag."""
    n = len(x)
    xc = x - x.mean()
    var = np.dot(xc, xc)
    if var < 1e-12:
        return np.zeros(max_lag)
    ac = np.correlate(xc, xc, mode="full")
    ac = ac[n - 1 :]  # positive lags only
    ac = ac / var
    return ac[1 : max_lag + 1]


def _catch22_single(x: np.ndarray) -> dict:
    """Compute 22 catch22-equivalent features for a single time series.

    Pure numpy/scipy implementation of the canonical catch22 feature set.
    """
    n = len(x)
    if n < 3:
        return {f"c22_{i:02d}": 0.0 for i in range(22)}

    xz = (x - x.mean()) / max(x.std(), 1e-12)  # z-scored
    feats = {}

    # 1. DN_HistogramMode_5: mode of 5-bin histogram
    counts, edges = np.histogram(x, bins=5)
    feats["c22_00"] = float(0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1]))

    # 2. DN_HistogramMode_10: mode of 10-bin histogram
    counts, edges = np.histogram(x, bins=10)
    feats["c22_01"] = float(0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1]))

    # 3. CO_f1ecac: first 1/e crossing of autocorrelation
    max_lag = min(n - 1, 40)
    if max_lag > 0:
        ac = _autocorr(x, max_lag)
        thresh = 1.0 / np.e
        crossings = np.where(ac < thresh)[0]
        feats["c22_02"] = float(crossings[0] + 1) if len(crossings) > 0 else float(max_lag)
    else:
        feats["c22_02"] = 0.0

    # 4. CO_FirstMin_ac: first minimum of autocorrelation
    if max_lag > 1:
        ac = _autocorr(x, max_lag)
        diffs = np.diff(ac)
        mins = np.where(diffs > 0)[0]  # where ac starts increasing = just past a minimum
        feats["c22_03"] = float(mins[0] + 1) if len(mins) > 0 else float(max_lag)
    else:
        feats["c22_03"] = 0.0

    # 5. CO_HistogramAMI_even_2_5: automutual information, m=2, tau=5
    tau = min(5, n // 2)
    if tau > 0 and n > tau:
        x1 = x[:-tau] if tau > 0 else x
        x2 = x[tau:]
        nbins = max(int(np.sqrt(len(x1))), 2)
        try:
            h_xy = np.histogram2d(x1, x2, bins=nbins)[0]
            h_xy = h_xy / h_xy.sum()
            h_x = h_xy.sum(axis=1)
            h_y = h_xy.sum(axis=0)
            # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
            mi = 0.0
            for i in range(nbins):
                for j in range(nbins):
                    if h_xy[i, j] > 1e-12 and h_x[i] > 1e-12 and h_y[j] > 1e-12:
                        mi += h_xy[i, j] * np.log(h_xy[i, j] / (h_x[i] * h_y[j]))
            feats["c22_04"] = float(mi)
        except Exception:
            feats["c22_04"] = 0.0
    else:
        feats["c22_04"] = 0.0

    # 6. CO_trev_1_num: time-reversibility statistic lag 1
    if n > 1:
        d = np.diff(xz)
        feats["c22_05"] = float(np.mean(d ** 3)) if len(d) > 0 else 0.0
    else:
        feats["c22_05"] = 0.0

    # 7. MD_hrv_classic_pnn40: fraction of successive differences exceeding 40% of std
    if n > 1:
        d = np.abs(np.diff(x))
        threshold = 0.4 * x.std()
        feats["c22_06"] = float(np.mean(d > threshold)) if threshold > 1e-12 else 0.0
    else:
        feats["c22_06"] = 0.0

    # 8. SB_BinaryStats_mean_longstretch1: longest stretch above mean
    above = (x > x.mean()).astype(int)
    if len(above) > 0:
        diffs_ab = np.diff(np.concatenate([[0], above, [0]]))
        starts = np.where(diffs_ab == 1)[0]
        ends = np.where(diffs_ab == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            feats["c22_07"] = float(np.max(ends[:len(starts)] - starts[:len(ends)]))
        else:
            feats["c22_07"] = 0.0
    else:
        feats["c22_07"] = 0.0

    # 9. SB_BinaryStats_diff_longstretch0: longest stretch of non-increasing values
    if n > 1:
        decreasing = (np.diff(x) <= 0).astype(int)
        diffs_d = np.diff(np.concatenate([[0], decreasing, [0]]))
        starts = np.where(diffs_d == 1)[0]
        ends = np.where(diffs_d == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            feats["c22_08"] = float(np.max(ends[:len(starts)] - starts[:len(ends)]))
        else:
            feats["c22_08"] = 0.0
    else:
        feats["c22_08"] = 0.0

    # 10. SB_MotifThree_quantile_hh: 3-symbol motif entropy
    if n >= 3:
        q33, q67 = np.percentile(x, [33.33, 66.67])
        symbols = np.digitize(x, [q33, q67])
        # Count 3-symbol motifs
        motifs = {}
        for i in range(len(symbols) - 2):
            key = (symbols[i], symbols[i + 1], symbols[i + 2])
            motifs[key] = motifs.get(key, 0) + 1
        total = sum(motifs.values())
        if total > 0:
            probs = np.array(list(motifs.values())) / total
            feats["c22_09"] = float(-np.sum(probs * np.log(probs + 1e-12)))
        else:
            feats["c22_09"] = 0.0
    else:
        feats["c22_09"] = 0.0

    # 11-12. SC_FluctAnal (DFA-like): detrended fluctuation analysis
    # Simplified: compute RMS of detrended segments at 2 scales
    for idx, scale in enumerate([5, 10]):
        if n >= scale * 2:
            n_segs = n // scale
            rmss = []
            for s in range(n_segs):
                seg = xz[s * scale : (s + 1) * scale]
                t_seg = np.arange(scale)
                p = np.polyfit(t_seg, seg, 1)
                detrended = seg - np.polyval(p, t_seg)
                rmss.append(np.sqrt(np.mean(detrended ** 2)))
            feats[f"c22_{10 + idx:02d}"] = float(np.mean(rmss)) if rmss else 0.0
        else:
            feats[f"c22_{10 + idx:02d}"] = 0.0

    # 13. SP_Summaries_welch_rect_area_5_1: spectral area in band
    if n >= 8:
        from scipy.signal import welch
        freqs, psd = welch(xz, fs=1.0, nperseg=min(n, 16), noverlap=0)
        total_power = np.sum(psd)
        if total_power > 1e-12 and len(freqs) >= 5:
            band = psd[:5]
            feats["c22_12"] = float(np.sum(band) / total_power)
        else:
            feats["c22_12"] = 0.0
    else:
        feats["c22_12"] = 0.0

    # 14. SP_Summaries_welch_rect_centroid: spectral centroid
    if n >= 8:
        from scipy.signal import welch
        freqs, psd = welch(xz, fs=1.0, nperseg=min(n, 16), noverlap=0)
        total_power = np.sum(psd)
        if total_power > 1e-12:
            feats["c22_13"] = float(np.sum(freqs * psd) / total_power)
        else:
            feats["c22_13"] = 0.0
    else:
        feats["c22_13"] = 0.0

    # 15. FC_LocalSimple_mean1_tauresrat: forecast error ratio
    if n >= 3:
        pred = x[:-1]  # predict x[t+1] = x[t]
        err = x[1:] - pred
        feats["c22_14"] = float(np.std(err) / max(np.std(x), 1e-12))
    else:
        feats["c22_14"] = 0.0

    # 16. FC_LocalSimple_mean3_stderr: 3-point moving avg forecast error
    if n >= 5:
        pred = np.convolve(x, np.ones(3) / 3, mode="valid")
        actual = x[3:]
        if len(pred) >= len(actual) and len(actual) > 0:
            err = actual - pred[:len(actual)]
            feats["c22_15"] = float(np.std(err))
        else:
            feats["c22_15"] = 0.0
    else:
        feats["c22_15"] = 0.0

    # 17. IN_AutoMutualInfoStats_40_gaussian_fmmi: first min of AMI
    # Simplified: first minimum of lagged correlation
    if n > 4:
        max_l = min(n // 2, 20)
        ac = _autocorr(x, max_l)
        if len(ac) > 1:
            dac = np.diff(ac)
            mins = np.where(dac > 0)[0]
            feats["c22_16"] = float(mins[0] + 1) if len(mins) > 0 else float(max_l)
        else:
            feats["c22_16"] = 0.0
    else:
        feats["c22_16"] = 0.0

    # 18-19. DN_OutlierInclude: outlier timing statistics
    threshold = 1.0  # 1 std deviation
    for idx, sign in enumerate([1, -1]):
        if sign == 1:
            outlier_mask = xz > threshold
        else:
            outlier_mask = xz < -threshold
        if outlier_mask.sum() > 0:
            outlier_times = np.where(outlier_mask)[0] / max(n - 1, 1)
            feats[f"c22_{17 + idx:02d}"] = float(np.median(outlier_times) - 0.5)
        else:
            feats[f"c22_{17 + idx:02d}"] = 0.0

    # 20. PD_PeriodicityWang_th0_01: dominant periodicity
    if n >= 8:
        ac = _autocorr(x, min(n - 1, 40))
        if len(ac) > 2:
            # Find first peak in autocorrelation
            dac = np.diff(ac)
            peaks = np.where((dac[:-1] > 0) & (dac[1:] <= 0))[0]
            if len(peaks) > 0 and ac[peaks[0] + 1] > 0.01:
                feats["c22_19"] = float(peaks[0] + 2)  # period
            else:
                feats["c22_19"] = 0.0
        else:
            feats["c22_19"] = 0.0
    else:
        feats["c22_19"] = 0.0

    # 21. CO_Embed2_Dist_tau_d_expfit_meandiff: embedding distance stat
    tau_e = max(1, int(feats.get("c22_02", 1)))
    tau_e = min(tau_e, n // 3, 10)
    if n > 2 * tau_e and tau_e > 0:
        x1 = xz[:-tau_e]
        x2 = xz[tau_e:]
        dists = np.sqrt((x1 - x2) ** 2)
        feats["c22_20"] = float(np.mean(dists))
    else:
        feats["c22_20"] = 0.0

    # 22. SB_TransitionMatrix_3ac_sumdiagcov: transition matrix diagonal
    if n >= 4:
        q33, q67 = np.percentile(x, [33.33, 66.67])
        symbols = np.digitize(x, [q33, q67])
        trans = np.zeros((3, 3))
        for i in range(len(symbols) - 1):
            trans[symbols[i], symbols[i + 1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        trans = trans / np.maximum(row_sums, 1)
        feats["c22_21"] = float(np.sum(np.diag(trans)))
    else:
        feats["c22_21"] = 0.0

    return feats


def extract_catch22_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract catch22 features for all 4 trajectory channels.

    4 channels (lon, lat, alt, RCS) × 22 features = 88 features.
    """
    pts = parse_ewkb_4d(hex_str)
    channels = {
        "alt": np.array([p[2] for p in pts]),
        "rcs": np.array([p[3] for p in pts]),
        "lon": np.array([p[0] for p in pts]),
        "lat": np.array([p[1] for p in pts]),
    }

    features = {}
    for ch_name, ch_data in channels.items():
        ch_feats = _catch22_single(ch_data)
        for feat_name, val in ch_feats.items():
            features[f"{ch_name}_{feat_name}"] = val

    return features


# ══════════════════════════════════════════════════════════════════════
# 4. PHYSICS SCORE FEATURES (8 compound domain indicators)
# ══════════════════════════════════════════════════════════════════════

def extract_physics_scores(df_feat: pd.DataFrame, df_raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Compute 8 physics-informed compound score features.

    Each score combines multiple features to indicate a specific ecological behavior.
    Designed to be month-invariant (based on physics, not timing).
    """
    n = len(df_feat)

    def _safe_col(name: str) -> np.ndarray:
        if name in df_feat.columns:
            return df_feat[name].values.astype(float)
        return np.zeros(n)

    airspeed = _safe_col("airspeed")
    alt_mean = _safe_col("alt_mean")
    alt_std = _safe_col("alt_std")
    straightness = _safe_col("straightness")
    speed_cv = _safe_col("speed_cv")
    rcs_mean_dB = _safe_col("rcs_mean_dB")
    rcs_scintillation = _safe_col("rcs_scintillation")
    heading_std = _safe_col("heading_std")
    curvature_mean = _safe_col("curvature_mean")
    rcs_ac1 = _safe_col("rcs_autocorr_lag1")
    radar_bird_size = _safe_col("radar_bird_size")

    # External features (may or may not be present)
    wind_speed = _safe_col("wx_wind_speed")
    wind_at_bird = _safe_col("wind_at_bird_alt")
    tidal_phase = _safe_col("tidal_phase")
    blh = _safe_col("boundary_layer_height")
    cape = _safe_col("cape_jkg")
    dist_water = _safe_col("dist_to_water_m")

    # 1. Cormorant wind score: |ground_speed - (0.70*wind + 14.4)| / SD
    # Low residual = likely Cormorant (tight speed-wind relationship)
    wind_eff = np.where(wind_at_bird > 0.1, wind_at_bird, wind_speed)
    expected = 0.70 * wind_eff + 14.4
    residual = np.abs(airspeed - expected)
    df_feat["phys_cormorant_wind"] = np.exp(-residual / 3.0)  # Gaussian-like, peak at 0

    # 2. Wader tidal score: activity correlated with tidal phase
    # Waders fly 3h before high tide (phase ~0.75 in 0-1 cycle)
    if np.any(tidal_phase > 0):
        tidal_signal = np.cos(2 * np.pi * (tidal_phase - 0.75))  # peak at phase 0.75
        df_feat["phys_wader_tidal"] = tidal_signal * (airspeed > 14.0).astype(float)
    else:
        df_feat["phys_wader_tidal"] = np.zeros(n)

    # 3. BoP soaring score: slow + high curvature + altitude gain + low straightness
    soaring = (
        np.exp(-(airspeed - 10.0) ** 2 / 20.0)  # speed near 10 m/s
        * (1 - straightness)  # circling
        * np.clip(curvature_mean / 0.01, 0, 1)  # curved path
    )
    df_feat["phys_bop_soaring"] = soaring

    # 4. Kestrel hover score: very slow + stable altitude + small bird
    hover = (
        (airspeed < 5.0).astype(float)
        * (alt_std < 10.0).astype(float)
        * (radar_bird_size <= 1).astype(float)  # Small bird
    )
    df_feat["phys_kestrel_hover"] = hover

    # 5. Flock interference score: RCS scintillation + size category
    # High SI = multiple birds in beam (flock). Geese, Waders travel in flocks.
    df_feat["phys_flock_signal"] = (
        np.log1p(rcs_scintillation)
        * np.clip(radar_bird_size / 4.0, 0, 1)
    )

    # 6. Clutter score: high RCS + low speed + erratic heading
    clutter = (
        (rcs_mean_dB > -18.0).astype(float) * 0.5
        + (airspeed < 5.0).astype(float) * 0.3
        + (heading_std > 1.5).astype(float) * 0.2
    )
    df_feat["phys_clutter"] = clutter

    # 7. Insect drift score: speed ≈ wind speed AND heading ≈ wind direction
    speed_ratio = airspeed / np.maximum(wind_eff, 0.5)
    df_feat["phys_insect_drift"] = np.exp(-((speed_ratio - 1.0) ** 2) / 0.5)

    # 8. Migration score: fast + straight + high altitude
    migration = (
        np.clip(airspeed / 20.0, 0, 1)
        * straightness
        * np.clip(alt_mean / 200.0, 0, 1)
    )
    df_feat["phys_migration"] = migration

    return df_feat


# ══════════════════════════════════════════════════════════════════════
# 5. FLOCK SIZE PREDICTOR
# ══════════════════════════════════════════════════════════════════════

def train_flock_predictor(train_df: pd.DataFrame, train_feats: pd.DataFrame) -> object:
    """Train a simple model to predict n_birds_observed from features.

    n_birds_observed is train-only (privileged). We train a regressor on
    features that exist in both train and test, then use predictions as
    a feature.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    target = train_df["n_birds_observed"].values.astype(float)
    valid = np.isfinite(target) & (target > 0)

    # Use features available in both train and test
    pred_cols = [
        c for c in ["rcs_mean_dB", "rcs_scintillation", "radar_bird_size",
                     "rcs_std_dB", "rcs_kurtosis_linear", "rcs_deep_fade_frac",
                     "rcs_linear_cv", "alt_std"]
        if c in train_feats.columns
    ]
    if len(pred_cols) < 3 or valid.sum() < 50:
        return None

    X = train_feats.loc[valid, pred_cols].values.astype(np.float32)
    y = np.log1p(target[valid])  # log transform for better regression

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X, y)
    model._flock_pred_cols = pred_cols
    return model


def predict_flock_size(model, df_feat: pd.DataFrame) -> np.ndarray:
    """Predict flock size using the trained model."""
    if model is None:
        return np.ones(len(df_feat))
    cols = model._flock_pred_cols
    X = df_feat[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.expm1(model.predict(X))  # inverse log1p


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════

def build_features_v3(
    df: pd.DataFrame,
    cache_path: Path | None = None,
    flock_model: object = None,
) -> pd.DataFrame:
    """Build complete v3 feature set (~324 features).

    Steps:
      1. Build v2 base features (76 features)
      2. Load all external features from E174 (~55 features)
      3. Add 10 new trajectory features
      4. Add 90 log-signature features
      5. Add 88 catch22 features
      6. Add 8 physics score features
      7. Add flock size prediction (1 feature)
      8. Handle inf/nan
    """
    if cache_path is not None and cache_path.exists():
        h = _features_v3_hash()
        meta = cache_path.with_suffix(".v3hash")
        if meta.exists() and meta.read_text().strip() == h:
            print(f"  Loading cached v3 features from {cache_path.name}", flush=True)
            return pd.read_pickle(cache_path)

    split = "train" if "bird_group" in df.columns else "test"
    n = len(df)

    # Step 1: v2 base features
    print("  [v3 1/7] Building v2 base features...", flush=True)
    # Only use v2 cache for full dataset (subset testing must recompute)
    from .data import load_train, load_test
    full_n = len(load_train()) if split == "train" else len(load_test())
    v2_cache_path = ROOT / "data" / f"_cached_{split}_features_v2.pkl"
    use_v2_cache = v2_cache_path if n == full_n else None
    df_feat = build_features_v2(df, cache_path=use_v2_cache)
    v2_count = df_feat.shape[1]
    print(f"    v2 base: {v2_count} features", flush=True)

    # Step 2: All external features (from E174)
    print("  [v3 2/7] Loading all external features...", flush=True)
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from e174_all_data import load_all_external_features
    df_feat = load_all_external_features(df_feat, df, split)
    ext_count = df_feat.shape[1] - v2_count
    print(f"    External: +{ext_count} features (total: {df_feat.shape[1]})", flush=True)

    # Step 3: New trajectory features (10)
    print("  [v3 3/7] Extracting new trajectory features...", flush=True)
    new_traj_rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    Progress: {idx}/{n}", flush=True)
        new_traj_rows.append(extract_new_trajectory_features(r.trajectory, r.trajectory_time))
    new_traj_df = pd.DataFrame(new_traj_rows)
    for col in new_traj_df.columns:
        df_feat[col] = new_traj_df[col].values
    print(f"    New trajectory: +{new_traj_df.shape[1]} features", flush=True)

    # Step 4: Log-signature features (90)
    print("  [v3 4/7] Extracting log-signature features...", flush=True)
    lsig_rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    Progress: {idx}/{n}", flush=True)
        lsig_rows.append(extract_log_signature_features(r.trajectory, r.trajectory_time))
    lsig_df = pd.DataFrame(lsig_rows)
    for col in lsig_df.columns:
        df_feat[col] = lsig_df[col].values
    print(f"    Log-signatures: +{lsig_df.shape[1]} features", flush=True)

    # Step 5: catch22 features (88)
    print("  [v3 5/7] Extracting catch22 features...", flush=True)
    c22_rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    Progress: {idx}/{n}", flush=True)
        c22_rows.append(extract_catch22_features(r.trajectory, r.trajectory_time))
    c22_df = pd.DataFrame(c22_rows)
    for col in c22_df.columns:
        df_feat[col] = c22_df[col].values
    print(f"    catch22: +{c22_df.shape[1]} features", flush=True)

    # Step 6: Physics scores (8)
    print("  [v3 6/7] Computing physics scores...", flush=True)
    df_feat = extract_physics_scores(df_feat, df, split)
    phys_cols = [c for c in df_feat.columns if c.startswith("phys_")]
    print(f"    Physics scores: +{len(phys_cols)} features", flush=True)

    # Step 7: Flock size prediction (1)
    if flock_model is not None:
        print("  [v3 7/7] Predicting flock size...", flush=True)
        df_feat["predicted_flock_size"] = predict_flock_size(flock_model, df_feat)
    else:
        print("  [v3 7/7] No flock model (skipped)", flush=True)

    # Handle inf/nan
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    # NOTE: Do NOT drop constant features here — do it in the experiment
    # script after both train and test are built, to ensure column alignment.

    print(f"    TOTAL: {df_feat.shape[1]} features", flush=True)

    if cache_path is not None:
        df_feat.to_pickle(cache_path)
        cache_path.with_suffix(".v3hash").write_text(_features_v3_hash())
        print(f"    Cached to {cache_path.name}", flush=True)

    return df_feat


def _features_v3_hash() -> str:
    """Hash of this file + features_v2.py for cache invalidation."""
    h = hashlib.sha256()
    for fname in ["features_v3.py", "features_v2.py"]:
        p = ROOT / "src" / fname
        if p.exists():
            h.update(p.read_text(encoding="utf-8").encode())
    return h.hexdigest()[:12]
