"""Clean feature extraction — single pipeline, no filtering, correct math.

Rules:
- ALL raw data points are used. No filtering, no interpolation, no detrending.
- Bearings use proper lat/lon scaling (67000/111000 m/deg at ~53N).
- RCS statistics that need linear scale use 10^(dB/10) conversion.
- SIZE_MAP is 1-indexed: {Small:1, Medium:2, Large:3, Flock:4}.
- One function per feature group. One build_features() entry point.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from .data import CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

SIZE_MAP = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}

# Temporal features that leak month identity — never use as model features
ALL_TEMPORAL = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring", "hour_bin_3h",
    "is_oct_nov", "migration_alt", "migration_speed", "is_night", "night_high_alt",
]


def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _parse_track(hex_str, traj_time_str):
    """Parse raw track into arrays. No modifications to raw values."""
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs_dB = np.array([p[3] for p in pts])
    return lons, lats, alts, rcs_dB, times, len(pts)


def _segments(lons, lats, alts, rcs_dB, times, n):
    """Compute per-segment quantities from ALL raw points. No filtering."""
    if n < 2:
        return {
            'dists': np.array([0.0]), 'dt': np.array([1.0]),
            'speeds': np.array([0.0]), 'alt_diffs': np.array([0.0]),
            'bearings': np.array([0.0]), 'bearing_changes': np.array([0.0]),
            'duration': 0.001, 'total_dist': 0.0, 'straight_dist': 0.0,
        }

    dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1]) for i in range(n-1)])
    dt = np.maximum(np.diff(times), 0.001)
    speeds = dists / dt
    alt_diffs = np.diff(alts)
    duration = times[-1] - times[0] if n > 1 else 0.001
    total_dist = dists.sum()
    straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1])

    # Bearings: CORRECT scaling (meters, not degrees)
    dx = np.diff(lons) * 67000.0
    dy = np.diff(lats) * 111000.0
    bearings = np.arctan2(dy, dx)

    if n > 2:
        bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
    else:
        bearing_changes = np.array([0.0])

    return {
        'dists': dists, 'dt': dt, 'speeds': speeds,
        'alt_diffs': alt_diffs, 'bearings': bearings,
        'bearing_changes': bearing_changes,
        'duration': duration, 'total_dist': total_dist,
        'straight_dist': straight_dist,
    }


# ─────────────────────────────────────────────────────────────
# Feature groups — each returns a dict of feature_name: value
# ─────────────────────────────────────────────────────────────

def _core_features(lons, lats, alts, rcs_dB, times, n, seg):
    """Basic statistics on trajectory, altitude, speed, RCS, turning."""
    speeds = seg['speeds']
    alt_diffs = seg['alt_diffs']
    bearing_changes = seg['bearing_changes']
    duration = seg['duration']
    total_dist = seg['total_dist']
    straight_dist = seg['straight_dist']
    dt = seg['dt']

    sinuosity = total_dist / max(straight_dist, 1e-6)
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    # Acceleration: dv / dt_midpoint (correct formula)
    if len(speeds) > 1:
        dt_mid = 0.5 * (dt[:-1] + dt[1:]) if len(dt) > 1 else dt[:1]
        n_accel = min(len(speeds) - 1, len(dt_mid))
        accel = np.diff(speeds)[:n_accel] / np.maximum(dt_mid[:n_accel], 0.001)
    else:
        accel = np.array([0.0])

    # Halves comparison
    mid = n // 2
    if mid > 1:
        alt_first, alt_second = np.mean(alts[:mid]), np.mean(alts[mid:])
        rcs_first, rcs_second = np.mean(rcs_dB[:mid]), np.mean(rcs_dB[mid:])
    else:
        alt_first = alt_second = np.mean(alts)
        rcs_first = rcs_second = np.mean(rcs_dB)

    return {
        "n_points": n,
        "duration": duration,
        "total_dist": total_dist,
        "straight_dist": straight_dist,
        "sinuosity": sinuosity,  # no clipping
        # altitude
        "alt_mean": np.mean(alts), "alt_std": np.std(alts),
        "alt_min": np.min(alts), "alt_max": np.max(alts),
        "alt_range": np.ptp(alts), "alt_median": np.median(alts),
        "alt_q25": np.percentile(alts, 25), "alt_q75": np.percentile(alts, 75),
        "alt_iqr": np.percentile(alts, 75) - np.percentile(alts, 25),
        "alt_diff_mean": np.mean(np.abs(alt_diffs)),
        "alt_diff_std": np.std(alt_diffs),
        "climb_rate": climbing.sum() / max(duration, 0.001) if len(climbing) > 0 else 0,
        "descent_rate": abs(descending.sum()) / max(duration, 0.001) if len(descending) > 0 else 0,
        "climb_frac": len(climbing) / max(len(alt_diffs), 1),
        # RCS (dB scale — these are raw stats on the dB values as given)
        "rcs_mean": np.mean(rcs_dB), "rcs_std": np.std(rcs_dB),
        "rcs_min": np.min(rcs_dB), "rcs_max": np.max(rcs_dB),
        "rcs_range": np.ptp(rcs_dB), "rcs_median": np.median(rcs_dB),
        "rcs_q25": np.percentile(rcs_dB, 25), "rcs_q75": np.percentile(rcs_dB, 75),
        "rcs_iqr": np.percentile(rcs_dB, 75) - np.percentile(rcs_dB, 25),
        "rcs_skew": float(pd.Series(rcs_dB).skew()) if n > 2 else 0,
        # speed
        "speed_mean": np.mean(speeds), "speed_std": np.std(speeds),
        "speed_max": np.max(speeds), "speed_min": np.min(speeds),
        "speed_median": np.median(speeds),
        "avg_ground_speed": total_dist / max(duration, 0.001),
        # acceleration
        "accel_mean": np.mean(accel), "accel_std": np.std(accel),
        "accel_max": np.max(np.abs(accel)),
        # turning (computed with correct bearing formula)
        "bearing_change_mean": np.mean(np.abs(bearing_changes)),
        "bearing_change_std": np.std(bearing_changes),
        "bearing_change_max": np.max(np.abs(bearing_changes)),
        "total_turning": np.sum(np.abs(bearing_changes)),
        "net_turning": np.abs(np.sum(bearing_changes)),
        # position
        "lon_mean": np.mean(lons), "lat_mean": np.mean(lats),
        "lon_std": np.std(lons), "lat_std": np.std(lats),
        "spatial_spread": np.std(lons) + np.std(lats),
        # halves
        "alt_change_halves": alt_second - alt_first,
        "rcs_change_halves": rcs_second - rcs_first,
        # interactions
        "speed_x_alt": np.mean(speeds) * np.mean(alts),
        "rcs_x_alt": np.mean(rcs_dB) * np.mean(alts),
        "dist_per_point": total_dist / max(n, 1),
    }


def _rcs_linear_features(rcs_dB, n):
    """RCS features computed on LINEAR scale (correct physics)."""
    rcs_lin = 10.0 ** (rcs_dB / 10.0)
    rcs_lin_mean = np.mean(rcs_lin)

    return {
        "rcs_linear_mean": rcs_lin_mean,
        "rcs_linear_std": np.std(rcs_lin),
        # CV on linear scale (correct — CV is meaningless on dB)
        "rcs_cv": np.std(rcs_lin) / max(rcs_lin_mean, 1e-12),
        "rcs_stability": 1.0 / (1.0 + np.std(rcs_lin) / max(rcs_lin_mean, 1e-12)),
        # Scintillation index (unbiased variance)
        "rcs_scintillation": float(np.var(rcs_lin, ddof=1) / max(rcs_lin_mean ** 2, 1e-12)) if n > 2 else 0,
        # Deep fade fraction
        "rcs_deep_fade_frac": float(np.mean(rcs_lin < 0.1 * rcs_lin_mean)) if rcs_lin_mean > 1e-12 else 0,
        # Kurtosis on linear
        "rcs_linear_kurtosis": float(pd.Series(rcs_lin).kurtosis()) if n > 3 else 0,
    }


def _rcs_spectral_features(rcs_dB, times, n):
    """Spectral features from RCS — computed on RAW data, no interpolation."""
    feats = {
        "rcs_peak_freq": 0, "rcs_peak_power": 0,
        "rcs_total_power": 0, "rcs_spectral_centroid": 0,
        "rcs_ac1": 0, "rcs_ac2": 0,
    }
    if n < 4:
        return feats

    # Autocorrelation on raw dB RCS (no interpolation needed)
    rcs_c = rcs_dB - np.mean(rcs_dB)
    var = np.var(rcs_dB)
    if var > 1e-12:
        feats["rcs_ac1"] = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var)
        if n > 4:
            feats["rcs_ac2"] = float(np.mean(rcs_c[:-2] * rcs_c[2:]) / var)

    # Power spectral density via Welch on RAW (non-uniform) data
    # The radar is 1 Hz, so times are approximately uniform
    # Use the raw data directly — no interpolation
    if n >= 8:
        try:
            from scipy.signal import welch
            fs = (n - 1) / max(times[-1] - times[0], 0.001)
            freqs, psd = welch(rcs_c, fs=fs, nperseg=min(n, 32))
            if len(psd) > 1:
                peak_idx = np.argmax(psd[1:]) + 1
                feats["rcs_peak_freq"] = float(freqs[peak_idx])
                feats["rcs_peak_power"] = float(psd[peak_idx])
                feats["rcs_total_power"] = float(psd[1:].sum())
                feats["rcs_spectral_centroid"] = float(np.sum(freqs[1:] * psd[1:]) / max(psd[1:].sum(), 1e-10))
        except Exception:
            pass

    return feats


def _flight_mode_features(lons, lats, alts, rcs_dB, speeds, seg, n):
    """Flight mode features: soaring, flapping, gliding, curvature."""
    duration = seg['duration']
    bearing_changes = seg['bearing_changes']
    dt = seg['dt']

    feats = {
        "soaring_index": 0, "alt_gain_rate": 0, "slow_flight_frac": 0,
        "flap_frac": 0, "glide_frac": 0,
        "alt_rate_mean": 0, "alt_rate_std": 0, "alt_rate_max": 0,
        "alt_accel_std": 0,
        "curvature_mean": 0, "curvature_max": 0,
        "alt_osc_freq": 0, "alt_osc_amplitude": 0,
        "turn_radius": 0,
    }

    if n < 3:
        return feats

    # Speed-based
    speed_below_10 = float(np.mean(speeds < 10.0))
    feats["slow_flight_frac"] = speed_below_10

    # Altitude rate (raw, no filtering)
    alt_rate = np.diff(alts) / np.maximum(dt, 0.001)
    feats["alt_rate_mean"] = float(np.mean(alt_rate))
    feats["alt_rate_std"] = float(np.std(alt_rate))
    feats["alt_rate_max"] = float(np.max(np.abs(alt_rate)))

    if len(alt_rate) > 1:
        alt_accel = np.diff(alt_rate)
        feats["alt_accel_std"] = float(np.std(alt_accel))

    # Soaring: altitude gain while turning
    alt_gain = max(np.max(alts) - alts[0], 0)
    mean_turn = np.mean(np.abs(bearing_changes)) if len(bearing_changes) > 0 else 0
    feats["soaring_index"] = float(alt_gain * mean_turn / max(duration, 0.001))
    feats["alt_gain_rate"] = float(alt_gain / max(duration, 0.001))

    # Flap/glide fractions from RCS variance in windows
    rcs_lin = 10.0 ** (rcs_dB / 10.0)
    win = max(3, n // 5)
    if n >= win:
        rcs_vars = [np.var(rcs_lin[i:i+win]) for i in range(n - win + 1)]
        median_var = np.median(rcs_vars) if rcs_vars else 0
        feats["flap_frac"] = float(np.mean([v > median_var for v in rcs_vars]))
        feats["glide_frac"] = 1.0 - feats["flap_frac"]

    # Curvature (correct: using meter-scaled coordinates)
    dx = np.diff(lons) * 67000.0
    dy = np.diff(lats) * 111000.0
    if len(dx) > 1:
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        ds = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
        cross = np.abs(dx[:-1] * ddy - dy[:-1] * ddx)
        curvature = cross / np.maximum(ds**3, 1e-10)
        feats["curvature_mean"] = float(np.mean(curvature))
        feats["curvature_max"] = float(np.max(curvature))

    # Altitude oscillation (mean-subtracted, NOT detrended)
    alt_centered = alts - np.mean(alts)
    zero_crossings = np.sum(np.diff(np.sign(alt_centered)) != 0)
    feats["alt_osc_freq"] = float(zero_crossings / max(duration, 0.001))
    feats["alt_osc_amplitude"] = float(np.std(alt_centered) * 2)

    # Turn radius (from raw speeds and bearing changes)
    if len(bearing_changes) > 1 and len(dt) > 1:
        dt_bc = 0.5 * (dt[:-1] + dt[1:]) if len(dt) > 1 else dt[:1]
        n_bc = min(len(bearing_changes), len(dt_bc))
        angular_vel = np.abs(bearing_changes[:n_bc]) / np.maximum(dt_bc[:n_bc], 0.001)
        mid_speeds = 0.5 * (speeds[:-1] + speeds[1:]) if len(speeds) > 1 else speeds[:1]
        mid_speeds = mid_speeds[:n_bc]
        turning = angular_vel > 0.01
        if turning.sum() > 0:
            radii = np.clip(mid_speeds[turning] / angular_vel[turning], 0, 10000)
            feats["turn_radius"] = float(np.median(radii))

    return feats


def _tabular_features(df_row, feats_dict):
    """Features from tabular columns (airspeed, radar_bird_size, min_z, max_z)."""
    airspeed = float(pd.to_numeric(df_row.get("airspeed", 0), errors="coerce") or 0)
    min_z = float(pd.to_numeric(df_row.get("min_z", 0), errors="coerce") or 0)
    max_z = float(pd.to_numeric(df_row.get("max_z", 0), errors="coerce") or 0)
    size_str = str(df_row.get("radar_bird_size", ""))
    size_val = SIZE_MAP.get(size_str, 0)

    avg_gs = feats_dict.get("avg_ground_speed", 0)
    rcs_mean = feats_dict.get("rcs_mean", -25)
    alt_mean = feats_dict.get("alt_mean", 0)

    # RCS per altitude: use LINEAR RCS / altitude (correct physics)
    rcs_lin_mean = feats_dict.get("rcs_linear_mean", 10.0 ** (rcs_mean / 10.0))
    rcs_per_alt = rcs_lin_mean / max(abs(alt_mean), 1.0)

    return {
        "airspeed": airspeed,
        "airspeed_vs_ground": airspeed / max(avg_gs, 0.01) if avg_gs > 0.01 else 1.0,
        "radar_bird_size": size_val,
        "size_x_alt": size_val * alt_mean,
        "size_x_airspeed": size_val * airspeed,
        "rcs_for_size": rcs_mean - (size_val * 3 - 30),  # dB residual (keeps dB scale)
        "rcs_per_alt": rcs_per_alt,
        "min_z": min_z,
        "max_z": max_z,
        "alt_mid": 0.5 * (min_z + max_z),
        "alt_tabular_range": max_z - min_z,
    }


def _heading_features(bearings, rcs_dB, n):
    """Heading consistency and RCS autocorrelation."""
    feats = {"heading_R": 0, "rcs_ac1_traj": 0}
    if n < 6:
        return feats

    # Heading resultant length (circular consistency, 0=erratic, 1=straight)
    if len(bearings) > 1:
        R = float(np.sqrt(np.mean(np.sin(bearings))**2 + np.mean(np.cos(bearings))**2))
        feats["heading_R"] = R if np.isfinite(R) else 0

    # RCS lag-1 autocorrelation
    rcs_c = rcs_dB - np.mean(rcs_dB)
    var = np.var(rcs_dB)
    if var > 1e-12 and n > 3:
        feats["rcs_ac1_traj"] = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var)

    return feats


# ─────────────────────────────────────────────────────────────
# External data loading
# ─────────────────────────────────────────────────────────────

def _load_external(df, split):
    """Load aligned external CSVs. Returns dict of column_name: values."""
    n = len(df)
    ext = {}

    datasets = [
        ('weather', 'wx_'),
        ('solar', 'sol_'),
        ('tidal', 'tid_'),
        ('water', 'wat_'),
        ('landuse', 'lu_'),
        ('turbines', 'turb_'),
        ('visibility', 'vis_'),
        ('altitude_winds', 'aw_'),
        ('pressure', 'pres_'),
        ('marine', 'mar_'),
        ('cape', 'cape_'),
        ('moon', 'moon_'),
        ('photoperiod', 'photo_'),
        ('natura2000', 'nat_'),
        ('insect', 'ins_'),
    ]

    for name, prefix in datasets:
        path = ROOT / "data" / f"{split}_{name}.csv"
        if path.exists():
            try:
                csv = pd.read_csv(path)
                if len(csv) == n:
                    for col in csv.columns:
                        vals = pd.to_numeric(csv[col], errors="coerce").values.astype(float)
                        ext[f"{prefix}{col}"] = vals
            except Exception:
                pass

    return ext


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def build_features(df, include_external=True):
    """Build feature DataFrame from raw competition data.

    Single pipeline. No versioning. No filtering. Correct math.

    Args:
        df: raw train or test DataFrame with trajectory, trajectory_time, etc.
        include_external: whether to load external CSV data (weather, solar, etc.)

    Returns:
        pd.DataFrame with one row per track, columns are features.
    """
    n_tracks = len(df)
    all_feats = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Features: {i}/{n_tracks}", flush=True)

        try:
            lons, lats, alts, rcs_dB, times, n = _parse_track(
                row["trajectory"], row["trajectory_time"]
            )
            seg = _segments(lons, lats, alts, rcs_dB, times, n)

            f = {}
            f.update(_core_features(lons, lats, alts, rcs_dB, times, n, seg))
            f.update(_rcs_linear_features(rcs_dB, n))
            f.update(_rcs_spectral_features(rcs_dB, times, n))
            f.update(_flight_mode_features(lons, lats, alts, rcs_dB, seg['speeds'], seg, n))
            f.update(_tabular_features(row, f))
            f.update(_heading_features(seg['bearings'], rcs_dB, n))

            all_feats.append(f)
        except Exception as e:
            # Single-point or corrupt tracks get zeros
            all_feats.append({"n_points": 1, "duration": 0.001})

    print(f"  Features: {n_tracks}/{n_tracks} done", flush=True)
    feat_df = pd.DataFrame(all_feats)

    # External data (weather, solar, tidal, etc.)
    if include_external:
        split = "train" if "bird_group" in df.columns else "test"
        ext = _load_external(df, split)
        for col_name, vals in ext.items():
            feat_df[col_name] = vals

    # Clean inf/nan at the END only (on computed features, never on raw data)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat_df
