"""Feature extraction v2 — rebuilt from ground up after 8 audits.

Every feature has a documented physical interpretation and verified correctness.
Key fixes vs v1:
  - RCS dual-scale (dB + linear) — correct physics for mean, CV, skew
  - start/end altitude (strongest missing altitude discriminator)
  - track_heading (strongest missing spatial discriminator — Pigeons 75% East)
  - heading_std (circular std — BoP 71° vs Cormorants 30°)
  - headwind_component (wind projected along heading — Pigeons +6.1 m/s)
  - rcs_scintillation (flock detection — SI>>1 for flock, ~1 single bird)
  - rcs_dB_per_log_alt (correct altitude-normalised RCS, 2x better than rcs_per_alt)
  - Removed: broken flap/glide (~50/50 always), redundant shape features,
    soaring_index (highest for Clutter not BoP), wrong rcs_cv/rcs_stability

Usage:
    from src.features_v2 import build_features_v2
    train_feats = build_features_v2(train_df)
    test_feats  = build_features_v2(test_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from .data import parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent

SIZE_MAP: dict[str, int] = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}


# ── Haversine ─────────────────────────────────────────────────────────

def _haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Core trajectory features (FIXED + NEW) ───────────────────────────

def extract_trajectory_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract ~35 clean trajectory features from EWKB + trajectory_time.

    All features have documented physical interpretation.
    Fixes: RCS dual-scale, start/end alt, track heading, heading std.
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs_dB = np.array([p[3] for p in pts])
    n = len(pts)
    duration = times[-1] - times[0] if n > 1 else 0.001

    # ── Segments (filter radar glitches with dt < 0.5s) ──
    if n > 1:
        dists = np.array([_haversine(lons[i], lats[i], lons[i + 1], lats[i + 1])
                          for i in range(n - 1)])
        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)
        raw_speeds = dists / dt
        valid_seg = raw_dt >= 0.5
        if valid_seg.sum() >= 1:
            speeds = raw_speeds[valid_seg]
            dt_valid = dt[valid_seg]
        else:
            speeds = raw_speeds
            dt_valid = dt
    else:
        dists = np.array([0.0])
        dt = np.array([1.0])
        raw_dt = dt
        speeds = np.array([0.0])
        dt_valid = dt
        valid_seg = np.array([True])

    total_dist = float(dists.sum())
    straight_dist = _haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0.0

    # ── Altitude (9 features) ──
    alt_diffs = np.diff(alts) if n > 1 else np.array([0.0])
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    alt_mean = float(np.mean(alts))
    alt_std = float(np.std(alts))
    alt_iqr = float(np.percentile(alts, 75) - np.percentile(alts, 25))
    start_alt = float(alts[0])
    end_alt = float(alts[-1])
    alt_diff_start_end = end_alt - start_alt
    climb_rate = float(climbing.sum() / max(duration, 0.001)) if len(climbing) > 0 else 0.0
    descent_rate = float(abs(descending.sum()) / max(duration, 0.001)) if len(descending) > 0 else 0.0
    climb_frac = float(len(climbing) / max(len(alt_diffs), 1))
    alt_max = float(np.max(alts))
    alt_median = float(np.median(alts))

    # Altitude rate (key Wader discriminator)
    if n > 1:
        alt_rate = alt_diffs / dt
        alt_rate_mean = float(np.mean(alt_rate))
    else:
        alt_rate_mean = 0.0

    # ── RCS dual-scale (10+3 features) ──
    rcs_mean_dB = float(np.mean(rcs_dB))
    rcs_std_dB = float(np.std(rcs_dB))
    rcs_iqr_dB = float(np.percentile(rcs_dB, 75) - np.percentile(rcs_dB, 25))
    rcs_median_dB = float(np.median(rcs_dB))
    rcs_q25_dB = float(np.percentile(rcs_dB, 25))
    rcs_q75_dB = float(np.percentile(rcs_dB, 75))

    # Convert to linear (mW) for physically correct statistics
    rcs_linear = 10.0 ** (rcs_dB / 10.0)
    rcs_linear_mean = float(np.mean(rcs_linear))
    rcs_linear_std = float(np.std(rcs_linear))
    rcs_linear_cv = rcs_linear_std / max(rcs_linear_mean, 1e-12)
    rcs_scintillation = float(np.var(rcs_linear) / max(rcs_linear_mean ** 2, 1e-12))

    # Skewness on linear scale (wins 7/9 classes per audit)
    if n > 2 and rcs_linear_std > 1e-12:
        rcs_skew_linear = float(np.mean(((rcs_linear - rcs_linear_mean) / rcs_linear_std) ** 3))
    else:
        rcs_skew_linear = 0.0

    # Deep fade fraction (flock interference — Geese SI>>1)
    rcs_deep_fade_fraction = float(np.mean(rcs_linear < 0.1 * rcs_linear_mean)) if rcs_linear_mean > 1e-12 else 0.0

    # Kurtosis on linear scale (spiky = flock)
    if n > 3 and rcs_linear_std > 1e-12:
        rcs_kurtosis_linear = float(
            np.mean(((rcs_linear - rcs_linear_mean) / rcs_linear_std) ** 4) - 3.0
        )
    else:
        rcs_kurtosis_linear = 0.0

    # Correct altitude-normalised RCS
    if alt_mean > 0.1:
        rcs_dB_per_log_alt = rcs_mean_dB - 10.0 * np.log10(max(alt_mean, 0.1))
    else:
        rcs_dB_per_log_alt = rcs_mean_dB

    # RCS autocorrelation lag-1 (best Cormorant discriminator d=0.62)
    rcs_centered = rcs_dB - rcs_mean_dB
    rcs_var = float(np.var(rcs_dB))
    if rcs_var > 1e-12 and n > 1:
        rcs_ac1 = float(np.mean(rcs_centered[:-1] * rcs_centered[1:]) / rcs_var)
        if not np.isfinite(rcs_ac1):
            rcs_ac1 = 0.0
    else:
        rcs_ac1 = 0.0

    # ── Speed (6 features) ──
    speed_median = float(np.median(speeds))
    avg_ground_speed = total_dist / max(duration, 0.001)

    if len(speeds) > 1 and len(dt_valid) > 1:
        dt_mid = 0.5 * (dt_valid[:-1] + dt_valid[1:]) if len(dt_valid) > 1 else dt_valid[:1]
        accel = np.diff(speeds) / np.maximum(dt_mid[:len(speeds) - 1], 0.001)
        accel_std_val = float(np.std(accel))
    else:
        accel_std_val = 0.0

    slow_flight_frac = float(np.mean(speeds < 10.0))

    # ── Heading & turning (4 features) ──
    if n > 2:
        dx = np.diff(lons) * 67000.0
        dy = np.diff(lats) * 111000.0
        seg_headings = np.arctan2(dy, dx)
        bearing_changes = np.arctan2(np.sin(np.diff(seg_headings)),
                                     np.cos(np.diff(seg_headings)))
        bearing_change_mean = float(np.mean(np.abs(bearing_changes)))

        # Circular std of segment headings
        sin_mean = float(np.mean(np.sin(seg_headings)))
        cos_mean = float(np.mean(np.cos(seg_headings)))
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        heading_std = float(np.sqrt(-2.0 * np.log(max(R, 1e-12)))) if R < 1.0 else 0.0

        # Curvature mean (key BoP/Cormorant discriminator)
        ds = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
        if len(dx) > 1:
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (ds[:-1] ** 3 + 1e-10)
            curvature_mean = float(np.mean(curvature))
        else:
            curvature_mean = 0.0
    else:
        bearing_change_mean = 0.0
        heading_std = 0.0
        curvature_mean = 0.0

    # Track heading = arctan2 of start->end displacement
    if n > 1 and straight_dist > 1.0:
        dx_net = (lons[-1] - lons[0]) * 67000.0
        dy_net = (lats[-1] - lats[0]) * 111000.0
        track_heading = float(np.arctan2(dy_net, dx_net))
    else:
        track_heading = 0.0

    # Straightness = net_displacement / total_path
    straightness = float(straight_dist / max(total_dist, 1e-6))
    straightness = min(straightness, 1.0)

    # ── Position (4 features) ──
    lon_mean = float(np.mean(lons))
    lat_mean = float(np.mean(lats))
    lon_std = float(np.std(lons))
    lat_std = float(np.std(lats))

    # ── Cross-features (2 features) ──
    speed_x_alt = speed_median * alt_mean

    mid = n // 2
    if mid > 1:
        alt_change_halves = float(np.mean(alts[mid:]) - np.mean(alts[:mid]))
    else:
        alt_change_halves = 0.0

    # ── Track metadata (2 features) ──
    n_points = n
    duration_val = duration

    return {
        # Track metadata
        "n_points": n_points,
        "duration": duration_val,
        # Altitude (12)
        "alt_mean": alt_mean,
        "alt_std": alt_std,
        "alt_iqr": alt_iqr,
        "alt_max": alt_max,
        "alt_median": alt_median,
        "alt_rate_mean": alt_rate_mean,
        "start_alt": start_alt,
        "end_alt": end_alt,
        "alt_diff_start_end": alt_diff_start_end,
        "climb_rate": climb_rate,
        "descent_rate": descent_rate,
        "climb_frac": climb_frac,
        # RCS dual-scale (13)
        "rcs_mean_dB": rcs_mean_dB,
        "rcs_std_dB": rcs_std_dB,
        "rcs_iqr_dB": rcs_iqr_dB,
        "rcs_median_dB": rcs_median_dB,
        "rcs_q25_dB": rcs_q25_dB,
        "rcs_q75_dB": rcs_q75_dB,
        "rcs_linear_mean": rcs_linear_mean,
        "rcs_linear_cv": rcs_linear_cv,
        "rcs_scintillation": rcs_scintillation,
        "rcs_skew_linear": rcs_skew_linear,
        "rcs_dB_per_log_alt": rcs_dB_per_log_alt,
        "rcs_autocorr_lag1": rcs_ac1,
        "rcs_deep_fade_frac": rcs_deep_fade_fraction,
        "rcs_kurtosis_linear": rcs_kurtosis_linear,
        # Speed (6)
        "airspeed_vs_ground": 0.0,  # placeholder, set in tabular
        "avg_ground_speed": avg_ground_speed,
        "speed_median": speed_median,
        "accel_std": accel_std_val,
        "slow_flight_frac": slow_flight_frac,
        # Heading & turning (5)
        "track_heading": track_heading,
        "heading_std": heading_std,
        "bearing_change_mean": bearing_change_mean,
        "straightness": straightness,
        "curvature_mean": curvature_mean,
        # Position (4)
        "lon_mean": lon_mean,
        "lat_mean": lat_mean,
        "lon_std": lon_std,
        "lat_std": lat_std,
        # Cross-features (2)
        "speed_x_alt": speed_x_alt,
        "alt_change_halves": alt_change_halves,
    }


# ── External data features ───────────────────────────────────────────

def _load_external_csv(name: str, split: str) -> pd.DataFrame:
    """Load an aligned external CSV. Returns empty DataFrame if missing.

    Validates row count matches the expected dataset size to catch
    silent misalignment issues.
    """
    path = ROOT / "data" / f"{split}_{name}.csv"
    if path.exists():
        ext_df = pd.read_csv(path)
        expected = _full_dataset_len(split)
        if len(ext_df) != expected:
            import warnings
            warnings.warn(
                f"{split}_{name}.csv has {len(ext_df)} rows, "
                f"expected {expected}. Returning empty DataFrame.",
                stacklevel=2,
            )
            return pd.DataFrame()
        return ext_df
    return pd.DataFrame()


def _full_dataset_len(split: str) -> int:
    """Return the expected number of rows in the full train/test dataset."""
    from .data import load_train, load_test
    if split == "train":
        return len(load_train())
    return len(load_test())


def add_external_features(
    df_feat: pd.DataFrame, split: str, row_indices: np.ndarray | None = None
) -> pd.DataFrame:
    """Add SAFE + INDEPENDENT external features only.

    Tier 1 (safe + independent): dist_to_water, over_water, water_level,
        hours_since_high_tide, tide_rising, dist_to_turbine
    Tier 2 (safe but partially redundant): turbines_within_500m, lifted_index
    Derived: headwind_component (requires track_heading + ERA5 winds)

    Args:
        row_indices: positional indices into the full CSV (for subsets). None = full.
    """
    def _vals(ext_df: pd.DataFrame, col: str) -> np.ndarray:
        v = ext_df[col].values
        if row_indices is not None:
            v = v[row_indices]
        return v

    # --- Tier 1: Water features ---
    water = _load_external_csv("water", split)
    if not water.empty:
        if "dist_to_water_m" in water.columns:
            df_feat["dist_to_water_m"] = _vals(water, "dist_to_water_m")
        if "over_water" in water.columns:
            df_feat["over_water"] = _vals(water, "over_water")

    # --- Tier 1: Tidal features ---
    tidal = _load_external_csv("tidal", split)
    if not tidal.empty:
        if "water_level_cm" in tidal.columns:
            df_feat["water_level_cm"] = _vals(tidal, "water_level_cm")
        if "hours_since_high_tide" in tidal.columns:
            df_feat["hours_since_high_tide"] = _vals(tidal, "hours_since_high_tide")
        if "tide_rising" in tidal.columns:
            df_feat["tide_rising"] = _vals(tidal, "tide_rising")
        if "tidal_phase" in tidal.columns:
            df_feat["tidal_phase"] = _vals(tidal, "tidal_phase")

    # --- Tier 1: Turbine features ---
    turbines = _load_external_csv("turbines", split)
    if not turbines.empty:
        if "dist_to_turbine_m" in turbines.columns:
            df_feat["dist_to_turbine_m"] = _vals(turbines, "dist_to_turbine_m")
        # Tier 2
        if "turbines_within_500m" in turbines.columns:
            df_feat["turbines_within_500m"] = _vals(turbines, "turbines_within_500m")

    # --- Tier 2: CAPE / lifted index ---
    cape = _load_external_csv("cape", split)
    if not cape.empty and "lifted_index" in cape.columns:
        df_feat["lifted_index"] = _vals(cape, "lifted_index")

    # --- Moon features (nocturnal migration signal) ---
    moon = _load_external_csv("moon", split)
    if not moon.empty:
        if "moon_illumination" in moon.columns:
            df_feat["moon_illumination"] = _vals(moon, "moon_illumination")
        if "is_day" in moon.columns:
            df_feat["is_day"] = _vals(moon, "is_day")

    # --- Pressure trends (migration trigger) ---
    pressure = _load_external_csv("pressure", split)
    if not pressure.empty:
        if "pressure_trend_3h" in pressure.columns:
            df_feat["pressure_trend_3h"] = _vals(pressure, "pressure_trend_3h")
        if "pressure_trend_12h" in pressure.columns:
            df_feat["pressure_trend_12h"] = _vals(pressure, "pressure_trend_12h")

    # --- Landuse (grassland distance — proven +0.0011 IW-mAP) ---
    landuse = _load_external_csv("landuse", split)
    if not landuse.empty:
        if "dist_to_grassland_m" in landuse.columns:
            df_feat["dist_to_grassland_m"] = _vals(landuse, "dist_to_grassland_m")

    # --- Altitude winds (boundary layer height — BoP thermal proxy) ---
    alt_winds = _load_external_csv("altitude_winds", split)
    if not alt_winds.empty:
        if "boundary_layer_height" in alt_winds.columns:
            df_feat["boundary_layer_height"] = _vals(alt_winds, "boundary_layer_height")

    # --- Derived: headwind component ---
    era5 = _load_external_csv("era5_winds", split)
    if not era5.empty and "track_heading" in df_feat.columns:
        wind_speed = None
        wind_dir = None
        if "era5_wind_10m" in era5.columns:
            wind_speed = _vals(era5, "era5_wind_10m")
        if "era5_wind_dir_at_alt" in era5.columns:
            wind_dir = _vals(era5, "era5_wind_dir_at_alt")

        if wind_speed is not None and wind_dir is not None:
            track_h = df_feat["track_heading"].values
            # Convert meteorological wind direction (deg CW from N, direction FROM)
            # to math convention (rad CCW from E)
            wind_from_math = np.pi / 2.0 - np.radians(wind_dir)
            # Headwind = component of wind opposing bird's motion
            # Positive = headwind (bird flies into wind)
            df_feat["headwind_component"] = wind_speed * np.cos(wind_from_math - track_h)

    return df_feat


# ── Tabular features (CLEANED) ───────────────────────────────────────

def add_tabular_features(
    df_feat: pd.DataFrame, df_orig: pd.DataFrame, row_indices: np.ndarray | None = None
) -> pd.DataFrame:
    """Add cleaned tabular features — no date-proxy leaks.

    Keeps: airspeed, min_z, max_z, z_range, z_mean, radar_bird_size,
           size_x_alt, sol_hours_since_sunrise, sol_daylight_fraction,
           sol_solar_elevation, wx_wind_speed, wx_wind_gust, wx_wind_v, wx_humidity
    Drops: sol_daylight_hours (0% within-date), wx_temp_c (3%),
           wx_dewpoint_c (2%), wx_wind_u (3%), all month/hour/temporal features

    Args:
        row_indices: original positional indices into the full dataset CSV
                     (for slicing external CSVs). None = full dataset.
    """
    # Core tabular
    df_feat["airspeed"] = pd.to_numeric(df_orig["airspeed"], errors="coerce").values
    df_feat["min_z"] = pd.to_numeric(df_orig["min_z"], errors="coerce").values
    df_feat["max_z"] = pd.to_numeric(df_orig["max_z"], errors="coerce").values
    df_feat["z_range"] = df_feat["max_z"] - df_feat["min_z"]
    df_feat["z_mean"] = (df_feat["max_z"] + df_feat["min_z"]) / 2.0

    # Radar bird size (numeric)
    df_feat["radar_bird_size"] = df_orig["radar_bird_size"].map(SIZE_MAP).values

    # Size interaction
    df_feat["size_x_alt"] = df_feat["radar_bird_size"] * df_feat["alt_mean"]

    # Fix airspeed_vs_ground (needs both airspeed and avg_ground_speed)
    if "avg_ground_speed" in df_feat.columns:
        df_feat["airspeed_vs_ground"] = (
            df_feat["airspeed"] / df_feat["avg_ground_speed"].clip(lower=0.01)
        )

    # RCS for size calibration
    if "rcs_mean_dB" in df_feat.columns:
        df_feat["rcs_for_size"] = df_feat["rcs_mean_dB"] / df_feat["radar_bird_size"].clip(lower=1)

    def _ext_vals(ext_df: pd.DataFrame, col: str) -> np.ndarray:
        """Extract values from external CSV, slicing by row_indices if needed."""
        vals = ext_df[col].values
        if row_indices is not None:
            vals = vals[row_indices]
        return vals

    # Solar features (safe ones only)
    split = "train" if "bird_group" in df_orig.columns else "test"
    solar = _load_external_csv("solar", split)
    if not solar.empty:
        if "hours_since_sunrise" in solar.columns:
            hrs = _ext_vals(solar, "hours_since_sunrise")
            df_feat["sol_hours_since_sunrise"] = hrs
            # Crepuscular index: within 1 hour of sunrise/sunset
            # Key Pigeon vs Duck discriminator (confusion pair 2)
            if "daylight_hours" in solar.columns:
                daylight = _ext_vals(solar, "daylight_hours")
                hrs_since_sunset = daylight - hrs
                df_feat["crepuscular_index"] = (
                    (hrs < 1.0) | (hrs_since_sunset < 1.0)
                ).astype(float)
        if "daylight_fraction" in solar.columns:
            df_feat["sol_daylight_fraction"] = _ext_vals(solar, "daylight_fraction")
        if "solar_elevation" in solar.columns:
            df_feat["sol_solar_elevation"] = _ext_vals(solar, "solar_elevation")

    # Weather features (safe ones only)
    weather = _load_external_csv("weather", split)
    if not weather.empty:
        for col in ["wind_speed", "wind_gust", "wind_v", "humidity"]:
            if col in weather.columns:
                df_feat[f"wx_{col}"] = _ext_vals(weather, col)

    return df_feat


# ── Main build function ──────────────────────────────────────────────

def build_features_v2(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Build clean v2 feature DataFrame from raw data.

    Steps:
      1. Extract trajectory features (per-track, ~35 features)
      2. Add tabular features (~14 features)
      3. Add external data features (~9 features)
      4. Handle inf/nan

    Args:
        df: raw train or test DataFrame
        cache_path: optional path to cache/load pickled features

    Returns:
        DataFrame with ~58 features, index aligned with df
    """
    if cache_path is not None and cache_path.exists():
        print(f"  Loading cached features from {cache_path.name}", flush=True)
        return pd.read_pickle(cache_path)

    # Step 1: Trajectory features
    print("  Extracting trajectory features...", flush=True)
    rows = []
    total = len(df)
    # Track original positional indices for external CSV slicing
    original_indices = np.arange(total) if df.index.equals(pd.RangeIndex(total)) else None
    if not df.index.equals(pd.RangeIndex(total)):
        # If df is a subset (e.g. head(5)), we need positional indices
        # to slice external CSVs correctly
        original_indices = np.array(df.index)

    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    Progress: {idx}/{total}", flush=True)
        feats = extract_trajectory_features(r.trajectory, r.trajectory_time)
        rows.append(feats)
    df_feat = pd.DataFrame(rows)

    # Step 2: Tabular features
    print("  Adding tabular features...", flush=True)
    split = "train" if "bird_group" in df.columns else "test"
    # Only pass row_indices if this is a subset of the full dataset
    row_idx = original_indices if original_indices is not None and len(df) != _full_dataset_len(split) else None
    df_feat = add_tabular_features(df_feat, df.reset_index(drop=True), row_idx)

    # Step 3: External features
    print("  Adding external features...", flush=True)
    df_feat = add_external_features(df_feat, split, row_idx)

    # Step 4: Handle inf/nan
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    if cache_path is not None:
        df_feat.to_pickle(cache_path)
        print(f"  Cached features to {cache_path.name}", flush=True)

    print(f"  Total features: {df_feat.shape[1]}", flush=True)
    return df_feat
