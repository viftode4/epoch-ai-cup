"""Feature extraction from radar trajectories.

Each feature set is a separate function that returns a dict.
Combine them in experiments as needed.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .data import CLASSES, parse_ewkb_4d, parse_trajectory_time

# Canonical list of ALL temporal features that cause train/test leakage.
# Train months [1,4,9,10] vs test months [2,5,9,10,12] -- 33% unseen.
# Import this instead of copy-pasting across experiments.
ALL_TEMPORAL = [
    # 18 original temporal features
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring", "hour_bin_3h",
    # 5 weakclass temporal leaks
    "is_oct_nov", "migration_alt", "migration_speed", "is_night", "night_high_alt",
]


SIZE_MAP: dict[str, int] = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}


def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Core trajectory features (v2 baseline set) ─────────────────────

def extract_core_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract the standard feature set used in v2 baseline."""
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)
    duration = times[-1] - times[0] if n > 1 else 0.001

    # Segments — filter out near-zero dt to avoid extreme speed outliers
    if n > 1:
        dists = np.array([haversine(lons[i], lats[i], lons[i + 1], lats[i + 1]) for i in range(n - 1)])
        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)
        raw_speeds = dists / dt
        # Mask out segments with dt < 0.5s (radar glitches)
        valid_seg = raw_dt >= 0.5
        if valid_seg.sum() >= 1:
            speeds = raw_speeds[valid_seg]
            dists_valid = dists[valid_seg]
            dt_valid = dt[valid_seg]
        else:
            speeds = raw_speeds
            dists_valid = dists
            dt_valid = dt
    else:
        dists, dt, speeds = np.array([0.0]), np.array([1.0]), np.array([0.0])
        raw_dt = dt
        valid_seg = np.array([True])
        dists_valid = dists
        dt_valid = dt
        raw_speeds = speeds

    total_dist = dists.sum()
    straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
    sinuosity = total_dist / max(straight_dist, 1e-6)

    alt_diffs = np.diff(alts) if n > 1 else np.array([0.0])
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    if n > 2:
        bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
        bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
    else:
        bearing_changes = np.array([0.0])

    # Acceleration (use filtered speeds + valid dt)
    if len(speeds) > 1 and len(dt_valid) > 1:
        dt_mid = 0.5 * (dt_valid[:-1] + dt_valid[1:]) if len(dt_valid) > 1 else dt_valid[:1]
        accel = np.diff(speeds) / np.maximum(dt_mid, 0.001)
    else:
        accel = np.array([0.0])

    # Halves
    mid = n // 2
    if mid > 1:
        alt_first, alt_second = np.mean(alts[:mid]), np.mean(alts[mid:])
        rcs_first, rcs_second = np.mean(rcs[:mid]), np.mean(rcs[mid:])
    else:
        alt_first = alt_second = np.mean(alts)
        rcs_first = rcs_second = np.mean(rcs)

    return {
        "n_points": n,
        "duration": duration,
        "total_dist": total_dist,
        "straight_dist": straight_dist,
        "sinuosity": min(sinuosity, 50),
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
        # RCS
        "rcs_mean": np.mean(rcs), "rcs_std": np.std(rcs),
        "rcs_min": np.min(rcs), "rcs_max": np.max(rcs),
        "rcs_range": np.ptp(rcs), "rcs_median": np.median(rcs),
        "rcs_q25": np.percentile(rcs, 25), "rcs_q75": np.percentile(rcs, 75),
        "rcs_iqr": np.percentile(rcs, 75) - np.percentile(rcs, 25),
        "rcs_skew": float(pd.Series(rcs).skew()) if n > 2 else 0,
        # speed
        "speed_mean": np.mean(speeds), "speed_std": np.std(speeds),
        "speed_max": np.max(speeds), "speed_min": np.min(speeds),
        "speed_median": np.median(speeds),
        "avg_ground_speed": total_dist / max(duration, 0.001),
        # acceleration
        "accel_mean": np.mean(accel), "accel_std": np.std(accel),
        "accel_max": np.max(np.abs(accel)),
        # turning
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
        # interactions (use filtered speeds for robustness)
        "speed_x_alt": np.median(speeds) * np.mean(alts),
        "rcs_x_alt": np.mean(rcs) * np.mean(alts),
        "dist_per_point": total_dist / max(n, 1),
    }


# ── RCS FFT features ───────────────────────────────────────────────

def extract_rcs_fft_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract FFT-based features from the RCS signal."""
    from scipy.interpolate import interp1d
    from scipy.signal import welch

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {"rcs_peak_freq": 0, "rcs_peak_power": 0, "rcs_total_power": 0, "rcs_spectral_centroid": 0}
    if n < 8:
        return defaults

    try:
        uniform_t = np.linspace(times[0], times[-1], n)
        fs = 1.0 / max(uniform_t[1] - uniform_t[0], 0.01)
        rcs_uniform = interp1d(times, rcs, kind="linear", fill_value="extrapolate")(uniform_t)
        freqs, psd = welch(rcs_uniform - np.mean(rcs_uniform), fs=fs, nperseg=min(n, 32))
        peak_idx = np.argmax(psd[1:]) + 1
        return {
            "rcs_peak_freq": freqs[peak_idx],
            "rcs_peak_power": psd[peak_idx],
            "rcs_total_power": psd[1:].sum(),
            "rcs_spectral_centroid": np.sum(freqs[1:] * psd[1:]) / max(psd[1:].sum(), 1e-10),
        }
    except Exception:
        return defaults


# ── Wavelet features (CWT on RCS — wingbeat extraction) ───────────

def extract_wavelet_features(hex_str: str, traj_time_str: str) -> dict:
    """CWT-based features from the RCS signal for wingbeat analysis.

    Replaces basic FFT with continuous wavelet transform (Morlet) which is
    better suited for short, non-stationary radar signals (Zaugg et al. 2008).
    """
    import pywt
    from scipy.signal import detrend

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {
        "cwt_energy_0_2hz": 0, "cwt_energy_2_5hz": 0,
        "cwt_energy_5_10hz": 0, "cwt_energy_10plus_hz": 0,
        "cwt_dominant_freq": 0, "cwt_freq_std": 0,
        "cwt_low_high_ratio": 0, "cwt_peak_scale_energy": 0,
        "cwt_total_energy": 0,
    }
    if n < 10:
        return defaults

    try:
        # Interpolate to uniform time grid
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults
        uniform_t = np.linspace(times[0], times[-1], n)
        dt = uniform_t[1] - uniform_t[0]
        fs = 1.0 / max(dt, 0.01)

        rcs_uniform = interp1d(times, rcs, kind="linear",
                               fill_value="extrapolate")(uniform_t)
        # Detrend (remove slow body-RCS drift, keep wingbeat oscillation)
        rcs_clean = detrend(rcs_uniform)

        # CWT with Morlet wavelet
        # Map frequency bands to scales: scale = central_freq * fs / freq
        central_freq = pywt.central_frequency("morl")
        max_freq = fs / 2
        # Scales corresponding to frequencies from 0.5 Hz to max_freq
        freqs_of_interest = np.linspace(0.5, min(max_freq, 25), 50)
        freqs_of_interest = freqs_of_interest[freqs_of_interest > 0]
        scales = central_freq * fs / freqs_of_interest

        coeffs, freqs = pywt.cwt(rcs_clean, scales, "morl",
                                 sampling_period=dt)
        power = np.abs(coeffs) ** 2

        # Energy in frequency bands
        energy_per_freq = power.mean(axis=1)  # average across time
        total_energy = energy_per_freq.sum() + 1e-10

        band_0_2 = energy_per_freq[(freqs >= 0) & (freqs < 2)].sum()
        band_2_5 = energy_per_freq[(freqs >= 2) & (freqs < 5)].sum()
        band_5_10 = energy_per_freq[(freqs >= 5) & (freqs < 10)].sum()
        band_10plus = energy_per_freq[freqs >= 10].sum()

        # Dominant frequency (peak of wavelet scalogram)
        peak_idx = np.argmax(energy_per_freq)
        dominant_freq = freqs[peak_idx]

        # Wingbeat regularity: weighted std of frequency
        freq_weights = energy_per_freq / total_energy
        freq_mean = np.sum(freqs * freq_weights)
        freq_std = np.sqrt(np.sum(freq_weights * (freqs - freq_mean) ** 2))

        low_energy = band_0_2 + band_2_5
        high_energy = band_5_10 + band_10plus
        low_high_ratio = low_energy / max(high_energy, 1e-10)

        return {
            "cwt_energy_0_2hz": band_0_2 / total_energy,
            "cwt_energy_2_5hz": band_2_5 / total_energy,
            "cwt_energy_5_10hz": band_5_10 / total_energy,
            "cwt_energy_10plus_hz": band_10plus / total_energy,
            "cwt_dominant_freq": dominant_freq,
            "cwt_freq_std": freq_std,
            "cwt_low_high_ratio": low_high_ratio,
            "cwt_peak_scale_energy": energy_per_freq[peak_idx],
            "cwt_total_energy": total_energy,
        }
    except Exception:
        return defaults


# ── Zaugg-style CWT features (32-band, for SVM) ──────────────────────

def extract_zaugg_cwt_features(hex_str: str, traj_time_str: str) -> dict:
    """Zaugg et al. 2008 style: 32 CWT frequency bands, per-band mean+std.

    Returns 64 band features + 3 signal stats = 67 features total.
    Designed for SVM (Laplace/RBF kernel), NOT for trees.
    """
    import pywt
    from scipy.signal import detrend

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    n_bands = 32
    defaults = {}
    for i in range(n_bands):
        defaults[f"zcwt_band{i:02d}_mean"] = 0.0
        defaults[f"zcwt_band{i:02d}_std"] = 0.0
    defaults["zcwt_signal_mean"] = 0.0
    defaults["zcwt_signal_std"] = 0.0
    defaults["zcwt_signal_energy"] = 0.0

    if n < 10:
        return defaults

    try:
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults

        # Interpolate to uniform time grid
        uniform_t = np.linspace(times[0], times[-1], n)
        dt = uniform_t[1] - uniform_t[0]
        fs = 1.0 / max(dt, 0.01)

        rcs_uniform = interp1d(times, rcs, kind="linear",
                               fill_value="extrapolate")(uniform_t)
        rcs_clean = detrend(rcs_uniform)

        # CWT with Morlet wavelet — 32 frequency bands from 0.31 to min(65, fs/2) Hz
        central_freq = pywt.central_frequency("morl")
        max_freq = min(fs / 2, 65.0)
        min_freq = 0.31
        if max_freq <= min_freq:
            return defaults

        # Logarithmically spaced frequencies (Zaugg style)
        band_freqs = np.geomspace(min_freq, max_freq, n_bands)
        scales = central_freq * fs / band_freqs

        coeffs, freqs = pywt.cwt(rcs_clean, scales, "morl",
                                 sampling_period=dt)
        power = np.abs(coeffs) ** 2

        result = {}
        for i in range(n_bands):
            band_power = power[i]
            result[f"zcwt_band{i:02d}_mean"] = float(np.mean(band_power))
            result[f"zcwt_band{i:02d}_std"] = float(np.std(band_power))

        # Signal-level stats
        result["zcwt_signal_mean"] = float(np.mean(rcs_clean))
        result["zcwt_signal_std"] = float(np.std(rcs_clean))
        result["zcwt_signal_energy"] = float(np.sum(rcs_clean ** 2))

        return result
    except Exception:
        return defaults


# ── Flight mode features (flap/glide segmentation + trajectory shape) ─

def extract_flight_mode_features(hex_str: str, traj_time_str: str) -> dict:
    """Segment track into flapping vs gliding, plus trajectory shape metrics.

    Hypothesis:
    - Pigeons: continuous flap (high flap_fraction, 0 glide)
    - Gulls: frequent gliding
    - Songbirds: bounding flight (periodic alt oscillation)
    - BoP: soaring (long glide, altitude gain, high curvature)
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {
        "flap_fraction": 0, "glide_fraction": 0,
        "n_mode_transitions": 0, "mean_flap_duration": 0,
        "mean_glide_duration": 0,
        "alt_osc_freq": 0, "alt_osc_amplitude": 0,
        "dir_autocorr_lag5": 0, "dir_autocorr_lag10": 0,
        "curvature_mean": 0, "curvature_max": 0,
        "effective_speed_ratio": 0,
    }

    if n < 6:
        return defaults

    try:
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults

        # === Flap/glide segmentation via RCS variance in sliding windows ===
        # High RCS variance = flapping (wing modulation), low = gliding
        win_size = max(3, n // 8)  # adaptive window
        rcs_var = np.array([
            np.var(rcs[max(0, i - win_size):i + 1])
            for i in range(n)
        ])
        # Threshold: median variance separates flap from glide
        threshold = np.median(rcs_var)
        is_flapping = rcs_var > threshold

        flap_frac = is_flapping.mean()
        glide_frac = 1.0 - flap_frac

        # Count mode transitions (flap↔glide switches)
        transitions = np.sum(np.diff(is_flapping.astype(int)) != 0)

        # Mean duration of flapping/gliding segments
        dt_arr = np.diff(times)
        flap_durations = []
        glide_durations = []
        seg_start = 0
        for i in range(1, n):
            if is_flapping[i] != is_flapping[seg_start]:
                seg_dur = times[i] - times[seg_start]
                if is_flapping[seg_start]:
                    flap_durations.append(seg_dur)
                else:
                    glide_durations.append(seg_dur)
                seg_start = i
        # Last segment
        seg_dur = times[-1] - times[seg_start]
        if is_flapping[seg_start]:
            flap_durations.append(seg_dur)
        else:
            glide_durations.append(seg_dur)

        mean_flap_dur = np.mean(flap_durations) if flap_durations else 0
        mean_glide_dur = np.mean(glide_durations) if glide_durations else 0

        # === Altitude oscillation (bounding flight detection) ===
        # Detrend altitude and find oscillation frequency
        alt_detrended = alts - np.linspace(alts[0], alts[-1], n)
        # Zero-crossing rate as proxy for oscillation frequency
        zero_crossings = np.sum(np.diff(np.sign(alt_detrended)) != 0)
        alt_osc_freq = zero_crossings / (2 * max(duration, 0.01))
        alt_osc_amplitude = np.std(alt_detrended) * 2  # ~peak-to-peak

        # === Direction autocorrelation ===
        if n > 2:
            bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
            cos_bearings = np.cos(bearings)

            def autocorr_at_lag(x, lag):
                if len(x) <= lag:
                    return 0
                x1 = x[:len(x) - lag]
                x2 = x[lag:]
                if np.std(x1) < 1e-10 or np.std(x2) < 1e-10:
                    return 1.0  # perfectly straight
                return np.corrcoef(x1, x2)[0, 1]

            dir_ac5 = autocorr_at_lag(cos_bearings, min(5, len(cos_bearings) - 1))
            dir_ac10 = autocorr_at_lag(cos_bearings, min(10, len(cos_bearings) - 1))
        else:
            dir_ac5 = dir_ac10 = 0

        # === Curvature (circling detection for BoP) ===
        # Use meter-scaled coordinates (same fix as bearings)
        if n > 3:
            dx = np.diff(lons) * 67000.0   # meters east
            dy = np.diff(lats) * 111000.0  # meters north
            ds = np.sqrt(dx**2 + dy**2) + 1e-10
            # Curvature = |x'y'' - y'x''| / |v|^3
            if len(dx) > 1:
                ddx = np.diff(dx)
                ddy = np.diff(dy)
                ds_mid = ds[:-1]
                curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (ds_mid**3 + 1e-10)
                curv_mean = np.mean(curvature)
                curv_max = np.max(curvature)
            else:
                curv_mean = curv_max = 0
        else:
            curv_mean = curv_max = 0

        # === Effective speed ratio (net displacement / total path) ===
        net_disp = haversine(lons[0], lats[0], lons[-1], lats[-1])
        total_path = sum(
            haversine(lons[i], lats[i], lons[i+1], lats[i+1])
            for i in range(n - 1)
        )
        eff_speed_ratio = net_disp / max(total_path, 1e-6)

        return {
            "flap_fraction": flap_frac,
            "glide_fraction": glide_frac,
            "n_mode_transitions": transitions,
            "mean_flap_duration": mean_flap_dur,
            "mean_glide_duration": mean_glide_dur,
            "alt_osc_freq": alt_osc_freq,
            "alt_osc_amplitude": alt_osc_amplitude,
            "dir_autocorr_lag5": dir_ac5 if np.isfinite(dir_ac5) else 0,
            "dir_autocorr_lag10": dir_ac10 if np.isfinite(dir_ac10) else 0,
            "curvature_mean": curv_mean,
            "curvature_max": curv_max,
            "effective_speed_ratio": min(eff_speed_ratio, 1.0),
        }
    except Exception:
        return defaults


# ── Tabular (non-trajectory) features ──────────────────────────────

def add_tabular_features(df_feat: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp, airspeed, altitude, and radar size features."""
    ts = pd.to_datetime(df_orig["timestamp_start_radar_utc"])
    te = pd.to_datetime(df_orig["timestamp_end_radar_utc"])
    hour = ts.dt.hour.values
    month = ts.dt.month.values

    df_feat["hour"] = hour
    df_feat["month"] = month
    df_feat["dayofweek"] = ts.dt.dayofweek.values
    df_feat["time_of_day"] = hour + ts.dt.minute.values / 60.0
    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12)
    df_feat["timestamp_duration"] = (te - ts).dt.total_seconds().values

    df_feat["airspeed"] = df_orig["airspeed"].values
    df_feat["min_z"] = df_orig["min_z"].values
    df_feat["max_z"] = df_orig["max_z"].values
    df_feat["z_range"] = df_orig["max_z"].values - df_orig["min_z"].values
    df_feat["z_mean"] = (df_orig["max_z"].values + df_orig["min_z"].values) / 2

    size_map = SIZE_MAP
    df_feat["radar_bird_size"] = df_orig["radar_bird_size"].map(size_map).values

    df_feat["airspeed_vs_ground"] = df_feat["airspeed"] / df_feat["avg_ground_speed"].clip(lower=0.01)

    return df_feat


# ── Targeted weak-class features (from E02's inline code) ─────────

def add_targeted_features(df_feat: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Hand-engineered features targeting weak classes (Pigeons, Clutter, etc.).

    These were in E02's inline code but missing from the modular pipeline.
    Each feature has a specific class hypothesis.
    """
    ts = pd.to_datetime(df_orig["timestamp_start_radar_utc"])
    hour = ts.dt.hour.values
    month = ts.dt.month.values

    # Pigeon signals (peak at 14:00 in October)
    df_feat["is_afternoon"] = (hour >= 13).astype(int)
    df_feat["is_october"] = (month == 10).astype(int)
    df_feat["oct_afternoon"] = ((month == 10) & (hour >= 13)).astype(int)
    df_feat["month_x_hour"] = month * 100 + hour

    # Seasonal (migration, clutter)
    df_feat["is_april"] = (month == 4).astype(int)
    df_feat["is_early_morning"] = (hour < 8).astype(int)
    df_feat["is_migration"] = ((month >= 9) & (month <= 11)).astype(int)
    df_feat["is_spring"] = ((month >= 3) & (month <= 5)).astype(int)
    df_feat["hour_bin_3h"] = (hour // 3).astype(int)

    # Size one-hot (helps trees split cleanly)
    size_map = SIZE_MAP
    size_val = df_orig["radar_bird_size"].map(size_map).values
    for name, val in [("small_bird", 0), ("medium", 1), ("large", 2), ("flock", 3)]:
        df_feat[f"is_{name}"] = (size_val == val).astype(int)

    # Speed bins
    df_feat["airspeed_high"] = (df_feat["airspeed"] > 17).astype(int)
    df_feat["airspeed_low"] = (df_feat["airspeed"] < 12).astype(int)

    # Duration bins (short = Clutter/Pigeons)
    df_feat["duration_short"] = (df_feat["duration"] < 25).astype(int)
    df_feat["duration_long"] = (df_feat["duration"] > 60).astype(int)

    # Size interaction terms
    df_feat["size_x_airspeed"] = df_feat["radar_bird_size"] * df_feat["airspeed"]
    df_feat["size_x_rcs"] = df_feat["radar_bird_size"] * df_feat["rcs_mean"]
    df_feat["size_x_alt"] = df_feat["radar_bird_size"] * df_feat["alt_mean"]

    return df_feat


# ── Weak-class targeted features ──────────────────────────────────

def extract_weakclass_features(hex_str: str, traj_time_str: str) -> dict:
    """Features specifically targeting weak classes.

    Cormorants: RCS stability, straight-line flight
    BoP: soaring index, circling, slow speed
    Waders: wingbeat proxy, altitude variability
    Geese: high altitude + large size + Oct migration
    """
    from scipy.signal import detrend

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {
        # RCS stability (Cormorants)
        "rcs_cv": 0, "rcs_autocorr_lag1": 0, "rcs_autocorr_lag3": 0,
        "rcs_stability": 0, "rcs_n_peaks_per_sec": 0,
        # Wingbeat proxy (Waders vs BoP)
        "rcs_zero_cross_rate": 0, "rcs_mean_cross_rate": 0,
        # Soaring index (BoP)
        "soaring_index": 0, "alt_gain_rate": 0, "slow_flight_frac": 0,
        # Trajectory shape (Cormorants = straight)
        "straightness": 0, "turn_angle_var": 0, "turn_angle_p90": 0,
        # Altitude dynamics (Waders, Geese)
        "alt_rate_mean": 0, "alt_rate_std": 0, "alt_rate_max": 0,
        "alt_accel_std": 0,
        # Speed profile (BoP)
        "speed_cv": 0, "speed_below_10_frac": 0, "speed_decel_max": 0,
        # RCS-altitude interaction (size consistency)
        "rcs_alt_corr": 0, "rcs_per_alt": 0,
    }

    if n < 4:
        return defaults

    try:
        duration = times[-1] - times[0]
        if duration < 0.1:
            return defaults
        dt = np.maximum(np.diff(times), 0.001)

        # ── RCS stability features (Cormorants have steady RCS) ──
        rcs_mean = np.mean(rcs)
        rcs_std = np.std(rcs)
        rcs_cv = rcs_std / max(abs(rcs_mean), 1e-10)

        # Autocorrelation of RCS
        rcs_centered = rcs - rcs_mean
        rcs_var = np.var(rcs) + 1e-10

        def safe_autocorr(x, lag, var):
            if len(x) <= lag:
                return 0
            return np.mean(x[:len(x)-lag] * x[lag:]) / var

        rcs_ac1 = safe_autocorr(rcs_centered, 1, rcs_var)
        rcs_ac3 = safe_autocorr(rcs_centered, min(3, n-1), rcs_var)

        # RCS stability = 1 - CV (higher = more stable)
        rcs_stability = 1.0 / (1.0 + rcs_cv)

        # ── Wingbeat proxy (RCS oscillation rate) ──
        # Zero-crossing rate of detrended RCS
        if n >= 6:
            rcs_detrended = detrend(rcs)
            zero_crossings = np.sum(np.diff(np.sign(rcs_detrended)) != 0)
            rcs_zc_rate = zero_crossings / (2 * max(duration, 0.01))
            # Mean-crossing rate (more robust)
            rcs_mc = np.sum(np.diff(np.sign(rcs - rcs_mean)) != 0)
            rcs_mc_rate = rcs_mc / (2 * max(duration, 0.01))
        else:
            rcs_zc_rate = 0
            rcs_mc_rate = 0

        # Number of RCS peaks per second
        if n >= 5:
            rcs_diff = np.diff(rcs)
            peaks = np.sum((rcs_diff[:-1] > 0) & (rcs_diff[1:] < 0))
            rcs_peaks_per_sec = peaks / max(duration, 0.01)
        else:
            rcs_peaks_per_sec = 0

        # ── Soaring index (BoP: altitude gain while slow) ──
        if n > 1:
            raw_dt = np.diff(times)
            dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                              for i in range(n-1)])
            raw_speeds = dists / dt
            # Filter out dt < 0.5s segments (same fix as core)
            valid_seg = raw_dt >= 0.5
            speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
            speed_mean = np.mean(speeds)

            alt_diffs = np.diff(alts)
            alt_gain = np.sum(alt_diffs[alt_diffs > 0])
            alt_gain_rate = alt_gain / max(duration, 0.01)

            # Soaring = gaining altitude while going slow
            if speed_mean > 0:
                soaring_index = alt_gain_rate / speed_mean
            else:
                soaring_index = 0

            # Fraction of time below 10 m/s (BoP soars slowly)
            slow_frac = np.mean(speeds < 10.0)

            # Speed variability
            speed_cv = np.std(speeds) / max(np.mean(speeds), 1e-10)

            # Max deceleration
            if len(speeds) > 1:
                speed_diff = np.diff(speeds)
                speed_decel_max = abs(np.min(speed_diff)) if len(speed_diff) > 0 else 0
            else:
                speed_decel_max = 0

            speed_below_10 = slow_frac
        else:
            speeds = np.array([0.0])
            soaring_index = alt_gain_rate = slow_frac = 0
            speed_cv = speed_below_10 = speed_decel_max = 0

        # ── Trajectory straightness (Cormorants fly very straight) ──
        total_dist = sum(haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                         for i in range(n-1)) if n > 1 else 0
        straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
        straightness = straight_dist / max(total_dist, 1e-6)
        straightness = min(straightness, 1.0)

        # Turn angle statistics
        if n > 2:
            bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
            turn_angles = np.abs(np.arctan2(np.sin(np.diff(bearings)),
                                             np.cos(np.diff(bearings))))
            turn_var = np.var(turn_angles)
            turn_p90 = np.percentile(turn_angles, 90)
        else:
            turn_var = turn_p90 = 0

        # ── Altitude dynamics (Waders: high variance, Geese: steady climb) ──
        if n > 1:
            alt_rate = np.diff(alts) / dt  # altitude change per second
            alt_rate_mean = np.mean(alt_rate)
            alt_rate_std = np.std(alt_rate)
            alt_rate_max = np.max(np.abs(alt_rate))

            if len(alt_rate) > 1:
                alt_accel = np.diff(alt_rate)
                alt_accel_std = np.std(alt_accel)
            else:
                alt_accel_std = 0
        else:
            alt_rate_mean = alt_rate_std = alt_rate_max = alt_accel_std = 0

        # ── RCS-altitude interaction ──
        if n > 2 and np.std(alts) > 0.01:
            rcs_alt_corr = np.corrcoef(rcs, alts)[0, 1]
            if not np.isfinite(rcs_alt_corr):
                rcs_alt_corr = 0
        else:
            rcs_alt_corr = 0

        mean_alt = np.mean(alts)
        rcs_per_alt = rcs_mean / max(abs(mean_alt), 1.0)

        return {
            "rcs_cv": rcs_cv,
            "rcs_autocorr_lag1": rcs_ac1 if np.isfinite(rcs_ac1) else 0,
            "rcs_autocorr_lag3": rcs_ac3 if np.isfinite(rcs_ac3) else 0,
            "rcs_stability": rcs_stability,
            "rcs_n_peaks_per_sec": rcs_peaks_per_sec,
            "rcs_zero_cross_rate": rcs_zc_rate,
            "rcs_mean_cross_rate": rcs_mc_rate,
            "soaring_index": soaring_index,
            "alt_gain_rate": alt_gain_rate,
            "slow_flight_frac": speed_below_10,
            "straightness": straightness,
            "turn_angle_var": turn_var,
            "turn_angle_p90": turn_p90,
            "alt_rate_mean": alt_rate_mean,
            "alt_rate_std": alt_rate_std,
            "alt_rate_max": alt_rate_max,
            "alt_accel_std": alt_accel_std,
            "speed_cv": speed_cv,
            "speed_below_10_frac": speed_below_10,
            "speed_decel_max": speed_decel_max,
            "rcs_alt_corr": rcs_alt_corr,
            "rcs_per_alt": rcs_per_alt,
        }
    except Exception:
        return defaults



def extract_flight_physics_features(hex_str: str, traj_time_str: str) -> dict:
    """Physics-based flight behavior features (E44).

    Season-invariant features based on aerodynamics and biomechanics:
    A. Cross-channel coupling (how altitude, RCS, speed, bearing relate)
    B. Biomechanics composites (bounding index, glide ratio, thermal score)
    C. Enhanced RCS modulation (modulation depth, periodicity, bimodality)
    D. 3D trajectory geometry (vertical/horizontal ratio, altitude trend)
    E. Multi-scale & complexity features (sinuosity ratio, permutation entropy)
    """
    from scipy.signal import detrend
    from scipy.stats import linregress

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {
        # A. Cross-channel coupling
        "phys_speed_alt_corr": 0, "phys_speed_rcs_corr": 0,
        "phys_bearing_rcs_corr": 0, "phys_alt_rate_rcs_corr": 0,
        "phys_speed_alt_rate_corr": 0, "phys_rcs_speed_interaction": 0,
        # B. Biomechanics composites
        "phys_bounding_index": 0, "phys_glide_ratio": 0,
        "phys_thermal_score": 0, "phys_wing_loading_proxy": 0,
        "phys_flap_regularity": 0, "phys_continuous_flap_score": 0,
        # C. Enhanced RCS modulation
        "phys_rcs_mod_depth": 0, "phys_rcs_periodicity_idx": 0,
        "phys_rcs_bimodality": 0, "phys_rcs_fluctuation_power": 0,
        # D. 3D trajectory geometry
        "phys_vert_horiz_ratio": 0, "phys_alt_trend_r2": 0,
        "phys_traj_aspect_ratio": 0, "phys_alt_entropy": 0,
        # E. Multi-scale & complexity
        "phys_sinuosity_ratio": 0, "phys_rcs_var_ratio": 0,
        "phys_speed_trend": 0, "phys_perm_entropy": 0,
    }

    if n < 8:
        return defaults

    try:
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults

        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)

        # Compute derived arrays
        dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                          for i in range(n - 1)])
        raw_speeds = dists / dt
        # Filter out dt < 0.5s segments (same fix as core)
        valid_seg = raw_dt >= 0.5
        if valid_seg.sum() >= 1:
            speeds = raw_speeds[valid_seg]
        else:
            speeds = raw_speeds
        alt_rate = np.diff(alts) / dt  # m/s vertical

        if n > 2:
            bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
            bearing_changes = np.abs(np.arctan2(
                np.sin(np.diff(bearings)), np.cos(np.diff(bearings))))
        else:
            bearing_changes = np.array([0.0])

        # Midpoints for correlation (align arrays to same length)
        rcs_mid = (rcs[:-1] + rcs[1:]) / 2  # length n-1
        alt_mid = (alts[:-1] + alts[1:]) / 2  # length n-1

        def safe_corr(a, b):
            """Pearson correlation, safe for constant arrays."""
            if len(a) < 3 or len(b) < 3:
                return 0.0
            if len(a) != len(b):
                m = min(len(a), len(b))
                a, b = a[:m], b[:m]
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            c = np.corrcoef(a, b)[0, 1]
            return c if np.isfinite(c) else 0.0

        # ═══ A. Cross-channel coupling ═══
        speed_alt_corr = safe_corr(speeds, alt_mid)
        speed_rcs_corr = safe_corr(speeds, rcs_mid)

        # bearing-RCS: use bearing_changes (len n-2) vs rcs (n-2 midpoints)
        if len(bearing_changes) >= 3:
            rcs_for_bearing = (rcs_mid[:-1] + rcs_mid[1:]) / 2 if len(rcs_mid) > 1 else rcs_mid[:len(bearing_changes)]
            bearing_rcs_corr = safe_corr(bearing_changes[:len(rcs_for_bearing)],
                                         rcs_for_bearing[:len(bearing_changes)])
        else:
            bearing_rcs_corr = 0.0

        alt_rate_rcs_corr = safe_corr(alt_rate, rcs_mid)
        speed_alt_rate_corr = safe_corr(speeds, alt_rate)
        rcs_speed_interaction = float(np.mean(rcs_mid * speeds))

        # ═══ B. Biomechanics composites ═══

        # Bounding index: altitude oscillation * alt-RCS correlation
        # Songbirds fold wings at altitude dips -> positive alt-RCS corr
        alt_detrended = alts - np.linspace(alts[0], alts[-1], n)
        alt_osc_amp = np.std(alt_detrended) * 2
        bounding_index = alt_osc_amp * max(alt_rate_rcs_corr, 0)

        # Glide ratio: horizontal distance / altitude loss during "glide" segments
        # Glide = low RCS variance windows
        win = max(3, n // 8)
        rcs_local_var = np.array([np.var(rcs[max(0, i-win):i+1]) for i in range(n)])
        glide_mask = rcs_local_var < np.median(rcs_local_var)
        if np.any(glide_mask[:-1]):
            glide_hdist = dists[glide_mask[:-1]].sum()
            alt_at_glide = alts[:-1][glide_mask[:-1]]
            alt_next_at_glide = alts[1:][glide_mask[:-1]]
            alt_loss = np.sum(np.maximum(alt_at_glide - alt_next_at_glide, 0))
            glide_ratio = glide_hdist / max(alt_loss, 1.0)
            glide_ratio = min(glide_ratio, 100.0)  # cap
        else:
            glide_ratio = 0.0

        # Thermal score: curvature * altitude gain rate / speed -> BoP
        if n > 3 and np.mean(speeds) > 0:
            dx = np.diff(lons) * 67000.0   # meters east
            dy = np.diff(lats) * 111000.0  # meters north
            ds = np.sqrt(dx**2 + dy**2) + 1e-10
            if len(dx) > 1:
                ddx = np.diff(dx)
                ddy = np.diff(dy)
                curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / (ds[:-1]**3 + 1e-10)
                curv_mean = np.mean(curvature)
            else:
                curv_mean = 0.0
            alt_gain = np.sum(np.maximum(np.diff(alts), 0))
            alt_gain_rate = alt_gain / max(duration, 0.01)
            thermal_score = curv_mean * alt_gain_rate / max(np.mean(speeds), 1e-6)
        else:
            thermal_score = 0.0

        # Wing loading proxy: speed^2 / |rcs_mean|
        # High wing loading birds (Cormorants, Ducks) fly faster for their size
        rcs_mean = np.mean(rcs)
        wing_loading_proxy = np.mean(speeds)**2 / max(abs(rcs_mean), 1.0)

        # Flap regularity: std of flap-segment durations
        flap_mask = ~glide_mask
        seg_durs = []
        seg_start = 0
        for i in range(1, n):
            if flap_mask[i] != flap_mask[seg_start]:
                if flap_mask[seg_start]:
                    seg_durs.append(times[i] - times[seg_start])
                seg_start = i
        if flap_mask[seg_start] and seg_start < n - 1:
            seg_durs.append(times[-1] - times[seg_start])
        flap_regularity = np.std(seg_durs) if len(seg_durs) > 1 else 0.0

        # Continuous flap score: flap_fraction * rcs_autocorrelation_lag1
        flap_fraction = flap_mask.mean()
        rcs_centered = rcs - rcs_mean
        rcs_var = np.var(rcs) + 1e-10
        rcs_ac1 = (np.mean(rcs_centered[:-1] * rcs_centered[1:]) / rcs_var
                   if n > 1 else 0.0)
        continuous_flap_score = flap_fraction * max(rcs_ac1, 0)

        # ═══ C. Enhanced RCS modulation ═══

        # Modulation depth (P90-P10, robust to outliers)
        rcs_mod_depth = np.percentile(rcs, 90) - np.percentile(rcs, 10)

        # Periodicity index: max autocorrelation at lags 2-10
        max_ac = 0.0
        for lag in range(2, min(11, n)):
            ac = np.mean(rcs_centered[:n-lag] * rcs_centered[lag:]) / rcs_var
            if np.isfinite(ac) and ac > max_ac:
                max_ac = ac
        rcs_periodicity_idx = max_ac

        # Bimodality: simplified dip test via bimodal coefficient
        # BC = (skewness^2 + 1) / kurtosis. BC > 5/9 suggests bimodality.
        rcs_std = np.std(rcs)
        if rcs_std > 1e-10 and n > 3:
            skew = float(np.mean(((rcs - rcs_mean) / rcs_std)**3))
            kurt = float(np.mean(((rcs - rcs_mean) / rcs_std)**4))
            rcs_bimodality = (skew**2 + 1) / max(kurt, 1e-10)
        else:
            rcs_bimodality = 0.0

        # Fluctuation power: variance of first-differenced RCS
        rcs_diff = np.diff(rcs)
        rcs_fluctuation_power = np.var(rcs_diff) if len(rcs_diff) > 0 else 0.0

        # ═══ D. 3D trajectory geometry ═══

        # Vertical/horizontal ratio
        mean_vert_speed = np.mean(np.abs(alt_rate))
        mean_horiz_speed = np.mean(speeds)
        vert_horiz_ratio = mean_vert_speed / max(mean_horiz_speed, 1e-6)

        # Altitude trend strength (R^2 of linear fit)
        if n > 2:
            t_arr = np.array(times) - times[0]
            slope, _, r_value, _, _ = linregress(t_arr, alts)
            alt_trend_r2 = r_value**2
        else:
            alt_trend_r2 = 0.0

        # Trajectory aspect ratio: altitude_range / horizontal_extent
        total_hdist = dists.sum()
        alt_range = np.ptp(alts)
        traj_aspect_ratio = alt_range / max(total_hdist, 1.0)

        # Altitude entropy: Shannon entropy of altitude histogram
        if alt_range > 0.1:
            n_bins = min(10, max(3, n // 5))
            counts, _ = np.histogram(alts, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            alt_entropy = -np.sum(probs * np.log(probs))
        else:
            alt_entropy = 0.0

        # ═══ E. Multi-scale & complexity ═══

        # Sinuosity ratio: first half vs second half
        mid = n // 2
        if mid > 2:
            h1_total = sum(haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                           for i in range(mid - 1))
            h1_straight = haversine(lons[0], lats[0], lons[mid-1], lats[mid-1])
            h2_total = sum(haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                           for i in range(mid, n - 1))
            h2_straight = haversine(lons[mid], lats[mid], lons[-1], lats[-1])
            sin1 = h1_total / max(h1_straight, 1e-6)
            sin2 = h2_total / max(h2_straight, 1e-6)
            sinuosity_ratio = sin1 / max(sin2, 1e-6)
            sinuosity_ratio = min(sinuosity_ratio, 10.0)  # cap
        else:
            sinuosity_ratio = 1.0

        # RCS variance ratio: first half vs second half
        if mid > 2:
            rv1 = np.var(rcs[:mid])
            rv2 = np.var(rcs[mid:])
            rcs_var_ratio = rv1 / max(rv2, 1e-10)
            rcs_var_ratio = min(rcs_var_ratio, 10.0)  # cap
        else:
            rcs_var_ratio = 1.0

        # Speed trend: slope of linear fit to speed
        if len(speeds) > 2:
            t_speed = np.cumsum(dt) - dt[0] / 2
            speed_trend = linregress(t_speed, speeds).slope
        else:
            speed_trend = 0.0

        # Permutation entropy of RCS (ordinal pattern complexity)
        # Order m=3: compare triplets of consecutive values
        if n >= 6:
            from itertools import permutations as _perms
            m = 3
            patterns = {}
            for i in range(n - m + 1):
                # Rank the m values
                window = rcs[i:i+m]
                pattern = tuple(np.argsort(window))
                patterns[pattern] = patterns.get(pattern, 0) + 1
            total_patterns = sum(patterns.values())
            probs = np.array(list(patterns.values())) / total_patterns
            perm_entropy = -np.sum(probs * np.log(probs))
            # Normalize by log(m!) so it's in [0, 1]
            import math as _math
            perm_entropy /= np.log(float(_math.factorial(m)))
        else:
            perm_entropy = 0.0

        return {
            # A. Cross-channel coupling
            "phys_speed_alt_corr": speed_alt_corr,
            "phys_speed_rcs_corr": speed_rcs_corr,
            "phys_bearing_rcs_corr": bearing_rcs_corr,
            "phys_alt_rate_rcs_corr": alt_rate_rcs_corr,
            "phys_speed_alt_rate_corr": speed_alt_rate_corr,
            "phys_rcs_speed_interaction": rcs_speed_interaction,
            # B. Biomechanics composites
            "phys_bounding_index": bounding_index,
            "phys_glide_ratio": glide_ratio,
            "phys_thermal_score": thermal_score,
            "phys_wing_loading_proxy": wing_loading_proxy,
            "phys_flap_regularity": flap_regularity,
            "phys_continuous_flap_score": continuous_flap_score,
            # C. Enhanced RCS modulation
            "phys_rcs_mod_depth": rcs_mod_depth,
            "phys_rcs_periodicity_idx": rcs_periodicity_idx,
            "phys_rcs_bimodality": rcs_bimodality,
            "phys_rcs_fluctuation_power": rcs_fluctuation_power,
            # D. 3D trajectory geometry
            "phys_vert_horiz_ratio": vert_horiz_ratio,
            "phys_alt_trend_r2": alt_trend_r2,
            "phys_traj_aspect_ratio": traj_aspect_ratio,
            "phys_alt_entropy": alt_entropy,
            # E. Multi-scale & complexity
            "phys_sinuosity_ratio": sinuosity_ratio,
            "phys_rcs_var_ratio": rcs_var_ratio,
            "phys_speed_trend": speed_trend,
            "phys_perm_entropy": perm_entropy,
        }
    except Exception:
        return defaults


def extract_absolute_wingbeat(hex_str: str, traj_time_str: str) -> dict:
    """Absolute Hz wingbeat frequency features (E16).

    Interpolates RCS to a fixed 20 Hz grid so that spectral peaks are in true
    biological Hz rather than normalised-by-Nyquist fractions.  This separates:
      - Geese/Cormorants  ~3-4 Hz
      - Ducks/Pigeons     ~5-8 Hz
      - Songbirds         ~10+ Hz / bursts
      - Gulls / BoP       <1 Hz or DC
    """
    from scipy.signal import spectrogram, welch
    from scipy.interpolate import interp1d as _interp1d

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)

    defaults = {
        "wb_hz_peak": 0.0, "wb_hz_stability": 0.0,
        "wb_band_0_1hz": 0.0, "wb_band_1_3hz": 0.0,
        "wb_band_3_6hz": 0.0, "wb_band_6_10hz": 0.0,
        "wb_energy_total": 0.0,
    }

    if len(pts) < 10:
        return defaults

    rcs = np.array([p[3] for p in pts])

    try:
        duration = times[-1] - times[0]
        if duration <= 0:
            return defaults
        fs = 20.0
        n_points = int(duration * fs)
        if n_points < 32:
            return defaults

        t_new = np.linspace(times[0], times[-1], n_points)
        f_interp = _interp1d(times, rcs, kind="linear", fill_value="extrapolate")
        rcs_ac = f_interp(t_new) - np.mean(f_interp(t_new))

        freqs, psd = welch(rcs_ac, fs=fs, nperseg=min(64, len(rcs_ac)))

        def _band(f_lo, f_hi):
            return np.sum(psd[(freqs >= f_lo) & (freqs < f_hi)])

        total_p = np.sum(psd) + 1e-9
        peak_hz = freqs[np.argmax(psd)]

        f_spec, _, sxx = spectrogram(rcs_ac, fs=fs, nperseg=32, noverlap=16)
        freq_stability = np.std(f_spec[np.argmax(sxx, axis=0)])

        return {
            "wb_hz_peak": float(peak_hz),
            "wb_hz_stability": float(freq_stability),
            "wb_band_0_1hz": float(_band(0, 1) / total_p),
            "wb_band_1_3hz": float(_band(1, 3) / total_p),
            "wb_band_3_6hz": float(_band(3, 6) / total_p),
            "wb_band_6_10hz": float(_band(6, 10) / total_p),
            "wb_energy_total": float(np.log1p(total_p)),
        }
    except Exception:
        return defaults


def extract_enhanced_bio_shape_features(hex_str: str, traj_time_str: str) -> dict:
    """Enhanced biological shape + RCS texture features (E68).

    Genuinely new features not covered by existing extractors:

    1. Turn direction consistency — ratio of net to total turning.
       High = consistent circling (BoP thermaling). Low = straight/erratic.
    2. Max sustained turn fraction — longest run of same-direction turns.
       Captures soaring spiral segments.
    3. Turn reversal rate — sign changes in bearing per second.
       High = bounding/erratic (Songbirds). Low = soaring or straight.
    4. RCS dominant autocorrelation lag — step lag where |AC| is largest.
       Proxy for dominant wingbeat period (Pigeons ~5-8 Hz lag).
    5. RCS flap regularity — CV of flap bout durations.
       Low CV = metronomic flapper (Pigeons/Waders). High = irregular.
    6. RCS glide/flap variance ratio — contrast between wing modes.
       High = clear flap-glide distinction (Gulls, BoP).
    7. RCS burst fraction — fraction of time with variance >2× median.
       High = bounding flight (Songbirds). Low = uniform flapping.
    8. Path loop fraction — how much of the trajectory doubles back.
       High = circling/hovering. Low = directed migration.
    """
    from scipy.stats import linregress

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)

    defaults = {
        "turn_dir_consistency": 0.0,
        "max_sustained_turn_frac": 0.0,
        "turn_reversal_rate": 0.0,
        "rcs_dominant_ac_lag": 0.0,
        "rcs_flap_regularity": 0.0,
        "rcs_glide_flap_var_ratio": 0.0,
        "rcs_burst_fraction": 0.0,
        "path_loop_fraction": 0.0,
    }

    if n < 6:
        return defaults

    try:
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults

        # ── 1. Turn direction consistency ──────────────────────────────
        if n > 3:
            bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
            b_changes = np.arctan2(np.sin(np.diff(bearings)),
                                   np.cos(np.diff(bearings)))
            total_turn = np.sum(np.abs(b_changes))
            net_turn = abs(np.sum(b_changes))
            turn_consistency = net_turn / max(total_turn, 1e-6)
            turn_consistency = min(turn_consistency, 1.0)

            # ── 2. Max sustained turn fraction ────────────────────────
            # Longest run of same-sign bearing changes / n
            signs = np.sign(b_changes)
            max_run = 0
            cur_run = 1
            for i in range(1, len(signs)):
                if signs[i] == signs[i - 1] and signs[i] != 0:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            max_sustained = max_run / max(len(b_changes), 1)

            # ── 3. Turn reversal rate ──────────────────────────────────
            sign_changes = np.sum(
                (signs[:-1] != signs[1:]) & (signs[:-1] != 0) & (signs[1:] != 0)
            )
            reversal_rate = sign_changes / max(duration, 0.01)
        else:
            b_changes = np.array([0.0])
            turn_consistency = 0.0
            max_sustained = 0.0
            reversal_rate = 0.0

        # ── 4. RCS dominant autocorrelation lag ───────────────────────
        rcs_centered = rcs - rcs.mean()
        rcs_var = np.var(rcs) + 1e-10
        best_ac = 0.0
        best_lag = 0.0
        for lag in range(1, min(n // 2, 20)):
            ac = np.mean(rcs_centered[:n - lag] * rcs_centered[lag:]) / rcs_var
            if abs(ac) > abs(best_ac):
                best_ac = ac
                best_lag = lag
        # Normalize lag to [0,1] (relative to half-length)
        rcs_dom_lag = best_lag / max(n // 2, 1)

        # ── 5 & 6: Flap/glide segmentation for regularity + contrast ──
        win = max(3, n // 8)
        rcs_local_var = np.array([
            np.var(rcs[max(0, i - win):i + 1]) for i in range(n)
        ])
        is_flap = rcs_local_var > np.median(rcs_local_var)

        # Flap bout durations
        flap_durs = []
        glide_durs = []
        seg_start = 0
        for i in range(1, n):
            if is_flap[i] != is_flap[seg_start]:
                seg_dur = times[i] - times[seg_start]
                (flap_durs if is_flap[seg_start] else glide_durs).append(seg_dur)
                seg_start = i
        seg_dur = times[-1] - times[seg_start]
        (flap_durs if is_flap[seg_start] else glide_durs).append(seg_dur)

        # Flap regularity: CV of bout durations
        if len(flap_durs) > 1:
            fdur_arr = np.array(flap_durs)
            rcs_flap_reg = np.std(fdur_arr) / max(np.mean(fdur_arr), 1e-6)
            rcs_flap_reg = min(rcs_flap_reg, 5.0)
        else:
            rcs_flap_reg = 0.0

        # Glide/flap variance contrast
        flap_var = rcs_local_var[is_flap].mean() if is_flap.any() else 1e-10
        glide_var = rcs_local_var[~is_flap].mean() if (~is_flap).any() else 1e-10
        gf_var_ratio = flap_var / max(glide_var, 1e-10)
        gf_var_ratio = min(gf_var_ratio, 20.0)

        # ── 7. RCS burst fraction ──────────────────────────────────────
        # Fraction of windows with variance > 2× median
        burst_threshold = 2.0 * np.median(rcs_local_var)
        burst_frac = float((rcs_local_var > burst_threshold).mean())

        # ── 8. Path loop fraction ──────────────────────────────────────
        # How often does the bird revisit its bounding box centre?
        # Measured as net displacement / total path × complement
        net_disp = haversine(lons[0], lats[0], lons[-1], lats[-1])
        total_path = sum(
            haversine(lons[i], lats[i], lons[i + 1], lats[i + 1])
            for i in range(n - 1)
        )
        # High loop fraction means the path loops back (net << total)
        loop_frac = 1.0 - min(net_disp / max(total_path, 1e-6), 1.0)

        return {
            "turn_dir_consistency": float(turn_consistency),
            "max_sustained_turn_frac": float(max_sustained),
            "turn_reversal_rate": float(reversal_rate),
            "rcs_dominant_ac_lag": float(rcs_dom_lag),
            "rcs_flap_regularity": float(rcs_flap_reg),
            "rcs_glide_flap_var_ratio": float(gf_var_ratio),
            "rcs_burst_fraction": float(burst_frac),
            "path_loop_fraction": float(loop_frac),
        }
    except Exception:
        return defaults


def extract_rcs_slope(hex_str: str, traj_time_str: str) -> dict:
    """Linear regression slope of RCS (dBm2) over normalised time [0,1].

    Separates species by how their radar cross-section changes during a track:
      Clutter: +1.5 dB (RCS increases as object approaches)
      Cormorants: +0.9 dB (large body, steady approach)
      Ducks: -2.0 dB (RCS decreases -- departing / altitude change)
      Others: ~flat

    Returns:
        rcs_slope: slope in dB per normalised time unit
    """
    from scipy.stats import linregress

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    if len(pts) < 4:
        return {"rcs_slope": 0.0}
    try:
        rcs = np.array([p[3] for p in pts], dtype=float)
        t = np.array(times, dtype=float)
        duration = t[-1] - t[0]
        if duration < 0.1:
            return {"rcs_slope": 0.0}
        t_norm = (t - t[0]) / duration  # normalise to [0, 1]
        slope = linregress(t_norm, rcs).slope
        return {"rcs_slope": float(slope) if np.isfinite(slope) else 0.0}
    except Exception:
        return {"rcs_slope": 0.0}


def extract_temporal_dynamics(hex_str: str, traj_time_str: str) -> dict:
    """Temporal dynamics features that capture flight behaviour over time.

    These features describe HOW measurements evolve along the track, not just
    their aggregate statistics. They are physics-based (MI ratio < 1.0) and
    specifically target the failure modes found in confusion analysis:

    heading_local_var:  Windowed heading consistency. BoP misclassified as Gulls
                        have heading_R=0.78 (looks straight) but heading_local_var=0.25
                        (actually erratic locally) vs Gulls 0.06. Uses 5-point windows.

    speed_consistency:  Ratio median/mean speed. Robust to outliers. Metronomic
                        flappers (Pigeons) ~1.0, variable flyers (BoP) < 0.8.

    speed_autocorr:     Lag-1 autocorrelation of speed. Misclassified Cormorants
                        have 0.23 (matching Gulls), correct ones -0.02.

    speed_slope:        Linear trend in speed over the track. Positive = accelerating
                        (departure), negative = decelerating (arrival). Normalised
                        by duration so units are (m/s)/s = m/s^2.

    alt_smoothness:     R^2 of linear fit to altitude. Cormorants/Geese fly level
                        (R^2 ~0.9), Songbirds bound (R^2 ~0.3).

    heading_change_rate: Standard deviation of bearing changes per second.
                        Captures turning aggressiveness independent of direction.

    rcs_trend:          Linear trend of RCS over time. Birds flying toward/away
                        from radar have systematic RCS changes (range effect).

    speed_variability:  IQR of speed / median speed. More robust than CV for
                        characterising speed consistency. Complements speed_consistency.
    """
    from scipy.stats import linregress

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    defaults = {
        "td_heading_local_var": 0.0,
        "td_speed_consistency": 1.0,
        "td_speed_autocorr": 0.0,
        "td_speed_slope": 0.0,
        "td_alt_smoothness": 0.0,
        "td_heading_change_rate": 0.0,
        "td_rcs_trend": 0.0,
        "td_speed_variability": 0.0,
    }

    if n < 6:
        return defaults

    try:
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs = np.array([p[3] for p in pts])
        duration = times[-1] - times[0]
        if duration < 0.5:
            return defaults

        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)
        valid_seg = raw_dt >= 0.5

        # Speeds (filtered)
        dlat = np.diff(lats) * 111000.0
        dlon = np.diff(lons) * 67000.0
        dists = np.sqrt(dlat**2 + dlon**2)
        raw_speeds = dists / dt
        speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds

        # Bearings (latitude-corrected)
        bearings = np.arctan2(dlat, dlon)

        # --- 1. Heading local variance ---
        # Compute heading_R in sliding windows of size 5, then take variance
        # of per-window heading_R. High variance = locally erratic even if
        # globally straight.
        win = 5
        if len(bearings) >= win:
            local_rs = []
            for j in range(len(bearings) - win + 1):
                b_win = bearings[j:j + win]
                r = np.sqrt(np.mean(np.cos(b_win))**2 + np.mean(np.sin(b_win))**2)
                local_rs.append(r)
            local_rs = np.array(local_rs)
            heading_local_var = float(np.var(local_rs))
        else:
            heading_local_var = 0.0

        # --- 2. Speed consistency (median / mean) ---
        speed_mean = np.mean(speeds)
        speed_consistency = float(np.median(speeds) / max(speed_mean, 1e-6))
        speed_consistency = min(speed_consistency, 2.0)

        # --- 3. Speed autocorrelation (lag-1) ---
        if len(speeds) > 3 and np.std(speeds) > 1e-6:
            sc = np.corrcoef(speeds[:-1], speeds[1:])
            speed_autocorr = float(sc[0, 1]) if np.isfinite(sc[0, 1]) else 0.0
        else:
            speed_autocorr = 0.0

        # --- 4. Speed slope (acceleration trend) ---
        if len(speeds) > 3:
            t_speed = np.linspace(0, 1, len(speeds))
            speed_slope = float(linregress(t_speed, speeds).slope)
            if not np.isfinite(speed_slope):
                speed_slope = 0.0
        else:
            speed_slope = 0.0

        # --- 5. Altitude smoothness (R^2 of linear fit) ---
        if n > 3:
            t_norm = np.linspace(0, 1, n)
            slope, intercept, r_value, _, _ = linregress(t_norm, alts)
            alt_smoothness = float(r_value**2)
        else:
            alt_smoothness = 0.0

        # --- 6. Heading change rate ---
        if len(bearings) > 1:
            b_changes = np.arctan2(np.sin(np.diff(bearings)),
                                   np.cos(np.diff(bearings)))
            heading_change_rate = float(np.std(b_changes) / max(np.mean(dt[1:]), 0.01))
            if not np.isfinite(heading_change_rate):
                heading_change_rate = 0.0
        else:
            heading_change_rate = 0.0

        # --- 7. RCS trend ---
        if n > 3:
            t_norm = np.linspace(0, 1, n)
            rcs_trend = float(linregress(t_norm, rcs).slope)
            if not np.isfinite(rcs_trend):
                rcs_trend = 0.0
        else:
            rcs_trend = 0.0

        # --- 8. Speed variability (robust IQR-based) ---
        if len(speeds) > 3:
            iqr = np.percentile(speeds, 75) - np.percentile(speeds, 25)
            speed_variability = float(iqr / max(np.median(speeds), 1e-6))
        else:
            speed_variability = 0.0

        return {
            "td_heading_local_var": heading_local_var,
            "td_speed_consistency": speed_consistency,
            "td_speed_autocorr": speed_autocorr,
            "td_speed_slope": speed_slope,
            "td_alt_smoothness": alt_smoothness,
            "td_heading_change_rate": heading_change_rate,
            "td_rcs_trend": rcs_trend,
            "td_speed_variability": speed_variability,
        }
    except Exception:
        return defaults


def extract_trajectory_separators(hex_str: str, traj_time_str: str) -> dict:
    """Features with strong class separation power from trajectory analysis.

    heading_R:             BoP d=-1.28, Clutter AUC=0.91 (circling vs straight)
    rcs_spectral_entropy:  Pigeons d=-1.42 (periodic wingbeat in RCS)
    speed_autocorr:        BoP d=+0.45 (consistent slow speed)
    alt_ascending_frac:    Waders/Pigeons d=+0.75 (climbing flight)
    alt_descending_frac:   Cormorants d=-0.54 (less descending than Gulls)
    alt_flat_frac:         Cormorants d=+0.73 (level flight)
    soaring_frac:          BoP d=+0.90 (circling + climbing)
    rcs_burst_frac:        RCS dynamics (sudden changes)
    rcs_smooth_frac:       RCS dynamics (stable periods)
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    result = {
        "heading_R": 1.0,
        "rcs_spectral_entropy": 0.0,
        "speed_autocorr": 0.0,
        "alt_ascending_frac": 0.0,
        "alt_descending_frac": 0.0,
        "alt_flat_frac": 1.0,
        "soaring_frac": 0.0,
        "rcs_burst_frac": 0.0,
        "rcs_smooth_frac": 1.0,
    }

    if n < 4:
        return result

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    raw_dt = np.diff(times)
    dt = np.maximum(raw_dt, 0.01)
    valid_seg = raw_dt >= 0.5  # filter glitched timestamps

    # Bearing and heading_R (corrected for latitude)
    dlat = np.diff(lats) * 111000.0
    dlon = np.diff(lons) * 67000.0
    bearings = np.arctan2(dlat, dlon)
    if len(bearings) >= 2:
        cos_b = np.cos(bearings)
        sin_b = np.sin(bearings)
        result["heading_R"] = float(np.sqrt(np.mean(cos_b)**2 + np.mean(sin_b)**2))

    # RCS spectral entropy (wingbeat periodicity)
    if n > 8:
        rcs_centered = rcs - np.mean(rcs)
        fft_mag = np.abs(np.fft.rfft(rcs_centered))
        if len(fft_mag) > 1:
            power = fft_mag[1:] ** 2
            total_power = power.sum()
            if total_power > 1e-10:
                p_norm = power / total_power
                result["rcs_spectral_entropy"] = float(
                    -np.sum(p_norm * np.log(p_norm + 1e-10))
                )

    # Speed autocorrelation
    dists = np.sqrt(dlat**2 + dlon**2)
    raw_speeds = dists / dt
    # Use filtered speeds for autocorrelation (avoid dt glitch contamination)
    clean_speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
    if len(clean_speeds) > 3:
        sc = np.corrcoef(clean_speeds[:-1], clean_speeds[1:])
        if np.isfinite(sc[0, 1]):
            result["speed_autocorr"] = float(sc[0, 1])

    # Altitude dynamics
    alt_diffs = np.diff(alts)
    if len(alt_diffs) > 0:
        result["alt_ascending_frac"] = float(np.mean(alt_diffs > 1))
        result["alt_descending_frac"] = float(np.mean(alt_diffs < -1))
        result["alt_flat_frac"] = float(np.mean(np.abs(alt_diffs) <= 1))

    # Soaring fraction (turning + climbing)
    if len(bearings) > 2 and len(alt_diffs) > 1:
        dbearing = np.abs(np.diff(bearings))
        dbearing = np.minimum(dbearing, 2 * np.pi - dbearing)
        min_len = min(len(dbearing), len(alt_diffs) - 1)
        if min_len > 0:
            soaring = (dbearing[:min_len] > 0.3) & (alt_diffs[1:min_len + 1] > 0.5)
            result["soaring_frac"] = float(np.mean(soaring))

    # RCS dynamics
    rcs_diff = np.abs(np.diff(rcs))
    if len(rcs_diff) > 0:
        result["rcs_burst_frac"] = float(np.mean(rcs_diff > 5))
        result["rcs_smooth_frac"] = float(np.mean(rcs_diff < 1))

    return result


def extract_raw_signal_features(hex_str: str, traj_time_str: str) -> dict:
    """Deep raw signal features capturing flight behaviour from radar data.

    These features describe physical behaviour only -- no temporal or
    weather information -- so they generalise across unseen months.

    heading_entropy:      direction diversity (8-sector Shannon entropy)
    turn_persistence:     longest consecutive same-direction turning / track length
    alt_profile_var:      altitude deviation from straight-line (undulation)
    rcs_mod_median:       median RCS modulation depth in 6-point window (body size proxy)
    rcs_mod_consistency:  CV of RCS modulation (consistent = regular wingbeat)
    speed_alt_coupling:   correlation between speed and vertical rate (soaring signature)
    speed_jerk_std:       std of speed acceleration (flight smoothness)
    vert_speed_skew:      skewness of vertical speed distribution
    vert_speed_kurt:      kurtosis of vertical speed distribution
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    result = {
        "heading_entropy": 0.0,
        "turn_persistence": 0.0,
        "alt_profile_var": 0.0,
        "rcs_mod_median": 0.0,
        "rcs_mod_consistency": 0.0,
        "speed_alt_coupling": 0.0,
        "speed_jerk_std": 0.0,
        "vert_speed_skew": 0.0,
        "vert_speed_kurt": 0.0,
    }

    if n < 5:
        return result

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    raw_dt = np.diff(times)
    dt = np.maximum(raw_dt, 0.01)
    valid_seg = raw_dt >= 0.5

    dlat = np.diff(lats) * 111000.0
    dlon = np.diff(lons) * 67000.0
    dists = np.sqrt(dlat**2 + dlon**2)
    raw_speeds = dists / dt
    speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
    bearings = np.arctan2(dlat, dlon)

    # heading_entropy: bin bearings into 8 sectors, Shannon entropy
    sectors = ((bearings + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    sector_counts = np.bincount(sectors, minlength=8)
    sector_probs = sector_counts / sector_counts.sum()
    result["heading_entropy"] = float(
        -np.sum(sector_probs * np.log(sector_probs + 1e-10))
    )

    # turn_persistence: longest run of consistent turn direction
    if len(bearings) > 2:
        dbearing = np.diff(bearings)
        dbearing = np.arctan2(np.sin(dbearing), np.cos(dbearing))
        same_dir = np.diff(np.sign(dbearing)) == 0
        if len(same_dir) > 0:
            boundaries = np.where(
                np.concatenate(([True], ~same_dir, [True]))
            )[0]
            runs = np.diff(boundaries)
            result["turn_persistence"] = float(runs.max() / len(same_dir))

    # alt_profile_var: deviation of altitude from straight-line path
    alt_line = np.linspace(alts[0], alts[-1], n)
    result["alt_profile_var"] = float(np.var(alts - alt_line))

    # rcs_mod_median / rcs_mod_consistency: rolling-window RCS modulation
    win = min(6, n)
    if win >= 3:
        rcs_mod = np.array([
            rcs[j:j + win].max() - rcs[j:j + win].min()
            for j in range(n - win + 1)
        ])
        result["rcs_mod_median"] = float(np.median(rcs_mod))
        mod_mean = np.mean(rcs_mod)
        if mod_mean > 1e-10:
            result["rcs_mod_consistency"] = float(np.std(rcs_mod) / mod_mean)

    # speed_alt_coupling: speed-vertical_rate correlation (soaring = slow + climbing)
    vert_speed = np.diff(alts) / dt
    min_l = min(len(speeds), len(vert_speed))
    if min_l > 2:
        sc = np.corrcoef(speeds[:min_l], vert_speed[:min_l])
        if np.isfinite(sc[0, 1]):
            result["speed_alt_coupling"] = float(sc[0, 1])

    # speed_jerk_std: variability of acceleration
    if len(speeds) > 2:
        dt_valid = dt[valid_seg] if valid_seg.sum() >= 1 else dt
        dt_jerk = dt_valid[1:] if len(dt_valid) > 1 else np.array([1.0])
        if len(dt_jerk) == len(speeds) - 1:
            speed_jerk = np.diff(speeds) / np.maximum(dt_jerk, 0.01)
        else:
            speed_jerk = np.diff(speeds)  # fallback without dt normalization
        result["speed_jerk_std"] = float(np.std(speed_jerk))

    # vert_speed_skew / vert_speed_kurt: shape of vertical speed distribution
    if len(vert_speed) > 3:
        vs_mean = np.mean(vert_speed)
        vs_std = np.std(vert_speed)
        if vs_std > 1e-10:
            vs_z = (vert_speed - vs_mean) / vs_std
            result["vert_speed_skew"] = float(np.mean(vs_z**3))
            result["vert_speed_kurt"] = float(np.mean(vs_z**4) - 3.0)

    return result


def extract_micro_pattern_features(hex_str: str, traj_time_str: str) -> dict:
    """Per-segment micro-pattern features from raw radar signals.

    These capture within-track behavioural dynamics that whole-track
    averages miss -- critical for separating hard cases from Gulls.

    speed_dip_frac:       BoP d=+1.81 correct vs wrong -- soaring speed drops
    speed_surge_frac:     Clutter d=+1.91 -- erratic speed spikes
    turn_frac_15deg:      Clutter d=+1.88, BoP d=+0.76 -- sharp turn fraction
    high_speed_low_alt:   Ducks d=+1.54 -- fast at low altitude
    speed_q_ratio:        BoP d=+0.56, Clutter d=+0.89 -- speed p90/p10
    max_alt_run:          Clutter d=+1.12, Pigeons d=+0.76 -- monotonic alt run
    vert_osc_freq:        Clutter d=-0.79, Cormorants d=+0.50 -- alt oscillation
    rcs_periodicity:      Geese d=+0.55 -- wingbeat in RCS ACF
    glide_frac:           Pigeons d=-0.50 -- fraction in glide mode
    powered_frac:         Cormorants correct d=+0.76 -- fraction in powered flight
    rcs_turn_diff:        Cormorants d=-0.58 correct/wrong -- RCS change in turns
    alt_stable_frac:      Ducks correct d=+0.69 -- fraction near mean altitude
    bounding_rate:        speed acceleration alternation (songbird wingbeat)
    turn_speed_corr:      do they slow in turns? (BoP: yes)
    alt_r2:               altitude linearity (ducks correct: low)
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    result = {
        "mp_speed_dip_frac": 0.0,
        "mp_speed_surge_frac": 0.0,
        "mp_turn_frac_15deg": 0.0,
        "mp_high_speed_low_alt": 0.0,
        "mp_speed_q_ratio": 1.0,
        "mp_max_alt_run": 0.0,
        "mp_vert_osc_freq": 0.0,
        "mp_rcs_periodicity": 0.0,
        "mp_glide_frac": 0.0,
        "mp_powered_frac": 0.0,
        "mp_rcs_turn_diff": 0.0,
        "mp_alt_stable_frac": 0.0,
        "mp_bounding_rate": 0.0,
        "mp_turn_speed_corr": 0.0,
        "mp_alt_r2": 0.0,
        "mp_rcs_local_range_cv": 0.0,
    }

    if n < 5:
        return result

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    raw_dt = np.diff(times)
    dt = np.maximum(raw_dt, 0.01)
    valid_seg = raw_dt >= 0.5

    dlat = np.diff(lats) * 111000.0
    dlon = np.diff(lons) * 67000.0
    dists = np.sqrt(dlat**2 + dlon**2)
    raw_speeds = dists / dt
    speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
    bearings = np.arctan2(dlat, dlon)
    vert_speed = np.diff(alts) / dt
    duration = times[-1] - times[0]

    # Speed dip/surge fraction (local anomalies vs 5-point running average)
    if len(speeds) > 5:
        local_avg = np.convolve(speeds, np.ones(5) / 5, mode="same")
        result["mp_speed_dip_frac"] = float(np.mean(speeds < 0.7 * local_avg))
        result["mp_speed_surge_frac"] = float(np.mean(speeds > 1.3 * local_avg))

    # Turn fraction (segments with >15 deg bearing change)
    if len(bearings) > 1:
        dbearing = np.diff(bearings)
        dbearing = np.arctan2(np.sin(dbearing), np.cos(dbearing))
        result["mp_turn_frac_15deg"] = float(
            np.mean(np.abs(dbearing) > np.radians(15))
        )
    else:
        dbearing = np.array([0.0])

    # High speed at low altitude (duck/cormorant signature)
    # Use raw speeds (aligned with alts[1:]) for per-segment features
    result["mp_high_speed_low_alt"] = float(
        np.mean((raw_speeds > 15) & (alts[1:] < 30))
    )

    # Speed quantile ratio (p90/p10)
    if len(speeds) > 3:
        p10 = np.percentile(speeds, 10)
        p90 = np.percentile(speeds, 90)
        result["mp_speed_q_ratio"] = float(p90 / (p10 + 0.1))

    # Max consecutive altitude direction run
    if len(vert_speed) > 2:
        vs_sign = np.sign(vert_speed)
        boundaries = np.where(np.diff(vs_sign) != 0)[0]
        if len(boundaries) > 0:
            runs = np.diff(np.concatenate([[0], boundaries, [len(vs_sign)]]))
            result["mp_max_alt_run"] = float(runs.max() / len(vs_sign))
        else:
            result["mp_max_alt_run"] = 1.0

    # Vertical oscillation frequency
    if len(vert_speed) > 3:
        vs_sign = np.sign(vert_speed)
        vs_changes = np.sum(np.diff(vs_sign) != 0)
        result["mp_vert_osc_freq"] = float(vs_changes / (duration + 1e-10))

    # RCS periodicity (max autocorrelation at lag 2-5)
    if n > 8:
        rcs_c = rcs - rcs.mean()
        acf_vals = []
        for lag in range(2, min(6, n // 2)):
            c = np.corrcoef(rcs_c[:-lag], rcs_c[lag:])[0, 1]
            if np.isfinite(c):
                acf_vals.append(c)
        if acf_vals:
            result["mp_rcs_periodicity"] = float(max(acf_vals))

    # Glide/powered fraction
    min_l = min(len(speeds) - 1, len(vert_speed) - 1)
    if min_l > 0:
        speed_dec = np.diff(speeds[: min_l + 1]) < 0
        alt_dec = vert_speed[: min_l + 1] < -0.5
        result["mp_glide_frac"] = float(
            np.mean(speed_dec[:min_l] & alt_dec[:min_l])
        )
        speed_inc = np.diff(speeds[: min_l + 1]) >= 0
        alt_up = vert_speed[: min_l + 1] >= -0.5
        result["mp_powered_frac"] = float(
            np.mean(speed_inc[:min_l] & alt_up[:min_l])
        )

    # RCS difference during turns vs straight (body aspect angle)
    if len(dbearing) > 2:
        turning = np.abs(dbearing) > np.radians(10)
        straight = ~turning
        min_l2 = min(len(turning), n - 2)
        rcs_seg = rcs[1 : min_l2 + 1]
        if turning[:min_l2].sum() > 0 and straight[:min_l2].sum() > 0:
            result["mp_rcs_turn_diff"] = float(
                np.mean(rcs_seg[turning[:min_l2]])
                - np.mean(rcs_seg[straight[:min_l2]])
            )

    # Altitude stability fraction (within 5m of mean)
    result["mp_alt_stable_frac"] = float(
        np.mean(np.abs(alts - np.mean(alts)) < 5)
    )

    # Bounding rate (speed acceleration sign alternation)
    if len(speeds) > 4:
        speed_sign = np.sign(np.diff(speeds))
        result["mp_bounding_rate"] = float(np.mean(np.diff(speed_sign) != 0))

    # Turn-speed correlation (slow down in turns?)
    if len(speeds) > 2 and len(dbearing) > 0:
        min_l3 = min(len(speeds) - 1, len(dbearing))
        c = np.corrcoef(
            np.abs(dbearing[:min_l3]), speeds[1 : min_l3 + 1]
        )[0, 1]
        if np.isfinite(c):
            result["mp_turn_speed_corr"] = float(c)

    # Altitude R^2 (linearity of altitude profile)
    if n > 3:
        t_norm = np.linspace(0, 1, n)
        coeffs = np.polyfit(t_norm, alts, 1)
        alt_pred = np.polyval(coeffs, t_norm)
        ss_res = np.sum((alts - alt_pred) ** 2)
        ss_tot = np.sum((alts - np.mean(alts)) ** 2)
        result["mp_alt_r2"] = float(max(0, 1 - ss_res / (ss_tot + 1e-10)))

    # RCS local range CV (consistency of RCS modulation across segments)
    if n > 6:
        seg_size = 5
        rcs_local_ranges = [
            np.max(rcs[j : j + seg_size]) - np.min(rcs[j : j + seg_size])
            for j in range(0, n - seg_size, seg_size)
        ]
        if rcs_local_ranges:
            mean_r = np.mean(rcs_local_ranges)
            if mean_r > 1e-10:
                result["mp_rcs_local_range_cv"] = float(
                    np.std(rcs_local_ranges) / mean_r
                )

    return result


def extract_radar_physics_features(hex_str: str, traj_time_str: str) -> dict:
    """Novel features derived from radar-bird physics first principles.

    1. Detection gap rate: fraction of missed radar scans. Body size proxy
       independent of RCS — smaller birds have lower detection probability.
    2. Wing loading proxy: speed^2 / RCS_linear. High-wing-loading birds
       (Cormorants, Pigeons) fly fast for their body size.
    3. RCS-speed cross-correlation: how RCS changes relate to speed changes.
       Encodes wing morphology (flapper vs glider).
    4. 3D soaring score: simultaneous turning + climbing. BoP thermal signature.
    5. Speed persistence: autocorrelation of speed. Pigeons are metronomic.
    6. RCS-altitude coupling: correlation between RCS and altitude changes.
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    defaults = {
        "rp_gap_fraction": 0.0,
        "rp_max_gap_s": 0.0,
        "rp_n_gaps": 0,
        "rp_wing_loading_proxy": 0.0,
        "rp_rcs_speed_corr": 0.0,
        "rp_soaring_score": 0.0,
        "rp_speed_persistence": 0.0,
        "rp_rcs_alt_corr": 0.0,
        "rp_power_proxy": 0.0,
    }

    if n < 4:
        return defaults

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    raw_dt = np.diff(times)
    dt = np.maximum(raw_dt, 0.001)
    median_dt = np.median(dt)
    valid_seg = raw_dt >= 0.5

    # --- 1. Detection gap features ---
    # A "gap" is when dt > 1.5x the median step (missed scan)
    gap_threshold = max(median_dt * 1.5, 1.5)
    is_gap = dt > gap_threshold
    gap_fraction = float(is_gap.mean())
    max_gap = float(dt.max())
    n_gaps = int(is_gap.sum())

    # --- 2. Wing loading proxy ---
    # Wing loading ∝ speed^2. RCS_linear ∝ body cross-section.
    # speed^2 / RCS_linear should separate high-WL (Cormorants, Pigeons)
    # from low-WL (Gulls, BoP soaring).
    dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                      for i in range(n-1)])
    raw_speeds = dists / dt
    speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
    mean_speed = float(np.mean(speeds))
    rcs_linear = 10.0 ** (np.mean(rcs) / 10.0)  # dBm2 → linear
    wing_loading_proxy = mean_speed**2 / max(rcs_linear, 1e-10)

    # --- 3. RCS-speed cross-correlation ---
    # At each measurement, correlate RCS with instantaneous speed.
    # Use midpoint RCS for each segment, filtered to valid segments.
    rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])
    rcs_mid_clean = rcs_mid[valid_seg] if valid_seg.sum() >= 1 else rcs_mid
    if len(speeds) > 3 and len(rcs_mid_clean) == len(speeds) and np.std(speeds) > 1e-6 and np.std(rcs_mid_clean) > 1e-6:
        rcs_speed_corr = float(np.corrcoef(rcs_mid_clean, speeds)[0, 1])
        if not np.isfinite(rcs_speed_corr):
            rcs_speed_corr = 0.0
    else:
        rcs_speed_corr = 0.0

    # --- 4. 3D soaring score ---
    # Fraction of track where bird is BOTH turning AND climbing.
    if n > 3:
        bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
        if len(bearings) > 1:
            bearing_changes = np.abs(
                np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
            )
            alt_rates = np.diff(alts[:-1]) / dt[:-1]
            # Soaring = high turning (> 0.1 rad/step) AND climbing (> 0.5 m/s)
            min_len = min(len(bearing_changes), len(alt_rates))
            is_soaring = (bearing_changes[:min_len] > 0.1) & (alt_rates[:min_len] > 0.5)
            soaring_score = float(is_soaring.mean())
        else:
            soaring_score = 0.0
    else:
        soaring_score = 0.0

    # --- 5. Speed persistence (lag-1 autocorrelation of speed) ---
    if len(speeds) > 5 and np.std(speeds) > 1e-6:
        sp_centered = speeds - np.mean(speeds)
        sp_var = np.var(speeds)
        speed_persistence = float(
            np.mean(sp_centered[:-1] * sp_centered[1:]) / max(sp_var, 1e-10)
        )
        if not np.isfinite(speed_persistence):
            speed_persistence = 0.0
    else:
        speed_persistence = 0.0

    # --- 6. RCS-altitude coupling ---
    if n > 3 and np.std(rcs) > 1e-6 and np.std(alts) > 1e-6:
        rcs_alt_corr = float(np.corrcoef(rcs, alts)[0, 1])
        if not np.isfinite(rcs_alt_corr):
            rcs_alt_corr = 0.0
    else:
        rcs_alt_corr = 0.0

    # --- 7. Flight power proxy ---
    # Proportional to drag power: ∝ speed^3 for aerodynamic drag.
    # Adjusted by climb rate for potential energy cost.
    duration = times[-1] - times[0]
    if duration > 0:
        alt_gain = float(np.sum(np.maximum(np.diff(alts), 0)))
        power_proxy = mean_speed**3 + 9.81 * alt_gain / max(duration, 1)
    else:
        power_proxy = 0.0

    return {
        "rp_gap_fraction": gap_fraction,
        "rp_max_gap_s": max_gap,
        "rp_n_gaps": n_gaps,
        "rp_wing_loading_proxy": float(wing_loading_proxy),
        "rp_rcs_speed_corr": rcs_speed_corr,
        "rp_soaring_score": soaring_score,
        "rp_speed_persistence": speed_persistence,
        "rp_rcs_alt_corr": rcs_alt_corr,
        "rp_power_proxy": power_proxy,
    }


def extract_linearity_features(hex_str: str, traj_time_str: str) -> dict:
    """Trajectory linearity features (E15/E16).

    Measures how closely the path follows a straight line – low deviation
    is characteristic of Cormorants flying on a fixed bearing.
    """
    pts = parse_ewkb_4d(hex_str)
    if len(pts) < 3:
        return {"lin_error_mean": 0.0}
    coords = np.array([p[:2] for p in pts])
    start, end = coords[0], coords[-1]
    vec_line = end - start
    len_line = np.linalg.norm(vec_line)
    if len_line < 1e-6:
        return {"lin_error_mean": 0.0}
    vec_line_norm = vec_line / len_line
    vec_pts = coords - start
    proj = np.dot(vec_pts, vec_line_norm)
    closest = np.outer(proj, vec_line_norm)
    errors = np.linalg.norm(vec_pts - closest, axis=1)
    return {"lin_error_mean": float(np.mean(errors / (len_line + 1e-6)))}


def extract_path_signature_features(hex_str: str, traj_time_str: str,
                                     depth: int = 2, lead_lag: bool = True) -> dict:
    """Path signature features (E45).

    Signatures are mathematically invariant to time reparameterization,
    meaning a bird's signature is the same regardless of sampling rate
    or calendar time. This directly addresses our temporal shift problem.

    Channels: altitude (normalized), RCS, speed, bearing_change.
    Optional lead-lag augmentation doubles channels for richer representation.
    """
    import esig

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    # Determine expected output size for naming
    n_channels = 4
    if lead_lag:
        n_channels *= 2
    # Signature length = sum of n_channels^k for k=1..depth
    sig_len = sum(n_channels**k for k in range(1, depth + 1))
    prefix = f"sig_d{depth}_ll" if lead_lag else f"sig_d{depth}"
    defaults = {f"{prefix}_{i}": 0.0 for i in range(sig_len)}

    if n < 8:
        return defaults

    try:
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs = np.array([p[3] for p in pts])
        dt = np.maximum(np.diff(times), 0.001)
        duration = times[-1] - times[0]

        # Channel 1: normalized altitude (0 to 1)
        alt_range = np.ptp(alts)
        if alt_range > 0.1:
            alt_norm = (alts - alts.min()) / alt_range
        else:
            alt_norm = np.zeros(n)

        # Channel 2: RCS (standardized)
        rcs_std = np.std(rcs)
        if rcs_std > 1e-10:
            rcs_norm = (rcs - np.mean(rcs)) / rcs_std
        else:
            rcs_norm = np.zeros(n)

        # Channel 3: speed (from haversine, normalized by mean)
        dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                          for i in range(n - 1)])
        raw_dt = np.diff(times)
        raw_speeds = dists / dt
        # Filter dt < 0.5s for mean computation (avoid outlier inflation)
        valid_seg = raw_dt >= 0.5
        speeds = raw_speeds  # keep all for per-point normalization
        clean_speeds = raw_speeds[valid_seg] if valid_seg.sum() >= 1 else raw_speeds
        speed_mean = np.mean(clean_speeds) if np.mean(clean_speeds) > 1e-6 else 1.0
        speed_norm = np.concatenate([[speeds[0] / speed_mean], speeds / speed_mean])

        # Channel 4: cumulative bearing change (normalized)
        if n > 2:
            bearings = np.arctan2(np.diff(lats) * 111000.0, np.diff(lons) * 67000.0)
            b_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
            cum_bearing = np.concatenate([[0, 0], np.cumsum(b_changes)])
            max_b = np.max(np.abs(cum_bearing)) if np.max(np.abs(cum_bearing)) > 1e-10 else 1.0
            bearing_norm = cum_bearing / max_b
        else:
            bearing_norm = np.zeros(n)

        # Assemble path: (n, 4)
        path = np.column_stack([alt_norm, rcs_norm, speed_norm, bearing_norm])

        # Lead-lag augmentation
        if lead_lag and n > 1:
            ll = np.zeros((2 * (n - 1), 8))
            for i in range(n - 1):
                ll[2 * i] = np.concatenate([path[i], path[i]])
                ll[2 * i + 1] = np.concatenate([path[i], path[i + 1]])
            path = ll

        # Compute signature
        sig = esig.stream2sig(path, depth)

        result = {}
        for i, val in enumerate(sig):
            result[f"{prefix}_{i}"] = float(val) if np.isfinite(val) else 0.0

        return result
    except Exception:
        return defaults

def add_weakclass_tabular(df_feat: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Tabular features targeting weak classes (migration timing, size interactions)."""
    ts = pd.to_datetime(df_orig["timestamp_start_radar_utc"])
    month = ts.dt.month.values
    hour = ts.dt.hour.values

    size_map = SIZE_MAP
    size_val = df_orig["radar_bird_size"].map(size_map).values

    # Migration timing (Geese, Waders peak Oct-Nov)
    df_feat["is_oct_nov"] = ((month == 10) | (month == 11)).astype(int)
    df_feat["migration_alt"] = df_feat["is_oct_nov"] * df_feat["alt_mean"]
    df_feat["migration_speed"] = df_feat["is_oct_nov"] * df_feat["airspeed"]

    # Size-altitude interaction (Geese: Large/Flock + high alt)
    df_feat["large_high_alt"] = ((size_val >= 2) & (df_feat["alt_mean"] > 100)).astype(int)
    df_feat["flock_indicator"] = (size_val == 3).astype(int)
    df_feat["size_alt_interaction"] = size_val * df_feat["alt_mean"]

    # Solitary bird indicator (BoP: always alone, Small/Medium)
    df_feat["solitary_slow"] = ((size_val <= 1) & (df_feat["airspeed"] < 13)).astype(int)

    # Night flight (some species are nocturnal migrants)
    df_feat["is_night"] = ((hour < 6) | (hour > 20)).astype(int)
    df_feat["night_high_alt"] = df_feat["is_night"] * df_feat["alt_mean"]

    # RCS-size consistency (Cormorants: Large bird but moderate RCS)
    df_feat["rcs_for_size"] = df_feat["rcs_mean"] - (size_val * 3 - 30)

    return df_feat


def add_external_prior_features(
    df_feat: pd.DataFrame,
    class_priors: dict[str, dict[str, float]],
    include_morph: bool = True,
    include_flight: bool = True,
) -> pd.DataFrame:
    """Add per-class prior-match features from external ornithology data.

    These features compare observed track attributes to class-level priors
    (speed, RCS proxy from mass, wing loading, wingbeat proxy).
    """
    if not include_morph and not include_flight:
        return df_feat

    required_cols = {"airspeed", "rcs_mean"}
    missing = [c for c in sorted(required_cols) if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required columns for external priors: {missing}")

    n = len(df_feat)
    airspeed = df_feat["airspeed"].values.astype(float)
    rcs_mean = df_feat["rcs_mean"].values.astype(float)
    radar_size = (
        df_feat["radar_bird_size"].values.astype(float)
        if "radar_bird_size" in df_feat.columns
        else np.zeros(n, dtype=float)
    )
    rcs_peak_freq = (
        df_feat["rcs_peak_freq"].values.astype(float)
        if "rcs_peak_freq" in df_feat.columns
        else np.zeros(n, dtype=float)
    )

    speed_matrix = []
    morph_matrix = []
    joint_matrix = []

    for cls in CLASSES:
        p = class_priors.get(cls, {})
        key = cls.lower().replace(" ", "_")
        speed_ms = float(p.get("speed_ms", 12.0))
        wingbeat_hz = float(p.get("wingbeat_hz", 6.0))
        expected_rcs = float(p.get("expected_rcs_db", -24.0))
        size_bin = float(p.get("size_bin", 1.0))
        wing_loading = float(p.get("wing_loading", 120.0))

        speed_gap = np.abs(airspeed - speed_ms) / max(speed_ms, 1.0)
        rcs_gap = np.abs(rcs_mean - expected_rcs) / max(abs(expected_rcs), 1.0)

        if include_flight:
            df_feat[f"prior_f_speed_gap_{key}"] = speed_gap
            df_feat[f"prior_f_speed_ratio_{key}"] = airspeed / max(speed_ms, 1.0)
            df_feat[f"prior_f_wb_gap_{key}"] = np.abs(rcs_peak_freq - wingbeat_hz)
            speed_matrix.append(speed_gap)

        if include_morph:
            speed_wl = 0.90 * np.sqrt(max(wing_loading, 1e-6))
            df_feat[f"prior_m_rcs_gap_{key}"] = rcs_gap
            df_feat[f"prior_m_size_gap_{key}"] = np.abs(radar_size - size_bin)
            df_feat[f"prior_m_wingload_gap_{key}"] = np.abs(airspeed - speed_wl) / max(speed_wl, 1.0)
            morph_matrix.append(rcs_gap)

        if include_morph and include_flight:
            joint_gap = 0.60 * speed_gap + 0.40 * rcs_gap
            df_feat[f"prior_mf_joint_gap_{key}"] = joint_gap
            joint_matrix.append(joint_gap)

    def _add_best_match_features(prefix: str, matrix_cols: list[np.ndarray]) -> None:
        if not matrix_cols:
            return
        mat = np.column_stack(matrix_cols)
        order = np.argsort(mat, axis=1)
        best_idx = order[:, 0]
        second_idx = order[:, 1] if mat.shape[1] > 1 else order[:, 0]
        best_val = mat[np.arange(mat.shape[0]), best_idx]
        second_val = mat[np.arange(mat.shape[0]), second_idx]
        df_feat[f"{prefix}_best_idx"] = best_idx
        df_feat[f"{prefix}_best_gap"] = best_val
        df_feat[f"{prefix}_margin"] = second_val - best_val
        df_feat[f"{prefix}_mean_gap"] = mat.mean(axis=1)

    _add_best_match_features("prior_f_summary", speed_matrix)
    _add_best_match_features("prior_m_summary", morph_matrix)
    _add_best_match_features("prior_mf_summary", joint_matrix)

    return df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)


# ── Build full feature matrix ──────────────────────────────────────

def build_features(df: pd.DataFrame, feature_sets: list[str] = None,
                   sig_depth: int = 2, sig_lead_lag: bool = True) -> pd.DataFrame:
    """
    Build feature DataFrame from raw data.

    Args:
        df: raw train or test DataFrame
        feature_sets: list of feature sets to include.
            Options: "core", "rcs_fft", "wavelet", "flight_mode", "tabular",
                     "targeted", "zaugg_cwt", "weakclass", "flight_physics",
                     "path_signature", "enhanced_bio_shape", "radar_physics",
                     "rcs_slope", "temporal_dynamics"
            Default: ["core", "rcs_fft", "tabular"]
        sig_depth: signature truncation depth (2 or 3)
        sig_lead_lag: whether to use lead-lag augmentation for signatures
    """
    if feature_sets is None:
        feature_sets = ["core", "rcs_fft", "tabular"]

    use_wavelet = "wavelet" in feature_sets
    use_flight = "flight_mode" in feature_sets
    use_zaugg = "zaugg_cwt" in feature_sets
    use_weakclass = "weakclass" in feature_sets
    use_physics = "flight_physics" in feature_sets
    use_signature = "path_signature" in feature_sets
    use_abs_wb = "absolute_wingbeat" in feature_sets
    use_linearity = "linearity" in feature_sets
    use_enhanced = "enhanced_bio_shape" in feature_sets
    use_radar_physics = "radar_physics" in feature_sets
    use_rcs_slope = "rcs_slope" in feature_sets
    use_traj_sep = "trajectory_separators" in feature_sets
    use_raw_signal = "raw_signal" in feature_sets
    use_micro = "micro_patterns" in feature_sets
    use_temporal_dynamics = "temporal_dynamics" in feature_sets

    rows = []
    total = len(df)
    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"  Features: {idx}/{total}", flush=True)
        feats = {}
        if "core" in feature_sets:
            feats.update(extract_core_features(r.trajectory, r.trajectory_time))
        if "rcs_fft" in feature_sets:
            feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
        if use_wavelet:
            feats.update(extract_wavelet_features(r.trajectory, r.trajectory_time))
        if use_flight:
            feats.update(extract_flight_mode_features(r.trajectory, r.trajectory_time))
        if use_zaugg:
            feats.update(extract_zaugg_cwt_features(r.trajectory, r.trajectory_time))
        if use_weakclass:
            feats.update(extract_weakclass_features(r.trajectory, r.trajectory_time))
        if use_physics:
            feats.update(extract_flight_physics_features(r.trajectory, r.trajectory_time))
        if use_signature:
            feats.update(extract_path_signature_features(
                r.trajectory, r.trajectory_time,
                depth=sig_depth, lead_lag=sig_lead_lag))
        if use_abs_wb:
            feats.update(extract_absolute_wingbeat(r.trajectory, r.trajectory_time))
        if use_linearity:
            feats.update(extract_linearity_features(r.trajectory, r.trajectory_time))
        if use_enhanced:
            feats.update(extract_enhanced_bio_shape_features(r.trajectory, r.trajectory_time))
        if use_radar_physics:
            feats.update(extract_radar_physics_features(r.trajectory, r.trajectory_time))
        if use_rcs_slope:
            feats.update(extract_rcs_slope(r.trajectory, r.trajectory_time))
        if use_traj_sep:
            feats.update(extract_trajectory_separators(r.trajectory, r.trajectory_time))
        if use_raw_signal:
            feats.update(extract_raw_signal_features(r.trajectory, r.trajectory_time))
        if use_micro:
            feats.update(extract_micro_pattern_features(r.trajectory, r.trajectory_time))
        if use_temporal_dynamics:
            feats.update(extract_temporal_dynamics(r.trajectory, r.trajectory_time))
        rows.append(feats)
    print(f"  Features: {total}/{total} done", flush=True)

    feat_df = pd.DataFrame(rows)

    if "tabular" in feature_sets:
        feat_df = add_tabular_features(feat_df, df)

    if "targeted" in feature_sets:
        feat_df = add_targeted_features(feat_df, df)

    if use_weakclass:
        feat_df = add_weakclass_tabular(feat_df, df)

    return feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
