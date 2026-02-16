"""Feature extraction from radar trajectories.

Each feature set is a separate function that returns a dict.
Combine them in experiments as needed.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .data import parse_ewkb_4d, parse_trajectory_time

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

    # Segments
    if n > 1:
        dists = np.array([haversine(lons[i], lats[i], lons[i + 1], lats[i + 1]) for i in range(n - 1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists / dt
    else:
        dists, dt, speeds = np.array([0.0]), np.array([1.0]), np.array([0.0])

    total_dist = dists.sum()
    straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
    sinuosity = total_dist / max(straight_dist, 1e-6)

    alt_diffs = np.diff(alts) if n > 1 else np.array([0.0])
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    if n > 2:
        bearings = np.arctan2(np.diff(lats), np.diff(lons))
        bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
    else:
        bearing_changes = np.array([0.0])

    # Acceleration
    accel = np.diff(speeds) / np.maximum(dt[:-1], 0.001) if len(speeds) > 1 and len(dt) > 1 else np.array([0.0])

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
        # interactions
        "speed_x_alt": np.mean(speeds) * np.mean(alts),
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
            bearings = np.arctan2(np.diff(lats), np.diff(lons))
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
        if n > 3:
            dx = np.diff(lons)
            dy = np.diff(lats)
            ds = np.sqrt(dx**2 + dy**2) + 1e-10
            # Curvature = |d(bearing)/ds|
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

    size_map = {"Small bird": 0, "Medium bird": 1, "Large bird": 2, "Flock": 3}
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
    size_map = {"Small bird": 0, "Medium bird": 1, "Large bird": 2, "Flock": 3}
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
            dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                              for i in range(n-1)])
            speeds = dists / dt
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
            bearings = np.arctan2(np.diff(lats), np.diff(lons))
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


def add_weakclass_tabular(df_feat: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Tabular features targeting weak classes (migration timing, size interactions)."""
    ts = pd.to_datetime(df_orig["timestamp_start_radar_utc"])
    month = ts.dt.month.values
    hour = ts.dt.hour.values

    size_map = {"Small bird": 0, "Medium bird": 1, "Large bird": 2, "Flock": 3}
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


# ── Build full feature matrix ──────────────────────────────────────

def build_features(df: pd.DataFrame, feature_sets: list[str] = None) -> pd.DataFrame:
    """
    Build feature DataFrame from raw data.

    Args:
        df: raw train or test DataFrame
        feature_sets: list of feature sets to include.
            Options: "core", "rcs_fft", "wavelet", "flight_mode", "tabular", "targeted"
            Default: ["core", "rcs_fft", "tabular"]
    """
    if feature_sets is None:
        feature_sets = ["core", "rcs_fft", "tabular"]

    use_wavelet = "wavelet" in feature_sets
    use_flight = "flight_mode" in feature_sets
    use_zaugg = "zaugg_cwt" in feature_sets
    use_weakclass = "weakclass" in feature_sets

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
