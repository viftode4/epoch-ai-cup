"""Feature extraction from radar trajectories.

Each feature set is a separate function that returns a dict.
Combine them in experiments as needed.
"""
import numpy as np
import pandas as pd
from .data import parse_ewkb_4d, parse_trajectory_time


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

    size_map = {"Small bird": 0, "Medium": 1, "Large": 2, "Flock": 3}
    df_feat["radar_bird_size"] = df_orig["radar_bird_size"].map(size_map).values

    df_feat["airspeed_vs_ground"] = df_feat["airspeed"] / df_feat["avg_ground_speed"].clip(lower=0.01)

    # Pigeon temporal window: strongest single Pigeon discriminator
    # (14:00 peak in October per EXPERIMENTS.md)
    df_feat["is_pigeon_window"] = (
        (hour >= 13) & (hour <= 16) & (month == 10)
    ).astype(float)

    return df_feat


# ── Wingbeat / CWT frequency features ──────────────────────────────

def extract_wingbeat_features(hex_str: str, traj_time_str: str) -> dict:
    """
    Extract RCS spectral and periodicity features for bird species discrimination.

    Radar tracks have low sampling rates (~0.08–0.5 Hz), so absolute wingbeat
    frequency bands (0.5–20 Hz, Zaugg 2008) cannot be resolved. Instead we use
    relative frequency bands (quartiles of Nyquist) that are always populated:

        wb_band_q1: 0–25% of Nyquist  (slowest relative variation)
        wb_band_q2: 25–50% of Nyquist
        wb_band_q3: 50–75% of Nyquist
        wb_band_q4: 75–100% of Nyquist (fastest relative variation)

    Soaring birds concentrate power in Q1; periodic flappers in Q2–Q3; clutter is
    broadband (flat across quartiles). All four features are always non-zero.

    Also:
        wb_dominant_freq:  peak frequency as fraction of Nyquist [0,1]
        wb_total_power:    total spectral power (log-scale energy of RCS signal)
        rcs_autocorr_lag1: normalised lag-1 autocorrelation (high → periodic flapper)
        rcs_autocorr_lag5: normalised lag-5 autocorrelation
        rcs_periodicity:   max autocorrelation in lags 2–15
    """
    from scipy.interpolate import interp1d
    from scipy.signal import welch

    pts   = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    rcs   = np.array([p[3] for p in pts])
    n     = len(pts)

    defaults = {
        "wb_band_q1": 0.0, "wb_band_q2": 0.0,
        "wb_band_q3": 0.0, "wb_band_q4": 0.0,
        "wb_dominant_freq": 0.0, "wb_total_power": 0.0,
        "rcs_autocorr_lag1": 0.0, "rcs_autocorr_lag5": 0.0,
        "rcs_periodicity": 0.0,
    }
    if n < 8:
        return defaults

    try:
        # Interpolate RCS to uniform time grid
        uniform_t   = np.linspace(times[0], times[-1], n)
        fs          = 1.0 / max(uniform_t[1] - uniform_t[0], 0.01)
        nyquist     = fs / 2.0
        rcs_uniform = interp1d(times, rcs, kind="linear",
                               fill_value="extrapolate")(uniform_t)
        rcs_dc      = rcs_uniform - rcs_uniform.mean()

        # Welch PSD
        nperseg = min(n, 32)
        freqs, psd = welch(rcs_dc, fs=fs, nperseg=nperseg)

        def band_power(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(psd[mask].sum()) if mask.any() else 0.0

        total = psd[1:].sum() + 1e-10   # exclude DC

        # Relative frequency quartile bands (always populated)
        p_q1 = band_power(0.0,          nyquist * 0.25)
        p_q2 = band_power(nyquist * 0.25, nyquist * 0.50)
        p_q3 = band_power(nyquist * 0.50, nyquist * 0.75)
        p_q4 = band_power(nyquist * 0.75, nyquist)

        # Dominant frequency as fraction of Nyquist (exclude DC bin)
        peak_idx  = int(np.argmax(psd[1:])) + 1
        dom_freq  = float(freqs[peak_idx]) / nyquist  # normalised [0,1]

        # RCS autocorrelation (normalised, positive lags only)
        rcs_norm = rcs_dc / (rcs_dc.std() + 1e-8)
        acf_full = np.correlate(rcs_norm, rcs_norm, mode="full")
        acf      = acf_full[n - 1:]          # lags 0, 1, 2, ...
        acf      = acf / (acf[0] + 1e-8)    # lag-0 = 1.0

        lag1 = float(acf[1]) if len(acf) > 1 else 0.0
        lag5 = float(acf[5]) if len(acf) > 5 else 0.0
        # Periodicity: max autocorrelation in lags 2–15
        max_lag  = min(15, len(acf) - 1)
        periodic = float(acf[2:max_lag + 1].max()) if max_lag >= 2 else 0.0

        return {
            "wb_band_q1":        p_q1 / total,
            "wb_band_q2":        p_q2 / total,
            "wb_band_q3":        p_q3 / total,
            "wb_band_q4":        p_q4 / total,
            "wb_dominant_freq":  dom_freq,
            "wb_total_power":    float(total),
            "rcs_autocorr_lag1": lag1,
            "rcs_autocorr_lag5": lag5,
            "rcs_periodicity":   periodic,
        }
    except Exception:
        return defaults


# ── Build full feature matrix ──────────────────────────────────────

def build_features(df: pd.DataFrame, feature_sets: list[str] = None) -> pd.DataFrame:
    """
    Build feature DataFrame from raw data.

    Args:
        df: raw train or test DataFrame
        feature_sets: list of feature sets to include.
            Options: "core", "rcs_fft", "wingbeat", "tabular"
            Default: all of them.
    """
    if feature_sets is None:
        feature_sets = ["core", "rcs_fft", "wingbeat", "tabular"]

    rows = []
    for _, r in df.iterrows():
        feats = {}
        if "core" in feature_sets:
            feats.update(extract_core_features(r.trajectory, r.trajectory_time))
        if "rcs_fft" in feature_sets:
            feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
        if "wingbeat" in feature_sets:
            feats.update(extract_wingbeat_features(r.trajectory, r.trajectory_time))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)

    if "tabular" in feature_sets:
        feat_df = add_tabular_features(feat_df, df)

    return feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ── Sequence features for 1D-CNN ───────────────────────────────────

def extract_sequence(hex_str: str, traj_time_str: str, n_steps: int = 64) -> np.ndarray:
    """
    Convert a variable-length radar trajectory to a fixed-length (6, n_steps) array.

    Channels: [alt_norm, rcs_norm, speed, bearing_change, lat_delta, lon_delta]
    - alt and rcs are z-score normalized per-track (removes absolute bias, keeps shape)
    - Short tracks: linearly interpolated to n_steps
    - Long tracks: uniformly subsampled to n_steps

    Returns np.ndarray of shape (6, n_steps), float32.
    """
    from scipy.interpolate import interp1d

    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    lons = np.array([p[0] for p in pts], dtype=np.float64)
    lats = np.array([p[1] for p in pts], dtype=np.float64)
    alts = np.array([p[2] for p in pts], dtype=np.float64)
    rcs  = np.array([p[3] for p in pts], dtype=np.float64)

    # Compute per-step derived quantities (length n-1, then pad to n)
    if n > 1:
        dt = np.maximum(np.diff(times), 0.001)
        dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1]) for i in range(n-1)])
        speeds = dists / dt
        if n > 2:
            bearings = np.arctan2(np.diff(lats), np.diff(lons))
            bchanges = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
            bchanges = np.concatenate([[0.0], bchanges])  # length n-1
        else:
            bchanges = np.zeros(n - 1)
        lat_deltas = np.diff(lats)
        lon_deltas = np.diff(lons)
        # Pad derived arrays to length n (repeat last value)
        speeds     = np.concatenate([speeds,     [speeds[-1]]])
        bchanges   = np.concatenate([bchanges,   [bchanges[-1]]])
        lat_deltas = np.concatenate([lat_deltas, [lat_deltas[-1]]])
        lon_deltas = np.concatenate([lon_deltas, [lon_deltas[-1]]])
    else:
        speeds     = np.zeros(n)
        bchanges   = np.zeros(n)
        lat_deltas = np.zeros(n)
        lon_deltas = np.zeros(n)

    # Z-score normalise alt and rcs per-track
    def znorm(arr):
        s = arr.std()
        return (arr - arr.mean()) / s if s > 1e-8 else arr - arr.mean()

    channels = np.stack([znorm(alts), znorm(rcs), speeds, bchanges, lat_deltas, lon_deltas])
    # channels shape: (6, n)

    # Resample to n_steps along the time axis
    if n == 1:
        out = np.repeat(channels, n_steps, axis=1)
    elif n >= n_steps:
        idx = np.round(np.linspace(0, n - 1, n_steps)).astype(int)
        out = channels[:, idx]
    else:
        t_orig = np.linspace(0, 1, n)
        t_new  = np.linspace(0, 1, n_steps)
        out = np.stack([
            interp1d(t_orig, channels[c], kind='linear')(t_new)
            for c in range(channels.shape[0])
        ])

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_sequences(df: pd.DataFrame, n_steps: int = 64) -> np.ndarray:
    """
    Extract fixed-length sequences for all rows in df.

    Returns np.ndarray of shape (len(df), 6, n_steps), float32.
    """
    seqs = []
    for _, row in df.iterrows():
        seqs.append(extract_sequence(row.trajectory, row.trajectory_time, n_steps))
    return np.stack(seqs)
