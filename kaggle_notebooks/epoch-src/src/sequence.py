"""Trajectory preprocessing for sequence models (1D-CNN, etc.).

Converts variable-length radar tracks into fixed-length multivariate
time series suitable for PyTorch models.

v1 (legacy): trajectory_to_channels / prepare_sequences
    Linear interpolation to fixed grid — destroys temporal ACF structure.

v2: prepare_sequences_v2 with multiple modes:
    "pad"      — raw sequences zero-padded with boolean mask
    "resample" — nearest-neighbor resample at 1 Hz in real time + pad
    "features" — fixed-size feature vector (ACF, FFT, cross-correlations)
"""
import numpy as np
from scipy.interpolate import interp1d
from .data import parse_ewkb_4d, parse_trajectory_time


# ── v2 channel definitions (dropped lon/lat deltas — not discriminative) ──
CHANNELS_V2 = ["altitude", "rcs", "speed", "bearing_change", "rcs_deriv", "alt_deriv"]
N_CHANNELS_V2 = 6


# ── v1 (legacy, kept for backward compatibility) ──────────────────────────

def trajectory_to_channels(hex_str: str, traj_time_str: str,
                           seq_len: int = 64) -> np.ndarray:
    """Convert a single trajectory to a fixed-length multichannel time series.

    WARNING: Linear interpolation to fixed grid destroys temporal ACF structure.
    Prefer prepare_sequences_v2() for new code.

    Channels (8):
        0: altitude (m)
        1: RCS (dBm2)
        2: speed (m/s, from haversine between consecutive points)
        3: bearing change (rad)
        4: lon delta (deg, from previous point)
        5: lat delta (deg, from previous point)
        6: RCS derivative (dBm2/s, wing motion signal)
        7: altitude derivative (m/s, vertical speed)

    Returns:
        np.ndarray of shape (8, seq_len), float32
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    if n < 2:
        return np.zeros((8, seq_len), dtype=np.float32)

    # Per-step derived channels (length n-1, pad first element with 0)
    dt = np.maximum(np.diff(times), 0.001)

    # Speed from coordinate deltas
    R = 6371000
    dlat = np.radians(np.diff(lats))
    dlon_rad = np.radians(np.diff(lons))
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_rad / 2) ** 2
    dists = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    speeds = np.concatenate([[0], dists / dt])

    # Bearing change (scale to meters at lat ~53.5)
    dx = np.diff(lons) * 67000.0
    dy = np.diff(lats) * 111000.0
    bearings = np.arctan2(dy, dx)
    if len(bearings) > 1:
        bc = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
        bearing_changes = np.concatenate([[0, 0], bc])
    else:
        bearing_changes = np.zeros(n)

    # Position deltas
    lon_delta = np.concatenate([[0], np.diff(lons)])
    lat_delta = np.concatenate([[0], np.diff(lats)])

    # Derivative channels (rate of change = motion signal)
    rcs_deriv = np.concatenate([[0], np.diff(rcs) / dt])  # RCS change rate (wing motion)
    alt_deriv = np.concatenate([[0], np.diff(alts) / dt])  # vertical speed (bounding flight)

    # Stack raw channels (8, n)
    raw = np.stack([alts, rcs, speeds, bearing_changes, lon_delta, lat_delta,
                    rcs_deriv, alt_deriv])

    # Interpolate to fixed length
    n_channels = raw.shape[0]
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, seq_len)
    channels = np.zeros((n_channels, seq_len), dtype=np.float32)
    for ch in range(n_channels):
        f = interp1d(t_orig, raw[ch], kind="linear", fill_value="extrapolate")
        channels[ch] = f(t_new)

    return channels


def prepare_sequences(df, seq_len: int = 64) -> np.ndarray:
    """Convert a DataFrame of tracks into a batch of sequences (v1 legacy).

    WARNING: Uses linear interpolation. Prefer prepare_sequences_v2().

    Returns:
        np.ndarray of shape (N, 8, seq_len), float32
    """
    total = len(df)
    sequences = np.zeros((total, 8, seq_len), dtype=np.float32)
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Sequences: {i}/{total}", flush=True)
        sequences[i] = trajectory_to_channels(r.trajectory, r.trajectory_time,
                                              seq_len=seq_len)
    print(f"  Sequences: {total}/{total} done", flush=True)
    return sequences


# ── v2: proper variable-length handling ──────────────────────────────────

def _extract_raw_channels(hex_str: str, traj_time_str: str):
    """Parse trajectory and compute 6 raw channels (no interpolation).

    Channels: altitude, rcs, speed, bearing_change, rcs_deriv, alt_deriv

    Returns:
        raw: (6, n) float32 array
        times: (n,) float64 array of elapsed seconds
    """
    pts = parse_ewkb_4d(hex_str)
    times = np.array(parse_trajectory_time(traj_time_str), dtype=np.float64)
    n = len(pts)

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    if n < 2:
        raw = np.stack([alts, rcs, np.zeros(n), np.zeros(n),
                        np.zeros(n), np.zeros(n)])
        return raw.astype(np.float32), times

    dt = np.maximum(np.diff(times), 0.001)

    # Speed (haversine)
    R_earth = 6371000
    dlat = np.radians(np.diff(lats))
    dlon_rad = np.radians(np.diff(lons))
    lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_rad / 2) ** 2
    dists = R_earth * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    speeds = np.concatenate([[dists[0] / dt[0]], dists / dt])  # repeat first speed instead of 0

    # Bearing change (latitude-corrected)
    dx = np.diff(lons) * 67000.0
    dy = np.diff(lats) * 111000.0
    bearings = np.arctan2(dy, dx)
    if len(bearings) > 1:
        bc = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
        bearing_changes = np.concatenate([[0, bc[0]], bc])  # repeat first bc instead of double-zero
    else:
        bearing_changes = np.zeros(n)

    # Derivatives
    rcs_deriv = np.concatenate([[np.diff(rcs)[0] / dt[0]], np.diff(rcs) / dt])
    alt_deriv = np.concatenate([[np.diff(alts)[0] / dt[0]], np.diff(alts) / dt])

    raw = np.stack([alts, rcs, speeds, bearing_changes, rcs_deriv, alt_deriv])
    return raw.astype(np.float32), times


def prepare_sequences_v2(df, mode="resample", max_len=200, resample_rate=1.0,
                         acf_lags=10):
    """Convert DataFrame of tracks to sequences with proper variable-length handling.

    Modes
    -----
    "pad":
        Raw variable-length sequences, zero-padded to max_len.
        Returns: (sequences (N, 6, max_len), mask (N, max_len), lengths (N,))
        Use for: Transformers with attention mask.

    "resample":
        Nearest-neighbor resample at resample_rate Hz in real time, then pad.
        Returns: (sequences (N, 6, max_len), mask (N, max_len), lengths (N,))
        Use for: CNN, ROCKET — preserves ACF structure.

    "features":
        Fixed-size feature vector from raw variable-length data:
        ACF lags, FFT, cross-channel correlations.
        Returns: (features (N, F), None, lengths (N,))
        Use for: Tree models alongside tabular features.
    """
    N = len(df)
    C = N_CHANNELS_V2
    lengths = np.zeros(N, dtype=np.int32)

    # Phase 1: extract all raw channels
    raw_list = []
    time_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  seq_v2 [{mode}]: {i}/{N}", flush=True)
        raw, times = _extract_raw_channels(row.trajectory, row.trajectory_time)
        raw_list.append(raw)
        time_list.append(times)
        lengths[i] = raw.shape[1]
    print(f"  seq_v2 [{mode}]: {N}/{N} done", flush=True)

    if mode == "pad":
        return _mode_pad(raw_list, lengths, max_len)
    elif mode == "resample":
        return _mode_resample(raw_list, time_list, lengths, max_len, resample_rate)
    elif mode == "features":
        return _mode_features(raw_list, time_list, lengths, acf_lags)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _mode_pad(raw_list, lengths, max_len):
    """Zero-pad raw sequences to max_len. No interpolation."""
    N = len(raw_list)
    C = N_CHANNELS_V2
    seq = np.zeros((N, C, max_len), dtype=np.float32)
    mask = np.zeros((N, max_len), dtype=bool)

    for i, raw in enumerate(raw_list):
        L = min(raw.shape[1], max_len)
        seq[i, :, :L] = raw[:, :L]
        mask[i, :L] = True
        lengths[i] = L

    return seq, mask, lengths


def _mode_resample(raw_list, time_list, lengths, max_len, rate):
    """Nearest-neighbor resample at `rate` Hz in real time, then pad.

    Preserves ACF structure because it copies actual measured values
    rather than creating artificial interpolated values.
    """
    N = len(raw_list)
    C = N_CHANNELS_V2
    seq = np.zeros((N, C, max_len), dtype=np.float32)
    mask = np.zeros((N, max_len), dtype=bool)

    for i, (raw, times) in enumerate(zip(raw_list, time_list)):
        n_orig = raw.shape[1]
        if n_orig < 2:
            if n_orig == 1:
                seq[i, :, 0] = raw[:, 0]
                mask[i, 0] = True
            lengths[i] = n_orig
            continue

        duration = times[-1] - times[0]
        step = 1.0 / rate
        n_new = min(int(duration / step) + 1, max_len)
        if n_new < 1:
            n_new = 1
        t_new = np.arange(n_new) * step + times[0]

        # Nearest-neighbor: for each t_new[j], find closest original point
        indices = np.searchsorted(times, t_new, side="left")
        indices = np.clip(indices, 0, n_orig - 1)
        # Refine: check if previous point is closer
        prev = np.clip(indices - 1, 0, n_orig - 1)
        use_prev = (np.abs(t_new - times[prev]) < np.abs(t_new - times[indices]))
        indices[use_prev] = prev[use_prev]

        seq[i, :, :n_new] = raw[:, indices]
        mask[i, :n_new] = True
        lengths[i] = n_new

    return seq, mask, lengths


def _mode_features(raw_list, time_list, lengths, acf_lags):
    """Extract fixed-size temporal feature vector from raw variable-length data.

    Features (~37):
        - RCS ACF lags 1..acf_lags (10)
        - alt_deriv ACF lags 1..acf_lags (10)
        - RCS FFT: top-3 frequencies + magnitudes (6)
        - Cross-channel correlations (6)
        - Sequence metadata (5)
    """
    N = len(raw_list)
    # Channel indices in CHANNELS_V2
    CH_ALT, CH_RCS, CH_SPD, CH_BC, CH_RCSD, CH_ALTD = range(6)

    n_feat = acf_lags + acf_lags + 6 + 6 + 5
    features = np.zeros((N, n_feat), dtype=np.float32)

    for i, (raw, times) in enumerate(zip(raw_list, time_list)):
        n_orig = raw.shape[1]
        if n_orig < 3:
            features[i, -5] = n_orig  # n_points
            if n_orig >= 2:
                features[i, -4] = times[-1] - times[0]  # duration
            continue

        col = 0

        # RCS ACF
        acf_rcs = _acf(raw[CH_RCS], acf_lags)
        features[i, col:col + acf_lags] = acf_rcs
        col += acf_lags

        # alt_deriv ACF
        acf_altd = _acf(raw[CH_ALTD], acf_lags)
        features[i, col:col + acf_lags] = acf_altd
        col += acf_lags

        # RCS FFT top-3
        fft_feats = _top_fft(raw[CH_RCS], times, n_top=3)
        features[i, col:col + 6] = fft_feats
        col += 6

        # Cross-channel correlations
        pairs = [
            (CH_RCS, CH_ALT),    # rcs vs altitude
            (CH_RCS, CH_SPD),    # rcs vs speed
            (CH_ALT, CH_SPD),    # altitude vs speed
            (CH_RCSD, CH_ALTD),  # rcs_deriv vs alt_deriv (dynamic coupling)
            (CH_SPD, CH_BC),     # speed vs bearing_change
            (CH_RCS, CH_RCSD),   # rcs vs rcs_deriv
        ]
        for ci, cj in pairs:
            features[i, col] = _pearson(raw[ci], raw[cj])
            col += 1

        # Metadata
        dt = np.diff(times)
        duration = times[-1] - times[0]
        features[i, col] = n_orig
        features[i, col + 1] = duration
        features[i, col + 2] = np.mean(dt)
        features[i, col + 3] = np.std(dt) if len(dt) > 1 else 0
        features[i, col + 4] = n_orig / max(duration, 0.01)

    # Build feature names
    feat_names = (
        [f"rcs_acf_lag{k}" for k in range(1, acf_lags + 1)]
        + [f"altd_acf_lag{k}" for k in range(1, acf_lags + 1)]
        + ["rcs_fft_freq1", "rcs_fft_mag1", "rcs_fft_freq2", "rcs_fft_mag2",
           "rcs_fft_freq3", "rcs_fft_mag3"]
        + ["xcorr_rcs_alt", "xcorr_rcs_spd", "xcorr_alt_spd",
           "xcorr_rcsd_altd", "xcorr_spd_bc", "xcorr_rcs_rcsd"]
        + ["seq_n_points", "seq_duration", "seq_mean_dt", "seq_std_dt", "seq_sampling_rate"]
    )

    return features, feat_names, lengths


# ── Helper functions for features mode ───────────────────────────────────

def _acf(x, max_lag):
    """Autocorrelation at lags 1..max_lag."""
    x_c = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag, dtype=np.float32)
    result = np.zeros(max_lag, dtype=np.float32)
    n = len(x_c)
    for lag in range(1, max_lag + 1):
        if n <= lag:
            break
        result[lag - 1] = np.mean(x_c[:n - lag] * x_c[lag:]) / var
    return result


def _top_fft(x, times, n_top=3):
    """Top-n FFT frequencies (in Hz) and magnitudes from raw data."""
    result = np.zeros(2 * n_top, dtype=np.float32)
    n = len(x)
    if n < 4:
        return result
    x_c = x - np.mean(x)
    fft_mag = np.abs(np.fft.rfft(x_c))
    fft_mag[0] = 0  # drop DC
    # Compute actual Hz frequencies using mean sampling interval
    mean_dt = (times[-1] - times[0]) / max(n - 1, 1)
    freqs = np.fft.rfftfreq(n, d=max(mean_dt, 0.01))
    top_idx = np.argsort(fft_mag)[-n_top:][::-1]
    for j, idx in enumerate(top_idx):
        if j >= n_top:
            break
        result[2 * j] = freqs[idx]
        result[2 * j + 1] = fft_mag[idx]
    return result


def _pearson(a, b):
    """Pearson correlation, safe for constant arrays."""
    if len(a) < 3:
        return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-10 or sb < 1e-10:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0
