"""Trajectory preprocessing for sequence models (1D-CNN, etc.).

Converts variable-length radar tracks into fixed-length multivariate
time series suitable for PyTorch models.
"""
import numpy as np
from scipy.interpolate import interp1d
from .data import parse_ewkb_4d, parse_trajectory_time


def trajectory_to_channels(hex_str: str, traj_time_str: str,
                           seq_len: int = 64) -> np.ndarray:
    """Convert a single trajectory to a fixed-length multichannel time series.

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

    # Bearing change
    dx = np.diff(lons)
    dy = np.diff(lats)
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
    """Convert a DataFrame of tracks into a batch of sequences.

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
