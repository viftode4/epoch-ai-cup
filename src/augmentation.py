"""Trajectory augmentation for rare bird classes.

Augments EWKB trajectories by applying physically-plausible transforms:
- Rotation: rotate entire trajectory (birds don't care about compass heading)
- Speed perturbation: scale time intervals (±20% natural variation)
- RCS noise: add Gaussian noise to radar cross-section (sensor variation)
- Altitude shift: shift altitude profile (same bird, different elevation)
- Spatial translation: shift lat/lon (same bird, different location)

All transforms preserve the physical structure of the trajectory.
"""

import numpy as np
import struct
from typing import Optional


def parse_trajectory_arrays(hex_str: str):
    """Parse EWKB into separate arrays (lons, lats, alts, rcs)."""
    raw = bytes.fromhex(hex_str)
    offset = 0
    bo = "<" if raw[offset] == 1 else ">"
    offset += 1
    geom_type = struct.unpack_from(f"{bo}I", raw, offset)[0]
    offset += 4
    srid = None
    if geom_type & 0x20000000:
        srid = struct.unpack_from(f"{bo}I", raw, offset)[0]
        offset += 4
    n_points = struct.unpack_from(f"{bo}I", raw, offset)[0]
    offset += 4
    lons, lats, alts, rcs = [], [], [], []
    for _ in range(n_points):
        lon, lat, alt, r = struct.unpack_from(f"{bo}4d", raw, offset)
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)
        rcs.append(r)
        offset += 32
    return (np.array(lons), np.array(lats), np.array(alts), np.array(rcs),
            bo, geom_type, srid)


def arrays_to_ewkb(lons, lats, alts, rcs, bo="<", geom_type=0xC0000002, srid=4326):
    """Encode arrays back to EWKB hex string."""
    parts = []
    parts.append(struct.pack("B", 1 if bo == "<" else 0))
    parts.append(struct.pack(f"{bo}I", geom_type))
    if geom_type & 0x20000000 and srid is not None:
        parts.append(struct.pack(f"{bo}I", srid))
    n = len(lons)
    parts.append(struct.pack(f"{bo}I", n))
    for i in range(n):
        parts.append(struct.pack(f"{bo}4d", lons[i], lats[i], alts[i], rcs[i]))
    return b"".join(parts).hex()


def augment_rotation(lons, lats, alts, rcs, angle_deg=None, rng=None):
    """Rotate trajectory around its centroid. Random angle if not specified."""
    if rng is None:
        rng = np.random.default_rng()
    if angle_deg is None:
        angle_deg = rng.uniform(0, 360)
    angle = np.radians(angle_deg)

    # Convert to local meters from centroid
    lat_center = np.mean(lats)
    lon_center = np.mean(lons)
    dx = (lons - lon_center) * 67000.0  # cos(53.5) * 111000
    dy = (lats - lat_center) * 111000.0

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # Convert back to lat/lon
    new_lons = lon_center + dx_rot / 67000.0
    new_lats = lat_center + dy_rot / 111000.0
    return new_lons, new_lats, alts.copy(), rcs.copy()


def augment_speed(times, scale=None, rng=None):
    """Scale trajectory time intervals (speed perturbation)."""
    if rng is None:
        rng = np.random.default_rng()
    if scale is None:
        scale = rng.uniform(0.8, 1.2)  # +/- 20%
    # Scale inter-point intervals
    dt = np.diff(times)
    dt_new = dt * scale
    new_times = np.zeros_like(times)
    new_times[1:] = np.cumsum(dt_new)
    return new_times


def augment_rcs_noise(rcs, sigma=1.0, rng=None):
    """Add Gaussian noise to RCS values (sensor measurement noise)."""
    if rng is None:
        rng = np.random.default_rng()
    return rcs + rng.normal(0, sigma, size=len(rcs))


def augment_altitude_shift(alts, shift_m=None, rng=None):
    """Shift entire altitude profile. Clamp to >= 0."""
    if rng is None:
        rng = np.random.default_rng()
    if shift_m is None:
        shift_m = rng.uniform(-20, 20)
    return np.maximum(alts + shift_m, 0.0)


def augment_spatial_translation(lons, lats, dx_m=None, dy_m=None, rng=None):
    """Translate trajectory in space (different radar location)."""
    if rng is None:
        rng = np.random.default_rng()
    if dx_m is None:
        dx_m = rng.uniform(-500, 500)
    if dy_m is None:
        dy_m = rng.uniform(-500, 500)
    new_lons = lons + dx_m / 67000.0
    new_lats = lats + dy_m / 111000.0
    return new_lons, new_lats


def augment_trajectory(hex_str: str, traj_time_str: str,
                       rotate: bool = True,
                       speed_perturb: bool = True,
                       rcs_noise: bool = True,
                       alt_shift: bool = True,
                       spatial_shift: bool = True,
                       rng: Optional[np.random.Generator] = None):
    """Apply a random combination of augmentations to a trajectory.

    Returns:
        (augmented_hex, augmented_times_str)
    """
    if rng is None:
        rng = np.random.default_rng()

    lons, lats, alts, rcs, bo, geom_type, srid = parse_trajectory_arrays(hex_str)
    times = np.array(eval(traj_time_str))

    if rotate:
        lons, lats, alts, rcs = augment_rotation(lons, lats, alts, rcs, rng=rng)
    if spatial_shift:
        lons, lats = augment_spatial_translation(lons, lats, rng=rng)
    if alt_shift:
        alts = augment_altitude_shift(alts, rng=rng)
    if rcs_noise:
        rcs = augment_rcs_noise(rcs, sigma=0.5, rng=rng)  # conservative noise
    if speed_perturb:
        times = augment_speed(times, rng=rng)

    new_hex = arrays_to_ewkb(lons, lats, alts, rcs, bo, geom_type, srid)
    new_times_str = str(times.tolist())
    return new_hex, new_times_str


def augment_rare_classes(train_df, y, classes, target_count=200, seed=42):
    """Augment rare classes to reach target_count samples each.

    Args:
        train_df: training DataFrame with trajectory, trajectory_time columns
        y: integer labels
        classes: list of class names
        target_count: target number of samples per rare class
        seed: random seed

    Returns:
        aug_df: DataFrame of augmented samples (new rows only)
        aug_y: integer labels for augmented samples
    """
    rng = np.random.default_rng(seed)
    aug_rows = []
    aug_labels = []

    for cls_idx, cls_name in enumerate(classes):
        mask = y == cls_idx
        n_existing = mask.sum()
        if n_existing >= target_count:
            continue

        n_needed = target_count - n_existing
        cls_df = train_df[mask]

        for i in range(n_needed):
            # Sample a random row from this class
            src_row = cls_df.iloc[rng.integers(0, len(cls_df))]
            new_hex, new_times = augment_trajectory(
                src_row.trajectory, src_row.trajectory_time, rng=rng
            )

            # Create augmented row (copy all columns, replace trajectory data)
            new_row = src_row.copy()
            new_row["trajectory"] = new_hex
            new_row["trajectory_time"] = new_times
            aug_rows.append(new_row)
            aug_labels.append(cls_idx)

    if not aug_rows:
        return train_df.iloc[:0], np.array([], dtype=int)

    import pandas as pd
    aug_df = pd.DataFrame(aug_rows).reset_index(drop=True)
    aug_y = np.array(aug_labels)
    return aug_df, aug_y
