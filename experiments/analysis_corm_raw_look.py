"""Just LOOK at the raw radar data. Cormorants vs others.

No models. No features. Just the actual trajectory measurements.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

print("=" * 60, flush=True)
print("RAW RADAR DATA: WHAT DO WE ACTUALLY MEASURE?", flush=True)
print("=" * 60, flush=True)

# What columns does each track have?
print("\nColumns available:", flush=True)
for col in train_df.columns:
    print(f"  {col}", flush=True)

print("\n\nThe radar gives us per track:", flush=True)
print("  - trajectory: sequence of (lon, lat, altitude_m, RCS_dBm2)", flush=True)
print("  - trajectory_time: elapsed seconds at each measurement", flush=True)
print("  - radar_bird_size: Small/Medium/Large/Flock", flush=True)
print("  - airspeed: average m/s", flush=True)
print("  - min_z, max_z: altitude range in meters", flush=True)

# ======================================================================
# Look at a few Cormorants raw
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("CORMORANT TRAJECTORIES (first 5)", flush=True)
print("=" * 60, flush=True)

corm_idx = train_df.index[train_df["bird_group"] == "Cormorants"].tolist()

for i, idx in enumerate(corm_idx[:5]):
    row = train_df.loc[idx]
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])

    rcs = [p[3] for p in pts]
    alt = [p[2] for p in pts]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]

    print(f"\n--- Cormorant #{i+1} (track {row['track_id']}) ---", flush=True)
    print(f"  Points: {len(pts)}, Duration: {times[-1]:.0f}s", flush=True)
    print(f"  Bird size: {row['radar_bird_size']}, Airspeed: {row['airspeed']:.1f} m/s", flush=True)
    print(f"  Altitude: {row['min_z']:.0f} - {row['max_z']:.0f} m", flush=True)
    print(f"  Species: {row['bird_species']}", flush=True)

    # Print first 20 points
    print(f"  Time(s)  Alt(m)   RCS(dBm2)  Lat         Lon", flush=True)
    for j in range(min(20, len(pts))):
        print(f"  {times[j]:6.1f}  {alt[j]:7.1f}  {rcs[j]:9.1f}  "
              f"{lats[j]:.6f}  {lons[j]:.6f}", flush=True)
    if len(pts) > 20:
        print(f"  ... ({len(pts)-20} more points)", flush=True)

# ======================================================================
# Compare with other species
# ======================================================================
for species in ["Gulls", "Ducks", "Pigeons", "Songbirds"]:
    print(f"\n" + "=" * 60, flush=True)
    print(f"{species.upper()} TRAJECTORY (1 example)", flush=True)
    print("=" * 60, flush=True)

    sp_idx = train_df.index[train_df["bird_group"] == species].tolist()
    idx = sp_idx[0]
    row = train_df.loc[idx]
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    rcs = [p[3] for p in pts]
    alt = [p[2] for p in pts]

    print(f"  Track {row['track_id']}: {len(pts)} pts, {times[-1]:.0f}s", flush=True)
    print(f"  Bird size: {row['radar_bird_size']}, Airspeed: {row['airspeed']:.1f} m/s", flush=True)
    print(f"  Altitude: {row['min_z']:.0f} - {row['max_z']:.0f} m", flush=True)

    print(f"  Time(s)  Alt(m)   RCS(dBm2)", flush=True)
    for j in range(min(15, len(pts))):
        print(f"  {times[j]:6.1f}  {alt[j]:7.1f}  {rcs[j]:9.1f}", flush=True)
    if len(pts) > 15:
        print(f"  ... ({len(pts)-15} more points)", flush=True)

# ======================================================================
# Summary stats of the raw measurements by class
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY: RAW TRAJECTORY STATS BY CLASS", flush=True)
print("=" * 60, flush=True)

print(f"\n{'Class':<15s} {'N':>4s} {'Pts':>6s} {'Dur(s)':>7s} "
      f"{'Speed':>6s} {'Alt_range':>10s} {'RCS_mean':>9s} {'RCS_std':>8s} "
      f"{'Size_mode':>12s}", flush=True)
print("-" * 90, flush=True)

for cls in CLASSES:
    mask = train_df["bird_group"] == cls
    subset = train_df[mask]

    n_pts = []
    durations = []
    rcs_means = []
    rcs_stds = []
    for _, row in subset.iterrows():
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])
        rcs = [p[3] for p in pts]
        n_pts.append(len(pts))
        durations.append(times[-1] if len(times) > 0 else 0)
        rcs_means.append(np.mean(rcs))
        rcs_stds.append(np.std(rcs))

    size_mode = subset["radar_bird_size"].mode().iloc[0] if len(subset) > 0 else "?"
    alt_range = f"{subset['min_z'].median():.0f}-{subset['max_z'].median():.0f}"

    print(f"{cls:<15s} {len(subset):>4d} {np.median(n_pts):>6.0f} "
          f"{np.median(durations):>7.0f} {subset['airspeed'].median():>6.1f} "
          f"{alt_range:>10s} {np.median(rcs_means):>9.1f} {np.median(rcs_stds):>8.1f} "
          f"{size_mode:>12s}", flush=True)

# ======================================================================
# What makes Cormorants unique in the RAW data?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("WHAT'S UNIQUE ABOUT CORMORANTS IN RAW DATA?", flush=True)
print("=" * 60, flush=True)

# Collect per-track raw stats
all_stats = []
for idx, row in train_df.iterrows():
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    rcs = np.array([p[3] for p in pts])
    alt = np.array([p[2] for p in pts])
    lats = np.array([p[1] for p in pts])
    lons = np.array([p[0] for p in pts])

    # Track length and duration
    n = len(pts)
    dur = times[-1] if len(times) > 0 else 0

    # Displacement (straight line start to end)
    if n > 1:
        dx = (lons[-1] - lons[0]) * 67000
        dy = (lats[-1] - lats[0]) * 111000
        dz = alt[-1] - alt[0]
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)

        # Path length (total distance traveled)
        path_len = 0
        for j in range(1, n):
            ddx = (lons[j] - lons[j-1]) * 67000
            ddy = (lats[j] - lats[j-1]) * 111000
            ddz = alt[j] - alt[j-1]
            path_len += np.sqrt(ddx**2 + ddy**2 + ddz**2)

        straightness = displacement / max(path_len, 1e-6)
    else:
        displacement = 0
        path_len = 0
        straightness = 0

    # Altitude stability
    alt_std = np.std(alt) if n > 1 else 0
    alt_range = alt.max() - alt.min() if n > 1 else 0

    # RCS stats
    rcs_mean = rcs.mean()
    rcs_std = rcs.std() if n > 1 else 0

    all_stats.append({
        "n_pts": n,
        "duration": dur,
        "displacement": displacement,
        "path_length": path_len,
        "straightness": straightness,
        "alt_std": alt_std,
        "alt_range": alt_range,
        "rcs_mean": rcs_mean,
        "rcs_std": rcs_std,
        "airspeed": row["airspeed"],
        "bird_size": row["radar_bird_size"],
    })

stats_df = pd.DataFrame(all_stats)
stats_df["is_corm"] = (train_df["bird_group"] == "Cormorants").values
stats_df["class"] = train_df["bird_group"].values

# Compare Cormorants vs rest on each raw stat
print(f"\n{'Metric':<18s} {'Corm_median':>12s} {'Rest_median':>12s} {'Ratio':>8s}", flush=True)
print("-" * 55, flush=True)

for col in ["n_pts", "duration", "displacement", "path_length",
            "straightness", "alt_std", "alt_range", "rcs_mean",
            "rcs_std", "airspeed"]:
    c_med = stats_df.loc[stats_df["is_corm"], col].median()
    r_med = stats_df.loc[~stats_df["is_corm"], col].median()
    ratio = c_med / max(r_med, 1e-6)
    print(f"{col:<18s} {c_med:>12.2f} {r_med:>12.2f} {ratio:>8.2f}x", flush=True)

# Bird size distribution
print(f"\nBird size distribution:", flush=True)
for cls in ["Cormorants", "Gulls", "Ducks", "Pigeons", "Songbirds"]:
    mask = stats_df["class"] == cls
    sizes = stats_df.loc[mask, "bird_size"].value_counts()
    total = mask.sum()
    parts = []
    for s in ["Small bird", "Medium bird", "Large bird", "Flock"]:
        if s in sizes.index:
            parts.append(f"{s}:{sizes[s]}/{total}")
    print(f"  {cls:<15s}: {', '.join(parts)}", flush=True)

print("\nDone!", flush=True)
