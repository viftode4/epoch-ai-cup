"""Cormorant fundamentals: what does the DATA actually tell us?

Stop modeling. Look at the data. What makes Cormorants unique?
- Where do they fly? (geographic position within radar range)
- What do field observers note? (observer_comment)
- How do they fly differently from each confusing class?
- What combinations of features are uniquely Cormorant?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

print("=" * 60, flush=True)
print("CORMORANT FUNDAMENTALS", flush=True)
print("=" * 60, flush=True)

# ======================================================================
# 1. Observer comments for Cormorants
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("1. WHAT DO FIELD OBSERVERS SAY?", flush=True)
print("=" * 60, flush=True)

corm = train_df[train_df["bird_group"] == "Cormorants"]
print(f"\n  Cormorant observer_comments:", flush=True)
for _, row in corm.iterrows():
    comment = row.get("observer_comment", "")
    if pd.notna(comment) and str(comment).strip():
        print(f"    Track {row['track_id']}: \"{comment}\"", flush=True)

print(f"\n  Cormorant n_birds_observed:", flush=True)
print(f"    {corm['n_birds_observed'].value_counts().to_dict()}", flush=True)

print(f"\n  Cormorant bird_species:", flush=True)
print(f"    {corm['bird_species'].value_counts().to_dict()}", flush=True)

print(f"\n  Cormorant radar_bird_size:", flush=True)
print(f"    {corm['radar_bird_size'].value_counts().to_dict()}", flush=True)

print(f"\n  Cormorant observer_position:", flush=True)
print(f"    {corm['observer_position'].value_counts().to_dict()}", flush=True)

# Compare with confusing classes
for cls in ["Gulls", "Ducks", "Pigeons", "Geese"]:
    sub = train_df[train_df["bird_group"] == cls]
    print(f"\n  {cls} n_birds_observed:", flush=True)
    print(f"    {sub['n_birds_observed'].value_counts().head(5).to_dict()}", flush=True)
    print(f"  {cls} radar_bird_size:", flush=True)
    print(f"    {sub['radar_bird_size'].value_counts().to_dict()}", flush=True)

# ======================================================================
# 2. WHERE do Cormorants fly? Geographic patterns
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("2. WHERE DO CORMORANTS FLY?", flush=True)
print("=" * 60, flush=True)

# Extract start/end/mean positions for all tracks
positions = []
for _, row in train_df.iterrows():
    pts = parse_ewkb_4d(row["trajectory"])
    lats = [p[1] for p in pts]
    lons = [p[0] for p in pts]
    alts = [p[2] for p in pts]

    positions.append({
        "class": row["bird_group"],
        "start_lat": lats[0], "start_lon": lons[0],
        "end_lat": lats[-1], "end_lon": lons[-1],
        "mean_lat": np.mean(lats), "mean_lon": np.mean(lons),
        "start_alt": alts[0], "end_alt": alts[-1],
        "mean_alt": np.mean(alts),
        "heading": np.degrees(np.arctan2(
            (lats[-1] - lats[0]) * 111000,
            (lons[-1] - lons[0]) * 67000
        )),
    })

pos_df = pd.DataFrame(positions)

# Cormorant vs rest positions
for metric in ["mean_lat", "mean_lon", "start_alt", "mean_alt", "heading"]:
    c_vals = pos_df.loc[pos_df["class"] == "Cormorants", metric]
    print(f"\n  Cormorant {metric}:", flush=True)
    print(f"    mean={c_vals.mean():.6f}, std={c_vals.std():.6f}, "
          f"range=[{c_vals.min():.6f}, {c_vals.max():.6f}]", flush=True)

    # Compare with other classes
    for cls in ["Gulls", "Ducks", "Geese", "Pigeons", "Birds of Prey"]:
        o_vals = pos_df.loc[pos_df["class"] == cls, metric]
        print(f"    {cls:<18s}: mean={o_vals.mean():.6f}, std={o_vals.std():.6f}", flush=True)

# Flight direction (heading) distribution
print(f"\n  Flight heading distribution:", flush=True)
for cls in ["Cormorants", "Gulls", "Ducks", "Geese", "Pigeons"]:
    headings = pos_df.loc[pos_df["class"] == cls, "heading"]
    # Bin into 8 compass directions
    bins = np.arange(-180, 225, 45)
    labels = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
    binned = pd.cut(headings, bins=bins, labels=labels)
    counts = binned.value_counts()
    top3 = counts.head(3)
    print(f"  {cls:<15s}: {', '.join(f'{d}:{n}' for d, n in top3.items())}", flush=True)

# ======================================================================
# 3. PAIRWISE confusion: what gets confused with Cormorants?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("3. CORMORANT vs EACH CONFUSING CLASS", flush=True)
print("=" * 60, flush=True)

# For each pair, what features BEST separate them?
from scipy.stats import ttest_ind
from src.features import build_features, ALL_TEMPORAL

print("\n  Building features...", flush=True)
X = build_features(train_df)
drop_cols = [c for c in ALL_TEMPORAL if c in X.columns]
X = X.drop(columns=drop_cols)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

corm_idx = np.where(y == CLASSES.index("Cormorants"))[0]

for cls in ["Gulls", "Ducks", "Pigeons", "Geese", "Birds of Prey"]:
    cls_idx = np.where(y == CLASSES.index(cls))[0]
    print(f"\n  Cormorant ({len(corm_idx)}) vs {cls} ({len(cls_idx)}):", flush=True)

    # Top 5 discriminative features
    t_stats = []
    for col in X.columns:
        c_vals = X.iloc[corm_idx][col].values
        o_vals = X.iloc[cls_idx][col].values
        t, p = ttest_ind(c_vals, o_vals, equal_var=False)
        t_stats.append((col, t, p))
    t_stats.sort(key=lambda x: -abs(x[1]))

    for feat, t, p in t_stats[:5]:
        c_med = X.iloc[corm_idx][feat].median()
        o_med = X.iloc[cls_idx][feat].median()
        direction = "Corm HIGHER" if t > 0 else "Corm LOWER"
        print(f"    {feat:<30s}: t={t:+.2f}, Corm={c_med:.3f}, {cls}={o_med:.3f} ({direction})", flush=True)

# ======================================================================
# 4. THE CORMORANT SIGNATURE: what combination is unique?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("4. THE CORMORANT COMBINATION", flush=True)
print("=" * 60, flush=True)

# What % of each class falls in the "Cormorant zone"?
# Cormorant zone: straight flight + long duration + large/medium size + moderate speed
print("\n  Defining Cormorant-like criteria from the data:", flush=True)

# Get Cormorant percentiles
for feat in ["sinuosity", "airspeed", "rcs_mean", "n_points"]:
    if feat in X.columns:
        c_vals = X.iloc[corm_idx][feat]
        p25, p75 = c_vals.quantile(0.25), c_vals.quantile(0.75)
        print(f"  {feat}: Cormorant IQR = [{p25:.3f}, {p75:.3f}]", flush=True)

        # What % of each class falls in this range?
        for cls in CLASSES:
            cls_i = np.where(y == CLASSES.index(cls))[0]
            vals = X.iloc[cls_i][feat]
            in_range = ((vals >= p25) & (vals <= p75)).mean()
            print(f"    {cls:<18s}: {in_range*100:5.1f}% in Cormorant IQR", flush=True)

# Multi-criteria: how many birds match ALL Cormorant criteria?
print(f"\n  Multi-criteria match (all 4 features in Cormorant IQR):", flush=True)
feats_to_check = ["sinuosity", "airspeed", "rcs_mean", "n_points"]
feats_available = [f for f in feats_to_check if f in X.columns]

if len(feats_available) >= 3:
    for cls in CLASSES:
        cls_i = np.where(y == CLASSES.index(cls))[0]
        all_match = np.ones(len(cls_i), dtype=bool)
        for feat in feats_available:
            c_vals = X.iloc[corm_idx][feat]
            p25, p75 = c_vals.quantile(0.25), c_vals.quantile(0.75)
            vals = X.iloc[cls_i][feat]
            all_match &= ((vals >= p25) & (vals <= p75)).values
        n_match = all_match.sum()
        print(f"    {cls:<18s}: {n_match:4d}/{len(cls_i)} ({n_match/len(cls_i)*100:5.1f}%)", flush=True)

# ======================================================================
# 5. What features do we NOT have that could help?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("5. WHAT ARE WE MISSING?", flush=True)
print("=" * 60, flush=True)

# Check if geographic features (lat/lon) are in our feature set
geo_feats = [c for c in X.columns if any(g in c.lower() for g in ["lat", "lon", "geo", "position"])]
print(f"  Geographic features in current set: {geo_feats if geo_feats else 'NONE'}", flush=True)

# Check heading/direction features
dir_feats = [c for c in X.columns if any(g in c.lower() for g in ["heading", "direction", "bearing"])]
print(f"  Direction/heading features: {dir_feats}", flush=True)

# Check n_birds features
nbird_feats = [c for c in X.columns if "n_bird" in c.lower()]
print(f"  N-birds features: {nbird_feats if nbird_feats else 'NONE'}", flush=True)

print("\n  Features we COULD add:", flush=True)
print("  - mean_lat, mean_lon (where in radar range)", flush=True)
print("  - flight_heading (compass direction of travel)", flush=True)
print("  - n_birds_observed (solo vs flock)", flush=True)
print("  - start_altitude, end_altitude (not just range)", flush=True)
print("  - distance_from_radar_center", flush=True)

print("\nDone!", flush=True)
