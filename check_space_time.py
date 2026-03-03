import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.data import CLASSES, parse_ewkb_4d
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

# Extract mean lon, lat
mean_lon = np.zeros(len(train))
mean_lat = np.zeros(len(train))

for i, row in train.iterrows():
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        mean_lon[i] = np.mean([p[0] for p in pts])
        mean_lat[i] = np.mean([p[1] for p in pts])
    except:
        mean_lon[i] = 0
        mean_lat[i] = 0

train["mean_lon"] = mean_lon
train["mean_lat"] = mean_lat

dt = pd.to_datetime(train["timestamp_start_radar_utc"])
time_s = dt.astype(int) / 10**9
speed = pd.to_numeric(train["airspeed"], errors="coerce").fillna(10.0).values
size_map = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}
size = train["radar_bird_size"].map(size_map).fillna(2).values

train["date"] = dt.dt.date
cluster_id = np.zeros(len(train), dtype=int)
current_id = 0

for date, group in train.groupby("date"):
    idx = group.index
    if len(idx) == 1:
        cluster_id[idx] = current_id
        current_id += 1
        continue
        
    lat_rad = np.deg2rad(mean_lat[idx].mean())
    x_m = mean_lon[idx] * 111000.0 * np.cos(lat_rad)
    y_m = mean_lat[idx] * 111000.0
    
    # Very strict distance metric
    X = np.column_stack([
        time_s[idx] / 15.0,   # 1 unit = 15 seconds
        x_m / 200.0,          # 1 unit = 200 meters
        y_m / 200.0,
        speed[idx] / 1.0,     # 1 unit = 1 m/s
        size[idx] * 10.0      # 1 unit = 0.1 size diff (forces exact size match)
    ])
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.5,
        linkage="complete"
    )
    labels = clustering.fit_predict(X)
    
    cluster_id[idx] = labels + current_id
    current_id += labels.max() + 1

train["cluster_id"] = cluster_id
print(f"Found {train['cluster_id'].nunique()} clusters (True groups: {train['primary_observation_id'].nunique()})")

smoothed_oof = oof.copy()
for cid, group in train.groupby("cluster_id"):
    idx = group.index
    if len(idx) > 1:
        smoothed_oof[idx] = np.mean(oof[idx], axis=0)

m_smooth, _ = compute_map(y, smoothed_oof)
print(f"Smoothed by strict space-time mAP: {m_smooth:.4f} (Delta: {m_smooth - m:+.4f})")

