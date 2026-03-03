import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

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
        
    # Center time for the day
    t_day = time_s[idx] - time_s[idx].min()
    
    X = np.column_stack([
        t_day / 30.0,      # 1 unit = 30 seconds
        speed[idx] / 2.0,  # 1 unit = 2 m/s
        size[idx] * 2.0    # 1 unit = 0.5 size diff
    ])
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.5, # max distance within cluster
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
print(f"Smoothed by cluster_id mAP: {m_smooth:.4f}")

