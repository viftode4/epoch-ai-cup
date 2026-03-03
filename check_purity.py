import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.data import parse_ewkb_4d

train = pd.read_csv("data/train.csv")
dt = pd.to_datetime(train["timestamp_start_radar_utc"])
time_s = dt.astype(int) / 10**9
speed = pd.to_numeric(train["airspeed"], errors="coerce").fillna(10.0).values
size_map = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}
size = train["radar_bird_size"].map(size_map).fillna(2).values

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
    
    X = np.column_stack([
        time_s[idx] / 15.0,
        x_m / 200.0,
        y_m / 200.0,
        speed[idx] / 1.0,
        size[idx] * 10.0
    ])
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage="complete")
    labels = clustering.fit_predict(X)
    cluster_id[idx] = labels + current_id
    current_id += labels.max() + 1

train["cluster_id"] = cluster_id

purity = []
for cid, group in train.groupby("cluster_id"):
    if len(group) > 1:
        purity.append(group["bird_group"].nunique() == 1)

print(f"Number of >1 clusters: {len(purity)}")
print(f"Purity: {np.mean(purity):.4f}")

