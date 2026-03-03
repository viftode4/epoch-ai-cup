import pandas as pd
import numpy as np
from src.data import parse_ewkb_4d

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])

mean_lon = np.zeros(len(train))
mean_lat = np.zeros(len(train))
for i, row in train.iterrows():
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        mean_lon[i] = np.mean([p[0] for p in pts])
        mean_lat[i] = np.mean([p[1] for p in pts])
    except:
        mean_lon[i] = np.nan
        mean_lat[i] = np.nan

train["mean_lon"] = mean_lon
train["mean_lat"] = mean_lat

# Calculate variance within primary_observation_id
variances = []
for name, group in train.groupby("primary_observation_id"):
    if len(group) > 1 and group["mean_lon"].notna().all():
        lat_rad = np.deg2rad(group["mean_lat"].mean())
        x_m = group["mean_lon"].values * 111000.0 * np.cos(lat_rad)
        y_m = group["mean_lat"].values * 111000.0
        
        dx = np.max(x_m) - np.min(x_m)
        dy = np.max(y_m) - np.min(y_m)
        dt = (group["dt"].max() - group["dt"].min()).total_seconds()
        dspeed = group["airspeed"].max() - group["airspeed"].min()
        dalt = group["min_z"].max() - group["min_z"].min()
        
        variances.append({
            "dx": dx, "dy": dy, "dt": dt, "dspeed": dspeed, "dalt": dalt,
            "size_nunique": group["radar_bird_size"].nunique()
        })

df_var = pd.DataFrame(variances)
print("90th percentile of within-flock spread:")
print(df_var.quantile(0.90))
print("\n95th percentile of within-flock spread:")
print(df_var.quantile(0.95))
print("\nMax within-flock spread:")
print(df_var.max())
