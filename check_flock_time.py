import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])

time_diffs = []
for name, group in train.groupby("primary_observation_id"):
    if len(group) > 1:
        group = group.sort_values("dt")
        diffs = group["dt"].diff().dt.total_seconds().dropna().values
        time_diffs.extend(diffs)

time_diffs = np.array(time_diffs)
print(f"Total pairs of consecutive tracks in same flock: {len(time_diffs)}")
print(f"Time diff == 0s: {(time_diffs == 0).sum()}")
print(f"Time diff <= 2s: {(time_diffs <= 2).sum()}")
print(f"Time diff <= 5s: {(time_diffs <= 5).sum()}")
print(f"Time diff <= 10s: {(time_diffs <= 10).sum()}")
print(f"Time diff <= 30s: {(time_diffs <= 30).sum()}")
print(f"Time diff <= 60s: {(time_diffs <= 60).sum()}")

