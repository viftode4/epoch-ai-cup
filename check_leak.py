import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train = train.sort_values("dt")

# Check if tracks close in time have the same class
train["time_diff"] = train["dt"].diff().dt.total_seconds()
train["same_class_as_prev"] = (train["bird_group"] == train["bird_group"].shift(1))

# For tracks within 60 seconds of each other
close_tracks = train[train["time_diff"] < 60]
print(f"Train tracks within 60s: {len(close_tracks)}")
print(f"Same class fraction: {close_tracks['same_class_as_prev'].mean():.3f}")

close_tracks_10s = train[train["time_diff"] < 10]
print(f"Train tracks within 10s: {len(close_tracks_10s)}")
print(f"Same class fraction: {close_tracks_10s['same_class_as_prev'].mean():.3f}")

# Check test set
test = pd.read_csv("data/test.csv")
test["dt"] = pd.to_datetime(test["timestamp_start_radar_utc"])
test = test.sort_values("dt")
test["time_diff"] = test["dt"].diff().dt.total_seconds()
print(f"Test tracks within 60s: {len(test[test['time_diff'] < 60])}")
print(f"Test tracks within 10s: {len(test[test['time_diff'] < 10])}")
