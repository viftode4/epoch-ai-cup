import pandas as pd
import numpy as np

test = pd.read_csv("data/test.csv")
test["dt"] = pd.to_datetime(test["timestamp_start_radar_utc"])
test = test.sort_values("dt")

# Group by connected components where time_diff <= 2 seconds
test["time_diff"] = test["dt"].diff().dt.total_seconds()
test["new_group"] = (test["time_diff"] > 10).cumsum()

print("Number of groups:", test["new_group"].nunique())
print("Groups with >1 track:", (test["new_group"].value_counts() > 1).sum())

# Let's check train set with same logic
train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train = train.sort_values("dt")
train["time_diff"] = train["dt"].diff().dt.total_seconds()
train["new_group"] = (train["time_diff"] > 10).cumsum()

print("\nTrain Number of groups:", train["new_group"].nunique())
print("True primary_observation_id:", train["primary_observation_id"].nunique())

# How pure are these groups in train?
purity = []
for name, group in train.groupby("new_group"):
    if len(group) > 1:
        purity.append((group["bird_group"].nunique() == 1))
print(f"Purity of >1 groups: {np.mean(purity):.3f}")

