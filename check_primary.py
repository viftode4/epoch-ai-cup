import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])

groups = train.groupby("primary_observation_id")
count = 0
for name, group in groups:
    if len(group) > 2:
        print(f"\nGroup {name}: {len(group)} tracks")
        print(group[["dt", "bird_group", "radar_bird_size", "airspeed", "min_z"]].to_string())
        count += 1
        if count >= 5:
            break
