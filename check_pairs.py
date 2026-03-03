import pandas as pd
import numpy as np
from src.data import parse_ewkb_4d
from itertools import combinations

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train = train.sort_values("dt").reset_index(drop=True)

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

# Create pairs within 120 seconds
time_s = train["dt"].astype(int) / 10**6
pairs = []

# Sliding window
for i in range(len(train)):
    j = i + 1
    while j < len(train) and time_s[j] - time_s[i] <= 120:
        is_same = (train.loc[i, "primary_observation_id"] == train.loc[j, "primary_observation_id"])
        pairs.append({
            "dt": time_s[j] - time_s[i],
            "dlon": abs(mean_lon[i] - mean_lon[j]),
            "dlat": abs(mean_lat[i] - mean_lat[j]),
            "dspeed": abs(train.loc[i, "airspeed"] - train.loc[j, "airspeed"]),
            "dalt": abs(train.loc[i, "min_z"] - train.loc[j, "min_z"]),
            "same_size": train.loc[i, "radar_bird_size"] == train.loc[j, "radar_bird_size"],
            "same_group": is_same
        })
        j += 1

df_pairs = pd.DataFrame(pairs)
print(f"Total pairs within 120s: {len(df_pairs)}")
print(f"Positive pairs (same group): {df_pairs['same_group'].sum()}")
print(f"Negative pairs (diff group): {(~df_pairs['same_group']).sum()}")

# Let's see if a simple model can predict this
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X = df_pairs[["dt", "dlon", "dlat", "dspeed", "dalt", "same_size"]].fillna(-1)
y = df_pairs["same_group"].astype(int)

clf = HistGradientBoostingClassifier()
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"Pairwise ROC AUC: {np.mean(scores):.4f}")

