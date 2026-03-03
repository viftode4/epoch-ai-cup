import pandas as pd
train = pd.read_csv("data/train.csv")
dt_tr = pd.to_datetime(train["timestamp_start_radar_utc"])
train["hour"] = dt_tr.dt.hour
print(train[train["bird_group"] == "Pigeons"]["hour"].value_counts().sort_index())
