import pandas as pd
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
dt_tr = pd.to_datetime(train["timestamp_start_radar_utc"])
dt_te = pd.to_datetime(test["timestamp_start_radar_utc"])
print("Train hours:\n", dt_tr.dt.hour.value_counts().sort_index())
print("Test hours:\n", dt_te.dt.hour.value_counts().sort_index())
