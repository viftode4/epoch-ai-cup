import pandas as pd
import numpy as np

test_df = pd.read_csv("data/test.csv")
test_df["dt"] = pd.to_datetime(test_df["timestamp_start_radar_utc"])
time_s = test_df["dt"].astype(int).values / 10**6

n_neighbors = np.zeros(len(test_df))
for i in range(len(test_df)):
    mask = (np.abs(time_s - time_s[i]) <= 60)
    mask[i] = False
    n_neighbors[i] = mask.sum()

print(f"Test tracks with >0 neighbors: {(n_neighbors > 0).sum()} / {len(test_df)}")
print(f"Mean neighbors: {n_neighbors.mean():.2f}")
print(f"Max neighbors: {n_neighbors.max()}")

i_max = np.argmax(n_neighbors)
print(f"Track with max neighbors: {test_df.loc[i_max, 'dt']}")
mask = (np.abs(time_s - time_s[i_max]) <= 60)
print(f"Neighbors min time: {test_df.loc[mask, 'dt'].min()}")
print(f"Neighbors max time: {test_df.loc[mask, 'dt'].max()}")
