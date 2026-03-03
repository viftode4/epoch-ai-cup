import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.data import CLASSES
from src.submission import save_submission

# Load Mega Ensemble
df = pd.read_csv("submissions/e111_mega_ensemble_geo5_20260302_1333.csv")
preds = df[CLASSES].values

# Load Test Set
test = pd.read_csv("data/test.csv")
test["dt"] = pd.to_datetime(test["timestamp_start_radar_utc"])
test["idx"] = np.arange(len(test))

# Sort by time
test_sorted = test.sort_values("dt")
test_sorted["time_diff"] = test_sorted["dt"].diff().dt.total_seconds()

# Group by connected components where time_diff <= 1 seconds
test_sorted["new_group"] = (test_sorted["time_diff"] > 1).cumsum()

# Smooth predictions
smoothed_preds = np.zeros_like(preds)
for name, group in test_sorted.groupby("new_group"):
    idx = group["idx"].values
    if len(idx) > 1:
        smoothed_preds[idx] = np.mean(preds[idx], axis=0)
    else:
        smoothed_preds[idx] = preds[idx]

save_submission(smoothed_preds, "e112_leak_smoothed_geo5_1s", cv_map=None)
print("Saved E112 Leak Smoothed Ensemble.")
