import pandas as pd
import numpy as np
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train["idx"] = np.arange(len(train))
train = train.sort_values("dt")
train["time_diff"] = train["dt"].diff().dt.total_seconds()

for thresh in [1, 2, 5, 10]:
    train["new_group"] = (train["time_diff"] > thresh).cumsum()
    
    smoothed_oof = np.zeros_like(oof)
    for name, group in train.groupby("new_group"):
        idx = group["idx"].values
        if len(idx) > 1:
            smoothed_oof[idx] = np.mean(oof[idx], axis=0)
        else:
            smoothed_oof[idx] = oof[idx]
            
    m_smooth, _ = compute_map(y, smoothed_oof)
    print(f"Thresh {thresh}s -> mAP: {m_smooth:.4f} (Delta: {m_smooth - m:+.4f})")

