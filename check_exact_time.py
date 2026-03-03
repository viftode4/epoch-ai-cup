import pandas as pd
import numpy as np
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

# Group by exact timestamp
smoothed_oof = oof.copy()
for ts, group in train.groupby("timestamp_start_radar_utc"):
    idx = group.index
    if len(idx) > 1:
        smoothed_oof[idx] = np.mean(oof[idx], axis=0)

m_smooth, _ = compute_map(y, smoothed_oof)
print(f"Smoothed by exact timestamp mAP: {m_smooth:.4f} (Delta: {m_smooth - m:+.4f})")

# Group by exact timestamp + radar_bird_size
smoothed_oof2 = oof.copy()
for ts, group in train.groupby(["timestamp_start_radar_utc", "radar_bird_size"]):
    idx = group.index
    if len(idx) > 1:
        smoothed_oof2[idx] = np.mean(oof[idx], axis=0)

m_smooth2, _ = compute_map(y, smoothed_oof2)
print(f"Smoothed by exact timestamp + size mAP: {m_smooth2:.4f} (Delta: {m_smooth2 - m:+.4f})")
