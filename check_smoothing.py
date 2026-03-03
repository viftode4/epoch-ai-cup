import numpy as np
import pandas as pd
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes

# Load E79 LOMO OOF
oof = np.load("oof_e50.npy")
m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

# Sort by time
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train["idx"] = np.arange(len(train))
train_sorted = train.sort_values("dt")

oof_sorted = oof[train_sorted["idx"]]

# Apply rolling average smoothing
for window_s in [10, 30, 60, 120, 300]:
    smoothed_oof = np.zeros_like(oof_sorted)
    times = train_sorted["dt"].values
    
    for i in range(len(times)):
        t = times[i]
        # Find indices within window_s seconds
        mask = np.abs((times - t).astype('timedelta64[s]').astype(float)) <= window_s
        # Average the probabilities
        smoothed_oof[i] = np.mean(oof_sorted[mask], axis=0)
        
    # Re-sort to original indices
    final_oof = np.zeros_like(oof)
    final_oof[train_sorted["idx"]] = smoothed_oof
    
    m_smooth, _ = compute_map(y, final_oof)
    print(f"Smoothed ({window_s}s) mAP: {m_smooth:.4f} (Delta: {m_smooth - m:+.4f})")
