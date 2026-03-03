import numpy as np
import pandas as pd
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

# Smooth by primary_observation_id
smoothed_oof = oof.copy()
for obs_id, group in train.groupby("primary_observation_id"):
    idx = group.index
    if len(idx) > 1:
        smoothed_oof[idx] = np.mean(oof[idx], axis=0)

m_smooth, _ = compute_map(y, smoothed_oof)
print(f"Smoothed by obs_id mAP: {m_smooth:.4f} (Delta: {m_smooth - m:+.4f})")

# What if we take the MAX instead of MEAN?
max_oof = oof.copy()
for obs_id, group in train.groupby("primary_observation_id"):
    idx = group.index
    if len(idx) > 1:
        max_oof[idx] = np.max(oof[idx], axis=0)
        # renormalize
        max_oof[idx] = max_oof[idx] / np.sum(max_oof[idx], axis=1, keepdims=True)

m_max, _ = compute_map(y, max_oof)
print(f"Max by obs_id mAP: {m_max:.4f} (Delta: {m_max - m:+.4f})")

