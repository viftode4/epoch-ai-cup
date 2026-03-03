import numpy as np
import pandas as pd
from src.data import CLASSES
from src.metrics import compute_map

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

dt = pd.to_datetime(train["timestamp_start_radar_utc"])
time_s = dt.astype(int).values / 10**9
speed = pd.to_numeric(train["airspeed"], errors="coerce").fillna(10.0).values
alt = pd.to_numeric(train["min_z"], errors="coerce").fillna(0.0).values
size = train["radar_bird_size"].values

# Compute margins
order = np.argsort(-oof, axis=1)
margin = oof[np.arange(len(oof)), order[:, 0]] - oof[np.arange(len(oof)), order[:, 1]]

TAU_LOW = 0.10
TAU_HIGH = 0.20
MAX_TIME = 60
MAX_DIST = 1.0

new_oof = oof.copy()
changed = 0

for i in range(len(train)):
    if margin[i] < TAU_LOW:
        # Find neighbors
        mask = (np.abs(time_s - time_s[i]) <= MAX_TIME) & (margin > TAU_HIGH)
        mask[i] = False
        
        # Must be same size category
        mask = mask & (size == size[i])
        
        idx = np.where(mask)[0]
        if len(idx) > 0:
            # Compute distance
            dv = np.abs(speed[idx] - speed[i]) / 5.0
            dz = np.abs(alt[idx] - alt[i]) / 50.0
            dist = dv + dz
            
            best_j = idx[np.argmin(dist)]
            if np.min(dist) < MAX_DIST:
                new_oof[i] = oof[best_j]
                changed += 1

m_new, _ = compute_map(y, new_oof)
print(f"Propagated OOF mAP: {m_new:.4f} (Delta: {m_new - m:+.4f}, Changed: {changed})")

