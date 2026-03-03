import numpy as np
import pandas as pd
from src.data import CLASSES
from src.metrics import compute_map
from experiments.e96_nbalt_heading_ac1 import build_gbif_priors, apply_gated_ratio_priors

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

train_months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values

# Apply prior tilt to shared months (9, 10)
alpha_map = {9: 0.15, 10: 0.15}
oof_adj, changed = apply_gated_ratio_priors(oof, train_months, p_train, priors, alpha_map, tau=0.15)

m_adj, _ = compute_map(y, oof_adj)
print(f"Shared-month prior OOF mAP: {m_adj:.4f} (Changed {changed} rows)")

# What if we don't gate it?
def apply_ungated_ratio_priors(preds, months, p_train, priors, alpha_map):
    out = preds.copy()
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0: continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out

oof_ungated = apply_ungated_ratio_priors(oof, train_months, p_train, priors, alpha_map)
m_ungated, _ = compute_map(y, oof_ungated)
print(f"Ungated shared-month prior OOF mAP: {m_ungated:.4f}")

