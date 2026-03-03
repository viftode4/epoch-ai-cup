import numpy as np
import pandas as pd
from src.data import CLASSES
from src.metrics import compute_map
from experiments.e96_nbalt_heading_ac1 import build_gbif_priors

train = pd.read_csv("data/train.csv")
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes
oof = np.load("oof_e50.npy")

m, _ = compute_map(y, oof)
print(f"Base OOF mAP: {m:.4f}")

counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

train_months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values

def ipf_adjust(preds, target_marginal, max_iter=50):
    q = preds.copy()
    for _ in range(max_iter):
        # Match column marginals
        col_sums = q.mean(axis=0)
        q = q * (target_marginal / np.clip(col_sums, 1e-12, None))
        # Match row marginals
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
    return q

oof_ipf = oof.copy()
for month in np.unique(train_months):
    mask = train_months == month
    if mask.sum() > 0:
        oof_ipf[mask] = ipf_adjust(oof[mask], priors[month])

m_ipf, _ = compute_map(y, oof_ipf)
print(f"IPF OOF mAP: {m_ipf:.4f} (Delta: {m_ipf - m:+.4f})")

# What if we only do 1 iteration? (This is basically prior tilt)
def ipf_1iter(preds, target_marginal):
    q = preds.copy()
    col_sums = q.mean(axis=0)
    q = q * (target_marginal / np.clip(col_sums, 1e-12, None))
    q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
    return q

oof_ipf1 = oof.copy()
for month in np.unique(train_months):
    mask = train_months == month
    if mask.sum() > 0:
        oof_ipf1[mask] = ipf_1iter(oof[mask], priors[month])

m_ipf1, _ = compute_map(y, oof_ipf1)
print(f"IPF (1 iter) OOF mAP: {m_ipf1:.4f} (Delta: {m_ipf1 - m:+.4f})")

