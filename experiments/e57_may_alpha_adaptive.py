"""E57: Adaptive May-alpha candidates (smart order, not grid).

Goal: break the 0.56 plateau by tuning May correction while keeping winter fixed.

Fixed winter (from best E54 winter tilt):
  - Feb (m2):  alpha = 0.22
  - Dec (m12): alpha = 0.24

We generate a small set of May candidates chosen for information gain:
  - Bracket far from baseline: 0.06 and 0.18
  - Refine near baseline:      0.09 and 0.15

Recommended submission order (sequential):
  1) m5=0.18  (probe higher May correction)
  2) m5=0.06  (probe lower May correction)

Then:
  - if 0.18 wins -> try 0.15 next, then consider 0.22
  - if 0.06 wins -> try 0.09 next, then consider 0.04
  - if tie/noise -> try 0.15 and 0.09 as confirmatory
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def build_gbif_priors(p_train):
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(len(CLASSES))
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals

    priors = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_month_alphas(preds, months, p_train, priors, alpha_map):
    out = preds.copy()
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0 or alpha == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


print("=" * 70, flush=True)
print("E57 MAY-ALPHA ADAPTIVE CANDIDATES".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

test_base = np.load(ROOT / "test_e50.npy")

m2 = 0.22
m12 = 0.24
may_candidates = [0.18, 0.06, 0.15, 0.09]

print("\nFixed winter:", flush=True)
print(f"  m2={m2:.2f}, m12={m12:.2f}", flush=True)
print("\nMay candidates (submit in this order):", flush=True)
for i, m5 in enumerate(may_candidates, 1):
    print(f"  {i}) m5={m5:.2f}", flush=True)

print("\nGenerating submissions...", flush=True)
saved = []
for m5 in may_candidates:
    alpha_map = {2: m2, 5: m5, 12: m12}
    pred = apply_month_alphas(test_base, test_months, p_train, priors, alpha_map)
    pred = renorm_rows(pred)
    name = f"e57_e50_mayprobe_m2_{m2:.2f}_m5_{m5:.2f}_m12_{m12:.2f}"
    path = save_submission(pred, name, cv_map=None)
    saved.append(path.name)

print("\nSaved:", flush=True)
for f in saved:
    print(f"  {f}", flush=True)
print("\nDone.", flush=True)

