"""E56: Test whether May adjustment is hurting (set May alpha=0).

We keep the winning winter corrections from E54 and simply turn off May:
  m2=0.22, m5=0.00, m12=0.24

This is the most informative next probe given:
- uniform a0.15 -> 0.55 LB
- winter tilt (0.22,0.12,0.24) -> 0.56 LB
- stronger winter didn't improve beyond 0.56
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
print("E56 MAY-OFF TEST".center(70), flush=True)
print("=" * 70, flush=True)

test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
test_base = np.load(ROOT / "test_e50.npy")

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

alpha_map = {2: 0.22, 5: 0.00, 12: 0.24}
pred = apply_month_alphas(test_base, test_months, p_train, priors, alpha_map)
pred = renorm_rows(pred)

print(f"\nalpha_map={alpha_map}", flush=True)

def dist_for_month(m):
    mask = test_months == m
    a = pred[mask].argmax(axis=1)
    d = np.bincount(a, minlength=len(CLASSES))
    return {CLASSES[i]: int(d[i]) for i in range(len(CLASSES))}

for m in [2, 5, 12]:
    d = dist_for_month(m)
    print(
        f"  month {m}: Gulls={d['Gulls']}, Waders={d['Waders']}, "
        f"Songbirds={d['Songbirds']}, Geese={d['Geese']}",
        flush=True,
    )

save_submission(pred, "e56_e50_may_off_m2_0.22_m5_0.00_m12_0.24", cv_map=None)
print("\nSaved submission for Kaggle.", flush=True)
print("Done.", flush=True)

