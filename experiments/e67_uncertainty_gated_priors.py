"""E67: Uncertainty-gated unseen-month GBIF prior tilt (post-processing).

Goal
----
E54/E55 improved LB via month-specific prior tilts, but further manual tweaks
plateaued. Hypothesis: uniform prior correction still harms *confident* cases.
We should apply the correction primarily to *uncertain* predictions.

Method
------
Start from base predictions `test_e50.npy` and apply the E54 month priors only
for samples where the model is uncertain:

  margin = p_top1 - p_top2
  apply correction iff margin < tau

We generate a small set of tau candidates for Kaggle probing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}

# Candidate uncertainty thresholds (smaller = fewer rows adjusted).
TAUS = [0.05, 0.10, 0.15]


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
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


def apply_gated_priors(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
    tau: float,
) -> tuple[np.ndarray, int]:
    out = preds.copy()
    # uncertainty margin computed once
    order = np.argsort(-out, axis=1)
    p1 = out[np.arange(out.shape[0]), order[:, 0]]
    p2 = out[np.arange(out.shape[0]), order[:, 1]]
    margin = p1 - p2

    changed = 0
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


print("=" * 70, flush=True)
print("E67 UNCERTAINTY-GATED UNSEEN-MONTH PRIORS".center(70), flush=True)
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

base = np.load(ROOT / "test_e50.npy").astype(float)
base = renorm_rows(base)

def month_dist(P, m):
    mask = test_months == m
    a = P[mask].argmax(axis=1)
    d = np.bincount(a, minlength=len(CLASSES))
    return {CLASSES[i]: int(d[i]) for i in range(len(CLASSES))}

print("\nBase argmax dist (before correction):", flush=True)
for m in [2, 5, 12]:
    d = month_dist(base, m)
    print(f"  m{m}: Gulls={d['Gulls']}, Waders={d['Waders']}, Geese={d['Geese']}, Songbirds={d['Songbirds']}", flush=True)

print("\nGenerating gated candidates...", flush=True)
for tau in TAUS:
    pred, changed = apply_gated_priors(base, test_months, p_train, priors, BASE_ALPHA, tau=tau)
    d2 = month_dist(pred, 2)
    d5 = month_dist(pred, 5)
    d12 = month_dist(pred, 12)
    print(
        f"\n  tau={tau:.2f}: changed_rows={changed}\n"
        f"    m2:  Gulls={d2['Gulls']} Waders={d2['Waders']}\n"
        f"    m5:  Gulls={d5['Gulls']} Waders={d5['Waders']}\n"
        f"    m12: Gulls={d12['Gulls']} Waders={d12['Waders']}",
        flush=True,
    )
    save_submission(pred, f"e67_gatedpriors_tau{tau:.2f}", cv_map=None)

print("\nDone.", flush=True)

