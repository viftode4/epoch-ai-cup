"""E58: Winter airspeed-gated Gulls↔Waders adjustment (post-processing).

Motivation from Kaggle evidence:
- Unseen-month prior adjustment helped (0.55 -> 0.56) but is now plateauing.
- Remaining errors likely come from *within-month* ambiguities, especially Gulls vs Waders.

Insight from best winter-tilt submission (E54):
- In months 2 and 12, cases where (top1=Gulls, top2=Waders) have lower airspeed
  than cases where (top1=Waders, top2=Gulls).

Hypothesis:
  For winter months, when a track is fast and model is unsure between Gulls/Waders,
  it is more often a Wader. We can boost Waders only for those ambiguous cases.

This script starts from the winning baseline adjustment:
  m2=0.22, m5=0.12, m12=0.24 (GBIF Bayes prior)
then applies an additional gated boost.
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

BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}

# Gating hyperparameters derived from exploratory stats on E54 predictions.
MARGIN_TAU = 0.15
SPEED_TAU = {2: 15.5, 12: 14.0}  # m/s

# Candidate boost factors for Waders in gated cases.
CANDIDATES = [
    {"name": "k135", "k": 1.35},
    {"name": "k155", "k": 1.55},
]


def renorm_rows(pred):
    pred = np.clip(pred, 1e-12, None)
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


def apply_bayes_month_alphas(preds, months, p_train, priors, alpha_map):
    out = preds.copy()
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0 or alpha == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


def apply_winter_gate(preds, months, airspeed, k, margin_tau=MARGIN_TAU):
    cols = CLASSES[:]  # submission uses sample-sub column order, but we operate in CLASSES order
    g = cols.index("Gulls")
    w = cols.index("Waders")

    out = preds.copy()
    changed = 0
    for m in [2, 12]:
        mask_m = months == m
        if mask_m.sum() == 0:
            continue
        Pm = out[mask_m]
        order = np.argsort(-Pm, axis=1)
        t1 = order[:, 0]
        t2 = order[:, 1]
        margins = Pm[:, g] - Pm[:, w]
        gate = (
            (t1 == g)
            & (t2 == w)
            & (margins < margin_tau)
            & (airspeed[mask_m] >= SPEED_TAU[m])
        )
        if gate.any():
            idxs = np.where(mask_m)[0][gate]
            out[idxs, w] *= k
            out[idxs] = out[idxs] / np.clip(out[idxs].sum(axis=1, keepdims=True), 1e-12, None)
            changed += len(idxs)
    return out, changed


print("=" * 70, flush=True)
print("E58 WINTER AIRSPEED-GATED ADJUSTMENT".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
airspeed = test_df["airspeed"].values.astype(float)

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

base = np.load(ROOT / "test_e50.npy")
base = apply_bayes_month_alphas(base, test_months, p_train, priors, BASE_ALPHA)
base = renorm_rows(base)

print("\nBaseline (E54 winter tilt) files exist; this regenerates from test_e50.npy.", flush=True)

def month_dist(P, m):
    mask = test_months == m
    a = P[mask].argmax(axis=1)
    d = np.bincount(a, minlength=len(CLASSES))
    return {CLASSES[i]: int(d[i]) for i in range(len(CLASSES))}

for m in [2, 12]:
    d = month_dist(base, m)
    print(f"  month {m}: Gulls={d['Gulls']}, Waders={d['Waders']}, Geese={d['Geese']}, Songbirds={d['Songbirds']}", flush=True)

print("\nGenerating candidates (recommended submit order: k135 -> k155).", flush=True)
for cfg in CANDIDATES:
    pred, n_changed = apply_winter_gate(base, test_months, airspeed, k=cfg["k"])
    pred = renorm_rows(pred)
    d2 = month_dist(pred, 2)
    d12 = month_dist(pred, 12)
    print(
        f"\n  {cfg['name']}: k={cfg['k']:.2f}, changed_rows={n_changed}\n"
        f"    month2:  Gulls={d2['Gulls']}, Waders={d2['Waders']}\n"
        f"    month12: Gulls={d12['Gulls']}, Waders={d12['Waders']}",
        flush=True,
    )
    save_submission(pred, f"e58_winter_airspeed_gate_{cfg['name']}", cv_map=None)

print("\nDone.", flush=True)

