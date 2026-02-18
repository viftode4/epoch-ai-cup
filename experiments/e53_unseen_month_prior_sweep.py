"""E53: Unseen-month prior adjustment sweep on top of E50/E52.

Applies GBIF-based Bayesian adjustment ONLY to unseen test months (2, 5, 12).
This leaves shared-month behavior mostly intact while steering difficult months.
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
UNSEEN_MONTHS = {2, 5, 12}
ALPHAS = [0.15, 0.25, 0.35, 0.50]


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def adjust_unseen_only(preds, months, priors, p_train, alpha):
    out = preds.copy()
    ratio_cache = {}
    for month in sorted(set(months)):
        if month not in UNSEEN_MONTHS:
            continue
        mask = months == month
        if month not in ratio_cache:
            ratio_cache[month] = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio_cache[month]
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


print("=" * 70, flush=True)
print("E53 UNSEEN-MONTH PRIOR SWEEP".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    month = int(row["month"])
    si = np.ones(len(CLASSES))
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_mean = gbif[cls].values.mean()
            si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    gbif_si[month] = si

priors = {}
for month in range(1, 13):
    raw = p_train * gbif_si[month]
    raw = np.maximum(raw, 1e-8)
    priors[month] = raw / raw.sum()

base_models = {
    "e50": np.load(ROOT / "test_e50.npy"),
}
e52_path = ROOT / "test_e52.npy"
if e52_path.exists():
    base_models["e52"] = np.load(e52_path)

print("\nBase argmax distributions:", flush=True)
for name, pred in base_models.items():
    dist = np.bincount(pred.argmax(axis=1), minlength=len(CLASSES))
    print(f"  {name:<6s}: " + " ".join(f"{c}:{int(dist[i])}" for i, c in enumerate(CLASSES)), flush=True)

print("\nGenerating unseen-month adjusted variants...", flush=True)
saved = []
for name, pred in base_models.items():
    for alpha in ALPHAS:
        adj = adjust_unseen_only(pred, test_months, priors, p_train, alpha)
        adj = renorm_rows(adj)
        dist = np.bincount(adj.argmax(axis=1), minlength=len(CLASSES))
        print(
            f"  {name} alpha={alpha:.2f}: "
            f"Gulls={dist[CLASSES.index('Gulls')]}, "
            f"Pigeons={dist[CLASSES.index('Pigeons')]}, "
            f"Waders={dist[CLASSES.index('Waders')]}, "
            f"Ducks={dist[CLASSES.index('Ducks')]}",
            flush=True,
        )
        out = save_submission(adj, f"e53_{name}_unseenprior_a{alpha:.2f}", cv_map=None)
        saved.append(out.name)

print(f"\nSaved {len(saved)} submissions:", flush=True)
for name in saved:
    print(f"  {name}", flush=True)
print("Done.", flush=True)
