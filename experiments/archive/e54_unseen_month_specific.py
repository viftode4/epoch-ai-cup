"""E54: Month-specific unseen-month prior adjustment on top of E50.

Builds two targeted variants from the successful e53_e50_a0.15 direction:
- Variant A (spring-tilted): stronger adjustment in May.
- Variant B (winter-tilted): stronger adjustment in Feb/Dec.
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
        raw = p_train * si[month]
        raw = np.maximum(raw, 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_month_specific_adjustment(preds, months, priors, p_train, alpha_map):
    """Apply Bayesian prior adjustment using alpha per month."""
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
print("E54 UNSEEN MONTH-SPECIFIC ADJUSTMENT".center(70), flush=True)
print("=" * 70, flush=True)

test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
test_e50 = np.load(ROOT / "test_e50.npy")

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

variants = {
    # Keep successful baseline behavior for winter, push May harder.
    "spring_tilt": {2: 0.15, 5: 0.28, 12: 0.15},
    # Push winter stronger, keep May conservative.
    "winter_tilt": {2: 0.22, 5: 0.12, 12: 0.24},
}

print("\nBase E50 distribution:", flush=True)
base_dist = np.bincount(test_e50.argmax(axis=1), minlength=len(CLASSES))
print("  " + " ".join(f"{cls}:{int(base_dist[i])}" for i, cls in enumerate(CLASSES)), flush=True)

print("\nGenerating variants...", flush=True)
for name, alpha_map in variants.items():
    pred = apply_month_specific_adjustment(test_e50, test_months, priors, p_train, alpha_map)
    pred = renorm_rows(pred)

    # Quick diagnostics for unseen months
    unseen_mask = np.isin(test_months, [2, 5, 12])
    unseen_dist = np.bincount(pred[unseen_mask].argmax(axis=1), minlength=len(CLASSES))
    full_dist = np.bincount(pred.argmax(axis=1), minlength=len(CLASSES))

    print(f"\n  Variant {name}: alpha_map={alpha_map}", flush=True)
    print(
        f"    unseen -> Gulls:{int(unseen_dist[CLASSES.index('Gulls')])} "
        f"Waders:{int(unseen_dist[CLASSES.index('Waders')])} "
        f"Pigeons:{int(unseen_dist[CLASSES.index('Pigeons')])} "
        f"Songbirds:{int(unseen_dist[CLASSES.index('Songbirds')])}",
        flush=True,
    )
    print(
        f"    full   -> Gulls:{int(full_dist[CLASSES.index('Gulls')])} "
        f"Waders:{int(full_dist[CLASSES.index('Waders')])} "
        f"Pigeons:{int(full_dist[CLASSES.index('Pigeons')])}",
        flush=True,
    )

    save_submission(pred, f"e54_e50_{name}_m2_{alpha_map[2]:.2f}_m5_{alpha_map[5]:.2f}_m12_{alpha_map[12]:.2f}", cv_map=None)

print("\nDone.", flush=True)
