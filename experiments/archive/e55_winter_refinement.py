"""E55: Winter-focused refinement around the current best E54 winter tilt.

Signal from Kaggle:
- E53 uniform unseen prior (0.15) -> 0.55 LB
- E54 winter-tilt (m2=0.22,m5=0.12,m12=0.24) -> 0.56 LB
- E54 spring-tilt underperformed

This script explores a local neighborhood:
1) Balanced stronger winter
2) Aggressive stronger winter
3) Aggressive + confidence gating (adjust only uncertain samples)
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


def apply_month_alphas_confidence_gated(preds, months, p_train, priors, alpha_map, tau=0.52):
    """Scale alpha by uncertainty: low-confidence rows get stronger correction."""
    out = preds.copy()
    conf = out.max(axis=1)
    gate = np.clip((tau - conf) / max(tau, 1e-6), 0.0, 1.0)
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0 or alpha == 0:
            continue
        # Row-wise effective alpha.
        eff_alpha = alpha * gate[mask]
        ratio_base = priors[month] / np.maximum(p_train, 1e-12)
        # Apply per-row exponent.
        ratio = np.power(ratio_base[np.newaxis, :], eff_alpha[:, np.newaxis])
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


print("=" * 72, flush=True)
print("E55 WINTER REFINEMENT".center(72), flush=True)
print("=" * 72, flush=True)

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

base_dist = np.bincount(test_base.argmax(axis=1), minlength=len(CLASSES))
print("\nBase E50 distribution:", flush=True)
print("  " + " ".join(f"{cls}:{int(base_dist[i])}" for i, cls in enumerate(CLASSES)), flush=True)

variants = [
    (
        "winter_balanced",
        {"m2": 0.24, "m5": 0.10, "m12": 0.26},
        False,
    ),
    (
        "winter_stronger",
        {"m2": 0.26, "m5": 0.10, "m12": 0.30},
        False,
    ),
    (
        "winter_stronger_gated",
        {"m2": 0.26, "m5": 0.10, "m12": 0.30},
        True,
    ),
]

print("\nGenerating refinement variants...", flush=True)
for name, cfg, gated in variants:
    alpha_map = {2: cfg["m2"], 5: cfg["m5"], 12: cfg["m12"]}
    if gated:
        pred = apply_month_alphas_confidence_gated(
            test_base, test_months, p_train, priors, alpha_map, tau=0.52
        )
    else:
        pred = apply_month_alphas(test_base, test_months, p_train, priors, alpha_map)
    pred = renorm_rows(pred)

    dist_full = np.bincount(pred.argmax(axis=1), minlength=len(CLASSES))
    unseen_mask = np.isin(test_months, [2, 5, 12])
    dist_unseen = np.bincount(pred[unseen_mask].argmax(axis=1), minlength=len(CLASSES))
    print(
        f"\n  {name}: m2={cfg['m2']:.2f}, m5={cfg['m5']:.2f}, m12={cfg['m12']:.2f}, gated={gated}",
        flush=True,
    )
    print(
        f"    unseen -> Gulls:{int(dist_unseen[CLASSES.index('Gulls')])} "
        f"Waders:{int(dist_unseen[CLASSES.index('Waders')])} "
        f"Songbirds:{int(dist_unseen[CLASSES.index('Songbirds')])} "
        f"Pigeons:{int(dist_unseen[CLASSES.index('Pigeons')])}",
        flush=True,
    )
    print(
        f"    full   -> Gulls:{int(dist_full[CLASSES.index('Gulls')])} "
        f"Waders:{int(dist_full[CLASSES.index('Waders')])} "
        f"Pigeons:{int(dist_full[CLASSES.index('Pigeons')])}",
        flush=True,
    )

    suffix = (
        f"m2_{cfg['m2']:.2f}_m5_{cfg['m5']:.2f}_m12_{cfg['m12']:.2f}"
        + ("_gated" if gated else "")
    )
    save_submission(pred, f"e55_{name}_{suffix}", cv_map=None)

print("\nDone.", flush=True)
