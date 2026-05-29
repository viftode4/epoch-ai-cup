"""E176: LOMO audit of ALL existing OOF predictions.

Quick script to check LOMO (generalization) for everything we've trained.
SKF is overfit — LOMO is the honest metric.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train
from src.metrics import compute_map
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values


def eval_both(oof, name):
    """Evaluate SKF mAP and LOMO mAP."""
    oof = renorm_rows(oof)
    skf, _ = compute_map(y, oof)
    lomo_scores = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, per_class = compute_map(y[mask], oof[mask])
            lomo_scores[m] = s
    lomo = np.mean(list(lomo_scores.values()))
    gap = skf - lomo
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_scores.items()))
    print(f"  {name:<35s}  SKF={skf:.4f}  LOMO={lomo:.4f}  gap={gap:.4f}  [{month_str}]")
    return skf, lomo


print("=" * 100)
print("  E176 LOMO AUDIT — All OOF Predictions")
print("=" * 100)
print()

# E175 models
for name in ["best", "lgb", "dro", "cb", "xgb", "ranker"]:
    path = ROOT / f"oof_e175_{name}.npy"
    if path.exists():
        eval_both(np.load(path), f"E175 {name}")

print()

# E176 Phase C models
for name in ["C2_BRF", "C1_specialists", "C_extra_gbdt", "C7_smoothap_mlp"]:
    path = ROOT / f"oof_e176_{name}.npy"
    if path.exists():
        eval_both(np.load(path), f"E176 {name}")

print()

# Blends
print("--- Blends ---")
oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))

# E175 best + lgb blends
for w in [0.3, 0.5, 0.7]:
    blend = w * oof_best + (1-w) * oof_lgb
    eval_both(renorm_rows(blend), f"E175 {w:.0%}best+{1-w:.0%}lgb")

# Phase C blends
for name in ["C2_BRF", "C1_specialists", "C_extra_gbdt"]:
    path = ROOT / f"oof_e176_{name}.npy"
    if path.exists():
        oof_c = np.load(path)
        for alpha in [0.10, 0.20, 0.30]:
            blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_c)
            eval_both(renorm_rows(blend), f"E175+{name}@{alpha}")

print()

# Per-class LOMO breakdown for baseline
print("--- Per-class LOMO (E175 best) ---")
for m in sorted(set(train_months)):
    mask = train_months == m
    if mask.sum() >= 10:
        _, pc = compute_map(y[mask], oof_best[mask])
        n_per_class = {CLASSES[c]: int((y[mask] == c).sum()) for c in range(N_CLASSES)}
        low = [(k, v) for k, v in pc.items() if v < 0.5 and n_per_class[k] > 0]
        low_str = ", ".join(f"{k[:4]}={v:.2f}(n={n_per_class[k]})" for k, v in sorted(low, key=lambda x: x[1]))
        print(f"  Month {m:2d} (n={mask.sum():4d}): mAP={np.mean(list(pc.values())):.4f}  weak: {low_str}")

print("\nDone.")
