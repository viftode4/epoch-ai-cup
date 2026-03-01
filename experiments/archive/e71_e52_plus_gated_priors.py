"""E71: Combine E52 shared-month blend with E67 gated unseen-month priors.

Scientific question
-------------------
E54/E67 hit a stable plateau at LB=0.56 by correcting unseen months (2/5/12)
via GBIF ratio-tilt priors (optionally uncertainty-gated). E70 introduced a
feature-based specialist perturbation on unseen months but did not move LB.

Hypothesis
----------
The plateau may now be limited by *shared-month* (9/10) ranking quality rather
than unseen-month correction. E52 is explicitly designed to improve shared
months by blending E42 into E50 only for Sep/Oct, while keeping unseen months
as E50.

So, if we take E52 predictions (shared-month improved) and then apply E67's
gated unseen-month priors (unseen-month improved), we might exceed the 0.56
plateau. If LB stays at 0.56, shared-month improvements are likely not the
bottleneck (or E52 does not help on the public test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.metrics import compute_map  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU = 0.15  # best-known gate from E67 (LB 0.56)


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
) -> tuple[np.ndarray, int, dict[int, int]]:
    out = preds.copy()
    order = np.argsort(-out, axis=1)
    p1 = out[np.arange(out.shape[0]), order[:, 0]]
    p2 = out[np.arange(out.shape[0]), order[:, 1]]
    margin = p1 - p2

    changed = 0
    changed_by_month: dict[int, int] = {}
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            changed_by_month[int(month)] = 0
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            changed_by_month[int(month)] = 0
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        n = int(gate.sum())
        changed += n
        changed_by_month[int(month)] = n
    return renorm_rows(out), changed, changed_by_month


print("=" * 70, flush=True)
print("E71: E52 BASE + E67 GATED UNSEEN PRIORS".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Diagnostics on OOF predictions (train-only signal; used for sanity checks).
oof50 = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
oof52 = renorm_rows(np.load(ROOT / "oof_e52.npy").astype(float))
m50, _ = compute_map(y, oof50)
m52, _ = compute_map(y, oof52)
mask_shared = np.isin(train_months, [9, 10])
m50_sh, _ = compute_map(y[mask_shared], oof50[mask_shared])
m52_sh, _ = compute_map(y[mask_shared], oof52[mask_shared])
print(f"\nOOF sanity check:", flush=True)
print(f"  E50 full mAP:     {m50:.4f}", flush=True)
print(f"  E52 full mAP:     {m52:.4f}", flush=True)
print(f"  E50 shared mAP:   {m50_sh:.4f}", flush=True)
print(f"  E52 shared mAP:   {m52_sh:.4f}", flush=True)

# Apply priors on the E52 test predictions.
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

base = renorm_rows(np.load(ROOT / "test_e52.npy").astype(float))
pred, changed, changed_by_month = apply_gated_priors(base, test_months, p_train, priors, BASE_ALPHA, tau=TAU)

print(f"\nApplied priors on unseen months with gate tau={TAU:.2f}", flush=True)
print(f"  changed_rows={changed}", flush=True)
for m in [2, 5, 12]:
    mm = test_months == m
    print(f"  month={m}: n={int(mm.sum())} changed={changed_by_month.get(int(m), 0)}", flush=True)

save_submission(
    pred,
    f"e71_e52_plus_gatedpriors_tau{TAU:.2f}",
    cv_map=m52,
)

print("\nDone.", flush=True)

