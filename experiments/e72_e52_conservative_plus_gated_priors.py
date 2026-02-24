"""E72: Conservative shared-month blend + E67 gated unseen-month priors.

This is the conservative counterpart to E71.

We rebuild the E52 *conservative* month-aware blend (weights shrunk by 20%)
and then apply the proven E67 gated unseen-month GBIF priors (tau=0.15).

Purpose: if E71 is too aggressive on months 9/10, E72 tests a safer blend.
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

# E52 tuned weights were logged as w9=0.75, w10=0.55.
# Conservative variant shrinks by 20% (see e52_month_aware_blend.py).
W9 = 0.60
W10 = 0.44

BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU = 0.15


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def apply_month_blend(p50: np.ndarray, p42: np.ndarray, months: np.ndarray, w9: float, w10: float) -> np.ndarray:
    out = p50.copy()
    m9 = months == 9
    m10 = months == 10
    out[m9] = (1.0 - w9) * p50[m9] + w9 * p42[m9]
    out[m10] = (1.0 - w10) * p50[m10] + w10 * p42[m10]
    return renorm_rows(out)


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
print("E72: E52 CONSERVATIVE + E67 GATED UNSEEN PRIORS".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

oof42 = renorm_rows(np.load(ROOT / "oof_e42.npy").astype(float))
oof50 = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
test42 = renorm_rows(np.load(ROOT / "test_e42.npy").astype(float))
test50 = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

oof_base = apply_month_blend(oof50, oof42, train_months, w9=W9, w10=W10)
m_base, _ = compute_map(y, oof_base)
print(f"\nOOF mAP (conservative month-blend): {m_base:.4f}", flush=True)

counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

test_base = apply_month_blend(test50, test42, test_months, w9=W9, w10=W10)
pred, changed = apply_gated_priors(test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU)
print(f"Applied priors (tau={TAU:.2f}) changed_rows={changed}", flush=True)

save_submission(
    pred,
    f"e72_e52cons_w9_{W9:.2f}_w10_{W10:.2f}_plus_gatedpriors_tau{TAU:.2f}",
    cv_map=m_base,
)

print("\nDone.", flush=True)

