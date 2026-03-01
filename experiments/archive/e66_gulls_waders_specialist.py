"""E66: Learned Gulls↔Waders correction layer (physics-driven).

Motivation
----------
Rule-based post-processing for Gulls↔Waders (E58) hurt LB. Instead of hard
thresholds (airspeed gates), learn a *pairwise specialist* on train data and
use it to redistribute probability mass between the two classes:

  S = p(Gulls) + p(Waders)
  p'(Waders) = S * s(x)
  p'(Gulls)  = S * (1 - s(x))

where s(x) is a learned estimate of P(Waders | x) using only non-temporal,
physics/kinematics features.

This preserves total mass assigned to the pair and only changes within-pair
ranking, which can improve macro-AP without destabilizing other classes.

Pipeline
--------
1) Train a binary LGBM specialist using LOMO-by-month CV on {Gulls,Waders}.
2) Predict s(x) for all train/test samples (OOF per month for train).
3) Apply correction to base predictions from E50:
   - OOF: oof_e50.npy  (LOMO mAP ~0.3625)
   - Test: test_e50.npy
4) Apply the proven unseen-month GBIF prior tilt (E54) after correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.features import ALL_TEMPORAL, build_features  # noqa: E402
from src.metrics import compute_map  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Base (E54) unseen-month prior strengths that reached LB 0.56.
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    """Build month priors from GBIF seasonal indices (as used in E38/E58)."""
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


def apply_bayes_month_alphas(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
) -> np.ndarray:
    out = preds.copy()
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0 or alpha == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


def apply_gulls_waders_specialist(preds: np.ndarray, s_waders: np.ndarray) -> np.ndarray:
    """Redistribute mass between Gulls and Waders using specialist s(x)."""
    g = CLASSES.index("Gulls")
    w = CLASSES.index("Waders")
    out = preds.copy()
    s = np.clip(s_waders.astype(float), 0.0, 1.0)
    S = out[:, g] + out[:, w]
    out[:, w] = S * s
    out[:, g] = S * (1.0 - s)
    return renorm_rows(out)


print("=" * 70, flush=True)
print("E66 GULLS↔WADERS SPECIALIST CORRECTION".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(np.unique(train_months))

g_idx = CLASSES.index("Gulls")
w_idx = CLASSES.index("Waders")

# Load base predictions (E50) for evaluation and submission generation.
oof_base = np.load(ROOT / "oof_e50.npy").astype(float)
test_base = np.load(ROOT / "test_e50.npy").astype(float)
oof_base = renorm_rows(oof_base)
test_base = renorm_rows(test_base)

# Compute training priors.
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Build non-temporal physics/kinematics features for specialist.
print("\nBuilding features for specialist (no temporal)...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
X_train_df = build_features(train_df, feature_sets=feat_sets)
X_test_df = build_features(test_df, feature_sets=feat_sets)
keep_cols = [c for c in X_train_df.columns if c not in ALL_TEMPORAL]
X_train_df = X_train_df[keep_cols]
X_test_df = X_test_df[keep_cols]

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)
fn = list(X_train_df.columns)

# Specialist training subset (true labels only).
mask_gw = (y == g_idx) | (y == w_idx)
y_gw = (y[mask_gw] == w_idx).astype(int)
idx_gw = np.where(mask_gw)[0]

print(f"  Specialist train subset: {mask_gw.sum()} rows (Gulls={np.sum(y==g_idx)}, Waders={np.sum(y==w_idx)})", flush=True)
print(f"  Train months: {unique_months}", flush=True)

# LOMO CV for specialist.
s_train = np.zeros(len(y), dtype=float)
s_test_acc = np.zeros(len(X_test), dtype=float)

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

print("\nTraining specialist (LOMO by month)...", flush=True)
for m in unique_months:
    va_all = np.where(train_months == m)[0]
    tr_all = np.where(train_months != m)[0]

    # Restrict training to GW subset within training months.
    tr_idx = tr_all[mask_gw[tr_all]]
    va_idx = va_all  # predict for all rows in the held-out month

    y_tr = (y[tr_idx] == w_idx).astype(int)
    X_tr = X_train[tr_idx]
    X_va = X_train[va_idx]

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int((1 - y_tr).sum()), 1)
    spw = n_neg / n_pos

    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = spw

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=(y[va_idx] == w_idx).astype(int), feature_name=fn)

    mdl = lgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    s_train[va_idx] = mdl.predict(X_va)
    s_test_acc += mdl.predict(X_test) / len(unique_months)

    # Binary AP on GW subset in this month (diagnostic).
    va_gw = va_all[mask_gw[va_all]]
    if len(va_gw) > 0:
        ap = average_precision_score((y[va_gw] == w_idx).astype(int), s_train[va_gw])
        print(f"  Month {m}: GW-AP={ap:.4f} (n_gw={len(va_gw)})", flush=True)
    else:
        print(f"  Month {m}: GW-AP=NA (no gw samples)", flush=True)

ap_all = average_precision_score(y_gw, s_train[idx_gw])
print(f"\nSpecialist overall AP (Waders vs Gulls): {ap_all:.4f}", flush=True)

# Evaluate effect on LOMO OOF mAP (train).
oof_corr = apply_gulls_waders_specialist(oof_base, s_train)
base_map, _ = compute_map(y, oof_base)
corr_map, _ = compute_map(y, oof_corr)
print(f"\nLOMO mAP on train:", flush=True)
print(f"  Base (E50 OOF): {base_map:.4f}", flush=True)
print(f"  + GW specialist: {corr_map:.4f}  (delta {corr_map - base_map:+.4f})", flush=True)

# Generate two test submissions:
# A) specialist -> unseen-month priors (E54)
# B) unseen-month priors (E54) -> specialist
print("\nGenerating test submissions...", flush=True)
test_a = apply_gulls_waders_specialist(test_base, s_test_acc)
test_a = apply_bayes_month_alphas(test_a, test_months, p_train, priors, BASE_ALPHA)
test_a = renorm_rows(test_a)
save_submission(test_a, "e66_gw_specialist_then_priors", cv_map=None)

test_b = apply_bayes_month_alphas(test_base, test_months, p_train, priors, BASE_ALPHA)
test_b = renorm_rows(test_b)
test_b = apply_gulls_waders_specialist(test_b, s_test_acc)
save_submission(test_b, "e66_priors_then_gw_specialist", cv_map=None)

print("\nDone.", flush=True)

