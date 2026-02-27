"""E90: Intrinsic-features-only model for label shift correction.

Build a base model using only features where P(X|Y) is approximately invariant
across months, satisfying the label shift assumption. This is a prerequisite
for principled BBSE/MLLS correction in E91.

Feature categorization of the 36 E79 features:
  INTRINSIC (21): RCS, altitude, speed, trajectory — bird physics invariant.
  SPATIAL (4): lon_mean, lat_mean, lon_std, lat_std — partial shift.
  ENVIRONMENTAL (11): wx_* (7), sol_* (4) — strongly encode month identity.

We train:
  A. 21 intrinsic features only (strict P(X|Y) invariance)
  B. 25 features (intrinsic + spatial)
  C. 36 features (E79 baseline, for comparison)

All use LGB+XGB+CB ensemble with E79 hyperparameters. Evaluated with both
SKF (5-fold) and LOMO. Saves oof_e90.npy and test_e90.npy for E91.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# ── Feature categorization ─────────────────────────────────────────
INTRINSIC_FEATURES = [
    # RCS (6)
    "rcs_mean", "rcs_median", "rcs_q25", "rcs_q75", "rcs_per_alt", "rcs_for_size",
    # Altitude (5)
    "alt_max", "alt_median", "alt_q75", "alt_change_halves", "alt_rate_mean",
    # Speed (7)
    "speed_median", "avg_ground_speed", "accel_std", "airspeed",
    "airspeed_vs_ground", "slow_flight_frac", "speed_x_alt",
    # Trajectory (3)
    "bearing_change_mean", "curvature_mean", "size_x_alt",
]

SPATIAL_FEATURES = ["lon_mean", "lat_mean", "lon_std", "lat_std"]

ENVIRONMENTAL_FEATURES = [
    # Weather (7)
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    # Solar (4)
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]

ALL_36 = INTRINSIC_FEATURES + SPATIAL_FEATURES + ENVIRONMENTAL_FEATURES

FEATURE_SETS = {
    "intrinsic_21": INTRINSIC_FEATURES,
    "intrinsic_spatial_25": INTRINSIC_FEATURES + SPATIAL_FEATURES,
    "full_36": ALL_36,
}


def add_weather_solar(train_feats, test_feats):
    """Add weather + solar features."""
    train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
    test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
    for col in train_weather.columns:
        train_feats[f"wx_{col}"] = train_weather[col].values
        test_feats[f"wx_{col}"] = test_weather[col].values
    train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
    test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
    for col in train_solar.columns:
        train_feats[f"sol_{col}"] = train_solar[col].values
        test_feats[f"sol_{col}"] = test_solar[col].values
    return train_feats, test_feats


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


# ====================================================================
print("=" * 70, flush=True)
print("E90: INTRINSIC FEATURES MODEL".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# -- Build all features (superset) ------------------------------------
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar (needed for full_36 variant)
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# -- Class weights (effective number, beta=0.999) ----------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# Store results
results = {}

for set_name, feat_list in FEATURE_SETS.items():
    available = [f for f in feat_list if f in train_feats.columns]
    missing = [f for f in feat_list if f not in train_feats.columns]
    if missing:
        print(f"\n  WARNING {set_name}: missing {missing}", flush=True)

    X = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X_test = test_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

    print(f"\n{'='*60}", flush=True)
    print(f"  {set_name.upper()} ({X.shape[1]} features)", flush=True)
    print(f"{'='*60}", flush=True)

    # ── SKF CV ──────────────────────────────────────────────────
    print(f"\n  --- SKF 5-fold ---", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold_i+1}/{N_FOLDS}", flush=True)

        # LightGBM
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
        test_lgb += lgb.predict_proba(X_test) / N_FOLDS

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                sample_weight=sample_weights[tr_idx], verbose=False)
        oof_xgb[va_idx] = xgb.predict_proba(X[va_idx])
        test_xgb += xgb.predict_proba(X_test) / N_FOLDS

        # CatBoost (E79-tuned HPs)
        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
            border_count=128, loss_function="MultiClass", eval_metric="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X[va_idx])
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Individual SKF scores
    for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
        m, _ = compute_map(y, oof)
        print(f"    {name} SKF: {m:.4f}", flush=True)

    # Weight optimization
    best_w, best_skf = None, -1.0
    for w_lgb in np.arange(0.0, 0.55, 0.05):
        for w_xgb in np.arange(0.0, 0.55, 0.05):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01 or w_cb > 1.01:
                continue
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_skf:
                best_skf = m
                best_w = (w_lgb, w_xgb, w_cb)

    w_lgb, w_xgb, w_cb = best_w
    oof_skf = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
    test_skf = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

    skf_map, skf_per = compute_map(y, oof_skf)
    print(f"    Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
    print_results(skf_map, skf_per, label=f"{set_name} SKF")

    # ── LOMO CV ─────────────────────────────────────────────────
    print(f"\n  --- LOMO ({len(unique_months)} folds) ---", flush=True)
    oof_lgb_lomo = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb_lomo = np.zeros((len(y), N_CLASSES), dtype=np.float64)

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        print(f"    Hold-out month {month}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof_lgb_lomo[va_idx] = lgb.predict_proba(X[va_idx])

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof_cb_lomo[va_idx] = cb.predict_proba(X[va_idx])

    # Find best LOMO blend weight
    best_w_lomo, best_lomo = 0.5, -1.0
    for w in np.arange(0.0, 1.05, 0.05):
        oof_ens = w * oof_lgb_lomo + (1 - w) * oof_cb_lomo
        m, _ = compute_map(y, oof_ens)
        if m > best_lomo:
            best_lomo = m
            best_w_lomo = w

    oof_lomo = best_w_lomo * oof_lgb_lomo + (1 - best_w_lomo) * oof_cb_lomo
    lomo_map, lomo_per = compute_map(y, oof_lomo)
    print(f"    Best LOMO weight: LGB={best_w_lomo:.2f} CB={1-best_w_lomo:.2f}", flush=True)
    print_results(lomo_map, lomo_per, label=f"{set_name} LOMO")

    # Per-month LOMO breakdown
    print(f"\n    Per-month LOMO breakdown:", flush=True)
    for month in unique_months:
        mask = train_months == month
        m, per = compute_map(y[mask], oof_lomo[mask])
        print(f"      Month {month:2d}: mAP={m:.4f} (n={mask.sum()})", flush=True)

    results[set_name] = {
        "skf_map": skf_map, "skf_per": skf_per,
        "lomo_map": lomo_map, "lomo_per": lomo_per,
        "oof_skf": oof_skf, "test_skf": test_skf,
        "oof_lomo": oof_lomo,
        "skf_weights": best_w, "lomo_weight": best_w_lomo,
    }

# ====================================================================
# COMPARISON
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("COMPARISON".center(70), flush=True)
print(f"{'='*70}", flush=True)

print(f"\n  {'Variant':25s} {'SKF':>8s} {'LOMO':>8s} {'Gap':>8s} {'N_feat':>6s}", flush=True)
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}", flush=True)
for name, r in results.items():
    gap = r["skf_map"] - r["lomo_map"]
    n = len(FEATURE_SETS[name])
    print(f"  {name:25s} {r['skf_map']:8.4f} {r['lomo_map']:8.4f} {gap:8.4f} {n:6d}", flush=True)

# Per-class LOMO comparison
print(f"\n  Per-class LOMO:", flush=True)
print(f"  {'Class':20s}", end="", flush=True)
for name in results:
    print(f" {name:>12s}", end="")
print()
for cls in CLASSES:
    print(f"  {cls:20s}", end="", flush=True)
    for name, r in results.items():
        print(f" {r['lomo_per'][cls]:12.4f}", end="")
    print()

# ====================================================================
# SAVE BEST INTRINSIC VARIANT FOR E91
# ====================================================================
# Use intrinsic_spatial_25 as default (better than 21 usually, still ~invariant)
# But save both for E91 to test
print(f"\n{'='*70}", flush=True)
print("SAVING ARTIFACTS".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Save intrinsic_21 OOF and test (strictest P(X|Y) invariance)
r21 = results["intrinsic_21"]
np.save(ROOT / "oof_e90_21.npy", r21["oof_skf"])
np.save(ROOT / "test_e90_21.npy", r21["test_skf"])
np.save(ROOT / "oof_e90_21_lomo.npy", r21["oof_lomo"])
print(f"  Saved oof_e90_21.npy (SKF), test_e90_21.npy, oof_e90_21_lomo.npy", flush=True)

# Save intrinsic_spatial_25
r25 = results["intrinsic_spatial_25"]
np.save(ROOT / "oof_e90_25.npy", r25["oof_skf"])
np.save(ROOT / "test_e90_25.npy", r25["test_skf"])
np.save(ROOT / "oof_e90_25_lomo.npy", r25["oof_lomo"])
print(f"  Saved oof_e90_25.npy (SKF), test_e90_25.npy, oof_e90_25_lomo.npy", flush=True)

# Primary artifacts for E91 (alias)
# Choose intrinsic_spatial_25 as default (plan says test both)
np.save(ROOT / "oof_e90.npy", r25["oof_skf"])
np.save(ROOT / "test_e90.npy", r25["test_skf"])
np.save(ROOT / "oof_e90_lomo.npy", r25["oof_lomo"])
print(f"  Saved oof_e90.npy = intrinsic_spatial_25 (default for E91)", flush=True)

# Save submissions for best intrinsic variants
save_submission(r21["test_skf"], "e90_intrinsic_21", cv_map=r21["skf_map"])
save_submission(r25["test_skf"], "e90_intrinsic_25", cv_map=r25["skf_map"])

# Also save train metadata needed by E91
np.save(ROOT / "e90_train_months.npy", train_months)
np.save(ROOT / "e90_test_months.npy", test_months)
np.save(ROOT / "e90_y.npy", y)
print(f"  Saved train/test months and labels for E91", flush=True)

print("\nDone.", flush=True)
