"""E170: Per-Month Model Routing.

The 0.59 LB ceiling comes from a Pareto tradeoff:
  - Shared months (Sep/Oct = 67% test): mAP ~0.79 with weather/solar features
  - Unseen months (Feb/May/Dec = 33% test): mAP ~0.18 (model confused by wrong weather)

Solution: Route test rows by month to specialized models:
  Model A (shared): Full 36+8 features (including weather/solar) -- tuned for Sep/Oct
  Model B (unseen): Physics-only features (no wx/sol) + turn_radius -- month-invariant

Both models: LGB+XGB+CB ensemble, E79 defaults, 5-fold SKF.
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
from src.data import CLASSES, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# ---- Feature definitions ------------------------------------------------
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

TD_FEATURES = [
    "td_heading_local_var", "td_speed_consistency", "td_speed_autocorr",
    "td_speed_slope", "td_alt_smoothness", "td_heading_change_rate",
    "td_rcs_trend", "td_speed_variability",
]

# Weather/solar = month proxies (good for shared, bad for unseen)
WX_SOL_FEATURES = [
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]

# Physics-only = month-invariant (for unseen months model)
PHYSICS_FEATURES = [f for f in KEEP_FEATURES if f not in WX_SOL_FEATURES]

# New physics feature
NEW_FEATURES = ["turn_radius"]

# Train months vs test months
TRAIN_MONTHS = {1, 4, 9, 10}
SHARED_MONTHS = {9, 10}        # in both train and test
UNSEEN_MONTHS = {2, 5, 12}     # in test only


def add_weather_solar(train_feats, test_feats):
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


def train_ensemble(X, y, X_test, sample_weights, label, skf):
    """Train LGB+XGB+CB ensemble, return OOF + test preds."""
    n_train, n_test = len(X), len(X_test)

    oof_lgb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    test_xgb = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((n_test, N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
        test_lgb += lgb.predict_proba(X_test) / N_FOLDS

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

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            random_strength=1.0, border_count=128,
            loss_function="MultiClass", eval_metric="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X[va_idx])
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Weight optimization
    best_w, best_map = None, 0
    for w_lgb in np.arange(0.0, 1.05, 0.1):
        for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.1):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01:
                continue
            oof_blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            m, _ = compute_map(y, oof_blend)
            if m > best_map:
                best_map = m
                best_w = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))

    print(f"    {label} weights: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}", flush=True)
    print(f"    {label} SKF mAP: {best_map:.4f}", flush=True)

    oof = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    return oof, test, best_map


# ======================================================================
# MAIN
# ======================================================================
print("=" * 70, flush=True)
print("E170: Per-Month Model Routing".center(70), flush=True)
print("=" * 70, flush=True)

# ---- Load data -------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
print(f"  Train months: {sorted(np.unique(train_months))}", flush=True)
print(f"  Test months:  {sorted(np.unique(test_months))}", flush=True)
print(f"  Test shared (Sep/Oct): {np.isin(test_months, list(SHARED_MONTHS)).sum()}", flush=True)
print(f"  Test unseen (Feb/May/Dec): {np.isin(test_months, list(UNSEEN_MONTHS)).sum()}", flush=True)

# ---- Build features --------------------------------------------------
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
             "weakclass", "temporal_dynamics"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal leakers
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# ---- Effective number class weights ----------------------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ======================================================================
# MODEL A: Full features (for shared months Sep/Oct)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("MODEL A: Full Features (shared months)".center(70), flush=True)
print("=" * 70, flush=True)

feat_a = KEEP_FEATURES + TD_FEATURES
feat_a_avail = [f for f in feat_a if f in train_feats.columns]
print(f"  Features: {len(feat_a_avail)}", flush=True)

X_a = train_feats[feat_a_avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_a_test = test_feats[feat_a_avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

oof_a, test_a, map_a = train_ensemble(X_a, y, X_a_test, sample_weights, "ModelA", skf)
_, per_a = compute_map(y, oof_a)
print_results(map_a, per_a, "Model A (Full, SKF)")

# ======================================================================
# MODEL B: Physics-only (for unseen months Feb/May/Dec)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("MODEL B: Physics-Only (unseen months)".center(70), flush=True)
print("=" * 70, flush=True)

feat_b = PHYSICS_FEATURES + TD_FEATURES + NEW_FEATURES
feat_b_avail = [f for f in feat_b if f in train_feats.columns]
print(f"  Features: {len(feat_b_avail)} (no wx/sol, + turn_radius)", flush=True)
print(f"  Dropped: {[f for f in WX_SOL_FEATURES if f in train_feats.columns]}", flush=True)

X_b = train_feats[feat_b_avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_b_test = test_feats[feat_b_avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

oof_b, test_b, map_b = train_ensemble(X_b, y, X_b_test, sample_weights, "ModelB", skf)
_, per_b = compute_map(y, oof_b)
print_results(map_b, per_b, "Model B (Physics, SKF)")

# ======================================================================
# LOMO validation (simulates unseen months)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("LOMO Validation (per-month holdout)".center(70), flush=True)
print("=" * 70, flush=True)

unique_months = sorted(np.unique(train_months))
oof_lomo_a = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_lomo_b = np.zeros((len(y), N_CLASSES), dtype=np.float64)

for month in unique_months:
    va_idx = np.where(train_months == month)[0]
    tr_idx = np.where(train_months != month)[0]
    print(f"\n  --- Hold out month {month} (n={len(va_idx)}) ---", flush=True)

    # Model A (full features)
    lgb_a = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb_a.fit(X_a[tr_idx], y[tr_idx], eval_set=[(X_a[va_idx], y[va_idx])])
    oof_lomo_a[va_idx] = lgb_a.predict_proba(X_a[va_idx])

    # Model B (physics-only)
    lgb_b = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb_b.fit(X_b[tr_idx], y[tr_idx], eval_set=[(X_b[va_idx], y[va_idx])])
    oof_lomo_b[va_idx] = lgb_b.predict_proba(X_b[va_idx])

    # Per-month comparison
    m_a, _ = compute_map(y[va_idx], oof_lomo_a[va_idx])
    m_b, _ = compute_map(y[va_idx], oof_lomo_b[va_idx])
    delta = m_b - m_a
    better = "B" if delta > 0 else "A"
    print(f"    ModelA (full): {m_a:.4f}, ModelB (physics): {m_b:.4f}, delta={delta:+.4f} -> {better}", flush=True)

lomo_a, _ = compute_map(y, oof_lomo_a)
lomo_b, _ = compute_map(y, oof_lomo_b)
print(f"\n  Overall LOMO: ModelA={lomo_a:.4f}, ModelB={lomo_b:.4f}, delta={lomo_b-lomo_a:+.4f}", flush=True)

# ======================================================================
# ROUTING: Model A for shared months, Model B for unseen months
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("ROUTING: A(shared) + B(unseen)".center(70), flush=True)
print("=" * 70, flush=True)

test_shared_mask = np.isin(test_months, list(SHARED_MONTHS))
test_unseen_mask = np.isin(test_months, list(UNSEEN_MONTHS))

n_shared = test_shared_mask.sum()
n_unseen = test_unseen_mask.sum()
print(f"  Shared rows:  {n_shared} ({100*n_shared/len(test_df):.1f}%)", flush=True)
print(f"  Unseen rows:  {n_unseen} ({100*n_unseen/len(test_df):.1f}%)", flush=True)

# Routed test predictions
test_routed = np.zeros_like(test_a)
test_routed[test_shared_mask] = test_a[test_shared_mask]
test_routed[test_unseen_mask] = test_b[test_unseen_mask]
test_routed = renorm_rows(test_routed)

# Also try: all Model A (baseline), all Model B, various blends
# For OOF: simulate routing by using LOMO per-month holdout
# Train month 9,10 -> "shared-like", month 1,4 -> "unseen-like"
train_shared_mask = np.isin(train_months, list(SHARED_MONTHS))
train_unseen_mask = ~train_shared_mask  # months 1, 4

oof_routed_lomo = np.zeros_like(oof_lomo_a)
oof_routed_lomo[train_shared_mask] = oof_lomo_a[train_shared_mask]
oof_routed_lomo[train_unseen_mask] = oof_lomo_b[train_unseen_mask]

routed_lomo, routed_per = compute_map(y, oof_routed_lomo)
print(f"\n  LOMO routed: {routed_lomo:.4f} (A-shared + B-unseen)", flush=True)
print(f"  LOMO ModelA: {lomo_a:.4f}", flush=True)
print(f"  LOMO ModelB: {lomo_b:.4f}", flush=True)
print(f"  Delta (routed vs A): {routed_lomo - lomo_a:+.4f}", flush=True)

# Also try soft blending on unseen months
print("\n--- Soft blend on unseen: alpha * B + (1-alpha) * A ---", flush=True)
best_blend_alpha, best_blend_lomo = 0.0, 0.0
for alpha in np.arange(0.0, 1.05, 0.1):
    oof_soft = oof_lomo_a.copy()
    oof_soft[train_unseen_mask] = (1.0 - alpha) * oof_lomo_a[train_unseen_mask] + alpha * oof_lomo_b[train_unseen_mask]
    m, _ = compute_map(y, oof_soft)
    marker = " <-- best" if m > best_blend_lomo else ""
    print(f"    alpha={alpha:.1f}: LOMO={m:.4f}{marker}", flush=True)
    if m > best_blend_lomo:
        best_blend_lomo = m
        best_blend_alpha = alpha

print(f"\n  Best unseen blend: alpha={best_blend_alpha:.1f}, LOMO={best_blend_lomo:.4f}", flush=True)

# Apply best blend to test
test_final = test_a.copy()
test_final[test_unseen_mask] = (
    (1.0 - best_blend_alpha) * test_a[test_unseen_mask]
    + best_blend_alpha * test_b[test_unseen_mask]
)
test_final = renorm_rows(test_final)

# ======================================================================
# SAVE
# ======================================================================
print("\n--- Saving ---", flush=True)
np.save(ROOT / "oof_e170_model_a.npy", oof_a)
np.save(ROOT / "oof_e170_model_b.npy", oof_b)
np.save(ROOT / "test_e170_model_a.npy", test_a)
np.save(ROOT / "test_e170_model_b.npy", test_b)
np.save(ROOT / "test_e170_routed.npy", test_routed)
np.save(ROOT / "test_e170_final.npy", test_final)

# Save both submissions
save_submission(test_routed, "e170_routed_hard", cv_map=routed_lomo)
save_submission(test_final, f"e170_blend_a{best_blend_alpha:.1f}", cv_map=best_blend_lomo)
save_submission(test_a, "e170_model_a_only", cv_map=map_a)

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("E170 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  Model A (full 36+8):     SKF={map_a:.4f}  LOMO={lomo_a:.4f}", flush=True)
print(f"  Model B (physics+TR):    SKF={map_b:.4f}  LOMO={lomo_b:.4f}", flush=True)
print(f"  Routed (hard):           LOMO={routed_lomo:.4f}  delta={routed_lomo-lomo_a:+.4f}", flush=True)
print(f"  Routed (soft a={best_blend_alpha:.1f}):    LOMO={best_blend_lomo:.4f}  delta={best_blend_lomo-lomo_a:+.4f}", flush=True)
print(f"\nDone.", flush=True)
