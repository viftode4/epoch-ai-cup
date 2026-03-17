"""E162: Physics-only features — no weather, no solar, no temporal.

Only model bird flight behaviour from the raw radar signal.
This should generalise to unseen months (Feb/May/Dec) because nothing
in the feature set encodes "which month is this".

Changes vs E79:
  - DROP all 11 weather/solar features
  - DROP all temporal features (already banned, but also targeted ones)
  - ADD new raw signal features (heading_entropy, turn_persistence, etc.)
  - ADD best existing separators from analysis (heading_R, turn_angle_p90, etc.)
  - Subsample Gulls to reduce dominance (57.8% -> ~25%)
  - Same E79 HPs, same ensemble approach
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
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

# ── Feature selection: physics only ──────────────────────────────

# E79's 25 pure radar/trajectory features (no weather/solar)
E79_RADAR = [
    "alt_max", "alt_median", "alt_q75",
    "rcs_mean", "rcs_median", "rcs_q25", "rcs_q75",
    "speed_median", "avg_ground_speed", "accel_std",
    "bearing_change_mean", "lon_mean", "lat_mean", "lon_std", "lat_std",
    "alt_change_halves", "speed_x_alt", "curvature_mean",
    "slow_flight_frac", "alt_rate_mean", "rcs_per_alt",
    "airspeed", "airspeed_vs_ground", "size_x_alt", "rcs_for_size",
]

# New features from our analysis, cherry-picked per class:
# - Multi-class: speed_below_10_frac (4 cls), heading_R (2), heading_entropy (2),
#                turn_persistence (3), straightness (2)
# - Cormorants: phys_traj_aspect_ratio, turn_reversal_rate, is_large, rcs_max, spatial_spread
# - Clutter: turn_angle_p90, rcs_min, turn_angle_var, rcs_stability, alt_profile_var, rcs_mod_median
# - BoP: path_loop_fraction, solitary_slow, airspeed_low, soaring_frac, speed_alt_coupling
# - Ducks: airspeed_high, dist_per_point, speed_min, rcs_x_alt
# - Pigeons: rcs_spectral_entropy, rcs_mod_consistency
# - Geese: is_small_bird, radar_bird_size, size_x_airspeed
# - Waders: alt_ascending_frac, rp_soaring_score, curvature_max
# - Songbirds: vert_speed_kurt, alt_profile_var (shared)
NEW_SIGNAL = [
    # Multi-class separators
    "speed_below_10_frac", "heading_R", "heading_entropy",
    "turn_persistence", "straightness",
    # Per-class separators
    "turn_angle_p90", "turn_angle_var", "rcs_min", "rcs_max",
    "rcs_stability", "rcs_spectral_entropy",
    "phys_traj_aspect_ratio", "turn_reversal_rate",
    "path_loop_fraction", "dist_per_point", "speed_min",
    "effective_speed_ratio", "lin_error_mean",
    "alt_ascending_frac", "alt_descending_frac", "alt_flat_frac",
    "soaring_frac", "rp_soaring_score", "curvature_max",
    "alt_profile_var", "rcs_mod_median", "rcs_mod_consistency",
    "speed_alt_coupling", "speed_jerk_std",
    "vert_speed_skew", "vert_speed_kurt",
    "speed_autocorr", "rcs_burst_frac", "rcs_smooth_frac",
    # Radar size indicators (from targeted, non-temporal)
    "airspeed_high", "airspeed_low",
    "radar_bird_size", "is_small_bird", "is_medium", "is_large", "is_flock",
    "size_x_airspeed",
    "rcs_x_alt", "spatial_spread",
]

KEEP_FEATURES = E79_RADAR + [f for f in NEW_SIGNAL if f not in E79_RADAR]

SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Gulls subsample ratio
GULL_SUBSAMPLE = 0.45  # ~675 Gulls from 1503, still largest class


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)


# ====================================================================
print("=" * 70, flush=True)
print("E162 PHYSICS-ONLY FEATURES".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data and build features -----------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Use all extractors that produce physics features
print("\nBuilding features...", flush=True)
feat_sets = [
    "core", "rcs_fft", "tabular", "targeted",
    "flight_mode", "weakclass",
    "rcs_slope", "trajectory_separators",
    "radar_physics", "flight_physics",
    "enhanced_bio_shape", "linearity",
    "raw_signal",
]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
print(f"  Raw features: {train_feats.shape[1]}", flush=True)

# Remove ALL temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Also remove explicitly temporal targeted features
TEMPORAL_TARGETED = [
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring",
    "hour_bin_3h", "timestamp_duration",
]
drop_temporal = [c for c in TEMPORAL_TARGETED if c in train_feats.columns]
if drop_temporal:
    train_feats = train_feats.drop(columns=drop_temporal)
    test_feats = test_feats.drop(columns=drop_temporal)
    print(f"  Dropped {len(drop_temporal)} temporal targeted features", flush=True)

# Prune to selected physics features
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
missing = [f for f in KEEP_FEATURES if f not in train_feats.columns]
if missing:
    print(f"  Missing ({len(missing)}): {missing}", flush=True)

# Check what we're actually using
from_e79 = [f for f in available if f in E79_RADAR]
from_new = [f for f in available if f not in E79_RADAR]
print(f"  E79 radar features: {len(from_e79)}/25", flush=True)
print(f"  New signal features: {len(from_new)}", flush=True)
print(f"  Total: {len(available)}", flush=True)

train_feats = train_feats[available]
test_feats = test_feats[available]

X_full = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
feat_names = available

# -- Class weights + Gull subsampling --------------------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights_full = class_weights_arr[y]

gull_idx = CLASSES.index("Gulls")
print(f"\n  Gulls: {int(counts[gull_idx])} samples ({counts[gull_idx]/len(y)*100:.1f}%)", flush=True)
print(f"  Gull subsample ratio: {GULL_SUBSAMPLE}", flush=True)

# -- SKF CV with Gull subsampling ----------------------------------
print("\n--- SKF ensemble training (5-fold, Gulls subsampled) ---", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
    # Subsample Gulls in training set
    tr_gulls = tr_idx[y[tr_idx] == gull_idx]
    tr_other = tr_idx[y[tr_idx] != gull_idx]
    rng = np.random.RandomState(SEED + fold_i)
    n_keep = int(len(tr_gulls) * GULL_SUBSAMPLE)
    tr_gulls_sub = rng.choice(tr_gulls, size=n_keep, replace=False)
    tr_sub = np.concatenate([tr_other, tr_gulls_sub])
    tr_sub.sort()

    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_sub)} (Gulls {n_keep}/{len(tr_gulls)}) val={len(va_idx)}", flush=True)

    X_tr = X_full[tr_sub]
    y_tr = y[tr_sub]
    sw_tr = sample_weights_full[tr_sub]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_full[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X_full[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_full[va_idx], y[va_idx])],
            sample_weight=sw_tr, verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_full[va_idx])
    test_xgb += xgb.predict_proba(X_test) / N_FOLDS

    # CatBoost
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
        border_count=128, loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_full[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X_full[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

# Individual model scores
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# -- Weight optimization on SKF OOF --------------------------------
print("\n--- Ensemble weight optimization ---", flush=True)
best_w = None
best_ens_map = -1.0
for w_lgb in np.arange(0.0, 1.05, 0.05):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.05):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01:
            continue
        oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        m, _ = compute_map(y, oof_ens)
        if m > best_ens_map:
            best_ens_map = m
            best_w = (w_lgb, w_xgb, w_cb)

w_lgb, w_xgb, w_cb = best_w
print(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Ensemble SKF mAP: {best_ens_map:.4f}", flush=True)

oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

base_map, base_per = compute_map(y, oof_ens)
print_results(base_map, base_per, label="E162 ensemble (SKF OOF)")

# -- Feature importance (top 30 from LGB) --------------------------
print("\n--- Top 30 feature importances (LGB last fold) ---", flush=True)
imp = lgb.feature_importances_
order = np.argsort(imp)[::-1]
for i, idx in enumerate(order[:30]):
    marker = " [NEW]" if feat_names[idx] not in E79_RADAR else ""
    print(f"  {i+1:3d}. {feat_names[idx]:>35s}: {imp[idx]:5d}{marker}", flush=True)

# -- Per-month breakdown -------------------------------------------
print("\n--- Per-month mAP (OOF) ---", flush=True)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
for m in sorted(np.unique(train_months)):
    mask = train_months == m
    aps = []
    for i in range(N_CLASSES):
        y_bin = (y[mask] == i).astype(int)
        if y_bin.sum() == 0:
            continue
        aps.append(average_precision_score(y_bin, oof_ens[mask, i]))
    print(f"  Month {m:2d} (n={mask.sum():4d}): mAP = {np.mean(aps):.4f}", flush=True)

# -- Binary specialists for Waders + Pigeons (SKF) ------------------
print("\n--- Training specialists (SKF, Gulls subsampled) ---", flush=True)
specialist_oof = {}
specialist_test = {}
ap_delta = {}

for cls in SPECIALIST_CLASSES:
    cidx = CLASSES.index(cls)
    y_bin = (y == cidx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
        # Subsample Gulls
        tr_gulls = tr_idx[y[tr_idx] == gull_idx]
        tr_other = tr_idx[y[tr_idx] != gull_idx]
        rng = np.random.RandomState(SEED + fold_i)
        n_keep = int(len(tr_gulls) * GULL_SUBSAMPLE)
        tr_gulls_sub = rng.choice(tr_gulls, size=n_keep, replace=False)
        tr_sub = np.concatenate([tr_other, tr_gulls_sub])
        tr_sub.sort()

        cb_spec = CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=80, task_type="GPU",
        )
        cb_spec.fit(X_full[tr_sub], y_bin[tr_sub],
                    eval_set=(X_full[va_idx], y_bin[va_idx]), verbose=0)
        oof_bin[va_idx] = cb_spec.predict_proba(X_full[va_idx])[:, 1]
        test_bin += cb_spec.predict_proba(X_test)[:, 1] / N_FOLDS

    ap_base = average_precision_score(y_bin, oof_ens[:, cidx])
    ap_spec = average_precision_score(y_bin, oof_bin)
    ap_delta[cls] = ap_spec - ap_base
    specialist_oof[cls] = oof_bin
    specialist_test[cls] = test_bin
    print(
        f"  {cls:<15s}: AP base={ap_base:.4f} | spec={ap_spec:.4f} | delta={ap_delta[cls]:+.4f}",
        flush=True,
    )

# -- Per-class alpha blend -----------------------------------------
improving = [cls for cls in SPECIALIST_CLASSES if ap_delta[cls] > 0.002]
print(f"\nImproving specialists (>0.002 AP): {improving}", flush=True)

if not improving:
    best_alpha_map = {}
    best_oof = oof_ens.copy()
    best_test = test_ens.copy()
    best_map = base_map
else:
    best_map = -1.0
    best_alpha_map = None
    best_oof = None

    for combo in itertools.product(ALPHA_GRID, repeat=len(improving)):
        alpha_map = {cls: alpha for cls, alpha in zip(improving, combo)}
        oof_blend = apply_blend(oof_ens, specialist_oof, alpha_map)
        m, _ = compute_map(y, oof_blend)
        if m > best_map:
            best_map = m
            best_alpha_map = alpha_map
            best_oof = oof_blend

    print(f"  Best alpha map: {best_alpha_map}", flush=True)
    print(f"  Best SKF mAP:   {best_map:.4f} ({best_map - base_map:+.4f})", flush=True)
    best_test = apply_blend(test_ens, specialist_test, best_alpha_map)

best_per = compute_map(y, best_oof)[1]
print_results(best_map, best_per, label="E162 final (SKF OOF)")

# -- Compare to E79 ------------------------------------------------
print(f"\n  E79 reference:     SKF 0.7736 (36 feats, 11 weather/solar)", flush=True)
print(f"  E161 reference:    SKF 0.7667 (188 feats, all extractors)", flush=True)
print(f"  E162 physics-only: SKF {best_map:.4f} ({len(available)} feats, 0 temporal)", flush=True)

# -- Save artifacts ------------------------------------------------
print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e162.npy", best_oof)
np.save(ROOT / "test_e162.npy", best_test)
save_submission(best_test, "e162_physics_only", cv_map=best_map)

# Save the feature list
with open(ROOT / "data" / "physics_features.txt", "w") as f:
    for feat in available:
        f.write(feat + "\n")
print(f"  Saved feature list: data/physics_features.txt ({len(available)} features)", flush=True)

print("\nDone.", flush=True)
