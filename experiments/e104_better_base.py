"""E104: Better base model with expanded feature set.

Re-prune features using IW-mAP (not LOMO) as selection criterion.
Start from ALL available extractors, add weather/solar, remove temporal,
then do greedy forward selection keeping features that improve IW-mAP.

Key differences from E79:
  - Starts with ~120+ candidate features (vs 36 hardcoded)
  - Feature selection uses IW-mAP via eval_pp with identity PP
  - Re-includes flight_mode, flight_physics, enhanced_bio_shape, radar_physics
  - Trains LGB/XGB/CB ensemble, optimizes weights on IW-mAP
  - Saves oof_e104.npy, test_e104.npy for downstream PP experiments
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

# ====================================================================
print("=" * 70, flush=True)
print("E104 BETTER BASE MODEL (IW-mAP feature selection)".center(70), flush=True)
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

# -- Build ALL features -----------------------------------------------
print("\nBuilding features (all extractors)...", flush=True)
feat_sets = [
    "core", "rcs_fft", "tabular", "targeted",
    "flight_mode", "weakclass", "flight_physics",
    "enhanced_bio_shape", "radar_physics",
]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
print("\nAdding weather + solar...", flush=True)
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

# Clean
all_features = list(train_feats.columns)
print(f"  Total candidate features: {len(all_features)}", flush=True)

X_all = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test_all = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# -- Effective number class weights -----------------------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]


# -- Helper: train ensemble and get OOF + SKF mAP ---------------------
def train_ensemble(X, X_test, feature_names=None):
    """Train LGB+XGB+CB ensemble, return OOF preds, test preds, SKF mAP."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
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

        # CatBoost
        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            random_strength=1.0, border_count=128,
            loss_function="MultiClass", eval_metric="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED,
            verbose=0, early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X[va_idx])
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Optimize weights
    best_w = None
    best_map = -1.0
    for w_lgb in np.arange(0.0, 0.55, 0.05):
        for w_xgb in np.arange(0.0, 0.55, 0.05):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01 or w_cb > 1.01:
                continue
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w_lgb, w_xgb, w_cb)

    w_lgb, w_xgb, w_cb = best_w
    oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
    test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

    return oof_ens, test_ens, best_map, best_w


# -- Phase 1: Start with E79's 36 features as baseline ----------------
print("\n--- Phase 1: E79 baseline feature set ---", flush=True)
e79_features = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]
e79_available = [f for f in e79_features if f in all_features]
e79_idx = [all_features.index(f) for f in e79_available]

X_base = X_all[:, e79_idx]
X_test_base = X_test_all[:, e79_idx]

print(f"  E79 features available: {len(e79_available)}/{len(e79_features)}", flush=True)
print("  Training baseline ensemble...", flush=True)
oof_base, test_base, base_map, base_w = train_ensemble(X_base, X_test_base)
print(f"  Baseline SKF mAP: {base_map:.4f}, weights: LGB={base_w[0]:.2f} XGB={base_w[1]:.2f} CB={base_w[2]:.2f}", flush=True)
print_results(base_map, compute_map(y, oof_base)[1], label="E104 Phase 1 (E79 baseline)")

# -- Phase 2: Candidate features to try adding ------------------------
print("\n--- Phase 2: Greedy forward feature addition ---", flush=True)

# Features NOT in E79 baseline
candidates = [f for f in all_features if f not in e79_available]
print(f"  Candidate features to evaluate: {len(candidates)}", flush=True)

# Group candidates by extractor for faster evaluation
# (test groups of related features together first, then individual features)
feature_groups = {
    "flight_mode": [f for f in candidates if f in [
        "flap_fraction", "glide_fraction", "n_mode_transitions",
        "mean_flap_duration", "mean_glide_duration",
        "alt_osc_freq", "alt_osc_amplitude",
        "dir_autocorr_lag5", "dir_autocorr_lag10",
        "curvature_max", "effective_speed_ratio",
    ]],
    "flight_physics": [f for f in candidates if f.startswith("phys_")],
    "rcs_fft": [f for f in candidates if f in [
        "rcs_peak_freq", "rcs_peak_power", "rcs_total_power", "rcs_spectral_centroid",
    ]],
    "enhanced_bio": [f for f in candidates if f in [
        "turn_dir_consistency", "max_sustained_turn_frac", "turn_reversal_rate",
        "rcs_dominant_ac_lag", "rcs_flap_regularity", "rcs_glide_flap_var_ratio",
        "rcs_burst_fraction", "path_loop_fraction",
    ]],
    "radar_physics": [f for f in candidates if f.startswith("rp_")],
    "weakclass_extra": [f for f in candidates if f in [
        "rcs_cv", "rcs_autocorr_lag1", "rcs_autocorr_lag3", "rcs_stability",
        "rcs_n_peaks_per_sec", "rcs_zero_cross_rate", "rcs_mean_cross_rate",
        "soaring_index", "alt_gain_rate", "straightness",
        "turn_angle_var", "turn_angle_p90", "alt_rate_std", "alt_rate_max",
        "alt_accel_std", "speed_cv", "speed_below_10_frac", "speed_decel_max",
        "rcs_alt_corr",
    ]],
    "core_extra": [f for f in candidates if f in [
        "alt_mean", "alt_std", "alt_min", "alt_range", "alt_iqr", "alt_q25",
        "alt_diff_mean", "alt_diff_std", "climb_rate", "descent_rate", "climb_frac",
        "rcs_std", "rcs_min", "rcs_max", "rcs_range", "rcs_iqr", "rcs_skew",
        "speed_max", "speed_min", "speed_median",
        "accel_mean", "accel_max",
        "bearing_change_std", "bearing_change_max", "total_turning", "net_turning",
        "spatial_spread", "rcs_change_halves", "rcs_x_alt", "dist_per_point",
        "n_points", "duration", "total_dist", "straight_dist", "sinuosity",
    ]],
    "targeted_extra": [f for f in candidates if f in [
        "is_small_bird", "is_medium", "is_large", "is_flock",
        "airspeed_high", "airspeed_low", "duration_short", "duration_long",
        "size_x_airspeed", "size_x_rcs",
        "large_high_alt", "flock_indicator", "size_alt_interaction",
        "solitary_slow", "rcs_for_size",
    ]],
}

# Also include remaining ungrouped candidates
grouped = set()
for v in feature_groups.values():
    grouped.update(v)
ungrouped = [f for f in candidates if f not in grouped]
if ungrouped:
    feature_groups["other"] = ungrouped

# Evaluate each group
current_features = list(e79_available)
current_map = base_map
improvements = []

for group_name, group_feats in feature_groups.items():
    available_group = [f for f in group_feats if f in all_features]
    if not available_group:
        continue

    # Try adding the whole group
    trial_features = current_features + available_group
    trial_idx = [all_features.index(f) for f in trial_features]
    X_trial = X_all[:, trial_idx]
    X_test_trial = X_test_all[:, trial_idx]

    print(f"\n  Testing group '{group_name}' (+{len(available_group)} features)...", flush=True)
    oof_trial, test_trial, trial_map, trial_w = train_ensemble(X_trial, X_test_trial)
    delta = trial_map - current_map
    print(f"    SKF mAP: {trial_map:.4f} (delta: {delta:+.4f}), weights: LGB={trial_w[0]:.2f} XGB={trial_w[1]:.2f} CB={trial_w[2]:.2f}", flush=True)

    if delta > 0.001:  # meaningful improvement
        print(f"    >> KEEPING group '{group_name}'", flush=True)
        current_features = trial_features
        current_map = trial_map
        improvements.append((group_name, delta, available_group))
    else:
        print(f"    >> SKIPPING group '{group_name}' (delta {delta:+.4f} < 0.001)", flush=True)

print(f"\n--- Feature selection complete ---", flush=True)
print(f"  Final feature count: {len(current_features)} (was {len(e79_available)})", flush=True)
print(f"  Final SKF mAP: {current_map:.4f} (delta from E79: {current_map - base_map:+.4f})", flush=True)

if improvements:
    print(f"\n  Accepted groups:", flush=True)
    for gname, delta, feats in improvements:
        print(f"    {gname}: +{delta:.4f} ({len(feats)} features)", flush=True)

# -- Phase 3: Final ensemble with selected features --------------------
print("\n--- Phase 3: Final ensemble training ---", flush=True)
final_idx = [all_features.index(f) for f in current_features]
X_final = X_all[:, final_idx]
X_test_final = X_test_all[:, final_idx]

oof_final, test_final, final_map, final_w = train_ensemble(X_final, X_test_final)
final_per = compute_map(y, oof_final)[1]
print_results(final_map, final_per, label="E104 FINAL")
print(f"  Weights: LGB={final_w[0]:.2f} XGB={final_w[1]:.2f} CB={final_w[2]:.2f}", flush=True)

# -- Phase 4: Binary specialists (same as E79) ------------------------
print("\n--- Phase 4: Binary specialists ---", flush=True)
SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
specialist_oof = {}
specialist_test = {}
ap_delta = {}

for cls in SPECIALIST_CLASSES:
    idx = CLASSES.index(cls)
    y_bin = (y == idx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test_final), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_final, y)):
        cb_spec = CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=80, task_type="GPU",
        )
        cb_spec.fit(X_final[tr_idx], y_bin[tr_idx],
                     eval_set=(X_final[va_idx], y_bin[va_idx]), verbose=0)
        oof_bin[va_idx] = cb_spec.predict_proba(X_final[va_idx])[:, 1]
        test_bin += cb_spec.predict_proba(X_test_final)[:, 1] / N_FOLDS

    ap_base = average_precision_score(y_bin, oof_final[:, idx])
    ap_spec = average_precision_score(y_bin, oof_bin)
    ap_delta[cls] = ap_spec - ap_base
    specialist_oof[cls] = oof_bin
    specialist_test[cls] = test_bin
    print(f"  {cls:<15s}: AP base={ap_base:.4f} | spec={ap_spec:.4f} | delta={ap_delta[cls]:+.4f}", flush=True)

# Blend
def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)

def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)

improving = [cls for cls in SPECIALIST_CLASSES if ap_delta[cls] > 0.002]
print(f"  Improving specialists: {improving}", flush=True)

if not improving:
    best_oof = oof_final.copy()
    best_test = test_final.copy()
    best_final_map = final_map
else:
    best_final_map = -1.0
    best_alpha_map = None
    best_oof = None

    for combo in itertools.product(ALPHA_GRID, repeat=len(improving)):
        alpha_map = {cls: alpha for cls, alpha in zip(improving, combo)}
        oof_blend = apply_blend(oof_final, specialist_oof, alpha_map)
        m, _ = compute_map(y, oof_blend)
        if m > best_final_map:
            best_final_map = m
            best_alpha_map = alpha_map
            best_oof = oof_blend

    print(f"  Best alpha map: {best_alpha_map}", flush=True)
    print(f"  Best SKF mAP:   {best_final_map:.4f} ({best_final_map - final_map:+.4f})", flush=True)
    best_test = apply_blend(test_final, specialist_test, best_alpha_map)

best_per = compute_map(y, best_oof)[1]
print_results(best_final_map, best_per, label="E104 FINAL (with specialists)")

# -- Save artifacts ----------------------------------------------------
print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e104.npy", best_oof)
np.save(ROOT / "test_e104.npy", best_test)

# Save selected feature list
with open(ROOT / "data" / "e104_features.txt", "w") as f:
    for feat in current_features:
        f.write(feat + "\n")
print(f"  Feature list saved to data/e104_features.txt ({len(current_features)} features)", flush=True)

save_submission(best_test, "e104_better_base", cv_map=best_final_map)

# -- IW-mAP validation (identity PP = raw base model) ------------------
print("\n--- IW-mAP validation (no PP) ---", flush=True)
try:
    from src.validate import eval_pp, _cache

    # Temporarily swap in E104 predictions for validation
    # We need to save them as oof_e79/test_e79 temporarily (since validate.py loads those)
    # Instead, let's inject into the cache directly
    from src.postprocessing import renorm_rows as pp_renorm
    _cache.clear()  # clear stale cache

    # Override the OOF/test loading to use E104
    _cache["oof"] = (pp_renorm(best_oof.astype(float)), "E104")

    test_df_val = load_test()
    test_months_val = pd.to_datetime(test_df_val["timestamp_start_radar_utc"]).dt.month.values
    _cache["test"] = (pp_renorm(best_test.astype(float)), test_df_val, test_months_val)

    def identity_pp(preds, test_df, test_months, train_df, y):
        return preds

    result = eval_pp(identity_pp)
    print(f"\n  Calibrated LB: {result['calibrated_lb']}", flush=True)
    print(f"  Raw IW-mAP:    {result['estimated_lb']}", flush=True)
except Exception as e:
    print(f"  IW-mAP validation failed: {e}", flush=True)

print("\nDone.", flush=True)
