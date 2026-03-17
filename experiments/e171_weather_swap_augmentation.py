"""E171: Weather-Swap Augmentation for Unseen Months.

Problem: Train months [1,4,9,10], test has unseen [2,5,12].
The model sees Feb/Dec weather and panics because it never trained with it.

Solution: Augment training data by swapping weather/solar features with
randomly sampled test-month weather. The bird's trajectory features stay
real, only the environmental context changes.

This teaches the model: "Gull + February weather = still Gull"

Variants:
  A) Baseline E79-style (no augmentation)
  B) Weather-swap: duplicate each training row N times with random
     unseen-month weather from test data
  C) Weather-noise: add heavy Gaussian noise to wx/sol features
  D) Weather-dropout: randomly zero out wx/sol features during training
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

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

TD_FEATURES = [
    "td_heading_local_var", "td_speed_consistency", "td_speed_autocorr",
    "td_speed_slope", "td_alt_smoothness", "td_heading_change_rate",
    "td_rcs_trend", "td_speed_variability",
]

# Indices of weather/solar features within the feature matrix
WX_SOL_NAMES = [
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]


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
    """Train LGB+XGB+CB, return OOF + test preds + mAP."""
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

        xgb_model = XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb_model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                       sample_weight=sample_weights[tr_idx], verbose=False)
        oof_xgb[va_idx] = xgb_model.predict_proba(X[va_idx])
        test_xgb += xgb_model.predict_proba(X_test) / N_FOLDS

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

    best_w, best_map = None, 0
    for w_lgb in np.arange(0.0, 1.05, 0.1):
        for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.1):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01:
                continue
            m, _ = compute_map(y, w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb)
            if m > best_map:
                best_map = m
                best_w = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))

    print(f"    {label} weights: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}", flush=True)
    print(f"    {label} SKF mAP: {best_map:.4f}", flush=True)

    oof = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    return oof, test, best_map


def train_augmented_ensemble(X_orig, y_orig, X_test, sample_weights_orig,
                             wx_sol_indices, test_wx_sol_unseen,
                             label, skf, n_aug=2, noise_std=0.0, dropout_p=0.0):
    """Train with weather-augmented data.

    For each fold:
      1. Take fold-train rows
      2. Create n_aug copies with swapped/noised weather features
      3. Train on original + augmented, validate on original only
    """
    n_train, n_test = len(X_orig), len(X_test)
    n_wx = len(wx_sol_indices)

    oof_lgb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    test_xgb = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((n_test, N_CLASSES), dtype=np.float64)

    rng = np.random.RandomState(SEED)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_orig, y_orig)):
        X_tr = X_orig[tr_idx]
        y_tr = y_orig[tr_idx]
        sw_tr = sample_weights_orig[tr_idx]

        # Create augmented copies
        aug_X_list = [X_tr]
        aug_y_list = [y_tr]
        aug_sw_list = [sw_tr]

        for _ in range(n_aug):
            X_aug = X_tr.copy()

            if test_wx_sol_unseen is not None and noise_std == 0.0 and dropout_p == 0.0:
                # Weather-swap: replace wx/sol with random unseen-month test weather
                n_unseen = len(test_wx_sol_unseen)
                swap_idx = rng.randint(0, n_unseen, size=len(X_tr))
                for j, feat_idx in enumerate(wx_sol_indices):
                    X_aug[:, feat_idx] = test_wx_sol_unseen[swap_idx, j]

            if noise_std > 0.0:
                # Weather-noise: add Gaussian noise to wx/sol
                for feat_idx in wx_sol_indices:
                    col_std = np.std(X_tr[:, feat_idx])
                    X_aug[:, feat_idx] += rng.normal(0, col_std * noise_std, size=len(X_tr))

            if dropout_p > 0.0:
                # Weather-dropout: zero out wx/sol features randomly
                mask = rng.random(size=(len(X_tr), n_wx)) < dropout_p
                for j, feat_idx in enumerate(wx_sol_indices):
                    X_aug[mask[:, j], feat_idx] = 0.0

            aug_X_list.append(X_aug)
            aug_y_list.append(y_tr)
            aug_sw_list.append(sw_tr * 0.5)  # half weight for augmented

        X_train_aug = np.concatenate(aug_X_list, axis=0)
        y_train_aug = np.concatenate(aug_y_list, axis=0)
        sw_train_aug = np.concatenate(aug_sw_list, axis=0)

        X_val = X_orig[va_idx]
        y_val = y_orig[va_idx]

        n_real = len(tr_idx)
        n_total = len(X_train_aug)
        print(f"    Fold {fold_i+1}/{N_FOLDS}: real={n_real} aug={n_total-n_real} val={len(va_idx)}", flush=True)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(X_train_aug, y_train_aug, eval_set=[(X_val, y_val)])
        oof_lgb[va_idx] = lgb.predict_proba(X_val)
        test_lgb += lgb.predict_proba(X_test) / N_FOLDS

        xgb_model = XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb_model.fit(X_train_aug, y_train_aug, eval_set=[(X_val, y_val)],
                       sample_weight=sw_train_aug, verbose=False)
        oof_xgb[va_idx] = xgb_model.predict_proba(X_val)
        test_xgb += xgb_model.predict_proba(X_test) / N_FOLDS

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            random_strength=1.0, border_count=128,
            loss_function="MultiClass", eval_metric="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X_train_aug, y_train_aug, eval_set=(X_val, y_val), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X_val)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    best_w, best_map = None, 0
    for w_lgb in np.arange(0.0, 1.05, 0.1):
        for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.1):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01:
                continue
            m, _ = compute_map(y_orig, w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb)
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
print("E171: Weather-Swap Augmentation".center(70), flush=True)
print("=" * 70, flush=True)

# ---- Load data -------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

# ---- Build features --------------------------------------------------
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
             "weakclass", "temporal_dynamics"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# Feature selection: 36 pruned + 8 TD + turn_radius
all_keep = KEEP_FEATURES + TD_FEATURES + ["turn_radius"]
available = [f for f in all_keep if f in train_feats.columns]
print(f"  Features: {len(available)}", flush=True)

X = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Find wx/sol column indices in the feature matrix
wx_sol_indices = [available.index(f) for f in WX_SOL_NAMES if f in available]
print(f"  Weather/solar feature indices: {len(wx_sol_indices)} cols", flush=True)

# Extract unseen-month test weather for swapping
unseen_mask = np.isin(test_months, [2, 5, 12])
test_wx_sol_unseen = X_test[unseen_mask][:, wx_sol_indices]
print(f"  Unseen-month test rows for weather pool: {len(test_wx_sol_unseen)}", flush=True)

# Also get ALL test weather (including shared months) for broader pool
test_wx_sol_all = X_test[:, wx_sol_indices]

# Class weights
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ======================================================================
# Variant A: Baseline (no augmentation)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("Variant A: Baseline (no augmentation)".center(70), flush=True)
print("=" * 70, flush=True)

oof_a, test_a, map_a = train_ensemble(X, y, X_test, sample_weights, "Baseline", skf)
_, per_a = compute_map(y, oof_a)
print_results(map_a, per_a, "A: Baseline")

# ======================================================================
# Variant B: Weather-swap (unseen months only)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("Variant B: Weather-Swap (unseen month weather)".center(70), flush=True)
print("=" * 70, flush=True)

oof_b, test_b, map_b = train_augmented_ensemble(
    X, y, X_test, sample_weights, wx_sol_indices,
    test_wx_sol_unseen=test_wx_sol_unseen,
    label="WxSwap-unseen", skf=skf, n_aug=2,
)
_, per_b = compute_map(y, oof_b)
print_results(map_b, per_b, "B: Weather-Swap (unseen)")

# ======================================================================
# Variant C: Weather-swap (all test weather)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("Variant C: Weather-Swap (all test weather)".center(70), flush=True)
print("=" * 70, flush=True)

oof_c, test_c, map_c = train_augmented_ensemble(
    X, y, X_test, sample_weights, wx_sol_indices,
    test_wx_sol_unseen=test_wx_sol_all,
    label="WxSwap-all", skf=skf, n_aug=2,
)
_, per_c = compute_map(y, oof_c)
print_results(map_c, per_c, "C: Weather-Swap (all)")

# ======================================================================
# Variant D: Weather-noise (heavy gaussian noise on wx/sol)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("Variant D: Weather-Noise (std=1.0)".center(70), flush=True)
print("=" * 70, flush=True)

oof_d, test_d, map_d = train_augmented_ensemble(
    X, y, X_test, sample_weights, wx_sol_indices,
    test_wx_sol_unseen=None,
    label="WxNoise", skf=skf, n_aug=2, noise_std=1.0,
)
_, per_d = compute_map(y, oof_d)
print_results(map_d, per_d, "D: Weather-Noise")

# ======================================================================
# Variant E: Weather-dropout (50% chance to zero wx/sol)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("Variant E: Weather-Dropout (p=0.5)".center(70), flush=True)
print("=" * 70, flush=True)

oof_e, test_e, map_e = train_augmented_ensemble(
    X, y, X_test, sample_weights, wx_sol_indices,
    test_wx_sol_unseen=None,
    label="WxDropout", skf=skf, n_aug=2, dropout_p=0.5,
)
_, per_e = compute_map(y, oof_e)
print_results(map_e, per_e, "E: Weather-Dropout")

# ======================================================================
# LOMO validation for best variants
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("LOMO Validation (quick, LGB only)".center(70), flush=True)
print("=" * 70, flush=True)

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(np.unique(train_months))
rng = np.random.RandomState(SEED)

# Test baseline vs best augmented variant on LOMO
variants_for_lomo = [
    ("A-baseline", None, 0, 0.0, 0.0),
    ("B-wxswap-unseen", test_wx_sol_unseen, 2, 0.0, 0.0),
    ("D-wxnoise", None, 2, 1.0, 0.0),
    ("E-wxdropout", None, 2, 0.0, 0.5),
]

for vname, wx_pool, n_aug, noise, drop in variants_for_lomo:
    oof_lomo = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]

        if n_aug > 0:
            aug_list = [X_tr]
            y_list = [y_tr]
            for _ in range(n_aug):
                X_a = X_tr.copy()
                if wx_pool is not None:
                    n_pool = len(wx_pool)
                    swap_idx = rng.randint(0, n_pool, size=len(X_tr))
                    for j, fi in enumerate(wx_sol_indices):
                        X_a[:, fi] = wx_pool[swap_idx, j]
                if noise > 0:
                    for fi in wx_sol_indices:
                        col_std = np.std(X_tr[:, fi])
                        X_a[:, fi] += rng.normal(0, col_std * noise, size=len(X_tr))
                if drop > 0:
                    n_wx = len(wx_sol_indices)
                    mask = rng.random(size=(len(X_tr), n_wx)) < drop
                    for j, fi in enumerate(wx_sol_indices):
                        X_a[mask[:, j], fi] = 0.0
                aug_list.append(X_a)
                y_list.append(y_tr)
            X_tr = np.concatenate(aug_list, axis=0)
            y_tr = np.concatenate(y_list, axis=0)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X_tr, y_tr, eval_set=[(X[va_idx], y[va_idx])])
        oof_lomo[va_idx] = lgb.predict_proba(X[va_idx])

    lomo_map, _ = compute_map(y, oof_lomo)

    # Per-month breakdown
    per_month = []
    for month in unique_months:
        m_idx = np.where(train_months == month)[0]
        m_map, _ = compute_map(y[m_idx], oof_lomo[m_idx])
        per_month.append(f"M{month}={m_map:.4f}")

    print(f"  {vname:20s}: LOMO={lomo_map:.4f}  {' '.join(per_month)}", flush=True)

# ======================================================================
# SAVE best variant
# ======================================================================
print("\n--- Summary ---", flush=True)
results = [
    ("A-baseline", map_a, oof_a, test_a),
    ("B-wxswap-unseen", map_b, oof_b, test_b),
    ("C-wxswap-all", map_c, oof_c, test_c),
    ("D-wxnoise", map_d, oof_d, test_d),
    ("E-wxdropout", map_e, oof_e, test_e),
]

for name, m, _, _ in results:
    delta = m - map_a
    print(f"  {name:20s}: SKF={m:.4f}  delta={delta:+.4f}", flush=True)

# Save all variants
for name, m, oof, test in results:
    tag = name.split("-")[0].lower()
    np.save(ROOT / f"oof_e171_{tag}.npy", oof)
    np.save(ROOT / f"test_e171_{tag}.npy", test)

# Save best submission
best_name, best_map, _, best_test = max(results, key=lambda x: x[1])
save_submission(best_test, f"e171_{best_name}", cv_map=best_map)

# Also save baseline for comparison
save_submission(test_a, "e171_baseline", cv_map=map_a)

print(f"\nBest: {best_name} (SKF={best_map:.4f})", flush=True)
print("Done.", flush=True)
