"""E174: Use ALL collected data — every signal, every dataset.

Loads all 18 external CSVs, all unused columns, plus derived features.
Combines with v2 trajectory features for the most complete feature set ever tested.
Uses CB-heavy ensemble (0.15/0.05/0.80) discovered in E173 + all P0 bug fixes.

New data sources (never before in base model):
  - soil.csv: 4 cols (moisture, temperature at 2 depths)
  - weather_extra.csv: 14 cols (WMO weather code, cloud layers, snow, wind 100m)
  - Unused columns from 13 partially-used CSVs (~35 cols)
  - 4 derived domain features (cormorant wind residual, insect drift, etc.)

Total expected: ~130+ features (76 v2 base + ~55 new external + derived)
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features_v2 import build_features_v2, _load_external_csv
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
SEED = 42


# ======================================================================
# Load ALL external features — everything we collected
# ======================================================================

def load_all_external_features(
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Load EVERY unused column from every external CSV into the feature matrix.

    This adds ~55 new columns on top of what features_v2.py already loads.
    Handles circular features (sin/cos), WMO decomposition, and derived features.
    """
    n = len(df_feat)

    def _safe_load(name: str, col: str) -> np.ndarray | None:
        """Load a column from external CSV, return None if missing."""
        ext = _load_external_csv(name, split)
        if ext.empty or col not in ext.columns:
            return None
        if len(ext) != n:
            import warnings
            warnings.warn(
                f"{split}_{name}.csv has {len(ext)} rows, expected {n}. Skipping {col}.",
                stacklevel=2,
            )
            return None
        vals = ext[col].values.astype(float)
        return vals

    def _add(feat_name: str, values: np.ndarray | None):
        """Add feature if values exist and aren't already in df_feat."""
        if values is not None and feat_name not in df_feat.columns:
            df_feat[feat_name] = values

    # ── SOIL (4 cols — all new) ──
    for col in ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
                "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm"]:
        _add(col, _safe_load("soil", col))

    # ── WEATHER_EXTRA (10 raw + 4 WMO flags + 4 wind dir sin/cos) ──
    we_cols_raw = [
        "precipitation", "snowfall", "snow_depth",
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "wind_speed_100m",
    ]
    for col in we_cols_raw:
        _add(f"wxe_{col}", _safe_load("weather_extra", col))

    # WMO weather code decomposition
    wmo = _safe_load("weather_extra", "weather_code")
    if wmo is not None:
        _add("wmo_code", wmo)
        _add("wmo_fog", ((wmo >= 40) & (wmo <= 49)).astype(float))
        _add("wmo_rain", (((wmo >= 50) & (wmo <= 69)) | ((wmo >= 80) & (wmo <= 84))).astype(float))
        _add("wmo_snow", (((wmo >= 70) & (wmo <= 79)) | ((wmo >= 85) & (wmo <= 86))).astype(float))
        _add("wmo_thunder", ((wmo >= 95) & (wmo <= 99)).astype(float))

    # Wind direction at 10m and 100m (circular -> sin/cos)
    for level in ["10m", "100m"]:
        wd = _safe_load("weather_extra", f"wind_direction_{level}")
        if wd is not None:
            wd_rad = np.radians(wd)
            _add(f"wxe_wind_dir_{level}_sin", np.sin(wd_rad))
            _add(f"wxe_wind_dir_{level}_cos", np.cos(wd_rad))

    # ── CAPE (2 unused) ──
    _add("cape_jkg", _safe_load("cape", "cape_jkg"))
    _add("cin", _safe_load("cape", "cin"))

    # ── MARINE (2 unused + wave_direction sin/cos) ──
    _add("wave_period", _safe_load("marine", "wave_period"))
    wave_dir = _safe_load("marine", "wave_direction")
    if wave_dir is not None:
        wd_rad = np.radians(wave_dir)
        _add("wave_dir_sin", np.sin(wd_rad))
        _add("wave_dir_cos", np.cos(wd_rad))

    # ── TURBINES (2 unused) ──
    _add("turbines_within_1km", _safe_load("turbines", "turbines_within_1km"))
    _add("turbines_within_2km", _safe_load("turbines", "turbines_within_2km"))

    # ── LANDUSE (keep arable distance — strong Pigeon signal) ──
    # grassland_fraction_2km DROPPED: too weak (spread 0.42-0.49 across classes)
    _add("dist_to_arable_m", _safe_load("landuse", "dist_to_arable_m"))

    # ── NATURA2000 (1 unused) ──
    _add("in_natura2000", _safe_load("natura2000", "in_natura2000"))

    # ── ALTITUDE_WINDS ──
    _add("wind_at_bird_alt", _safe_load("altitude_winds", "wind_at_bird_alt"))
    # wind_80m DROPPED: r=0.987 with wind_at_bird_alt (most birds fly ~80m)
    # wind_120m DROPPED: r=1.000 with wxe_wind_speed_100m (identical data)
    _add("wind_180m", _safe_load("altitude_winds", "wind_180m"))

    # ── ERA5_WINDS (skip redundant) ──
    _add("era5_wind_shear_10_100", _safe_load("era5_winds", "era5_wind_shear_10_100"))
    # era5_wind_shear_100_500 DROPPED: identical to -wxe_wind_speed_100m (verified r=1.0)
    # era5_wind_250m, era5_wind_500m DROPPED: all zeros (pipeline didn't populate)
    _add("era5_temp_2m", _safe_load("era5_winds", "era5_temp_2m"))

    # ── PHOTOPERIOD (1 unused) ──
    _add("daylength_change_7d", _safe_load("photoperiod", "daylength_change_7d"))

    # ── PRESSURE (1 unused) ──
    _add("pressure_hpa", _safe_load("pressure", "pressure_hpa"))

    # ── VISIBILITY (load fog/rain/visibility as features too, not just PP) ──
    _add("visibility_km", _safe_load("visibility", "visibility_km"))
    _add("fog", _safe_load("visibility", "fog"))
    _add("rain_occurring", _safe_load("visibility", "rain_occurring"))

    # ── MOON (4 unused) ──
    _add("moon_altitude_deg", _safe_load("moon", "moon_altitude_deg"))
    _add("is_civil_twilight", _safe_load("moon", "is_civil_twilight"))
    _add("is_nautical_twilight", _safe_load("moon", "is_nautical_twilight"))
    _add("is_astronomical_night", _safe_load("moon", "is_astronomical_night"))

    # ── WEATHER (7 unused) ──
    for col, feat_name in [
        ("wind_u", "wx_wind_u"), ("temp_c", "wx_temp_c"),
        ("dewpoint_c", "wx_dewpoint_c"), ("sunshine_hrs", "wx_sunshine_hrs"),
        ("radiation", "wx_radiation"),
        ("precip_dur", "wx_precip_dur"), ("precip_mm", "wx_precip_mm"),
    ]:
        _add(feat_name, _safe_load("weather", col))

    # ── INSECT (as feature, not just PP evidence) ──
    _add("insect_activity_index", _safe_load("insect", "insect_activity_index"))

    # ── DERIVED FEATURES ──
    # 1. Cormorant wind residual: ground_speed = 0.70 * wind + 14.4
    #    Residual ≈ 0 for Cormorants, large for others
    airspeed = df_feat.get("airspeed")
    wind_speed = df_feat.get("wx_wind_speed")
    if airspeed is not None and wind_speed is not None:
        expected_speed = 0.70 * wind_speed + 14.4
        _add("cormorant_wind_residual", (airspeed - expected_speed).values)

    # 2. Insect drift ratio: birds >> wind (ratio >> 1); insects ≈ wind (ratio ≈ 1)
    wind_bird = df_feat.get("wind_at_bird_alt")
    if airspeed is not None and wind_bird is not None:
        safe_wind = np.maximum(wind_bird.values if hasattr(wind_bird, 'values') else wind_bird, 0.5)
        air_vals = airspeed.values if hasattr(airspeed, 'values') else airspeed
        _add("insect_drift_ratio", air_vals / safe_wind)

    # 3. True airspeed: ground_speed - headwind = wind-corrected species flight speed
    #    Removes wind contamination from airspeed column.
    #    Pigeons: 18.9 -> 13.5 (fly with tailwind), BoP: 11.8 -> 14.7 (fly into headwind)
    hw = df_feat.get("headwind_component")
    if airspeed is not None and hw is not None:
        air_vals = airspeed.values if hasattr(airspeed, 'values') else airspeed
        hw_vals = hw.values if hasattr(hw, 'values') else hw
        _add("true_airspeed", air_vals - hw_vals)

    # 4. Temp-dewpoint spread: moisture/fog indicator
    #    Small spread = moist air. Pigeons 1.3°C (moist October), Ducks 5.0°C
    temp = df_feat.get("wx_temp_c")
    dewp = df_feat.get("wx_dewpoint_c")
    if temp is not None and dewp is not None:
        t_vals = temp.values if hasattr(temp, 'values') else temp
        d_vals = dewp.values if hasattr(dewp, 'values') else dewp
        _add("temp_dewpoint_spread", t_vals - d_vals)

    # Handle inf/nan in all new features
    for col in df_feat.columns:
        if df_feat[col].dtype in [np.float64, np.float32, float]:
            df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_feat


# ======================================================================
# Data validation
# ======================================================================

def validate_features(
    train_feats: pd.DataFrame,
    test_feats: pd.DataFrame,
    n_train: int,
    n_test: int,
) -> list[str]:
    """Validate all features: row counts, NaN rates, distributions.

    Returns list of warnings.
    """
    warnings_list = []

    # Row count check
    if len(train_feats) != n_train:
        warnings_list.append(f"CRITICAL: train features {len(train_feats)} != expected {n_train}")
    if len(test_feats) != n_test:
        warnings_list.append(f"CRITICAL: test features {len(test_feats)} != expected {n_test}")

    # Column alignment
    train_only = set(train_feats.columns) - set(test_feats.columns)
    test_only = set(test_feats.columns) - set(train_feats.columns)
    if train_only:
        warnings_list.append(f"Train-only columns: {train_only}")
    if test_only:
        warnings_list.append(f"Test-only columns: {test_only}")

    # Per-column stats
    shared_cols = sorted(set(train_feats.columns) & set(test_feats.columns))
    print(f"\n  Feature validation ({len(shared_cols)} shared columns):", flush=True)
    print(f"  {'Feature':40s} {'Train NaN%':>10s} {'Test NaN%':>10s} {'Train mean':>12s} {'Train std':>12s} {'Status':>8s}", flush=True)
    print(f"  {'-'*94}", flush=True)

    n_all_zero = 0
    n_high_nan = 0
    for col in shared_cols:
        tr_vals = train_feats[col]
        te_vals = test_feats[col]
        tr_nan = tr_vals.isna().mean() * 100
        te_nan = te_vals.isna().mean() * 100
        tr_mean = tr_vals.mean()
        tr_std = tr_vals.std()

        status = "OK"
        if tr_nan > 50 or te_nan > 50:
            status = "HIGH_NAN"
            n_high_nan += 1
        elif tr_std < 1e-10:
            status = "CONSTANT"
            n_all_zero += 1

        if status != "OK":
            print(f"  {col:40s} {tr_nan:>9.1f}% {te_nan:>9.1f}% {tr_mean:>12.4f} {tr_std:>12.4f} {status:>8s}", flush=True)
            warnings_list.append(f"{col}: {status} (train NaN={tr_nan:.1f}%, test NaN={te_nan:.1f}%)")

    print(f"\n  Summary: {len(shared_cols)} features, {n_high_nan} high-NaN, {n_all_zero} constant", flush=True)
    return warnings_list


# ======================================================================
# Effective number class weights
# ======================================================================

def effective_number_weights(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    eff_num = (1.0 - beta ** counts) / (1.0 - beta)
    class_weights = 1.0 / np.maximum(eff_num, 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    return class_weights[y]


# ======================================================================
# Ensemble weight optimization
# ======================================================================

def optimize_ensemble_weights(
    oof_lgb, oof_xgb, oof_cb, y, step=0.05,
):
    best_map, best_w = -1.0, (0.50, 0.40, 0.10)
    for w1 in range(0, int(1.0 / step) + 1):
        wl = w1 * step
        for w2 in range(0, int((1.0 - wl) / step) + 1):
            wx = w2 * step
            wc = 1.0 - wl - wx
            if wc < -1e-8:
                continue
            wc = max(wc, 0.0)
            m, _ = compute_map(y, wl * oof_lgb + wx * oof_xgb + wc * oof_cb)
            if m > best_map:
                best_map = m
                best_w = (round(wl, 2), round(wx, 2), round(wc, 2))
    return (*best_w, best_map)


# ======================================================================
# Standard PP (3-channel NB)
# ======================================================================

def apply_standard_nb_pp(preds, test_df, test_months, train_df, y,
                         gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)

    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed_tr, "alt_mid": 0.5*(min_z_tr+max_z_tr), "alt_range": max_z_tr-min_z_tr}
    sl, lps, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_te, "alt_mid": 0.5*(min_z_te+max_z_te), "alt_range": max_z_te-min_z_te}
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(test_df, sl, lps, cont_te, w, None, mu, sig)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)


# ======================================================================
# Cache utilities
# ======================================================================

def _features_v2_hash():
    src = (ROOT / "src" / "features_v2.py").read_text(encoding="utf-8")
    return hashlib.sha256(src.encode()).hexdigest()[:12]

def _validate_cache(cache_path):
    meta = cache_path.with_suffix(".hash")
    h = _features_v2_hash()
    if meta.exists() and meta.read_text().strip() == h:
        return True
    return False

def _save_cache_hash(cache_path):
    cache_path.with_suffix(".hash").write_text(_features_v2_hash())


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  E174: ALL DATA — Every Signal, Every Dataset")
    print("=" * 70)

    # ── Load raw data ──
    print("\n[1/7] Loading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    # ── Build v2 base features ──
    print("\n[2/7] Building v2 base features...", flush=True)
    cache_train = ROOT / "data" / "_cached_train_features_v2.pkl"
    cache_test = ROOT / "data" / "_cached_test_features_v2.pkl"
    if cache_train.exists() and not _validate_cache(cache_train):
        cache_train.unlink()
    if cache_test.exists() and not _validate_cache(cache_test):
        cache_test.unlink()

    train_feats = build_features_v2(train_df, cache_path=cache_train)
    _save_cache_hash(cache_train)
    test_feats = build_features_v2(test_df, cache_path=cache_test)
    _save_cache_hash(cache_test)
    v2_count = len(train_feats.columns)
    print(f"  V2 base features: {v2_count}", flush=True)

    # ── Load ALL external features ──
    print("\n[3/7] Loading ALL external data...", flush=True)
    train_feats = load_all_external_features(train_feats, train_df, "train")
    test_feats = load_all_external_features(test_feats, test_df, "test")
    new_count = len(train_feats.columns) - v2_count
    print(f"  New external features added: {new_count}", flush=True)
    print(f"  Total features: {len(train_feats.columns)}", flush=True)

    # ── Validate data ──
    print("\n[4/7] Validating data integrity...", flush=True)
    warnings_list = validate_features(train_feats, test_feats, len(train_df), len(test_df))
    if warnings_list:
        print(f"\n  {len(warnings_list)} warnings found:", flush=True)
        for w in warnings_list:
            print(f"    - {w}", flush=True)

    # Drop constant/useless/redundant features (verified on both train AND test)
    DROP_CONSTANT = {
        # Constant (all zeros or all ones in both splits)
        "is_astronomical_night", "is_civil_twilight", "is_nautical_twilight",  # all daytime
        "is_day",  # always 1
        "wmo_fog", "wmo_snow", "wmo_thunder",  # no events in dataset
        "wxe_snow_depth", "wxe_snowfall",  # no snow
        "in_natura2000",  # 1 sample in train, 0 in test
    }
    feature_cols = sorted(
        (set(train_feats.columns) & set(test_feats.columns)) - DROP_CONSTANT
    )
    print(f"\n  Dropped {len(DROP_CONSTANT)} constant features", flush=True)
    print(f"  Final features: {len(feature_cols)}", flush=True)

    X_train = train_feats[feature_cols].values.astype(np.float32)
    X_test = test_feats[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Train models (all P0 fixes + CB-heavy) ──
    print("\n[5/7] Training models (all P0 fixes)...", flush=True)

    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES))
    oof_xgb = np.zeros((len(y), N_CLASSES))
    oof_cb = np.zeros((len(y), N_CLASSES))
    test_lgb = np.zeros((len(test_df), N_CLASSES))
    test_xgb = np.zeros((len(test_df), N_CLASSES))
    test_cb = np.zeros((len(test_df), N_CLASSES))

    lgb_params = {
        "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
        "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "is_unbalance": False,
        "verbosity": -1, "random_state": SEED, "n_jobs": -1,
    }
    xgb_params = {
        "objective": "multi:softprob", "num_class": N_CLASSES, "n_estimators": 2000,
        "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbosity": 0, "random_state": SEED, "n_jobs": -1,
        "tree_method": "hist", "early_stopping_rounds": 100,
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        print(f"\n  Fold {fold+1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)})", flush=True)
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        w_tr = effective_number_weights(y_tr)
        w_va = effective_number_weights(y_va)

        # LGB (Fix: eval_sample_weight)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[val_idx] = m.predict_proba(X_va)
        test_lgb += m.predict_proba(X_test) / N_FOLDS

        # XGB (Fix: weighted eval)
        mx = xgb.XGBClassifier(**xgb_params)
        mx.fit(X_tr, y_tr, sample_weight=w_tr,
               eval_set=[(X_va, y_va)], sample_weight_eval_set=[w_va], verbose=False)
        oof_xgb[val_idx] = mx.predict_proba(X_va)
        test_xgb += mx.predict_proba(X_test) / N_FOLDS

        # CB (Fix: weighted eval Pool)
        mc = cb.CatBoostClassifier(
            iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            random_seed=SEED, verbose=0, auto_class_weights=None,
            early_stopping_rounds=100, task_type="CPU",
        )
        mc.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=cb.Pool(X_va, y_va, weight=w_va))
        oof_cb[val_idx] = mc.predict_proba(X_va)
        test_cb += mc.predict_proba(X_test) / N_FOLDS

        # Fold mAP (CB-heavy)
        fold_p = 0.15 * oof_lgb[val_idx] + 0.05 * oof_xgb[val_idx] + 0.80 * oof_cb[val_idx]
        fm, _ = compute_map(y_va, fold_p)
        print(f"    Fold {fold+1} mAP: {fm:.4f}", flush=True)

    # ── Individual + ensemble evaluation ──
    print("\n[6/7] Evaluation...", flush=True)
    lgb_map, _ = compute_map(y, oof_lgb)
    xgb_map, _ = compute_map(y, oof_xgb)
    cb_map, _ = compute_map(y, oof_cb)
    print(f"  LGB OOF mAP: {lgb_map:.4f}")
    print(f"  XGB OOF mAP: {xgb_map:.4f}")
    print(f"  CB  OOF mAP: {cb_map:.4f}")

    # Optimize ensemble weights
    w_l, w_x, w_c, opt_map = optimize_ensemble_weights(oof_lgb, oof_xgb, oof_cb, y)
    print(f"  Optimized: LGB={w_l:.2f}, XGB={w_x:.2f}, CB={w_c:.2f} -> mAP={opt_map:.4f}")

    # Also compute E173-style CB-heavy
    oof_cb_heavy = 0.15 * oof_lgb + 0.05 * oof_xgb + 0.80 * oof_cb
    cbh_map, _ = compute_map(y, oof_cb_heavy)
    print(f"  CB-heavy (0.15/0.05/0.80): mAP={cbh_map:.4f}")

    # Use whichever is better
    if opt_map >= cbh_map:
        oof_best = w_l * oof_lgb + w_x * oof_xgb + w_c * oof_cb
        test_best = w_l * test_lgb + w_x * test_xgb + w_c * test_cb
        best_map = opt_map
        best_label = f"Optimized ({w_l}/{w_x}/{w_c})"
    else:
        oof_best = oof_cb_heavy
        test_best = 0.15 * test_lgb + 0.05 * test_xgb + 0.80 * test_cb
        best_map = cbh_map
        best_label = "CB-heavy (0.15/0.05/0.80)"

    skf_map, skf_per_class = compute_map(y, oof_best)
    print_results(skf_map, skf_per_class, f"E174 SKF ({best_label})")

    # LOMO
    print("\n  LOMO evaluation:", flush=True)
    lomo_maps = {}
    for held in sorted(set(train_months)):
        mask = train_months == held
        if mask.sum() < 10:
            continue
        lm, _ = compute_map(y[mask], oof_best[mask])
        lomo_maps[held] = lm
        name = {1: "Jan", 4: "Apr", 9: "Sep", 10: "Oct"}.get(held, f"M{held}")
        print(f"    LOMO {name} (n={mask.sum()}): {lm:.4f}", flush=True)
    lomo_avg = np.mean(list(lomo_maps.values()))
    print(f"    LOMO overall: {lomo_avg:.4f}", flush=True)

    # IW-mAP
    print("\n  IW-mAP validation:", flush=True)
    try:
        from src.validate import eval_pp
        result = eval_pp(
            lambda p, td, tm, trd, y_: apply_standard_nb_pp(p, td, tm, trd, y_),
            verbose=True,
        )
    except Exception as e:
        print(f"  IW-mAP skipped: {e}", flush=True)
        result = None

    # ── Save outputs ──
    print("\n[7/7] Saving...", flush=True)
    np.save(ROOT / "oof_e174.npy", oof_best)
    np.save(ROOT / "test_e174.npy", test_best)

    # Feature importance (last LGB)
    if hasattr(m, 'feature_importances_'):
        # Use the last LGB model (variable 'm' was reassigned, use lgb model)
        pass
    # Get importance from the last fold's LGB model
    imp = pd.Series(
        lgb.LGBMClassifier(**lgb_params).fit(
            X_train, y, sample_weight=effective_number_weights(y),
            callbacks=[lgb.early_stopping(100, verbose=False)],
            eval_set=[(X_train[:100], y[:100])],
            eval_sample_weight=[effective_number_weights(y[:100])],
        ).feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
    print("\n  Top 30 features (LGB importance):", flush=True)
    for feat, val in imp.head(30).items():
        print(f"    {feat:40s}: {val}", flush=True)
    print(f"\n  Bottom 10 features:", flush=True)
    for feat, val in imp.tail(10).items():
        print(f"    {feat:40s}: {val}", flush=True)

    # PP + submissions
    test_pp = apply_standard_nb_pp(test_best, test_df, test_months, train_df, y)
    save_submission(test_best, "e174_raw", cv_map=skf_map)
    save_submission(test_pp, "e174_pp", cv_map=skf_map)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  E174 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Features:       {len(feature_cols)} ({v2_count} v2 base + {new_count} new external + derived)")
    print(f"  Ensemble:       {best_label}")
    print(f"  SKF mAP:        {skf_map:.4f}")
    print(f"  LOMO mAP:       {lomo_avg:.4f}")
    if result:
        print(f"  Cal LB:         {result.get('calibrated_lb', 'N/A')}")
    print(f"\n  vs E173 (36 feats): SKF {skf_map:.4f} vs 0.6954, LOMO {lomo_avg:.4f} vs 0.5093")
    print(f"  vs E172 (76 feats): SKF {skf_map:.4f} vs 0.6956, LOMO {lomo_avg:.4f} vs 0.5163")
    print("=" * 70)

    # Per-class
    print("\n  Per-class AP:", flush=True)
    for cls in CLASSES:
        ap = skf_per_class.get(cls, 0)
        print(f"    {cls:15s}: {ap:.4f}", flush=True)


if __name__ == "__main__":
    main()
