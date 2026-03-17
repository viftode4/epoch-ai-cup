"""E164: LOMO-Trained Invariant Base Model.

Key changes vs E79:
  1. Remove weather/solar features (Phase 0 analysis shows they're month proxies)
     - All 7 weather features have MI(month)/MI(class) > 3.5
     - sol_daylight_hours has ratio 6.0
     - Keep ONLY 25 physics features (+ optionally sol_hours_since_sunrise, sol_daylight_fraction)
  2. LOMO CV training (4 folds: Jan, Apr, Sep, Oct)
     - Optimize ensemble weights per LOMO fold, not global SKF
  3. Month-aware test prediction routing:
     - Feb/Dec test -> Jan-held-out model (winter proxy)
     - May test -> Apr-held-out model (spring proxy)
     - Sep/Oct test -> respective held-out models
  4. Also train standard SKF model on invariant features for comparison

Hypothesis: Weather/solar features cause temporal overfitting. Removing them
and training with LOMO should improve cross-month generalization, especially
for unseen months (Feb/May/Dec = 33% of test).
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

# Phase 0 results: 25 invariant features (MI month/class ratio < 1.0)
INVARIANT_FEATURES = [
    # Altitude
    "alt_max", "alt_median", "alt_q75", "alt_change_halves",
    # RCS
    "rcs_mean", "rcs_median", "rcs_q25", "rcs_q75", "rcs_per_alt", "rcs_for_size",
    # Speed
    "speed_median", "avg_ground_speed", "airspeed", "airspeed_vs_ground",
    "slow_flight_frac", "accel_std",
    # Trajectory shape
    "bearing_change_mean", "curvature_mean",
    # Position
    "lon_mean", "lat_mean", "lon_std", "lat_std",
    # Interactions
    "speed_x_alt", "size_x_alt",
    # Altitude dynamics
    "alt_rate_mean",
]

# Variant B: + 2 borderline solar features (ratio 1.18 and 1.52)
BORDERLINE_SOLAR = ["sol_hours_since_sunrise", "sol_daylight_fraction"]

# Test month -> which LOMO fold (training month) is proxy
TEST_TO_TRAIN_PROXY = {
    2:  1,   # Feb -> Jan (winter)
    5:  4,   # May -> Apr (spring)
    9:  9,   # Sep -> Sep (shared)
    10: 10,  # Oct -> Oct (shared)
    12: 1,   # Dec -> Jan (winter)
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


def train_ensemble_fold(X_tr, y_tr, X_va, y_va, X_test, sample_weights_tr):
    """Train LGB + XGB + CB on a single fold, return per-model predictions."""
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    va_lgb = lgb.predict_proba(X_va)
    te_lgb = lgb.predict_proba(X_test)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            sample_weight=sample_weights_tr, verbose=False)
    va_xgb = xgb.predict_proba(X_va)
    te_xgb = xgb.predict_proba(X_test)

    # CatBoost
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5,
        random_strength=1.0, border_count=128,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    va_cb = cb.predict_proba(X_va)
    te_cb = cb.predict_proba(X_test)

    return (va_lgb, va_xgb, va_cb), (te_lgb, te_xgb, te_cb)


def optimize_weights(va_lgb, va_xgb, va_cb, y_va):
    """Grid search ensemble weights on validation set."""
    best_w = (0.5, 0.3, 0.2)
    best_map = -1.0
    for w_lgb in np.arange(0.0, 0.75, 0.05):
        for w_xgb in np.arange(0.0, 0.75, 0.05):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01 or w_cb > 1.01:
                continue
            oof = w_lgb * va_lgb + w_xgb * va_xgb + w_cb * va_cb
            m, _ = compute_map(y_va, oof)
            if m > best_map:
                best_map = m
                best_w = (w_lgb, w_xgb, max(w_cb, 0))
    return best_w, best_map


def main():
    print("=" * 70, flush=True)
    print("E164 LOMO-INVARIANT BASE MODEL".center(70), flush=True)
    print("=" * 70, flush=True)

    # -- Load data ---
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    unique_months = sorted(np.unique(train_months))

    # -- Build features ---
    print("\nBuilding features...", flush=True)
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    # Remove temporal features
    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]

    # Add weather+solar for variant B
    train_feats, test_feats = add_weather_solar(train_feats, test_feats)

    # Effective number class weights
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    beta = 0.999
    eff_n = (1.0 - beta ** counts) / (1.0 - beta)
    class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
    class_weights_arr /= class_weights_arr.sum() / N_CLASSES
    sample_weights = class_weights_arr[y]

    # ============================================================
    # Variant A: 25 invariant features (no weather/solar)
    # ============================================================
    for variant_name, feat_list in [
        ("A_invariant25", INVARIANT_FEATURES),
        ("B_invariant27", INVARIANT_FEATURES + BORDERLINE_SOLAR),
    ]:
        avail = [f for f in feat_list if f in train_feats.columns]
        missing = [f for f in feat_list if f not in train_feats.columns]
        if missing:
            print(f"  WARNING: missing features for {variant_name}: {missing}", flush=True)

        X = train_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
        X_test = test_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

        print(f"\n{'='*70}", flush=True)
        print(f"  VARIANT {variant_name}: {len(avail)} features", flush=True)
        print(f"{'='*70}", flush=True)

        # ============================================================
        # Part 1: LOMO training with month-aware routing
        # ============================================================
        print("\n--- LOMO training (4 folds) with month routing ---", flush=True)

        lomo_oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        lomo_test_per_month = {}  # month -> test predictions from that fold

        for held_out_month in unique_months:
            tr_mask = train_months != held_out_month
            va_mask = train_months == held_out_month
            tr_idx = np.where(tr_mask)[0]
            va_idx = np.where(va_mask)[0]

            print(f"\n  LOMO fold: hold out M{held_out_month:02d} "
                  f"(train={tr_mask.sum()}, val={va_mask.sum()})", flush=True)

            va_preds, te_preds = train_ensemble_fold(
                X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X_test,
                sample_weights[tr_idx],
            )

            # Optimize weights on this fold's validation set
            w, fold_map = optimize_weights(
                va_preds[0], va_preds[1], va_preds[2], y[va_idx],
            )
            print(f"    Fold weights: LGB={w[0]:.2f} XGB={w[1]:.2f} CB={w[2]:.2f} "
                  f"mAP={fold_map:.4f}", flush=True)

            # Store OOF
            lomo_oof[va_idx] = w[0] * va_preds[0] + w[1] * va_preds[1] + w[2] * va_preds[2]

            # Store test predictions (with this fold's optimized weights)
            te_ens = w[0] * te_preds[0] + w[1] * te_preds[1] + w[2] * te_preds[2]
            lomo_test_per_month[held_out_month] = te_ens

        # LOMO OOF evaluation
        lomo_map, lomo_per = compute_map(y, lomo_oof)
        print_results(lomo_map, lomo_per, label=f"E164 {variant_name} LOMO OOF")

        # Month-routed test predictions
        test_routed = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        for test_m in sorted(np.unique(test_months)):
            proxy_month = TEST_TO_TRAIN_PROXY.get(test_m, 1)  # default to Jan
            mask = test_months == test_m
            test_routed[mask] = lomo_test_per_month[proxy_month][mask]
            print(f"  Test M{test_m:02d} ({mask.sum()} rows) -> "
                  f"LOMO fold M{proxy_month:02d}", flush=True)

        # Also create averaged test predictions (like E79)
        test_avg = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        for m, preds in lomo_test_per_month.items():
            test_avg += preds / len(unique_months)

        # ============================================================
        # Part 2: SKF training (same features, for comparison)
        # ============================================================
        print("\n--- SKF training (5-fold, same features) ---", flush=True)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            print(f"  SKF Fold {fold_i+1}/{N_FOLDS}", flush=True)
            va_preds, te_preds = train_ensemble_fold(
                X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X_test,
                sample_weights[tr_idx],
            )
            oof_lgb[va_idx] = va_preds[0]
            oof_xgb[va_idx] = va_preds[1]
            oof_cb[va_idx] = va_preds[2]
            test_lgb += te_preds[0] / N_FOLDS
            test_xgb += te_preds[1] / N_FOLDS
            test_cb += te_preds[2] / N_FOLDS

        # Optimize SKF ensemble weights
        w_skf, skf_map = optimize_weights(oof_lgb, oof_xgb, oof_cb, y)
        print(f"  SKF weights: LGB={w_skf[0]:.2f} XGB={w_skf[1]:.2f} CB={w_skf[2]:.2f}", flush=True)

        oof_skf = w_skf[0] * oof_lgb + w_skf[1] * oof_xgb + w_skf[2] * oof_cb
        test_skf = w_skf[0] * test_lgb + w_skf[1] * test_xgb + w_skf[2] * test_cb

        skf_map_final, skf_per = compute_map(y, oof_skf)
        print_results(skf_map_final, skf_per, label=f"E164 {variant_name} SKF OOF")

        # ============================================================
        # Save all variants
        # ============================================================
        print("\nSaving artifacts...", flush=True)

        # LOMO routed
        np.save(ROOT / f"oof_e164_{variant_name}_lomo.npy", lomo_oof)
        np.save(ROOT / f"test_e164_{variant_name}_routed.npy", test_routed)
        save_submission(test_routed, f"e164_{variant_name}_routed", cv_map=lomo_map)

        # LOMO averaged (not routed)
        np.save(ROOT / f"test_e164_{variant_name}_avg.npy", test_avg)
        save_submission(test_avg, f"e164_{variant_name}_avg", cv_map=lomo_map)

        # SKF
        np.save(ROOT / f"oof_e164_{variant_name}_skf.npy", oof_skf)
        np.save(ROOT / f"test_e164_{variant_name}_skf.npy", test_skf)
        save_submission(test_skf, f"e164_{variant_name}_skf", cv_map=skf_map_final)

        print(f"\n  SUMMARY {variant_name}:", flush=True)
        print(f"    LOMO mAP: {lomo_map:.4f}", flush=True)
        print(f"    SKF mAP:  {skf_map_final:.4f}", flush=True)
        print(f"    E79 ref:  0.7736 (SKF, 36 feats with weather/solar)", flush=True)

    print("\nDone. Submit routed variant if LOMO mAP > E79.", flush=True)


if __name__ == "__main__":
    main()
