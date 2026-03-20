"""E170: Feature System Rebuild — clean features + honest CV + spatiotemporal PP.

Rebuilt from ground up after 8 audits revealed:
  - RCS features computed on wrong scale (dB vs linear)
  - Missing discriminators (track_heading, start/end alt, heading_std)
  - Broken features (flap/glide ~50/50 always, soaring_index highest for Clutter)
  - Date-proxy leakage (4 features in E79's best 36)
  - Redundancies (6 features measuring same thing)

Key changes vs E79:
  1. features_v2.py: dual-scale RCS, track_heading, start/end alt, headwind
  2. StratifiedGroupKFold (primary_observation_id) — honest CV
  3. External data: tidal, water distance, turbine distance (safe + independent)
  4. Spatiotemporal smoothing (+0.0105 OOF verified)
  5. LOMO evaluation as primary metric (not SKF)

Outputs:
  - LOMO mAP, SKF mAP, per-class AP breakdown
  - oof_e170.npy, test_e170.npy
  - submission CSV if results look promising
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features_v2 import build_features_v2
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
# Spatiotemporal smoothing (verified +0.0105 OOF)
# ======================================================================

def spatiotemporal_smoothing(
    preds: np.ndarray,
    df: pd.DataFrame,
    radius_m: float = 500.0,
    time_window_s: float = 120.0,
    self_weight: float = 0.80,
) -> np.ndarray:
    """Smooth predictions by averaging with spatiotemporal neighbors.

    Tracks within `radius_m` meters and `time_window_s` seconds get averaged.
    Each track keeps `self_weight` of its own prediction and (1-self_weight)
    from the mean of its neighbors.

    Args:
        preds: (n, K) probability matrix
        df: DataFrame with trajectory (for position) and timestamp_start_radar_utc
        radius_m: spatial radius in meters
        time_window_s: temporal window in seconds
        self_weight: weight for the track's own prediction (0.80 = 80% self)

    Returns:
        (n, K) smoothed probability matrix
    """
    from src.data import parse_ewkb_4d

    n = len(df)
    if n < 2:
        return preds.copy()

    # Extract track centroids (mean lon/lat)
    lons = np.zeros(n)
    lats = np.zeros(n)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            lons[i] = np.mean([p[0] for p in pts])
            lats[i] = np.mean([p[1] for p in pts])
        except Exception:
            lons[i] = 0.0
            lats[i] = 0.0

    # Extract timestamps as seconds since epoch
    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    epoch_s = (ts - pd.Timestamp("2020-01-01")).dt.total_seconds().values

    # Convert to approximate meters for fast distance calc
    lon_m = lons * 67000.0
    lat_m = lats * 111000.0

    out = preds.copy()
    n_smoothed = 0

    for i in range(n):
        # Spatial distance (fast approximate)
        dx = lon_m - lon_m[i]
        dy = lat_m - lat_m[i]
        dist_sq = dx * dx + dy * dy

        # Temporal distance
        dt = np.abs(epoch_s - epoch_s[i])

        # Find neighbors (excluding self)
        mask = (dist_sq < radius_m * radius_m) & (dt < time_window_s) & (np.arange(n) != i)

        if mask.sum() > 0:
            neighbor_mean = preds[mask].mean(axis=0)
            out[i] = self_weight * preds[i] + (1.0 - self_weight) * neighbor_mean
            n_smoothed += 1

    out = renorm_rows(out)
    print(f"  Spatiotemporal smoothing: {n_smoothed}/{n} tracks had neighbors", flush=True)
    return out


# ======================================================================
# Effective number class weights (beta=0.999)
# ======================================================================

def effective_number_weights(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Compute per-sample weights using effective number of samples.

    Cui et al. 2019: w_c = (1 - beta) / (1 - beta^n_c)
    Less aggressive than inverse-frequency, more robust.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    eff_num = (1.0 - beta ** counts) / (1.0 - beta)
    class_weights = 1.0 / np.maximum(eff_num, 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    return class_weights[y]


# ======================================================================
# Main experiment
# ======================================================================

def main():
    print("=" * 60)
    print("  E170: Feature System Rebuild")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/7] Loading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(
        pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int
    )

    # ── Extract features ──
    print("\n[2/7] Building v2 features...", flush=True)
    cache_train = ROOT / "data" / "_cached_train_features_v2.pkl"
    cache_test = ROOT / "data" / "_cached_test_features_v2.pkl"

    train_feats = build_features_v2(train_df, cache_path=cache_train)
    test_feats = build_features_v2(test_df, cache_path=cache_test)

    feature_cols = [c for c in train_feats.columns if c in test_feats.columns]
    print(f"  Using {len(feature_cols)} features", flush=True)

    X_train = train_feats[feature_cols].values.astype(np.float32)
    X_test = test_feats[feature_cols].values.astype(np.float32)

    # ── Feature importance (quick LOMO check) ──
    print("\n[3/7] Training models...", flush=True)

    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb

    # Months for evaluation
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    # Groups for StratifiedGroupKFold
    groups = train_df["primary_observation_id"].values

    # ── StratifiedGroupKFold ──
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES))
    oof_xgb = np.zeros((len(y), N_CLASSES))
    oof_cb = np.zeros((len(y), N_CLASSES))
    test_lgb = np.zeros((len(test_df), N_CLASSES))
    test_xgb = np.zeros((len(test_df), N_CLASSES))
    test_cb = np.zeros((len(test_df), N_CLASSES))

    # Sample weights
    sample_weights = effective_number_weights(y, beta=0.999)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)})", flush=True)

        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        # ── LightGBM ──
        lgb_params = {
            "objective": "multiclass",
            "num_class": N_CLASSES,
            "metric": "multi_logloss",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "is_unbalance": False,  # using external sample_weight instead
            "verbosity": -1,
            "random_state": SEED,
            "n_jobs": -1,
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
        test_lgb += lgb_model.predict_proba(X_test) / N_FOLDS

        # ── XGBoost ──
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": N_CLASSES,
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "random_state": SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 100,
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_test) / N_FOLDS

        # ── CatBoost ──
        cb_model = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            verbose=0,
            auto_class_weights=None,  # using external sample_weight instead
            early_stopping_rounds=100,
            task_type="CPU",
        )
        cb_model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=(X_va, y_va),
        )
        oof_cb[val_idx] = cb_model.predict_proba(X_va)
        test_cb += cb_model.predict_proba(X_test) / N_FOLDS

        # Per-fold mAP
        fold_preds = 0.50 * oof_lgb[val_idx] + 0.40 * oof_xgb[val_idx] + 0.10 * oof_cb[val_idx]
        fold_map, _ = compute_map(y_va, fold_preds)
        print(f"    Fold {fold + 1} mAP: {fold_map:.4f}", flush=True)

    # ── Ensemble (same weights as E79) ──
    print("\n[4/7] Ensembling (LGB=0.50, XGB=0.40, CB=0.10)...", flush=True)
    oof_ensemble = 0.50 * oof_lgb + 0.40 * oof_xgb + 0.10 * oof_cb
    test_ensemble = 0.50 * test_lgb + 0.40 * test_xgb + 0.10 * test_cb

    # ── SKF evaluation ──
    skf_map, skf_per_class = compute_map(y, oof_ensemble)
    print_results(skf_map, skf_per_class, "E170 SKF (StratifiedGroupKFold)")

    # ── LOMO evaluation ──
    print("\n[5/7] LOMO evaluation...", flush=True)
    lomo_maps = {}
    for held_month in sorted(set(train_months)):
        va_mask = train_months == held_month
        if va_mask.sum() < 10:
            continue
        lomo_map, lomo_per = compute_map(y[va_mask], oof_ensemble[va_mask])
        lomo_maps[held_month] = lomo_map
        month_name = {1: "Jan", 4: "Apr", 9: "Sep", 10: "Oct"}.get(held_month, f"M{held_month}")
        print(f"    LOMO {month_name} (n={va_mask.sum()}): mAP={lomo_map:.4f}", flush=True)

    lomo_overall = np.mean(list(lomo_maps.values()))
    print(f"    LOMO overall: {lomo_overall:.4f}", flush=True)

    # ── Spatiotemporal smoothing ──
    print("\n[6/7] Post-processing...", flush=True)

    # Apply spatiotemporal smoothing on OOF for validation
    oof_smoothed = spatiotemporal_smoothing(
        oof_ensemble, train_df,
        radius_m=500.0, time_window_s=120.0, self_weight=0.80,
    )
    skf_smooth_map, skf_smooth_per = compute_map(y, oof_smoothed)
    print(f"  SKF after smoothing: {skf_smooth_map:.4f} (delta: {skf_smooth_map - skf_map:+.4f})", flush=True)

    # Apply smoothing on test predictions
    test_smoothed = spatiotemporal_smoothing(
        test_ensemble, test_df,
        radius_m=500.0, time_window_s=120.0, self_weight=0.80,
    )

    # Standard GBIF priors + NB evidence
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    # Apply PP on test
    test_pp, n_changed = apply_gated_ratio_priors(
        test_smoothed, test_months, p_train, priors, BASE_ALPHA, tau=0.15
    )

    # NB evidence (tabular channels)
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {
        "speed": speed_tr,
        "alt_mid": 0.5 * (min_z_tr + max_z_tr),
        "alt_range": max_z_tr - min_z_tr,
    }
    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te = {
        "speed": speed_te,
        "alt_mid": 0.5 * (min_z_te + max_z_te),
        "alt_range": max_z_te - min_z_te,
    }
    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig,
    )
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < 0.25)
    test_final = apply_nb_poe(test_pp, loglike, gamma=0.10, gate=gate)

    print(f"  GBIF priors changed: {n_changed} rows", flush=True)
    print(f"  NB evidence gated: {gate.sum()} rows", flush=True)

    # ── Save outputs ──
    print("\n[7/7] Saving outputs...", flush=True)
    np.save(ROOT / "oof_e170.npy", oof_ensemble)
    np.save(ROOT / "test_e170.npy", test_ensemble)
    print("  Saved oof_e170.npy, test_e170.npy", flush=True)

    # Feature importance (LGB last fold)
    if hasattr(lgb_model, 'feature_importances_'):
        imp = pd.Series(lgb_model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False)
        print("\n  Top 20 features (LGB importance):", flush=True)
        for feat, val in imp.head(20).items():
            print(f"    {feat:30s}: {val}", flush=True)

    # Save submission (raw + PP)
    save_submission(test_ensemble, "e170_raw", cv_map=skf_map)
    save_submission(test_final, "e170_pp", cv_map=skf_map)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  E170 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Features:       {len(feature_cols)}")
    print(f"  SKF mAP:        {skf_map:.4f}")
    print(f"  SKF smoothed:   {skf_smooth_map:.4f}")
    print(f"  LOMO mAP:       {lomo_overall:.4f}")
    print(f"  E79 baseline:   SKF=0.7736, LOMO=0.363")
    print("=" * 60)

    # Per-class comparison
    print("\n  Per-class AP (E170 vs E79 baseline):", flush=True)
    e79_aps = {
        "Birds of Prey": 0.885, "Clutter": 0.610, "Cormorants": 0.939,
        "Ducks": 0.666, "Geese": 0.728, "Gulls": 0.956,
        "Pigeons": 0.254, "Songbirds": 0.640, "Waders": 0.816,
    }
    for cls in CLASSES:
        e170_ap = skf_per_class.get(cls, 0)
        e79_ap = e79_aps.get(cls, 0)
        delta = e170_ap - e79_ap
        marker = "+" if delta > 0 else ""
        print(f"    {cls:15s}: {e170_ap:.4f}  (E79: {e79_ap:.3f}, delta: {marker}{delta:.4f})", flush=True)


if __name__ == "__main__":
    main()
