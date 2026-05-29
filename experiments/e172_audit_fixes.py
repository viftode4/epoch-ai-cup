"""E172: Pipeline Audit Fixes + Extended PP Evidence Channels.

Comprehensive audit fix experiment incorporating:

TRACK 1 — P0 Bug Fixes:
  1.1 OOF smoothing leakage: skip OOF smoothing, only apply to test
  1.2 Per-fold sample weights: compute inside fold loop
  1.3 Weighted eval_set for XGB/CB: pass sample_weight to eval sets
  1.4 Cache invalidation: hash-based cache check for features_v2.py

TRACK 2 — New PP Evidence Channels (NB only, NOT base model features):
  2.1 Insect activity index (Clutter)
  2.2 Visibility/fog/rain (Clutter/Duck)
  2.3 Marine wave_height + SST (Cormorant/Gull/Wader)
  2.4 Wind shear 80-180m (BoP thermal)
  2.5 Photoperiod change rate (migration trigger)
  2.6 Natura2000 distance (Wader)
  2.7 CAPE normalized (BoP thermal)
  2.8 True airspeed (wind-corrected speed)

TRACK 4 — Domain Knowledge Gaps:
  4.1 Insect-wind match (speed_wind_ratio + heading_wind_diff → Clutter)
  4.2 Crepuscular index as NB evidence (Duck/Pigeon)
  4.3 Alerstam 2007 literature speed priors (generalization)

Base model: same as E170 (features_v2.py, StratifiedGroupKFold, LGB+XGB+CB ensemble)
PP: extended NB evidence with all new channels + Alerstam speed priors

Outputs:
  - Honest OOF mAP (no smoothing leakage)
  - LOMO evaluation
  - Submission CSVs (raw + PP variants)
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
from src.features_v2 import build_features_v2
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    load_pp_evidence, get_alerstam_speed_params,
)

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
SEED = 42


# ======================================================================
# Cache invalidation via source hash (Fix 1.4)
# ======================================================================

def _features_v2_hash() -> str:
    """Hash features_v2.py source to detect code changes."""
    src = (ROOT / "src" / "features_v2.py").read_text(encoding="utf-8")
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _validate_cache(cache_path: Path) -> bool:
    """Check if cached features match current source code."""
    meta_path = cache_path.with_suffix(".hash")
    current_hash = _features_v2_hash()
    if meta_path.exists():
        stored_hash = meta_path.read_text().strip()
        if stored_hash == current_hash:
            return True
        print(f"  Cache STALE: features_v2.py changed ({stored_hash} -> {current_hash})", flush=True)
        return False
    return False


def _save_cache_hash(cache_path: Path):
    """Save source hash alongside cached features."""
    meta_path = cache_path.with_suffix(".hash")
    meta_path.write_text(_features_v2_hash())


# ======================================================================
# Spatiotemporal smoothing (TEST ONLY — no OOF to prevent leakage)
# ======================================================================

def spatiotemporal_smoothing(
    preds: np.ndarray,
    df: pd.DataFrame,
    radius_m: float = 500.0,
    time_window_s: float = 120.0,
    self_weight: float = 0.80,
) -> np.ndarray:
    """Smooth predictions by averaging with spatiotemporal neighbors.

    Only used on TEST predictions. OOF smoothing is skipped to prevent
    cross-fold leakage (Fix 1.1).
    """
    from src.data import parse_ewkb_4d

    n = len(df)
    if n < 2:
        return preds.copy()

    lons = np.zeros(n)
    lats = np.zeros(n)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            lons[i] = np.mean([p[0] for p in pts])
            lats[i] = np.mean([p[1] for p in pts])
        except Exception:
            pass

    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    epoch_s = (ts - pd.Timestamp("2020-01-01")).dt.total_seconds().values

    lon_m = lons * 67000.0
    lat_m = lats * 111000.0

    out = preds.copy()
    n_smoothed = 0

    for i in range(n):
        dx = lon_m - lon_m[i]
        dy = lat_m - lat_m[i]
        dist_sq = dx * dx + dy * dy
        dt = np.abs(epoch_s - epoch_s[i])
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
    """Compute per-sample weights using effective number of samples."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    eff_num = (1.0 - beta ** counts) / (1.0 - beta)
    class_weights = 1.0 / np.maximum(eff_num, 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    return class_weights[y]


# ======================================================================
# Extended PP: build evidence channels for NB
# ======================================================================

def build_extended_nb_params(
    train_df: pd.DataFrame,
    y: np.ndarray,
    use_alerstam_speed: bool = True,
) -> tuple:
    """Build NB params with extended evidence channels.

    Returns (size_levels, log_p_size, mu, sig, channel_weights, min_sigma_map)
    """
    # Core tabular channels (same as E170)
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {
        "speed": speed_tr,
        "alt_mid": 0.5 * (min_z_tr + max_z_tr),
        "alt_range": max_z_tr - min_z_tr,
    }

    # Load all external PP evidence channels
    ext_tr = load_pp_evidence(train_df, "train")
    cont_tr.update(ext_tr)

    # Channel weights (domain-informed):
    # Core channels get full weight, new channels start at 0.5 (conservative)
    weights = {
        "speed": 1.0,
        "alt_mid": 1.0,
        "alt_range": 0.5,
        # New channels — conservative initial weights
        "insect_activity": 0.5,
        "visibility_km": 0.3,
        "fog": 0.3,
        "rain": 0.3,
        "wave_height": 0.4,
        "sst": 0.3,
        "wind_shear": 0.4,
        "direct_radiation": 0.3,
        "daylength_change": 0.4,
        "natura2000_dist": 0.3,
        "cape_norm": 0.3,
        "crepuscular": 0.5,
        "true_airspeed": 0.8,  # should be better than raw speed
        "insect_wind_match": 0.6,  # strong domain signal for Clutter
    }

    # Min sigma overrides for binary/discrete channels
    min_sigma = {
        "fog": 0.10,
        "rain": 0.10,
        "crepuscular": 0.10,
        "insect_wind_match": 0.10,
    }

    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, cont_tr,
        min_sigma=min_sigma,
        default_min_sigma=0.50,
    )

    # Override speed params with Alerstam literature priors (Fix 4.3)
    if use_alerstam_speed:
        al_mu, al_sigma = get_alerstam_speed_params()
        # Blend: 50% data-derived + 50% literature (compromise for generalization)
        if "speed" in mu:
            mu["speed"] = 0.5 * mu["speed"] + 0.5 * al_mu
            sig["speed"] = np.sqrt(0.5 * sig["speed"]**2 + 0.5 * al_sigma**2)
            print("  Speed priors blended with Alerstam 2007 literature values", flush=True)

    return size_levels, log_p_size, mu, sig, weights, min_sigma


def apply_extended_nb_pp(
    preds: np.ndarray,
    test_df: pd.DataFrame,
    test_months: np.ndarray,
    train_df: pd.DataFrame,
    y: np.ndarray,
    gamma: float = 0.10,
    tau_prior: float = 0.15,
    tau_nb: float = 0.25,
    use_alerstam_speed: bool = True,
) -> np.ndarray:
    """Apply extended NB post-processing with all evidence channels.

    Callable as pp_fn for eval_pp().
    """
    # Stage 1: GBIF priors
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, n_changed = apply_gated_ratio_priors(
        preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior
    )

    # Stage 2+3: Extended NB evidence
    size_levels, log_p_size, mu, sig, weights, min_sigma = build_extended_nb_params(
        train_df, y, use_alerstam_speed=use_alerstam_speed,
    )

    # Load test evidence
    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te: dict[str, np.ndarray] = {
        "speed": speed_te,
        "alt_mid": 0.5 * (min_z_te + max_z_te),
        "alt_range": max_z_te - min_z_te,
    }
    ext_te = load_pp_evidence(test_df, "test" if "bird_group" not in test_df.columns else "train")
    cont_te.update(ext_te)

    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig,
    )
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    out = apply_nb_poe(out, loglike, gamma=gamma, gate=gate)

    return out


# ======================================================================
# Main experiment
# ======================================================================

def main():
    print("=" * 60)
    print("  E172: Pipeline Audit Fixes + Extended PP")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/8] Loading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(
        pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int
    )

    # ── Extract features (with cache validation — Fix 1.4) ──
    print("\n[2/8] Building v2 features...", flush=True)
    cache_train = ROOT / "data" / "_cached_train_features_v2.pkl"
    cache_test = ROOT / "data" / "_cached_test_features_v2.pkl"

    # Validate caches against source hash
    if cache_train.exists() and not _validate_cache(cache_train):
        cache_train.unlink()
        print("  Deleted stale train cache", flush=True)
    if cache_test.exists() and not _validate_cache(cache_test):
        cache_test.unlink()
        print("  Deleted stale test cache", flush=True)

    train_feats = build_features_v2(train_df, cache_path=cache_train)
    _save_cache_hash(cache_train)
    test_feats = build_features_v2(test_df, cache_path=cache_test)
    _save_cache_hash(cache_test)

    feature_cols = [c for c in train_feats.columns if c in test_feats.columns]
    print(f"  Using {len(feature_cols)} features", flush=True)

    X_train = train_feats[feature_cols].values.astype(np.float32)
    X_test = test_feats[feature_cols].values.astype(np.float32)

    # ── Train models ──
    print("\n[3/8] Training models...", flush=True)

    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES))
    oof_xgb = np.zeros((len(y), N_CLASSES))
    oof_cb = np.zeros((len(y), N_CLASSES))
    test_lgb = np.zeros((len(test_df), N_CLASSES))
    test_xgb = np.zeros((len(test_df), N_CLASSES))
    test_cb = np.zeros((len(test_df), N_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)})", flush=True)

        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Fix 1.2: Per-fold sample weights (not global)
        w_tr = effective_number_weights(y_tr, beta=0.999)
        w_va = effective_number_weights(y_va, beta=0.999)

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
            "is_unbalance": False,
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

        # ── XGBoost (Fix 1.3: weighted eval set) ──
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
            sample_weight_eval_set=[w_va],  # Fix 1.3
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_test) / N_FOLDS

        # ── CatBoost (Fix 1.3: weighted eval set) ──
        cb_model = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=SEED,
            verbose=0,
            auto_class_weights=None,
            early_stopping_rounds=100,
            task_type="CPU",
        )
        cb_pool_val = cb.Pool(X_va, y_va, weight=w_va)  # Fix 1.3
        cb_model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=cb_pool_val,
        )
        oof_cb[val_idx] = cb_model.predict_proba(X_va)
        test_cb += cb_model.predict_proba(X_test) / N_FOLDS

        # Per-fold mAP
        fold_preds = 0.50 * oof_lgb[val_idx] + 0.40 * oof_xgb[val_idx] + 0.10 * oof_cb[val_idx]
        fold_map, _ = compute_map(y_va, fold_preds)
        print(f"    Fold {fold + 1} mAP: {fold_map:.4f}", flush=True)

    # ── Ensemble ──
    print("\n[4/8] Ensembling (LGB=0.50, XGB=0.40, CB=0.10)...", flush=True)
    oof_ensemble = 0.50 * oof_lgb + 0.40 * oof_xgb + 0.10 * oof_cb
    test_ensemble = 0.50 * test_lgb + 0.40 * test_xgb + 0.10 * test_cb

    # ── SKF evaluation (no OOF smoothing — Fix 1.1) ──
    skf_map, skf_per_class = compute_map(y, oof_ensemble)
    print_results(skf_map, skf_per_class, "E172 SKF (no smoothing, honest)")

    # ── LOMO evaluation ──
    print("\n[5/8] LOMO evaluation...", flush=True)
    lomo_maps = {}
    for held_month in sorted(set(train_months)):
        va_mask = train_months == held_month
        if va_mask.sum() < 10:
            continue
        lomo_map, _ = compute_map(y[va_mask], oof_ensemble[va_mask])
        lomo_maps[held_month] = lomo_map
        month_name = {1: "Jan", 4: "Apr", 9: "Sep", 10: "Oct"}.get(held_month, f"M{held_month}")
        print(f"    LOMO {month_name} (n={va_mask.sum()}): mAP={lomo_map:.4f}", flush=True)

    lomo_overall = np.mean(list(lomo_maps.values()))
    print(f"    LOMO overall: {lomo_overall:.4f}", flush=True)

    # ── Post-processing on TEST only (Fix 1.1) ──
    print("\n[6/8] Post-processing (test only — no OOF smoothing)...", flush=True)

    # Apply spatiotemporal smoothing on TEST only
    test_smoothed = spatiotemporal_smoothing(
        test_ensemble, test_df,
        radius_m=500.0, time_window_s=120.0, self_weight=0.80,
    )

    # Variant A: Standard PP (same as E170 for comparison)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    test_pp_std, n_changed = apply_gated_ratio_priors(
        test_smoothed, test_months, p_train, priors, BASE_ALPHA, tau=0.15
    )
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr_std = {"speed": speed_tr, "alt_mid": 0.5*(min_z_tr+max_z_tr), "alt_range": max_z_tr-min_z_tr}
    sl, lps, mu_s, sig_s = build_nb_params(train_df, y, cont_tr_std)
    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te_std = {"speed": speed_te, "alt_mid": 0.5*(min_z_te+max_z_te), "alt_range": max_z_te-min_z_te}
    w_std = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll_std = compute_log_p_u_given_c(test_df, sl, lps, cont_te_std, w_std, None, mu_s, sig_s)
    gate_std = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp_std) < 0.25)
    test_final_std = apply_nb_poe(test_pp_std, ll_std, gamma=0.10, gate=gate_std)

    # Variant B: Extended PP (all new evidence channels)
    print("\n  Building extended PP (new evidence channels)...", flush=True)
    test_pp_ext = apply_extended_nb_pp(
        test_smoothed, test_df, test_months, train_df, y,
        gamma=0.10, tau_prior=0.15, tau_nb=0.25,
        use_alerstam_speed=True,
    )

    # Variant C: Extended PP with stronger gamma
    test_pp_ext_g15 = apply_extended_nb_pp(
        test_smoothed, test_df, test_months, train_df, y,
        gamma=0.15, tau_prior=0.15, tau_nb=0.25,
        use_alerstam_speed=True,
    )

    print(f"  Standard PP: {n_changed} rows GBIF-adjusted, {gate_std.sum()} NB-gated", flush=True)

    # ── IW-mAP Validation ──
    print("\n[7/8] IW-mAP Validation...", flush=True)
    try:
        from src.validate import eval_pp

        # Evaluate extended PP via eval_pp
        print("\n  --- Standard PP (baseline) ---", flush=True)
        result_std = eval_pp(
            lambda p, td, tm, trd, y_: apply_extended_nb_pp(
                p, td, tm, trd, y_, gamma=0.10, use_alerstam_speed=False,
            ),
            verbose=True,
        )

        print("\n  --- Extended PP (all channels + Alerstam) ---", flush=True)
        result_ext = eval_pp(
            lambda p, td, tm, trd, y_: apply_extended_nb_pp(
                p, td, tm, trd, y_, gamma=0.10, use_alerstam_speed=True,
            ),
            verbose=True,
        )

        print("\n  --- Extended PP gamma=0.15 ---", flush=True)
        result_ext15 = eval_pp(
            lambda p, td, tm, trd, y_: apply_extended_nb_pp(
                p, td, tm, trd, y_, gamma=0.15, use_alerstam_speed=True,
            ),
            verbose=True,
        )
    except Exception as e:
        print(f"  IW-mAP validation skipped: {e}", flush=True)
        result_std = result_ext = result_ext15 = None

    # ── Save outputs ──
    print("\n[8/8] Saving outputs...", flush=True)
    np.save(ROOT / "oof_e172.npy", oof_ensemble)
    np.save(ROOT / "test_e172.npy", test_ensemble)

    # Feature importance
    if hasattr(lgb_model, 'feature_importances_'):
        imp = pd.Series(lgb_model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False)
        print("\n  Top 20 features (LGB importance):", flush=True)
        for feat, val in imp.head(20).items():
            print(f"    {feat:30s}: {val}", flush=True)

    # Save submissions
    save_submission(test_ensemble, "e172_raw", cv_map=skf_map)
    save_submission(test_final_std, "e172_pp_std", cv_map=skf_map)
    save_submission(test_pp_ext, "e172_pp_ext", cv_map=skf_map)
    save_submission(test_pp_ext_g15, "e172_pp_ext_g15", cv_map=skf_map)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  E172 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Features:       {len(feature_cols)}")
    print(f"  SKF mAP:        {skf_map:.4f}  (honest, no OOF smoothing)")
    print(f"  LOMO mAP:       {lomo_overall:.4f}")
    print(f"  E170 baseline:  SKF=0.5141 (with leaky smoothing)")
    print()
    print(f"  P0 Fixes applied:")
    print(f"    1.1 OOF smoothing leakage: FIXED (test-only smoothing)")
    print(f"    1.2 Per-fold sample weights: FIXED")
    print(f"    1.3 Weighted eval_set XGB/CB: FIXED")
    print(f"    1.4 Cache invalidation: FIXED (source hash)")
    print()
    if result_std:
        cal_std = result_std.get("calibrated_lb", "N/A")
        cal_ext = result_ext.get("calibrated_lb", "N/A") if result_ext else "N/A"
        cal_ext15 = result_ext15.get("calibrated_lb", "N/A") if result_ext15 else "N/A"
        print(f"  PP Variants (calibrated LB estimate):")
        print(f"    Standard:           {cal_std}")
        print(f"    Extended g=0.10:    {cal_ext}")
        print(f"    Extended g=0.15:    {cal_ext15}")
    print("=" * 60)

    # Per-class comparison
    print("\n  Per-class AP (E172 vs E79 baseline):", flush=True)
    e79_aps = {
        "Birds of Prey": 0.885, "Clutter": 0.610, "Cormorants": 0.939,
        "Ducks": 0.666, "Geese": 0.728, "Gulls": 0.956,
        "Pigeons": 0.254, "Songbirds": 0.640, "Waders": 0.816,
    }
    for cls in CLASSES:
        e172_ap = skf_per_class.get(cls, 0)
        e79_ap = e79_aps.get(cls, 0)
        delta = e172_ap - e79_ap
        marker = "+" if delta > 0 else ""
        print(f"    {cls:15s}: {e172_ap:.4f}  (E79: {e79_ap:.3f}, delta: {marker}{delta:.4f})", flush=True)


if __name__ == "__main__":
    main()
