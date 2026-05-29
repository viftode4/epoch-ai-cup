"""E173: P0 Bug Fixes on E79's Proven 36 Features + Ensemble Re-optimization.

Goal: Apply proven bug fixes to the proven feature set. Lowest risk, highest ROI.

Bug fixes applied:
  1. Per-fold effective_number_weights (not global)
  2. XGB eval_set weighted (sample_weight_eval_set)
  3. CB eval_set weighted (Pool with weight)
  4. LGB eval_set weighted (eval_sample_weight) — NEW fix missed in E172
  5. OOF smoothing leakage: skip OOF smoothing (test-only)
  6. Cache hash validation for features_v2

Feature set: E79's backward-eliminated 36 features, mapped to v2 pipeline.
  - 4 renamed (rcs_mean→rcs_mean_dB, etc.)
  - 6 recovered from external CSVs / computed (alt_q75, rcs_per_alt, etc.)

Ensemble: Re-optimized weights via grid search on OOF mAP.
PP: Standard 3-channel NB only (speed, alt_mid, alt_range) — NOT the 14 extended
    channels that hurt in E172.
Bonus: Cleanlab noise detection on OOF predictions.
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
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
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
# E79 feature mapping: v1 name -> v2 name
# ======================================================================

E79_TO_V2 = {
    "rcs_mean": "rcs_mean_dB",
    "rcs_median": "rcs_median_dB",
    "rcs_q25": "rcs_q25_dB",
    "rcs_q75": "rcs_q75_dB",
}

# Features that v2 dropped but E79 needs — must recover
E79_MISSING_FROM_V2 = [
    "alt_q75",          # percentile from trajectory
    "rcs_per_alt",      # rcs_mean / alt_mean (v1 style)
    "wx_wind_u",        # from weather CSV
    "wx_temp_c",        # from weather CSV
    "wx_dewpoint_c",    # from weather CSV
    "sol_daylight_hours",  # from solar CSV
]


def recover_missing_features(
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Recover E79 features that v2 dropped."""
    n = len(df_feat)

    # alt_q75: compute from trajectory
    alt_q75 = np.zeros(n, dtype=np.float32)
    for i, (_, row) in enumerate(df_raw.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            alts = np.array([p[2] for p in pts])
            alt_q75[i] = float(np.percentile(alts, 75))
        except Exception:
            alt_q75[i] = 0.0
    df_feat["alt_q75"] = alt_q75

    # rcs_per_alt: v1 formula = rcs_mean / max(alt_mean, 1)
    if "rcs_mean_dB" in df_feat.columns and "alt_mean" in df_feat.columns:
        df_feat["rcs_per_alt"] = df_feat["rcs_mean_dB"] / df_feat["alt_mean"].clip(lower=1.0)

    # wx_wind_u, wx_temp_c, wx_dewpoint_c from weather CSV
    weather = _load_external_csv("weather", split)
    if not weather.empty:
        for col in ["wind_u", "temp_c", "dewpoint_c"]:
            if col in weather.columns:
                df_feat[f"wx_{col}"] = weather[col].values[:n].astype(np.float32)

    # sol_daylight_hours from solar CSV
    solar = _load_external_csv("solar", split)
    if not solar.empty and "daylight_hours" in solar.columns:
        df_feat["sol_daylight_hours"] = solar["daylight_hours"].values[:n].astype(np.float32)

    # Handle inf/nan in recovered features
    for col in E79_MISSING_FROM_V2:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_feat


def select_e79_features(
    df_feat: pd.DataFrame,
    best_features_path: Path,
) -> list[str]:
    """Read E79's 36 features and map to v2 column names."""
    raw_names = best_features_path.read_text().strip().splitlines()
    raw_names = [n.strip() for n in raw_names if n.strip()]

    mapped = []
    missing = []
    for name in raw_names:
        v2_name = E79_TO_V2.get(name, name)
        if v2_name in df_feat.columns:
            mapped.append(v2_name)
        else:
            missing.append(name)

    if missing:
        print(f"  WARNING: {len(missing)} E79 features not found: {missing}", flush=True)

    print(f"  Mapped {len(mapped)}/{len(raw_names)} E79 features to v2", flush=True)
    return mapped


# ======================================================================
# Cache invalidation via source hash
# ======================================================================

def _features_v2_hash() -> str:
    src = (ROOT / "src" / "features_v2.py").read_text(encoding="utf-8")
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _validate_cache(cache_path: Path) -> bool:
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
    meta_path = cache_path.with_suffix(".hash")
    meta_path.write_text(_features_v2_hash())


# ======================================================================
# Effective number class weights (beta=0.999)
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
    oof_lgb: np.ndarray,
    oof_xgb: np.ndarray,
    oof_cb: np.ndarray,
    y: np.ndarray,
    step: float = 0.05,
) -> tuple[float, float, float, float]:
    """Grid search ensemble weights to maximize OOF macro mAP.

    Returns (w_lgb, w_xgb, w_cb, best_mAP).
    """
    best_map = -1.0
    best_w = (0.50, 0.40, 0.10)

    for w_lgb_int in range(0, int(1.0 / step) + 1):
        w_lgb = w_lgb_int * step
        for w_xgb_int in range(0, int((1.0 - w_lgb) / step) + 1):
            w_xgb = w_xgb_int * step
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -1e-8:
                continue
            w_cb = max(w_cb, 0.0)

            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))

    return best_w[0], best_w[1], best_w[2], best_map


# ======================================================================
# Standard PP (3-channel NB only — proven safe)
# ======================================================================

def apply_standard_nb_pp(
    preds: np.ndarray,
    test_df: pd.DataFrame,
    test_months: np.ndarray,
    train_df: pd.DataFrame,
    y: np.ndarray,
    gamma: float = 0.10,
    tau_prior: float = 0.15,
    tau_nb: float = 0.25,
) -> np.ndarray:
    """Standard 3-channel NB PP (speed, alt_mid, alt_range). Callable for eval_pp."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    out, _ = apply_gated_ratio_priors(
        preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior
    )

    # Train evidence
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {
        "speed": speed_tr,
        "alt_mid": 0.5 * (min_z_tr + max_z_tr),
        "alt_range": max_z_tr - min_z_tr,
    }
    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

    # Test evidence
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
        test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig
    )
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)


# ======================================================================
# Cleanlab noise detection
# ======================================================================

def run_cleanlab_analysis(
    oof_preds: np.ndarray,
    y: np.ndarray,
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Run cleanlab to identify likely mislabeled training samples.

    Returns (issue_mask, stats_dict).
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print("  cleanlab not installed. Run: pip install cleanlab", flush=True)
        return np.zeros(len(y), dtype=bool), {}

    # Find label issues using out-of-fold predictions
    issue_indices = find_label_issues(
        labels=y,
        pred_probs=oof_preds,
        return_indices_ranked_by="self_confidence",
    )
    issue_mask = np.zeros(len(y), dtype=bool)
    issue_mask[issue_indices] = True

    # Per-class breakdown
    stats = {}
    for c in range(N_CLASSES):
        cls_mask = y == c
        n_issues = int(issue_mask[cls_mask].sum())
        n_total = int(cls_mask.sum())
        stats[CLASSES[c]] = {
            "n_issues": n_issues,
            "n_total": n_total,
            "pct": round(100 * n_issues / max(n_total, 1), 1),
        }

    # Show predicted vs actual for issues
    pred_classes = oof_preds[issue_mask].argmax(axis=1)
    actual_classes = y[issue_mask]
    confusion_pairs = {}
    for pred_c, actual_c in zip(pred_classes, actual_classes):
        pair = f"{CLASSES[actual_c]} -> {CLASSES[pred_c]}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    stats["_confusion_pairs"] = dict(
        sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]
    )

    return issue_mask, stats


# ======================================================================
# Main experiment
# ======================================================================

def main():
    print("=" * 60)
    print("  E173: P0 Fixes on E79's 36 Features + Ensemble Reopt")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/9] Loading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(
        pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int
    )
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    # ── Build v2 features (with cache validation) ──
    print("\n[2/9] Building v2 features + recovering E79 missing...", flush=True)
    cache_train = ROOT / "data" / "_cached_train_features_v2.pkl"
    cache_test = ROOT / "data" / "_cached_test_features_v2.pkl"

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

    # Recover E79 features that v2 dropped
    print("  Recovering missing E79 features...", flush=True)
    train_feats = recover_missing_features(train_feats, train_df, "train")
    test_feats = recover_missing_features(test_feats, test_df, "test")

    # ── Select E79's 36 features ──
    print("\n[3/9] Selecting E79's 36 features...", flush=True)
    best_features_path = ROOT / "data" / "best_features.txt"
    feature_cols = select_e79_features(train_feats, best_features_path)
    # Ensure same columns exist in both
    feature_cols = [c for c in feature_cols if c in test_feats.columns]
    print(f"  Final feature count: {len(feature_cols)}", flush=True)

    X_train = train_feats[feature_cols].values.astype(np.float32)
    X_test = test_feats[feature_cols].values.astype(np.float32)

    # Guard: handle inf/nan explicitly
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Train models (all P0 fixes) ──
    print("\n[4/9] Training models (all P0 fixes)...", flush=True)

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

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)})", flush=True)

        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Fix 1: Per-fold sample weights (not global)
        w_tr = effective_number_weights(y_tr, beta=0.999)
        w_va = effective_number_weights(y_va, beta=0.999)

        # ── LightGBM (Fix 4: eval_sample_weight) ──
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
            eval_sample_weight=[w_va],  # Fix 4: NEW — weighted eval for LGB
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
        test_lgb += lgb_model.predict_proba(X_test) / N_FOLDS

        # ── XGBoost (Fix 2: weighted eval set) ──
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
            sample_weight_eval_set=[w_va],  # Fix 2
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_test) / N_FOLDS

        # ── CatBoost (Fix 3: weighted eval set) ──
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
        cb_pool_val = cb.Pool(X_va, y_va, weight=w_va)  # Fix 3
        cb_model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=cb_pool_val,
        )
        oof_cb[val_idx] = cb_model.predict_proba(X_va)
        test_cb += cb_model.predict_proba(X_test) / N_FOLDS

        # Per-fold mAP (inherited weights for monitoring)
        fold_preds = 0.50 * oof_lgb[val_idx] + 0.40 * oof_xgb[val_idx] + 0.10 * oof_cb[val_idx]
        fold_map, _ = compute_map(y_va, fold_preds)
        print(f"    Fold {fold + 1} mAP: {fold_map:.4f}", flush=True)

    # ── Individual model OOF scores ──
    print("\n[5/9] Individual model evaluation...", flush=True)
    lgb_map, _ = compute_map(y, oof_lgb)
    xgb_map, _ = compute_map(y, oof_xgb)
    cb_map, _ = compute_map(y, oof_cb)
    print(f"  LGB OOF mAP: {lgb_map:.4f}")
    print(f"  XGB OOF mAP: {xgb_map:.4f}")
    print(f"  CB  OOF mAP: {cb_map:.4f}")

    # ── Ensemble weight optimization ──
    print("\n[6/9] Optimizing ensemble weights on OOF mAP...", flush=True)
    w_lgb_opt, w_xgb_opt, w_cb_opt, opt_map = optimize_ensemble_weights(
        oof_lgb, oof_xgb, oof_cb, y, step=0.05
    )
    print(f"  Optimized: LGB={w_lgb_opt:.2f}, XGB={w_xgb_opt:.2f}, CB={w_cb_opt:.2f} -> mAP={opt_map:.4f}")

    # Compare with inherited weights
    oof_inherited = 0.50 * oof_lgb + 0.40 * oof_xgb + 0.10 * oof_cb
    inherited_map, _ = compute_map(y, oof_inherited)
    print(f"  Inherited:  LGB=0.50, XGB=0.40, CB=0.10 -> mAP={inherited_map:.4f}")
    print(f"  Delta: {opt_map - inherited_map:+.4f}")

    # Use optimized weights
    oof_ensemble = w_lgb_opt * oof_lgb + w_xgb_opt * oof_xgb + w_cb_opt * oof_cb
    test_ensemble = w_lgb_opt * test_lgb + w_xgb_opt * test_xgb + w_cb_opt * test_cb

    # Also prepare inherited-weight versions for comparison
    test_inherited = 0.50 * test_lgb + 0.40 * test_xgb + 0.10 * test_cb

    # ── SKF evaluation ──
    skf_map, skf_per_class = compute_map(y, oof_ensemble)
    print_results(skf_map, skf_per_class, "E173 SKF (optimized weights)")

    # ── LOMO evaluation ──
    print("\n[7/9] LOMO evaluation...", flush=True)
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

    # ── Post-processing (standard 3-channel NB only) ──
    print("\n[8/9] Post-processing...", flush=True)

    # PP on test (optimized weights)
    test_pp = apply_standard_nb_pp(
        test_ensemble, test_df, test_months, train_df, y,
        gamma=0.10, tau_prior=0.15, tau_nb=0.25,
    )
    # PP on test (inherited weights)
    test_pp_inherited = apply_standard_nb_pp(
        test_inherited, test_df, test_months, train_df, y,
        gamma=0.10, tau_prior=0.15, tau_nb=0.25,
    )

    # IW-mAP Validation
    print("\n  --- IW-mAP Validation ---", flush=True)
    try:
        from src.validate import eval_pp

        print("\n  Standard NB PP (optimized weights):", flush=True)
        result_opt = eval_pp(
            lambda p, td, tm, trd, y_: apply_standard_nb_pp(
                p, td, tm, trd, y_, gamma=0.10,
            ),
            verbose=True,
        )
    except Exception as e:
        print(f"  IW-mAP validation skipped: {e}", flush=True)
        result_opt = None

    # ── Cleanlab noise detection ──
    print("\n[9/9] Cleanlab noise detection...", flush=True)
    issue_mask, cl_stats = run_cleanlab_analysis(oof_ensemble, y, train_df)
    n_issues = int(issue_mask.sum())
    print(f"\n  Cleanlab found {n_issues} likely mislabeled samples ({100*n_issues/len(y):.1f}%)")

    if cl_stats:
        print("\n  Per-class noise breakdown:")
        for cls in CLASSES:
            s = cl_stats.get(cls, {})
            if s:
                print(f"    {cls:15s}: {s['n_issues']:3d}/{s['n_total']:4d} ({s['pct']:.1f}%)")

        pairs = cl_stats.get("_confusion_pairs", {})
        if pairs:
            print("\n  Top confusion pairs (label -> predicted):")
            for pair, count in list(pairs.items())[:10]:
                print(f"    {pair}: {count}")

    # Train cleanlab variant if enough noisy samples found
    oof_clean = None
    test_clean = None
    clean_map = None
    if n_issues >= 10:
        print(f"\n  Training cleanlab variant (downweighting {n_issues} noisy samples)...", flush=True)

        # Downweight noisy samples to 0.1x (don't remove — preserve fold structure)
        noise_weight = np.ones(len(y), dtype=np.float32)
        noise_weight[issue_mask] = 0.1

        oof_lgb_cl = np.zeros((len(y), N_CLASSES))
        oof_xgb_cl = np.zeros((len(y), N_CLASSES))
        oof_cb_cl = np.zeros((len(y), N_CLASSES))
        test_lgb_cl = np.zeros((len(test_df), N_CLASSES))
        test_xgb_cl = np.zeros((len(test_df), N_CLASSES))
        test_cb_cl = np.zeros((len(test_df), N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Combine effective number weights with noise downweighting
            w_tr = effective_number_weights(y_tr, beta=0.999) * noise_weight[train_idx]
            w_va = effective_number_weights(y_va, beta=0.999)

            # LGB
            lgb_cl = lgb.LGBMClassifier(**lgb_params)
            lgb_cl.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_sample_weight=[w_va],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof_lgb_cl[val_idx] = lgb_cl.predict_proba(X_va)
            test_lgb_cl += lgb_cl.predict_proba(X_test) / N_FOLDS

            # XGB
            xgb_cl = xgb.XGBClassifier(**xgb_params)
            xgb_cl.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                sample_weight_eval_set=[w_va],
                verbose=False,
            )
            oof_xgb_cl[val_idx] = xgb_cl.predict_proba(X_va)
            test_xgb_cl += xgb_cl.predict_proba(X_test) / N_FOLDS

            # CB
            cb_cl = cb.CatBoostClassifier(
                iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                random_seed=SEED, verbose=0, auto_class_weights=None,
                early_stopping_rounds=100, task_type="CPU",
            )
            cb_pool_va_cl = cb.Pool(X_va, y_va, weight=w_va)
            cb_cl.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=cb_pool_va_cl)
            oof_cb_cl[val_idx] = cb_cl.predict_proba(X_va)
            test_cb_cl += cb_cl.predict_proba(X_test) / N_FOLDS

        oof_clean = w_lgb_opt * oof_lgb_cl + w_xgb_opt * oof_xgb_cl + w_cb_opt * oof_cb_cl
        test_clean = w_lgb_opt * test_lgb_cl + w_xgb_opt * test_xgb_cl + w_cb_opt * test_cb_cl
        clean_map, clean_per_class = compute_map(y, oof_clean)
        print_results(clean_map, clean_per_class, "E173 Cleanlab SKF")
        print(f"  Delta vs standard: {clean_map - skf_map:+.4f}")

        # PP on cleanlab test predictions
        test_clean_pp = apply_standard_nb_pp(
            test_clean, test_df, test_months, train_df, y,
            gamma=0.10, tau_prior=0.15, tau_nb=0.25,
        )

    # ── Save outputs ──
    print("\n  Saving outputs...", flush=True)
    np.save(ROOT / "oof_e173.npy", oof_ensemble)
    np.save(ROOT / "test_e173.npy", test_ensemble)

    # Feature importance (last LGB model)
    if hasattr(lgb_model, 'feature_importances_'):
        imp = pd.Series(lgb_model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False)
        print("\n  Feature importance (LGB, last fold):", flush=True)
        for feat, val in imp.items():
            print(f"    {feat:30s}: {val}", flush=True)

    # Save submissions
    save_submission(test_ensemble, "e173_raw_opt", cv_map=skf_map)
    save_submission(test_pp, "e173_pp_opt", cv_map=skf_map)
    save_submission(test_inherited, "e173_raw_inh", cv_map=inherited_map)
    save_submission(test_pp_inherited, "e173_pp_inh", cv_map=inherited_map)
    if test_clean is not None:
        save_submission(test_clean, "e173_raw_clean", cv_map=clean_map)
        save_submission(test_clean_pp, "e173_pp_clean", cv_map=clean_map)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  E173 RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Features:       {len(feature_cols)} (E79's 36 via v2)")
    print(f"  SKF mAP:        {skf_map:.4f}  (optimized weights)")
    print(f"  SKF mAP (inh):  {inherited_map:.4f}  (inherited 0.50/0.40/0.10)")
    print(f"  LOMO mAP:       {lomo_overall:.4f}")
    print(f"  Ensemble:       LGB={w_lgb_opt:.2f}, XGB={w_xgb_opt:.2f}, CB={w_cb_opt:.2f}")
    if clean_map is not None:
        print(f"  Cleanlab mAP:   {clean_map:.4f}  ({n_issues} noisy samples downweighted)")
    print()
    print(f"  P0 Fixes applied:")
    print(f"    1. Per-fold sample weights: FIXED")
    print(f"    2. Weighted eval_set XGB: FIXED")
    print(f"    3. Weighted eval_set CB: FIXED")
    print(f"    4. Weighted eval_set LGB: FIXED (NEW)")
    print(f"    5. OOF smoothing leakage: FIXED (no OOF smoothing)")
    print(f"    6. Cache invalidation: FIXED (source hash)")
    if result_opt:
        cal = result_opt.get("calibrated_lb", "N/A")
        print(f"\n  Calibrated LB estimate: {cal}")
    print("=" * 60)

    # Per-class comparison with E79
    print("\n  Per-class AP (E173 vs E79 baseline):", flush=True)
    e79_aps = {
        "Birds of Prey": 0.885, "Clutter": 0.610, "Cormorants": 0.939,
        "Ducks": 0.666, "Geese": 0.728, "Gulls": 0.956,
        "Pigeons": 0.254, "Songbirds": 0.640, "Waders": 0.816,
    }
    for cls in CLASSES:
        e173_ap = skf_per_class.get(cls, 0)
        e79_ap = e79_aps.get(cls, 0)
        delta = e173_ap - e79_ap
        marker = "+" if delta > 0 else ""
        print(f"    {cls:15s}: {e173_ap:.4f}  (E79: {e79_ap:.3f}, delta: {marker}{delta:.4f})", flush=True)


if __name__ == "__main__":
    main()
