"""E166: Per-Month Ensemble Routing.

Use different ensemble weights and PP strength per test month.

Steps:
  1. Detect test months from timestamps
  2. Per-month ensemble weights: optimize LGB/XGB/CB separately per LOMO fold
  3. Per-month PP gamma: shared months get minimal PP, unseen months moderate
  4. Combine with MLLS calibration from E165

Uses E79's existing per-model OOF/test predictions if available,
otherwise retrains from scratch.
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
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)
from src.validate import (
    _temperature_scale, _find_temperature, _mlls_estimate,
    eval_pp,
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# LOMO fold mapping
FOLD_TO_PROXY = {
    9:  [9],       # Sep -> Sep (shared)
    10: [10],      # Oct -> Oct (shared)
    1:  [2, 12],   # Jan -> Feb + Dec (winter)
    4:  [5],       # Apr -> May (spring)
}

TEST_TO_TRAIN_PROXY = {
    2:  1, 5: 4, 9: 9, 10: 10, 12: 1,
}

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]


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


def train_lgb(X_tr, y_tr, X_va, y_va, X_test, sample_weights_tr):
    from lightgbm import LGBMClassifier
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    return lgb.predict_proba(X_va), lgb.predict_proba(X_test)


def train_xgb(X_tr, y_tr, X_va, y_va, X_test, sample_weights_tr):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            sample_weight=sample_weights_tr, verbose=False)
    return xgb.predict_proba(X_va), xgb.predict_proba(X_test)


def train_cb(X_tr, y_tr, X_va, y_va, X_test):
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5,
        random_strength=1.0, border_count=128,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    return cb.predict_proba(X_va), cb.predict_proba(X_test)


def optimize_weights(va_lgb, va_xgb, va_cb, y_va):
    """Grid search ensemble weights."""
    best_w = (0.5, 0.3, 0.2)
    best_map = -1.0
    for w_lgb in np.arange(0.0, 0.80, 0.05):
        for w_xgb in np.arange(0.0, 0.80, 0.05):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01 or w_cb > 1.01:
                continue
            oof = w_lgb * va_lgb + w_xgb * va_xgb + max(w_cb, 0) * va_cb
            m, _ = compute_map(y_va, oof)
            if m > best_map:
                best_map = m
                best_w = (w_lgb, w_xgb, max(w_cb, 0))
    return best_w, best_map


def main():
    print("=" * 70, flush=True)
    print("E166 PER-MONTH ENSEMBLE ROUTING".center(70), flush=True)
    print("=" * 70, flush=True)

    # -- Load data ---
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    unique_months = sorted(np.unique(train_months))

    # Effective number class weights
    counts_arr = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts_arr / counts_arr.sum()
    beta = 0.999
    eff_n = (1.0 - beta ** counts_arr) / (1.0 - beta)
    class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
    class_weights_arr /= class_weights_arr.sum() / N_CLASSES
    sample_weights = class_weights_arr[y]

    # -- Build features ---
    print("\nBuilding features...", flush=True)
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]
    train_feats, test_feats = add_weather_solar(train_feats, test_feats)

    available = [f for f in KEEP_FEATURES if f in train_feats.columns]
    X = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X_test = test_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    print(f"  Features: {X.shape[1]}", flush=True)

    # ============================================================
    # LOMO training with per-fold weight optimization
    # ============================================================
    print("\n--- LOMO training with per-fold weights ---", flush=True)

    lomo_oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    lomo_test = {}  # month -> (test_lgb, test_xgb, test_cb, weights)

    for held_out in unique_months:
        tr_mask = train_months != held_out
        va_mask = train_months == held_out
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        print(f"\n  LOMO M{held_out:02d}: train={tr_mask.sum()} val={va_mask.sum()}", flush=True)

        va_lgb, te_lgb = train_lgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                                    X_test, sample_weights[tr_idx])
        va_xgb, te_xgb = train_xgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                                     X_test, sample_weights[tr_idx])
        va_cb, te_cb = train_cb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X_test)

        w, fold_map = optimize_weights(va_lgb, va_xgb, va_cb, y[va_idx])
        print(f"    Weights: LGB={w[0]:.2f} XGB={w[1]:.2f} CB={w[2]:.2f} mAP={fold_map:.4f}", flush=True)

        lomo_oof[va_idx] = w[0] * va_lgb + w[1] * va_xgb + w[2] * va_cb
        lomo_test[held_out] = {
            "lgb": te_lgb, "xgb": te_xgb, "cb": te_cb, "weights": w,
        }

    lomo_map, lomo_per = compute_map(y, lomo_oof)
    print_results(lomo_map, lomo_per, label="E166 LOMO OOF (per-fold weights)")

    # ============================================================
    # Route test predictions by month
    # ============================================================
    print("\n--- Test month routing ---", flush=True)

    test_routed = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for test_m in sorted(np.unique(test_months)):
        proxy = TEST_TO_TRAIN_PROXY.get(test_m, 1)
        mask = test_months == test_m
        d = lomo_test[proxy]
        w = d["weights"]
        preds = w[0] * d["lgb"] + w[1] * d["xgb"] + w[2] * d["cb"]
        test_routed[mask] = preds[mask]
        print(f"  M{test_m:02d} ({mask.sum()} rows) -> fold M{proxy:02d} "
              f"(w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}])", flush=True)

    # ============================================================
    # Per-month PP gamma sweep
    # ============================================================
    print("\n--- Per-month PP gamma sweep ---", flush=True)

    # Build NB parameters from full training data
    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed_tr, "alt_mid": 0.5*(min_z_tr+max_z_tr), "alt_range": max_z_tr-min_z_tr}
    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_te, "alt_mid": 0.5*(min_z_te+max_z_te), "alt_range": max_z_te-min_z_te}
    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig,
    )

    priors = build_gbif_priors(p_train)

    pp_configs = [
        # name, tau_prior, tau_nb, gamma_shared, gamma_unseen
        ("A_no_pp", 0.15, 0.25, 0.0, 0.0),
        ("B_unseen_only_g010", 0.15, 0.25, 0.0, 0.10),
        ("C_unseen_only_g020", 0.15, 0.25, 0.0, 0.20),
        ("D_light_shared_g005", 0.15, 0.25, 0.05, 0.10),
        ("E_uniform_g010", 0.15, 0.25, 0.10, 0.10),
        ("F_differential_g003_g015", 0.15, 0.25, 0.03, 0.15),
    ]

    for pp_name, tau_prior, tau_nb, gamma_s, gamma_u in pp_configs:
        # Apply GBIF priors (unseen months only)
        out, _ = apply_gated_ratio_priors(
            test_routed, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior,
        )

        # Apply NB PoE with different gamma for shared vs unseen
        margin = top2_margin(out)
        for test_m in sorted(np.unique(test_months)):
            mask_m = test_months == test_m
            is_unseen = test_m in UNSEEN_MONTHS
            gamma = gamma_u if is_unseen else gamma_s
            if gamma == 0:
                continue
            gate = mask_m & (margin < tau_nb)
            if gate.sum() == 0:
                continue
            # Apply NB PoE only to this month's rows
            ll = loglike[gate]
            ll = ll - ll.max(axis=1, keepdims=True)
            fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
            out[gate] = out[gate] * fac
            out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)

        out = renorm_rows(out)

        # Evaluate via IW-mAP
        def make_pp_eval(pp_n=pp_name, tp=tau_prior, tn=tau_nb, gs=gamma_s, gu=gamma_u):
            def pp_fn(preds, test_df_v, test_months_v, train_df_v, y_v):
                counts_v = np.bincount(y_v, minlength=N_CLASSES).astype(float)
                p_tr_v = counts_v / counts_v.sum()
                priors_v = build_gbif_priors(p_tr_v)
                out_v, _ = apply_gated_ratio_priors(
                    preds, test_months_v, p_tr_v, priors_v, BASE_ALPHA, tau=tp,
                )

                speed_v = pd.to_numeric(train_df_v["airspeed"], errors="coerce").values.astype(float)
                minz_v = pd.to_numeric(train_df_v["min_z"], errors="coerce").values.astype(float)
                maxz_v = pd.to_numeric(train_df_v["max_z"], errors="coerce").values.astype(float)
                cont_v = {"speed": speed_v, "alt_mid": 0.5*(minz_v+maxz_v), "alt_range": maxz_v-minz_v}
                sl, lps, mu_v, sig_v = build_nb_params(train_df_v, y_v, cont_v)

                speed_tv = pd.to_numeric(test_df_v["airspeed"], errors="coerce").values.astype(float)
                minz_tv = pd.to_numeric(test_df_v["min_z"], errors="coerce").values.astype(float)
                maxz_tv = pd.to_numeric(test_df_v["max_z"], errors="coerce").values.astype(float)
                cont_tv = {"speed": speed_tv, "alt_mid": 0.5*(minz_tv+maxz_tv), "alt_range": maxz_tv-minz_tv}
                wts = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
                ll_v = compute_log_p_u_given_c(test_df_v, sl, lps, cont_tv, wts, None, mu_v, sig_v)

                margin_v = top2_margin(out_v)
                for m_v in np.unique(test_months_v):
                    mask_mv = test_months_v == m_v
                    is_unseen_v = m_v in UNSEEN_MONTHS
                    g = gu if is_unseen_v else gs
                    if g == 0:
                        continue
                    gate_v = mask_mv & (margin_v < tn)
                    if gate_v.sum() == 0:
                        continue
                    ll_g = ll_v[gate_v]
                    ll_g = ll_g - ll_g.max(axis=1, keepdims=True)
                    fac_v = np.exp(np.clip(g * ll_g, -50.0, 50.0))
                    out_v[gate_v] = out_v[gate_v] * fac_v
                    out_v[gate_v] = out_v[gate_v] / np.clip(
                        out_v[gate_v].sum(axis=1, keepdims=True), 1e-12, None)
                return renorm_rows(out_v)
            return pp_fn

        result = eval_pp(make_pp_eval(), verbose=False)
        cal_lb = result.get("calibrated_lb", "N/A")
        iw = result["estimated_lb"]
        delta = result["estimated_delta"]
        shared = result["shared_delta"]
        print(f"  {pp_name:<30s}  IW={iw:.4f} d={delta:+.4f}  "
              f"shared={shared:+.4f}  calLB={cal_lb}  {result['recommendation']}", flush=True)

        save_submission(out, f"e166_{pp_name}")

    # ============================================================
    # Also save the routed-only (no PP) variant
    # ============================================================
    save_submission(test_routed, f"e166_routed_no_pp", cv_map=lomo_map)

    # Save artifacts
    np.save(ROOT / "oof_e166_lomo.npy", lomo_oof)
    np.save(ROOT / "test_e166_routed.npy", test_routed)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
