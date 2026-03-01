"""E48: Pruned Features + Optuna HP Tuning

Based on analysis (2026-02-18):
  - Backward elimination found 36 features optimal (LOMO 0.3949 vs 139-feat 0.3756)
  - Zero HP tuning done across 47 experiments -- biggest remaining opportunity

Strategy:
  Part A: 36-feature baseline with default params (confirm backward elimination gain)
  Part B: Optuna HP tuning for CatBoost (dominates ensemble)
  Part C: Optuna HP tuning for LGB + XGB
  Part D: Full tuned ensemble with LOMO + SKF + test predictions
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999
P = lambda *a, **kw: print(*a, **kw, flush=True)

# The 36 features from backward elimination
BEST_36 = [
    "alt_max", "alt_median", "alt_q75",
    "rcs_mean", "rcs_median", "rcs_q25", "rcs_q75",
    "speed_median", "avg_ground_speed", "accel_std",
    "bearing_change_mean", "lon_mean", "lat_mean", "lon_std", "lat_std",
    "alt_change_halves", "speed_x_alt", "curvature_mean",
    "slow_flight_frac", "alt_rate_mean", "rcs_per_alt",
    "airspeed", "airspeed_vs_ground", "size_x_alt", "rcs_for_size",
    "speed_mean", "speed_std",
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]

# ======================================================================
# Load data
# ======================================================================
P("=" * 60)
P("E48: PRUNED FEATURES + OPTUNA HP TUNING")
P("=" * 60)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# Build features
P("\nBuilding features...")
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Load weather
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

# Load solar
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

# Clean
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

# Select 36 features
available = [c for c in BEST_36 if c in train_feats.columns]
missing = [c for c in BEST_36 if c not in train_feats.columns]
if missing:
    P(f"WARNING: Missing features: {missing}")
P(f"Using {len(available)} / {len(BEST_36)} features")

X_all = train_feats[available].values.astype(np.float32)
X_test_all = test_feats[available].values.astype(np.float32)
fn = list(available)


def lomo_evaluate_cb(params, X, y, sw, months, unique_m):
    """Evaluate CatBoost with given params using LOMO."""
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_m:
        va = np.where(months == m)[0]
        tr = np.where(months != m)[0]
        cb = CatBoostClassifier(**params)
        cb.fit(X[tr], y[tr], eval_set=(X[va], y[va]), verbose=0, sample_weight=sw[tr])
        oof[va] = cb.predict_proba(X[va])
    score, _ = compute_map(y, oof)
    return score


def lomo_evaluate_lgb(params, X, y, sw, months, unique_m, fn):
    """Evaluate LGB with given params using LOMO."""
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_m:
        va = np.where(months == m)[0]
        tr = np.where(months != m)[0]
        dtrain = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], feature_name=fn)
        dval = lgb.Dataset(X[va], label=y[va], feature_name=fn, reference=dtrain)
        model = lgb.train(params, dtrain, 2000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va] = model.predict(X[va])
    score, _ = compute_map(y, oof)
    return score


def lomo_evaluate_xgb(params, X, y, sw, months, unique_m, fn):
    """Evaluate XGB with given params using LOMO."""
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_m:
        va = np.where(months == m)[0]
        tr = np.where(months != m)[0]
        dtrain = xgb.DMatrix(X[tr], label=y[tr], weight=sw[tr], feature_names=fn)
        dval = xgb.DMatrix(X[va], label=y[va], feature_names=fn)
        model = xgb.train(params, dtrain, 2000,
                          evals=[(dval, "val")],
                          early_stopping_rounds=80, verbose_eval=0)
        oof[va] = model.predict(dval)
    score, _ = compute_map(y, oof)
    return score


# ======================================================================
# Part A: Baseline with 36 features + default params
# ======================================================================
P("\n" + "=" * 60)
P("PART A: 36-FEATURE BASELINE (default params)")
P("=" * 60)

LGB_DEFAULT = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}
XGB_DEFAULT = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

t0 = time.time()

# LOMO with default params, 36 features
oof_baseline = np.zeros((len(y), N_CLASSES))
for m in unique_months:
    va = np.where(train_months == m)[0]
    tr = np.where(train_months != m)[0]
    X_tr, X_va = X_all[tr], X_all[va]
    y_tr, y_va = y[tr], y[va]
    sw_tr = sample_weights[tr]

    # LGB
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(LGB_DEFAULT, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_va)

    # XGB
    m_xgb = xgb.train(XGB_DEFAULT, xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, feature_names=fn),
                       2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                       early_stopping_rounds=80, verbose_eval=0)
    p_xgb = m_xgb.predict(xgb.DMatrix(X_va, feature_names=fn))

    # CB
    cb = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=sw_tr)
    p_cb = cb.predict_proba(X_va)

    oof_baseline[va] = 0.33 * p_lgb + 0.33 * p_xgb + 0.34 * p_cb
    fold_map, _ = compute_map(y_va, oof_baseline[va])
    P(f"  Month {m}: mAP={fold_map:.4f} (n={len(va)})")

baseline_map, baseline_per = compute_map(y, oof_baseline)
P(f"\n  Part A Baseline LOMO: {baseline_map:.4f} ({time.time()-t0:.0f}s)")
P(f"  Per-class:")
for cls in CLASSES:
    P(f"    {cls:<15s}: {baseline_per.get(cls, 0):.4f}")

# ======================================================================
# Part B: Optuna HP tuning for CatBoost
# ======================================================================
P("\n" + "=" * 60)
P("PART B: OPTUNA HP TUNING -- CatBoost (50 trials)")
P("=" * 60)

t0 = time.time()


def cb_objective(trial):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 80,
        "task_type": "GPU",
    }
    score = lomo_evaluate_cb(params, X_all, y, sample_weights, train_months, unique_months)
    return score


study_cb = optuna.create_study(direction="maximize", study_name="catboost_lomo")
study_cb.optimize(cb_objective, n_trials=50, show_progress_bar=False)

P(f"\n  CatBoost best LOMO: {study_cb.best_value:.4f} ({time.time()-t0:.0f}s)")
P(f"  Best params: {study_cb.best_params}")

# ======================================================================
# Part C: Optuna HP tuning for LGB + XGB
# ======================================================================
P("\n" + "=" * 60)
P("PART C: OPTUNA HP TUNING -- LGB (30 trials) + XGB (30 trials)")
P("=" * 60)

t0 = time.time()


def lgb_objective(trial):
    params = {
        "objective": "multiclass", "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
    }
    score = lomo_evaluate_lgb(params, X_all, y, sample_weights, train_months, unique_months, fn)
    return score


study_lgb = optuna.create_study(direction="maximize", study_name="lgb_lomo")
study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=False)

P(f"\n  LGB best LOMO: {study_lgb.best_value:.4f} ({time.time()-t0:.0f}s)")
P(f"  Best params: {study_lgb.best_params}")

t0 = time.time()


def xgb_objective(trial):
    params = {
        "objective": "multi:softprob", "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "seed": 42, "nthread": -1, "verbosity": 0,
        "device": "cuda", "tree_method": "hist",
    }
    score = lomo_evaluate_xgb(params, X_all, y, sample_weights, train_months, unique_months, fn)
    return score


study_xgb = optuna.create_study(direction="maximize", study_name="xgb_lomo")
study_xgb.optimize(xgb_objective, n_trials=30, show_progress_bar=False)

P(f"\n  XGB best LOMO: {study_xgb.best_value:.4f} ({time.time()-t0:.0f}s)")
P(f"  Best params: {study_xgb.best_params}")

# ======================================================================
# Part D: Full tuned ensemble -- LOMO + weight search + SKF + test preds
# ======================================================================
P("\n" + "=" * 60)
P("PART D: FULL TUNED ENSEMBLE")
P("=" * 60)

# Build best params
best_lgb = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
    **study_lgb.best_params,
}
best_xgb = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
    **study_xgb.best_params,
}
best_cb_params = {
    "iterations": 2000,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42, "verbose": 0,
    "early_stopping_rounds": 80, "task_type": "GPU",
    **study_cb.best_params,
}

# LOMO with tuned params -- collect per-model OOF for weight optimization
P("\n  Running LOMO with tuned params...")
oof_lgb_all = np.zeros((len(y), N_CLASSES))
oof_xgb_all = np.zeros((len(y), N_CLASSES))
oof_cb_all = np.zeros((len(y), N_CLASSES))

t0 = time.time()
for m in unique_months:
    va = np.where(train_months == m)[0]
    tr = np.where(train_months != m)[0]
    X_tr, X_va = X_all[tr], X_all[va]
    y_tr, y_va = y[tr], y[va]
    sw_tr = sample_weights[tr]

    # LGB tuned
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_l = lgb.train(best_lgb, dtrain, 2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb_all[va] = m_l.predict(X_va)

    # XGB tuned
    m_x = xgb.train(best_xgb, xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, feature_names=fn),
                     2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                     early_stopping_rounds=80, verbose_eval=0)
    oof_xgb_all[va] = m_x.predict(xgb.DMatrix(X_va, feature_names=fn))

    # CB tuned
    cb = CatBoostClassifier(**best_cb_params)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=sw_tr)
    oof_cb_all[va] = cb.predict_proba(X_va)

    # Per-month scores
    for name, oof_m in [("LGB", oof_lgb_all), ("XGB", oof_xgb_all), ("CB", oof_cb_all)]:
        s, _ = compute_map(y_va, oof_m[va])
        P(f"    Month {m} {name}: {s:.4f}")
    ens = 0.33 * oof_lgb_all[va] + 0.33 * oof_xgb_all[va] + 0.34 * oof_cb_all[va]
    s, _ = compute_map(y_va, ens)
    P(f"    Month {m} ENS(0.33/0.33/0.34): {s:.4f}")

# Per-model LOMO
for name, oof_m in [("LGB", oof_lgb_all), ("XGB", oof_xgb_all), ("CB", oof_cb_all)]:
    s, _ = compute_map(y, oof_m)
    P(f"\n  {name} tuned LOMO: {s:.4f}")

# Fixed weight ensemble
oof_fixed = 0.33 * oof_lgb_all + 0.33 * oof_xgb_all + 0.34 * oof_cb_all
fixed_map, fixed_per = compute_map(y, oof_fixed)
P(f"\n  Ensemble (0.33/0.33/0.34) tuned LOMO: {fixed_map:.4f}")

# Simple weight grid search
P("\n  Weight grid search...")
best_w, best_w_score = (0.33, 0.33, 0.34), fixed_map
for w1 in np.arange(0.0, 1.05, 0.05):
    for w2 in np.arange(0.0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        oof_w = w1 * oof_lgb_all + w2 * oof_xgb_all + w3 * oof_cb_all
        s, _ = compute_map(y, oof_w)
        if s > best_w_score:
            best_w_score = s
            best_w = (round(w1, 2), round(w2, 2), round(w3, 2))

P(f"  Best weights: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}")
P(f"  Best weighted LOMO: {best_w_score:.4f}")

# But use fixed weights for submission (weight opt on OOF is biased)
# Just report the grid search result for information
P(f"\n  NOTE: Using fixed 0.33/0.33/0.34 for submission (weight opt on OOF overfits)")

# Final LOMO per-class
oof_final = oof_fixed
final_map, final_per = compute_map(y, oof_final)
P(f"\n  FINAL TUNED LOMO: {final_map:.4f}")
P(f"  Per-class:")
for cls in CLASSES:
    base_ap = baseline_per.get(cls, 0)
    tuned_ap = final_per.get(cls, 0)
    delta = tuned_ap - base_ap
    P(f"    {cls:<15s}: {tuned_ap:.4f} (delta {delta:+.4f} vs default params)")

P(f"\n  vs Part A baseline: {final_map:.4f} vs {baseline_map:.4f} (delta {final_map - baseline_map:+.4f})")
P(f"  ({time.time()-t0:.0f}s)")

# ======================================================================
# SKF + test predictions with tuned params
# ======================================================================
P("\n" + "=" * 60)
P("SKF EVALUATION + TEST PREDICTIONS (tuned params)")
P("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.zeros((len(y), N_CLASSES))
test_lgb_pred = np.zeros((len(X_test_all), N_CLASSES))
test_xgb_pred = np.zeros((len(X_test_all), N_CLASSES))
test_cb_pred = np.zeros((len(X_test_all), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr = sample_weights[tr_idx]

    # LGB
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_l = lgb.train(best_lgb, dtrain, 2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_skf[va_idx] += 0.33 * m_l.predict(X_va)
    test_lgb_pred += m_l.predict(X_test_all) / 5

    # XGB
    m_x = xgb.train(best_xgb, xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, feature_names=fn),
                     2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                     early_stopping_rounds=80, verbose_eval=0)
    oof_skf[va_idx] += 0.33 * m_x.predict(xgb.DMatrix(X_va, feature_names=fn))
    test_xgb_pred += m_x.predict(xgb.DMatrix(X_test_all, feature_names=fn)) / 5

    # CB
    cb = CatBoostClassifier(**best_cb_params)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=sw_tr)
    oof_skf[va_idx] += 0.34 * cb.predict_proba(X_va)
    test_cb_pred += cb.predict_proba(X_test_all) / 5

    fold_map, _ = compute_map(y_va, oof_skf[va_idx])
    P(f"  SKF Fold {fold_idx}: mAP={fold_map:.4f} (n={len(va_idx)})")

test_pred = 0.33 * test_lgb_pred + 0.33 * test_xgb_pred + 0.34 * test_cb_pred
skf_map, skf_per = compute_map(y, oof_skf)

P(f"\n  SKF CV mAP: {skf_map:.4f}")
P(f"  LOMO mAP:   {final_map:.4f}")
P(f"  Gap:        {skf_map - final_map:.4f}")

P(f"\n  Per-class SKF:")
P(f"  {'Class':<15s} {'SKF':>7s} {'LOMO':>7s}")
for cls in CLASSES:
    s = skf_per.get(cls, 0)
    l = final_per.get(cls, 0)
    P(f"  {cls:<15s} {s:>7.4f} {l:>7.4f}")

# Test distribution
P(f"\n  Test class distribution (argmax):")
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    P(f"    {cls:<15s}: {dist[i]}")

# Save
np.save(ROOT / "oof_e48.npy", oof_skf)
np.save(ROOT / "test_e48.npy", test_pred)
save_submission(test_pred, "e48_pruned_tuned", cv_map=skf_map)

# ======================================================================
# Summary
# ======================================================================
P("\n" + "=" * 60)
P("E48 SUMMARY")
P("=" * 60)
P(f"  Features: {len(available)} (pruned from 139)")
P(f"  Part A baseline (default params): LOMO = {baseline_map:.4f}")
P(f"  Part D tuned ensemble:            LOMO = {final_map:.4f} (delta {final_map - baseline_map:+.4f})")
P(f"  SKF CV mAP:                       {skf_map:.4f}")
P(f"  Best CB params: {study_cb.best_params}")
P(f"  Best LGB params: {study_lgb.best_params}")
P(f"  Best XGB params: {study_xgb.best_params}")
P(f"  Weight grid best: {best_w} -> {best_w_score:.4f} (NOT used for submission)")
P("\nDone!")
