"""E79: Pruned base model with HP-tuned CatBoost.

Rebuild E50-equivalent pipeline using validated 36-feature set from backward
elimination + Optuna-tuned CatBoost.

Key change vs E50: uses SKF (5-fold StratifiedKFold) for base model training
instead of LOMO. SKF produces higher-quality OOF + test preds. Post-processing
params will be validated on shared-month (Sep/Oct) OOF in downstream experiments.
Optuna tuning still uses LOMO to avoid picking HPs that overfit to temporal leakage.

Pipeline:
  1. Build 36-feature set (core + flight_mode + weakclass + tabular + weather + solar)
  2. Optuna-tune CatBoost HPs using LOMO objective (honest HP selection)
  3. Train LGB + XGB + CB ensemble with 5-fold SKF (better OOF + test preds)
  4. Binary specialists for Waders + Pigeons (SKF)
  5. Per-class alpha blend (optimized on SKF OOF)
  6. Save oof_e79.npy, test_e79.npy
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

# 36 validated features from backward elimination
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def add_weather_solar(train_feats, test_feats):
    """Add weather + solar features (no GBIF -- eliminated by backward elim)."""
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


def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)


# ====================================================================
print("=" * 70, flush=True)
print("E79 PRUNED TUNED BASE (SKF)".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data and build features -----------------------------------
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
unique_months = sorted(np.unique(train_months))

print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# Prune to 36 validated features
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
missing = [f for f in KEEP_FEATURES if f not in train_feats.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing: {missing}", flush=True)
print(f"  Using {len(available)}/{len(KEEP_FEATURES)} pruned features", flush=True)

train_feats = train_feats[available]
test_feats = test_feats[available]

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X.shape[1]}", flush=True)

# -- Effective number class weights (beta=0.999, from E15) ----------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- Optuna tuning for CatBoost on LOMO (honest HP selection) -------
# Use LOMO for HP search only -- avoids temporal leakage in HP selection
print("\n--- Optuna CatBoost tuning (LOMO objective) ---", flush=True)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def cb_lomo_objective(trial):
        params = {
            "iterations": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "auto_class_weights": "Balanced",
            "random_seed": SEED,
            "verbose": 0,
            "early_stopping_rounds": 100,
            "task_type": "GPU",
        }
        oof_preds = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        for month in unique_months:
            va_idx = np.where(train_months == month)[0]
            tr_idx = np.where(train_months != month)[0]
            cb = CatBoostClassifier(**params)
            cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
            oof_preds[va_idx] = cb.predict_proba(X[va_idx])
        lomo_map, _ = compute_map(y, oof_preds)
        return lomo_map

    study = optuna.create_study(direction="maximize")
    study.optimize(cb_lomo_objective, n_trials=20, show_progress_bar=False)
    best_cb_params = study.best_params
    print(f"  Best LOMO mAP: {study.best_value:.4f}", flush=True)
    print(f"  Best params: {best_cb_params}", flush=True)
except ImportError:
    print("  Optuna not installed, using defaults", flush=True)
    best_cb_params = {
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "border_count": 128,
    }

# -- SKF CV with LGB + XGB + CB ------------------------------------
# SKF for OOF quality + test pred averaging (5 folds > LOMO's 4)
print("\n--- SKF ensemble training (5-fold) ---", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

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

    # CatBoost (Optuna-tuned HPs)
    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=best_cb_params.get("learning_rate", 0.03),
        depth=best_cb_params.get("depth", 6),
        l2_leaf_reg=best_cb_params.get("l2_leaf_reg", 3.0),
        bagging_temperature=best_cb_params.get("bagging_temperature", 0.5),
        random_strength=best_cb_params.get("random_strength", 1.0),
        border_count=best_cb_params.get("border_count", 128),
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=0,
        early_stopping_rounds=100,
        task_type="GPU",
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

# Individual model scores (SKF)
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# -- Weight optimization on SKF OOF --------------------------------
print("\n--- Ensemble weight optimization ---", flush=True)
best_w = None
best_ens_map = -1.0
for w_lgb in np.arange(0.0, 0.55, 0.05):
    for w_xgb in np.arange(0.0, 0.55, 0.05):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01 or w_cb > 1.01:
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
print_results(base_map, base_per, label="E79 ensemble (SKF OOF)")

# -- Binary specialists for Waders + Pigeons (SKF) ------------------
print("\n--- Training specialists (SKF) ---", flush=True)
specialist_oof = {}
specialist_test = {}
ap_delta = {}

for cls in SPECIALIST_CLASSES:
    idx = CLASSES.index(cls)
    y_bin = (y == idx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        cb_spec = CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=80, task_type="GPU",
        )
        cb_spec.fit(X[tr_idx], y_bin[tr_idx], eval_set=(X[va_idx], y_bin[va_idx]), verbose=0)
        oof_bin[va_idx] = cb_spec.predict_proba(X[va_idx])[:, 1]
        test_bin += cb_spec.predict_proba(X_test)[:, 1] / N_FOLDS

    ap_base = average_precision_score(y_bin, oof_ens[:, idx])
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
print_results(best_map, best_per, label="E79 final (SKF OOF)")

# -- Save artifacts ------------------------------------------------
print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e79.npy", best_oof)
np.save(ROOT / "test_e79.npy", best_test)
save_submission(best_test, "e79_pruned_tuned_base", cv_map=best_map)

print("\nDone.", flush=True)
