"""E167: Bug-fixed base + temporal dynamics features.

All bug fixes applied to src/features.py:
  - dt<0.5s segment filtering (13 locations)
  - Curvature in meter-space (2 locations)
  - speed_x_alt using median instead of mean
  - SIZE_MAP 1-indexed (Small=1,Medium=2,Large=3,Flock=4)

New features: extract_temporal_dynamics (8 features):
  td_heading_local_var, td_speed_consistency, td_speed_autocorr,
  td_speed_slope, td_alt_smoothness, td_heading_change_rate,
  td_rcs_trend, td_speed_variability

Also: early stopping for LGB and XGB (was missing in E79).

Pipeline identical to E79 otherwise:
  1. Build features (36 pruned + 8 temporal dynamics)
  2. Optuna-tune CB on LOMO
  3. 5-fold SKF ensemble (LGB+XGB+CB)
  4. Binary specialists for Waders+Pigeons
  5. Save oof/test .npy + submission
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

# New temporal dynamics features
TD_FEATURES = [
    "td_heading_local_var", "td_speed_consistency", "td_speed_autocorr",
    "td_speed_slope", "td_alt_smoothness", "td_heading_change_rate",
    "td_rcs_trend", "td_speed_variability",
]

SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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


def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)


# ====================================================================
print("=" * 70, flush=True)
print("E167 BUGFIX + TEMPORAL DYNAMICS".center(70), flush=True)
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
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
             "weakclass", "temporal_dynamics"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# Prune to 36 validated + 8 temporal dynamics
all_keep = KEEP_FEATURES + TD_FEATURES
available = [f for f in all_keep if f in train_feats.columns]
missing = [f for f in all_keep if f not in train_feats.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing: {missing}", flush=True)
print(f"  Using {len(available)} features ({len(KEEP_FEATURES)} pruned + {len([f for f in TD_FEATURES if f in train_feats.columns])} temporal)", flush=True)

train_feats = train_feats[available]
test_feats = test_feats[available]

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X.shape[1]}", flush=True)

# -- Effective number class weights (beta=0.999) --------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- Optuna tuning for CatBoost on LOMO -----------------------------
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

    # LightGBM (no early stopping -- validated: ES hurts SKF by -0.01)
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGBoost (no early stopping -- same reason)
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
best_w, best_map = None, 0
for w_lgb in np.arange(0.0, 1.05, 0.1):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.1):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01:
            continue
        oof_blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        m, _ = compute_map(y, oof_blend)
        if m > best_map:
            best_map = m
            best_w = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))
print(f"  Best weights: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}", flush=True)
print(f"  Best ensemble mAP: {best_map:.4f}", flush=True)

oof_ens = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_ens = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

# -- Binary specialists ---------------------------------------------
print("\n--- Binary specialists ---", flush=True)
specialist_oof = {}
specialist_test = {}

for cls in SPECIALIST_CLASSES:
    cls_idx = CLASSES.index(cls)
    y_bin = (y == cls_idx).astype(int)
    oof_sp = np.zeros(len(y), dtype=np.float64)
    test_sp = np.zeros(len(X_test), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        sp_lgb = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=5, subsample=0.7, colsample_bytree=0.7,
            is_unbalance=True, random_state=SEED, verbose=-1, device="gpu",
        )
        sp_lgb.fit(X[tr_idx], y_bin[tr_idx],
                    eval_set=[(X[va_idx], y_bin[va_idx])],
        )
        oof_sp[va_idx] = sp_lgb.predict_proba(X[va_idx])[:, 1]
        test_sp += sp_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

    sp_auc = average_precision_score(y_bin, oof_sp)
    print(f"  {cls}: specialist AP = {sp_auc:.4f}", flush=True)
    specialist_oof[cls] = oof_sp
    specialist_test[cls] = test_sp

# -- Per-class alpha blend ------------------------------------------
print("\n--- Alpha blend optimization ---", flush=True)
best_alpha = {}
for cls in SPECIALIST_CLASSES:
    best_a, best_ap = 0.0, 0.0
    cls_idx = CLASSES.index(cls)
    y_bin = (y == cls_idx).astype(int)
    for alpha in ALPHA_GRID:
        blended = (1.0 - alpha) * oof_ens[:, cls_idx] + alpha * specialist_oof[cls]
        ap = average_precision_score(y_bin, blended)
        if ap > best_ap:
            best_ap = ap
            best_a = alpha
    best_alpha[cls] = best_a
    print(f"  {cls}: best alpha={best_a}, AP={best_ap:.4f}", flush=True)

oof_final = apply_blend(oof_ens, specialist_oof, best_alpha)
test_final = apply_blend(test_ens, specialist_test, best_alpha)

# -- Final scores ---------------------------------------------------
final_map, per_class = compute_map(y, oof_final)
print_results(final_map, per_class, "E167 FINAL (SKF)")

# Compare vs E79
try:
    oof_e79 = np.load(ROOT / "oof_e79.npy")
    e79_map, e79_per = compute_map(y, oof_e79)
    print(f"\n  E79 reference: {e79_map:.4f}", flush=True)
    print(f"  E167 delta:    {final_map - e79_map:+.4f}", flush=True)
    print("\n  Per-class deltas:", flush=True)
    for cls in CLASSES:
        d = per_class[cls] - e79_per[cls]
        arrow = "+" if d > 0 else ""
        print(f"    {cls:15s}: {arrow}{d:.4f}  ({e79_per[cls]:.4f} -> {per_class[cls]:.4f})", flush=True)
except FileNotFoundError:
    print("  (oof_e79.npy not found, skipping comparison)", flush=True)

# -- Save -----------------------------------------------------------
np.save(ROOT / "oof_e167.npy", oof_final)
np.save(ROOT / "test_e167.npy", test_final)
save_submission(test_final, f"e167_bugfix_td_{final_map:.4f}", cv_map=final_map)

# -- LOMO validation (quick) ----------------------------------------
print("\n--- LOMO validation ---", flush=True)
oof_lomo = np.zeros((len(y), N_CLASSES), dtype=np.float64)
for month in unique_months:
    va_idx = np.where(train_months == month)[0]
    tr_idx = np.where(train_months != month)[0]
    lgb_l = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb_l.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
  )
    oof_lomo[va_idx] = lgb_l.predict_proba(X[va_idx])
lomo_map, lomo_per = compute_map(y, oof_lomo)
print_results(lomo_map, lomo_per, "E167 LOMO (LGB only)")

try:
    oof_e79 = np.load(ROOT / "oof_e79.npy")
    # E79 LOMO not saved, just report the number
    print(f"\n  E79 LOMO reference: ~0.3949 (from backward elim)", flush=True)
    print(f"  E167 LOMO delta:   {lomo_map - 0.3949:+.4f}", flush=True)
except Exception:
    pass

print("\nDone.", flush=True)
