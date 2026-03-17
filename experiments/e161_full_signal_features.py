"""E161: E79 retrain with corrected preprocessing + full radar signal features.

E79 used 36 backward-eliminated features, but:
  1. Bearing was distorted (arctan2 without lat scaling) -- now fixed
  2. New feature extractors added but never tested: trajectory_separators,
     rcs_slope, radar_physics, flight_physics, enhanced_bio_shape, wingbeat, linearity
  3. The old 36 features were selected on broken preprocessing

This experiment:
  - Uses ALL feature extractors to capture the full radar signal shape
  - Removes temporal features (proven to overfit)
  - Same E79 HPs + ensemble approach (no Optuna -- avoid E160's overfit)
  - Keeps the 36 originals + adds ~70 new signal features (~106 total)
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

SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# ALL feature sets -- full radar signal characterization
ALL_FEAT_SETS = [
    "core",              # basic trajectory stats
    "rcs_fft",           # RCS frequency domain
    "tabular",           # extended stats
    "targeted",          # targeted features
    "flight_mode",       # flap/glide, oscillation, curvature
    "weakclass",         # RCS dynamics, soaring, turn angles
    "rcs_slope",         # RCS linear trend
    "trajectory_separators",  # heading_R, spectral entropy, speed autocorr, alt fracs
    "radar_physics",     # gap analysis, wing loading, power proxy
    "flight_physics",    # cross-correlations, bounding, thermal, periodicity
    "enhanced_bio_shape",  # turn consistency, RCS flap regularity
    "absolute_wingbeat", # wingbeat frequency bands
    "linearity",         # trajectory linearity
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
print("E161 FULL SIGNAL FEATURES".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data and build features -----------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

print("\nBuilding features (all extractors)...", flush=True)
train_feats = build_features(train_df, feature_sets=ALL_FEAT_SETS)
test_feats = build_features(test_df, feature_sets=ALL_FEAT_SETS)
print(f"  Raw features: {train_feats.shape[1]}", flush=True)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
print(f"  After temporal removal: {train_feats.shape[1]}", flush=True)

# Add weather + solar
train_feats, test_feats = add_weather_solar(train_feats, test_feats)
print(f"  With weather+solar: {train_feats.shape[1]}", flush=True)

# Print which features are new vs E79's 36
e79_features = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]
all_features = list(train_feats.columns)
new_features = [f for f in all_features if f not in e79_features]
kept_from_e79 = [f for f in e79_features if f in all_features]
print(f"  E79 features kept: {len(kept_from_e79)}/36", flush=True)
print(f"  New features: {len(new_features)}", flush=True)
print(f"  Total: {len(all_features)}", flush=True)

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# -- Effective number class weights (beta=0.999, from E15) ----------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- SKF CV with LGB + XGB + CB (E79 HPs exactly) ------------------
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

    # LightGBM (E79 HPs)
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGBoost (E79 HPs)
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

    # CatBoost (E79 default HPs, no Optuna)
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
        border_count=128, loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

# Individual model scores
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# -- Weight optimization on SKF OOF --------------------------------
print("\n--- Ensemble weight optimization ---", flush=True)
best_w = None
best_ens_map = -1.0
for w_lgb in np.arange(0.0, 1.05, 0.05):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.05):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01:
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
print_results(base_map, base_per, label="E161 ensemble (SKF OOF)")

# -- Feature importance (top 30 from LGB) --------------------------
print("\n--- Top 30 feature importances (LGB last fold) ---", flush=True)
imp = lgb.feature_importances_
feat_names = all_features
order = np.argsort(imp)[::-1]
for i, idx in enumerate(order[:30]):
    marker = " [NEW]" if feat_names[idx] not in e79_features else ""
    print(f"  {i+1:3d}. {feat_names[idx]:>35s}: {imp[idx]:5d}{marker}", flush=True)

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
print_results(best_map, best_per, label="E161 final (SKF OOF)")

# -- Compare to E79 ------------------------------------------------
print(f"\n  E79 reference:  SKF 0.7736 (36 feats)", flush=True)
print(f"  E161 result:    SKF {best_map:.4f} ({len(all_features)} feats)", flush=True)
print(f"  Delta:          {best_map - 0.7736:+.4f}", flush=True)

# -- Save artifacts ------------------------------------------------
print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e161.npy", best_oof)
np.save(ROOT / "test_e161.npy", best_test)
save_submission(best_test, "e161_full_signal", cv_map=best_map)

print("\nDone.", flush=True)
