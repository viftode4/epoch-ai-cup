"""E168: Stacking — E79 Base Predictions + External Features Meta-Learner.

Key insight from E166: adding external features to the tree model HURTS LOMO.
Key insight from E167: NB PoE uses external data but with fixed Gaussian assumptions.

This experiment uses STACKING: a lightweight meta-learner that takes E79's
OOF predictions (9 class probabilities) + selected external features as input
and learns to CORRECT the base predictions. The meta-learner can discover
non-linear interactions between base predictions and external evidence that
the NB PoE's Gaussian assumption misses.

Pipeline:
  1. Load E79 OOF predictions (9 cols) + external features (selected)
  2. Train a lightweight LGB meta-learner on LOMO
  3. LOMO evaluation to check if meta-learner generalizes
  4. If LOMO improves: train on SKF, save stacked predictions
  5. Apply PP on top of stacked predictions

The meta-learner is deliberately small (few estimators, high regularization)
to prevent overfitting to the base predictions.
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
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_FOLDS = 5


def load_external_csv(name, split):
    path = ROOT / "data" / f"{split}_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ======================================================================
# MAIN
# ======================================================================

print("=" * 70, flush=True)
print("E168: STACKING - E79 + EXTERNAL META-LEARNER".center(70), flush=True)
print("=" * 70, flush=True)

# Load data
train_df = load_train()
test_df = load_test()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# Check base predictions
oof_path = ROOT / "oof_e79.npy"
test_path = ROOT / "test_e79.npy"
if not oof_path.exists() or not test_path.exists():
    print("ERROR: E79 predictions not found", flush=True)
    sys.exit(1)

oof_base = np.load(oof_path).astype(float)
test_base = np.load(test_path).astype(float)
print(f"  E79 OOF: {oof_base.shape}, Test: {test_base.shape}", flush=True)

# Effective number class weights
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# ── Build stacking features ─────────────────────────────────────────
print("\n--- Building stacking features ---", flush=True)

# Base predictions as features
train_stack = pd.DataFrame(oof_base, columns=[f"pred_{cls}" for cls in CLASSES])
test_stack = pd.DataFrame(test_base, columns=[f"pred_{cls}" for cls in CLASSES])

# Derived prediction features
train_stack["pred_max"] = oof_base.max(axis=1)
train_stack["pred_entropy"] = -np.sum(
    np.clip(oof_base, 1e-8, 1) * np.log(np.clip(oof_base, 1e-8, 1)), axis=1
)
train_stack["pred_margin"] = top2_margin(oof_base)
train_stack["pred_top2_ratio"] = np.sort(oof_base, axis=1)[:, -1] / np.maximum(
    np.sort(oof_base, axis=1)[:, -2], 1e-8
)

test_stack["pred_max"] = test_base.max(axis=1)
test_stack["pred_entropy"] = -np.sum(
    np.clip(test_base, 1e-8, 1) * np.log(np.clip(test_base, 1e-8, 1)), axis=1
)
test_stack["pred_margin"] = top2_margin(test_base)
test_stack["pred_top2_ratio"] = np.sort(test_base, axis=1)[:, -1] / np.maximum(
    np.sort(test_base, axis=1)[:, -2], 1e-8
)

# Select high-value external features (month-invariant spatial + tidal)
external_features = {
    "tidal": ["hours_since_high_tide", "tidal_phase"],
    "water": ["dist_to_water_m"],
    "visibility": ["rain_occurring"],
    "altitude_winds": ["boundary_layer_height", "wind_at_bird_alt"],
    "landuse": ["dist_to_grassland_m"],
    "turbines": ["dist_to_turbine_m"],
    "marine": ["sea_surface_temperature"],
    "cape": ["cape_jkg", "lifted_index"],
    "pressure": ["pressure_trend_3h"],
    "natura2000": ["dist_to_natura2000_m"],
}

for csv_name, cols in external_features.items():
    tr_ext = load_external_csv(csv_name, "train")
    te_ext = load_external_csv(csv_name, "test")
    for col in cols:
        if not tr_ext.empty and col in tr_ext.columns:
            train_stack[f"ext_{col}"] = pd.to_numeric(
                tr_ext[col], errors="coerce"
            ).fillna(0).values
        if not te_ext.empty and col in te_ext.columns:
            test_stack[f"ext_{col}"] = pd.to_numeric(
                te_ext[col], errors="coerce"
            ).fillna(0).values

# Add raw tabular features that the meta-learner can interact with predictions
train_stack["airspeed"] = pd.to_numeric(train_df["airspeed"], errors="coerce").fillna(0).values
train_stack["min_z"] = pd.to_numeric(train_df["min_z"], errors="coerce").fillna(0).values
train_stack["max_z"] = pd.to_numeric(train_df["max_z"], errors="coerce").fillna(0).values

test_stack["airspeed"] = pd.to_numeric(test_df["airspeed"], errors="coerce").fillna(0).values
test_stack["min_z"] = pd.to_numeric(test_df["min_z"], errors="coerce").fillna(0).values
test_stack["max_z"] = pd.to_numeric(test_df["max_z"], errors="coerce").fillna(0).values

# Size encoding
from src.features import SIZE_MAP
train_stack["radar_bird_size"] = train_df["radar_bird_size"].map(SIZE_MAP).fillna(2).values
test_stack["radar_bird_size"] = test_df["radar_bird_size"].map(SIZE_MAP).fillna(2).values

# Ensure same columns
common_cols = sorted(set(train_stack.columns) & set(test_stack.columns))
train_stack = train_stack[common_cols]
test_stack = test_stack[common_cols]

X_stack = train_stack.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test_stack = test_stack.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
feature_names = list(train_stack.columns)
print(f"  Stacking features: {X_stack.shape[1]}", flush=True)
print(f"  Feature list: {feature_names}", flush=True)

# ── LOMO evaluation of stacking ─────────────────────────────────────
print("\n--- LOMO Evaluation ---", flush=True)

from lightgbm import LGBMClassifier

unique_months = sorted(np.unique(train_months))

# Test multiple meta-learner configs (small = prevents overfitting to base OOF)
configs = [
    {"n_estimators": 200, "num_leaves": 15, "max_depth": 3, "reg_lambda": 10.0, "label": "tiny"},
    {"n_estimators": 300, "num_leaves": 31, "max_depth": 4, "reg_lambda": 5.0, "label": "small"},
    {"n_estimators": 500, "num_leaves": 31, "max_depth": 5, "reg_lambda": 2.0, "label": "medium"},
    {"n_estimators": 800, "num_leaves": 63, "max_depth": 6, "reg_lambda": 1.0, "label": "large"},
]

# E79 baseline LOMO for comparison
oof_e79_lomo = np.zeros((len(y), N_CLASSES))
for month in unique_months:
    va = train_months == month
    oof_e79_lomo[va] = oof_base[va]  # E79 is SKF, not LOMO, so we use it as-is
e79_map, _ = compute_map(y, oof_base)
print(f"  E79 SKF mAP: {e79_map:.4f} (reference)", flush=True)

best_config = None
best_lomo = -1.0

for cfg in configs:
    label = cfg.pop("label")
    oof_meta = np.zeros((len(y), N_CLASSES), dtype=np.float64)

    for month in unique_months:
        va = train_months == month
        tr = ~va
        lgb = LGBMClassifier(
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=SEED, verbose=-1,
            device="gpu", n_jobs=-1, **cfg,
        )
        lgb.fit(X_stack[tr], y[tr], sample_weight=sample_weights[tr])
        oof_meta[va] = lgb.predict_proba(X_stack[va])

    lomo_map, per_lomo = compute_map(y, oof_meta)
    tag = " *** BEST" if lomo_map > best_lomo else ""
    print(f"  {label:8s}: LOMO mAP = {lomo_map:.4f}{tag}", flush=True)

    if lomo_map > best_lomo:
        best_lomo = lomo_map
        best_config = dict(cfg)
        best_config["label"] = label

    cfg["label"] = label  # restore

print(f"\n  Best meta-learner: {best_config['label']} -> LOMO mAP: {best_lomo:.4f}", flush=True)

# ── SKF training with best config ────────────────────────────────────
print("\n--- SKF Training ---", flush=True)

from sklearn.model_selection import StratifiedKFold

label = best_config.pop("label")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_stacked = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_stacked = np.zeros((len(X_test_stack), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_stack, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}", flush=True)
    lgb = LGBMClassifier(
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED, verbose=-1,
        device="gpu", n_jobs=-1, **best_config,
    )
    lgb.fit(X_stack[tr_idx], y[tr_idx],
            eval_set=[(X_stack[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx])
    oof_stacked[va_idx] = lgb.predict_proba(X_stack[va_idx])
    test_stacked += lgb.predict_proba(X_test_stack) / N_FOLDS

    # Feature importance (last fold only)
    if fold_i == N_FOLDS - 1:
        imp = lgb.feature_importances_
        top_idx = np.argsort(-imp)[:20]
        print("\n  Top 20 features by importance:", flush=True)
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1:3d}. {feature_names[idx]:35s} imp={imp[idx]}", flush=True)

skf_map, skf_per = compute_map(y, oof_stacked)
print_results(skf_map, skf_per, label="E168 Stacked (SKF OOF)")
print(f"  vs E79 SKF mAP: {e79_map:.4f} -> {skf_map:.4f} (delta: {skf_map - e79_map:+.4f})", flush=True)

# ── Ensemble: blend base + stacked ──────────────────────────────────
print("\n--- Blend E79 + Stacked ---", flush=True)

best_blend_map = -1.0
best_alpha = 0.0
for alpha in np.arange(0.0, 1.05, 0.05):
    oof_blend = (1 - alpha) * oof_base + alpha * oof_stacked
    m, _ = compute_map(y, oof_blend)
    if m > best_blend_map:
        best_blend_map = m
        best_alpha = alpha

print(f"  Best blend: alpha={best_alpha:.2f} (E79:{1-best_alpha:.2f} + Stack:{best_alpha:.2f})", flush=True)
print(f"  Blended SKF mAP: {best_blend_map:.4f}", flush=True)

oof_final = (1 - best_alpha) * oof_base + best_alpha * oof_stacked
test_final = (1 - best_alpha) * test_base + best_alpha * test_stacked

# ── Save artifacts ──────────────────────────────────────────────────
print("\n--- Saving ---", flush=True)
np.save(ROOT / "oof_e168.npy", oof_final)
np.save(ROOT / "test_e168.npy", test_final)
save_submission(test_final, "e168_stacked_raw", cv_map=best_blend_map)

# ── Apply standard PP on stacked predictions ────────────────────────
print("\n--- Post-Processing on Stacked ---", flush=True)

p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)
test_pp, n_ch = apply_gated_ratio_priors(
    test_final.copy(), test_months, p_train, priors, BASE_ALPHA, tau=0.15
)

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
gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < 0.25)
test_pp_final = apply_nb_poe(test_pp, ll, gamma=0.10, gate=gate)

save_submission(test_pp_final, "e168_stacked_pp", cv_map=best_blend_map)

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("E168 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  Meta-learner: {label}", flush=True)
print(f"  Stacking features: {X_stack.shape[1]}", flush=True)
print(f"  LOMO mAP (stacked): {best_lomo:.4f}", flush=True)
print(f"  SKF mAP (stacked):  {skf_map:.4f}", flush=True)
print(f"  SKF mAP (E79):      {e79_map:.4f}", flush=True)
print(f"  Blend alpha:         {best_alpha:.2f}", flush=True)
print(f"  Blend SKF mAP:       {best_blend_map:.4f}", flush=True)
print("=" * 70, flush=True)
print("\nDone.", flush=True)
