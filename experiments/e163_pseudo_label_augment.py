"""E163: Pseudo-labeling + rare-class augmentation.

Strategy:
  A) Baseline: E162b config (69 physics + 15 wx/solar = 84 features)
  B) + Pseudo-labels: Add E79's confident test predictions (>0.80) as extra training
  C) + Augmentation: Augment rare classes (Cormorants=40, Ducks=58, Clutter=84) to 150
  D) + Both: Pseudo-labels + augmentation combined

Pseudo-labeling adds real unseen-month data (Feb/May/Dec) that training lacks entirely.
Augmentation adds diversity to rare classes via physical transforms (rotation, speed, RCS noise).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.augmentation import augment_rare_classes

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# E162b feature config: physics + raw signal features (no explicit temporal)
FEAT_SETS = [
    "core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass",
    "rcs_slope", "trajectory_separators", "radar_physics", "flight_physics",
    "enhanced_bio_shape", "linearity", "raw_signal",
]

PHYSICS_FEATURES = [
    f.strip() for f in (ROOT / "data" / "physics_features.txt").read_text().splitlines()
    if f.strip()
]

# Pseudo-label config
PL_CONFIDENCE = 0.80  # only use predictions with >80% confidence
PL_PRED_FILE = ROOT / "test_e79.npy"  # E79 test predictions

# Augmentation config
AUG_TARGET = 150  # augment rare classes to this count


def load_weather_solar(split, ext_dir=None):
    """Load weather + solar features."""
    if ext_dir is None:
        ext_dir = ROOT / "data"
    weather = pd.read_csv(ext_dir / f"{split}_weather.csv")
    solar = pd.read_csv(ext_dir / f"{split}_solar.csv")
    return weather, solar


def prepare_features(df, feat_sets=FEAT_SETS, physics_cols=PHYSICS_FEATURES):
    """Build features, remove temporal, add weather/solar, keep physics list."""
    feats = build_features(df, feature_sets=feat_sets)
    keep = [c for c in feats.columns if c not in ALL_TEMPORAL]
    feats = feats[keep]
    return feats


def add_weather_solar(feats, weather, solar, prefix_wx="wx_", prefix_sol="sol_"):
    """Add weather and solar columns to feature DataFrame."""
    for col in weather.columns:
        feats[f"{prefix_wx}{col}"] = weather[col].values
    for col in solar.columns:
        feats[f"{prefix_sol}{col}"] = solar[col].values
    return feats


def select_features(feats, physics_cols):
    """Keep only physics features that exist in the DataFrame."""
    available = [f for f in physics_cols if f in feats.columns]
    return feats[available], available


def to_matrix(feats):
    """Convert DataFrame to clean float32 matrix."""
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)


def train_ensemble(X, y, X_test, sample_weights, tag=""):
    """Train LGB+XGB+CB ensemble with weight optimization."""
    print(f"\n--- {tag} ---", flush=True)
    print(f"  Train: {X.shape[0]} x {X.shape[1]}, Test: {X_test.shape[0]}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
        test_lgb += lgb.predict_proba(X_test) / N_FOLDS

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

    # Weight optimization
    best_w, best_map = None, -1.0
    for w_lgb in np.arange(0.0, 1.05, 0.05):
        for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.05):
            w_cb = 1.0 - w_lgb - w_xgb
            if w_cb < -0.01:
                continue
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w_lgb, w_xgb, w_cb)

    w_lgb, w_xgb, w_cb = best_w
    oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
    test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

    print(f"  Weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
    m, per = compute_map(y, oof_ens)
    print_results(m, per, label=tag)
    return oof_ens, test_ens, m, per


def make_sample_weights(y_arr, n_classes):
    """Effective number class weights (beta=0.999)."""
    counts = np.bincount(y_arr, minlength=n_classes).astype(float)
    beta = 0.999
    eff_n = (1.0 - beta ** counts) / (1.0 - beta)
    cw = 1.0 / np.maximum(eff_n, 1e-6)
    cw /= cw.sum() / n_classes
    return cw[y_arr]


# ====================================================================
print("=" * 70, flush=True)
print("E163 PSEUDO-LABEL + AUGMENTATION".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data -------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# -- Build features --------------------------------------------------
print("\nBuilding features...", flush=True)
train_feats_raw = prepare_features(train_df)
test_feats_raw = prepare_features(test_df)

train_weather, train_solar = load_weather_solar("train")
test_weather, test_solar = load_weather_solar("test")

train_feats_full = add_weather_solar(train_feats_raw.copy(), train_weather, train_solar)
test_feats_full = add_weather_solar(test_feats_raw.copy(), test_weather, test_solar)

train_feats, feat_names = select_features(train_feats_full, PHYSICS_FEATURES)
test_feats, _ = select_features(test_feats_full, PHYSICS_FEATURES)
print(f"  Features: {len(feat_names)}", flush=True)

X_train = to_matrix(train_feats)
X_test = to_matrix(test_feats)

# =====================================================================
# VARIANT A: Baseline (E162b reproduction)
# =====================================================================
sw = make_sample_weights(y, N_CLASSES)
oof_a, test_a, map_a, per_a = train_ensemble(X_train, y, X_test, sw, tag="A: Baseline (E162b)")

# =====================================================================
# VARIANT B: + Pseudo-labels from E79
# =====================================================================
print("\n\n--- Preparing pseudo-labels ---", flush=True)
test_preds = np.load(PL_PRED_FILE)
test_conf = test_preds.max(axis=1)
test_labels = test_preds.argmax(axis=1)
pl_mask = test_conf >= PL_CONFIDENCE

print(f"  E79 test predictions: {len(test_preds)}", flush=True)
print(f"  Confident (>={PL_CONFIDENCE}): {pl_mask.sum()} ({pl_mask.mean()*100:.1f}%)", flush=True)
for cidx, cls in enumerate(CLASSES):
    n = ((test_labels == cidx) & pl_mask).sum()
    if n > 0:
        print(f"    {cls}: {n}", flush=True)

# Extract test months for context
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month
for m in sorted(test_months.unique()):
    m_mask = (test_months == m).values & pl_mask
    n = m_mask.sum()
    print(f"    Month {m:2d}: {n} pseudo-labels", flush=True)

# Build features for pseudo-labeled test samples
pl_test_df = test_df[pl_mask].reset_index(drop=True)
pl_y = test_labels[pl_mask]

if len(pl_test_df) > 0:
    pl_feats_raw = prepare_features(pl_test_df)
    # Weather/solar for pseudo-labeled rows (subset of test)
    pl_weather = test_weather[pl_mask].reset_index(drop=True)
    pl_solar = test_solar[pl_mask].reset_index(drop=True)
    pl_feats_full = add_weather_solar(pl_feats_raw.copy(), pl_weather, pl_solar)
    pl_feats, _ = select_features(pl_feats_full, PHYSICS_FEATURES)
    X_pl = to_matrix(pl_feats)

    # Combine train + pseudo-labels
    X_train_pl = np.vstack([X_train, X_pl])
    y_pl = np.concatenate([y, pl_y])
    sw_pl = make_sample_weights(y_pl, N_CLASSES)

    print(f"\n  Combined: {len(y_pl)} samples ({len(y)} train + {len(pl_y)} pseudo)", flush=True)

    # For OOF evaluation, we only evaluate on original train samples
    # So we train on all data but report OOF only for original indices
    oof_b, test_b, map_b, per_b = train_ensemble(
        X_train_pl, y_pl, X_test, sw_pl, tag="B: + Pseudo-labels"
    )
    # Extract original train OOF for fair comparison
    oof_b_orig = oof_b[:len(y)]
    map_b_orig, per_b_orig = compute_map(y, oof_b_orig)
    print(f"  B original-train-only mAP: {map_b_orig:.4f}", flush=True)
    print_results(map_b_orig, per_b_orig, label="B: orig-train OOF")
else:
    print("  No pseudo-labels available!", flush=True)
    oof_b, test_b, map_b = oof_a, test_a, map_a
    oof_b_orig = oof_a
    map_b_orig = map_a

# =====================================================================
# VARIANT C: + Rare class augmentation (no pseudo-labels)
# =====================================================================
print("\n\n--- Preparing augmented data ---", flush=True)
aug_df, aug_y = augment_rare_classes(train_df, y, CLASSES, target_count=AUG_TARGET, seed=SEED)
print(f"  Augmented: {len(aug_y)} new samples", flush=True)
for cidx, cls in enumerate(CLASSES):
    n_orig = (y == cidx).sum()
    n_aug = (aug_y == cidx).sum()
    if n_aug > 0:
        print(f"    {cls}: {n_orig} -> {n_orig + n_aug} (+{n_aug})", flush=True)

if len(aug_df) > 0:
    # Build features for augmented samples
    aug_feats_raw = prepare_features(aug_df)

    # For augmented samples, we duplicate the weather/solar from their source rows
    # Since augmentation picks random rows, we need to build weather/solar for them
    # Quick approach: use the mean weather/solar values (augmented = synthetic location)
    aug_weather = pd.DataFrame(
        np.tile(train_weather.mean().values, (len(aug_df), 1)),
        columns=train_weather.columns
    )
    aug_solar = pd.DataFrame(
        np.tile(train_solar.mean().values, (len(aug_df), 1)),
        columns=train_solar.columns
    )
    aug_feats_full = add_weather_solar(aug_feats_raw.copy(), aug_weather, aug_solar)
    aug_feats, _ = select_features(aug_feats_full, PHYSICS_FEATURES)
    X_aug = to_matrix(aug_feats)

    X_train_aug = np.vstack([X_train, X_aug])
    y_aug = np.concatenate([y, aug_y])
    sw_aug = make_sample_weights(y_aug, N_CLASSES)

    print(f"\n  Combined: {len(y_aug)} samples ({len(y)} train + {len(aug_y)} augmented)", flush=True)

    oof_c, test_c, map_c, per_c = train_ensemble(
        X_train_aug, y_aug, X_test, sw_aug, tag="C: + Augmentation"
    )
    oof_c_orig = oof_c[:len(y)]
    map_c_orig, per_c_orig = compute_map(y, oof_c_orig)
    print(f"  C original-train-only mAP: {map_c_orig:.4f}", flush=True)
    print_results(map_c_orig, per_c_orig, label="C: orig-train OOF")
else:
    print("  No augmentation needed!", flush=True)
    oof_c, test_c, map_c = oof_a, test_a, map_a
    oof_c_orig = oof_a
    map_c_orig = map_a

# =====================================================================
# VARIANT D: + Both pseudo-labels and augmentation
# =====================================================================
if len(pl_test_df) > 0 and len(aug_df) > 0:
    X_train_both = np.vstack([X_train, X_pl, X_aug])
    y_both = np.concatenate([y, pl_y, aug_y])
    sw_both = make_sample_weights(y_both, N_CLASSES)

    print(f"\n  Combined: {len(y_both)} samples ({len(y)} + {len(pl_y)} PL + {len(aug_y)} aug)", flush=True)

    oof_d, test_d, map_d, per_d = train_ensemble(
        X_train_both, y_both, X_test, sw_both, tag="D: PL + Augmentation"
    )
    oof_d_orig = oof_d[:len(y)]
    map_d_orig, per_d_orig = compute_map(y, oof_d_orig)
    print(f"  D original-train-only mAP: {map_d_orig:.4f}", flush=True)
    print_results(map_d_orig, per_d_orig, label="D: orig-train OOF")
else:
    oof_d, test_d = oof_a, test_a
    map_d_orig = map_a

# =====================================================================
# Summary + save best
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  E79 reference:   SKF 0.7736, LB 0.59", flush=True)
print(f"  E162b reference: SKF 0.7753, LB 0.54", flush=True)
print(f"  A: Baseline:     SKF {map_a:.4f}", flush=True)
print(f"  B: + PL:         SKF {map_b_orig:.4f} (full={map_b:.4f})", flush=True)
print(f"  C: + Aug:        SKF {map_c_orig:.4f} (full={map_c:.4f})", flush=True)
print(f"  D: + PL + Aug:   SKF {map_d_orig:.4f}", flush=True)

# Pick best by original-train OOF
variants = {
    "A": (map_a, oof_a, test_a),
    "B": (map_b_orig, oof_b_orig, test_b),
    "C": (map_c_orig, oof_c_orig, test_c),
    "D": (map_d_orig, oof_d[:len(y)] if len(pl_test_df) > 0 and len(aug_df) > 0 else oof_a,
          test_d),
}
best_var = max(variants, key=lambda k: variants[k][0])
best_map, best_oof, best_test = variants[best_var]
print(f"\n  Best: {best_var} (SKF {best_map:.4f})", flush=True)

# Save
np.save(ROOT / "oof_e163.npy", best_oof)
np.save(ROOT / "test_e163.npy", best_test)
save_submission(best_test, f"e163_{best_var}_pl_aug", cv_map=best_map)

# Also save all test predictions for comparison
for var, (m, _, t) in variants.items():
    save_submission(t, f"e163_{var}", cv_map=m)

print("\nDone.", flush=True)
