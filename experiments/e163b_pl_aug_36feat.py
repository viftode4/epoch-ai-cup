"""E163b: Pseudo-labeling + augmentation on E79's proven 36-feature set.

Key insight: E79's 36 features generalize to LB 0.59 while E162b's 69 features
only reach LB 0.54 despite higher SKF. So we use E79's features but add:
  - Pseudo-labels from E79 test predictions (confident >0.80 threshold)
  - Rare class augmentation (trajectory rotation/speed/RCS transforms)

This should give the best of both worlds: generalizable features + more data.
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

# E79's exact feature config
FEAT_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

PL_CONFIDENCE = 0.80
PL_PRED_FILE = ROOT / "test_e79.npy"
AUG_TARGETS = [150, 200]  # try both


def build_e79_features(df):
    """Build E79-style features: 36 pruned + weather/solar."""
    feats = build_features(df, feature_sets=FEAT_SETS)
    keep = [c for c in feats.columns if c not in ALL_TEMPORAL]
    feats = feats[keep]
    return feats


def add_wx_solar(feats, weather, solar):
    for col in weather.columns:
        feats[f"wx_{col}"] = weather[col].values
    for col in solar.columns:
        feats[f"sol_{col}"] = solar[col].values
    return feats


def finalize(feats):
    available = [f for f in KEEP_FEATURES if f in feats.columns]
    return feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32), available


def make_weights(y_arr):
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    beta = 0.999
    eff_n = (1.0 - beta ** counts) / (1.0 - beta)
    cw = 1.0 / np.maximum(eff_n, 1e-6)
    cw /= cw.sum() / N_CLASSES
    return cw[y_arr]


def train_ensemble_skf(X, y, X_test, sw, tag, eval_n=None):
    """Train LGB+XGB+CB. eval_n = number of original samples for OOF eval."""
    print(f"\n--- {tag} ---", flush=True)
    print(f"  Train: {X.shape[0]} x {X.shape[1]}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb = np.zeros_like(oof_lgb)
    oof_cb = np.zeros_like(oof_lgb)
    t_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    t_xgb = np.zeros_like(t_lgb)
    t_cb = np.zeros_like(t_lgb)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}", flush=True)

        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu", n_jobs=-1,
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
        t_lgb += lgb.predict_proba(X_test) / N_FOLDS

        xgb = XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                sample_weight=sw[tr_idx], verbose=False)
        oof_xgb[va_idx] = xgb.predict_proba(X[va_idx])
        t_xgb += xgb.predict_proba(X_test) / N_FOLDS

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
            border_count=128, loss_function="MultiClass", eval_metric="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X[va_idx])
        t_cb += cb.predict_proba(X_test) / N_FOLDS

    # Weight optimization
    best_w, best_map = None, -1.0
    for w1 in np.arange(0.0, 1.05, 0.05):
        for w2 in np.arange(0.0, 1.05 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            m, _ = compute_map(y, oof)
            if m > best_map:
                best_map = m
                best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
    test_ens = w1 * t_lgb + w2 * t_xgb + w3 * t_cb
    print(f"  Weights: LGB={w1:.2f} XGB={w2:.2f} CB={w3:.2f}", flush=True)

    # Evaluate on original train only
    if eval_n is not None and eval_n < len(y):
        m_orig, per_orig = compute_map(y[:eval_n], oof_ens[:eval_n])
        print(f"  Original-train mAP: {m_orig:.4f}", flush=True)
        print_results(m_orig, per_orig, label=f"{tag} (orig)")
    else:
        m_orig = best_map

    m_full, per_full = compute_map(y, oof_ens)
    print_results(m_full, per_full, label=tag)
    return oof_ens, test_ens, m_orig, m_full


# ====================================================================
print("=" * 70, flush=True)
print("E163b: PL + AUG on E79's 36 FEATURES".center(70), flush=True)
print("=" * 70, flush=True)

# Load data
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
n_train = len(y)

# Build features
print("\nBuilding features (E79 config)...", flush=True)
train_feats = build_e79_features(train_df)
test_feats = build_e79_features(test_df)

train_wx = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_wx = pd.read_csv(ROOT / "data" / "test_weather.csv")
train_sol = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_sol = pd.read_csv(ROOT / "data" / "test_solar.csv")

train_feats = add_wx_solar(train_feats, train_wx, train_sol)
test_feats = add_wx_solar(test_feats, test_wx, test_sol)

X_train, feat_names = finalize(train_feats)
X_test, _ = finalize(test_feats)
print(f"  Features: {len(feat_names)}", flush=True)

# ---- A: Baseline (E79 reproduction) ----
sw = make_weights(y)
oof_a, test_a, _, map_a = train_ensemble_skf(X_train, y, X_test, sw, "A: E79 Baseline")

# ---- Prepare pseudo-labels ----
print("\n--- Pseudo-labels ---", flush=True)
test_preds = np.load(PL_PRED_FILE)
test_conf = test_preds.max(axis=1)
test_labels = test_preds.argmax(axis=1)
pl_mask = test_conf >= PL_CONFIDENCE
print(f"  Confident: {pl_mask.sum()}/{len(test_preds)}", flush=True)

pl_df = test_df[pl_mask].reset_index(drop=True)
pl_y = test_labels[pl_mask]

pl_feats = build_e79_features(pl_df)
pl_wx = test_wx[pl_mask].reset_index(drop=True)
pl_sol = test_sol[pl_mask].reset_index(drop=True)
pl_feats = add_wx_solar(pl_feats, pl_wx, pl_sol)
X_pl, _ = finalize(pl_feats)

# ---- Prepare augmentation ----
results = {}

for aug_target in AUG_TARGETS:
    print(f"\n{'='*70}", flush=True)
    print(f"  AUG_TARGET = {aug_target}", flush=True)

    aug_df, aug_y = augment_rare_classes(train_df, y, CLASSES, target_count=aug_target, seed=SEED)
    print(f"  Augmented: {len(aug_y)} new samples", flush=True)
    for cidx, cls in enumerate(CLASSES):
        n_aug = (aug_y == cidx).sum()
        if n_aug > 0:
            print(f"    {cls}: {(y==cidx).sum()} -> {(y==cidx).sum()+n_aug}", flush=True)

    aug_feats = build_e79_features(aug_df)
    # Use mean weather/solar for augmented (synthetic location/time)
    aug_wx_data = pd.DataFrame(
        np.tile(train_wx.mean().values, (len(aug_df), 1)), columns=train_wx.columns
    )
    aug_sol_data = pd.DataFrame(
        np.tile(train_sol.mean().values, (len(aug_df), 1)), columns=train_sol.columns
    )
    aug_feats = add_wx_solar(aug_feats, aug_wx_data, aug_sol_data)
    X_aug, _ = finalize(aug_feats)

    # B: PL only
    X_b = np.vstack([X_train, X_pl])
    y_b = np.concatenate([y, pl_y])
    sw_b = make_weights(y_b)
    oof_b, test_b, map_b_orig, map_b = train_ensemble_skf(
        X_b, y_b, X_test, sw_b, f"B: +PL (aug={aug_target})", eval_n=n_train
    )

    # C: Aug only
    X_c = np.vstack([X_train, X_aug])
    y_c = np.concatenate([y, aug_y])
    sw_c = make_weights(y_c)
    oof_c, test_c, map_c_orig, map_c = train_ensemble_skf(
        X_c, y_c, X_test, sw_c, f"C: +Aug({aug_target})", eval_n=n_train
    )

    # D: PL + Aug
    X_d = np.vstack([X_train, X_pl, X_aug])
    y_d = np.concatenate([y, pl_y, aug_y])
    sw_d = make_weights(y_d)
    oof_d, test_d, map_d_orig, map_d = train_ensemble_skf(
        X_d, y_d, X_test, sw_d, f"D: +PL+Aug({aug_target})", eval_n=n_train
    )

    results[aug_target] = {
        "B": (map_b_orig, test_b),
        "C": (map_c_orig, test_c),
        "D": (map_d_orig, test_d),
    }

# ---- Summary ----
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  E79 reference:  SKF 0.7736, LB 0.59", flush=True)
print(f"  A: Baseline:    SKF {map_a:.4f}", flush=True)
for aug_target, r in results.items():
    for var, (m, _) in r.items():
        print(f"  {var} (aug={aug_target}): SKF {m:.4f} (delta={m-map_a:+.4f})", flush=True)

# Save best
all_variants = {"A": (map_a, test_a)}
for aug_target, r in results.items():
    for var, (m, t) in r.items():
        all_variants[f"{var}_{aug_target}"] = (m, t)

best_key = max(all_variants, key=lambda k: all_variants[k][0])
best_map, best_test = all_variants[best_key]
print(f"\n  Best: {best_key} (SKF {best_map:.4f})", flush=True)

save_submission(best_test, f"e163b_{best_key}", cv_map=best_map)
np.save(ROOT / "test_e163b.npy", best_test)

print("\nDone.", flush=True)
