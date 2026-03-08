"""E157: MultiRocket temporal features + LGB.

Test MultiRocket (from aeon) on properly preprocessed 6-channel radar sequences,
fed into LGB instead of LogReg. Previous E13c got 0.47 with broken preprocessing + LogReg.

Variants:
  A: MultiRocket features only + LGB
  B: MultiRocket features + 36 tabular features + LGB
  C: If B improves, LGB + XGB ensemble on combined features
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.sequence import prepare_sequences_v2
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

print("=" * 70, flush=True)
print("E157 MULTIROCKET + LGB".center(70), flush=True)
print("=" * 70, flush=True)

# ── Load data ─────────────────────────────────────────────────────────
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# ── Prepare sequences (v2, resample mode) ─────────────────────────────
print("\nPreparing sequences (v2, resample, max_len=200)...", flush=True)
train_seq, train_mask, train_lengths = prepare_sequences_v2(
    train_df, mode="resample", max_len=200, resample_rate=1.0,
)
test_seq, test_mask, test_lengths = prepare_sequences_v2(
    test_df, mode="resample", max_len=200, resample_rate=1.0,
)
print(f"  Train sequences: {train_seq.shape}", flush=True)
print(f"  Test sequences:  {test_seq.shape}", flush=True)
print(f"  Train lengths: min={train_lengths.min()}, median={int(np.median(train_lengths))}, max={train_lengths.max()}", flush=True)

# ── Extract MultiRocket features ──────────────────────────────────────
print("\nExtracting MultiRocket features...", flush=True)
from aeon.transformations.collection.convolution_based import MultiRocket

# Use reduced kernels to fit in memory (6250 default -> ~50k features, too much)
# 2000 kernels -> ~16k features, manageable
N_KERNELS = 2000
mr = MultiRocket(n_kernels=N_KERNELS, random_state=SEED, n_jobs=-1)
print(f"  Using {N_KERNELS} kernels", flush=True)

# aeon expects (n_samples, n_channels, n_timepoints) -- matches our v2 output
mr.fit(train_seq, y)
X_mr_train = mr.transform(train_seq)
X_mr_test = mr.transform(test_seq)

# Convert to numpy if needed
if hasattr(X_mr_train, 'values'):
    X_mr_train = X_mr_train.values
if hasattr(X_mr_test, 'values'):
    X_mr_test = X_mr_test.values

X_mr_train = np.asarray(X_mr_train, dtype=np.float32)
X_mr_test = np.asarray(X_mr_test, dtype=np.float32)

print(f"  MultiRocket features: {X_mr_train.shape[1]}", flush=True)

# Handle inf/nan
X_mr_train = np.where(np.isfinite(X_mr_train), X_mr_train, 0.0)
X_mr_test = np.where(np.isfinite(X_mr_test), X_mr_test, 0.0)

# ── Build tabular features (36 pruned) ────────────────────────────────
print("\nBuilding tabular features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
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

# Prune to 36 validated features
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
missing = [f for f in KEEP_FEATURES if f not in train_feats.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing: {missing}", flush=True)
print(f"  Using {len(available)}/{len(KEEP_FEATURES)} tabular features", flush=True)

train_feats = train_feats[available]
test_feats = test_feats[available]

X_tab_train = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_tab_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# ── Combine features for variant B ────────────────────────────────────
X_combined_train = np.hstack([X_mr_train, X_tab_train])
X_combined_test = np.hstack([X_mr_test, X_tab_test])
print(f"\n  MultiRocket-only features: {X_mr_train.shape[1]}", flush=True)
print(f"  Tabular features: {X_tab_train.shape[1]}", flush=True)
print(f"  Combined features: {X_combined_train.shape[1]}", flush=True)

# ── Effective number class weights ────────────────────────────────────
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# ── SKF CV ────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

results = {}


def train_lgb_cv(X_train, X_test_full, tag, extra_params=None):
    """Train LGB with 5-fold SKF CV and return OOF + test preds."""
    print(f"\n--- Variant {tag} ---", flush=True)
    print(f"  Features: {X_train.shape[1]}", flush=True)

    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_preds = np.zeros((len(X_test_full), N_CLASSES), dtype=np.float64)

    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=0.01,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
        device="gpu",
        n_jobs=-1,
    )
    if extra_params:
        params.update(extra_params)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)
        lgb = LGBMClassifier(**params)
        lgb.fit(
            X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])],
        )
        oof[va_idx] = lgb.predict_proba(X_train[va_idx])
        test_preds += lgb.predict_proba(X_test_full) / N_FOLDS

    m, per = compute_map(y, oof)
    print_results(m, per, label=f"E157 {tag}")
    results[tag] = {"map": m, "per_class": per, "oof": oof, "test": test_preds}
    return oof, test_preds, m


# ── Variant A: MultiRocket only ───────────────────────────────────────
oof_a, test_a, map_a = train_lgb_cv(X_mr_train, X_mr_test, "A: MultiRocket only")

# ── Variant B: MultiRocket + Tabular ─────────────────────────────────
oof_b, test_b, map_b = train_lgb_cv(X_combined_train, X_combined_test, "B: MultiRocket + Tabular")

# ── Variant C: LGB + XGB ensemble on combined (only if B improves) ───
if map_b > map_a:
    print("\n--- Variant C: LGB + XGB ensemble on combined features ---", flush=True)

    oof_lgb_c = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb_c = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb_c = np.zeros((len(X_combined_test), N_CLASSES), dtype=np.float64)
    test_xgb_c = np.zeros((len(X_combined_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_combined_train, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        # LGB
        lgb = LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(
            X_combined_train[tr_idx], y[tr_idx],
            eval_set=[(X_combined_train[va_idx], y[va_idx])],
        )
        oof_lgb_c[va_idx] = lgb.predict_proba(X_combined_train[va_idx])
        test_lgb_c += lgb.predict_proba(X_combined_test) / N_FOLDS

        # XGB
        xgb = XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb.fit(
            X_combined_train[tr_idx], y[tr_idx],
            eval_set=[(X_combined_train[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False,
        )
        oof_xgb_c[va_idx] = xgb.predict_proba(X_combined_train[va_idx])
        test_xgb_c += xgb.predict_proba(X_combined_test) / N_FOLDS

    # Optimize ensemble weights
    best_w_c = None
    best_map_c = -1.0
    for w_lgb in np.arange(0.0, 1.05, 0.05):
        w_xgb = 1.0 - w_lgb
        oof_ens = w_lgb * oof_lgb_c + w_xgb * oof_xgb_c
        m, _ = compute_map(y, oof_ens)
        if m > best_map_c:
            best_map_c = m
            best_w_c = (w_lgb, w_xgb)

    w_lgb_c, w_xgb_c = best_w_c
    oof_c = w_lgb_c * oof_lgb_c + w_xgb_c * oof_xgb_c
    test_c = w_lgb_c * test_lgb_c + w_xgb_c * test_xgb_c
    _, per_c = compute_map(y, oof_c)

    print(f"  Best weights: LGB={w_lgb_c:.2f} XGB={w_xgb_c:.2f}", flush=True)
    print_results(best_map_c, per_c, label="E157 C: LGB+XGB ensemble (combined)")
    results["C: LGB+XGB ensemble"] = {"map": best_map_c, "per_class": per_c, "oof": oof_c, "test": test_c}
else:
    print("\n--- Skipping variant C (B did not improve over A) ---", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  E79 reference:         SKF 0.7736", flush=True)
for tag, r in results.items():
    print(f"  E157 {tag}: SKF {r['map']:.4f}", flush=True)

# ── Save best variant ─────────────────────────────────────────────────
best_tag = max(results, key=lambda k: results[k]["map"])
best_r = results[best_tag]
print(f"\nBest variant: {best_tag} (SKF {best_r['map']:.4f})", flush=True)

np.save(ROOT / "oof_e157.npy", best_r["oof"])
np.save(ROOT / "test_e157.npy", best_r["test"])
save_submission(best_r["test"], f"e157_multirocket_lgb", cv_map=best_r["map"])

print("\nDone.", flush=True)
