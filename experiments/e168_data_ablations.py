"""E168: Data-level ablations to break 0.59 LB ceiling.

Tests four independent hypotheses via LGB-only SKF (fast):
  A. Baseline (bug-fixed 36 features, E79 equivalent)
  B. Hour-filter: drop train tracks with hour >= 13 (test has 0 tracks after 12h)
  C. September-as-unseen: exclude Sep from shared months in LOMO
  D. Add speed_ratio_var feature (per-segment airspeed/ground variance)

Each ablation is independent. We compare SKF mAP and LOMO mAP.
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]


def add_weather_solar(train_feats, test_feats):
    """Add weather + solar features."""
    for prefix, fname in [("wx_", "weather"), ("sol_", "solar")]:
        tr = pd.read_csv(ROOT / "data" / f"train_{fname}.csv")
        te = pd.read_csv(ROOT / "data" / f"test_{fname}.csv")
        for col in tr.columns:
            train_feats[f"{prefix}{col}"] = tr[col].values
            test_feats[f"{prefix}{col}"] = te[col].values
    return train_feats, test_feats


def build_base_features(train_df, test_df, extra_feats=None):
    """Build the standard 36-feature set, optionally adding extras."""
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    # Remove temporal leakers
    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]

    # Add weather + solar
    train_feats, test_feats = add_weather_solar(train_feats, test_feats)

    # Prune to 36 validated features
    use_feats = list(KEEP_FEATURES)
    if extra_feats:
        use_feats += extra_feats

    available = [f for f in use_feats if f in train_feats.columns]
    train_feats = train_feats[available]
    test_feats = test_feats[available]

    X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    return X, X_test, available


def run_skf(X, y, X_test, label=""):
    """Run LGB 5-fold SKF, return OOF preds and mAP."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_pred = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
            n_jobs=-1,
        )
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof[va_idx] = lgb.predict_proba(X[va_idx])
        test_pred += lgb.predict_proba(X_test) / N_FOLDS

    m, per = compute_map(y, oof)
    print(f"  {label} SKF mAP: {m:.4f}", flush=True)
    return oof, test_pred, m, per


def run_lomo(X, y, train_months, label=""):
    """Run LGB LOMO, return OOF preds and mAP."""
    unique = sorted(np.unique(train_months))
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)

    for month in unique:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = lgb.predict_proba(X[va_idx])

    m, per = compute_map(y, oof)
    print(f"  {label} LOMO mAP: {m:.4f}", flush=True)
    return oof, m, per


# ======================================================================
print("=" * 70, flush=True)
print("E168 DATA ABLATIONS".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data ---------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_hours = train_ts.dt.hour.values
test_hours = test_ts.dt.hour.values
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

# Quick stats
print(f"\n  Train hours >= 13: {(train_hours >= 13).sum()} / {len(train_hours)} "
      f"({100*(train_hours >= 13).mean():.1f}%)", flush=True)
print(f"  Test hours >= 13:  {(test_hours >= 13).sum()} / {len(test_hours)} "
      f"({100*(test_hours >= 13).mean():.1f}%)", flush=True)
print(f"  Train Sep tracks:  {(train_months == 9).sum()}", flush=True)
print(f"  Test Sep tracks:   {(test_months == 9).sum()}", flush=True)

# -- Build features (once) ---------------------------------------------
print("\nBuilding features...", flush=True)
X_all, X_test, feat_names = build_base_features(train_df, test_df)
print(f"  {X_all.shape[1]} features: {feat_names[:5]}...", flush=True)

# ======================================================================
# A. BASELINE (bug-fixed 36 features)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("A. BASELINE (36 features, full train)", flush=True)
print("=" * 70, flush=True)
oof_a, test_a, skf_a, per_a = run_skf(X_all, y, X_test, "A")
_, lomo_a, lomo_per_a = run_lomo(X_all, y, train_months, "A")

# ======================================================================
# B. HOUR FILTER: drop train tracks with hour >= 13
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("B. HOUR FILTER (drop train hour >= 13)", flush=True)
print("=" * 70, flush=True)

mask_hour = train_hours < 13
print(f"  Keeping {mask_hour.sum()} / {len(mask_hour)} train tracks", flush=True)
X_hourfilter = X_all[mask_hour]
y_hourfilter = y[mask_hour]
months_hourfilter = train_months[mask_hour]

# For SKF, we train on filtered data but predict on all test
oof_b_partial, test_b, skf_b, per_b = run_skf(X_hourfilter, y_hourfilter, X_test, "B")
# LOMO on filtered data
_, lomo_b, lomo_per_b = run_lomo(X_hourfilter, y_hourfilter, months_hourfilter, "B")

# Also check: what classes are we dropping?
print("\n  Dropped tracks by class:", flush=True)
dropped_classes = y[~mask_hour]
for i, cls in enumerate(CLASSES):
    n_dropped = (dropped_classes == i).sum()
    n_total = (y == i).sum()
    if n_dropped > 0:
        print(f"    {cls:15s}: {n_dropped}/{n_total} ({100*n_dropped/n_total:.1f}%)", flush=True)

# ======================================================================
# C. SEPTEMBER-AS-UNSEEN (LOMO only -- exclude Sep from "shared")
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("C. SEPTEMBER AS UNSEEN (LOMO-only test)", flush=True)
print("=" * 70, flush=True)

# Standard LOMO: train on all months except held-out
# Sep-as-unseen LOMO: when evaluating, weight Sep differently
# But the real test: does training WITHOUT Sep improve test preds?
# Train on {Jan, Apr, Oct} only, predict Sep as unseen month

# Test 1: LOMO excluding Sep from training folds
print("\n  C1: Train only on {Jan, Apr, Oct}, predict Sep OOF", flush=True)
non_sep_mask = train_months != 9
sep_mask = train_months == 9
X_nosep = X_all[non_sep_mask]
y_nosep = y[non_sep_mask]
months_nosep = train_months[non_sep_mask]

# Train on {Jan,Apr,Oct} and predict Sep
lgb_c = LGBMClassifier(
    n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
    subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
    class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
)
lgb_c.fit(X_nosep, y_nosep)
sep_preds = lgb_c.predict_proba(X_all[sep_mask])
sep_map, sep_per = compute_map(y[sep_mask], sep_preds)
print(f"  Sep OOF mAP (trained w/o Sep): {sep_map:.4f}", flush=True)

# Compare to LOMO Sep fold (trained on Jan+Apr+Oct, predict Sep) -- same thing
# The interesting comparison is SKF:
# In SKF, Sep samples are mixed across folds, so the model sees Sep in training
# If Sep has shifted distribution, SKF overestimates performance on Sep-like test data

# Test 2: SKF on non-Sep data only
print("\n  C2: SKF on non-Sep data only", flush=True)
oof_c2, test_c2, skf_c2, per_c2 = run_skf(X_nosep, y_nosep, X_test, "C2")
_, lomo_c2, lomo_per_c2 = run_lomo(X_nosep, y_nosep, months_nosep, "C2")

print(f"\n  Non-Sep train size: {len(y_nosep)} vs full: {len(y)}", flush=True)

# ======================================================================
# D. HOUR FILTER + NO SEP (combined)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("D. HOUR FILTER + NO SEP (combined)", flush=True)
print("=" * 70, flush=True)

mask_d = (train_hours < 13) & (train_months != 9)
print(f"  Keeping {mask_d.sum()} / {len(y)} train tracks", flush=True)
X_d = X_all[mask_d]
y_d = y[mask_d]
months_d = train_months[mask_d]

oof_d, test_d, skf_d, per_d = run_skf(X_d, y_d, X_test, "D")
_, lomo_d, lomo_per_d = run_lomo(X_d, y_d, months_d, "D")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)

results = [
    ("A. Baseline (36 feat)", skf_a, lomo_a, per_a),
    ("B. Hour < 13 filter", skf_b, lomo_b, per_b),
    ("C2. No September", skf_c2, lomo_c2, per_c2),
    ("D. Hour<13 + No Sep", skf_d, lomo_d, per_d),
]

print(f"\n  {'Variant':<25s} {'SKF':>8s} {'LOMO':>8s} {'dSKF':>8s} {'dLOMO':>8s}", flush=True)
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", flush=True)
for name, skf_val, lomo_val, _ in results:
    d_skf = skf_val - skf_a
    d_lomo = lomo_val - lomo_a
    print(f"  {name:<25s} {skf_val:8.4f} {lomo_val:8.4f} {d_skf:+8.4f} {d_lomo:+8.4f}", flush=True)

# Per-class comparison
print(f"\n  Per-class SKF deltas vs baseline:", flush=True)
print(f"  {'Class':<15s}", end="", flush=True)
for name, _, _, _ in results[1:]:
    short = name.split(".")[0] + "." + name.split(".")[1][:8]
    print(f" {short:>10s}", end="", flush=True)
print(flush=True)

for cls in CLASSES:
    print(f"  {cls:<15s}", end="", flush=True)
    for _, _, _, per in results[1:]:
        d = per[cls] - per_a[cls]
        print(f" {d:+10.4f}", end="", flush=True)
    print(flush=True)

# Save best test predictions
best_idx = np.argmax([r[1] for r in results])  # best SKF
best_name = results[best_idx][0]
test_preds = [test_a, test_b, test_c2, test_d][best_idx]
print(f"\n  Best variant: {best_name}", flush=True)

# Save all test predictions
np.save(ROOT / "test_e168_A_baseline.npy", test_a)
np.save(ROOT / "test_e168_B_hourfilter.npy", test_b)
np.save(ROOT / "test_e168_C_nosep.npy", test_c2)
np.save(ROOT / "test_e168_D_combined.npy", test_d)

print("\nDone.", flush=True)
