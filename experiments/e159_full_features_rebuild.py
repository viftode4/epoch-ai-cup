"""E159: Full feature rebuild with trajectory separators.

Rebuilds from ALL available non-temporal features including new
trajectory separators (heading_R, rcs_spectral_entropy, soaring_frac, etc).

Evaluates using proxy fold per-class APs (not SKF, not calibrated IW-mAP).
The question: do the new features help on unseen-month proxy folds?
"""
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

from src.data import CLASSES, ROOT, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results

N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42
DATA_DIR = ROOT / "data"

# ── Load data ─────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("E159: FULL FEATURE REBUILD".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

# ── Build ALL features ────────────────────────────────────────────
print("\nBuilding ALL feature sets...", flush=True)
feat_sets = [
    "core", "rcs_fft", "tabular", "targeted",
    "flight_mode", "weakclass", "rcs_slope", "trajectory_separators",
]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
for name, prefix in [("weather", "wx_"), ("solar", "sol_")]:
    train_ext = pd.read_csv(DATA_DIR / f"train_{name}.csv")
    test_ext = pd.read_csv(DATA_DIR / f"test_{name}.csv")
    for col in train_ext.columns:
        train_feats[f"{prefix}{col}"] = train_ext[col].values
        test_feats[f"{prefix}{col}"] = test_ext[col].values

print(f"  Total features: {len(train_feats.columns)}", flush=True)

# Check new separator features are present
new_feats = ["heading_R", "rcs_spectral_entropy", "speed_autocorr",
             "alt_ascending_frac", "alt_descending_frac", "alt_flat_frac",
             "soaring_frac", "rcs_burst_frac", "rcs_smooth_frac"]
present = [f for f in new_feats if f in train_feats.columns]
print(f"  New trajectory separators: {len(present)}/{len(new_feats)}", flush=True)
for f in present:
    print(f"    {f}: mean={train_feats[f].mean():.4f}, std={train_feats[f].std():.4f}", flush=True)

X_train = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# ── Class weights ─────────────────────────────────────────────────
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# ── Proxy fold evaluation ─────────────────────────────────────────
FOLD_TO_PROXY = {
    9:  ("Sep", "Sep (shared)"),
    10: ("Oct", "Oct (shared)"),
    1:  ("Jan", "Feb+Dec (unseen)"),
    4:  ("Apr", "May (unseen)"),
}


def evaluate_proxy_folds(oof, label):
    """Evaluate per-class AP on each proxy fold."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"  PROXY FOLD EVALUATION: {label}", flush=True)
    print(f"{'=' * 70}", flush=True)

    for held_out in [1, 4, 9, 10]:
        name, proxy = FOLD_TO_PROXY[held_out]
        va_mask = months == held_out
        va_y = y[va_mask]
        va_oof = oof[va_mask]
        n = va_mask.sum()

        y_bin = np.zeros((len(va_y), N_CLASSES), dtype=int)
        y_bin[np.arange(len(va_y)), va_y] = 1

        print(f"\n  --- {name} (n={n}) -> {proxy} ---", flush=True)
        aps = []
        for c in range(N_CLASSES):
            nc = y_bin[:, c].sum()
            if nc == 0:
                continue
            ap = average_precision_score(y_bin[:, c], va_oof[:, c])
            aps.append(ap)
            status = "OK" if ap > 0.5 else ("WEAK" if ap > 0.2 else "BROKEN")
            print(f"    {CLASSES[c]:>15s}  n={nc:>3d}  AP={ap:.3f}  [{status}]", flush=True)
        print(f"    {'macro mAP':>15s}  n={n:>3d}  AP={np.mean(aps):.3f}", flush=True)


# ── SKF training ──────────────────────────────────────────────────
print("\n--- SKF ensemble training (5-fold) ---", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    # LGB
    lgb = LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1,
        device="gpu", n_jobs=-1,
    )
    lgb.fit(X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X_train[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGB
    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_train[va_idx])
    test_xgb += xgb.predict_proba(X_test) / N_FOLDS

    # CatBoost
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=7,
        l2_leaf_reg=5.0, bagging_temperature=1.0,
        class_weights={i: class_weights_arr[i] for i in range(N_CLASSES)},
        random_seed=SEED, verbose=0, task_type="GPU",
    )
    cb.fit(X_train[tr_idx], y[tr_idx],
           eval_set=(X_train[va_idx], y[va_idx]))
    oof_cb[va_idx] = cb.predict_proba(X_train[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

# Individual model scores
m_lgb, _ = compute_map(y, oof_lgb)
m_xgb, _ = compute_map(y, oof_xgb)
m_cb, _ = compute_map(y, oof_cb)
print(f"  LGB SKF mAP: {m_lgb:.4f}", flush=True)
print(f"  XGB SKF mAP: {m_xgb:.4f}", flush=True)
print(f"  CB  SKF mAP: {m_cb:.4f}", flush=True)

# ── Ensemble weight optimization ──────────────────────────────────
print("\n--- Ensemble weight optimization ---", flush=True)
best_w, best_map = None, -1.0
for w_lgb in np.arange(0.0, 1.05, 0.10):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.10):
        w_cb = round(1.0 - w_lgb - w_xgb, 2)
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

print(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Ensemble SKF mAP: {best_map:.4f}", flush=True)

# ── Print full results ────────────────────────────────────────────
m_ens, per_ens = compute_map(y, oof_ens)
print_results(m_ens, per_ens, label="E159 ensemble")

# ── Proxy fold evaluation ─────────────────────────────────────────
evaluate_proxy_folds(oof_ens, "E159 full features")

# ── Compare with E79/E156 baseline ────────────────────────────────
for baseline_name, baseline_path in [("E79", ROOT / "oof_e79.npy"),
                                      ("E156", ROOT / "oof_e156.npy")]:
    if baseline_path.exists():
        oof_base = np.load(baseline_path).astype(np.float64)
        evaluate_proxy_folds(oof_base, f"{baseline_name} baseline")

# ── Feature importance (top 30) ───────────────────────────────────
print("\n--- Top 30 feature importances (LGB last fold) ---", flush=True)
imp = lgb.feature_importances_
idx = np.argsort(imp)[::-1][:30]
for rank, i in enumerate(idx):
    marker = " <<<" if feature_names[i] in new_feats else ""
    print(f"  {rank+1:>2d}. {feature_names[i]:>30s}: {imp[i]:>5d}{marker}", flush=True)

# ── Save ──────────────────────────────────────────────────────────
from src.submission import save_submission
save_submission(test_ens, f"e159_full", cv_map=best_map)
np.save(ROOT / "oof_e159.npy", oof_ens)
np.save(ROOT / "test_e159.npy", test_ens)
print("\nDone.", flush=True)
