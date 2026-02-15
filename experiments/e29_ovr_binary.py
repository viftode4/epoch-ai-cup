"""E29: One-vs-Rest Binary Classifiers.

For each of 9 classes: train LGB + CatBoost binary classifiers with LOMO CV.
LGB uses metric='average_precision' (directly optimizes AP for early stopping).
CatBoost uses eval_metric='PRAUC:type=Classic'.
Per-class hyperparameter adjustment based on class size.
Per-class LGB/CB blend weight optimization on OOF.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

ALL_TEMPORAL = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring", "hour_bin_3h",
    "is_oct_nov", "migration_alt", "migration_speed", "is_night", "night_high_alt",
]


def get_lgb_params_for_class(n_pos):
    """Per-class hyperparameters based on class size."""
    if n_pos >= 400:
        return {"num_leaves": 47, "min_child_samples": 5, "max_depth": 6, "reg_lambda": 3}
    elif n_pos >= 100:
        return {"num_leaves": 31, "min_child_samples": 8, "max_depth": 5, "reg_lambda": 5}
    elif n_pos >= 50:
        return {"num_leaves": 23, "min_child_samples": 10, "max_depth": 5, "reg_lambda": 7}
    else:
        return {"num_leaves": 15, "min_child_samples": 15, "max_depth": 4, "reg_lambda": 10}


def get_cb_params_for_class(n_pos):
    """Per-class CatBoost params based on class size."""
    if n_pos >= 400:
        return {"depth": 6, "l2_leaf_reg": 3}
    elif n_pos >= 100:
        return {"depth": 5, "l2_leaf_reg": 5}
    elif n_pos >= 50:
        return {"depth": 5, "l2_leaf_reg": 7}
    else:
        return {"depth": 4, "l2_leaf_reg": 10}


# ── Main ──────────────────────────────────────────────────────────

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

print("Building features (no temporal + weakclass)...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
print(f"  Features: {train_feats.shape[1]}", flush=True)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)

# ── Build LOMO folds ─────────────────────────────────────────────
ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = ts.dt.month.values
unique_months = sorted(np.unique(train_months))
print(f"  Train months: {unique_months}", flush=True)

lomo_folds = []
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    lomo_folds.append((tr_idx, va_idx))
n_lomo = len(lomo_folds)

# Also build SKF folds for comparison
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_folds = list(skf.split(np.zeros(len(y)), y))

# ── Train OvR binary classifiers ─────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("One-vs-Rest Binary Classifiers (LOMO)", flush=True)
print("=" * 60, flush=True)

oof_lomo = np.zeros((len(X), N_CLASSES))
test_lomo = np.zeros((len(X_test), N_CLASSES))
oof_skf = np.zeros((len(X), N_CLASSES))
test_skf = np.zeros((len(X_test), N_CLASSES))

for c in range(N_CLASSES):
    cls_name = CLASSES[c]
    y_c = (y == c).astype(int)
    n_pos = y_c.sum()
    n_neg = len(y_c) - n_pos
    ratio = n_neg / max(n_pos, 1)

    lgb_cls_params = get_lgb_params_for_class(n_pos)
    cb_cls_params = get_cb_params_for_class(n_pos)

    print(f"\n  Class {c}: {cls_name} (n_pos={n_pos}, ratio={ratio:.1f})", flush=True)

    lgb_base = {
        "objective": "binary", "metric": "average_precision",
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "verbose": -1, "seed": 42, "n_jobs": -1,
        "device": "gpu", "scale_pos_weight": ratio,
    }
    lgb_base.update(lgb_cls_params)

    # ── LOMO ──
    oof_lgb_c = np.zeros(len(X))
    oof_cb_c = np.zeros(len(X))
    test_lgb_c = np.zeros(len(X_test))
    test_cb_c = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(lomo_folds):
        y_tr_c = y_c[tr_idx]
        y_va_c = y_c[va_idx]

        n_pos_fold = y_va_c.sum()
        if n_pos_fold == 0:
            print(f"    WARNING: Fold {fold} has 0 positives for {cls_name}, skipping", flush=True)
            continue

        # LGB
        dtrain = lgb.Dataset(X[tr_idx], label=y_tr_c, feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y_va_c, feature_name=fn, reference=dtrain)
        mdl = lgb.train(lgb_base, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb_c[va_idx] = mdl.predict(X[va_idx])
        test_lgb_c += mdl.predict(X_test) / n_lomo

        # CatBoost
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05,
            depth=cb_cls_params["depth"], l2_leaf_reg=cb_cls_params["l2_leaf_reg"],
            loss_function="Logloss", eval_metric="PRAUC:type=Classic",
            scale_pos_weight=ratio,
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X[tr_idx], y_tr_c, eval_set=(X[va_idx], y_va_c), verbose=0)
        oof_cb_c[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
        test_cb_c += cb.predict_proba(X_test)[:, 1] / n_lomo

    # Per-class blend optimization on OOF
    best_alpha = 0.5
    best_ap = 0
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = alpha * oof_lgb_c + (1 - alpha) * oof_cb_c
        # Only evaluate on samples that were in validation sets
        valid_mask = (oof_lgb_c != 0) | (oof_cb_c != 0) | (y_c == 1)
        if y_c[valid_mask].sum() > 0:
            ap = average_precision_score(y_c[valid_mask], blended[valid_mask])
        else:
            ap = 0
        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha

    oof_lomo[:, c] = best_alpha * oof_lgb_c + (1 - best_alpha) * oof_cb_c
    test_lomo[:, c] = best_alpha * test_lgb_c + (1 - best_alpha) * test_cb_c
    print(f"    LOMO blend: alpha(LGB)={best_alpha:.2f} -> AP={best_ap:.4f}", flush=True)

    # ── SKF ──
    oof_lgb_s = np.zeros(len(X))
    oof_cb_s = np.zeros(len(X))
    test_lgb_s = np.zeros(len(X_test))
    test_cb_s = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf_folds):
        y_tr_c = y_c[tr_idx]
        y_va_c = y_c[va_idx]

        dtrain = lgb.Dataset(X[tr_idx], label=y_tr_c, feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y_va_c, feature_name=fn, reference=dtrain)
        mdl = lgb.train(lgb_base, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb_s[va_idx] = mdl.predict(X[va_idx])
        test_lgb_s += mdl.predict(X_test) / 5

        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05,
            depth=cb_cls_params["depth"], l2_leaf_reg=cb_cls_params["l2_leaf_reg"],
            loss_function="Logloss", eval_metric="PRAUC:type=Classic",
            scale_pos_weight=ratio,
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X[tr_idx], y_tr_c, eval_set=(X[va_idx], y_va_c), verbose=0)
        oof_cb_s[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
        test_cb_s += cb.predict_proba(X_test)[:, 1] / 5

    # SKF blend
    best_alpha_s = 0.5
    best_ap_s = 0
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = alpha * oof_lgb_s + (1 - alpha) * oof_cb_s
        ap = average_precision_score(y_c, blended)
        if ap > best_ap_s:
            best_ap_s = ap
            best_alpha_s = alpha

    oof_skf[:, c] = best_alpha_s * oof_lgb_s + (1 - best_alpha_s) * oof_cb_s
    test_skf[:, c] = best_alpha_s * test_lgb_s + (1 - best_alpha_s) * test_cb_s
    print(f"    SKF blend:  alpha(LGB)={best_alpha_s:.2f} -> AP={best_ap_s:.4f}", flush=True)


# ── Evaluate ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("RESULTS", flush=True)
print("=" * 60, flush=True)

map_lomo, per_lomo = compute_map(y, oof_lomo)
print_results(map_lomo, per_lomo, "E29 OvR LOMO")

map_skf, per_skf = compute_map(y, oof_skf)
print_results(map_skf, per_skf, "E29 OvR SKF")

# Comparison
print(f"\n  {'Method':<20s} {'mAP':>8s}", flush=True)
print(f"  {'OvR LOMO':<20s} {map_lomo:>8.4f}", flush=True)
print(f"  {'OvR SKF':<20s} {map_skf:>8.4f}", flush=True)
print(f"  {'Delta (SKF-LOMO)':<20s} {map_skf - map_lomo:>+8.4f}", flush=True)

# Test distribution
print("\nTest prediction distributions:", flush=True)
for name, preds in [("LOMO", test_lomo), ("SKF", test_skf)]:
    pred_classes = preds.argmax(axis=1)
    dist = np.bincount(pred_classes, minlength=N_CLASSES)
    print(f"  {name}: {dict(zip(CLASSES, dist))}", flush=True)

# ── Save ──────────────────────────────────────────────────────────
np.save(ROOT / "oof_e29_lomo.npy", oof_lomo)
np.save(ROOT / "test_e29_lomo.npy", test_lomo)
np.save(ROOT / "oof_e29_skf.npy", oof_skf)
np.save(ROOT / "test_e29_skf.npy", test_skf)

# Save SKF version as primary (more likely to perform on LB)
np.save(ROOT / "oof_e29.npy", oof_skf)
np.save(ROOT / "test_e29.npy", test_skf)

save_submission(test_skf, "e29_ovr_skf", cv_map=map_skf)
print("\nDone!", flush=True)
