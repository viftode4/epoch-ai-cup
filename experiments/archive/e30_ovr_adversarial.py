"""E30: OvR + Adversarial Weights.

OvR binary classifiers with adversarial-weighted samples.
Loads adversarial weights from E28 (or recomputes if missing).
LOMO + SKF CV evaluation.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
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
    if n_pos >= 400:
        return {"num_leaves": 47, "min_child_samples": 5, "max_depth": 6, "reg_lambda": 3}
    elif n_pos >= 100:
        return {"num_leaves": 31, "min_child_samples": 8, "max_depth": 5, "reg_lambda": 5}
    elif n_pos >= 50:
        return {"num_leaves": 23, "min_child_samples": 10, "max_depth": 5, "reg_lambda": 7}
    else:
        return {"num_leaves": 15, "min_child_samples": 15, "max_depth": 4, "reg_lambda": 10}


def get_cb_params_for_class(n_pos):
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

# ── Load or recompute adversarial weights ─────────────────────────
adv_path = ROOT / "adv_weights_e28.npy"
if adv_path.exists():
    print("Loading adversarial weights from E28...", flush=True)
    adv_w = np.load(adv_path)
else:
    print("Recomputing adversarial weights...", flush=True)
    X_adv = np.vstack([X, X_test])
    y_adv = np.array([0] * len(X) + [1] * len(X_test))
    adv_params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": 5, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "verbose": -1, "seed": 42, "device": "gpu",
    }
    skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_adv = np.zeros(len(y_adv))
    for fold, (tr_idx, va_idx) in enumerate(skf_adv.split(X_adv, y_adv)):
        dtrain = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx])
        dval = lgb.Dataset(X_adv[va_idx], label=y_adv[va_idx])
        mdl = lgb.train(adv_params, dtrain, num_boost_round=500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_adv[va_idx] = mdl.predict(X_adv[va_idx])
    p_test = np.clip(oof_adv[:len(X)], 0.01, 0.99)
    adv_w = np.clip(p_test / (1.0 - p_test), 0.1, 10.0)
    adv_w = adv_w / adv_w.mean()
    np.save(adv_path, adv_w)

print(f"  Adversarial weight stats: min={adv_w.min():.3f} mean={adv_w.mean():.3f} max={adv_w.max():.3f}", flush=True)

# ── Build folds ───────────────────────────────────────────────────
ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = ts.dt.month.values
unique_months = sorted(np.unique(train_months))

lomo_folds = []
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    lomo_folds.append((tr_idx, va_idx))
n_lomo = len(lomo_folds)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_folds = list(skf.split(np.zeros(len(y)), y))


# ── Train OvR with adversarial weights ────────────────────────────
print("\n" + "=" * 60, flush=True)
print("OvR + Adversarial Weights (SKF)", flush=True)
print("=" * 60, flush=True)

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

    print(f"\n  Class {c}: {cls_name} (n_pos={n_pos})", flush=True)

    # Bake imbalance ratio into per-sample weights
    # positive samples get adv_w * ratio, negative get adv_w * 1.0
    sample_w = adv_w.copy()
    sample_w[y_c == 1] *= ratio

    lgb_base = {
        "objective": "binary", "metric": "average_precision",
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "verbose": -1, "seed": 42, "n_jobs": -1,
        "device": "gpu",
        # No scale_pos_weight -- baked into sample_w
    }
    lgb_base.update(lgb_cls_params)

    oof_lgb_s = np.zeros(len(X))
    oof_cb_s = np.zeros(len(X))
    test_lgb_s = np.zeros(len(X_test))
    test_cb_s = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf_folds):
        y_tr_c = y_c[tr_idx]
        y_va_c = y_c[va_idx]
        w_tr = sample_w[tr_idx]

        dtrain = lgb.Dataset(X[tr_idx], label=y_tr_c, weight=w_tr, feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y_va_c, feature_name=fn, reference=dtrain)
        mdl = lgb.train(lgb_base, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb_s[va_idx] = mdl.predict(X[va_idx])
        test_lgb_s += mdl.predict(X_test) / 5

        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05,
            depth=cb_cls_params["depth"], l2_leaf_reg=cb_cls_params["l2_leaf_reg"],
            loss_function="Logloss", eval_metric="PRAUC:type=Classic",
            # No scale_pos_weight -- baked into sample_w
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X[tr_idx], y_tr_c, eval_set=(X[va_idx], y_va_c), verbose=0,
               sample_weight=w_tr)
        oof_cb_s[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
        test_cb_s += cb.predict_proba(X_test)[:, 1] / 5

    # SKF blend
    best_alpha = 0.5
    best_ap = 0
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = alpha * oof_lgb_s + (1 - alpha) * oof_cb_s
        ap = average_precision_score(y_c, blended)
        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha

    oof_skf[:, c] = best_alpha * oof_lgb_s + (1 - best_alpha) * oof_cb_s
    test_skf[:, c] = best_alpha * test_lgb_s + (1 - best_alpha) * test_cb_s
    print(f"    SKF blend:  alpha(LGB)={best_alpha:.2f} -> AP={best_ap:.4f}", flush=True)


# ── Evaluate ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("RESULTS", flush=True)
print("=" * 60, flush=True)

map_skf, per_skf = compute_map(y, oof_skf)
print_results(map_skf, per_skf, "E30 OvR+Adversarial SKF")

# Compare with E29 OvR (no adversarial) if available
e29_path = ROOT / "oof_e29_skf.npy"
if e29_path.exists():
    oof_e29 = np.load(e29_path)
    map_e29, per_e29 = compute_map(y, oof_e29)
    print(f"\n  Comparison: E29 OvR (no adv) = {map_e29:.4f}, E30 OvR+adv = {map_skf:.4f}, delta = {map_skf - map_e29:+.4f}", flush=True)
    print(f"\n  {'Class':<15s} {'E29 (no adv)':>12s} {'E30 (+adv)':>12s} {'Delta':>8s}", flush=True)
    for cls in CLASSES:
        d = per_skf.get(cls, 0) - per_e29.get(cls, 0)
        print(f"  {cls:<15s} {per_e29.get(cls, 0):>12.4f} {per_skf.get(cls, 0):>12.4f} {d:>+8.4f}", flush=True)

# Test distribution
print("\nTest prediction distributions:", flush=True)
pred_classes = test_skf.argmax(axis=1)
dist = np.bincount(pred_classes, minlength=N_CLASSES)
print(f"  E30: {dict(zip(CLASSES, dist))}", flush=True)

# ── Save ──────────────────────────────────────────────────────────
np.save(ROOT / "oof_e30.npy", oof_skf)
np.save(ROOT / "test_e30.npy", test_skf)

save_submission(test_skf, "e30_ovr_adversarial", cv_map=map_skf)
print("\nDone!", flush=True)
