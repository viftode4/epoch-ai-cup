"""E27: LOMO CV Baseline -- StratifiedKFold vs Leave-One-Month-Out.

Runs E25D config (LGB+XGB+CB, effective number weights, no temporal features)
twice: SKF (5 folds) and LOMO (4 folds, one month per fold).
Logit adjustment on both. Side-by-side comparison.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
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
BETA = 0.999

ALL_TEMPORAL = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring", "hour_bin_3h",
    # 5 weakclass temporal leaks
    "is_oct_nov", "migration_alt", "migration_speed", "is_night", "night_high_alt",
]

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}


# ── Reusable functions ────────────────────────────────────────────

def train_ensemble(X, y, X_test, fn, sample_weights, folds_list, n_folds, label):
    """Train LGB+XGB+CB ensemble on given folds. Returns (oof, test_pred, best_w)."""
    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(folds_list):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx]

        # LGB
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
        mdl = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = mdl.predict(X_va)
        test_lgb += mdl.predict(X_test) / n_folds

        # XGB
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
        mdl = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
        oof_xgb[va_idx] = mdl.predict(dval_xgb)
        test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=fn)) / n_folds

        # CatBoost
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / n_folds

        # Per-fold mAP
        fold_oof = 0.33 * oof_lgb[va_idx] + 0.33 * oof_xgb[va_idx] + 0.34 * oof_cb[va_idx]
        fold_map, _ = compute_map(y_va, fold_oof)
        print(f"  [{label}] Fold {fold} done -- mAP={fold_map:.4f} (n={len(va_idx)})", flush=True)

    # Optimize weights
    best_map = 0
    best_w = (0.33, 0.33, 0.34)
    for w1 in np.arange(0.05, 0.50, 0.05):
        for w2 in np.arange(0.05, 0.50, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w1, w2, w3)

    oof = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test_pred = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    print(f"  [{label}] Best weights: LGB={best_w[0]:.2f} XGB={best_w[1]:.2f} CB={best_w[2]:.2f} -> mAP={best_map:.4f}", flush=True)
    return oof, test_pred, best_w


def apply_logit_adj(oof, test_pred, y, counts, label):
    """Apply per-class logit adjustment. Returns (oof_adj, test_adj, final_map, per_class)."""
    priors = counts / counts.sum()
    tau = np.zeros(N_CLASSES)
    base_map, _ = compute_map(y, oof)
    best_map_adj = base_map

    for iteration in range(3):
        improved = False
        for c in range(N_CLASSES):
            best_t = tau[c]
            best_m = best_map_adj
            for t in np.arange(-0.5, 1.51, 0.02):
                tau[c] = t
                adj = priors ** (-tau)
                a = oof * adj[None, :]
                a = a / a.sum(axis=1, keepdims=True)
                m, _ = compute_map(y, a)
                if m > best_m:
                    best_m = m
                    best_t = t
            tau[c] = best_t
            if best_m > best_map_adj:
                best_map_adj = best_m
                improved = True
        print(f"  [{label}] Logit adj round {iteration+1}: {best_map_adj:.4f}", flush=True)
        if not improved:
            break

    adj = priors ** (-tau)
    oof_adj = oof * adj[None, :]
    oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
    test_adj = test_pred * adj[None, :]
    test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

    final_map, final_per = compute_map(y, oof_adj)
    return oof_adj, test_adj, final_map, final_per


# ── Main ──────────────────────────────────────────────────────────

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Effective number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# Build features
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

# ── StratifiedKFold (5 folds) ────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("StratifiedKFold (5 folds)", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_folds = list(skf.split(np.zeros(len(y)), y))

oof_skf, test_skf, w_skf = train_ensemble(X, y, X_test, fn, sample_weights, skf_folds, 5, "SKF")
oof_skf_adj, test_skf_adj, map_skf, per_skf = apply_logit_adj(oof_skf, test_skf, y, counts, "SKF")
print_results(map_skf, per_skf, "E27 SKF: StratifiedKFold (5 folds)")

# ── LOMO (4 folds) ───────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("LOMO -- Leave-One-Month-Out (4 folds)", flush=True)
print("=" * 60, flush=True)

ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = ts.dt.month.values
unique_months = sorted(np.unique(train_months))
print(f"  Train months: {unique_months}", flush=True)
for m in unique_months:
    n_m = np.sum(train_months == m)
    print(f"    Month {m:2d}: {n_m} samples", flush=True)

lomo_folds = []
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    lomo_folds.append((tr_idx, va_idx))

n_lomo = len(lomo_folds)
oof_lomo, test_lomo, w_lomo = train_ensemble(X, y, X_test, fn, sample_weights, lomo_folds, n_lomo, "LOMO")
oof_lomo_adj, test_lomo_adj, map_lomo, per_lomo = apply_logit_adj(oof_lomo, test_lomo, y, counts, "LOMO")
print_results(map_lomo, per_lomo, "E27 LOMO: Leave-One-Month-Out (4 folds)")

# ── Side-by-side comparison ──────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("COMPARISON: SKF vs LOMO", flush=True)
print("=" * 60, flush=True)
print(f"  {'Method':<20s} {'mAP':>8s}", flush=True)
print(f"  {'SKF (5 folds)':<20s} {map_skf:>8.4f}", flush=True)
print(f"  {'LOMO (4 folds)':<20s} {map_lomo:>8.4f}", flush=True)
print(f"  {'Delta (SKF-LOMO)':<20s} {map_skf - map_lomo:>+8.4f}", flush=True)
print(flush=True)
print(f"  {'Class':<15s} {'SKF':>8s} {'LOMO':>8s} {'Delta':>8s}", flush=True)
for cls in CLASSES:
    d = per_skf.get(cls, 0) - per_lomo.get(cls, 0)
    print(f"  {cls:<15s} {per_skf.get(cls, 0):>8.4f} {per_lomo.get(cls, 0):>8.4f} {d:>+8.4f}", flush=True)

# ── Per-fold LOMO mAP ────────────────────────────────────────────
print("\n  LOMO per-fold mAP:", flush=True)
for i, m in enumerate(unique_months):
    va_idx = lomo_folds[i][1]
    fold_map, fold_per = compute_map(y[va_idx], oof_lomo_adj[va_idx])
    classes_in_fold = np.unique(y[va_idx])
    print(f"    Month {m:2d}: mAP={fold_map:.4f} (n={len(va_idx)}, classes={len(classes_in_fold)})", flush=True)

# ── Test distribution ─────────────────────────────────────────────
print("\nTest prediction distributions:", flush=True)
for label_name, preds in [("SKF", test_skf_adj), ("LOMO", test_lomo_adj)]:
    pred_classes = preds.argmax(axis=1)
    dist = np.bincount(pred_classes, minlength=N_CLASSES)
    print(f"  {label_name}: {dict(zip(CLASSES, dist))}", flush=True)

# ── Save ──────────────────────────────────────────────────────────
np.save(ROOT / "oof_e27_skf.npy", oof_skf_adj)
np.save(ROOT / "test_e27_skf.npy", test_skf_adj)
np.save(ROOT / "oof_e27_lomo.npy", oof_lomo_adj)
np.save(ROOT / "test_e27_lomo.npy", test_lomo_adj)

save_submission(test_lomo_adj, "e27_lomo", cv_map=map_lomo)
print("\nDone!", flush=True)
