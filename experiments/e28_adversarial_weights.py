"""E28: Adversarial Sample Weighting.

Trains adversarial LGB (train=0, test=1) to get P(test|x) for each train sample.
Computes adversarial weights: w = p/(1-p), clipped [0.1, 10], mean-normalized.
Combined with effective number class weights.
Three comparisons: LOMO-base, LOMO-adv, SKF-adv.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
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


def train_ensemble(X, y, X_test, fn, sample_weights, folds_list, n_folds, label):
    """Train LGB+XGB+CB ensemble. Returns (oof, test_pred, best_w)."""
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

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
        mdl = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = mdl.predict(X_va)
        test_lgb += mdl.predict(X_test) / n_folds

        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
        mdl = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
        oof_xgb[va_idx] = mdl.predict(dval_xgb)
        test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=fn)) / n_folds

        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / n_folds

        fold_oof = 0.33 * oof_lgb[va_idx] + 0.33 * oof_xgb[va_idx] + 0.34 * oof_cb[va_idx]
        fold_map, _ = compute_map(y_va, fold_oof)
        print(f"  [{label}] Fold {fold} done -- mAP={fold_map:.4f} (n={len(va_idx)})", flush=True)

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
    """Apply per-class logit adjustment."""
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

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

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


# ── Step 1: Train adversarial model ──────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Step 1: Adversarial validation model", flush=True)
print("=" * 60, flush=True)

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

auc = roc_auc_score(y_adv, oof_adv)
print(f"  Adversarial AUC: {auc:.4f}", flush=True)

# Extract P(test|x) for training samples only
p_test = np.clip(oof_adv[:len(X)], 0.01, 0.99)

# Adversarial weights: p/(1-p), clipped, normalized
adv_w_raw = p_test / (1.0 - p_test)
adv_w = np.clip(adv_w_raw, 0.1, 10.0)
adv_w = adv_w / adv_w.mean()

print(f"  Adversarial weight stats: min={adv_w.min():.3f} mean={adv_w.mean():.3f} max={adv_w.max():.3f} std={adv_w.std():.3f}", flush=True)

# Per-class mean adversarial weight
print("\n  Per-class mean adversarial weight:", flush=True)
for c in range(N_CLASSES):
    mask = y == c
    print(f"    {CLASSES[c]:15s}: {adv_w[mask].mean():.3f} (n={mask.sum()})", flush=True)

# Save adversarial weights
np.save(ROOT / "adv_weights_e28.npy", adv_w)
print(f"  Saved adv_weights_e28.npy", flush=True)


# ── Step 2: Combine class weights with adversarial weights ───────
class_sample_w = np.array([class_w[yi] for yi in y])
adv_class_w = class_sample_w * adv_w
adv_class_w = adv_class_w / adv_class_w.mean() * class_sample_w.mean()  # re-normalize to same scale

print(f"\n  Combined weight stats: min={adv_class_w.min():.3f} mean={adv_class_w.mean():.3f} max={adv_class_w.max():.3f}", flush=True)


# ── Step 3: Build folds ──────────────────────────────────────────
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


# ── Step 4: LOMO-base (no adversarial, same as E27) ─────────────
print("\n" + "=" * 60, flush=True)
print("LOMO-base (no adversarial weights)", flush=True)
print("=" * 60, flush=True)

oof_base, test_base, _ = train_ensemble(X, y, X_test, fn, class_sample_w, lomo_folds, n_lomo, "LOMO-base")
oof_base_adj, test_base_adj, map_base, per_base = apply_logit_adj(oof_base, test_base, y, counts, "LOMO-base")
print_results(map_base, per_base, "E28 LOMO-base")


# ── Step 5: LOMO-adv ─────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("LOMO-adv (with adversarial weights)", flush=True)
print("=" * 60, flush=True)

oof_adv_ens, test_adv_ens, _ = train_ensemble(X, y, X_test, fn, adv_class_w, lomo_folds, n_lomo, "LOMO-adv")
oof_adv_adj, test_adv_adj, map_adv, per_adv = apply_logit_adj(oof_adv_ens, test_adv_ens, y, counts, "LOMO-adv")
print_results(map_adv, per_adv, "E28 LOMO-adv")


# ── Step 6: SKF-adv ──────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SKF-adv (adversarial weights, StratifiedKFold)", flush=True)
print("=" * 60, flush=True)

oof_skf_adv, test_skf_adv, _ = train_ensemble(X, y, X_test, fn, adv_class_w, skf_folds, 5, "SKF-adv")
oof_skf_adj, test_skf_adj, map_skf, per_skf = apply_logit_adj(oof_skf_adv, test_skf_adv, y, counts, "SKF-adv")
print_results(map_skf, per_skf, "E28 SKF-adv")


# ── Comparison ────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("COMPARISON", flush=True)
print("=" * 60, flush=True)
print(f"  {'Method':<25s} {'mAP':>8s}", flush=True)
print(f"  {'LOMO-base':<25s} {map_base:>8.4f}", flush=True)
print(f"  {'LOMO-adv':<25s} {map_adv:>8.4f}", flush=True)
print(f"  {'SKF-adv':<25s} {map_skf:>8.4f}", flush=True)
print(f"  {'Delta LOMO (adv-base)':<25s} {map_adv - map_base:>+8.4f}", flush=True)
print(flush=True)

print(f"  {'Class':<15s} {'LOMO-base':>10s} {'LOMO-adv':>10s} {'SKF-adv':>10s} {'Delta':>8s}", flush=True)
for cls in CLASSES:
    d = per_adv.get(cls, 0) - per_base.get(cls, 0)
    print(f"  {cls:<15s} {per_base.get(cls, 0):>10.4f} {per_adv.get(cls, 0):>10.4f} {per_skf.get(cls, 0):>10.4f} {d:>+8.4f}", flush=True)

# Test distribution
print("\nTest prediction distributions:", flush=True)
for name, preds in [("LOMO-base", test_base_adj), ("LOMO-adv", test_adv_adj), ("SKF-adv", test_skf_adj)]:
    pred_classes = preds.argmax(axis=1)
    dist = np.bincount(pred_classes, minlength=N_CLASSES)
    print(f"  {name}: {dict(zip(CLASSES, dist))}", flush=True)

# ── Save best ─────────────────────────────────────────────────────
# Save SKF-adv as the primary E28 result (best for LB)
np.save(ROOT / "oof_e28.npy", oof_skf_adj)
np.save(ROOT / "test_e28.npy", test_skf_adj)

save_submission(test_skf_adj, "e28_adversarial_skf", cv_map=map_skf)
print("\nDone!", flush=True)
