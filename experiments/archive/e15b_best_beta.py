"""E15b: Retrain tree ensemble with best beta=0.999 (from E15 sweep)

Quick re-run of the winning configuration to save OOF + test predictions.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
BETA = 0.999

print(f"CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})", flush=True)

# ── Data ─────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print(f"Extracting features...", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# Effective Number weights
counts = np.bincount(y, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

print(f"\nbeta={BETA} weights:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:15s} (n={counts[i]:4d}): weight={class_w[i]:.3f}", flush=True)

# GPU params
lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1,
    "device": "gpu",
}

xgb_params = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), N_CLASSES))
oof_xgb = np.zeros((len(X), N_CLASSES))
oof_cb = np.zeros((len(X), N_CLASSES))
test_lgb = np.zeros((len(X_test), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))
test_cb = np.zeros((len(X_test), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold} ---", flush=True)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(X_va)
    test_lgb += mdl.predict(X_test) / N_FOLDS

    # XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                    evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = mdl.predict(dval_xgb)
    test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

    # CatBoost (GPU, manual weights)
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb[va_idx] = cb.predict_proba(X_va)
    test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Per-fold check
    oof_ens = 0.15 * oof_lgb[va_idx] + 0.10 * oof_xgb[va_idx] + 0.75 * oof_cb[va_idx]
    fold_map, _ = compute_map(y_va, oof_ens)
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

# Optimize weights
print("\nOptimizing ensemble weights...", flush=True)
best_map = 0
best_w = None
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

print(f"Best weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}", flush=True)

oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, f"E15 Tree Ensemble (beta={BETA})")

print(f"\nE10 was: 0.7322", flush=True)
print(f"E15:     {final_map:.4f} ({final_map - 0.7322:+.4f})", flush=True)

np.save(ROOT / "oof_e15.npy", oof_final)
np.save(ROOT / "test_e15.npy", test_final)
print("Saved oof_e15.npy and test_e15.npy", flush=True)

# ── Stacking: rebuild 4-model stack with E15 replacing E10 ───────
print(f"\n{'='*60}", flush=True)
print("Rebuilding 4-model stack with E15 trees", flush=True)
print(f"{'='*60}", flush=True)

oof_e08 = np.load(ROOT / "oof_e08.npy")
oof_e06 = np.load(ROOT / "oof_e06.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")
test_e08 = np.load(ROOT / "test_e08.npy")
test_e06 = np.load(ROOT / "test_e06.npy")
test_e09 = np.load(ROOT / "test_e09.npy")

# Optimize stack weights
best_stack_map = 0
best_stack_w = None
for w0 in np.arange(0.50, 0.90, 0.05):
    for w1 in np.arange(0.05, 0.25, 0.05):
        for w2 in np.arange(0.05, 0.25, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.05:
                continue
            oof_stack = w0 * oof_final + w1 * oof_e08 + w2 * oof_e06 + w3 * oof_e09
            m, _ = compute_map(y, oof_stack)
            if m > best_stack_map:
                best_stack_map = m
                best_stack_w = (w0, w1, w2, w3)

print(f"Best stack: tree={best_stack_w[0]:.2f} rocket={best_stack_w[1]:.2f} "
      f"cnn={best_stack_w[2]:.2f} svm={best_stack_w[3]:.2f}", flush=True)
print(f"Stack mAP: {best_stack_map:.4f} (E11 was 0.7396, delta={best_stack_map - 0.7396:+.4f})",
      flush=True)

oof_stack = (best_stack_w[0] * oof_final + best_stack_w[1] * oof_e08 +
             best_stack_w[2] * oof_e06 + best_stack_w[3] * oof_e09)
test_stack = (best_stack_w[0] * test_final + best_stack_w[1] * test_e08 +
              best_stack_w[2] * test_e06 + best_stack_w[3] * test_e09)

stack_map, stack_per = compute_map(y, oof_stack)
print_results(stack_map, stack_per, "E15 Stack (improved trees)")

# Apply logit adjustment (T08) on top
counts = np.bincount(y, minlength=N_CLASSES)
priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = stack_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj = priors ** (-per_class_tau)
            adjusted = oof_stack * adj[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c
        per_class_tau[c] = best_c_tau
        if best_c_map > current_best:
            current_best = best_c_map
            improved = True
    print(f"  Logit adj round {iteration + 1}: mAP={current_best:.4f}", flush=True)
    if not improved:
        break

adj = priors ** (-per_class_tau)
oof_adj = oof_stack * adj[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
test_adj = test_stack * adj[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, "E15 Stack + Logit Adjustment")

print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E10 tree alone:          0.7322", flush=True)
print(f"  E15 tree alone:          {final_map:.4f} ({final_map - 0.7322:+.4f})", flush=True)
print(f"  E11 old stack:           0.7396", flush=True)
print(f"  E15 new stack:           {stack_map:.4f} ({stack_map - 0.7396:+.4f})", flush=True)
print(f"  E12 old stack+logit:     0.7451", flush=True)
print(f"  E15 new stack+logit:     {adj_map:.4f} ({adj_map - 0.7451:+.4f})", flush=True)

# Save best
np.save(ROOT / "oof_e15_stack.npy", oof_adj)
np.save(ROOT / "test_e15_stack.npy", test_adj)
save_submission(test_adj, "e15_stack_logitadj", cv_map=adj_map)
print("\nDone!", flush=True)
