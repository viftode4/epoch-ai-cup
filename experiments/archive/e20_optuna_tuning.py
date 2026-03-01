"""E20: Optuna Hyperparameter Tuning for Tree Ensemble

CatBoost controls 80% of the ensemble. Even a small improvement matters.
This script tunes CatBoost, XGBoost, and LightGBM hyperparameters jointly
using Optuna with 5-fold CV macro mAP as the objective.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import torch
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
print("Extracting features...", flush=True)
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

# Fixed fold split for fair comparison
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

# ── Phase 1: Tune CatBoost (80% weight) ─────────────────────────
print(f"\n{'='*60}", flush=True)
print("Phase 1: Tuning CatBoost (80% of ensemble weight)", flush=True)
print(f"{'='*60}", flush=True)

def cb_objective(trial):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 15),
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 80,
        "task_type": "GPU",
    }

    oof = np.zeros((len(X), N_CLASSES))
    for fold, (tr_idx, va_idx) in enumerate(folds):
        cb = CatBoostClassifier(**params)
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
               verbose=0, sample_weight=sample_weights[tr_idx])
        oof[va_idx] = cb.predict_proba(X[va_idx])

    m, _ = compute_map(y, oof)
    return m

cb_study = optuna.create_study(direction="maximize", study_name="catboost")
cb_study.enqueue_trial({  # E15 baseline as first trial
    "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
    "random_strength": 1.0, "bagging_temperature": 1.0, "min_data_in_leaf": 1,
})
cb_study.optimize(cb_objective, n_trials=40, show_progress_bar=False)

print(f"\n  Best CatBoost mAP: {cb_study.best_value:.4f}", flush=True)
print(f"  Best params: {cb_study.best_params}", flush=True)

# ── Phase 2: Tune LightGBM ──────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Phase 2: Tuning LightGBM", flush=True)
print(f"{'='*60}", flush=True)

def lgb_objective(trial):
    params = {
        "objective": "multiclass", "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "verbose": -1, "seed": 42, "n_jobs": -1,
        "device": "gpu",
    }

    oof = np.zeros((len(X), N_CLASSES))
    for fold, (tr_idx, va_idx) in enumerate(folds):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx],
                             feature_name=feature_names)
        dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=feature_names,
                           reference=dtrain)
        mdl = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va_idx] = mdl.predict(X[va_idx])

    m, _ = compute_map(y, oof)
    return m

lgb_study = optuna.create_study(direction="maximize", study_name="lightgbm")
lgb_study.enqueue_trial({  # E15 baseline
    "learning_rate": 0.05, "num_leaves": 47, "max_depth": 7,
    "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
})
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=False)

print(f"\n  Best LGB mAP: {lgb_study.best_value:.4f}", flush=True)
print(f"  Best params: {lgb_study.best_params}", flush=True)

# ── Phase 3: Retrain with best params ────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Phase 3: Retrain with optimized params", flush=True)
print(f"{'='*60}", flush=True)

best_cb_params = cb_study.best_params
best_lgb_params = lgb_study.best_params

oof_lgb = np.zeros((len(X), N_CLASSES))
oof_xgb = np.zeros((len(X), N_CLASSES))
oof_cb = np.zeros((len(X), N_CLASSES))
test_lgb_p = np.zeros((len(X_test), N_CLASSES))
test_xgb_p = np.zeros((len(X_test), N_CLASSES))
test_cb_p = np.zeros((len(X_test), N_CLASSES))

lgb_params_final = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
    **best_lgb_params,
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

for fold, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n--- Fold {fold} ---", flush=True)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    # LGB (optimized)
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    mdl = lgb.train(lgb_params_final, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(X_va)
    test_lgb_p += mdl.predict(X_test) / N_FOLDS

    # XGB (keep E15 params for now)
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                    evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = mdl.predict(dval_xgb)
    test_xgb_p += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

    # CB (optimized)
    cb = CatBoostClassifier(
        iterations=2000,
        learning_rate=best_cb_params["learning_rate"],
        depth=best_cb_params["depth"],
        l2_leaf_reg=best_cb_params["l2_leaf_reg"],
        random_strength=best_cb_params["random_strength"],
        bagging_temperature=best_cb_params["bagging_temperature"],
        min_data_in_leaf=best_cb_params["min_data_in_leaf"],
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb[va_idx] = cb.predict_proba(X_va)
    test_cb_p += cb.predict_proba(X_test) / N_FOLDS

    fold_ens = 0.15 * oof_lgb[va_idx] + 0.05 * oof_xgb[va_idx] + 0.80 * oof_cb[va_idx]
    fold_map, _ = compute_map(y_va, fold_ens)
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

# Optimize ensemble weights
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

print(f"\nBest weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}", flush=True)

oof_tree = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_tree = best_w[0] * test_lgb_p + best_w[1] * test_xgb_p + best_w[2] * test_cb_p

tree_map, tree_per = compute_map(y, oof_tree)
print_results(tree_map, tree_per, "E20 Tuned Tree Ensemble")
print(f"\nE15 tree was: 0.7451", flush=True)
print(f"E20 tree:     {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)

np.save(ROOT / "oof_e20.npy", oof_tree)
np.save(ROOT / "test_e20.npy", test_tree)

# ── Rebuild stack + logit adj ────────────────────────────────────
oof_e08 = np.load(ROOT / "oof_e08.npy")
oof_e06 = np.load(ROOT / "oof_e06.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")
test_e08 = np.load(ROOT / "test_e08.npy")
test_e06 = np.load(ROOT / "test_e06.npy")
test_e09 = np.load(ROOT / "test_e09.npy")

best_stack_map = 0
best_stack_w = None
for w0 in np.arange(0.50, 0.90, 0.05):
    for w1 in np.arange(0.05, 0.25, 0.05):
        for w2 in np.arange(0.05, 0.25, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.05:
                continue
            oof_stack = w0 * oof_tree + w1 * oof_e08 + w2 * oof_e06 + w3 * oof_e09
            m, _ = compute_map(y, oof_stack)
            if m > best_stack_map:
                best_stack_map = m
                best_stack_w = (w0, w1, w2, w3)

print(f"\nBest stack: tree={best_stack_w[0]:.2f} rocket={best_stack_w[1]:.2f} "
      f"cnn={best_stack_w[2]:.2f} svm={best_stack_w[3]:.2f}", flush=True)
print(f"Stack mAP: {best_stack_map:.4f} (E15 stack was 0.7493)", flush=True)

oof_stack = (best_stack_w[0] * oof_tree + best_stack_w[1] * oof_e08 +
             best_stack_w[2] * oof_e06 + best_stack_w[3] * oof_e09)
test_stack = (best_stack_w[0] * test_tree + best_stack_w[1] * test_e08 +
              best_stack_w[2] * test_e06 + best_stack_w[3] * test_e09)

# Logit adjustment
priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = best_stack_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj_ = priors ** (-per_class_tau)
            adjusted = oof_stack * adj_[np.newaxis, :]
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

adj_ = priors ** (-per_class_tau)
oof_adj = oof_stack * adj_[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
test_adj = test_stack * adj_[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, "E20 Stack + Logit Adjustment")

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E15 tree:          0.7451", flush=True)
print(f"  E20 tree (tuned):  {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)
print(f"  E15 stack:         0.7493", flush=True)
print(f"  E20 stack:         {best_stack_map:.4f} ({best_stack_map - 0.7493:+.4f})", flush=True)
print(f"  E15 stack+logit:   0.7535", flush=True)
print(f"  E20 stack+logit:   {adj_map:.4f} ({adj_map - 0.7535:+.4f})", flush=True)

if adj_map > 0.7535:
    np.save(ROOT / "oof_e20_stack.npy", oof_adj)
    np.save(ROOT / "test_e20_stack.npy", test_adj)
    save_submission(test_adj, "e20_optuna", cv_map=adj_map)
    print(f"\nNEW BEST! Saved submission.", flush=True)
else:
    print(f"\nNo improvement over E15.", flush=True)

print("\nDone!", flush=True)
