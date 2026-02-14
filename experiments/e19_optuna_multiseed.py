"""E19: Optuna CatBoost Tuning + Multi-Seed Averaging

CatBoost is 80% of our ensemble. Tuning it = direct impact on final score.
Then average over 3 random seeds for stability.
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
SEEDS = [42, 123, 777]

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

# ── Phase 1: Optuna CatBoost Tuning ─────────────────────────────
print("\n" + "="*60, flush=True)
print("Phase 1: Optuna CatBoost hyperparameter tuning", flush=True)
print("="*60, flush=True)

def objective(trial):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 80,
        "task_type": "GPU",
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        cb = CatBoostClassifier(**params)
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
               verbose=0, sample_weight=sample_weights[tr_idx])
        oof[va_idx] = cb.predict_proba(X[va_idx])

    m, _ = compute_map(y, oof)
    return m

def print_callback(study, trial):
    print(f"  Trial {trial.number}: mAP={trial.value:.4f} "
          f"(best so far: {study.best_value:.4f})", flush=True)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, callbacks=[print_callback])

print(f"\nBest CatBoost mAP: {study.best_value:.4f}", flush=True)
print(f"Best params:", flush=True)
for k, v in study.best_params.items():
    print(f"  {k}: {v}", flush=True)

# ── Phase 2: Train full ensemble with best CB params, multi-seed ─
print("\n" + "="*60, flush=True)
print("Phase 2: Full ensemble with tuned CB, multi-seed averaging", flush=True)
print("="*60, flush=True)

best_cb_params = study.best_params

lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "n_jobs": -1,
    "device": "gpu",
}

xgb_params = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

# Accumulate over seeds
oof_lgb_all = np.zeros((len(X), N_CLASSES))
oof_xgb_all = np.zeros((len(X), N_CLASSES))
oof_cb_all = np.zeros((len(X), N_CLASSES))
test_lgb_all = np.zeros((len(X_test), N_CLASSES))
test_xgb_all = np.zeros((len(X_test), N_CLASSES))
test_cb_all = np.zeros((len(X_test), N_CLASSES))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---", flush=True)

    lgb_params["seed"] = seed
    xgb_params["seed"] = seed

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
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

        # CatBoost (tuned)
        cb = CatBoostClassifier(
            iterations=2000,
            learning_rate=best_cb_params["learning_rate"],
            depth=best_cb_params["depth"],
            l2_leaf_reg=best_cb_params["l2_leaf_reg"],
            random_strength=best_cb_params["random_strength"],
            bagging_temperature=best_cb_params["bagging_temperature"],
            border_count=best_cb_params["border_count"],
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=seed, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    oof_lgb_all += oof_lgb / len(SEEDS)
    oof_xgb_all += oof_xgb / len(SEEDS)
    oof_cb_all += oof_cb / len(SEEDS)
    test_lgb_all += test_lgb / len(SEEDS)
    test_xgb_all += test_xgb / len(SEEDS)
    test_cb_all += test_cb / len(SEEDS)

    # Per-seed score
    oof_ens = 0.15 * oof_lgb + 0.05 * oof_xgb + 0.80 * oof_cb
    seed_map, _ = compute_map(y, oof_ens)
    print(f"  Seed {seed} tree mAP: {seed_map:.4f}", flush=True)

# ── Optimize ensemble weights on multi-seed average ──────────────
print("\nOptimizing ensemble weights...", flush=True)
best_map = 0
best_w = None
for w1 in np.arange(0.05, 0.50, 0.05):
    for w2 in np.arange(0.05, 0.50, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        oof_ens = w1 * oof_lgb_all + w2 * oof_xgb_all + w3 * oof_cb_all
        m, _ = compute_map(y, oof_ens)
        if m > best_map:
            best_map = m
            best_w = (w1, w2, w3)

print(f"Best weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}", flush=True)

oof_tree = best_w[0] * oof_lgb_all + best_w[1] * oof_xgb_all + best_w[2] * oof_cb_all
test_tree = best_w[0] * test_lgb_all + best_w[1] * test_xgb_all + best_w[2] * test_cb_all

tree_map, tree_per = compute_map(y, oof_tree)
print_results(tree_map, tree_per, "E19 Tree (tuned CB + 3-seed avg)")
print(f"\nE15 tree was: 0.7451", flush=True)
print(f"E19 tree:     {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)

np.save(ROOT / "oof_e19.npy", oof_tree)
np.save(ROOT / "test_e19.npy", test_tree)

# ── Rebuild stack ────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Rebuilding 4-model stack", flush=True)
print(f"{'='*60}", flush=True)

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

print(f"Best stack: tree={best_stack_w[0]:.2f} rocket={best_stack_w[1]:.2f} "
      f"cnn={best_stack_w[2]:.2f} svm={best_stack_w[3]:.2f}", flush=True)

oof_stack = (best_stack_w[0] * oof_tree + best_stack_w[1] * oof_e08 +
             best_stack_w[2] * oof_e06 + best_stack_w[3] * oof_e09)
test_stack = (best_stack_w[0] * test_tree + best_stack_w[1] * test_e08 +
              best_stack_w[2] * test_e06 + best_stack_w[3] * test_e09)

stack_map, stack_per = compute_map(y, oof_stack)
print_results(stack_map, stack_per, "E19 Stack")

# ── Logit adjustment ────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Per-class logit adjustment", flush=True)
print(f"{'='*60}", flush=True)

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
print_results(adj_map, adj_per, "E19 Stack + Logit Adjustment")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Optuna best CB standalone: {study.best_value:.4f}", flush=True)
print(f"  E15 tree (1 seed):         0.7451", flush=True)
print(f"  E19 tree (tuned+3seed):    {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)
print(f"  E15 stack:                 0.7493", flush=True)
print(f"  E19 stack:                 {stack_map:.4f} ({stack_map - 0.7493:+.4f})", flush=True)
print(f"  E15 stack+logit:           0.7535", flush=True)
print(f"  E19 stack+logit:           {adj_map:.4f} ({adj_map - 0.7535:+.4f})", flush=True)

np.save(ROOT / "oof_e19_stack.npy", oof_adj)
np.save(ROOT / "test_e19_stack.npy", test_adj)
save_submission(test_adj, "e19_optuna_multiseed", cv_map=adj_map)

print("\nDone!", flush=True)
