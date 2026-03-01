"""E21: SMOTE on Feature Space + GroupKFold

Apply SMOTE to minority class features within each training fold.
Generates synthetic samples for Cormorants (40), Ducks (58), etc.

SMOTE operates on the 105-dim feature space, NOT on raw time series.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
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
N_FOLDS = 5
BETA = 0.999

# ── Data ─────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
groups = train_df["primary_observation_id"].values

FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print("Extracting features...", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

counts = np.bincount(y, minlength=N_CLASSES)

# Effective Number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

# Model params
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


def train_with_smote(min_target, label):
    """Train LGB+XGB+CB with SMOTE oversampling in each fold."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Apply SMOTE to training fold only
        tr_counts = np.bincount(y_tr, minlength=N_CLASSES)
        strategy = {}
        for c in range(N_CLASSES):
            if tr_counts[c] < min_target:
                strategy[c] = min_target

        if strategy:
            # k_neighbors must be < smallest class size
            min_class_size = tr_counts[tr_counts > 0].min()
            k = min(5, min_class_size - 1)
            k = max(k, 1)
            smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        # Recompute sample weights for SMOTE'd data
        w_tr = np.array([class_w[yi] for yi in y_tr_sm])

        n_original = len(X_tr)
        n_smoted = len(X_tr_sm)
        print(f"  Fold {fold}: {n_original} -> {n_smoted} (+{n_smoted - n_original} synthetic), "
              f"val={len(va_idx)}", flush=True)

        # LightGBM
        dtrain = lgb.Dataset(X_tr_sm, label=y_tr_sm, weight=w_tr, feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
        mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = mdl.predict(X_va)
        test_lgb += mdl.predict(X_test) / N_FOLDS

        # XGBoost
        dtrain_xgb = xgb.DMatrix(X_tr_sm, label=y_tr_sm, weight=w_tr, feature_names=feature_names)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
        mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
        oof_xgb[va_idx] = mdl.predict(dval_xgb)
        test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

        # CatBoost
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr_sm, y_tr_sm, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Optimize weights
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

    oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    final_map, final_per = compute_map(y, oof_final)

    print(f"\n  {label}: mAP={final_map:.4f} (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f})",
          flush=True)
    return oof_final, test_final, final_map, final_per


# ── Sweep SMOTE targets ─────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SMOTE sweep: varying minimum class target", flush=True)
print("=" * 60, flush=True)

results = []
for min_target in [100, 150, 200, 300]:
    print(f"\n--- min_target={min_target} ---", flush=True)
    oof, test, m, per = train_with_smote(min_target, f"SMOTE-{min_target}")
    results.append((min_target, m, per, oof, test))

# Pick best
results.sort(key=lambda x: -x[1])
best_target, best_map, best_per, best_oof, best_test = results[0]

print(f"\n{'='*60}", flush=True)
print("COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E20 GroupKFold (no SMOTE, ref):   load from oof_e20.npy", flush=True)
for target, m, per, _, _ in sorted(results, key=lambda x: x[0]):
    print(f"  SMOTE min={target:3d}:                 {m:.4f}", flush=True)

print(f"\nBest: min_target={best_target} ({best_map:.4f})", flush=True)
print_results(best_map, best_per, f"E21 Best (SMOTE-{best_target})")

# ── Logit adjustment ────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Logit adjustment on best SMOTE", flush=True)
print("=" * 60, flush=True)

priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = best_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj = priors ** (-per_class_tau)
            adjusted = best_oof * adj[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c
        per_class_tau[c] = best_c_tau
        if best_c_map > current_best:
            current_best = best_c_map
            improved = True
    print(f"  Round {iteration + 1}: mAP={current_best:.4f}", flush=True)
    if not improved:
        break

adj = priors ** (-per_class_tau)
oof_adj = best_oof * adj[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
test_adj = best_test * adj[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, f"E21 SMOTE-{best_target} + Logit Adj")

np.save(ROOT / "oof_e21.npy", oof_adj)
np.save(ROOT / "test_e21.npy", test_adj)
save_submission(test_adj, f"e21_smote_{best_target}", cv_map=adj_map)
print("\nDone!", flush=True)
