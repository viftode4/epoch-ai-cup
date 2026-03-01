"""E24: Weak-Class Targeted Features

Test new features designed for Cormorants, BoP, Waders, Geese:
- RCS stability (Cormorants: steady flight)
- Wingbeat proxy (Waders: fast flap, BoP: none)
- Soaring index (BoP: altitude gain while slow)
- Straightness (Cormorants: very straight flight)
- Altitude dynamics (Waders: high variability)
- Speed profile (BoP: slow, variable)
- Migration timing interactions (Geese/Waders: Oct-Nov)

Uses GroupKFold for honest evaluation.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
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
counts = np.bincount(y, minlength=N_CLASSES)

# ── Feature sets to compare ──────────────────────────────────────
configs = {
    "baseline": ["core", "rcs_fft", "tabular", "targeted", "flight_mode"],
    "with_weakclass": ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"],
}

# Model params
lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
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

# Effective Number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(sgkf.split(np.zeros(len(y)), y, groups))

results = {}

for config_name, feat_sets in configs.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name} -- {feat_sets}", flush=True)
    print(f"{'='*60}", flush=True)

    print("Extracting features...", flush=True)
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)
    print(f"  Feature count: {train_feats.shape[1]}", flush=True)

    X = train_feats.values.astype(np.float32)
    X_test = test_feats.values.astype(np.float32)
    feature_names = list(train_feats.columns)
    sample_weights = np.array([class_w[yi] for yi in y])

    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(folds):
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

        # CatBoost
        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

        print(f"  Fold {fold} done", flush=True)

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

    print(f"  Weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}", flush=True)
    print_results(final_map, final_per, f"{config_name} ({train_feats.shape[1]} feats)")

    results[config_name] = {
        "map": final_map, "per_class": final_per, "oof": oof_final,
        "test": test_final, "n_feats": train_feats.shape[1],
    }

# ── Logit adjustment on best config ─────────────────────────────
best_config = max(results.keys(), key=lambda k: results[k]["map"])
print(f"\n{'='*60}", flush=True)
print(f"Logit adjustment on best: {best_config}", flush=True)
print(f"{'='*60}", flush=True)

oof_best = results[best_config]["oof"]
test_best = results[best_config]["test"]

priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = results[best_config]["map"]

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj = priors ** (-per_class_tau)
            adjusted = oof_best * adj[np.newaxis, :]
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
oof_adj = oof_best * adj[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
test_adj = test_best * adj[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, f"E24 {best_config} + Logit Adj")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY (GroupKFold)", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E20 baseline (105 feats):     0.6898 (ref)", flush=True)
for name, r in results.items():
    print(f"  {name:30s} ({r['n_feats']} feats): {r['map']:.4f}", flush=True)
print(f"  + logit adj:                  {adj_map:.4f}", flush=True)

# Per-class comparison
print(f"\nPer-class delta (weakclass - baseline):", flush=True)
if "baseline" in results and "with_weakclass" in results:
    for cls in CLASSES:
        base_ap = results["baseline"]["per_class"][cls]
        new_ap = results["with_weakclass"]["per_class"][cls]
        marker = " ***" if abs(new_ap - base_ap) > 0.01 else ""
        print(f"  {cls:15s}: {base_ap:.4f} -> {new_ap:.4f} ({new_ap - base_ap:+.4f}){marker}",
              flush=True)

# Feature importance for new features
print(f"\nNew feature importance (top from weakclass set):", flush=True)
new_feat_names = [f for f in results.get("with_weakclass", {}).get("per_class", {})
                  if f.startswith(("rcs_cv", "rcs_auto", "rcs_stab", "rcs_n_peaks",
                                   "rcs_zero", "rcs_mean_cross", "soaring", "alt_gain",
                                   "slow_flight", "straight", "turn_angle", "alt_rate",
                                   "alt_accel", "speed_cv", "speed_below", "speed_decel",
                                   "rcs_alt", "rcs_per", "migration", "large_high",
                                   "flock_ind", "size_alt", "solitary", "night", "rcs_for"))]

# Save
np.save(ROOT / "oof_e24.npy", oof_adj)
np.save(ROOT / "test_e24.npy", test_adj)
save_submission(test_adj, "e24_weakclass_features", cv_map=adj_map)
print("\nDone!", flush=True)
