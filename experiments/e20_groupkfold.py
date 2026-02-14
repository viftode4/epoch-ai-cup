"""E20: GroupKFold Honest Baseline

Re-measure tree ensemble with StratifiedGroupKFold on primary_observation_id.
Tracks from the same observation event stay in the same fold (no leakage).

Compare to E15 tree (0.7451 with StratifiedKFold).
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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

FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print("Extracting features...", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)
print(f"  Feature count: {train_feats.shape[1]}", flush=True)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# Effective Number weights
counts = np.bincount(y, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

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


def train_ensemble(X, y, X_test, sample_weights, splitter, label):
    """Train LGB+XGB+CB with given CV splitter."""
    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(splitter):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx]

        print(f"  Fold {fold}: train={len(tr_idx)}, val={len(va_idx)}, "
              f"val_classes={np.bincount(y_va, minlength=N_CLASSES).tolist()}", flush=True)

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


# ── A) StratifiedKFold (E15 reproduction) ────────────────────────
print("\n" + "=" * 60, flush=True)
print("A) StratifiedKFold (E15 reproduction)", flush=True)
print("=" * 60, flush=True)

skf_splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(X, y))
oof_skf, test_skf, map_skf, per_skf = train_ensemble(
    X, y, X_test, sample_weights, skf_splits, "StratifiedKFold")
print_results(map_skf, per_skf, "E15 Repro (StratifiedKFold)")

# ── B) StratifiedGroupKFold ──────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("B) StratifiedGroupKFold on primary_observation_id", flush=True)
print("=" * 60, flush=True)

# Show group stats
from collections import Counter
group_sizes = Counter(groups)
multi = {g: c for g, c in group_sizes.items() if c > 1}
print(f"  Total groups: {len(group_sizes)}", flush=True)
print(f"  Multi-track groups: {len(multi)} ({sum(multi.values())} tracks)", flush=True)

sgkf_splits = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(X, y, groups))
oof_gkf, test_gkf, map_gkf, per_gkf = train_ensemble(
    X, y, X_test, sample_weights, sgkf_splits, "GroupKFold")
print_results(map_gkf, per_gkf, "E20 GroupKFold")

# ── Logit adjustment on GroupKFold ───────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Logit adjustment on GroupKFold predictions", flush=True)
print("=" * 60, flush=True)

priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = map_gkf

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj = priors ** (-per_class_tau)
            adjusted = oof_gkf * adj[np.newaxis, :]
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
oof_adj = oof_gkf * adj[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, "E20 GroupKFold + Logit Adj")

# Apply same tau to test
test_adj = test_gkf * adj[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SUMMARY: Impact of GroupKFold", flush=True)
print("=" * 60, flush=True)
print(f"  E15 tree (StratifiedKFold, ref): 0.7451", flush=True)
print(f"  E15 repro (StratifiedKFold):     {map_skf:.4f}", flush=True)
print(f"  E20 tree (GroupKFold):           {map_gkf:.4f} ({map_gkf - map_skf:+.4f} vs SKF)", flush=True)
print(f"  E20 + logit adj:                 {adj_map:.4f} ({adj_map - map_gkf:+.4f} from logit adj)", flush=True)
print(f"\nPer-class comparison (GroupKFold vs StratifiedKFold):", flush=True)
for cls in CLASSES:
    gkf_ap = per_gkf[cls]
    skf_ap = per_skf[cls]
    print(f"  {cls:15s}: GKF={gkf_ap:.4f}  SKF={skf_ap:.4f}  delta={gkf_ap - skf_ap:+.4f}", flush=True)

# Save
np.save(ROOT / "oof_e20.npy", oof_adj)
np.save(ROOT / "test_e20.npy", test_adj)
save_submission(test_adj, "e20_groupkfold", cv_map=adj_map)
print("\nDone!", flush=True)
