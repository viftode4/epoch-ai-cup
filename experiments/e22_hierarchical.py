"""E22: Hierarchical Classification

Stage 1: Binary Gull vs Non-Gull classifier
Stage 2: 8-class classifier on non-Gull data only

Final: P(Gull) from stage 1, P(class_i) = P(NonGull) * P(class_i|NonGull)

This directly attacks the "Gulls black hole" where 33-48% of minority class
predictions get pulled toward Gulls by the strong majority prior.
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
GULL_IDX = CLASSES.index("Gulls")  # = 5

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

# Binary labels: Gull=1, NonGull=0
y_binary = (y == GULL_IDX).astype(int)

# 8-class labels: remap non-Gull classes to 0-7
# Classes: BoP(0), Clutter(1), Cormorants(2), Ducks(3), Geese(4), [Gulls(5)], Pigeons(6), Songbirds(7), Waders(8)
# Remapped: BoP(0), Clutter(1), Cormorants(2), Ducks(3), Geese(4), Pigeons(5), Songbirds(6), Waders(7)
non_gull_mask = y != GULL_IDX
remap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7}  # skip 5 (Gulls)
unmap = {v: k for k, v in remap.items()}  # 8-class idx -> original 9-class idx
y_8class = np.array([remap[yi] for yi in y[non_gull_mask]])
N_8CLASS = 8

non_gull_classes = [c for i, c in enumerate(CLASSES) if i != GULL_IDX]
print(f"Non-Gull classes ({N_8CLASS}): {non_gull_classes}", flush=True)
print(f"Non-Gull distribution: {np.bincount(y_8class, minlength=N_8CLASS).tolist()}", flush=True)

# Effective number weights for 8-class
counts_8 = np.bincount(y_8class, minlength=N_8CLASS)
effective_n_8 = 1.0 - np.power(BETA, counts_8)
class_w_8 = (1.0 - BETA) / (effective_n_8 + 1e-10)
class_w_8 = class_w_8 / class_w_8.mean()

# Model params
lgb_binary_params = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 47, "max_depth": 7,
    "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

lgb_multi_params = {
    "objective": "multiclass", "num_class": N_8CLASS,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

xgb_binary_params = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

xgb_multi_params = {
    "objective": "multi:softprob", "num_class": N_8CLASS,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

# ── GroupKFold splits ────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(sgkf.split(X, y, groups))

# ── Stage 1: Binary Gull vs Non-Gull ────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Stage 1: Binary Gull vs Non-Gull", flush=True)
print("=" * 60, flush=True)

oof_binary = np.zeros(len(X))  # P(Gull)
test_binary = np.zeros(len(X_test))

# Use inverse-frequency weights for binary
n_gull = (y_binary == 1).sum()
n_nongull = (y_binary == 0).sum()
binary_weight = np.where(y_binary == 1, len(y) / (2 * n_gull), len(y) / (2 * n_nongull))

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    yb_tr, yb_va = y_binary[tr_idx], y_binary[va_idx]
    wb_tr = binary_weight[tr_idx]

    # LGB binary
    dtrain = lgb.Dataset(X_tr, label=yb_tr, weight=wb_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=yb_va, feature_name=feature_names, reference=dtrain)
    m_lgb = lgb.train(lgb_binary_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_va)
    t_lgb = m_lgb.predict(X_test)

    # XGB binary
    dtrain_xgb = xgb.DMatrix(X_tr, label=yb_tr, weight=wb_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=yb_va, feature_names=feature_names)
    m_xgb = xgb.train(xgb_binary_params, dtrain_xgb, num_boost_round=2000,
                       evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    p_xgb = m_xgb.predict(dval_xgb)
    t_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=feature_names))

    # CB binary
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="Logloss", eval_metric="Logloss",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU", auto_class_weights="Balanced",
    )
    cb.fit(X_tr, yb_tr, eval_set=(X_va, yb_va), verbose=0)
    p_cb = cb.predict_proba(X_va)[:, 1]
    t_cb = cb.predict_proba(X_test)[:, 1]

    # Ensemble binary (80% CB, 15% LGB, 5% XGB like E15)
    oof_binary[va_idx] = 0.15 * p_lgb + 0.05 * p_xgb + 0.80 * p_cb
    test_binary += (0.15 * t_lgb + 0.05 * t_xgb + 0.80 * t_cb) / N_FOLDS

    acc = ((oof_binary[va_idx] > 0.5) == yb_va).mean()
    print(f"  Fold {fold}: binary acc={acc:.4f}", flush=True)

binary_acc = ((oof_binary > 0.5) == y_binary).mean()
print(f"\n  Overall binary accuracy: {binary_acc:.4f}", flush=True)
print(f"  Gull recall: {((oof_binary > 0.5) & (y_binary == 1)).sum() / n_gull:.4f}", flush=True)
print(f"  NonGull recall: {((oof_binary <= 0.5) & (y_binary == 0)).sum() / n_nongull:.4f}", flush=True)

# ── Stage 2: 8-class on Non-Gull data ───────────────────────────
print("\n" + "=" * 60, flush=True)
print("Stage 2: 8-class on Non-Gull data", flush=True)
print("=" * 60, flush=True)

oof_8class = np.zeros((len(X), N_8CLASS))  # P(class|NonGull) for ALL samples
test_8class = np.zeros((len(X_test), N_8CLASS))

for fold, (tr_idx, va_idx) in enumerate(folds):
    # Training: only non-Gull samples
    tr_nongull = tr_idx[y[tr_idx] != GULL_IDX]
    X_tr_ng = X[tr_nongull]
    y_tr_ng = np.array([remap[yi] for yi in y[tr_nongull]])
    w_tr_ng = np.array([class_w_8[yi] for yi in y_tr_ng])

    # Validation: ALL samples (we need predictions for everyone)
    X_va = X[va_idx]

    print(f"  Fold {fold}: train_nongull={len(tr_nongull)}, val_all={len(va_idx)}", flush=True)

    # LGB 8-class
    dtrain = lgb.Dataset(X_tr_ng, label=y_tr_ng, weight=w_tr_ng, feature_name=feature_names)
    # For early stopping, use non-Gull validation samples
    va_nongull = va_idx[y[va_idx] != GULL_IDX]
    y_va_ng = np.array([remap[yi] for yi in y[va_nongull]])
    dval = lgb.Dataset(X[va_nongull], label=y_va_ng, feature_name=feature_names, reference=dtrain)

    m_lgb = lgb.train(lgb_multi_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    p_lgb = m_lgb.predict(X_va)
    t_lgb = m_lgb.predict(X_test)

    # XGB 8-class
    dtrain_xgb = xgb.DMatrix(X_tr_ng, label=y_tr_ng, weight=w_tr_ng, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X[va_nongull], label=y_va_ng, feature_names=feature_names)
    m_xgb = xgb.train(xgb_multi_params, dtrain_xgb, num_boost_round=2000,
                       evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    p_xgb = m_xgb.predict(xgb.DMatrix(X_va, feature_names=feature_names))
    t_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=feature_names))

    # CB 8-class
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr_ng, y_tr_ng, eval_set=(X[va_nongull], y_va_ng), verbose=0, sample_weight=w_tr_ng)
    p_cb = cb.predict_proba(X_va)
    t_cb = cb.predict_proba(X_test)

    # Ensemble 8-class
    oof_8class[va_idx] = 0.15 * p_lgb + 0.05 * p_xgb + 0.80 * p_cb
    test_8class += (0.15 * t_lgb + 0.05 * t_xgb + 0.80 * t_cb) / N_FOLDS

# ── Combine: hierarchical probabilities ──────────────────────────
print("\n" + "=" * 60, flush=True)
print("Combining hierarchical predictions", flush=True)
print("=" * 60, flush=True)

oof_hier = np.zeros((len(X), N_CLASSES))
test_hier = np.zeros((len(X_test), N_CLASSES))

# P(Gull) = binary prediction
oof_hier[:, GULL_IDX] = oof_binary
test_hier[:, GULL_IDX] = test_binary

# P(class_i) = P(NonGull) * P(class_i | NonGull)
for new_idx, orig_idx in unmap.items():
    oof_hier[:, orig_idx] = (1 - oof_binary) * oof_8class[:, new_idx]
    test_hier[:, orig_idx] = (1 - test_binary) * test_8class[:, new_idx]

# Normalize
oof_hier = oof_hier / oof_hier.sum(axis=1, keepdims=True)
test_hier = test_hier / test_hier.sum(axis=1, keepdims=True)

hier_map, hier_per = compute_map(y, oof_hier)
print_results(hier_map, hier_per, "E22 Hierarchical (raw)")

# ── Also try blending hierarchical with flat model ───────────────
print("\n" + "=" * 60, flush=True)
print("Blending hierarchical with flat 9-class (E20)", flush=True)
print("=" * 60, flush=True)

# Load E20 GroupKFold OOF if available, else try E15
try:
    oof_flat = np.load(ROOT / "oof_e20.npy")
    flat_label = "E20"
except FileNotFoundError:
    oof_flat = np.load(ROOT / "oof_e15.npy")
    flat_label = "E15"
print(f"  Using {flat_label} as flat model", flush=True)

best_blend_map = 0
best_alpha = 0
for alpha in np.arange(0.0, 1.01, 0.05):
    oof_blend = alpha * oof_hier + (1 - alpha) * oof_flat
    m, _ = compute_map(y, oof_blend)
    if m > best_blend_map:
        best_blend_map = m
        best_alpha = alpha

oof_blend = best_alpha * oof_hier + (1 - best_alpha) * oof_flat
blend_map, blend_per = compute_map(y, oof_blend)
print(f"  Best blend: alpha={best_alpha:.2f} (hier={best_alpha:.0%}, flat={1-best_alpha:.0%})",
      flush=True)
print_results(blend_map, blend_per, f"E22 Blend (hier={best_alpha:.0%} + {flat_label})")

# ── Logit adjustment ────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Logit adjustment on best result", flush=True)
print("=" * 60, flush=True)

# Pick whichever is better: pure hierarchical or blend
if blend_map > hier_map:
    oof_best = oof_blend
    base_label = "blend"
    base_map = blend_map
else:
    oof_best = oof_hier
    base_label = "hierarchical"
    base_map = hier_map

priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = base_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj_factor = priors ** (-per_class_tau)
            adjusted = oof_best * adj_factor[np.newaxis, :]
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

adj_factor = priors ** (-per_class_tau)
oof_adj = oof_best * adj_factor[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, f"E22 {base_label} + Logit Adj")

# Test predictions
if base_label == "blend":
    try:
        test_flat = np.load(ROOT / "test_e20.npy")
    except FileNotFoundError:
        test_flat = np.load(ROOT / "test_e15.npy")
    test_best = best_alpha * test_hier + (1 - best_alpha) * test_flat
else:
    test_best = test_hier
test_adj = test_best * adj_factor[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  E15 tree+logit (StratifiedKFold): 0.7535", flush=True)
print(f"  E22 hierarchical (raw):           {hier_map:.4f}", flush=True)
print(f"  E22 blend ({base_label}):         {blend_map:.4f}", flush=True)
print(f"  E22 + logit adj:                  {adj_map:.4f}", flush=True)

np.save(ROOT / "oof_e22.npy", oof_adj)
np.save(ROOT / "test_e22.npy", test_adj)
save_submission(test_adj, "e22_hierarchical", cv_map=adj_map)
print("\nDone!", flush=True)
