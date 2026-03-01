"""E17: Pseudo-Labeling with Multi-Model Agreement (T20-T23)

Uses 4 diverse model families to identify high-confidence test samples:
- T21: Multi-model agreement filter (3+/4 models agree on top-1 class)
- T23: Per-class adaptive confidence thresholds
- T22: Soft probability labels (not hard labels)
- T20: Distribution alignment (cap pseudo-labels to ~2x class prior)

Retrains tree ensemble (E15-style, beta=0.999) on original + pseudo-labeled data.
Rebuilds 4-model stack + logit adjustment.
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
PSEUDO_WEIGHT = 0.5  # pseudo-labeled samples get 50% weight vs real samples

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

X_orig = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# ── Step 1: Analyze multi-model agreement on test set (T21) ──────
print("\n" + "="*60, flush=True)
print("Step 1: Multi-model agreement analysis", flush=True)
print("="*60, flush=True)

# Load test predictions from 4 diverse model families
test_trees = np.load(ROOT / "test_e15.npy")      # Tree ensemble (best, 0.7451)
test_cnn   = np.load(ROOT / "test_e06.npy")       # 1D-CNN (0.5238)
test_rocket = np.load(ROOT / "test_e08.npy")      # MiniRocket (0.4799)
test_svm   = np.load(ROOT / "test_e09.npy")       # SVM (0.5238)

# Also load the best stacked predictions for soft labels
test_stack = np.load(ROOT / "test_e15_stack.npy")  # Stacked + logit adj (0.7535)

# Top-1 predictions from each model
pred_trees = test_trees.argmax(axis=1)
pred_cnn = test_cnn.argmax(axis=1)
pred_rocket = test_rocket.argmax(axis=1)
pred_svm = test_svm.argmax(axis=1)

# Count agreements
all_preds = np.stack([pred_trees, pred_cnn, pred_rocket, pred_svm], axis=1)  # (1872, 4)

agreement_counts = np.zeros(len(test_stack), dtype=int)
for i in range(len(test_stack)):
    # Count how many models agree with the tree ensemble (strongest model)
    agreement_counts[i] = np.sum(all_preds[i] == pred_trees[i])

print(f"  4/4 agree: {np.sum(agreement_counts == 4)} samples ({np.sum(agreement_counts == 4)/len(test_stack)*100:.1f}%)", flush=True)
print(f"  3/4 agree: {np.sum(agreement_counts >= 3)} samples ({np.sum(agreement_counts >= 3)/len(test_stack)*100:.1f}%)", flush=True)
print(f"  2/4 agree: {np.sum(agreement_counts >= 2)} samples ({np.sum(agreement_counts >= 2)/len(test_stack)*100:.1f}%)", flush=True)

# Per-class agreement breakdown
print("\n  Per-class 4/4 agreement:", flush=True)
for c in range(N_CLASSES):
    mask = (pred_trees == c) & (agreement_counts == 4)
    print(f"    {CLASSES[c]:15s}: {mask.sum():4d} samples", flush=True)

# ── Step 2: Per-class adaptive thresholds (T23) ─────────────────
print("\n" + "="*60, flush=True)
print("Step 2: Per-class adaptive confidence thresholds", flush=True)
print("="*60, flush=True)

# Use stacked predictions for confidence (best calibrated)
max_probs = test_stack.max(axis=1)

# Set thresholds based on class frequency:
# - Majority classes (Gulls, Songbirds): high threshold (0.90+)
# - Medium classes: moderate (0.80)
# - Minority classes (Cormorants, Ducks): lower (0.70) to get enough pseudo-labels
counts_train = np.bincount(y, minlength=N_CLASSES)
thresholds = np.zeros(N_CLASSES)
for c in range(N_CLASSES):
    if counts_train[c] >= 500:   # Gulls, Songbirds
        thresholds[c] = 0.90
    elif counts_train[c] >= 100:  # BoP, Pigeons, Waders
        thresholds[c] = 0.80
    elif counts_train[c] >= 70:   # Clutter, Geese
        thresholds[c] = 0.75
    else:                          # Cormorants (40), Ducks (58)
        thresholds[c] = 0.65

print("  Adaptive thresholds:", flush=True)
for c in range(N_CLASSES):
    print(f"    {CLASSES[c]:15s}: thresh={thresholds[c]:.2f} (n_train={counts_train[c]})", flush=True)

# ── Step 3: Filter pseudo-labels (T21 + T23 combined) ───────────
print("\n" + "="*60, flush=True)
print("Step 3: Selecting pseudo-labeled samples", flush=True)
print("="*60, flush=True)

# DARP-style distribution cap (T20): don't allow more than 2x the original
# class proportion in pseudo-labels
train_prior = counts_train / counts_train.sum()
max_pseudo_per_class = (2.0 * train_prior * len(test_stack)).astype(int)

# Selection criteria: agreement >= 3 AND max_prob > per-class threshold
pseudo_mask = np.zeros(len(test_stack), dtype=bool)
pseudo_labels_hard = pred_trees.copy()  # hard labels for stratification
pseudo_selected_per_class = np.zeros(N_CLASSES, dtype=int)

for c in range(N_CLASSES):
    # Candidates: tree predicts class c, 3+ models agree, above threshold
    candidates = (
        (pred_trees == c) &
        (agreement_counts >= 3) &
        (max_probs > thresholds[c])
    )
    candidate_indices = np.where(candidates)[0]

    # Sort by confidence (highest first) and cap at max_pseudo_per_class
    if len(candidate_indices) > 0:
        confidences = max_probs[candidate_indices]
        sorted_idx = candidate_indices[np.argsort(-confidences)]
        n_select = min(len(sorted_idx), max_pseudo_per_class[c])
        pseudo_mask[sorted_idx[:n_select]] = True
        pseudo_selected_per_class[c] = n_select

n_pseudo = pseudo_mask.sum()
print(f"\n  Total pseudo-labeled: {n_pseudo} / {len(test_stack)} ({n_pseudo/len(test_stack)*100:.1f}%)", flush=True)
print(f"  Training expands: {len(y)} -> {len(y) + n_pseudo} ({n_pseudo/len(y)*100:.1f}% increase)", flush=True)
print("\n  Per-class pseudo-labels selected:", flush=True)
for c in range(N_CLASSES):
    orig = counts_train[c]
    pseudo = pseudo_selected_per_class[c]
    print(f"    {CLASSES[c]:15s}: +{pseudo:4d} (was {orig:4d}, now {orig+pseudo:4d}, +{pseudo/max(orig,1)*100:.0f}%)", flush=True)

# ── Step 4: Prepare expanded training set ────────────────────────
print("\n" + "="*60, flush=True)
print("Step 4: Preparing expanded training set", flush=True)
print("="*60, flush=True)

# Use SOFT labels (T22) from stacked model for pseudo-labeled samples
pseudo_soft_labels = test_stack[pseudo_mask]  # (n_pseudo, 9) probability vectors
pseudo_hard_labels = pred_trees[pseudo_mask]   # for stratification
X_pseudo = X_test[pseudo_mask]

# Combine original + pseudo features
X_combined = np.vstack([X_orig, X_pseudo])
y_combined_hard = np.concatenate([y, pseudo_hard_labels])

print(f"  Combined X: {X_combined.shape}", flush=True)
print(f"  Combined y: {y_combined_hard.shape}", flush=True)

# ── Step 5: Effective Number weights for original samples ────────
# Original samples: beta=0.999 effective number weights
# Pseudo samples: flat weight * PSEUDO_WEIGHT multiplier
counts_combined = np.bincount(y_combined_hard, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts_train)  # weights based on ORIGINAL counts
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

# Sample weights: original get full effective number weight, pseudo get reduced
sample_weights = np.zeros(len(X_combined))
for i in range(len(X_orig)):
    sample_weights[i] = class_w[y[i]]
for i in range(len(X_pseudo)):
    sample_weights[len(X_orig) + i] = class_w[pseudo_hard_labels[i]] * PSEUDO_WEIGHT

print(f"  Original sample weights: mean={sample_weights[:len(X_orig)].mean():.3f}", flush=True)
print(f"  Pseudo sample weights:   mean={sample_weights[len(X_orig):].mean():.3f}", flush=True)

# ── Step 6: Train tree ensemble on expanded data ─────────────────
print("\n" + "="*60, flush=True)
print("Step 6: Training tree ensemble on expanded data", flush=True)
print("="*60, flush=True)

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

# CV on ORIGINAL data only (pseudo-labels only in training, never in validation)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X_orig), N_CLASSES))
oof_xgb = np.zeros((len(X_orig), N_CLASSES))
oof_cb = np.zeros((len(X_orig), N_CLASSES))
test_lgb = np.zeros((len(X_test), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))
test_cb = np.zeros((len(X_test), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_orig, y)):
    print(f"\n--- Fold {fold} ---", flush=True)

    # Training: original fold training + ALL pseudo-labeled samples
    X_tr_orig = X_orig[tr_idx]
    y_tr_orig = y[tr_idx]
    w_tr_orig = sample_weights[tr_idx]

    X_tr = np.vstack([X_tr_orig, X_pseudo])
    y_tr = np.concatenate([y_tr_orig, pseudo_hard_labels])
    w_tr = np.concatenate([w_tr_orig, sample_weights[len(X_orig):]])

    # Validation: ONLY original samples (clean evaluation)
    X_va = X_orig[va_idx]
    y_va = y[va_idx]

    print(f"  Train: {len(X_tr)} ({len(X_tr_orig)} orig + {len(X_pseudo)} pseudo)", flush=True)

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

    # CatBoost (GPU)
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
    oof_ens = 0.15 * oof_lgb[va_idx] + 0.05 * oof_xgb[va_idx] + 0.80 * oof_cb[va_idx]
    fold_map, _ = compute_map(y_va, oof_ens)
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

# ── Step 7: Optimize ensemble weights ────────────────────────────
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

oof_tree = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_tree = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

tree_map, tree_per = compute_map(y, oof_tree)
print_results(tree_map, tree_per, "E17 Tree Ensemble (pseudo-labeled)")
print(f"\nE15 tree was: 0.7451", flush=True)
print(f"E17 tree:     {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)

np.save(ROOT / "oof_e17.npy", oof_tree)
np.save(ROOT / "test_e17.npy", test_tree)

# ── Step 8: Rebuild 4-model stack ────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Step 8: Rebuilding 4-model stack with pseudo-labeled trees", flush=True)
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
print(f"Stack mAP: {best_stack_map:.4f} (E15 stack was 0.7493, delta={best_stack_map - 0.7493:+.4f})",
      flush=True)

oof_stack = (best_stack_w[0] * oof_tree + best_stack_w[1] * oof_e08 +
             best_stack_w[2] * oof_e06 + best_stack_w[3] * oof_e09)
test_stack_new = (best_stack_w[0] * test_tree + best_stack_w[1] * test_e08 +
                  best_stack_w[2] * test_e06 + best_stack_w[3] * test_e09)

stack_map, stack_per = compute_map(y, oof_stack)
print_results(stack_map, stack_per, "E17 Stack (pseudo-labeled trees)")

# ── Step 9: Logit adjustment (T08) ───────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Step 9: Per-class logit adjustment", flush=True)
print(f"{'='*60}", flush=True)

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
test_adj = test_stack_new * adj[np.newaxis, :]
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, "E17 Stack + Logit Adjustment")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Pseudo-labeled samples:     {n_pseudo} / {len(test_stack)} test", flush=True)
print(f"  E15 tree alone:             0.7451", flush=True)
print(f"  E17 tree (pseudo):          {tree_map:.4f} ({tree_map - 0.7451:+.4f})", flush=True)
print(f"  E15 stack:                  0.7493", flush=True)
print(f"  E17 stack (pseudo):         {stack_map:.4f} ({stack_map - 0.7493:+.4f})", flush=True)
print(f"  E15 stack+logit (best):     0.7535", flush=True)
print(f"  E17 stack+logit (pseudo):   {adj_map:.4f} ({adj_map - 0.7535:+.4f})", flush=True)

# Save
np.save(ROOT / "oof_e17_stack.npy", oof_adj)
np.save(ROOT / "test_e17_stack.npy", test_adj)
save_submission(test_adj, "e17_pseudo_labeling", cv_map=adj_map)

print("\nDone!", flush=True)
