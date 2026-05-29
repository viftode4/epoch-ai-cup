"""E206: CatBoost on ALL 316 v3 features — no selection.

Hypothesis: Feature dilution finding was validated on LOMO which anti-correlates
with private LB. Winning team used ~100 features. Our stability selection may have
been too aggressive. CatBoost handles many features well with built-in L2 reg.

Train CatBoost on all 316 cached v3 features, save OOF + test predictions,
then blend into e188 pool.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, CLASSES, load_train, load_test
from src.metrics import compute_map, print_results
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

# ---- Load data + cached v3 features ----
print("Loading data...")
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train_df["primary_observation_id"].values

train_feats = pickle.load(open(ROOT / "data" / "_cached_train_features_v3.pkl", "rb"))
test_feats = pickle.load(open(ROOT / "data" / "_cached_test_features_v3.pkl", "rb"))

# Align columns
common_cols = sorted(set(train_feats.columns) & set(test_feats.columns))
X_train = train_feats[common_cols].values.astype(np.float32)
X_test = test_feats[common_cols].values.astype(np.float32)

# Clean
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Features: {X_train.shape[1]} (all v3, no selection)")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ---- CatBoost with multiple seeds ----
N_FOLDS = 5
SEEDS = [42, 7, 123, 456, 789]

oof_preds = np.zeros((len(y), 9), dtype=np.float64)
test_preds = np.zeros((X_test.shape[0], 9), dtype=np.float64)
oof_counts = np.zeros(len(y), dtype=int)

t0 = time.time()

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y[train_idx], y[val_idx]

        # Class weights
        counts = np.bincount(ytr, minlength=9).astype(float)
        max_count = counts.max()
        class_weights = {i: max_count / max(c, 1) for i, c in enumerate(counts)}

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed + fold,
            verbose=0,
            auto_class_weights='Balanced',
            task_type='CPU',
            eval_metric='MultiClass',
            early_stopping_rounds=100,
        )

        model.fit(
            Xtr, ytr,
            eval_set=(Xval, yval),
            verbose=0,
        )

        val_pred = model.predict_proba(Xval)
        oof_preds[val_idx] += val_pred
        oof_counts[val_idx] += 1

        test_pred = model.predict_proba(X_test)
        test_preds += test_pred

        # Per-fold score
        fold_aps = []
        for c in range(9):
            fold_aps.append(average_precision_score((yval == c).astype(int), val_pred[:, c]))
        print(f"  Fold {fold}: mAP={np.mean(fold_aps):.4f}")

# Average
oof_preds /= np.maximum(oof_counts[:, None], 1)
test_preds /= (N_FOLDS * len(SEEDS))

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s")

# ---- Evaluate ----
overall, per_class = compute_map(y, oof_preds)
print_results(overall, per_class, "E206: CatBoost ALL 316 features")

# ---- Save predictions ----
np.save(ROOT / "oof_e206_cb_all316.npy", oof_preds)
np.save(ROOT / "test_e206_cb_all316.npy", test_preds)
print(f"Saved oof_e206_cb_all316.npy and test_e206_cb_all316.npy")

# ---- Also train LGB for diversity ----
print("\n" + "=" * 60)
print("LGB on all 316 features")
print("=" * 60)

import lightgbm as lgb

oof_lgb = np.zeros((len(y), 9), dtype=np.float64)
test_lgb = np.zeros((X_test.shape[0], 9), dtype=np.float64)
oof_lgb_counts = np.zeros(len(y), dtype=int)

for seed in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(Xtr, ytr)
        dval = lgb.Dataset(Xval, yval, reference=dtrain)

        params = {
            'objective': 'multiclass',
            'num_class': 9,
            'metric': 'multi_logloss',
            'boosting_type': 'dart',
            'num_leaves': 63,
            'max_depth': 7,
            'learning_rate': 0.05,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'is_unbalance': True,
            'verbose': -1,
            'seed': seed + fold,
            'drop_rate': 0.1,
        }

        model_lgb = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        val_pred = model_lgb.predict(Xval)
        oof_lgb[val_idx] += val_pred
        oof_lgb_counts[val_idx] += 1
        test_lgb += model_lgb.predict(X_test)

oof_lgb /= np.maximum(oof_lgb_counts[:, None], 1)
test_lgb /= (N_FOLDS * len(SEEDS))

overall_lgb, per_class_lgb = compute_map(y, oof_lgb)
print_results(overall_lgb, per_class_lgb, "E206: LGB DART ALL 316 features")

np.save(ROOT / "oof_e206_lgb_all316.npy", oof_lgb)
np.save(ROOT / "test_e206_lgb_all316.npy", test_lgb)
print(f"Saved oof_e206_lgb_all316.npy and test_e206_lgb_all316.npy")

# ---- Blend into e188 pool ----
print("\n" + "=" * 60)
print("BLENDING INTO E188 POOL")
print("=" * 60)

from scipy.optimize import minimize

SUB_CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']

def macro_map_fast(preds):
    aps = []
    for i in range(9):
        aps.append(average_precision_score((y == i).astype(int), preds[:, i]))
    return np.mean(aps)

def softmax_w(w):
    w = np.abs(np.array(w))
    return w / max(w.sum(), 1e-12)

# Load e188 models + add our new ones
e188_models = ['e79', 'e175_best', 'e175_lgb', 'e179_best',
               'e185_tabpfn_relabel', 'e185_tabpfn_all', 'e186_ovo',
               'e180_cnn', 'e187_blend', 'e173', 'e179_cb']

pool = {}
for name in e188_models:
    oof_path = ROOT / f'oof_{name}.npy'
    test_path = ROOT / f'test_{name}.npy'
    if oof_path.exists() and test_path.exists():
        o = np.load(oof_path, allow_pickle=True)
        t = np.load(test_path, allow_pickle=True)
        if hasattr(o, 'shape') and o.shape == (2601, 9) and t.shape == (1872, 9):
            pool[name] = {'oof': o.astype(float), 'test': t.astype(float)}

# Add new models
pool['e206_cb_all316'] = {'oof': oof_preds, 'test': test_preds}
pool['e206_lgb_all316'] = {'oof': oof_lgb, 'test': test_lgb}

print(f"Pool: {len(pool)} models ({len(e188_models)} e188 + 2 new)")
pool_names = list(pool.keys())
pool_oofs = [pool[n]['oof'] for n in pool_names]
pool_tests = [pool[n]['test'] for n in pool_names]

def neg_map(w):
    w = softmax_w(w)
    blend = sum(wi * o for wi, o in zip(w, pool_oofs))
    return -macro_map_fast(blend)

# Multi-restart
best_m = 0
best_w = None
for seed in range(10):
    rng = np.random.RandomState(seed)
    w0 = rng.dirichlet(np.ones(len(pool_names)))
    res = minimize(neg_map, w0, method='Nelder-Mead',
                   options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-6})
    w = softmax_w(res.x)
    m = macro_map_fast(sum(wi * o for wi, o in zip(w, pool_oofs)))
    if m > best_m:
        best_m = m
        best_w = w
    print(f"  seed {seed}: {m:.4f}")

print(f"\nBest blend OOF: {best_m:.4f}")
print("Weights (>1%):")
for i, n in enumerate(pool_names):
    if best_w[i] > 0.01:
        print(f"  {n:30s}: {best_w[i]:.3f}")

# Generate submissions
track_ids = test_df['track_id'].values

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e206_{tag}.csv"
    df.to_csv(fname, index=False)
    print(f"  Saved {fname}")

# Raw blend
test_blend = sum(wi * t for wi, t in zip(best_w, pool_tests))
save_sub(test_blend, "e188_plus_316feat")

# T=0.9 sharpening
sharp = test_blend ** (1.0 / 0.9)
sharp = sharp / sharp.sum(axis=1, keepdims=True)
save_sub(sharp, "e188_plus_316feat_T09")

# CB 316 standalone
save_sub(test_preds, "cb_all316_raw")

print("\nDone!")
