"""E201: Diverse model zoo — build many different models, each good at something.

Models:
1. CatBoost multi-seed (10 seeds, strong reg) — robust base
2. 9x OvR binary CatBoost — per-class specialists
3. Random Forest — can't overfit as hard as boosting
4. Logistic Regression — simplest possible, maximum regularization
5. LightGBM DART — dropout prevents overfitting
6. Per-class column selection — pick best model per class
7. Simple average of diverse models — diversity > optimization

Then: submit raw + blends + per-class + averaged.
"""
import sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb

ROOT = Path('G:/Projects/epoch-ai-cup')
from src.data import CLASSES, load_train, load_test
from src.features import build_features, ALL_TEMPORAL

N = len(CLASSES)
SEED = 42
SUB = ['Clutter','Cormorants','Pigeons','Ducks','Geese','Gulls','Birds of Prey','Waders','Songbirds']

# ── Load data ──
print("="*60, flush=True)
print("E201: DIVERSE MODEL ZOO", flush=True)
print("="*60, flush=True)

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df['bird_group'], categories=CLASSES).codes, dtype=int)

# Build features (same as E79)
print("\nBuilding features...", flush=True)
fsets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=fsets)
test_feats = build_features(test_df, feature_sets=fsets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]; test_feats = test_feats[keep]
for prefix, fname in [('wx_', 'weather'), ('sol_', 'solar')]:
    for split, feats in [('train', train_feats), ('test', test_feats)]:
        ext = pd.read_csv(ROOT / f'data/{split}_{fname}.csv')
        for col in ext.columns:
            feats[f'{prefix}{col}'] = ext[col].values

feat_names = [f.strip() for f in open(ROOT/'data/best_features.txt').read().splitlines() if f.strip()]
avail = [f for f in feat_names if f in train_feats.columns]
print(f"  {len(avail)}/{len(feat_names)} E79 features available", flush=True)

X = train_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
Xt = test_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Also build ALL features (for models that might benefit)
all_cols = [c for c in train_feats.columns]
X_all = train_feats[all_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
Xt_all = test_feats[all_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  All features: {X_all.shape[1]}", flush=True)

# Effective number weights
counts = np.bincount(y, minlength=N).astype(float)
beta = 0.999
eff_n = (1.0 - beta**counts) / (1.0 - beta)
cw = 1.0 / np.maximum(eff_n, 1e-6)
cw /= cw.sum() / N
sw = cw[y]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def eval_oof(oof, label):
    yb = np.zeros((len(y), N))
    for i in range(N): yb[:, i] = (y == i).astype(int)
    aps = [average_precision_score(yb[:, i], oof[:, i]) for i in range(N)]
    m = np.mean(aps)
    weak = [0, 2, 3, 6, 8]  # BoP, Corm, Duck, Pig, Wad
    print(f"  {label}: SKF={m:.4f}  " +
          " ".join(f"{CLASSES[c][:4]}={aps[c]:.3f}" for c in weak), flush=True)
    return m, aps

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': test_df['track_id'].values})
    for cls in SUB: df[cls] = preds[:, CLASSES.index(cls)]
    df.to_csv(ROOT / f'submissions/{tag}.csv', index=False)

all_oofs = {}
all_tests = {}

# ══════════════════════════════════════════════════════════════
# MODEL 1: CatBoost multi-seed (10 seeds, strong regularization)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n1. CatBoost Multi-Seed (10 seeds)\n{'='*60}", flush=True)
t0 = time.time()
oof_cb_avg = np.zeros((len(y), N))
test_cb_avg = np.zeros((len(Xt), N))

for seed in range(10):
    oof_s = np.zeros((len(y), N))
    test_s = np.zeros((len(Xt), N))
    skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed*11+7)
    for _, (tr, va) in enumerate(skf_s.split(X, y)):
        mc = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                                l2_leaf_reg=5.0, rsm=0.6,
                                bootstrap_type='MVS', subsample=0.7,
                                auto_class_weights='Balanced', verbose=0,
                                early_stopping_rounds=50, random_seed=seed*11+7)
        mc.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        oof_s[va] = mc.predict_proba(X[va])
        test_s += mc.predict_proba(Xt) / 5
    oof_cb_avg += oof_s / 10
    test_cb_avg += test_s / 10
    if seed % 3 == 0:
        print(f"  Seed {seed+1}/10 done", flush=True)

eval_oof(oof_cb_avg, "CB 10-seed avg")
all_oofs['cb_10seed'] = oof_cb_avg
all_tests['cb_10seed'] = test_cb_avg
save_sub(test_cb_avg, 'e201_cb_10seed')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# MODEL 2: 9x OvR Binary CatBoost
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n2. OvR Binary CatBoost (9 models)\n{'='*60}", flush=True)
t0 = time.time()
oof_ovr = np.zeros((len(y), N))
test_ovr = np.zeros((len(Xt), N))

for c in range(N):
    y_bin = (y == c).astype(int)
    oof_c = np.zeros(len(y))
    test_c = np.zeros(len(Xt))

    for _, (tr, va) in enumerate(skf.split(X, y)):
        mc = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                                l2_leaf_reg=5.0, auto_class_weights='Balanced',
                                verbose=0, early_stopping_rounds=50, random_seed=SEED)
        mc.fit(X[tr], y_bin[tr], eval_set=(X[va], y_bin[va]))
        proba = mc.predict_proba(X[va])
        oof_c[va] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        tp = mc.predict_proba(Xt)
        test_c += (tp[:, 1] if tp.shape[1] > 1 else tp[:, 0]) / 5

    oof_ovr[:, c] = oof_c
    test_ovr[:, c] = test_c

# Normalize rows
oof_ovr = np.clip(oof_ovr, 1e-10, None)
oof_ovr /= oof_ovr.sum(axis=1, keepdims=True)
test_ovr = np.clip(test_ovr, 1e-10, None)
test_ovr /= test_ovr.sum(axis=1, keepdims=True)

eval_oof(oof_ovr, "OvR Binary CB")
all_oofs['ovr_cb'] = oof_ovr
all_tests['ovr_cb'] = test_ovr
save_sub(test_ovr, 'e201_ovr_cb')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# MODEL 3: Random Forest (can't overfit as hard)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n3. Random Forest\n{'='*60}", flush=True)
t0 = time.time()
oof_rf = np.zeros((len(y), N))
test_rf = np.zeros((len(Xt), N))

for _, (tr, va) in enumerate(skf.split(X, y)):
    rf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                max_features='sqrt', class_weight='balanced',
                                random_state=SEED, n_jobs=-1)
    rf.fit(X[tr], y[tr], sample_weight=sw[tr])
    oof_rf[va] = rf.predict_proba(X[va])
    test_rf += rf.predict_proba(Xt) / 5

eval_oof(oof_rf, "Random Forest")
all_oofs['rf'] = oof_rf
all_tests['rf'] = test_rf
save_sub(test_rf, 'e201_rf')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# MODEL 4: Logistic Regression (maximum simplicity)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n4. Logistic Regression\n{'='*60}", flush=True)
t0 = time.time()
oof_lr = np.zeros((len(y), N))
test_lr = np.zeros((len(Xt), N))

from sklearn.preprocessing import StandardScaler
for _, (tr, va) in enumerate(skf.split(X, y)):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X[tr])
    Xs_va = scaler.transform(X[va])
    Xs_te = scaler.transform(Xt)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                            multi_class='multinomial', solver='lbfgs', random_state=SEED)
    lr.fit(Xs_tr, y[tr], sample_weight=sw[tr])
    oof_lr[va] = lr.predict_proba(Xs_va)
    test_lr += lr.predict_proba(Xs_te) / 5

eval_oof(oof_lr, "Logistic Regression")
all_oofs['lr'] = oof_lr
all_tests['lr'] = test_lr
save_sub(test_lr, 'e201_lr')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# MODEL 5: LightGBM DART (dropout prevents overfitting)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n5. LightGBM DART\n{'='*60}", flush=True)
t0 = time.time()
oof_dart = np.zeros((len(y), N))
test_dart = np.zeros((len(Xt), N))

for _, (tr, va) in enumerate(skf.split(X, y)):
    dtrain = lgb.Dataset(X[tr], y[tr], weight=sw[tr])
    dval = lgb.Dataset(X[va], y[va])
    m = lgb.train({'objective': 'multiclass', 'num_class': 9, 'metric': 'multi_logloss',
                   'boosting_type': 'dart', 'drop_rate': 0.15, 'skip_drop': 0.5,
                   'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6,
                   'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
                   'min_child_samples': 20, 'lambda_l1': 0.1, 'lambda_l2': 1.0,
                   'verbose': -1, 'is_unbalance': True},
                  dtrain, 500, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_dart[va] = m.predict(X[va])
    test_dart += m.predict(Xt) / 5

eval_oof(oof_dart, "LGB DART")
all_oofs['dart'] = oof_dart
all_tests['dart'] = test_dart
save_sub(test_dart, 'e201_dart')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# MODEL 6: CatBoost on ALL features (more signal, more noise)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n6. CatBoost ALL features ({X_all.shape[1]})\n{'='*60}", flush=True)
t0 = time.time()
oof_cb_all = np.zeros((len(y), N))
test_cb_all = np.zeros((len(Xt_all), N))

for _, (tr, va) in enumerate(skf.split(X_all, y)):
    mc = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                            l2_leaf_reg=5.0, rsm=0.3,
                            bootstrap_type='MVS', subsample=0.7,
                            auto_class_weights='Balanced', verbose=0,
                            early_stopping_rounds=50, random_seed=SEED)
    mc.fit(X_all[tr], y[tr], eval_set=(X_all[va], y[va]))
    oof_cb_all[va] = mc.predict_proba(X_all[va])
    test_cb_all += mc.predict_proba(Xt_all) / 5

eval_oof(oof_cb_all, "CB ALL feats")
all_oofs['cb_all'] = oof_cb_all
all_tests['cb_all'] = test_cb_all
save_sub(test_cb_all, 'e201_cb_all_feats')
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# ENSEMBLES
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nENSEMBLES\n{'='*60}", flush=True)

model_names = list(all_oofs.keys())
print(f"Models: {model_names}", flush=True)

# Simple average of ALL models
avg_oof = np.mean([all_oofs[k] for k in model_names], axis=0)
avg_test = np.mean([all_tests[k] for k in model_names], axis=0)
eval_oof(avg_oof, "Simple avg ALL")
save_sub(avg_test, 'e201_avg_all')

# Average of best 3 (CB 10-seed + DART + OvR)
best3_oof = (all_oofs['cb_10seed'] + all_oofs['dart'] + all_oofs['ovr_cb']) / 3
best3_test = (all_tests['cb_10seed'] + all_tests['dart'] + all_tests['ovr_cb']) / 3
eval_oof(best3_oof, "Avg (CB+DART+OvR)")
save_sub(best3_test, 'e201_avg_cb_dart_ovr')

# CB 10-seed + RF (most diverse pair)
div_oof = 0.7 * all_oofs['cb_10seed'] + 0.3 * all_oofs['rf']
div_test = 0.7 * all_tests['cb_10seed'] + 0.3 * all_tests['rf']
eval_oof(div_oof, "CB70+RF30")
save_sub(div_test, 'e201_cb70_rf30')

# Per-class best model selection
print("\nPer-class best model:", flush=True)
yb = np.zeros((len(y), N))
for i in range(N): yb[:, i] = (y == i).astype(int)

perclass_test = np.zeros((len(Xt), N))
for c in range(N):
    best_name = None
    best_ap = 0
    for name in model_names:
        ap = average_precision_score(yb[:, c], all_oofs[name][:, c])
        if ap > best_ap:
            best_ap = ap
            best_name = name
    perclass_test[:, c] = all_tests[best_name][:, c]
    print(f"  {CLASSES[c]:15s}: {best_name} (AP={best_ap:.3f})", flush=True)

perclass_test = np.clip(perclass_test, 1e-10, None)
perclass_test /= perclass_test.sum(axis=1, keepdims=True)
save_sub(perclass_test, 'e201_perclass_best')

# Temperature variants on CB 10-seed
for T in [0.85, 0.9, 0.95]:
    logits = np.log(np.clip(test_cb_avg, 1e-8, 1.0))
    s = logits / T; s -= s.max(axis=1, keepdims=True)
    sharp = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    save_sub(sharp, f'e201_cb10seed_T{T}')

# Temperature on simple average
for T in [0.85, 0.9]:
    logits = np.log(np.clip(avg_test, 1e-8, 1.0))
    s = logits / T; s -= s.max(axis=1, keepdims=True)
    sharp = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    save_sub(sharp, f'e201_avg_T{T}')

print(f"\n{'='*60}\nALL DONE\n{'='*60}", flush=True)

# Save OOFs for later analysis
for name in model_names:
    np.save(ROOT / f'oof_e201_{name}.npy', all_oofs[name])
    np.save(ROOT / f'test_e201_{name}.npy', all_tests[name])
print("OOFs and test preds saved.", flush=True)
