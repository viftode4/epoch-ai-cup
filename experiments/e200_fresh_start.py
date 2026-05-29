"""E200: Fresh start. Simple, clean, aimed at 0.59+.

What we know from 187 experiments:
- E79 (36 features, LGB+XGB+CB, effective weights) = 0.59 LB
- More features = worse (dilution confirmed at scale)
- TabPFN/OvO hurt LB despite looking good on OOF
- Post-processing is neutral at best
- Temperature T=0.9 matched 0.59 on one submission

Strategy: Rebuild clean with slightly different hyperparams/seeds to generate
diverse submissions, then submit all and pick the best.
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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path('G:/Projects/epoch-ai-cup')
from src.data import CLASSES, load_train, load_test
from src.features import build_features, ALL_TEMPORAL

# ── Load & build features ──
print("Loading...", flush=True)
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df['bird_group'], categories=CLASSES).codes, dtype=int)
N = len(CLASSES)

print("Building features...", flush=True)
fsets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=fsets)
test_feats = build_features(test_df, feature_sets=fsets)

# Remove temporal
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
for prefix, fname in [('wx_', 'weather'), ('sol_', 'solar')]:
    for split, feats in [('train', train_feats), ('test', test_feats)]:
        ext = pd.read_csv(ROOT / f'data/{split}_{fname}.csv')
        for col in ext.columns:
            feats[f'{prefix}{col}'] = ext[col].values

print(f"  Total features: {train_feats.shape[1]}", flush=True)

# Select E79's 36 pruned features
feat_names = [f.strip() for f in open(ROOT / 'data/best_features.txt').read().splitlines() if f.strip()]
avail = [f for f in feat_names if f in train_feats.columns]
miss = [f for f in feat_names if f not in train_feats.columns]
print(f"  E79 features: {len(avail)}/{len(feat_names)}, missing: {miss}", flush=True)

X = train_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats[avail].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Effective number weights
counts = np.bincount(y, minlength=N).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
cw = 1.0 / np.maximum(eff_n, 1e-6)
cw /= cw.sum() / N
sw = cw[y]

SUB = ['Clutter','Cormorants','Pigeons','Ducks','Geese','Gulls','Birds of Prey','Waders','Songbirds']

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': test_df['track_id'].values})
    for cls in SUB:
        df[cls] = preds[:, CLASSES.index(cls)]
    df.to_csv(ROOT / f'submissions/{tag}.csv', index=False)
    print(f"  Saved {tag}.csv", flush=True)

def eval_oof(oof, label):
    yb = np.zeros((len(y), N))
    for i in range(N): yb[:, i] = (y == i).astype(int)
    aps = [average_precision_score(yb[:, i], oof[:, i]) for i in range(N)]
    m = np.mean(aps)
    print(f"  {label}: SKF={m:.4f}  " + " ".join(f"{CLASSES[c][:4]}={aps[c]:.3f}" for c in range(N)), flush=True)
    return m

# ── Train multiple seeds/configs and save each ──
configs = [
    # (name, lgb_params, xgb_params, cb_params, weights, seed)
    ("A_original",
     dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
          class_weight='balanced', verbose=-1, n_jobs=-1),
     dict(n_estimators=1500, learning_rate=0.03, max_depth=6,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
          objective='multi:softprob', eval_metric='mlogloss', verbosity=0, tree_method='hist'),
     dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
          auto_class_weights='Balanced', verbose=0, early_stopping_rounds=100),
     [0.50, 0.40, 0.10], 42),

    ("B_cb_heavy",
     dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
          class_weight='balanced', verbose=-1, n_jobs=-1),
     dict(n_estimators=1500, learning_rate=0.03, max_depth=6,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
          objective='multi:softprob', eval_metric='mlogloss', verbosity=0, tree_method='hist'),
     dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
          auto_class_weights='Balanced', verbose=0, early_stopping_rounds=100),
     [0.15, 0.05, 0.80], 42),  # CB dominant (E173 showed CB=0.80 is best)

    ("C_seed7",
     dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
          class_weight='balanced', verbose=-1, n_jobs=-1),
     dict(n_estimators=1500, learning_rate=0.03, max_depth=6,
          subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
          objective='multi:softprob', eval_metric='mlogloss', verbosity=0, tree_method='hist'),
     dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
          auto_class_weights='Balanced', verbose=0, early_stopping_rounds=100),
     [0.50, 0.40, 0.10], 7),  # Different seed

    ("D_deeper",
     dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=8,
          subsample=0.7, colsample_bytree=0.4, reg_alpha=0.1, reg_lambda=1.0,
          class_weight='balanced', verbose=-1, n_jobs=-1),
     dict(n_estimators=2000, learning_rate=0.02, max_depth=8,
          subsample=0.7, colsample_bytree=0.4, reg_alpha=0.1, reg_lambda=1.0,
          objective='multi:softprob', eval_metric='mlogloss', verbosity=0, tree_method='hist'),
     dict(iterations=2000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0,
          auto_class_weights='Balanced', verbose=0, early_stopping_rounds=100),
     [0.40, 0.30, 0.30], 42),
]

all_tests = []
for name, lgb_p, xgb_p, cb_p, weights, seed in configs:
    print(f"\n{'='*60}\nConfig: {name} (seed={seed})\n{'='*60}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_l = np.zeros((len(y), N)); oof_x = np.zeros((len(y), N)); oof_c = np.zeros((len(y), N))
    tl = np.zeros((len(X_test), N)); tx = np.zeros((len(X_test), N)); tc = np.zeros((len(X_test), N))

    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5...", flush=True)

        lgb_p_fold = {**lgb_p, 'random_state': seed}
        m = LGBMClassifier(**lgb_p_fold)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], sample_weight=sw[tr])
        oof_l[va] = m.predict_proba(X[va]); tl += m.predict_proba(X_test) / 5

        xgb_p_fold = {**xgb_p, 'random_state': seed, 'num_class': N}
        mx = XGBClassifier(**xgb_p_fold)
        mx.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], sample_weight=sw[tr], verbose=False)
        oof_x[va] = mx.predict_proba(X[va]); tx += mx.predict_proba(X_test) / 5

        cb_p_fold = {**cb_p, 'random_seed': seed}
        mc = CatBoostClassifier(**cb_p_fold)
        mc.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        oof_c[va] = mc.predict_proba(X[va]); tc += mc.predict_proba(X_test) / 5

    wl, wx, wc = weights
    oof = wl*oof_l + wx*oof_x + wc*oof_c
    test_pred = wl*tl + wx*tx + wc*tc

    m = eval_oof(oof, name)
    save_sub(test_pred, f'e200_{name}')
    all_tests.append(test_pred)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# Also save average of all configs
if len(all_tests) > 1:
    avg = np.mean(all_tests, axis=0)
    save_sub(avg, 'e200_avg_all')

    # And sharpened versions
    for T in [0.85, 0.9, 0.95]:
        logits = np.log(np.clip(avg, 1e-8, 1.0))
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        sharp = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
        save_sub(sharp, f'e200_avg_T{T}')

print("\nDone!", flush=True)
