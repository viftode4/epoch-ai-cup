"""E186b: Finish remaining tests + validate OvO on TRUE LOMO.

1. Curriculum learning (fixed init_model issue)
2. LGB + XGB + CatBoost ensemble (ALL features, relabeled)
3. TRUE LOMO for OvO and best approaches
4. OvO + GBDT ensemble blend
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

feats_train = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
X_train = np.nan_to_num(feats_train.values.astype(np.float32), nan=0, posinf=0, neginf=0)

cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
agreed_noisy = cache['agreed_noisy'].tolist()
consensus_labels = cache['consensus_labels']
quality = cache['quality']

y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def eval_skf(name, oof):
    skf, pc = compute_map(y, oof)
    print(f"  {name}: SKF={skf:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} "
          f"BoP={pc['Birds of Prey']:.4f} Gulls={pc['Gulls']:.4f}", flush=True)
    return skf, pc

def rpe(preds_list, weights, power=1.5):
    n_s = preds_list[0].shape[0]
    out = np.zeros((n_s, N))
    for c in range(N):
        for p, w in zip(preds_list, weights):
            out[:, c] += w * (rankdata(p[:, c]) / n_s) ** power
    return out

print("=" * 90, flush=True)
print("E186b: REMAINING TESTS + VALIDATION", flush=True)
print("=" * 90, flush=True)

# Load OvO results from E186
oof_ovo = np.load(ROOT / "oof_e186_ovo.npy")
oof_tabpfn_relabel = np.load(ROOT / "oof_e185_tabpfn_relabel.npy")
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
eval_skf("REF: OvO Pairwise", oof_ovo)
eval_skf("REF: TabPFN-ALL relabel", oof_tabpfn_relabel)

# ══════════════════════════════════════════════════════════════
# 1. CURRICULUM LEARNING (fixed)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n1. CURRICULUM LEARNING (fixed)\n{'='*90}", flush=True)

clean_mask = quality > 0.3
print(f"  Clean samples (q>0.3): {clean_mask.sum()} / {len(y)}", flush=True)

oof_curriculum = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    # Stage 1: clean only
    tr_clean = tr[clean_mask[tr] & keep_mask[tr]]
    clf1 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                               random_state=42, verbose=-1, n_jobs=1)
    clf1.fit(X_train[tr_clean], y_relabeled[tr_clean])
    # Save booster for init
    booster1 = clf1.booster_

    # Stage 2: all data, continue from stage 1
    tr_all = tr[keep_mask[tr]]
    clf2 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.02, num_leaves=31,
                               class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                               random_state=42, verbose=-1, n_jobs=1)
    clf2.fit(X_train[tr_all], y_relabeled[tr_all], init_model=booster1)
    oof_curriculum[va] = clf2.predict_proba(X_train[va])

eval_skf("Curriculum (clean->all)", oof_curriculum)

# ══════════════════════════════════════════════════════════════
# 2. LGB + XGB + CB ENSEMBLE
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n2. LGB + XGB + CatBoost ENSEMBLE\n{'='*90}", flush=True)

t0 = time.time()

# LGB
oof_lgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              class_weight="balanced", subsample=0.7, colsample_bytree=0.6,
                              random_state=42, verbose=-1, n_jobs=1)
    clf.fit(X_train[tr_use], y_relabeled[tr_use],
            eval_set=[(X_train[va], y[va])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = clf.predict_proba(X_train[va])
eval_skf("LGB (relabeled, ALL)", oof_lgb)

# XGB
oof_xgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    cc = np.bincount(y_relabeled[tr_use], minlength=N)
    sw = np.array([1.0 / max(cc[c], 1) for c in y_relabeled[tr_use]])
    sw = sw / sw.mean()
    clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6,
                              subsample=0.7, colsample_bytree=0.6,
                              random_state=42, verbosity=0, n_jobs=1,
                              early_stopping_rounds=50, eval_metric="mlogloss")
    clf.fit(X_train[tr_use], y_relabeled[tr_use], sample_weight=sw,
            eval_set=[(X_train[va], y[va])], verbose=False)
    oof_xgb[va] = clf.predict_proba(X_train[va])
eval_skf("XGB (relabeled, ALL)", oof_xgb)

# CatBoost
oof_cb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    clf = cb.CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                                 auto_class_weights="Balanced", subsample=0.7, rsm=0.6,
                                 random_seed=42, verbose=0, task_type="CPU",
                                 early_stopping_rounds=50)
    clf.fit(cb.Pool(X_train[tr_use], y_relabeled[tr_use]),
            eval_set=cb.Pool(X_train[va], y[va]))
    oof_cb[va] = clf.predict_proba(X_train[va])
eval_skf("CatBoost (relabeled, ALL)", oof_cb)

print(f"  3-model training: {time.time()-t0:.0f}s", flush=True)

# Ensembles
oof_3way = (oof_lgb + oof_xgb + oof_cb) / 3
eval_skf("LGB+XGB+CB average", oof_3way)

oof_3way_rp = rpe([oof_lgb, oof_xgb, oof_cb], [0.4, 0.3, 0.3], 1.5)
eval_skf("LGB+XGB+CB rank-power", oof_3way_rp)

# + TabPFN
oof_4way = rpe([oof_lgb, oof_xgb, oof_cb, oof_tabpfn_relabel], [0.25, 0.2, 0.2, 0.35], 1.5)
eval_skf("GBDT+TabPFN rank-power", oof_4way)

# + OvO
oof_5way = rpe([oof_lgb, oof_xgb, oof_cb, oof_tabpfn_relabel, oof_ovo],
               [0.2, 0.15, 0.15, 0.25, 0.25], 1.5)
eval_skf("GBDT+TabPFN+OvO rank-power", oof_5way)

# Save
np.save(ROOT / "oof_e186_lgb.npy", oof_lgb)
np.save(ROOT / "oof_e186_xgb.npy", oof_xgb)
np.save(ROOT / "oof_e186_cb.npy", oof_cb)

# ══════════════════════════════════════════════════════════════
# 3. BLEND OPTIMIZATION (LOMO)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n3. BLEND OPTIMIZATION\n{'='*90}", flush=True)

def lomo(oof_pred):
    maps = {}
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof_pred[mask])
            maps[m] = lm
    return np.mean(list(maps.values())), maps

# Search: OvO + TabPFN + best GBDT
components = [oof_ovo, oof_tabpfn_relabel, oof_3way_rp]
comp_names = ["OvO", "TabPFN", "GBDT"]

best_lomo = -1
best_cfg = None
for power in [1.0, 1.5, 2.0]:
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            w2 = round(1.0 - w0 - w1, 2)
            if w2 < -0.01:
                continue
            blend = rpe(components, [w0, w1, w2], power)
            l, _ = lomo(blend)
            skf, pc = compute_map(y, blend)
            if l > best_lomo:
                best_lomo = l
                best_cfg = (w0, w1, w2, power, skf, pc, l)

w0, w1, w2, pw, skf_best, pc_best, lomo_best = best_cfg
print(f"  Best 3-way: OvO={w0:.1f} TabPFN={w1:.1f} GBDT={w2:.1f} power={pw}", flush=True)
print(f"  SKF={skf_best:.4f}  LOMO={lomo_best:.4f}", flush=True)
for cls in CLASSES:
    print(f"    {cls:15s}: {pc_best[cls]:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════
# 4. TRUE LOMO VALIDATION
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n4. TRUE LOMO VALIDATION (honest generalization)\n{'='*90}", flush=True)

LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
             subsample=0.7, colsample_bytree=0.6, random_state=42, verbose=-1, n_jobs=1)

def true_lomo(name, X, y_labels, mask=None):
    oof = np.zeros((len(y), N))
    for m in sorted(set(months)):
        va = months == m
        tr = ~va
        if mask is not None:
            tr = tr & mask
        clf = lgb.LGBMClassifier(**LGB_P)
        clf.fit(X[tr], y_labels[tr])
        oof[va] = clf.predict_proba(X[va])
    overall, pc = compute_map(y, oof)
    ms = {}
    for m in sorted(set(months)):
        mask_m = months == m
        s, _ = compute_map(y[mask_m], oof[mask_m])
        ms[m] = s
    l = np.mean(list(ms.values()))
    month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms.items()))
    print(f"  {name}: TRUE_LOMO={l:.4f} ({month_str})", flush=True)
    print(f"    Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} BoP={pc['Birds of Prey']:.4f}", flush=True)
    return l, pc

true_lomo("LGB baseline (orig labels)", X_train, y)
true_lomo("LGB relabeled+clean", X_train, y_relabeled, mask=keep_mask)

# OvO TRUE LOMO: need to retrain per month
print(f"\n  OvO TRUE LOMO (36 pairs × 4 months)...", flush=True)
from tabpfn import TabPFNClassifier
t0 = time.time()
oof_ovo_lomo = np.zeros((len(y), N))
for m in sorted(set(months)):
    va = months == m
    tr = (~va) & keep_mask
    # Train 36 pairs on 3 months, predict held-out month
    for i in range(N):
        for j in range(i+1, N):
            pair_tr = tr & ((y_relabeled == i) | (y_relabeled == j))
            if pair_tr.sum() < 6:
                continue
            y_pair = (y_relabeled[pair_tr] == i).astype(int)
            clf = TabPFNClassifier(n_estimators=4, random_state=42)
            clf.fit(X_train[pair_tr], y_pair)
            probs = clf.predict_proba(X_train[va])
            p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            oof_ovo_lomo[va, i] += p1
            oof_ovo_lomo[va, j] += (1 - p1)

# Normalize
oof_ovo_lomo = oof_ovo_lomo / np.maximum(oof_ovo_lomo.sum(axis=1, keepdims=True), 1e-10)
overall_ovo, pc_ovo = compute_map(y, oof_ovo_lomo)
ms_ovo = {}
for m in sorted(set(months)):
    mask_m = months == m
    s, _ = compute_map(y[mask_m], oof_ovo_lomo[mask_m])
    ms_ovo[m] = s
lomo_ovo = np.mean(list(ms_ovo.values()))
month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms_ovo.items()))
print(f"  OvO TabPFN: TRUE_LOMO={lomo_ovo:.4f} ({month_str})", flush=True)
print(f"    Corm={pc_ovo['Cormorants']:.4f} Wader={pc_ovo['Waders']:.4f} BoP={pc_ovo['Birds of Prey']:.4f}", flush=True)
print(f"    OvO TRUE LOMO took {time.time()-t0:.0f}s", flush=True)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\nFINAL SUMMARY\n{'='*90}", flush=True)

all_results = [
    ("E175 (LB=0.59)", oof_e175),
    ("TabPFN-ALL relabel", oof_tabpfn_relabel),
    ("OvO Pairwise (SKF)", oof_ovo),
    ("Curriculum learning", oof_curriculum),
    ("LGB (relabeled, ALL)", oof_lgb),
    ("XGB (relabeled, ALL)", oof_xgb),
    ("CatBoost (relabeled, ALL)", oof_cb),
    ("LGB+XGB+CB average", oof_3way),
    ("LGB+XGB+CB rank-power", oof_3way_rp),
    ("GBDT+TabPFN", oof_4way),
    ("GBDT+TabPFN+OvO", oof_5way),
]

print(f"\n  {'Config':35s} {'SKF':>7s} {'LOMO*':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Gulls':>7s}", flush=True)
print(f"  {'-'*90}", flush=True)
for name, oof in all_results:
    skf, pc = compute_map(y, oof)
    l, _ = lomo(oof)
    print(f"  {name:35s} {skf:7.4f} {l:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} "
          f"{pc['Birds of Prey']:7.4f} {pc['Gulls']:7.4f}", flush=True)

print(f"\n  * LOMO is FAKE (post-hoc on SKF OOF)", flush=True)
print(f"\nDone.", flush=True)
