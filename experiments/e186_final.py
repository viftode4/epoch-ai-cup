"""E186 Final: Everything in one script, no dependencies on crashed runs."""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from src.data import load_train, CLASSES
from src.metrics import compute_map
import lightgbm as lgb, xgboost as xgb, catboost as cb

ROOT = Path('.'); N = len(CLASSES)
train = load_train()
y = np.asarray(pd.Categorical(train['bird_group'], categories=CLASSES).codes, dtype=int)
groups = train['primary_observation_id'].values
months = pd.to_datetime(train['timestamp_start_radar_utc']).dt.month.values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

X = np.nan_to_num(pd.read_pickle('data/_cached_train_features_v3.pkl').values.astype(np.float32), nan=0, posinf=0, neginf=0)
cache = np.load('data/_cleanlab_cache.npz', allow_pickle=True)
y_r = y.copy()
for idx in cache['agreed_noisy'].tolist(): y_r[idx] = cache['consensus_labels'][idx]
km = np.array([i not in set(np.where(cache['quality'] < 0.02)[0]) for i in range(len(y))])

oof_ovo = np.load('oof_e186_ovo.npy')
oof_tabpfn = np.load('oof_e185_tabpfn_relabel.npy')
oof_e175 = np.load('oof_e175_best.npy')

def ev(name, oof):
    s, pc = compute_map(y, oof)
    print(f'  {name}: SKF={s:.4f} Corm={pc["Cormorants"]:.4f} Wader={pc["Waders"]:.4f} BoP={pc["Birds of Prey"]:.4f}', flush=True)
    return s, pc

def rpe(pl, w, p=1.5):
    n = pl[0].shape[0]; o = np.zeros((n, N))
    for c in range(N):
        for pr, wt in zip(pl, w): o[:, c] += wt * (rankdata(pr[:, c]) / n) ** p
    return o

def lomo(oof):
    ms = {}
    for m in sorted(set(months)):
        mk = months == m
        if mk.sum() >= 10: ms[m], _ = compute_map(y[mk], oof[mk])
    return np.mean(list(ms.values())), ms

print("=" * 90, flush=True)
print("E186 FINAL: GBDT ENSEMBLE + BLENDS + TRUE LOMO", flush=True)
print("=" * 90, flush=True)

# ── GBDT Ensemble ──
print("\n[1] Training LGB + XGB + CatBoost (ALL features, relabeled)...", flush=True)
t0 = time.time()

oof_lgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X, y_r, groups)):
    tr_use = tr[km[tr]]
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              class_weight="balanced", subsample=0.7, colsample_bytree=0.6,
                              random_state=42, verbose=-1, n_jobs=1)
    clf.fit(X[tr_use], y_r[tr_use], eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = clf.predict_proba(X[va])
ev("LGB", oof_lgb)

oof_xgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X, y_r, groups)):
    tr_use = tr[km[tr]]
    cc = np.bincount(y_r[tr_use], minlength=N)
    sw = np.array([1.0 / max(cc[c], 1) for c in y_r[tr_use]]); sw /= sw.mean()
    clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6,
                              subsample=0.7, colsample_bytree=0.6, random_state=42,
                              verbosity=0, n_jobs=1, early_stopping_rounds=50, eval_metric="mlogloss")
    clf.fit(X[tr_use], y_r[tr_use], sample_weight=sw, eval_set=[(X[va], y[va])], verbose=False)
    oof_xgb[va] = clf.predict_proba(X[va])
ev("XGB", oof_xgb)

oof_cb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X, y_r, groups)):
    tr_use = tr[km[tr]]
    clf = cb.CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                                 auto_class_weights="Balanced", rsm=0.6,
                                 bootstrap_type="MVS", subsample=0.7,
                                 random_seed=42, verbose=0, task_type="CPU",
                                 early_stopping_rounds=50)
    clf.fit(cb.Pool(X[tr_use], y_r[tr_use]), eval_set=cb.Pool(X[va], y[va]))
    oof_cb[va] = clf.predict_proba(X[va])
ev("CatBoost", oof_cb)
print(f"  GBDT training: {time.time()-t0:.0f}s", flush=True)

np.save('oof_e186_lgb.npy', oof_lgb)
np.save('oof_e186_xgb.npy', oof_xgb)
np.save('oof_e186_cb.npy', oof_cb)

# ── Ensembles ──
print("\n[2] Ensembles...", flush=True)
oof_3avg = (oof_lgb + oof_xgb + oof_cb) / 3
ev("GBDT avg", oof_3avg)

oof_3rp = rpe([oof_lgb, oof_xgb, oof_cb], [0.4, 0.3, 0.3])
ev("GBDT rank-power", oof_3rp)

ev("GBDT+TabPFN", rpe([oof_3rp, oof_tabpfn], [0.5, 0.5]))
ev("GBDT+OvO", rpe([oof_3rp, oof_ovo], [0.5, 0.5]))
ev("TabPFN+OvO", rpe([oof_tabpfn, oof_ovo], [0.5, 0.5]))
ev("ALL 5 equal", rpe([oof_lgb, oof_xgb, oof_cb, oof_tabpfn, oof_ovo], [0.2]*5))

# ── Blend search ──
print("\n[3] Blend search (OvO + TabPFN + GBDT, optimize LOMO)...", flush=True)
comps = [oof_ovo, oof_tabpfn, oof_3rp]
best_l = -1; best_c = None
for pw in [1.0, 1.5, 2.0]:
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            w2 = round(1 - w0 - w1, 2)
            if w2 < -0.01: continue
            b = rpe(comps, [w0, w1, w2], pw)
            l, _ = lomo(b); s, pc = compute_map(y, b)
            if l > best_l: best_l = l; best_c = (w0, w1, w2, pw, s, pc, l)

w0, w1, w2, pw, s, pc, l = best_c
print(f"  Best LOMO blend: OvO={w0:.1f} TabPFN={w1:.1f} GBDT={w2:.1f} power={pw}", flush=True)
print(f"  SKF={s:.4f} LOMO={l:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}", flush=True)

# Also optimize SKF
best_s = -1; best_cs = None
for pw in [1.0, 1.5, 2.0]:
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            w2 = round(1 - w0 - w1, 2)
            if w2 < -0.01: continue
            b = rpe(comps, [w0, w1, w2], pw)
            s, pc = compute_map(y, b)
            if s > best_s: best_s = s; best_cs = (w0, w1, w2, pw, s, pc)

w0, w1, w2, pw, s, pc = best_cs
print(f"  Best SKF blend:  OvO={w0:.1f} TabPFN={w1:.1f} GBDT={w2:.1f} power={pw}", flush=True)
print(f"  SKF={s:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}", flush=True)

# ── TRUE LOMO ──
print(f"\n[4] TRUE LOMO validation...", flush=True)

LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
             subsample=0.7, colsample_bytree=0.6, random_state=42, verbose=-1, n_jobs=1)

def true_lomo(name, X_data, y_labels, mask=None):
    oof = np.zeros((len(y), N))
    for m in sorted(set(months)):
        va = months == m; tr = ~va
        if mask is not None: tr = tr & mask
        clf = lgb.LGBMClassifier(**LGB_P)
        clf.fit(X_data[tr], y_labels[tr])
        oof[va] = clf.predict_proba(X_data[va])
    _, pc = compute_map(y, oof)
    ms = {}
    for m in sorted(set(months)):
        mk = months == m; ms[m], _ = compute_map(y[mk], oof[mk])
    l = np.mean(list(ms.values()))
    print(f"  {name}: TRUE_LOMO={l:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} BoP={pc['Birds of Prey']:.4f}", flush=True)
    return l, pc

true_lomo("LGB original", X, y)
true_lomo("LGB relabeled", X, y_r, mask=km)

# OvO TRUE LOMO
print(f"\n  OvO TRUE LOMO (36 pairs x 4 months)...", flush=True)
from tabpfn import TabPFNClassifier
t0 = time.time()
oof_ovo_tl = np.zeros((len(y), N))
for mi, m in enumerate(sorted(set(months))):
    va = months == m; tr = (~va) & km
    print(f"    Month {m} ({mi+1}/4)...", flush=True)
    for i in range(N):
        for j in range(i + 1, N):
            pm = tr & ((y_r == i) | (y_r == j))
            if pm.sum() < 6: continue
            yp = (y_r[pm] == i).astype(int)
            clf = TabPFNClassifier(n_estimators=4, random_state=42)
            clf.fit(X[pm], yp)
            p = clf.predict_proba(X[va])
            p1 = p[:, 1] if p.shape[1] > 1 else p[:, 0]
            oof_ovo_tl[va, i] += p1
            oof_ovo_tl[va, j] += (1 - p1)

oof_ovo_tl /= np.maximum(oof_ovo_tl.sum(axis=1, keepdims=True), 1e-10)
_, pc_otl = compute_map(y, oof_ovo_tl)
ms_otl = {}
for m in sorted(set(months)):
    mk = months == m; ms_otl[m], _ = compute_map(y[mk], oof_ovo_tl[mk])
l_otl = np.mean(list(ms_otl.values()))
ms_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms_otl.items()))
print(f"  OvO TRUE_LOMO={l_otl:.4f} ({ms_str})", flush=True)
print(f"    Corm={pc_otl['Cormorants']:.4f} Wader={pc_otl['Waders']:.4f} BoP={pc_otl['Birds of Prey']:.4f}", flush=True)
print(f"    Took {time.time()-t0:.0f}s", flush=True)

# TabPFN TRUE LOMO
print(f"\n  TabPFN TRUE LOMO...", flush=True)
t0 = time.time()
oof_tab_tl = np.zeros((len(y), N))
for m in sorted(set(months)):
    va = months == m; tr = (~va) & km
    clf = TabPFNClassifier(n_estimators=8, random_state=42)
    clf.fit(X[tr], y_r[tr])
    oof_tab_tl[va] = clf.predict_proba(X[va])
_, pc_ttl = compute_map(y, oof_tab_tl)
ms_ttl = {}
for m in sorted(set(months)):
    mk = months == m; ms_ttl[m], _ = compute_map(y[mk], oof_tab_tl[mk])
l_ttl = np.mean(list(ms_ttl.values()))
print(f"  TabPFN TRUE_LOMO={l_ttl:.4f} Corm={pc_ttl['Cormorants']:.4f} Wader={pc_ttl['Waders']:.4f}", flush=True)

# ── FINAL SUMMARY ──
print(f"\n{'='*90}\nFINAL SUMMARY\n{'='*90}", flush=True)
print(f"\n  SKF (in-distribution):", flush=True)
print(f"  {'Config':35s} {'SKF':>7s} {'LOMO*':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s}", flush=True)
print(f"  {'-'*75}", flush=True)
for name, oof in [("E175 (LB=0.59)", oof_e175), ("TabPFN-ALL relabel", oof_tabpfn),
                   ("OvO Pairwise", oof_ovo), ("LGB relabeled", oof_lgb),
                   ("XGB relabeled", oof_xgb), ("CatBoost relabeled", oof_cb),
                   ("GBDT avg", oof_3avg), ("GBDT rank-power", oof_3rp)]:
    s, pc = compute_map(y, oof); l, _ = lomo(oof)
    print(f"  {name:35s} {s:7.4f} {l:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} {pc['Birds of Prey']:7.4f}", flush=True)

print(f"\n  TRUE LOMO (generalization):", flush=True)
print(f"    LGB relabeled:  0.3381 (from earlier run)")
print(f"    TabPFN relabel: {l_ttl:.4f}")
print(f"    OvO Pairwise:   {l_otl:.4f}")
print(f"\nDone.", flush=True)
