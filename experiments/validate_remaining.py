"""Remaining tests: TabPFN and soft-label distillation on TRUE LOMO.
Tests 5-11 from validate_all_fixes.py (1-4 already completed)."""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from src.data import load_train, CLASSES
from src.metrics import compute_map

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(set(months))

# Features
feats = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
X_all = np.nan_to_num(feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)

from src.features import ALL_TEMPORAL
wx_kw = ['wx_', 'wxe_', 'sol_', 'era5_', 'soil_', 'moon_', 'tide_', 'wave_',
         'fog', 'rain', 'wmo_', 'pressure', 'humidity', 'temp_', 'daylength',
         'cin', 'cape', 'insect', 'crepuscular', 'cloud', 'precip', 'sunshine',
         'wind_180', 'over_water']
phys_cols = [f for f in feats.columns if not any(k in f.lower() for k in wx_kw) and f not in ALL_TEMPORAL]
X_phys = np.nan_to_num(feats[phys_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)

sel100 = [l.strip() for l in open(ROOT / "data/best_features_e175.txt") if l.strip()]
sel100 = [f for f in sel100 if f in feats.columns]
X_100 = np.nan_to_num(feats[sel100].values.astype(np.float32), nan=0, posinf=0, neginf=0)

print(f"ALL={X_all.shape[1]}, phys={X_phys.shape[1]}, sel100={X_100.shape[1]}", flush=True)

# Load cleanlab cache
cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
agreed_noisy = cache['agreed_noisy'].tolist()
quality = cache['quality']
consensus_labels = cache['consensus_labels']

y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])

def true_lomo(name, train_fn):
    t0 = time.time()
    oof = np.zeros((len(y), N))
    for m in unique_months:
        va = months == m
        tr = ~va
        try:
            oof[va] = train_fn(tr, va)
        except Exception as e:
            print(f"    ERROR month {m}: {e}", flush=True)
    overall, pc = compute_map(y, oof)
    ms = {}
    for m in unique_months:
        mask = months == m
        if mask.sum() >= 5:
            s, _ = compute_map(y[mask], oof[mask])
            ms[m] = s
    lomo = np.mean(list(ms.values()))
    elapsed = time.time() - t0
    month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms.items()))
    print(f"\n  {name} ({elapsed:.0f}s):", flush=True)
    print(f"    TRUE LOMO={lomo:.4f}  ({month_str})", flush=True)
    print(f"    Corm={pc['Cormorants']:.4f}  Wader={pc['Waders']:.4f}  BoP={pc['Birds of Prey']:.4f}  "
          f"Gulls={pc['Gulls']:.4f}  Geese={pc['Geese']:.4f}  Ducks={pc['Ducks']:.4f}", flush=True)
    return lomo, pc

results = []

# Previous results for reference
print("PREVIOUS RESULTS (from validate_corm_fix.py):", flush=True)
print("  [2] LGB-100 baseline:      TRUE LOMO=0.3350  Corm=0.0347", flush=True)
print("  [3] LGB-100 + conf weights: TRUE LOMO=0.3407  Corm=0.0556", flush=True)
print("  [4] LGB-100 relabel+clean:  TRUE LOMO=0.3370  Corm=0.0285", flush=True)

# 5. TabPFN ALL features
print("\n[5] TabPFN ALL features", flush=True)
from tabpfn import TabPFNClassifier
def tabpfn_all(tr, va):
    m = TabPFNClassifier(n_estimators=8, random_state=42)
    m.fit(X_all[tr], y[tr])
    return m.predict_proba(X_all[va])
l5, p5 = true_lomo("[5] TabPFN-ALL", tabpfn_all)
results.append(("[5] TabPFN-ALL", l5, p5))

# 6. TabPFN physics-only
print("\n[6] TabPFN physics-only", flush=True)
def tabpfn_phys(tr, va):
    m = TabPFNClassifier(n_estimators=8, random_state=42)
    m.fit(X_phys[tr], y[tr])
    return m.predict_proba(X_phys[va])
l6, p6 = true_lomo("[6] TabPFN-phys", tabpfn_phys)
results.append(("[6] TabPFN-phys", l6, p6))

# 7. TabPFN ALL + relabeled
print("\n[7] TabPFN ALL + relabeled", flush=True)
def tabpfn_relabel(tr, va):
    tr_use = tr & keep_mask
    m = TabPFNClassifier(n_estimators=8, random_state=42)
    m.fit(X_all[tr_use], y_relabeled[tr_use])
    return m.predict_proba(X_all[va])
l7, p7 = true_lomo("[7] TabPFN-ALL relabeled", tabpfn_relabel)
results.append(("[7] TabPFN-ALL relabel", l7, p7))

# 8. Soft-label OvR distillation
print("\n[8] Soft-label OvR distillation", flush=True)
import lightgbm as lgb
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")
def soft_ovr(tr, va):
    preds = np.zeros((va.sum(), N))
    for c in range(N):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               subsample=0.7, colsample_bytree=0.6,
                               random_state=42, verbose=-1, n_jobs=1)
        m.fit(X_100[tr], oof_tabpfn[tr, c])
        preds[:, c] = np.clip(m.predict(X_100[va]), 0, 1)
    row_sums = preds.sum(axis=1, keepdims=True)
    return preds / np.maximum(row_sums, 1e-10)
l8, p8 = true_lomo("[8] Soft OvR distill", soft_ovr)
results.append(("[8] Soft OvR distill", l8, p8))

# 9. Soft OvR + relabeled targets
print("\n[9] Soft OvR + corrected targets", flush=True)
soft_corrected = oof_tabpfn.copy()
for idx in agreed_noisy:
    soft_corrected[idx] = 0
    soft_corrected[idx, consensus_labels[idx]] = 1.0
def soft_ovr_corrected(tr, va):
    preds = np.zeros((va.sum(), N))
    for c in range(N):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               subsample=0.7, colsample_bytree=0.6,
                               random_state=42, verbose=-1, n_jobs=1)
        m.fit(X_100[tr], soft_corrected[tr, c])
        preds[:, c] = np.clip(m.predict(X_100[va]), 0, 1)
    row_sums = preds.sum(axis=1, keepdims=True)
    return preds / np.maximum(row_sums, 1e-10)
l9, p9 = true_lomo("[9] Soft OvR corrected", soft_ovr_corrected)
results.append(("[9] Soft OvR corrected", l9, p9))

# Summary
print("\n\n" + "=" * 90, flush=True)
print("FULL SUMMARY (all tests, TRUE LOMO)", flush=True)
print("=" * 90, flush=True)
print(f"\n  Previous (LGB-100):", flush=True)
print(f"    [2] Baseline:      0.3350  Corm=0.035  Wader=0.051", flush=True)
print(f"    [3] Conf weights:  0.3407  Corm=0.056  Wader=0.047", flush=True)
print(f"    [4] Relabel+clean: 0.3370  Corm=0.029  Wader=0.054", flush=True)

print(f"\n  {'Config':35s} {'LOMO':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Gulls':>7s}", flush=True)
print(f"  {'-'*75}", flush=True)
for name, l, p in results:
    print(f"  {name:35s} {l:7.4f} {p['Cormorants']:7.4f} {p['Waders']:7.4f} "
          f"{p['Birds of Prey']:7.4f} {p['Gulls']:7.4f}", flush=True)
