"""Validate ALL Cormorant fixes on TRUE LOMO. Compare to E175 baseline (LB 0.59).

Tests:
  1. E175 baseline (OvR LambdaRank OOF — the model that gets 0.59 on Kaggle)
  2. LGB baseline (simple multiclass — calibration point)
  3. LGB + confidence weighting (cleanlab quality as sample weight)
  4. LGB + relabeled data (config 6: relabel all noise + remove extreme)
  5. TabPFN baseline (ALL 327 features)
  6. TabPFN physics-only (268 features, no weather)
  7. TabPFN + relabeled data
  8. OvR soft regression (soft-label distillation approximation)

All evaluated with TRUE LOMO (train 3 months, predict 4th).
Cleanlab results cached to data/_cleanlab_cache.npz for reuse.
"""
import sys, time, os
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from src.data import load_train, CLASSES
from src.metrics import compute_map
from scipy.special import softmax

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)
CORM = 2

# ── Load data ──
train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(set(months))

# ── Load features ──
feats = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
sel100 = [l.strip() for l in open(ROOT / "data/best_features_e175.txt") if l.strip()]
sel100 = [f for f in sel100 if f in feats.columns]

X_100 = np.nan_to_num(feats[sel100].values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_all = np.nan_to_num(feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)

# Physics-only: remove weather/temporal
from src.features import ALL_TEMPORAL
wx_kw = ['wx_', 'wxe_', 'sol_', 'era5_', 'soil_', 'moon_', 'tide_', 'wave_',
         'fog', 'rain', 'wmo_', 'pressure', 'humidity', 'temp_', 'daylength',
         'cin', 'cape', 'insect', 'crepuscular', 'cloud', 'precip', 'sunshine',
         'wind_180', 'over_water']
phys_cols = [f for f in feats.columns if not any(k in f.lower() for k in wx_kw) and f not in ALL_TEMPORAL]
X_phys = np.nan_to_num(feats[phys_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)

print(f"Features: 100-sel={X_100.shape[1]}, all={X_all.shape[1]}, physics={X_phys.shape[1]}", flush=True)

# ── Cleanlab: cache or load ──
cache_path = ROOT / "data/_cleanlab_cache.npz"
if cache_path.exists():
    print("Loading cached cleanlab results...", flush=True)
    cache = np.load(cache_path, allow_pickle=True)
    agreed_noisy = cache['agreed_noisy'].tolist()
    quality = cache['quality']
    consensus_labels = cache['consensus_labels']
else:
    print("Running cleanlab (will cache)...", flush=True)
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    oof_t = np.load(ROOT / "oof_e183_tabpfn.npy")
    oof_e = softmax(np.load(ROOT / "oof_e175_best.npy"), axis=1)
    oof_c = np.load(ROOT / "oof_e175_cb.npy")
    oof_r = np.load(ROOT / "oof_e175_ranker.npy")

    issues_t = set(find_label_issues(labels=y, pred_probs=oof_t, return_indices_ranked_by="self_confidence", n_jobs=1))
    issues_e = set(find_label_issues(labels=y, pred_probs=oof_e, return_indices_ranked_by="self_confidence", n_jobs=1))
    agreed_noisy = sorted(issues_t & issues_e)
    quality = get_label_quality_scores(labels=y, pred_probs=oof_t)

    # Consensus labels from 4 models
    consensus_labels = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        preds = [oof_t[i].argmax(), oof_e[i].argmax(), oof_c[i].argmax(), oof_r[i].argmax()]
        consensus_labels[i] = Counter(preds).most_common(1)[0][0]

    np.savez(cache_path, agreed_noisy=np.array(agreed_noisy), quality=quality, consensus_labels=consensus_labels)
    print(f"  Cached to {cache_path}", flush=True)

print(f"Agreed noisy: {len(agreed_noisy)}", flush=True)

# ── Build label variants ──
# Config 6 from previous test: relabel all noise + remove extreme
y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])
print(f"Config6: relabeled {np.sum(y_relabeled != y)}, removing {len(extreme)} extreme", flush=True)

# ── TRUE LOMO evaluation ──
def true_lomo(name, train_fn, eval_y=None):
    """Train with train_fn(X_tr, y_tr, X_va) for each held-out month. Eval on original y."""
    if eval_y is None:
        eval_y = y
    t0 = time.time()
    oof = np.zeros((len(eval_y), N))

    for m in unique_months:
        va = months == m
        tr = ~va
        try:
            oof[va] = train_fn(tr, va)
        except Exception as e:
            print(f"    ERROR month {m}: {e}", flush=True)

    overall, pc = compute_map(eval_y, oof)
    ms = {}
    for m in unique_months:
        mask = months == m
        if mask.sum() >= 5:
            s, _ = compute_map(eval_y[mask], oof[mask])
            ms[m] = s
    lomo = np.mean(list(ms.values()))
    elapsed = time.time() - t0

    month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms.items()))
    print(f"\n  {name} ({elapsed:.0f}s):", flush=True)
    print(f"    TRUE LOMO={lomo:.4f}  ({month_str})", flush=True)
    print(f"    Corm={pc['Cormorants']:.4f}  Wader={pc['Waders']:.4f}  BoP={pc['Birds of Prey']:.4f}  "
          f"Gulls={pc['Gulls']:.4f}  Geese={pc['Geese']:.4f}  Ducks={pc['Ducks']:.4f}  "
          f"Pigeons={pc['Pigeons']:.4f}  Songbirds={pc['Songbirds']:.4f}  Clutter={pc['Clutter']:.4f}", flush=True)
    return lomo, pc

# ── Model builders ──
import lightgbm as lgb

LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
             subsample=0.7, colsample_bytree=0.6, class_weight="balanced",
             random_state=42, verbose=-1, n_jobs=1)

def make_lgb(X, y_train, weights=None, mask=None):
    """Returns a train_fn for LGB."""
    def train_fn(tr, va):
        tr_use = tr if mask is None else tr & mask
        w = weights[tr_use] if weights is not None else None
        m = lgb.LGBMClassifier(**LGB_P)
        m.fit(X[tr_use], y_train[tr_use], sample_weight=w)
        return m.predict_proba(X[va])
    return train_fn

def make_tabpfn(X, y_train, mask=None):
    """Returns a train_fn for TabPFN."""
    from tabpfn import TabPFNClassifier
    def train_fn(tr, va):
        tr_use = tr if mask is None else tr & mask
        m = TabPFNClassifier(n_estimators=8, random_state=42)
        m.fit(X[tr_use], y_train[tr_use])
        return m.predict_proba(X[va])
    return train_fn

def make_soft_ovr(X, soft_targets, mask=None):
    """OvR regression with soft targets (distillation)."""
    def train_fn(tr, va):
        tr_use = tr if mask is None else tr & mask
        preds = np.zeros((va.sum(), N))
        for c in range(N):
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                   subsample=0.7, colsample_bytree=0.6,
                                   random_state=42, verbose=-1, n_jobs=1)
            m.fit(X[tr_use], soft_targets[tr_use, c])
            preds[:, c] = np.clip(m.predict(X[va]), 0, 1)
        # Normalize rows
        row_sums = preds.sum(axis=1, keepdims=True)
        preds = preds / np.maximum(row_sums, 1e-10)
        return preds
    return train_fn

# ══════════════════════════════════════════════════════════════════════
print("=" * 90, flush=True)
print("COMPREHENSIVE CORMORANT FIX VALIDATION (TRUE LOMO)", flush=True)
print("=" * 90, flush=True)

results = []

# 1. E175 baseline (the model that gets 0.59 on Kaggle) — post-hoc LOMO on its OOF
print("\n[1] E175 BASELINE (post-hoc LOMO on LB=0.59 OOF — reference only)", flush=True)
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
_, pc1 = compute_map(y, oof_e175)
ms1 = {}
for m in unique_months:
    mask = months == m
    s, _ = compute_map(y[mask], oof_e175[mask])
    ms1[m] = s
fake_lomo = np.mean(list(ms1.values()))
print(f"  E175 FAKE LOMO={fake_lomo:.4f} (reference — this is NOT true LOMO)", flush=True)
print(f"  Corm={pc1['Cormorants']:.4f}  Wader={pc1['Waders']:.4f}", flush=True)

# 2. LGB baseline (100 features, original labels)
print("\n[2] LGB BASELINE (100 features, original labels)", flush=True)
l2, p2 = true_lomo("[2] LGB-100 baseline", make_lgb(X_100, y))
results.append(("[2] LGB-100 baseline", l2, p2))

# 3. LGB + confidence weighting
print("\n[3] LGB + confidence weighting", flush=True)
l3, p3 = true_lomo("[3] LGB-100 + conf weights", make_lgb(X_100, y, weights=quality))
results.append(("[3] LGB + conf weights", l3, p3))

# 4. LGB + relabeled (config 6)
print("\n[4] LGB + relabeled + remove extreme", flush=True)
l4, p4 = true_lomo("[4] LGB-100 relabeled+clean", make_lgb(X_100, y_relabeled, mask=keep_mask))
results.append(("[4] LGB relabel+clean", l4, p4))

# 5. LGB ALL features
print("\n[5] LGB ALL 327 features", flush=True)
l5, p5 = true_lomo("[5] LGB-ALL baseline", make_lgb(X_all, y))
results.append(("[5] LGB-ALL baseline", l5, p5))

# 6. LGB ALL + relabeled
print("\n[6] LGB ALL + relabeled", flush=True)
l6, p6 = true_lomo("[6] LGB-ALL relabeled+clean", make_lgb(X_all, y_relabeled, mask=keep_mask))
results.append(("[6] LGB-ALL relabel+clean", l6, p6))

# 7. TabPFN ALL features
print("\n[7] TabPFN ALL features", flush=True)
l7, p7 = true_lomo("[7] TabPFN-ALL baseline", make_tabpfn(X_all, y))
results.append(("[7] TabPFN-ALL baseline", l7, p7))

# 8. TabPFN physics-only
print("\n[8] TabPFN physics-only", flush=True)
l8, p8 = true_lomo("[8] TabPFN-phys baseline", make_tabpfn(X_phys, y))
results.append(("[8] TabPFN-phys", l8, p8))

# 9. TabPFN ALL + relabeled
print("\n[9] TabPFN ALL + relabeled", flush=True)
l9, p9 = true_lomo("[9] TabPFN-ALL relabeled+clean", make_tabpfn(X_all, y_relabeled, mask=keep_mask))
results.append(("[9] TabPFN-ALL relabel+clean", l9, p9))

# 10. Soft-label OvR distillation (using TabPFN OOF as targets)
print("\n[10] Soft-label OvR distillation", flush=True)
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")
l10, p10 = true_lomo("[10] Soft OvR distill (TabPFN targets)", make_soft_ovr(X_100, oof_tabpfn))
results.append(("[10] Soft OvR distill", l10, p10))

# 11. Soft-label OvR + relabeled targets
print("\n[11] Soft-label OvR + relabeled soft targets", flush=True)
# For relabeled samples, replace soft target with one-hot of consensus
soft_relabeled = oof_tabpfn.copy()
for idx in agreed_noisy:
    soft_relabeled[idx] = 0
    soft_relabeled[idx, consensus_labels[idx]] = 1.0
l11, p11 = true_lomo("[11] Soft OvR + relabeled targets", make_soft_ovr(X_100, soft_relabeled))
results.append(("[11] Soft OvR relabeled", l11, p11))

# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("FINAL SUMMARY (TRUE LOMO — train on 3 months, predict 4th)", flush=True)
print("=" * 90, flush=True)
print(f"\n  E175 reference (FAKE LOMO, LB=0.59): {fake_lomo:.4f}  Corm={pc1['Cormorants']:.4f}  Wader={pc1['Waders']:.4f}", flush=True)
print(f"\n  {'Config':40s} {'LOMO':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Gulls':>7s} {'Delta':>7s}", flush=True)
print(f"  {'-'*85}", flush=True)

base_lomo = results[0][1]  # LGB-100 baseline
for name, lomo_val, pc in results:
    d = lomo_val - base_lomo
    print(f"  {name:40s} {lomo_val:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} "
          f"{pc['Birds of Prey']:7.4f} {pc['Gulls']:7.4f} {d:+7.4f}", flush=True)

print(f"\nCompleted.", flush=True)
