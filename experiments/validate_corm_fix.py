"""Validate: does fixing Cormorant labels help? TRUE LOMO (train 3 months, predict 4th)."""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np
import pandas as pd
from collections import Counter
from src.data import load_train, CLASSES
from src.metrics import compute_map
from scipy.special import softmax
from cleanlab.filter import find_label_issues
import lightgbm as lgb

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
N = len(CLASSES)
CORM = 2

# 100 stability-selected features
feats = pd.read_pickle("G:/Projects/epoch-ai-cup/data/_cached_train_features_v3.pkl")
sel = [l.strip() for l in open("G:/Projects/epoch-ai-cup/data/best_features_e175.txt") if l.strip()]
sel = [f for f in sel if f in feats.columns]
X = np.nan_to_num(feats[sel].values.astype(np.float32), nan=0, posinf=0, neginf=0)

# Consensus from 4 models
oof_t = np.load("G:/Projects/epoch-ai-cup/oof_e183_tabpfn.npy")
oof_e = softmax(np.load("G:/Projects/epoch-ai-cup/oof_e175_best.npy"), axis=1)
oof_c = np.load("G:/Projects/epoch-ai-cup/oof_e175_cb.npy")
oof_r = np.load("G:/Projects/epoch-ai-cup/oof_e175_ranker.npy")

def consensus(idx):
    preds = [oof_t[idx].argmax(), oof_e[idx].argmax(), oof_c[idx].argmax(), oof_r[idx].argmax()]
    v = Counter(preds).most_common(1)[0]
    return v[0], v[1]

# Agreed noisy labels
issues_t = set(find_label_issues(labels=y, pred_probs=oof_t, return_indices_ranked_by="self_confidence", n_jobs=1))
issues_e = set(find_label_issues(labels=y, pred_probs=oof_e, return_indices_ranked_by="self_confidence", n_jobs=1))
agreed = sorted(issues_t & issues_e)
print(f"Agreed noisy: {len(agreed)}", flush=True)

# Classify Cormorants
corm_idx = np.where(y == CORM)[0]
real, relabel, ambig = [], [], []
for idx in corm_idx:
    cl, ag = consensus(idx)
    if cl == CORM:
        real.append(idx)
    elif ag >= 3 and oof_t[idx, cl] > 0.3:
        relabel.append((idx, cl))
    else:
        ambig.append(idx)
print(f"Cormorants: {len(real)} real, {len(relabel)} relabel, {len(ambig)} ambiguous", flush=True)

# TRUE LOMO
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                  subsample=0.7, colsample_bytree=0.6,
                  class_weight="balanced", random_state=42, verbose=-1, n_jobs=1)

def true_lomo(X_data, y_data, months_data, label, keep_mask=None):
    t0 = time.time()
    uniq = sorted(set(months_data))
    oof = np.zeros((len(y), N))  # always full size for eval against original y
    for m in uniq:
        va = months_data == m
        if keep_mask is not None:
            tr = (~va) & keep_mask
        else:
            tr = ~va
        # Ensure enough classes
        tr_classes = len(set(y_data[tr]))
        if tr_classes < 2:
            continue
        mdl = lgb.LGBMClassifier(**LGB_PARAMS)
        mdl.fit(X_data[tr], y_data[tr])
        oof[va] = mdl.predict_proba(X_data[va])

    # Evaluate on ORIGINAL labels (y, not y_data) for honest comparison
    overall, pc = compute_map(y, oof)
    ms = {}
    for m in uniq:
        mask = months_data == m
        if mask.sum() >= 5:
            s, _ = compute_map(y[mask], oof[mask])
            ms[m] = s
    lomo = np.mean(list(ms.values()))
    elapsed = time.time() - t0

    print(f"\n  {label} ({elapsed:.0f}s):", flush=True)
    print(f"    TRUE LOMO={lomo:.4f}  SKF-approx={overall:.4f}", flush=True)
    print(f"    Months: {' '.join(f'm{m}={v:.3f}' for m,v in sorted(ms.items()))}", flush=True)
    for c in CLASSES:
        mk = " ***" if c == "Cormorants" else " **" if c == "Waders" else ""
        print(f"    {c:15s}: {pc[c]:.4f}{mk}", flush=True)
    return lomo, pc

print("=" * 80, flush=True)
print("CORMORANT FIX: TRUE LOMO VALIDATION", flush=True)
print("=" * 80, flush=True)

# 1. Baseline
l1, p1 = true_lomo(X, y, months, "[1] Baseline")

# 2. Relabel high-confidence Corm mislabels
y2 = y.copy()
for idx, nl in relabel:
    y2[idx] = nl
l2, p2 = true_lomo(X, y2, months, f"[2] Relabel {len(relabel)} Corm mislabels")

# 3. Relabel ALL 133 agreed noise
y3 = y.copy()
for idx in agreed:
    cl, _ = consensus(idx)
    y3[idx] = cl
l3, p3 = true_lomo(X, y3, months, f"[3] Relabel ALL {len(agreed)} noise")

# 4. Remove Corm mislabels
rm_set = set(idx for idx, _ in relabel)
keep4 = np.array([i not in rm_set for i in range(len(y))])
l4, p4 = true_lomo(X, y, months, f"[4] Remove {len(rm_set)} Corm mislabels", keep_mask=keep4)

# 5. Soft-approx: relabel ALL suspect Corm to TabPFN argmax
y5 = y.copy()
for idx, nl in relabel:
    y5[idx] = nl
for idx in ambig:
    y5[idx] = oof_t[idx].argmax()
l5, p5 = true_lomo(X, y5, months, f"[5] Relabel ALL {len(relabel)+len(ambig)} suspect Corm")

# 6. Relabel ALL noise + remove extreme (quality < 0.02)
from cleanlab.rank import get_label_quality_scores
qual = get_label_quality_scores(labels=y, pred_probs=oof_t)
y6 = y3.copy()  # start from relabeled
extreme = np.where(qual < 0.02)[0]
keep6 = np.array([i not in set(extreme) for i in range(len(y))])
l6, p6 = true_lomo(X, y6, months, f"[6] Relabel ALL + remove {len(extreme)} extreme", keep_mask=keep6)

# Summary
print("\n" + "=" * 80, flush=True)
print("SUMMARY", flush=True)
print("=" * 80, flush=True)
print(f"{'Config':50s} {'LOMO':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Delta':>7s}", flush=True)
print("-" * 90, flush=True)
for name, l, p in [
    ("[1] Baseline", l1, p1),
    (f"[2] Relabel {len(relabel)} Corm mislabels", l2, p2),
    (f"[3] Relabel ALL {len(agreed)} noise", l3, p3),
    (f"[4] Remove {len(rm_set)} Corm mislabels", l4, p4),
    (f"[5] Relabel ALL suspect Corm", l5, p5),
    (f"[6] Relabel ALL + remove extreme", l6, p6),
]:
    d = l - l1
    print(f"  {name:50s} {l:7.4f} {p['Cormorants']:7.4f} {p['Waders']:7.4f} {p['Birds of Prey']:7.4f} {d:+7.4f}", flush=True)
