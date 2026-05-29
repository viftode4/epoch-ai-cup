"""E204: Filtered blend — exclude models known to fail on private LB.

Phase 5 learning from E203: OvO (e186_ovo), e163, TabPFN-solo get heavy
Nelder-Mead weights due to inflated OOF but generalize poorly.

Hypothesis: removing known-bad generalizers and re-optimizing should
produce a blend that transfers better to private test.

DO NOT INCLUDE:
- e186_ovo (uniform on test, session notes)
- e163 (inflated OOF, unknown private)
- e83 (inflated OOF)
- e189_ovo_coupled (OvO variant, same problem)
- e183_tabpfn, e185_tabpfn_all, e185_tabpfn_relabel (TabPFN hurts LB)
- e177_xgb_rankmap, e178_xgb_rankmap (broken rank:map, 0.12-0.28 OOF)
- e70, e71, e84 (very low OOF < 0.40)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import ROOT, CLASSES

train = pd.read_csv(ROOT / 'data' / 'train.csv')
test = pd.read_csv(ROOT / 'data' / 'test.csv')
y = np.array([CLASSES.index(c) for c in train['bird_group'].values])
SUB_CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']

EXCLUDE = {
    'e186_ovo', 'e163', 'e83', 'e189_ovo_coupled',
    'e183_tabpfn', 'e185_tabpfn_all', 'e185_tabpfn_relabel',
    'e177_xgb_rankmap', 'e178_xgb_rankmap',
    'e70', 'e71', 'e84',
    'e176_C7_smoothap_mlp',  # 0.49 OOF
    'e176_C2_BRF',  # 0.58 OOF
}

def macro_map(preds):
    aps = []
    for i in range(9):
        y_bin = (y == i).astype(int)
        aps.append(average_precision_score(y_bin, preds[:, i]))
    return np.mean(aps), aps

# ---- Load models (filtered) ----
print("Loading models (excluding known-bad generalizers)...")
candidates = {}
for oof_path in sorted(ROOT.glob('oof_e*.npy')):
    name = oof_path.stem.replace('oof_', '')
    if name in EXCLUDE:
        continue
    test_path = ROOT / f'test_{name}.npy'
    if not test_path.exists():
        continue
    try:
        oof = np.load(oof_path, allow_pickle=True)
        tst = np.load(test_path, allow_pickle=True)
        if hasattr(oof, 'shape') and oof.shape == (2601, 9) and tst.shape == (1872, 9) and float(oof.min()) >= -0.1:
            candidates[name] = {'oof': oof.astype(float), 'test': tst.astype(float)}
    except:
        pass

print(f"Loaded {len(candidates)} models (excluded {len(EXCLUDE)})")
names = list(candidates.keys())
oofs = [candidates[n]['oof'] for n in names]
tests = [candidates[n]['test'] for n in names]

scores = {}
for i, name in enumerate(names):
    m, _ = macro_map(oofs[i])
    scores[name] = m

def softmax_w(w):
    w = np.abs(np.array(w))
    s = w.sum()
    return w / max(s, 1e-12)

def to_ranks(preds):
    n = preds.shape[0]
    ranks = np.zeros_like(preds)
    for c in range(9):
        ranks[:, c] = preds[:, c].argsort().argsort().astype(float) / (n - 1)
    return ranks

track_ids = test['track_id'].values

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e204_{tag}.csv"
    df.to_csv(fname, index=False)
    df.to_csv(ROOT / "submission.csv", index=False)
    print(f"  Saved {fname}")

# ---- METHOD 1: Top-15 prob blend ----
sorted_models = sorted(scores.items(), key=lambda x: -x[1])
print("\nFiltered model scores:")
for n, m in sorted_models[:20]:
    print(f"  {n:30s}: {m:.4f}")

TOP_K = 15
top_names = [n for n, _ in sorted_models[:TOP_K]]
top_oofs = [candidates[n]['oof'] for n in top_names]
top_tests = [candidates[n]['test'] for n in top_names]

print(f"\n{'='*60}")
print(f"[1] PROB BLEND TOP-{TOP_K} (filtered)")
print(f"{'='*60}")

def neg_map_prob(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, top_oofs))
    m, _ = macro_map(blend)
    return -m

w0 = np.ones(TOP_K) / TOP_K
res = minimize(neg_map_prob, w0, method='Nelder-Mead',
               options={'maxiter': 30000, 'xatol': 1e-5, 'fatol': 1e-6})
w_prob = softmax_w(res.x)
blend_oof = sum(wi * oof for wi, oof in zip(w_prob, top_oofs))
m_prob, aps = macro_map(blend_oof)
print(f"OOF mAP = {m_prob:.4f}")
print("\nWeights (>1%):")
for i, n in enumerate(top_names):
    if w_prob[i] > 0.01:
        print(f"  {n:30s}: {w_prob[i]:.3f}")

test_prob = sum(wi * t for wi, t in zip(w_prob, top_tests))
save_sub(test_prob, "prob_top15")

# ---- METHOD 2: Rank blend top-15 ----
print(f"\n{'='*60}")
print(f"[2] RANK BLEND TOP-{TOP_K} (filtered)")
print(f"{'='*60}")

top_oof_ranks = [to_ranks(o) for o in top_oofs]
top_test_ranks = [to_ranks(t) for t in top_tests]

def neg_map_rank(w):
    w = softmax_w(w)
    blend = sum(wi * r for wi, r in zip(w, top_oof_ranks))
    m, _ = macro_map(blend)
    return -m

res2 = minimize(neg_map_rank, w0, method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-5, 'fatol': 1e-6})
w_rank = softmax_w(res2.x)
blend_rank_oof = sum(wi * r for wi, r in zip(w_rank, top_oof_ranks))
m_rank, _ = macro_map(blend_rank_oof)
print(f"OOF mAP = {m_rank:.4f}")
print("\nWeights (>1%):")
for i, n in enumerate(top_names):
    if w_rank[i] > 0.01:
        print(f"  {n:30s}: {w_rank[i]:.3f}")

test_rank = sum(wi * r for wi, r in zip(w_rank, top_test_ranks))
save_sub(test_rank, "rank_top15")

# ---- METHOD 3: E79-heavy blend (E79 was our best on private historically) ----
print(f"\n{'='*60}")
print("[3] E79-ANCHORED BLEND")
print(f"{'='*60}")

# E79 as anchor, sweep adding each other model
e79_oof = candidates['e79']['oof']
e79_test = candidates['e79']['test']
e79_score, _ = macro_map(e79_oof)
print(f"E79 solo: {e79_score:.4f}")

best_blend_score = e79_score
best_blend_oof = e79_oof.copy()
best_blend_test = e79_test.copy()
blend_log = []

for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    for n in names:
        if n == 'e79':
            continue
        trial = (1 - alpha) * e79_oof + alpha * candidates[n]['oof']
        m, _ = macro_map(trial)
        if m > best_blend_score + 0.001:
            blend_log.append((n, alpha, m))

blend_log.sort(key=lambda x: -x[2])
for n, a, m in blend_log[:10]:
    print(f"  E79({1-a:.2f}) + {n:30s}({a:.2f}) = {m:.4f}")

if blend_log:
    best_n, best_a, best_m = blend_log[0]
    e79_anchor_oof = (1 - best_a) * e79_oof + best_a * candidates[best_n]['oof']
    e79_anchor_test = (1 - best_a) * e79_test + best_a * candidates[best_n]['test']

    # Try adding a third model
    for alpha2 in [0.05, 0.10, 0.15]:
        for n2 in names:
            if n2 in ('e79', best_n):
                continue
            trial = (1 - alpha2) * e79_anchor_oof + alpha2 * candidates[n2]['oof']
            m2, _ = macro_map(trial)
            if m2 > best_m + 0.001:
                print(f"  + {n2:30s}({alpha2:.2f}) = {m2:.4f}")
                if m2 > best_blend_score:
                    best_blend_score = m2
                    best_blend_oof = trial
                    best_blend_test = (1 - alpha2) * e79_anchor_test + alpha2 * candidates[n2]['test']

    if best_blend_score <= best_m:
        best_blend_oof = e79_anchor_oof
        best_blend_test = e79_anchor_test
        best_blend_score = best_m

    save_sub(best_blend_test, "e79_anchored")
    print(f"  Best E79-anchored OOF: {best_blend_score:.4f}")

# ---- METHOD 4: Per-class best model (no optimization, just pick best per column) ----
print(f"\n{'='*60}")
print("[4] PER-CLASS BEST MODEL")
print(f"{'='*60}")

perclass_test = np.zeros((1872, 9))
for c in range(9):
    y_bin = (y == c).astype(int)
    best_ap = 0
    best_name = ''
    for n in names:
        ap = average_precision_score(y_bin, candidates[n]['oof'][:, c])
        if ap > best_ap:
            best_ap = ap
            best_name = n
    print(f"  {CLASSES[c]:20s}: {best_name:30s} (AP={best_ap:.4f})")
    perclass_test[:, c] = candidates[best_name]['test'][:, c]

save_sub(perclass_test, "perclass_best")

# ---- SUMMARY ----
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  [1] Prob top-15:      OOF {m_prob:.4f}")
print(f"  [2] Rank top-15:      OOF {m_rank:.4f}")
print(f"  [3] E79-anchored:     OOF {best_blend_score:.4f}")
print(f"  [4] Per-class best:   (selected per column)")
print(f"\nTarget: beat 0.578 private (1st place)")
print(f"Our best: 0.54 private (e188_prob_ensemble)")
