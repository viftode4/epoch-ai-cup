"""E188: Final ensemble — last day submission.

Tries 5 ensemble strategies on OOF, generates test submissions for top ones.
"""
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from src.data import ROOT, CLASSES

train = pd.read_csv(ROOT / 'data' / 'train.csv')
test = pd.read_csv(ROOT / 'data' / 'test.csv')
SUB_CLASSES = ['Clutter','Cormorants','Pigeons','Ducks','Geese','Gulls','Birds of Prey','Waders','Songbirds']
y = np.array([CLASSES.index(c) for c in train['bird_group'].values])

def macro_map(preds):
    y_bin = np.zeros((len(y), 9))
    for i in range(9):
        y_bin[:, i] = (y == i).astype(int)
    aps = [average_precision_score(y_bin[:, i], preds[:, i]) for i in range(9)]
    return np.mean(aps), aps

# Load all candidates
candidates = {}
for name, oof_f, test_f in [
    ('e79',           'oof_e79.npy',                    'test_e79.npy'),
    ('e175_best',     'oof_e175_best.npy',              'test_e175_best.npy'),
    ('e175_lgb',      'oof_e175_lgb.npy',               'test_e175_lgb.npy'),
    ('e179_best',     'oof_e179_best.npy',              'test_e179_best.npy'),
    ('e185_tpfn_rel', 'oof_e185_tabpfn_relabel.npy',    'test_e185_tabpfn_relabel.npy'),
    ('e185_tpfn_all', 'oof_e185_tabpfn_all.npy',        'test_e185_tabpfn_all.npy'),
    ('e186_ovo',      'oof_e186_ovo.npy',               'test_e186_ovo.npy'),
    ('e180_cnn',      'oof_e180_cnn.npy',               'test_e180_cnn.npy'),
    ('e187_blend',    'oof_e187_blend.npy',              'test_e187_blend.npy'),
    ('e173',          'oof_e173.npy',                    'test_e173.npy'),
    ('e179_cb',       'oof_e179_cb.npy',                'test_e179_cb.npy'),
]:
    oof = np.load(ROOT / oof_f)
    tst = np.load(ROOT / test_f)
    if oof.shape == (2601, 9) and tst.shape == (1872, 9) and oof.min() >= -0.1:
        candidates[name] = {'oof': oof, 'test': tst}

print(f"Loaded {len(candidates)} models")
names = list(candidates.keys())
oofs = [candidates[n]['oof'] for n in names]
tests = [candidates[n]['test'] for n in names]

base_m, base_aps = macro_map(candidates['e79']['oof'])
print(f"E79 baseline: mAP={base_m:.4f}")

# ---- Helpers ----
def to_ranks(preds):
    n = preds.shape[0]
    ranks = np.zeros_like(preds)
    for c in range(9):
        ranks[:, c] = preds[:, c].argsort().argsort().astype(float) / (n - 1)
    return ranks

def softmax_w(w):
    w = np.array(w)
    w = np.abs(w) / max(np.abs(w).sum(), 1e-12)
    return w

# ---- METHOD 1: Probability average ----
print("\n" + "="*60)
print("[1] PROBABILITY AVERAGE (Nelder-Mead)")
print("="*60)

def neg_map_prob(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, oofs))
    m, _ = macro_map(blend)
    return -m

w0 = np.ones(len(names)) / len(names)
res = minimize(neg_map_prob, w0, method='Nelder-Mead',
               options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-5})
w_prob = softmax_w(res.x)
blend_prob_oof = sum(wi * oof for wi, oof in zip(w_prob, oofs))
m_prob, _ = macro_map(blend_prob_oof)
print(f"mAP={m_prob:.4f} (delta={m_prob-base_m:+.4f})")
for i, n in enumerate(names):
    if w_prob[i] > 0.01:
        print(f"  {n:20s}: {w_prob[i]:.3f}")

# ---- METHOD 2: Rank average ----
print("\n" + "="*60)
print("[2] RANK AVERAGE (Nelder-Mead)")
print("="*60)

oof_ranks = [to_ranks(oof) for oof in oofs]

def neg_map_rank(w):
    w = softmax_w(w)
    blend = sum(wi * r for wi, r in zip(w, oof_ranks))
    m, _ = macro_map(blend)
    return -m

res2 = minimize(neg_map_rank, w0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-5})
w_rank = softmax_w(res2.x)
blend_rank_oof = sum(wi * r for wi, r in zip(w_rank, oof_ranks))
m_rank, _ = macro_map(blend_rank_oof)
print(f"mAP={m_rank:.4f} (delta={m_rank-base_m:+.4f})")
for i, n in enumerate(names):
    if w_rank[i] > 0.01:
        print(f"  {n:20s}: {w_rank[i]:.3f}")

# ---- METHOD 3: Geometric mean ----
print("\n" + "="*60)
print("[3] GEOMETRIC MEAN (Nelder-Mead)")
print("="*60)

def neg_map_geo(w):
    w = softmax_w(w)
    log_blend = sum(wi * np.log(np.clip(oof, 1e-10, None)) for wi, oof in zip(w, oofs))
    blend = np.exp(log_blend)
    blend = blend / blend.sum(axis=1, keepdims=True)
    m, _ = macro_map(blend)
    return -m

res3 = minimize(neg_map_geo, w0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-5})
w_geo = softmax_w(res3.x)
log_blend_oof = sum(wi * np.log(np.clip(oof, 1e-10, None)) for wi, oof in zip(w_geo, oofs))
blend_geo_oof = np.exp(log_blend_oof)
blend_geo_oof = blend_geo_oof / blend_geo_oof.sum(axis=1, keepdims=True)
m_geo, _ = macro_map(blend_geo_oof)
print(f"mAP={m_geo:.4f} (delta={m_geo-base_m:+.4f})")
for i, n in enumerate(names):
    if w_geo[i] > 0.01:
        print(f"  {n:20s}: {w_geo[i]:.3f}")

# ---- METHOD 4: Per-class rank optimization ----
print("\n" + "="*60)
print("[4] PER-CLASS RANK (pairwise optimization)")
print("="*60)

perclass_selections = {}
blend_pcr_oof = np.zeros((2601, 9))
for c in range(9):
    y_bin_c = (y == c).astype(int)
    best_ap = 0
    best_config = (0, 0, 0.0)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            for alpha in np.arange(0.0, 1.05, 0.1):
                col = (1-alpha) * oof_ranks[i][:, c] + alpha * oof_ranks[j][:, c]
                ap = average_precision_score(y_bin_c, col)
                if ap > best_ap:
                    best_ap = ap
                    best_config = (i, j, alpha)
    bi, bj, ba = best_config
    blend_pcr_oof[:, c] = (1-ba) * oof_ranks[bi][:, c] + ba * oof_ranks[bj][:, c]
    perclass_selections[c] = best_config
    print(f"  {CLASSES[c]:15s}: {names[bi]}({1-ba:.1f}) + {names[bj]}({ba:.1f}) = {best_ap:.3f}")

m_pcr, _ = macro_map(blend_pcr_oof)
print(f"  Overall mAP={m_pcr:.4f} (delta={m_pcr-base_m:+.4f})")

# ---- METHOD 5: Targeted surgery ----
print("\n" + "="*60)
print("[5] TARGETED SURGERY (BoP/Corm/Waders)")
print("="*60)

weak = [CLASSES.index('Birds of Prey'), CLASSES.index('Cormorants'), CLASSES.index('Waders')]
best_alpha_surg = 0
best_m_surg = 0
for alpha in np.arange(0.05, 0.50, 0.05):
    blended = candidates['e79']['oof'].copy()
    for c in weak:
        blended[:, c] = (1-alpha) * candidates['e79']['oof'][:, c] + alpha * candidates['e186_ovo']['oof'][:, c]
    m, _ = macro_map(blended)
    if m > best_m_surg:
        best_m_surg = m
        best_alpha_surg = alpha
    print(f"  alpha={alpha:.2f}: mAP={m:.4f}")
print(f"  Best: alpha={best_alpha_surg:.2f}, mAP={best_m_surg:.4f} (delta={best_m_surg-base_m:+.4f})")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}")
print(f"SUMMARY (E79 baseline = {base_m:.4f})")
print(f"{'='*60}")
results = [
    ("[1] Prob avg", m_prob),
    ("[2] Rank avg", m_rank),
    ("[3] Geo mean", m_geo),
    ("[4] Per-class rank", m_pcr),
    ("[5] Targeted surg", best_m_surg),
]
for name, m in sorted(results, key=lambda x: -x[1]):
    print(f"  {name:25s}: {m:.4f}  (delta={m-base_m:+.4f})")

# ================================================================
# GENERATE SUBMISSIONS
# ================================================================
print(f"\n{'='*60}")
print("GENERATING SUBMISSIONS")
print(f"{'='*60}")

track_ids = test['track_id'].values
test_ranks = [to_ranks(t) for t in tests]

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e188_{tag}.csv"
    df.to_csv(fname, index=False)
    df.to_csv(ROOT / "submission.csv", index=False)
    print(f"  Saved {fname}")

# 1. Rank ensemble
test_rank_blend = sum(wi * r for wi, r in zip(w_rank, test_ranks))
save_sub(test_rank_blend, "rank_ensemble")

# 2. Per-class rank
test_pcr = np.zeros((1872, 9))
for c in range(9):
    bi, bj, ba = perclass_selections[c]
    test_pcr[:, c] = (1-ba) * test_ranks[bi][:, c] + ba * test_ranks[bj][:, c]
save_sub(test_pcr, "perclass_rank")

# 3. Prob ensemble
test_prob_blend = sum(wi * t for wi, t in zip(w_prob, tests))
save_sub(test_prob_blend, "prob_ensemble")

# 4. Geo ensemble
log_test = sum(wi * np.log(np.clip(t, 1e-10, None)) for wi, t in zip(w_geo, tests))
test_geo = np.exp(log_test)
test_geo = test_geo / test_geo.sum(axis=1, keepdims=True)
save_sub(test_geo, "geo_ensemble")

# 5. Targeted surgery
test_surg = candidates['e79']['test'].copy()
for c in weak:
    test_surg[:, c] = (1-best_alpha_surg) * candidates['e79']['test'][:, c] + best_alpha_surg * candidates['e186_ovo']['test'][:, c]
save_sub(test_surg, "targeted_surgery")

print("\nDone! 5 submissions generated.")
