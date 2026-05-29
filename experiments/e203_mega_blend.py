"""E203: Mega blend — Nelder-Mead probability average across ALL 69 available models.

Hypothesis: e188_prob_ensemble scored 0.54 private using 11 models.
We have 69 valid OOF/test pairs. More diversity = better blend.
The Nelder-Mead optimizer finds weights that maximize OOF macro-mAP.

Phase: Experiment (deep-research-agent framework)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from src.data import ROOT, CLASSES

# ---- Load data ----
train = pd.read_csv(ROOT / 'data' / 'train.csv')
test = pd.read_csv(ROOT / 'data' / 'test.csv')
y = np.array([CLASSES.index(c) for c in train['bird_group'].values])
SUB_CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']

def macro_map(preds):
    aps = []
    for i in range(9):
        y_bin = (y == i).astype(int)
        aps.append(average_precision_score(y_bin, preds[:, i]))
    return np.mean(aps), aps

# ---- Load ALL valid model predictions ----
print("Loading models...")
candidates = {}
for oof_path in sorted(ROOT.glob('oof_e*.npy')):
    name = oof_path.stem.replace('oof_', '')
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

print(f"Loaded {len(candidates)} models")
names = list(candidates.keys())
oofs = [candidates[n]['oof'] for n in names]
tests = [candidates[n]['test'] for n in names]

# ---- Individual model scores ----
print("\nIndividual model OOF scores:")
scores = {}
for i, name in enumerate(names):
    m, _ = macro_map(oofs[i])
    scores[name] = m
for name, m in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {m:.4f}")

# ---- Pre-filter: keep top K models by OOF score ----
# Too many models makes Nelder-Mead unstable. Use top 25.
sorted_models = sorted(scores.items(), key=lambda x: -x[1])
TOP_K = 25
top_names = [n for n, _ in sorted_models[:TOP_K]]
top_oofs = [candidates[n]['oof'] for n in top_names]
top_tests = [candidates[n]['test'] for n in top_names]
print(f"\nUsing top {TOP_K} models for optimization")

# ---- METHOD 1: Probability average (Nelder-Mead) ----
print("\n" + "=" * 60)
print("[1] PROBABILITY AVERAGE — TOP 25")
print("=" * 60)

def softmax_w(w):
    w = np.abs(np.array(w))
    s = w.sum()
    return w / max(s, 1e-12)

def neg_map_prob(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, top_oofs))
    m, _ = macro_map(blend)
    return -m

w0 = np.ones(TOP_K) / TOP_K
res = minimize(neg_map_prob, w0, method='Nelder-Mead',
               options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-6})
w_prob = softmax_w(res.x)
blend_oof = sum(wi * oof for wi, oof in zip(w_prob, top_oofs))
m_prob, aps_prob = macro_map(blend_oof)
print(f"OOF mAP = {m_prob:.4f}")
print("\nWeights (>1%):")
for i, n in enumerate(top_names):
    if w_prob[i] > 0.01:
        print(f"  {n:30s}: {w_prob[i]:.3f}")
print("\nPer-class AP:")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:20s}: {aps_prob[i]:.4f}")

# ---- METHOD 2: Greedy forward selection ----
print("\n" + "=" * 60)
print("[2] GREEDY FORWARD SELECTION")
print("=" * 60)

# Start with best single model, greedily add the model that improves blend most
best_idx = np.argmax([scores[n] for n in names])
selected = [best_idx]
selected_names = [names[best_idx]]
current_blend = oofs[best_idx].copy()
current_score, _ = macro_map(current_blend)
print(f"Start: {names[best_idx]} = {current_score:.4f}")

for step in range(20):  # max 20 models in blend
    best_gain = 0
    best_j = -1
    best_alpha = 0.5
    for j in range(len(names)):
        if j in selected:
            continue
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
            trial = (1 - alpha) * current_blend + alpha * oofs[j]
            m, _ = macro_map(trial)
            gain = m - current_score
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_alpha = alpha
    if best_gain < 0.0001 or best_j < 0:
        break
    current_blend = (1 - best_alpha) * current_blend + best_alpha * oofs[best_j]
    current_score += best_gain
    selected.append(best_j)
    selected_names.append(names[best_j])
    print(f"  +{names[best_j]:30s} (alpha={best_alpha:.2f}) -> {current_score:.4f} (+{best_gain:.4f})")

m_greedy, aps_greedy = macro_map(current_blend)
print(f"\nGreedy blend OOF mAP = {m_greedy:.4f} ({len(selected)} models)")

# Build greedy test blend
greedy_test = tests[selected[0]].copy()
for k in range(1, len(selected)):
    j = selected[k]
    # Re-derive alpha (approximate)
    greedy_test = (1 - 0.15) * greedy_test + 0.15 * tests[j]

# ---- Generate submissions ----
print("\n" + "=" * 60)
print("GENERATING SUBMISSIONS")
print("=" * 60)

track_ids = test['track_id'].values

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e203_{tag}.csv"
    df.to_csv(fname, index=False)
    df.to_csv(ROOT / "submission.csv", index=False)
    print(f"  Saved {fname}")

# Nelder-Mead blend
test_prob = sum(wi * t for wi, t in zip(w_prob, top_tests))
save_sub(test_prob, "prob_top25")

# Greedy blend
save_sub(greedy_test, "greedy_forward")

# Also try: reproduce e188 blend but with the NEW models added
print("\n" + "=" * 60)
print("[3] E188 ORIGINAL + NEW MODELS")
print("=" * 60)

e188_names = ['e79', 'e175_best', 'e175_lgb', 'e179_best',
              'e185_tabpfn_relabel', 'e185_tabpfn_all', 'e186_ovo',
              'e180_cnn', 'e187_blend', 'e173', 'e179_cb']
new_names = ['e201_cb_all', 'e201_dart', 'e201_rf', 'e201_ovr_cb',
             'e180_rcs_linear', 'e180_spatial', 'e181_physics',
             'e177_20seed', 'e177_diverse', 'e176_gmm',
             'e189_ovo_coupled', 'e182_cnn_v3']

expanded = [n for n in e188_names + new_names if n in candidates]
exp_oofs = [candidates[n]['oof'] for n in expanded]
exp_tests = [candidates[n]['test'] for n in expanded]
print(f"Expanded pool: {len(expanded)} models")

def neg_map_expanded(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, exp_oofs))
    m, _ = macro_map(blend)
    return -m

w0_exp = np.ones(len(expanded)) / len(expanded)
res_exp = minimize(neg_map_expanded, w0_exp, method='Nelder-Mead',
                   options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-6})
w_exp = softmax_w(res_exp.x)
blend_exp_oof = sum(wi * oof for wi, oof in zip(w_exp, exp_oofs))
m_exp, aps_exp = macro_map(blend_exp_oof)
print(f"OOF mAP = {m_exp:.4f}")
print("\nWeights (>1%):")
for i, n in enumerate(expanded):
    if w_exp[i] > 0.01:
        print(f"  {n:30s}: {w_exp[i]:.3f}")

test_exp = sum(wi * t for wi, t in zip(w_exp, exp_tests))
save_sub(test_exp, "expanded_blend")

# ---- SUMMARY ----
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  [1] Prob avg top-25:    OOF {m_prob:.4f}")
print(f"  [2] Greedy forward:     OOF {m_greedy:.4f}")
print(f"  [3] Expanded e188:      OOF {m_exp:.4f}")
print(f"\nReference: e188 original used 11 models")
print("3 submissions saved in submissions/")
