"""E205: Variations on e188_prob_ensemble (our best at 0.54 private).

Knowledge base says: e188 prob blend of 11 diverse models = 0.54 private.
E203/E204 showed: removing "bad" models hurts (diversity > individual quality).
E188 included TabPFN + OvO + CNN alongside trees = maximum diversity.

Hypothesis: small variations on the e188 weights/composition could push
past 0.54 without retraining. Try:
1. Multiple Nelder-Mead random restarts (e188 used single start)
2. Temperature sharpening on the blend output
3. Power-mean blending (interpolates between arithmetic and geometric)
4. Slight composition changes (+/- 1-2 models)
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

def macro_map(preds):
    aps = []
    for i in range(9):
        y_bin = (y == i).astype(int)
        aps.append(average_precision_score(y_bin, preds[:, i]))
    return np.mean(aps), aps

def softmax_w(w):
    w = np.abs(np.array(w))
    s = w.sum()
    return w / max(s, 1e-12)

track_ids = test['track_id'].values

def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e205_{tag}.csv"
    df.to_csv(fname, index=False)
    print(f"  Saved {fname}")

# ---- Load the EXACT e188 model pool ----
e188_models = ['e79', 'e175_best', 'e175_lgb', 'e179_best',
               'e185_tabpfn_relabel', 'e185_tabpfn_all', 'e186_ovo',
               'e180_cnn', 'e187_blend', 'e173', 'e179_cb']

candidates = {}
for name in e188_models:
    oof_path = ROOT / f'oof_{name}.npy'
    test_path = ROOT / f'test_{name}.npy'
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path, allow_pickle=True)
        tst = np.load(test_path, allow_pickle=True)
        if hasattr(oof, 'shape') and oof.shape == (2601, 9) and tst.shape == (1872, 9):
            candidates[name] = {'oof': oof.astype(float), 'test': tst.astype(float)}

print(f"Loaded {len(candidates)} of {len(e188_models)} e188 models")
names = list(candidates.keys())
oofs = [candidates[n]['oof'] for n in names]
tests = [candidates[n]['test'] for n in names]
N = len(names)

# ---- METHOD 1: Multi-restart Nelder-Mead ----
print(f"\n{'='*60}")
print("[1] MULTI-RESTART NELDER-MEAD (10 random starts)")
print(f"{'='*60}")

def neg_map(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, oofs))
    m, _ = macro_map(blend)
    return -m

best_m = 0
best_w = None
for seed in range(10):
    rng = np.random.RandomState(seed)
    w0 = rng.dirichlet(np.ones(N))
    res = minimize(neg_map, w0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-6})
    w = softmax_w(res.x)
    m, _ = macro_map(sum(wi * oof for wi, oof in zip(w, oofs)))
    if m > best_m:
        best_m = m
        best_w = w
    print(f"  seed {seed}: OOF {m:.4f}")

print(f"  Best: OOF {best_m:.4f}")
test_multi = sum(wi * t for wi, t in zip(best_w, tests))

# Temperature sharpening on the best blend
print("\n  Temperature sweep on best blend:")
for T in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]:
    sharpened = test_multi ** (1.0 / T)
    sharpened = sharpened / sharpened.sum(axis=1, keepdims=True)
    # Can't eval on test, just generate

save_sub(test_multi, "multi_restart_raw")

# Sharpen with T=0.85 (historically worked)
sharp = test_multi ** (1.0 / 0.85)
sharp = sharp / sharp.sum(axis=1, keepdims=True)
save_sub(sharp, "multi_restart_T085")

sharp9 = test_multi ** (1.0 / 0.9)
sharp9 = sharp9 / sharp9.sum(axis=1, keepdims=True)
save_sub(sharp9, "multi_restart_T09")

# ---- METHOD 2: Power mean blending ----
print(f"\n{'='*60}")
print("[2] POWER MEAN (p=2, p=3)")
print(f"{'='*60}")

for p in [2, 3]:
    def neg_map_power(w, power=p):
        w = softmax_w(w)
        blend_p = sum(wi * (oof ** power) for wi, oof in zip(w, oofs))
        blend = blend_p ** (1.0 / power)
        blend = np.nan_to_num(blend, nan=0.0)
        m, _ = macro_map(blend)
        return -m

    res_p = minimize(neg_map_power, np.ones(N)/N, method='Nelder-Mead',
                     options={'maxiter': 10000})
    w_p = softmax_w(res_p.x)
    test_p = sum(wi * (t ** p) for wi, t in zip(w_p, tests))
    test_p = test_p ** (1.0 / p)
    test_p = np.nan_to_num(test_p, nan=0.0)
    test_p = test_p / np.maximum(test_p.sum(axis=1, keepdims=True), 1e-10)

    blend_oof = sum(wi * (oof ** p) for wi, oof in zip(w_p, oofs))
    blend_oof = blend_oof ** (1.0 / p)
    m_p, _ = macro_map(np.nan_to_num(blend_oof, nan=0.0))
    print(f"  p={p}: OOF {m_p:.4f}")
    save_sub(test_p, f"power_p{p}")

# ---- METHOD 3: +2 diverse models to e188 pool ----
print(f"\n{'='*60}")
print("[3] E188 + EXTRA DIVERSE MODELS")
print(f"{'='*60}")

extras = ['e201_dart', 'e201_rf', 'e182_cnn_v3', 'e180_rcs_linear',
          'e177_diverse', 'e176_gmm', 'e180_spatial']

for extra_name in extras:
    oof_path = ROOT / f'oof_{extra_name}.npy'
    test_path = ROOT / f'test_{extra_name}.npy'
    if oof_path.exists() and test_path.exists():
        try:
            oof = np.load(oof_path, allow_pickle=True)
            tst = np.load(test_path, allow_pickle=True)
            if hasattr(oof, 'shape') and oof.shape == (2601, 9) and tst.shape == (1872, 9):
                candidates[extra_name] = {'oof': oof.astype(float), 'test': tst.astype(float)}
        except:
            pass

# Try e188 + each extra individually
for extra_name in extras:
    if extra_name not in candidates:
        continue
    aug_names = names + [extra_name]
    aug_oofs = oofs + [candidates[extra_name]['oof']]
    aug_tests = tests + [candidates[extra_name]['test']]

    def neg_map_aug(w):
        w = softmax_w(w)
        blend = sum(wi * oof for wi, oof in zip(w, aug_oofs))
        m, _ = macro_map(blend)
        return -m

    w0 = np.ones(len(aug_names)) / len(aug_names)
    res_aug = minimize(neg_map_aug, w0, method='Nelder-Mead',
                       options={'maxiter': 10000})
    w_aug = softmax_w(res_aug.x)
    m_aug, _ = macro_map(sum(wi * o for wi, o in zip(w_aug, aug_oofs)))
    extra_weight = w_aug[-1]
    print(f"  +{extra_name:25s}: OOF {m_aug:.4f}, extra_w={extra_weight:.3f}")

# Build the best augmented blend: e188 + e201_dart + e182_cnn_v3
aug2_names = names + ['e201_dart', 'e182_cnn_v3']
aug2_oofs = oofs + [candidates['e201_dart']['oof'], candidates['e182_cnn_v3']['oof']]
aug2_tests = tests + [candidates['e201_dart']['test'], candidates['e182_cnn_v3']['test']]

def neg_map_aug2(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, aug2_oofs))
    m, _ = macro_map(blend)
    return -m

best_m2 = 0
best_w2 = None
for seed in range(5):
    rng = np.random.RandomState(seed + 100)
    w0 = rng.dirichlet(np.ones(len(aug2_names)))
    res2 = minimize(neg_map_aug2, w0, method='Nelder-Mead',
                    options={'maxiter': 10000})
    w2 = softmax_w(res2.x)
    m2, _ = macro_map(sum(wi * o for wi, o in zip(w2, aug2_oofs)))
    if m2 > best_m2:
        best_m2 = m2
        best_w2 = w2

print(f"\n  E188+dart+cnn: OOF {best_m2:.4f}")
test_aug2 = sum(wi * t for wi, t in zip(best_w2, aug2_tests))
save_sub(test_aug2, "e188_plus_dart_cnn")

# Temperature on augmented
sharp_aug = test_aug2 ** (1.0 / 0.9)
sharp_aug = sharp_aug / sharp_aug.sum(axis=1, keepdims=True)
save_sub(sharp_aug, "e188_plus_dart_cnn_T09")

print(f"\n{'='*60}")
print("DONE - 7 submissions generated")
print(f"{'='*60}")
