"""E205 weight extraction (for TNO request).

Reproduces the EXACT, deterministic (seeded) ensemble weight optimization from
e205_e188_variations.py and writes the resulting blend weights to a clean
artifact: submissions/e205_weights.json (+ a human-readable .txt).

The original e205 script only printed the weights; this recovers them verbatim
since the optimization is fully seeded (np.random.RandomState + Nelder-Mead).
"""
import json
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
y = np.array([CLASSES.index(c) for c in train['bird_group'].values])


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


def load_pool(model_names):
    pool = {}
    for name in model_names:
        oof_path = ROOT / f'oof_{name}.npy'
        test_path = ROOT / f'test_{name}.npy'
        if oof_path.exists() and test_path.exists():
            oof = np.load(oof_path, allow_pickle=True)
            tst = np.load(test_path, allow_pickle=True)
            if hasattr(oof, 'shape') and oof.shape == (2601, 9) and tst.shape == (1872, 9):
                pool[name] = {'oof': oof.astype(float), 'test': tst.astype(float)}
    return pool


e188_models = ['e79', 'e175_best', 'e175_lgb', 'e179_best',
               'e185_tabpfn_relabel', 'e185_tabpfn_all', 'e186_ovo',
               'e180_cnn', 'e187_blend', 'e173', 'e179_cb']

candidates = load_pool(e188_models)
names = list(candidates.keys())
oofs = [candidates[n]['oof'] for n in names]
N = len(names)
print(f"Loaded {N} of {len(e188_models)} e188 models")

results = {}

# ---- METHOD 1: Multi-restart Nelder-Mead (10 seeded starts) ----
def neg_map(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, oofs))
    m, _ = macro_map(blend)
    return -m

best_m, best_w = 0, None
for seed in range(10):
    rng = np.random.RandomState(seed)
    w0 = rng.dirichlet(np.ones(N))
    res = minimize(neg_map, w0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-6})
    w = softmax_w(res.x)
    m, _ = macro_map(sum(wi * oof for wi, oof in zip(w, oofs)))
    if m > best_m:
        best_m, best_w = m, w

results['multi_restart'] = {
    'submission_files': ['e205_multi_restart_raw.csv',
                         'e205_multi_restart_T085.csv',
                         'e205_multi_restart_T09.csv'],
    'oof_macro_map': float(best_m),
    'model_names': names,
    'weights': {n: float(wv) for n, wv in zip(names, best_w)},
}
print(f"\n[1] multi_restart  OOF mAP={best_m:.4f}")
for n, wv in zip(names, best_w):
    print(f"    {n:24s} {wv:.4f}")

# ---- METHOD 2: Power-mean blends (p=2, p=3) ----
for p in [2, 3]:
    def neg_map_power(w, power=p):
        w = softmax_w(w)
        blend_p = sum(wi * (oof ** power) for wi, oof in zip(w, oofs))
        blend = np.nan_to_num(blend_p ** (1.0 / power), nan=0.0)
        m, _ = macro_map(blend)
        return -m

    res_p = minimize(neg_map_power, np.ones(N) / N, method='Nelder-Mead',
                     options={'maxiter': 10000})
    w_p = softmax_w(res_p.x)
    blend_oof = sum(wi * (oof ** p) for wi, oof in zip(w_p, oofs))
    blend_oof = np.nan_to_num(blend_oof ** (1.0 / p), nan=0.0)
    m_p, _ = macro_map(blend_oof)
    results[f'power_p{p}'] = {
        'submission_files': [f'e205_power_p{p}.csv'],
        'oof_macro_map': float(m_p),
        'model_names': names,
        'weights': {n: float(wv) for n, wv in zip(names, w_p)},
    }
    print(f"\n[2] power_p{p}  OOF mAP={m_p:.4f}")
    for n, wv in zip(names, w_p):
        print(f"    {n:24s} {wv:.4f}")

# ---- METHOD 3: E188 + e201_dart + e182_cnn_v3 ----
extra_pool = load_pool(['e201_dart', 'e182_cnn_v3'])
candidates.update(extra_pool)
aug2_names = names + ['e201_dart', 'e182_cnn_v3']
aug2_oofs = oofs + [candidates['e201_dart']['oof'], candidates['e182_cnn_v3']['oof']]

def neg_map_aug2(w):
    w = softmax_w(w)
    blend = sum(wi * oof for wi, oof in zip(w, aug2_oofs))
    m, _ = macro_map(blend)
    return -m

best_m2, best_w2 = 0, None
for seed in range(5):
    rng = np.random.RandomState(seed + 100)
    w0 = rng.dirichlet(np.ones(len(aug2_names)))
    res2 = minimize(neg_map_aug2, w0, method='Nelder-Mead',
                    options={'maxiter': 10000})
    w2 = softmax_w(res2.x)
    m2, _ = macro_map(sum(wi * o for wi, o in zip(w2, aug2_oofs)))
    if m2 > best_m2:
        best_m2, best_w2 = m2, w2

results['e188_plus_dart_cnn'] = {
    'submission_files': ['e205_e188_plus_dart_cnn.csv',
                         'e205_e188_plus_dart_cnn_T09.csv'],
    'oof_macro_map': float(best_m2),
    'model_names': aug2_names,
    'weights': {n: float(wv) for n, wv in zip(aug2_names, best_w2)},
}
print(f"\n[3] e188_plus_dart_cnn  OOF mAP={best_m2:.4f}")
for n, wv in zip(aug2_names, best_w2):
    print(f"    {n:24s} {wv:.4f}")

# ---- Write artifacts ----
out_json = ROOT / 'submissions' / 'e205_weights.json'
out_json.write_text(json.dumps(results, indent=2))

lines = ["E205 ensemble blend weights (extracted for TNO)",
         "Source: experiments/e205_e188_variations.py (deterministic, seeded)",
         "=" * 60, ""]
for variant, info in results.items():
    lines.append(f"## {variant}  (OOF macro-mAP = {info['oof_macro_map']:.4f})")
    lines.append(f"   submissions: {', '.join(info['submission_files'])}")
    for n, wv in info['weights'].items():
        lines.append(f"     {n:24s} {wv:.6f}")
    lines.append("")
out_txt = ROOT / 'submissions' / 'e205_weights.txt'
out_txt.write_text("\n".join(lines))

print(f"\nWrote {out_json}")
print(f"Wrote {out_txt}")
