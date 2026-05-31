"""Re-derive the E205 multi_restart blend weights from scratch (full reproduction).

This reproduces the EXACT, deterministic (seeded) weight optimization that
produced weights.json["multi_restart"]. It uses the out-of-fold (OOF) training
predictions in models/oof_*.npy and the training labels in y.npy.

Run it to confirm the shipped weights are reproducible:
    python optimize_weights.py

Output weights should match weights.json (max abs diff < 1e-4). The optimizer is
Nelder-Mead with 10 fixed-seed Dirichlet restarts; the best OOF macro-mAP wins.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent

MODELS = ["e79", "e175_best", "e175_lgb", "e179_best", "e185_tabpfn_relabel",
          "e185_tabpfn_all", "e186_ovo", "e180_cnn", "e187_blend", "e173", "e179_cb"]

y = np.load(HERE / "y.npy")                       # (2601,) class indices, 0..8
oofs = [np.load(HERE / "models" / f"oof_{n}.npy", allow_pickle=True).astype(float)
        for n in MODELS]
N = len(MODELS)


def macro_map(preds):
    return float(np.mean([average_precision_score((y == i).astype(int), preds[:, i])
                          for i in range(9)]))


def softmax_w(w):
    w = np.abs(np.array(w))
    return w / max(w.sum(), 1e-12)


def neg_map(w):
    w = softmax_w(w)
    return -macro_map(sum(wi * o for wi, o in zip(w, oofs)))


best_m, best_w = 0.0, None
for seed in range(10):
    rng = np.random.RandomState(seed)
    res = minimize(neg_map, rng.dirichlet(np.ones(N)), method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-5, "fatol": 1e-6})
    w = softmax_w(res.x)
    m = macro_map(sum(wi * o for wi, o in zip(w, oofs)))
    print(f"  seed {seed}: OOF {m:.4f}")
    if m > best_m:
        best_m, best_w = m, w

print(f"\nBest OOF macro-mAP = {best_m:.4f}")
derived = {n: float(wi) for n, wi in zip(MODELS, best_w)}
for n, wi in derived.items():
    print(f"  {n:24s} {wi:.6f}")

shipped = json.loads((HERE / "weights.json").read_text())["multi_restart"]["weights"]
max_diff = max(abs(derived[n] - shipped[n]) for n in MODELS)
print(f"\nMax abs diff vs shipped weights.json: {max_diff:.3e}  "
      f"({'MATCH' if max_diff < 1e-4 else 'MISMATCH'})")
