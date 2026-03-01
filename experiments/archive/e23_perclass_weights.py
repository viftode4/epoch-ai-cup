"""E23: Per-Class Stacking Weight Optimization

Instead of one set of weights for all classes, optimize tree/CNN/SVM/rocket
weights separately per class to maximize per-class AP.

Uses existing OOF files (StratifiedKFold). No retraining needed.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ── Load data and OOF predictions ────────────────────────────────
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

oof_tree   = np.load(ROOT / "oof_e15.npy")       # 0.7451
oof_cnn    = np.load(ROOT / "oof_e06.npy")        # 0.5238
oof_rocket = np.load(ROOT / "oof_e08.npy")        # 0.4799
oof_svm    = np.load(ROOT / "oof_e09.npy")        # 0.5238

models = [("tree", oof_tree), ("cnn", oof_cnn), ("rocket", oof_rocket), ("svm", oof_svm)]

# ── Baseline: Global weights (E15 approach) ──────────────────────
print("=" * 60, flush=True)
print("Baseline: Global weight optimization", flush=True)
print("=" * 60, flush=True)

best_global_map = 0
best_global_w = None
for w0 in np.arange(0.50, 0.90, 0.05):
    for w1 in np.arange(0.05, 0.25, 0.05):
        for w2 in np.arange(0.05, 0.25, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.02:
                continue
            oof = w0 * oof_tree + w1 * oof_cnn + w2 * oof_rocket + w3 * oof_svm
            m, _ = compute_map(y, oof)
            if m > best_global_map:
                best_global_map = m
                best_global_w = (w0, w1, w2, w3)

oof_global = (best_global_w[0] * oof_tree + best_global_w[1] * oof_cnn +
              best_global_w[2] * oof_rocket + best_global_w[3] * oof_svm)
global_map, global_per = compute_map(y, oof_global)
print(f"Best global: tree={best_global_w[0]:.2f} cnn={best_global_w[1]:.2f} "
      f"rocket={best_global_w[2]:.2f} svm={best_global_w[3]:.2f}", flush=True)
print_results(global_map, global_per, "Global Weights")

# ── Per-class weight optimization ────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Per-class weight optimization", flush=True)
print("=" * 60, flush=True)

oof_perclass = np.zeros((len(y), N_CLASSES))
per_class_weights = {}

for c in range(N_CLASSES):
    y_binary = (y == c).astype(int)
    best_ap = 0
    best_w = None

    # Finer grid for per-class optimization
    for w0 in np.arange(0.0, 1.01, 0.05):
        for w1 in np.arange(0.0, 1.01 - w0, 0.05):
            for w2 in np.arange(0.0, 1.01 - w0 - w1, 0.05):
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -0.01:
                    continue
                w3 = max(w3, 0.0)
                probs = (w0 * oof_tree[:, c] + w1 * oof_cnn[:, c] +
                         w2 * oof_rocket[:, c] + w3 * oof_svm[:, c])
                ap = average_precision_score(y_binary, probs)
                if ap > best_ap:
                    best_ap = ap
                    best_w = (w0, w1, w2, w3)

    per_class_weights[c] = best_w
    oof_perclass[:, c] = (best_w[0] * oof_tree[:, c] + best_w[1] * oof_cnn[:, c] +
                          best_w[2] * oof_rocket[:, c] + best_w[3] * oof_svm[:, c])

    global_ap = global_per[CLASSES[c]]
    delta = best_ap - global_ap
    print(f"  {CLASSES[c]:15s}: AP={best_ap:.4f} (global={global_ap:.4f}, {delta:+.4f}) "
          f"w=[tree={best_w[0]:.2f} cnn={best_w[1]:.2f} rkt={best_w[2]:.2f} svm={best_w[3]:.2f}]",
          flush=True)

perclass_map, perclass_per = compute_map(y, oof_perclass)
print_results(perclass_map, perclass_per, "Per-Class Weights")

# ── Add logit adjustment on top ──────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Per-class weights + logit adjustment", flush=True)
print("=" * 60, flush=True)

priors = counts / counts.sum()
per_class_tau = np.zeros(N_CLASSES)
current_best = perclass_map

# Normalize for logit adjustment
oof_norm = oof_perclass / oof_perclass.sum(axis=1, keepdims=True)

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = per_class_tau[c]
        best_c_map = current_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adj = priors ** (-per_class_tau)
            adjusted = oof_norm * adj[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c
        per_class_tau[c] = best_c_tau
        if best_c_map > current_best:
            current_best = best_c_map
            improved = True
    print(f"  Logit adj round {iteration + 1}: mAP={current_best:.4f}", flush=True)
    if not improved:
        break

adj = priors ** (-per_class_tau)
oof_adj = oof_norm * adj[np.newaxis, :]
oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)

adj_map, adj_per = compute_map(y, oof_adj)
print_results(adj_map, adj_per, "Per-Class Weights + Logit Adj")

# ── Also apply logit adjustment to global weights ────────────────
print("\n" + "=" * 60, flush=True)
print("Global weights + logit adjustment (E15 baseline)", flush=True)
print("=" * 60, flush=True)

global_tau = np.zeros(N_CLASSES)
global_best = global_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_tau = global_tau[c]
        best_c_map = global_best
        for tau_c in np.arange(-0.5, 1.51, 0.02):
            global_tau[c] = tau_c
            gadj = priors ** (-global_tau)
            adjusted = oof_global * gadj[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c
        global_tau[c] = best_c_tau
        if best_c_map > global_best:
            global_best = best_c_map
            improved = True
    if not improved:
        break

gadj = priors ** (-global_tau)
oof_global_adj = oof_global * gadj[np.newaxis, :]
oof_global_adj = oof_global_adj / oof_global_adj.sum(axis=1, keepdims=True)
global_adj_map, global_adj_per = compute_map(y, oof_global_adj)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  E15 stack (ref):                  0.7493", flush=True)
print(f"  E15 stack + logit adj (ref):      0.7535", flush=True)
print(f"  Global weights:                   {global_map:.4f}", flush=True)
print(f"  Global weights + logit adj:       {global_adj_map:.4f}", flush=True)
print(f"  Per-class weights:                {perclass_map:.4f} ({perclass_map - global_map:+.4f} vs global)", flush=True)
print(f"  Per-class weights + logit adj:    {adj_map:.4f} ({adj_map - global_adj_map:+.4f} vs global+logit)", flush=True)

# Save best
np.save(ROOT / "oof_e23.npy", oof_adj)
print(f"\nSaved oof_e23.npy", flush=True)
print("Done!", flush=True)
