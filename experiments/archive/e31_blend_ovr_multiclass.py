"""E31: Blend OvR + Multi-class.

Loads all available OOF/test predictions from E25D, E27-E30.
Per-class blend optimization using average_precision_score.
Greedy forward selection: start with best single, add models that improve.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ── Load data ─────────────────────────────────────────────────────
print("Loading labels...", flush=True)
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# ── Load all available predictions ────────────────────────────────
print("Loading predictions...", flush=True)

models = {}

def try_load(name, oof_file, test_file):
    oof_path = ROOT / oof_file
    test_path = ROOT / test_file
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path)
        test = np.load(test_path)
        if oof.shape == (len(y), N_CLASSES) and test.shape[1] == N_CLASSES:
            models[name] = {"oof": oof, "test": test}
            m, _ = compute_map(y, oof)
            print(f"  Loaded {name}: mAP={m:.4f}", flush=True)
            return True
    return False

# Multi-class models (SKF-based, most likely to generalize)
try_load("E25D_multiclass", "oof_e25d.npy", "test_e25d.npy")
try_load("E27_SKF", "oof_e27_skf.npy", "test_e27_skf.npy")
try_load("E28_SKF_adv", "oof_e28.npy", "test_e28.npy")

# OvR models
try_load("E29_OvR_SKF", "oof_e29_skf.npy", "test_e29_skf.npy")
try_load("E30_OvR_adv", "oof_e30.npy", "test_e30.npy")

# LOMO models (for diversity)
try_load("E27_LOMO", "oof_e27_lomo.npy", "test_e27_lomo.npy")
try_load("E29_OvR_LOMO", "oof_e29_lomo.npy", "test_e29_lomo.npy")

model_names = list(models.keys())
n_models = len(model_names)
print(f"\n  Total models loaded: {n_models}", flush=True)

if n_models == 0:
    print("ERROR: No models found!", flush=True)
    sys.exit(1)


# ── Per-class model ranking ───────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Per-class model ranking (AP)", flush=True)
print("=" * 60, flush=True)

# Build per-class AP matrix
per_class_ap = np.zeros((n_models, N_CLASSES))
for i, name in enumerate(model_names):
    oof = models[name]["oof"]
    for c in range(N_CLASSES):
        y_c = (y == c).astype(int)
        per_class_ap[i, c] = average_precision_score(y_c, oof[:, c])

# Print ranking
header = f"  {'Model':<25s}" + "".join(f" {CLASSES[c]:>10s}" for c in range(N_CLASSES)) + f" {'mAP':>8s}"
print(header, flush=True)
for i, name in enumerate(model_names):
    aps = per_class_ap[i]
    row = f"  {name:<25s}" + "".join(f" {aps[c]:>10.4f}" for c in range(N_CLASSES)) + f" {aps.mean():>8.4f}"
    print(row, flush=True)


# ── Per-class alpha blend optimization ────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Per-class pairwise blend optimization", flush=True)
print("=" * 60, flush=True)

# For each class, find best blend of all model pairs
best_per_class_blend = {}
for c in range(N_CLASSES):
    cls_name = CLASSES[c]
    y_c = (y == c).astype(int)

    best_ap = 0
    best_config = None

    # Single best
    for i, name in enumerate(model_names):
        ap = per_class_ap[i, c]
        if ap > best_ap:
            best_ap = ap
            best_config = (name, 1.0)

    # Pairwise blends
    for i in range(n_models):
        for j in range(i + 1, n_models):
            for alpha in np.arange(0.0, 1.01, 0.05):
                blended = alpha * models[model_names[i]]["oof"][:, c] + (1 - alpha) * models[model_names[j]]["oof"][:, c]
                ap = average_precision_score(y_c, blended)
                if ap > best_ap:
                    best_ap = ap
                    best_config = (model_names[i], alpha, model_names[j], 1 - alpha)

    best_per_class_blend[c] = (best_ap, best_config)
    if len(best_config) == 2:
        print(f"  {cls_name:<15s}: AP={best_ap:.4f} -- {best_config[0]} (100%)", flush=True)
    else:
        print(f"  {cls_name:<15s}: AP={best_ap:.4f} -- {best_config[0]}({best_config[1]:.0%}) + {best_config[2]}({best_config[3]:.0%})", flush=True)


# ── Greedy forward selection (global) ─────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Greedy forward selection", flush=True)
print("=" * 60, flush=True)

# Start with best single model
best_single_map = 0
best_single = None
for name in model_names:
    m, _ = compute_map(y, models[name]["oof"])
    if m > best_single_map:
        best_single_map = m
        best_single = name

print(f"  Best single model: {best_single} (mAP={best_single_map:.4f})", flush=True)

# Greedy: try adding each remaining model with weight optimization
selected = [best_single]
remaining = [n for n in model_names if n != best_single]
current_oof = models[best_single]["oof"].copy()
current_test = models[best_single]["test"].copy()
current_map = best_single_map

for step in range(min(4, len(remaining))):  # max 4 additions
    best_improvement = 0
    best_add = None
    best_alpha = 0

    for name in remaining:
        # Try blending current with this model
        for alpha in np.arange(0.05, 0.96, 0.05):
            blended = (1 - alpha) * current_oof + alpha * models[name]["oof"]
            m, _ = compute_map(y, blended)
            improvement = m - current_map
            if improvement > best_improvement:
                best_improvement = improvement
                best_add = name
                best_alpha = alpha

    if best_improvement > 0.0005:  # minimum improvement threshold
        current_oof = (1 - best_alpha) * current_oof + best_alpha * models[best_add]["oof"]
        current_test = (1 - best_alpha) * current_test + best_alpha * models[best_add]["test"]
        current_map += best_improvement
        selected.append(best_add)
        remaining.remove(best_add)
        print(f"  + {best_add} (alpha={best_alpha:.2f}) -> mAP={current_map:.4f} (+{best_improvement:.4f})", flush=True)
    else:
        print(f"  No further improvement (best candidate: +{best_improvement:.4f})", flush=True)
        break

print(f"\n  Final greedy blend: {selected}", flush=True)


# ── Build best per-class blend ────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Per-class optimal blend", flush=True)
print("=" * 60, flush=True)

oof_perclass = np.zeros((len(y), N_CLASSES))
test_perclass = np.zeros((models[model_names[0]]["test"].shape[0], N_CLASSES))

for c in range(N_CLASSES):
    best_ap, config = best_per_class_blend[c]
    if len(config) == 2:
        name, w = config
        oof_perclass[:, c] = models[name]["oof"][:, c]
        test_perclass[:, c] = models[name]["test"][:, c]
    else:
        name1, w1, name2, w2 = config
        oof_perclass[:, c] = w1 * models[name1]["oof"][:, c] + w2 * models[name2]["oof"][:, c]
        test_perclass[:, c] = w1 * models[name1]["test"][:, c] + w2 * models[name2]["test"][:, c]

map_perclass, per_perclass = compute_map(y, oof_perclass)
print_results(map_perclass, per_perclass, "E31 Per-class optimal blend")


# ── Compare all approaches ────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("FINAL COMPARISON", flush=True)
print("=" * 60, flush=True)

greedy_map, greedy_per = compute_map(y, current_oof)

results = [
    ("Best single", best_single_map),
    ("Greedy blend", greedy_map),
    ("Per-class blend", map_perclass),
]

print(f"  {'Method':<25s} {'mAP':>8s}", flush=True)
for name, m in results:
    print(f"  {name:<25s} {m:>8.4f}", flush=True)

# Pick the best
best_method = max(results, key=lambda x: x[1])
print(f"\n  Winner: {best_method[0]} (mAP={best_method[1]:.4f})", flush=True)

if best_method[0] == "Per-class blend":
    final_oof = oof_perclass
    final_test = test_perclass
    final_map = map_perclass
    final_per = per_perclass
elif best_method[0] == "Greedy blend":
    final_oof = current_oof
    final_test = current_test
    final_map = greedy_map
    final_per = greedy_per
else:
    final_oof = models[best_single]["oof"]
    final_test = models[best_single]["test"]
    final_map = best_single_map
    _, final_per = compute_map(y, final_oof)

print_results(final_map, final_per, "E31 FINAL")

# Test distribution
pred_classes = final_test.argmax(axis=1)
dist = np.bincount(pred_classes, minlength=N_CLASSES)
print(f"\nTest predictions: {dict(zip(CLASSES, dist))}", flush=True)

# ── Save ──────────────────────────────────────────────────────────
np.save(ROOT / "oof_e31.npy", final_oof)
np.save(ROOT / "test_e31.npy", final_test)

save_submission(final_test, "e31_blend_ovr_multiclass", cv_map=final_map)
print("\nDone!", flush=True)
