"""E07: Blend E05 (tabular ensemble) + E06 (1D-CNN)

Optimize blending weight on OOF mAP. Even a weak CNN can help
if its errors are uncorrelated with the tabular models.
"""
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

# ── Load OOF predictions ──────────────────────────────────────────
oof_e05 = np.load("oof_e05.npy")
oof_e06 = np.load("oof_e06.npy")
test_e05 = np.load("test_e05.npy")
test_e06 = np.load("test_e06.npy")

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(load_train()["bird_group"])

# ── Baseline scores ───────────────────────────────────────────────
e05_map, e05_per = compute_map(y, oof_e05)
e06_map, e06_per = compute_map(y, oof_e06)
print_results(e05_map, e05_per, "E05 Tabular Ensemble")
print_results(e06_map, e06_per, "E06 1D-CNN")

# ── Grid search for optimal blend weight ──────────────────────────
print("\nSearching blend weights...", flush=True)
best_map = 0
best_w = 1.0

for w_tabular in np.arange(0.50, 1.01, 0.01):
    w_cnn = 1 - w_tabular
    blend = w_tabular * oof_e05 + w_cnn * oof_e06
    bmap, _ = compute_map(y, blend)
    if bmap > best_map:
        best_map = bmap
        best_w = w_tabular

print(f"Best: tabular={best_w:.2f}, cnn={1-best_w:.2f} -> mAP={best_map:.4f}")

# ── Final blend ───────────────────────────────────────────────────
oof_final = best_w * oof_e05 + (1 - best_w) * oof_e06
test_final = best_w * test_e05 + (1 - best_w) * test_e06

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E07 Blend (tabular + CNN)")

# ── Comparison ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  COMPARISON")
print(f"{'='*50}")
print(f"  E02 baseline:      0.7214")
print(f"  E05 tabular:       {e05_map:.4f}")
print(f"  E06 CNN:           {e06_map:.4f}")
print(f"  E07 blend:         {final_map:.4f}")
print(f"{'='*50}")

# Per-class comparison
print(f"\n  Per-class AP comparison (E02 -> E07):")
e02_aps = {"Clutter": 0.610, "Cormorants": 0.939, "Pigeons": 0.254,
           "Ducks": 0.666, "Geese": 0.728, "Gulls": 0.956,
           "Birds of Prey": 0.885, "Waders": 0.816, "Songbirds": 0.640}
for cls in CLASSES:
    old = e02_aps[cls]
    new = final_per[cls]
    delta = new - old
    arrow = "+" if delta > 0 else ""
    print(f"  {cls:15s}: {old:.3f} -> {new:.3f} ({arrow}{delta:.3f})")

# ── Save ──────────────────────────────────────────────────────────
save_submission(test_final, "e07_blend", cv_map=final_map)
