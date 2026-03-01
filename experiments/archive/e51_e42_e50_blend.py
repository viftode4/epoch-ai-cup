"""E51: Blend E42 historical model with E50 per-class specialist blend.

This is a low-risk post-processing ensemble:
- E42 remains the strongest robust baseline.
- E50 contributes external-prior specialist diversity (especially Pigeons/Waders).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission

ALPHAS = [i / 20.0 for i in range(21)]  # 0.00 .. 1.00 step 0.05


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


print("=" * 70, flush=True)
print("E51 E42 + E50 BLEND".center(70), flush=True)
print("=" * 70, flush=True)

oof42 = np.load(ROOT / "oof_e42.npy")
oof50 = np.load(ROOT / "oof_e50.npy")
test42 = np.load(ROOT / "test_e42.npy")
test50 = np.load(ROOT / "test_e50.npy")

train = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train["bird_group"])

best_alpha = None
best_map = -1.0
best_oof = None

print("\nAlpha sweep on OOF:", flush=True)
for alpha in ALPHAS:
    oof_blend = (1.0 - alpha) * oof42 + alpha * oof50
    oof_blend = renorm_rows(oof_blend)
    m, _ = compute_map(y, oof_blend)
    print(f"  alpha_e50={alpha:.2f}: mAP={m:.4f}", flush=True)
    if m > best_map:
        best_map = m
        best_alpha = alpha
        best_oof = oof_blend

print(f"\nBest alpha_e50: {best_alpha:.2f}", flush=True)
best_per = compute_map(y, best_oof)[1]
print_results(best_map, best_per, label="E51 blended (LOMO OOF)")

test_blend = (1.0 - best_alpha) * test42 + best_alpha * test50
test_blend = renorm_rows(test_blend)

np.save(ROOT / "oof_e51.npy", best_oof)
np.save(ROOT / "test_e51.npy", test_blend)
save_submission(test_blend, "e51_e42_e50_blend", cv_map=best_map)

print("\nSaved: oof_e51.npy, test_e51.npy", flush=True)
print("Done.", flush=True)
