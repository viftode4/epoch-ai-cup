"""E185b: Fine-grained refinement of best blend from E185.

Refines the 3-way blend (e166 + e176_gmm + e175_lgb) + corm_det found in E185.
Also tries combining physics prior + detector, and fine-grained weight search.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
TRAIN_MONTHS = [1, 4, 9, 10]
N = len(y)
CORM_IDX = CLASSES.index("Cormorants")

def lomo_map(oof_preds):
    fold_maps = []
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        mAP, _ = compute_map(y[mask], oof_preds[mask])
        fold_maps.append(mAP)
    return np.mean(fold_maps)

def lomo_map_perclass(oof_preds):
    per_class = {c: [] for c in CLASSES}
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        _, pc = compute_map(y[mask], oof_preds[mask])
        for c in CLASSES:
            per_class[c].append(pc[c])
    return {c: np.mean(v) for c, v in per_class.items()}

def simple_avg_blend(preds_list, weights):
    blended = np.zeros((N, 9))
    total_w = sum(weights)
    for pred, w in zip(preds_list, weights):
        blended += w * pred
    blended /= total_w
    return blended

# Load key predictions
e166 = np.load(ROOT / "oof_e166.npy")
e176_gmm = np.load(ROOT / "oof_e176_gmm.npy")
e176_iso = np.load(ROOT / "oof_e176_iso.npy")
e176_igk = np.load(ROOT / "oof_e176_iso_gmm_knn.npy")
e175_lgb = np.load(ROOT / "oof_e175_lgb.npy")
e175_best = np.load(ROOT / "oof_e175_best.npy")
e183_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")
e175_cb = np.load(ROOT / "oof_e175_cb.npy")
e175_dro = np.load(ROOT / "oof_e175_dro.npy")
corm_det = np.load(ROOT / "oof_e184_corm_det.npy")
wader_det = np.load(ROOT / "oof_e184_wader_det.npy")

print("=" * 70)
print("  E185b: FINE-GRAINED BLEND REFINEMENT")
print("=" * 70)

# ── Refine 3-way: e166 + e176_gmm + e175_lgb ─────────────────────
print("\n  Refining 3-way: e166 + e176_gmm + e175_lgb")
best_3way = {"score": 0, "weights": None}
for w1 in np.arange(0.30, 0.70, 0.02):
    for w2 in np.arange(0.05, 0.50, 0.02):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.60:
            continue
        blend = simple_avg_blend([e166, e176_gmm, e175_lgb], [w1, w2, w3])
        score = lomo_map(blend)
        if score > best_3way["score"]:
            best_3way = {"score": score, "weights": (w1, w2, w3)}

print(f"  Best: w={best_3way['weights']}, LOMO={best_3way['score']:.4f}")

# ── Try adding a 4th model to the best 3-way ─────────────────────
print("\n  Adding 4th model to best 3-way...")
w1, w2, w3 = best_3way["weights"]
base_3way = simple_avg_blend([e166, e176_gmm, e175_lgb], [w1, w2, w3])

extras = {
    "e176_iso_gmm_knn": e176_igk,
    "e176_iso": e176_iso,
    "e183_tabpfn": e183_tabpfn,
    "e175_best": e175_best,
    "e175_cb": e175_cb,
    "e175_dro": e175_dro,
}

best_4th = {"score": best_3way["score"], "name": "none", "alpha": 0}
for name, extra in extras.items():
    for alpha in np.arange(0.05, 0.50, 0.05):
        blend = (1.0 - alpha) * base_3way + alpha * extra
        score = lomo_map(blend)
        if score > best_4th["score"]:
            best_4th = {"score": score, "name": name, "alpha": alpha}

print(f"  Best 4th: {best_4th['name']}, alpha={best_4th['alpha']:.2f}, LOMO={best_4th['score']:.4f}")

# Build the actual best base blend
if best_4th["name"] != "none":
    base_blend = (1.0 - best_4th["alpha"]) * base_3way + best_4th["alpha"] * extras[best_4th["name"]]
    base_lomo = best_4th["score"]
else:
    base_blend = base_3way
    base_lomo = best_3way["score"]

print(f"\n  Base blend LOMO: {base_lomo:.4f}")

# ── Apply corm_det with fine-grained alpha ───────────────────────
print("\n  Fine-grained corm_det sweep on base blend...")
best_corm = {"score": base_lomo, "alpha": 0, "preds": base_blend}
for alpha in np.arange(0.0, 5.1, 0.1):
    boosted = base_blend.copy()
    boosted[:, CORM_IDX] *= (1.0 + alpha * corm_det)
    row_sums = boosted.sum(axis=1, keepdims=True)
    boosted = boosted / np.maximum(row_sums, 1e-12)
    score = lomo_map(boosted)
    if score > best_corm["score"]:
        best_corm = {"score": score, "alpha": alpha, "preds": boosted}

print(f"  Best corm_det alpha: {best_corm['alpha']:.1f}, LOMO: {best_corm['score']:.4f}")

# ── Apply wader_det on top ───────────────────────────────────────
print("\n  Adding wader_det on top...")
WADER_IDX = CLASSES.index("Waders")
best_wader = {"score": best_corm["score"], "alpha": 0, "preds": best_corm["preds"]}
for alpha in np.arange(0.0, 5.1, 0.1):
    boosted = best_corm["preds"].copy()
    boosted[:, WADER_IDX] *= (1.0 + alpha * wader_det)
    row_sums = boosted.sum(axis=1, keepdims=True)
    boosted = boosted / np.maximum(row_sums, 1e-12)
    score = lomo_map(boosted)
    if score > best_wader["score"]:
        best_wader = {"score": score, "alpha": alpha, "preds": boosted}

print(f"  Best wader_det alpha: {best_wader['alpha']:.1f}, LOMO: {best_wader['score']:.4f}")

# ── Per-class boost sweep (all classes) ──────────────────────────
print("\n  Per-class probability boost sweep...")
final_preds = best_wader["preds"].copy()
final_lomo = best_wader["score"]

for c_idx, c_name in enumerate(CLASSES):
    best_boost = {"score": final_lomo, "factor": 1.0}
    for factor in np.arange(0.5, 2.5, 0.05):
        boosted = final_preds.copy()
        boosted[:, c_idx] *= factor
        row_sums = boosted.sum(axis=1, keepdims=True)
        boosted = boosted / np.maximum(row_sums, 1e-12)
        score = lomo_map(boosted)
        if score > best_boost["score"]:
            best_boost = {"score": score, "factor": factor}

    if best_boost["factor"] != 1.0:
        print(f"  {c_name:15s}: boost={best_boost['factor']:.2f}, LOMO={best_boost['score']:.4f} (delta={best_boost['score']-final_lomo:+.4f})")
        final_preds[:, c_idx] *= best_boost["factor"]
        row_sums = final_preds.sum(axis=1, keepdims=True)
        final_preds = final_preds / np.maximum(row_sums, 1e-12)
        final_lomo = best_boost["score"]
    else:
        print(f"  {c_name:15s}: no improvement from boosting")

# ── Final report ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)

pc = lomo_map_perclass(final_preds)
print(f"\n  Final LOMO: {final_lomo:.4f}")
print(f"\n  Per-class AP breakdown:")
for c in CLASSES:
    marker = " <-- weak" if pc[c] < 0.5 else ""
    print(f"    {c:15s}: {pc[c]:.4f}{marker}")

# SKF reference
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_aps = []
for _, val_idx in skf.split(np.zeros(N), y):
    mAP, _ = compute_map(y[val_idx], final_preds[val_idx])
    skf_aps.append(mAP)
print(f"\n  Reference SKF: {np.mean(skf_aps):.4f}")

# Per-month breakdown
print(f"\n  Per-month LOMO breakdown:")
for m in TRAIN_MONTHS:
    mask = months == m
    mAP, pc_m = compute_map(y[mask], final_preds[mask])
    print(f"    Month {m:2d}: n={mask.sum():4d}, mAP={mAP:.4f}, "
          f"Corm={pc_m['Cormorants']:.4f}")

# Compare vs baselines
print(f"\n  Comparison:")
print(f"    e166 alone:     LOMO={lomo_map(e166):.4f}")
print(f"    e176_igk alone: LOMO={lomo_map(e176_igk):.4f}")
print(f"    e183_tabpfn:    LOMO={lomo_map(e183_tabpfn):.4f}")
print(f"    THIS BLEND:     LOMO={final_lomo:.4f}")
print(f"    Delta vs best single: {final_lomo - lomo_map(e166):+.4f}")

print(f"\n{'='*70}")
