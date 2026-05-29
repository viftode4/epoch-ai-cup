"""E185: Exhaustive ensemble blend optimization with physics Cormorant prior.

Searches for optimal blend of existing OOF predictions using LOMO validation.
Then applies physics-based Cormorant prior on the best blend.

LOMO = Leave-One-Month-Out (4 folds: months 1, 4, 9, 10).
Metric = macro-averaged mAP (sklearn).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from itertools import combinations
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map

# ── Load labels and months ──────────────────────────────────────────
train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
TRAIN_MONTHS = [1, 4, 9, 10]
N = len(y)
assert N == 2601

CORM_IDX = CLASSES.index("Cormorants")  # 2

# ── LOMO evaluation ────────────────────────────────────────────────
def lomo_map(oof_preds):
    """Compute LOMO macro mAP (4-fold, leave-one-month-out)."""
    fold_maps = []
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        mAP, _ = compute_map(y[mask], oof_preds[mask])
        fold_maps.append(mAP)
    return np.mean(fold_maps)


def lomo_map_perclass(oof_preds):
    """Compute LOMO per-class APs (averaged across folds)."""
    per_class = {c: [] for c in CLASSES}
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        _, pc = compute_map(y[mask], oof_preds[mask])
        for c in CLASSES:
            per_class[c].append(pc[c])
    return {c: np.mean(v) for c, v in per_class.items()}


# ── Load all candidate predictions ─────────────────────────────────
def load_oof(name):
    path = ROOT / f"oof_{name}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape != (N, 9):
        return None
    # Must be probabilities (rows sum to ~1)
    row_sums = arr.sum(axis=1)
    if np.abs(row_sums - 1.0).max() > 0.5:
        return None  # logits or ranker scores, skip
    return arr


CANDIDATES = {
    "e175_best": load_oof("e175_best"),
    "e175_cb": load_oof("e175_cb"),
    "e175_lgb": load_oof("e175_lgb"),
    "e175_xgb": load_oof("e175_xgb"),
    "e175_dro": load_oof("e175_dro"),
    "e183_tabpfn": load_oof("e183_tabpfn"),
    "e180_cnn": load_oof("e180_cnn"),
    "e182_cnn_v3": load_oof("e182_cnn_v3"),
    "e179_cb": load_oof("e179_cb"),
    "e179_lgb": load_oof("e179_lgb"),
    "e180_spatial": load_oof("e180_spatial"),
    "e181_physics": load_oof("e181_physics"),
    "e180_rcs_linear": load_oof("e180_rcs_linear"),
    "e176_gmm": load_oof("e176_gmm"),
    "e176_iso": load_oof("e176_iso"),
    "e176_iso_gmm_knn": load_oof("e176_iso_gmm_knn"),
    "e177_20seed": load_oof("e177_20seed"),
    "e177_diverse": load_oof("e177_diverse"),
    "e177_dro": load_oof("e177_dro"),
    "e177_tta": load_oof("e177_tta"),
    "e166": load_oof("e166"),
    "e172": load_oof("e172"),
    "e173": load_oof("e173"),
    "e173_clean": load_oof("e173_clean"),
    "e170": load_oof("e170"),
}

# Filter out None
CANDIDATES = {k: v for k, v in CANDIDATES.items() if v is not None}
print(f"\nLoaded {len(CANDIDATES)} candidate predictions:")

# ── Individual LOMO scores ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  INDIVIDUAL LOMO SCORES")
print("=" * 70)
individual_scores = {}
for name, oof in sorted(CANDIDATES.items()):
    score = lomo_map(oof)
    individual_scores[name] = score
    pc = lomo_map_perclass(oof)
    corm_ap = pc["Cormorants"]
    print(f"  {name:25s}  LOMO={score:.4f}  Corm_AP={corm_ap:.4f}")

# Sort by LOMO
ranked = sorted(individual_scores.items(), key=lambda x: -x[1])
print(f"\n  Top 5: {', '.join(f'{n}({s:.4f})' for n, s in ranked[:5])}")

# ── Rank-power blend function ──────────────────────────────────────
def rank_power_blend(preds_list, weights, power=1.0):
    """Rank-power ensemble: rank-transform, raise to power, weighted average.

    Rank transform makes ensemble robust to different calibration scales.
    Power parameter controls sharpness (>1 = sharper, <1 = smoother).
    """
    n_samples, n_classes = preds_list[0].shape
    blended = np.zeros((n_samples, n_classes))
    total_w = sum(weights)

    for pred, w in zip(preds_list, weights):
        # Rank transform per class (handles different calibration scales)
        ranked = np.zeros_like(pred)
        for c in range(n_classes):
            order = pred[:, c].argsort().argsort()
            ranked[:, c] = (order + 1) / n_samples  # ranks in [1/n, 1]

        # Power transform
        if power != 1.0:
            ranked = ranked ** power

        blended += w * ranked

    blended /= total_w

    # Renormalize rows to sum to 1
    row_sums = blended.sum(axis=1, keepdims=True)
    blended = blended / np.maximum(row_sums, 1e-12)

    return blended


def simple_avg_blend(preds_list, weights):
    """Simple weighted average blend (probability space)."""
    n_samples = preds_list[0].shape[0]
    blended = np.zeros((n_samples, 9))
    total_w = sum(weights)
    for pred, w in zip(preds_list, weights):
        blended += w * pred
    blended /= total_w
    return blended


# ── Phase 1: 2-way blends (fine grid) ─────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1: 2-WAY BLEND SEARCH")
print("=" * 70)

top_names = [n for n, _ in ranked[:10]]  # Top 10 candidates
best_2way = {"score": 0, "names": None, "weights": None, "method": None, "power": None}

for n1, n2 in combinations(top_names, 2):
    p1, p2 = CANDIDATES[n1], CANDIDATES[n2]

    # Simple average blend: sweep weights
    for w1 in np.arange(0.1, 1.0, 0.05):
        w2 = 1.0 - w1
        blend = simple_avg_blend([p1, p2], [w1, w2])
        score = lomo_map(blend)
        if score > best_2way["score"]:
            best_2way = {"score": score, "names": (n1, n2),
                        "weights": (w1, w2), "method": "avg", "power": None}

    # Rank-power blend: sweep weights and power
    for w1 in np.arange(0.1, 1.0, 0.1):
        w2 = 1.0 - w1
        for power in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            blend = rank_power_blend([p1, p2], [w1, w2], power)
            score = lomo_map(blend)
            if score > best_2way["score"]:
                best_2way = {"score": score, "names": (n1, n2),
                            "weights": (w1, w2), "method": "rank", "power": power}

print(f"\n  Best 2-way: {best_2way['names']}")
print(f"  Weights: {best_2way['weights']}")
print(f"  Method: {best_2way['method']}, Power: {best_2way['power']}")
print(f"  LOMO: {best_2way['score']:.4f}")

# Per-class breakdown of best 2-way
if best_2way["names"]:
    n1, n2 = best_2way["names"]
    w1, w2 = best_2way["weights"]
    if best_2way["method"] == "avg":
        best_2way_preds = simple_avg_blend([CANDIDATES[n1], CANDIDATES[n2]], [w1, w2])
    else:
        best_2way_preds = rank_power_blend([CANDIDATES[n1], CANDIDATES[n2]], [w1, w2], best_2way["power"])
    pc = lomo_map_perclass(best_2way_preds)
    print(f"\n  Per-class APs:")
    for c in CLASSES:
        print(f"    {c:15s}: {pc[c]:.4f}")


# ── Phase 2: 3-way blends (coarser grid, top candidates) ──────────
print("\n" + "=" * 70)
print("  PHASE 2: 3-WAY BLEND SEARCH")
print("=" * 70)

top6 = [n for n, _ in ranked[:8]]
best_3way = {"score": 0, "names": None, "weights": None, "method": None, "power": None}

for combo in combinations(top6, 3):
    preds = [CANDIDATES[n] for n in combo]

    # Coarser weight grid for 3-way
    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue

            # Simple average
            blend = simple_avg_blend(preds, [w1, w2, w3])
            score = lomo_map(blend)
            if score > best_3way["score"]:
                best_3way = {"score": score, "names": combo,
                            "weights": (w1, w2, w3), "method": "avg", "power": None}

            # Rank-power (fewer powers for speed)
            for power in [0.5, 1.0, 2.0]:
                blend = rank_power_blend(preds, [w1, w2, w3], power)
                score = lomo_map(blend)
                if score > best_3way["score"]:
                    best_3way = {"score": score, "names": combo,
                                "weights": (w1, w2, w3), "method": "rank", "power": power}

print(f"\n  Best 3-way: {best_3way['names']}")
print(f"  Weights: {best_3way['weights']}")
print(f"  Method: {best_3way['method']}, Power: {best_3way['power']}")
print(f"  LOMO: {best_3way['score']:.4f}")


# ── Phase 3: 4-way blends (very coarse, top candidates) ───────────
print("\n" + "=" * 70)
print("  PHASE 3: 4-WAY BLEND SEARCH")
print("=" * 70)

top5 = [n for n, _ in ranked[:6]]
best_4way = {"score": 0, "names": None, "weights": None, "method": None, "power": None}

for combo in combinations(top5, 4):
    preds = [CANDIDATES[n] for n in combo]

    # Very coarse grid: step 0.15
    for w1 in np.arange(0.1, 0.7, 0.15):
        for w2 in np.arange(0.1, 0.8 - w1, 0.15):
            for w3 in np.arange(0.1, 0.9 - w1 - w2, 0.15):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0.05:
                    continue

                blend = simple_avg_blend(preds, [w1, w2, w3, w4])
                score = lomo_map(blend)
                if score > best_4way["score"]:
                    best_4way = {"score": score, "names": combo,
                                "weights": (w1, w2, w3, w4), "method": "avg", "power": None}

                for power in [1.0, 2.0]:
                    blend = rank_power_blend(preds, [w1, w2, w3, w4], power)
                    score = lomo_map(blend)
                    if score > best_4way["score"]:
                        best_4way = {"score": score, "names": combo,
                                    "weights": (w1, w2, w3, w4), "method": "rank", "power": power}

print(f"\n  Best 4-way: {best_4way['names']}")
print(f"  Weights: {best_4way['weights']}")
print(f"  Method: {best_4way['method']}, Power: {best_4way['power']}")
print(f"  LOMO: {best_4way['score']:.4f}")


# ── Phase 4: Physics Cormorant prior ──────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 4: PHYSICS CORMORANT PRIOR ON BEST BLEND")
print("=" * 70)

# Select the best blend overall
best_blends = [
    ("best_single", ranked[0][1], CANDIDATES[ranked[0][0]]),
    ("best_2way", best_2way["score"], best_2way_preds if best_2way["names"] else None),
]

# Build 3-way preds if found
if best_3way["names"]:
    n1, n2, n3 = best_3way["names"]
    w1, w2, w3 = best_3way["weights"]
    if best_3way["method"] == "avg":
        best_3way_preds = simple_avg_blend(
            [CANDIDATES[n1], CANDIDATES[n2], CANDIDATES[n3]], [w1, w2, w3])
    else:
        best_3way_preds = rank_power_blend(
            [CANDIDATES[n1], CANDIDATES[n2], CANDIDATES[n3]], [w1, w2, w3], best_3way["power"])
    best_blends.append(("best_3way", best_3way["score"], best_3way_preds))

if best_4way["names"]:
    ns = best_4way["names"]
    ws = best_4way["weights"]
    preds_list = [CANDIDATES[n] for n in ns]
    if best_4way["method"] == "avg":
        best_4way_preds = simple_avg_blend(preds_list, list(ws))
    else:
        best_4way_preds = rank_power_blend(preds_list, list(ws), best_4way["power"])
    best_blends.append(("best_4way", best_4way["score"], best_4way_preds))

# Pick the overall best blend
best_blends = [(n, s, p) for n, s, p in best_blends if p is not None]
best_blends.sort(key=lambda x: -x[1])
best_name, best_score, best_preds = best_blends[0]
print(f"\n  Using '{best_name}' with LOMO={best_score:.4f} as base for Cormorant prior")

# Compute physics features from raw trajectory data
print("\n  Computing Cormorant physics features from raw trajectories...")

def compute_cormorant_likelihood(df):
    """Compute per-track Cormorant likelihood from raw trajectory data.

    Cormorant signature (from domain knowledge):
    - Speed: 13-18 m/s (distinctive narrow range)
    - Straightness: >0.85 (very straight flyers)
    - Altitude: <60m (low flyers)
    - RCS: >-27 dBm2 (larger than most songbirds)
    - RCS scintillation: >2.5 dB (high RCS variance from continuous flapping)
    - RCS autocorrelation lag-1: >0.35 (periodic wingbeat pattern)
    - Speed CV: <0.25 (very steady speed)
    """
    n_rows = len(df)
    likelihood = np.zeros(n_rows)

    for i in range(n_rows):
        try:
            pts = parse_ewkb_4d(df.iloc[i]["trajectory"])
            times = parse_trajectory_time(df.iloc[i]["trajectory_time"])
        except Exception:
            continue

        if len(pts) < 3:
            continue

        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs = np.array([p[3] for p in pts])
        n = len(pts)

        # Speed
        if n > 1:
            from src.features import haversine
            dists = np.array([haversine(lons[j], lats[j], lons[j+1], lats[j+1])
                             for j in range(n-1)])
            dt = np.maximum(np.diff(times), 0.5)
            speeds = dists / dt
            speed_mean = np.mean(speeds)
            speed_cv = np.std(speeds) / max(speed_mean, 1e-6)
        else:
            speed_mean = 0
            speed_cv = 1.0

        # Straightness
        total_dist = sum(dists) if n > 1 else 0
        straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
        straightness = straight_dist / max(total_dist, 1e-6) if total_dist > 0 else 0

        # Altitude
        alt_mean = np.mean(alts)

        # RCS stats
        rcs_mean = np.mean(rcs)
        rcs_std = np.std(rcs)

        # RCS autocorrelation lag-1
        if n > 2:
            rcs_centered = rcs - rcs_mean
            var = np.var(rcs)
            if var > 1e-8:
                rcs_ac1 = np.corrcoef(rcs_centered[:-1], rcs_centered[1:])[0, 1]
            else:
                rcs_ac1 = 0
        else:
            rcs_ac1 = 0

        # Score each criterion (soft sigmoid-like scoring)
        score = 0.0
        n_criteria = 0

        # Speed 13-18 m/s (Cormorant cruise speed)
        if 13.0 <= speed_mean <= 18.0:
            score += 1.0
        elif 10.0 <= speed_mean <= 21.0:
            score += 0.5
        n_criteria += 1

        # Straightness > 0.85
        if straightness > 0.85:
            score += 1.0
        elif straightness > 0.70:
            score += 0.5
        n_criteria += 1

        # Altitude < 60m
        if alt_mean < 60:
            score += 1.0
        elif alt_mean < 100:
            score += 0.5
        n_criteria += 1

        # RCS > -27 dBm2
        if rcs_mean > -27:
            score += 1.0
        elif rcs_mean > -30:
            score += 0.5
        n_criteria += 1

        # RCS scintillation > 2.5 dB
        if rcs_std > 2.5:
            score += 1.0
        elif rcs_std > 1.5:
            score += 0.5
        n_criteria += 1

        # RCS AC1 > 0.35
        if rcs_ac1 > 0.35:
            score += 1.0
        elif rcs_ac1 > 0.20:
            score += 0.5
        n_criteria += 1

        # Speed CV < 0.25
        if speed_cv < 0.25:
            score += 1.0
        elif speed_cv < 0.40:
            score += 0.5
        n_criteria += 1

        likelihood[i] = score / n_criteria

    return likelihood


corm_likelihood = compute_cormorant_likelihood(train_df)
print(f"  Likelihood stats: mean={corm_likelihood.mean():.3f}, "
      f"max={corm_likelihood.max():.3f}, "
      f">0.5: {(corm_likelihood > 0.5).sum()} tracks")

# Check likelihood on actual cormorants vs others
corm_mask = y == CORM_IDX
print(f"  On actual Cormorants (n={corm_mask.sum()}): mean={corm_likelihood[corm_mask].mean():.3f}")
print(f"  On non-Cormorants: mean={corm_likelihood[~corm_mask].mean():.3f}")

# Sweep gamma for Cormorant boost
print("\n  Sweeping gamma for Cormorant prior boost...")
best_gamma = {"score": best_score, "gamma": 0, "preds": best_preds}

for gamma in np.arange(0.0, 3.1, 0.1):
    boosted = best_preds.copy()
    boost = 1.0 + corm_likelihood * gamma
    boosted[:, CORM_IDX] *= boost
    # Renormalize
    row_sums = boosted.sum(axis=1, keepdims=True)
    boosted = boosted / np.maximum(row_sums, 1e-12)

    score = lomo_map(boosted)
    if score > best_gamma["score"]:
        best_gamma = {"score": score, "gamma": gamma, "preds": boosted}

    if gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pc = lomo_map_perclass(boosted)
        print(f"    gamma={gamma:.1f}: LOMO={score:.4f}, Corm_AP={pc['Cormorants']:.4f}")

print(f"\n  Best gamma: {best_gamma['gamma']:.1f}, LOMO: {best_gamma['score']:.4f}")


# ── Phase 5: Per-class blend weights (Cormorant column) ──────────
print("\n" + "=" * 70)
print("  PHASE 5: PER-CLASS BLEND WEIGHTS (CORMORANT COLUMN)")
print("=" * 70)

# Take the two best individual models and try different weights for Cormorant col
best_perclass = {"score": 0, "config": None}

# Use top 3 individual models
top3 = [n for n, _ in ranked[:3]]
print(f"  Using top 3: {top3}")

for n1, n2 in combinations(top3, 2):
    p1, p2 = CANDIDATES[n1], CANDIDATES[n2]

    # For each global weight ratio, also vary the Cormorant column weight
    for w_global in np.arange(0.1, 0.9, 0.1):
        w_other = 1.0 - w_global

        for w_corm_boost in np.arange(0.0, 1.0, 0.1):
            # Blend: use w_global for all classes, but override Cormorant column
            blend = np.zeros_like(p1)
            for c in range(9):
                if c == CORM_IDX:
                    # Different weight for Cormorant
                    blend[:, c] = w_corm_boost * p1[:, c] + (1.0 - w_corm_boost) * p2[:, c]
                else:
                    blend[:, c] = w_global * p1[:, c] + w_other * p2[:, c]

            # Renormalize
            row_sums = blend.sum(axis=1, keepdims=True)
            blend = blend / np.maximum(row_sums, 1e-12)

            score = lomo_map(blend)
            if score > best_perclass["score"]:
                best_perclass = {
                    "score": score,
                    "config": f"{n1}+{n2}, global_w={w_global:.1f}, corm_w={w_corm_boost:.1f}"
                }

# Also try 3-model per-class blend
for w1 in np.arange(0.1, 0.7, 0.15):
    for w2 in np.arange(0.1, 0.8 - w1, 0.15):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue

        p1, p2, p3 = CANDIDATES[top3[0]], CANDIDATES[top3[1]], CANDIDATES[top3[2]]

        for corm_src_idx in range(3):
            # Use one model's Cormorant column exclusively
            blend = w1 * p1 + w2 * p2 + w3 * p3

            # Override Cormorant column with single model
            corm_sources = [p1, p2, p3]
            blend[:, CORM_IDX] = corm_sources[corm_src_idx][:, CORM_IDX]

            # Renormalize
            row_sums = blend.sum(axis=1, keepdims=True)
            blend = blend / np.maximum(row_sums, 1e-12)

            score = lomo_map(blend)
            if score > best_perclass["score"]:
                best_perclass = {
                    "score": score,
                    "config": f"3-way {top3}, w=({w1:.2f},{w2:.2f},{w3:.2f}), "
                             f"Corm col from {top3[corm_src_idx]}"
                }

print(f"\n  Best per-class config: {best_perclass['config']}")
print(f"  LOMO: {best_perclass['score']:.4f}")


# ── Phase 6: Detector injection ───────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 6: DETECTOR INJECTION (e184_corm_det, e184_wader_det)")
print("=" * 70)

# Load 1D detector scores
corm_det = None
wader_det = None
try:
    cd = np.load(ROOT / "oof_e184_corm_det.npy")
    if cd.shape == (N,):
        corm_det = cd
        print(f"  Loaded corm_det: mean={cd.mean():.4f}, on actual corms={cd[corm_mask].mean():.4f}")
except Exception:
    pass

try:
    wd = np.load(ROOT / "oof_e184_wader_det.npy")
    if wd.shape == (N,):
        wader_det = wd
        WADER_IDX = CLASSES.index("Waders")
        wader_mask = y == WADER_IDX
        print(f"  Loaded wader_det: mean={wd.mean():.4f}, on actual waders={wd[wader_mask].mean():.4f}")
except Exception:
    pass

best_det = {"score": best_gamma["score"], "config": "no detector", "preds": best_gamma["preds"]}

if corm_det is not None:
    base = best_gamma["preds"]
    for alpha in np.arange(0.0, 2.1, 0.1):
        boosted = base.copy()
        boosted[:, CORM_IDX] *= (1.0 + alpha * corm_det)
        row_sums = boosted.sum(axis=1, keepdims=True)
        boosted = boosted / np.maximum(row_sums, 1e-12)
        score = lomo_map(boosted)
        if score > best_det["score"]:
            best_det = {"score": score, "config": f"corm_det alpha={alpha:.1f}", "preds": boosted}

if wader_det is not None:
    # Try injecting wader detector too
    base_for_wader = best_det["preds"]
    WADER_IDX = CLASSES.index("Waders")
    for alpha in np.arange(0.0, 2.1, 0.1):
        boosted = base_for_wader.copy()
        boosted[:, WADER_IDX] *= (1.0 + alpha * wader_det)
        row_sums = boosted.sum(axis=1, keepdims=True)
        boosted = boosted / np.maximum(row_sums, 1e-12)
        score = lomo_map(boosted)
        if score > best_det["score"]:
            best_det = {"score": score, "config": f"corm_det+wader_det alpha={alpha:.1f}", "preds": boosted}

print(f"\n  Best with detectors: {best_det['config']}")
print(f"  LOMO: {best_det['score']:.4f}")


# ── Phase 7: Combined best ────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 7: FINAL SUMMARY")
print("=" * 70)

all_results = [
    ("Best single", ranked[0][0], ranked[0][1]),
    ("Best 2-way", str(best_2way["names"]), best_2way["score"]),
    ("Best 3-way", str(best_3way["names"]), best_3way["score"]),
    ("Best 4-way", str(best_4way["names"]) if best_4way["names"] else "N/A", best_4way["score"]),
    ("Best + Corm prior", f"gamma={best_gamma['gamma']:.1f}", best_gamma["score"]),
    ("Best per-class blend", best_perclass["config"], best_perclass["score"]),
    ("Best + detectors", best_det["config"], best_det["score"]),
]

print(f"\n  {'Method':30s}  {'Config':50s}  {'LOMO':>7}")
print(f"  {'-'*30}  {'-'*50}  {'-'*7}")
for method, config, score in sorted(all_results, key=lambda x: -x[2]):
    marker = " <<<" if score == max(r[2] for r in all_results) else ""
    print(f"  {method:30s}  {str(config):50s}  {score:.4f}{marker}")

# Overall best
overall_best_score = max(r[2] for r in all_results)
overall_best = [r for r in all_results if r[2] == overall_best_score][0]
print(f"\n  OVERALL BEST: {overall_best[0]} -> LOMO={overall_best_score:.4f}")

# Per-class breakdown of overall best
if overall_best[0] == "Best + detectors":
    final_preds = best_det["preds"]
elif overall_best[0] == "Best + Corm prior":
    final_preds = best_gamma["preds"]
elif overall_best[0] == "Best 2-way":
    final_preds = best_2way_preds
elif overall_best[0] == "Best 3-way":
    final_preds = best_3way_preds
elif overall_best[0] == "Best 4-way":
    final_preds = best_4way_preds
else:
    final_preds = CANDIDATES[ranked[0][0]]

pc = lomo_map_perclass(final_preds)
print(f"\n  Per-class AP breakdown (LOMO):")
for c in CLASSES:
    marker = " <-- weak" if pc[c] < 0.5 else ""
    print(f"    {c:15s}: {pc[c]:.4f}{marker}")

# SKF for reference
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_aps = []
for _, val_idx in skf.split(np.zeros(N), y):
    mAP, _ = compute_map(y[val_idx], final_preds[val_idx])
    skf_aps.append(mAP)
skf_score = np.mean(skf_aps)
print(f"\n  Reference SKF (5-fold): {skf_score:.4f}")

print(f"\n{'='*70}")
print(f"  DONE. Best achievable LOMO with existing predictions: {overall_best_score:.4f}")
print(f"{'='*70}")
