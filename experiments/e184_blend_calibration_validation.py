"""E184 — Exhaustive blend/calibration validation using existing predictions ONLY.

NO model training. Uses only saved OOF/test .npy files.

Sections:
  1. Exhaustive blend search (rank-power, LOMO-optimized)
  2. Dirichlet (per-class temperature) calibration on TabPFN
  3. Per-class blend weights (column-wise optimization)
  4. Physics prior on Cormorant column of best blend
  5. Correlation analysis (diversity between models)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score

from src.data import CLASSES, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map

N_CLASSES = len(CLASSES)


# ===========================================================================
# Data loading
# ===========================================================================

def load_data():
    """Load train data, labels, and months."""
    train_df = load_train()
    y = np.asarray(
        pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int
    )
    months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    return train_df, y, months


def lomo_map(y, preds, months):
    """Leave-one-month-out mAP: hold out each month, compute mAP, average."""
    unique_months = sorted(set(months))
    scores = []
    for m in unique_months:
        mask = months == m
        if mask.sum() < 5:
            continue
        # Need at least 2 classes present
        classes_present = len(set(y[mask]))
        if classes_present < 2:
            continue
        score, _ = compute_map(y[mask], preds[mask])
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def lomo_per_class(y, preds, months):
    """LOMO per-class AP: for each class, average across months."""
    unique_months = sorted(set(months))
    class_aps = {c: [] for c in range(N_CLASSES)}
    for m in unique_months:
        mask = months == m
        if mask.sum() < 5:
            continue
        y_bin = np.eye(N_CLASSES)[y[mask]]
        for c in range(N_CLASSES):
            if y_bin[:, c].sum() > 0:
                ap = average_precision_score(y_bin[:, c], preds[mask, c])
                class_aps[c].append(ap)
    result = {}
    for c in range(N_CLASSES):
        result[CLASSES[c]] = float(np.mean(class_aps[c])) if class_aps[c] else 0.0
    return result


def load_oof_safe(name):
    """Load OOF predictions, return None if missing or wrong shape."""
    path = ROOT / f"oof_{name}.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(float)
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        print(f"  SKIP {name}: shape {arr.shape} (expected (N, {N_CLASSES}))")
        return None
    return arr


def load_test_safe(name):
    """Load test predictions, return None if missing or wrong shape."""
    path = ROOT / f"test_{name}.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(float)
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        return None
    return arr


# ===========================================================================
# Utilities
# ===========================================================================

def rank_power_blend(preds_list, weights, power=0.5):
    """Blend predictions using rank-power averaging.

    For each model: rank each class column, raise to power, weight, sum, re-rank.
    """
    n_samples = preds_list[0].shape[0]
    blended = np.zeros((n_samples, N_CLASSES))
    total_w = sum(weights)
    for pred, w in zip(preds_list, weights):
        for c in range(N_CLASSES):
            ranks = rankdata(pred[:, c]) / n_samples  # normalize to [0,1]
            blended[:, c] += w * (ranks ** power)
    blended /= total_w
    return blended


def simple_blend(preds_list, weights):
    """Weighted average blend of probability arrays."""
    total_w = sum(weights)
    blended = np.zeros_like(preds_list[0])
    for pred, w in zip(preds_list, weights):
        blended += w * pred
    blended /= total_w
    # Ensure valid probabilities
    blended = np.clip(blended, 1e-8, None)
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


# ===========================================================================
# SECTION 1: Exhaustive blend search
# ===========================================================================

def section1_exhaustive_blend(y, months, n_train):
    print("\n" + "=" * 70)
    print("  SECTION 1: EXHAUSTIVE BLEND SEARCH (LOMO-optimized)")
    print("=" * 70)

    # Define models to try
    model_names = [
        "e175_best", "e175_ranker", "e175_cb", "e183_tabpfn",
        "e180_cnn", "e182_cnn_v3",
        # Also include some other potentially useful ones
        "e175_lgb", "e175_dro", "e175_xgb",
        "e176_iso", "e176_gmm", "e176_iso_gmm_knn",
        "e177_20seed", "e177_diverse",
        "e180_spatial", "e180_rcs_linear",
    ]

    # Load available models
    models = {}
    for name in model_names:
        oof = load_oof_safe(name)
        if oof is not None and oof.shape[0] == n_train:
            models[name] = oof
            score = lomo_map(y, oof, months)
            print(f"  Loaded {name:25s} shape={oof.shape}  LOMO={score:.4f}")
        elif oof is not None:
            print(f"  SKIP {name}: shape mismatch ({oof.shape[0]} vs {n_train})")

    if len(models) < 2:
        print("  Not enough models for blending. Need at least 2.")
        return None, None

    model_list = list(models.keys())
    model_preds = [models[k] for k in model_list]

    # --- Pairs ---
    print(f"\n  Searching {len(model_list)} models, {len(list(combinations(range(len(model_list)), 2)))} pairs...")
    results = []

    # Weight grid for pairs
    pair_weights = np.arange(0.1, 1.0, 0.1)

    for i, j in combinations(range(len(model_list)), 2):
        for w1 in pair_weights:
            w2 = 1.0 - w1
            # Rank-power blend
            blended = rank_power_blend([model_preds[i], model_preds[j]], [w1, w2], power=0.5)
            score = lomo_map(y, blended, months)
            results.append({
                "type": "pair_rank",
                "models": f"{model_list[i]} + {model_list[j]}",
                "weights": f"{w1:.1f}/{w2:.1f}",
                "lomo": score,
                "indices": (i, j),
                "w": (w1, w2),
            })
            # Simple average blend
            blended2 = simple_blend([model_preds[i], model_preds[j]], [w1, w2])
            score2 = lomo_map(y, blended2, months)
            results.append({
                "type": "pair_avg",
                "models": f"{model_list[i]} + {model_list[j]}",
                "weights": f"{w1:.1f}/{w2:.1f}",
                "lomo": score2,
                "indices": (i, j),
                "w": (w1, w2),
            })

    # --- Triples (use coarser grid) ---
    triple_weights = [0.2, 0.4, 0.6, 0.8]
    n_triples = len(list(combinations(range(len(model_list)), 3)))
    print(f"  Searching {n_triples} triples...")

    for i, j, k in combinations(range(len(model_list)), 3):
        for w1 in triple_weights:
            for w2 in triple_weights:
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                blended = rank_power_blend(
                    [model_preds[i], model_preds[j], model_preds[k]],
                    [w1, w2, w3], power=0.5
                )
                score = lomo_map(y, blended, months)
                results.append({
                    "type": "triple_rank",
                    "models": f"{model_list[i]} + {model_list[j]} + {model_list[k]}",
                    "weights": f"{w1:.1f}/{w2:.1f}/{w3:.1f}",
                    "lomo": score,
                    "indices": (i, j, k),
                    "w": (w1, w2, w3),
                })

    # Sort and report top 10
    results.sort(key=lambda x: x["lomo"], reverse=True)
    print(f"\n  TOP 10 BLENDS BY LOMO ({len(results)} total searched):")
    print(f"  {'Rank':>4}  {'LOMO':>7}  {'Type':>12}  {'Weights':>15}  Models")
    for rank, r in enumerate(results[:10], 1):
        print(f"  {rank:4d}  {r['lomo']:.4f}  {r['type']:>12}  {r['weights']:>15}  {r['models']}")

    # Per-class AP for the best blend
    best = results[0]
    print(f"\n  BEST BLEND per-class AP:")
    if best["type"].startswith("pair"):
        i, j = best["indices"]
        if "rank" in best["type"]:
            best_preds = rank_power_blend(
                [model_preds[i], model_preds[j]], list(best["w"]), power=0.5
            )
        else:
            best_preds = simple_blend(
                [model_preds[i], model_preds[j]], list(best["w"])
            )
    else:
        i, j, k = best["indices"]
        best_preds = rank_power_blend(
            [model_preds[i], model_preds[j], model_preds[k]],
            list(best["w"]), power=0.5
        )

    per_class = lomo_per_class(y, best_preds, months)
    for cls, ap in per_class.items():
        marker = " <-- weak" if ap < 0.5 else ""
        print(f"    {cls:15s}: {ap:.4f}{marker}")

    # Also report SKF for comparison
    skf_score, skf_per_class = compute_map(y, best_preds)
    print(f"\n  Best blend SKF: {skf_score:.4f}")

    return best_preds, models


# ===========================================================================
# SECTION 2: Dirichlet (per-class temperature) calibration
# ===========================================================================

def section2_dirichlet_calibration(y, months):
    print("\n" + "=" * 70)
    print("  SECTION 2: DIRICHLET (PER-CLASS TEMPERATURE) CALIBRATION")
    print("=" * 70)

    oof_tabpfn = load_oof_safe("e183_tabpfn")
    if oof_tabpfn is None:
        print("  TabPFN OOF not available, trying e175_best...")
        oof_tabpfn = load_oof_safe("e175_best")
    if oof_tabpfn is None or oof_tabpfn.shape[0] != len(y):
        print("  No suitable predictions available. Skipping.")
        return

    # Baseline
    base_lomo = lomo_map(y, oof_tabpfn, months)
    base_per_class = lomo_per_class(y, oof_tabpfn, months)
    print(f"\n  Baseline LOMO: {base_lomo:.4f}")
    for cls, ap in base_per_class.items():
        print(f"    {cls:15s}: {ap:.4f}")

    # Per-class temperature search
    T_grid = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    print(f"\n  Searching per-class temperatures in {T_grid}...")

    logits = np.log(np.clip(oof_tabpfn, 1e-8, 1.0))

    # For each class, find optimal temperature
    best_temps = np.ones(N_CLASSES)
    for c in range(N_CLASSES):
        best_ap = -1
        best_T = 1.0
        for T in T_grid:
            # Scale only this class's logits
            scaled_logits = logits.copy()
            scaled_logits[:, c] = logits[:, c] / T
            # Re-softmax
            scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
            exp_s = np.exp(scaled_logits)
            calibrated = exp_s / exp_s.sum(axis=1, keepdims=True)
            # Evaluate LOMO AP for this class
            per_class = lomo_per_class(y, calibrated, months)
            ap = per_class[CLASSES[c]]
            if ap > best_ap:
                best_ap = ap
                best_T = T
        best_temps[c] = best_T
        print(f"    {CLASSES[c]:15s}: best T={best_T:5.1f}  AP={best_ap:.4f}  (was {base_per_class[CLASSES[c]]:.4f}, delta={best_ap - base_per_class[CLASSES[c]]:+.4f})")

    # Apply all best temperatures jointly
    scaled_logits = logits.copy()
    for c in range(N_CLASSES):
        scaled_logits[:, c] = logits[:, c] / best_temps[c]
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled_logits)
    calibrated = exp_s / exp_s.sum(axis=1, keepdims=True)

    joint_lomo = lomo_map(y, calibrated, months)
    joint_per_class = lomo_per_class(y, calibrated, months)
    print(f"\n  Joint per-class temperature LOMO: {joint_lomo:.4f} (baseline: {base_lomo:.4f}, delta: {joint_lomo - base_lomo:+.4f})")
    for cls, ap in joint_per_class.items():
        delta = ap - base_per_class[cls]
        marker = f" {'+'if delta>=0 else ''}{delta:.4f}"
        print(f"    {cls:15s}: {ap:.4f}{marker}")

    # Also try global temperature sweep
    print(f"\n  Global temperature sweep:")
    for T in T_grid:
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        calibrated = exp_s / exp_s.sum(axis=1, keepdims=True)
        score = lomo_map(y, calibrated, months)
        print(f"    T={T:5.1f}  LOMO={score:.4f}  delta={score - base_lomo:+.4f}")


# ===========================================================================
# SECTION 3: Per-class blend weights
# ===========================================================================

def section3_per_class_blend(y, months, n_train):
    print("\n" + "=" * 70)
    print("  SECTION 3: PER-CLASS BLEND WEIGHTS (column-wise optimization)")
    print("=" * 70)

    # Load key models
    model_names = [
        "e175_best", "e175_ranker", "e175_cb", "e183_tabpfn",
        "e182_cnn_v3", "e175_lgb", "e175_xgb", "e175_dro",
        "e176_iso", "e180_spatial", "e180_rcs_linear",
    ]

    models = {}
    for name in model_names:
        oof = load_oof_safe(name)
        if oof is not None and oof.shape[0] == n_train:
            models[name] = oof

    if len(models) < 2:
        print("  Not enough models. Skipping.")
        return

    model_list = list(models.keys())
    model_preds = [models[k] for k in model_list]
    n_models = len(model_list)

    print(f"\n  Models available: {model_list}")

    # For each class, find the best single model and best pair
    print(f"\n  PER-CLASS BEST SINGLE MODEL:")
    unique_months = sorted(set(months))

    for c in range(N_CLASSES):
        best_model = None
        best_ap = -1
        for mi, name in enumerate(model_list):
            # LOMO for this specific class
            aps = []
            for m in unique_months:
                mask = months == m
                y_bin = (y[mask] == c).astype(int)
                if y_bin.sum() == 0:
                    continue
                ap = average_precision_score(y_bin, model_preds[mi][mask, c])
                aps.append(ap)
            avg_ap = float(np.mean(aps)) if aps else 0
            if avg_ap > best_ap:
                best_ap = avg_ap
                best_model = name
        print(f"    {CLASSES[c]:15s}: {best_model:25s} AP={best_ap:.4f}")

    # Per-class optimal pair blend
    print(f"\n  PER-CLASS BEST PAIR BLEND:")
    per_class_best_blend = {}
    for c in range(N_CLASSES):
        best_score = -1
        best_config = None
        for i, j in combinations(range(n_models), 2):
            for w in np.arange(0.1, 1.0, 0.1):
                blended_col = w * model_preds[i][:, c] + (1 - w) * model_preds[j][:, c]
                # LOMO for this class
                aps = []
                for m in unique_months:
                    mask = months == m
                    y_bin = (y[mask] == c).astype(int)
                    if y_bin.sum() == 0:
                        continue
                    ap = average_precision_score(y_bin, blended_col[mask])
                    aps.append(ap)
                avg_ap = float(np.mean(aps)) if aps else 0
                if avg_ap > best_score:
                    best_score = avg_ap
                    best_config = (model_list[i], model_list[j], w)
        per_class_best_blend[c] = best_config
        print(f"    {CLASSES[c]:15s}: {best_config[0]:20s} ({best_config[2]:.1f}) + {best_config[1]:20s} ({1-best_config[2]:.1f})  AP={best_score:.4f}")

    # Build a Frankenstein blend: per-class optimal columns
    print(f"\n  FRANKENSTEIN BLEND (per-class optimal columns):")
    frank = np.zeros((n_train, N_CLASSES))
    for c in range(N_CLASSES):
        name_a, name_b, w = per_class_best_blend[c]
        frank[:, c] = w * models[name_a][:, c] + (1 - w) * models[name_b][:, c]

    # Normalize
    frank = np.clip(frank, 1e-8, None)
    frank /= frank.sum(axis=1, keepdims=True)

    frank_lomo = lomo_map(y, frank, months)
    frank_per_class = lomo_per_class(y, frank, months)
    print(f"    LOMO: {frank_lomo:.4f}")
    for cls, ap in frank_per_class.items():
        print(f"      {cls:15s}: {ap:.4f}")

    # Compare to best single model
    best_single_lomo = max(lomo_map(y, models[name], months) for name in model_list)
    print(f"\n    Best single model LOMO: {best_single_lomo:.4f}")
    print(f"    Frankenstein delta:     {frank_lomo - best_single_lomo:+.4f}")

    return frank, models


# ===========================================================================
# SECTION 4: Physics prior on Cormorant column
# ===========================================================================

def section4_physics_prior(y, months, train_df, best_blend_preds, models):
    print("\n" + "=" * 70)
    print("  SECTION 4: PHYSICS PRIOR ON CORMORANT COLUMN")
    print("=" * 70)

    if best_blend_preds is None:
        # Fall back to e175_best
        best_blend_preds = load_oof_safe("e175_best")
        if best_blend_preds is None:
            print("  No predictions available. Skipping.")
            return

    n = len(train_df)
    CORM_IDX = CLASSES.index("Cormorants")

    # Parse trajectories to compute physics features
    print("  Parsing trajectories for physics features...")
    speed_vals = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)

    # Compute straightness, rcs stats, speed_cv from trajectories
    straightness = np.zeros(n)
    rcs_mean_arr = np.zeros(n)
    rcs_std_arr = np.zeros(n)
    rcs_ac1_arr = np.zeros(n)
    speed_cv_arr = np.zeros(n)

    for idx in range(n):
        try:
            pts = parse_ewkb_4d(train_df.iloc[idx]["trajectory"])
            times = parse_trajectory_time(train_df.iloc[idx]["trajectory_time"])
            lons = np.array([p[0] for p in pts])
            lats = np.array([p[1] for p in pts])
            rcs = np.array([p[3] for p in pts])

            # Straightness
            if len(pts) > 1:
                from src.features import haversine
                total_dist = sum(
                    haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                    for i in range(len(pts)-1)
                )
                straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1])
                straightness[idx] = straight_dist / max(total_dist, 1e-6)
            else:
                straightness[idx] = 0.0

            # RCS stats
            rcs_mean_arr[idx] = np.mean(rcs)
            rcs_std_arr[idx] = np.std(rcs)

            # RCS autocorrelation lag-1
            if len(rcs) > 2:
                rcs_centered = rcs - np.mean(rcs)
                var = np.var(rcs)
                if var > 1e-10:
                    rcs_ac1_arr[idx] = np.mean(rcs_centered[:-1] * rcs_centered[1:]) / var
                else:
                    rcs_ac1_arr[idx] = 0.0
            else:
                rcs_ac1_arr[idx] = 0.0

            # Speed CV from segments
            if len(pts) > 2:
                dists = []
                for i in range(len(pts) - 1):
                    d = haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                    dists.append(d)
                dists = np.array(dists)
                dt = np.maximum(np.diff(times), 0.001)
                seg_speeds = dists / dt
                if np.mean(seg_speeds) > 1e-6:
                    speed_cv_arr[idx] = np.std(seg_speeds) / np.mean(seg_speeds)
                else:
                    speed_cv_arr[idx] = 0.0
            else:
                speed_cv_arr[idx] = 0.0
        except Exception:
            pass

    if idx % 500 == 0 and idx > 0:
        pass  # silent progress

    print(f"  Parsed {n} trajectories.")

    # RCS scintillation (std of linear RCS)
    rcs_linear = 10 ** (rcs_mean_arr / 10)
    # Actually we need per-point linear, let's use rcs_std as proxy for scintillation

    # Cormorant physics likelihood
    # speed 13-18, straight>0.85, alt<60, rcs>-27, scintillation>2.5, rcs_ac1>0.35, speed_cv<0.25
    print(f"\n  Computing Cormorant physics likelihood...")

    def corm_likelihood(speed, straight, alt, rcs_m, rcs_s, rcs_ac1, spd_cv):
        """Soft physics likelihood for Cormorant. Returns value in [0,1]."""
        # Each feature contributes a sigmoid-like score
        scores = []

        # Speed: peak at 13-18 m/s
        speed_score = np.exp(-0.5 * ((speed - 15.5) / 3.0) ** 2)
        scores.append(speed_score)

        # Straightness: high (>0.85)
        straight_score = 1.0 / (1.0 + np.exp(-10 * (straight - 0.80)))
        scores.append(straight_score)

        # Altitude: low (<60m)
        alt_score = 1.0 / (1.0 + np.exp(0.1 * (alt - 60)))
        scores.append(alt_score)

        # RCS: larger than typical birds (> -27 dB)
        rcs_score = 1.0 / (1.0 + np.exp(-0.5 * (rcs_m - (-27))))
        scores.append(rcs_score)

        # RCS std (scintillation proxy): > 2.5 dB
        scint_score = 1.0 / (1.0 + np.exp(-1.0 * (rcs_s - 2.0)))
        scores.append(scint_score)

        # RCS autocorrelation: > 0.35
        ac1_score = 1.0 / (1.0 + np.exp(-10 * (rcs_ac1 - 0.30)))
        scores.append(ac1_score)

        # Speed CV: low (< 0.25)
        cv_score = 1.0 / (1.0 + np.exp(10 * (spd_cv - 0.30)))
        scores.append(cv_score)

        return np.prod(scores, axis=0)

    phys_lik = corm_likelihood(
        speed_vals, straightness, alt_mid, rcs_mean_arr, rcs_std_arr, rcs_ac1_arr, speed_cv_arr
    )

    # Check distribution of likelihood
    print(f"    Physics likelihood stats:")
    print(f"      mean={phys_lik.mean():.4f}, median={np.median(phys_lik):.4f}, "
          f"max={phys_lik.max():.4f}, >0.1: {(phys_lik>0.1).sum()}")
    corm_mask = y == CORM_IDX
    print(f"      Cormorant mean lik: {phys_lik[corm_mask].mean():.4f}")
    print(f"      Non-Cormorant mean: {phys_lik[~corm_mask].mean():.4f}")

    # Apply as multiplicative boost to Cormorant column
    base_lomo = lomo_map(y, best_blend_preds, months)
    base_per_class = lomo_per_class(y, best_blend_preds, months)
    print(f"\n  Baseline LOMO: {base_lomo:.4f}")
    print(f"  Baseline Cormorant AP: {base_per_class['Cormorants']:.4f}")

    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"\n  {'Gamma':>7}  {'LOMO':>7}  {'dLOMO':>7}  {'Corm_AP':>8}  {'dCorm':>7}")
    for gamma in gammas:
        boosted = best_blend_preds.copy()
        boost = 1.0 + gamma * phys_lik
        boosted[:, CORM_IDX] *= boost
        # Renormalize
        boosted = np.clip(boosted, 1e-8, None)
        boosted /= boosted.sum(axis=1, keepdims=True)

        new_lomo = lomo_map(y, boosted, months)
        new_per_class = lomo_per_class(y, boosted, months)
        d_lomo = new_lomo - base_lomo
        d_corm = new_per_class["Cormorants"] - base_per_class["Cormorants"]
        print(f"  {gamma:7.1f}  {new_lomo:.4f}  {d_lomo:+.4f}  {new_per_class['Cormorants']:.4f}  {d_corm:+.4f}")

    # Also try Waders
    WADER_IDX = CLASSES.index("Waders")
    print(f"\n  Wader physics boost (high alt variance, continuous flap, speed>10):")

    def wader_likelihood(speed, straight, alt, alt_range, rcs_m):
        scores = []
        # Speed: moderate-fast (10-20 m/s)
        speed_score = np.exp(-0.5 * ((speed - 15) / 5.0) ** 2)
        scores.append(speed_score)
        # High altitude range
        alt_range_score = 1.0 / (1.0 + np.exp(-0.05 * (alt_range - 30)))
        scores.append(alt_range_score)
        # Straightness: moderate-high
        straight_score = 1.0 / (1.0 + np.exp(-8 * (straight - 0.70)))
        scores.append(straight_score)
        # Small RCS (< -25 dB)
        rcs_score = 1.0 / (1.0 + np.exp(0.5 * (rcs_m - (-25))))
        scores.append(rcs_score)
        return np.prod(scores, axis=0)

    alt_range = max_z - min_z
    wader_lik = wader_likelihood(speed_vals, straightness, alt_mid, alt_range, rcs_mean_arr)
    wader_mask = y == WADER_IDX
    print(f"    Wader mean lik: {wader_lik[wader_mask].mean():.4f}")
    print(f"    Non-Wader mean: {wader_lik[~wader_mask].mean():.4f}")

    print(f"\n  {'Gamma':>7}  {'LOMO':>7}  {'dLOMO':>7}  {'Wader_AP':>8}  {'dWader':>7}")
    for gamma in gammas:
        boosted = best_blend_preds.copy()
        boost = 1.0 + gamma * wader_lik
        boosted[:, WADER_IDX] *= boost
        boosted = np.clip(boosted, 1e-8, None)
        boosted /= boosted.sum(axis=1, keepdims=True)

        new_lomo = lomo_map(y, boosted, months)
        new_per_class = lomo_per_class(y, boosted, months)
        d_lomo = new_lomo - base_lomo
        d_wader = new_per_class["Waders"] - base_per_class["Waders"]
        print(f"  {gamma:7.1f}  {new_lomo:.4f}  {d_lomo:+.4f}  {new_per_class['Waders']:.4f}  {d_wader:+.4f}")


# ===========================================================================
# SECTION 5: Correlation analysis (diversity)
# ===========================================================================

def section5_correlation(y, months, n_train):
    print("\n" + "=" * 70)
    print("  SECTION 5: CORRELATION ANALYSIS (model diversity)")
    print("=" * 70)

    model_names = [
        "e175_best", "e175_ranker", "e175_cb", "e183_tabpfn",
        "e182_cnn_v3", "e175_lgb", "e175_xgb", "e175_dro",
        "e176_iso", "e180_spatial", "e180_rcs_linear",
        "e180_cnn",
    ]

    models = {}
    for name in model_names:
        oof = load_oof_safe(name)
        if oof is not None and oof.shape[0] == n_train:
            models[name] = oof

    model_list = list(models.keys())
    n_models = len(model_list)

    if n_models < 2:
        print("  Not enough models. Skipping.")
        return

    # Per-class Spearman correlation
    for c in range(N_CLASSES):
        print(f"\n  {CLASSES[c]} (class {c}):")
        # Compute pairwise correlation
        corrs = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i + 1, n_models):
                rho, _ = spearmanr(models[model_list[i]][:, c], models[model_list[j]][:, c])
                corrs[i, j] = rho
                corrs[j, i] = rho

        # Find most diverse pairs
        pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pairs.append((model_list[i], model_list[j], corrs[i, j]))
        pairs.sort(key=lambda x: x[2])

        print(f"    Most diverse pairs (lowest Spearman rho):")
        for name_a, name_b, rho in pairs[:5]:
            print(f"      {name_a:25s} vs {name_b:25s}  rho={rho:.4f}")

    # Overall diversity summary
    print(f"\n  OVERALL DIVERSITY (mean Spearman across all classes):")
    overall_corrs = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            rhos = []
            for c in range(N_CLASSES):
                rho, _ = spearmanr(models[model_list[i]][:, c], models[model_list[j]][:, c])
                rhos.append(rho)
            mean_rho = np.mean(rhos)
            overall_corrs[(model_list[i], model_list[j])] = mean_rho

    sorted_pairs = sorted(overall_corrs.items(), key=lambda x: x[1])
    print(f"\n  TOP 10 MOST DIVERSE PAIRS:")
    for (a, b), rho in sorted_pairs[:10]:
        lomo_a = lomo_map(y, models[a], months)
        lomo_b = lomo_map(y, models[b], months)
        print(f"    {a:25s} ({lomo_a:.4f}) vs {b:25s} ({lomo_b:.4f})  mean_rho={rho:.4f}")

    print(f"\n  TOP 10 MOST CORRELATED PAIRS:")
    for (a, b), rho in sorted_pairs[-10:]:
        print(f"    {a:25s} vs {b:25s}  mean_rho={rho:.4f}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  E184: EXHAUSTIVE BLEND & CALIBRATION VALIDATION")
    print("  No model training — uses existing predictions only")
    print("=" * 70)

    train_df, y, months = load_data()
    n_train = len(y)
    print(f"\n  Train samples: {n_train}")
    print(f"  Classes: {N_CLASSES}")
    print(f"  Months: {sorted(set(months))}")

    # Section 5 first (fast, informative)
    section5_correlation(y, months, n_train)

    # Section 1: Exhaustive blend
    best_blend_preds, models = section1_exhaustive_blend(y, months, n_train)

    # Section 2: Dirichlet calibration
    section2_dirichlet_calibration(y, months)

    # Section 3: Per-class blend
    frank_preds, models2 = section3_per_class_blend(y, months, n_train)

    # Section 4: Physics prior — use Frankenstein blend (probability-based) not rank blend
    # Rank-power blends from Section 1 are NOT valid probabilities, so physics prior fails on them
    best_preds_for_physics = frank_preds if frank_preds is not None else (load_oof_safe("e176_iso_gmm_knn") if load_oof_safe("e176_iso_gmm_knn") is not None else best_blend_preds)
    section4_physics_prior(y, months, train_df, best_preds_for_physics, models or models2)

    print("\n" + "=" * 70)
    print("  DONE — All sections complete")
    print("=" * 70)
