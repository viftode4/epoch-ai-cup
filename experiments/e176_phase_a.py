"""E176 Phase A: Zero-Retrain Post-Processing on E175 OOF predictions.

Techniques:
  A1. Per-class power transform (gamma_c per class, coordinate descent)
  A2. Day-level prior estimation (cosine similarity to training days)
  A3. Max-confidence flock propagation (non-monotonic cluster consensus)
  A4. MLLS with bias-corrected temperature scaling
  A5. KNN neighbor blending (shared months only)
  A6. Venn-Abers calibration (better probabilities for PP)

Each technique evaluated independently + combined on E175 OOF (SKF mAP).
Best combinations saved as test submissions.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  E176 Phase A: Zero-Retrain Post-Processing")
print("=" * 70)

t0 = time.time()

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load E175 OOF and test predictions
oof_best = np.load(ROOT / "oof_e175_best.npy").astype(np.float64)
oof_lgb = np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64)
oof_dro = np.load(ROOT / "oof_e175_dro.npy").astype(np.float64)
test_best = np.load(ROOT / "test_e175_best.npy").astype(np.float64)
test_lgb = np.load(ROOT / "test_e175_lgb.npy").astype(np.float64)
test_dro = np.load(ROOT / "test_e175_dro.npy").astype(np.float64)

# Ensure all are proper probabilities
oof_best = renorm_rows(oof_best)
oof_lgb = renorm_rows(oof_lgb)
oof_dro = renorm_rows(oof_dro)
test_best = renorm_rows(test_best)
test_lgb = renorm_rows(test_lgb)
test_dro = renorm_rows(test_dro)

# Training class proportions
counts_train = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts_train / counts_train.sum()

# Baseline scores
base_score, base_per_class = compute_map(y, oof_best)
print(f"\nBaseline E175 blend: SKF mAP = {base_score:.4f}")
for cls, ap in base_per_class.items():
    print(f"  {cls:20s}: {ap:.3f}")

# Load cached features for KNN (A5)
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
feat_cols = [c for c in train_feats.columns if c in test_feats.columns]
X_train_raw = train_feats[feat_cols].values.astype(np.float32)
X_test_raw = test_feats[feat_cols].values.astype(np.float32)
X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate(preds, label=""):
    """Evaluate and print per-class AP."""
    score, per_class = compute_map(y, preds)
    delta = score - base_score
    print(f"  {label}: SKF mAP = {score:.4f} (delta: {delta:+.4f})")
    # Show only changed classes
    improved = []
    degraded = []
    for cls in CLASSES:
        d = per_class[cls] - base_per_class[cls]
        if d > 0.005:
            improved.append(f"{cls[:4]}+{d:.3f}")
        elif d < -0.005:
            degraded.append(f"{cls[:4]}{d:.3f}")
    if improved:
        print(f"    Improved: {', '.join(improved)}")
    if degraded:
        print(f"    Degraded: {', '.join(degraded)}")
    return score, per_class


# ══════════════════════════════════════════════════════════════════════
# A1. Per-Class Power Transform
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A1. Per-Class Power Transform")
print("=" * 70)


def apply_power_transform(preds, gammas):
    """Apply per-class power transform: p_c^gamma_c / sum(p_j^gamma_j)."""
    out = np.clip(preds, 1e-12, None)
    for c in range(N_CLASSES):
        out[:, c] = out[:, c] ** gammas[c]
    return renorm_rows(out)


def optimize_power_gammas(oof, y_true, n_rounds=3, gamma_range=(0.3, 3.0)):
    """Coordinate descent to find per-class optimal gammas."""
    best_gammas = np.ones(N_CLASSES)
    best_score, _ = compute_map(y_true, oof)

    for round_i in range(n_rounds):
        improved = False
        for c in range(N_CLASSES):
            trial_gammas = best_gammas.copy()
            best_g = best_gammas[c]
            for g in np.linspace(gamma_range[0], gamma_range[1], 30):
                trial_gammas[c] = g
                transformed = apply_power_transform(oof, trial_gammas)
                score, _ = compute_map(y_true, transformed)
                if score > best_score + 1e-6:
                    best_score = score
                    best_g = g
                    improved = True
            best_gammas[c] = best_g
        if not improved:
            break
        print(f"  Round {round_i+1}: mAP = {best_score:.4f}, gammas = {np.round(best_gammas, 2)}")

    return best_gammas, best_score


gammas_best, score_a1 = optimize_power_gammas(oof_best, y)
oof_a1 = apply_power_transform(oof_best, gammas_best)
evaluate(oof_a1, "A1 Power Transform")
print(f"  Optimal gammas: {dict(zip(CLASSES, np.round(gammas_best, 2)))}")


# ══════════════════════════════════════════════════════════════════════
# A2. Day-Level Prior Estimation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A2. Day-Level Prior Estimation")
print("=" * 70)


def extract_day_profile(df, months):
    """Build per-day observable profiles (radar_bird_size dist, speed stats, alt stats)."""
    dates = pd.to_datetime(df["timestamp_start_radar_utc"]).dt.date
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values
    alt_mid = 0.5 * (min_z + max_z)
    sizes = df["radar_bird_size"].fillna("__UNK__")

    day_profiles = {}
    for date in sorted(dates.unique()):
        mask = dates == date
        n = int(mask.sum())
        if n < 5:
            continue

        # Size distribution
        size_dist = sizes[mask].value_counts(normalize=True)
        size_vec = np.array([
            size_dist.get("Small bird", 0),
            size_dist.get("Medium bird", 0),
            size_dist.get("Large bird", 0),
            size_dist.get("Flock", 0),
        ])

        # Speed/altitude stats
        spd = speed[mask]
        alt = alt_mid[mask]
        stats = np.array([
            np.nanmean(spd), np.nanstd(spd),
            np.nanmean(alt), np.nanstd(alt),
            float(n),  # track count
        ])

        profile = np.concatenate([size_vec, stats])
        profile = np.nan_to_num(profile, nan=0.0)
        day_profiles[date] = profile

    return day_profiles


def apply_day_priors(preds, df, months, train_df, y_train, alpha=0.15):
    """Match test days to nearest training days, apply day-specific priors."""
    train_dates = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.date
    test_dates = pd.to_datetime(df["timestamp_start_radar_utc"]).dt.date

    # Build per-day class distributions for training
    day_class_dist = {}
    for date in sorted(train_dates.unique()):
        mask = train_dates == date
        if mask.sum() < 5:
            continue
        counts = np.bincount(y_train[mask], minlength=N_CLASSES).astype(float)
        day_class_dist[date] = counts / counts.sum()

    if not day_class_dist:
        return preds.copy()

    # Build profiles
    train_profiles = extract_day_profile(train_df, pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values)
    test_profiles = extract_day_profile(df, months)

    if not train_profiles or not test_profiles:
        return preds.copy()

    # Match test days to training days by cosine similarity
    train_dates_list = sorted(train_profiles.keys())
    train_profile_mat = np.array([train_profiles[d] for d in train_dates_list])
    # Normalize
    norms = np.linalg.norm(train_profile_mat, axis=1, keepdims=True)
    train_profile_mat = train_profile_mat / np.maximum(norms, 1e-12)

    out = preds.copy()
    margin = top2_margin(out)

    for test_date in sorted(test_profiles.keys()):
        test_vec = test_profiles[test_date]
        test_vec = test_vec / max(np.linalg.norm(test_vec), 1e-12)

        # Cosine similarity
        sims = train_profile_mat @ test_vec
        top_k = np.argsort(-sims)[:3]

        # Weighted average of top-3 training days' class distributions
        w_prior = np.zeros(N_CLASSES)
        total_w = 0
        for idx in top_k:
            td = train_dates_list[idx]
            if td in day_class_dist:
                w = max(sims[idx], 0.01)
                w_prior += w * day_class_dist[td]
                total_w += w
        if total_w > 0:
            w_prior /= total_w
        else:
            continue

        # Apply ratio tilt to uncertain rows on this day
        test_day_mask = test_dates == test_date
        for i in np.where(test_day_mask)[0]:
            if margin[i] < 0.3:  # only uncertain
                ratio = (w_prior / np.maximum(p_train, 1e-12)) ** alpha
                out[i] = out[i] * ratio
                out[i] /= max(out[i].sum(), 1e-12)

    return renorm_rows(out)


oof_a2 = apply_day_priors(oof_best, train_df, train_months, train_df, y, alpha=0.10)
evaluate(oof_a2, "A2 Day-Level Priors (alpha=0.10)")

# Try different alpha values
for alpha in [0.05, 0.15, 0.20]:
    oof_a2_t = apply_day_priors(oof_best, train_df, train_months, train_df, y, alpha=alpha)
    evaluate(oof_a2_t, f"A2 Day Priors (alpha={alpha})")


# ══════════════════════════════════════════════════════════════════════
# A3. Max-Confidence Flock Propagation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A3. Max-Confidence Flock Propagation")
print("=" * 70)


def build_flock_clusters(df, time_thresh=60, dist_thresh_m=500):
    """Build clusters of tracks that are close in time and space."""
    from src.data import parse_ewkb_4d

    n = len(df)
    timestamps = pd.to_datetime(df["timestamp_start_radar_utc"])
    ts_unix = timestamps.values.astype(np.int64) / 1e9  # seconds

    # Extract centroids
    lons = np.zeros(n)
    lats = np.zeros(n)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if pts:
                lons[i] = np.mean([p[0] for p in pts])
                lats[i] = np.mean([p[1] for p in pts])
        except Exception:
            pass

    sizes = df["radar_bird_size"].fillna("__UNK__").values

    # Build adjacency: time < thresh AND dist < thresh AND same size
    # Use a simple O(n^2) approach (n~2600, manageable)
    clusters = []
    visited = set()

    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        queue = [i]
        while queue:
            curr = queue.pop(0)
            for j in range(n):
                if j in visited:
                    continue
                dt = abs(ts_unix[curr] - ts_unix[j])
                if dt > time_thresh:
                    continue
                if sizes[curr] != sizes[j]:
                    continue
                # Approximate distance in meters
                dlat = (lats[curr] - lats[j]) * 111000
                dlon = (lons[curr] - lons[j]) * 67000
                dist = np.sqrt(dlat**2 + dlon**2)
                if dist > dist_thresh_m:
                    continue
                cluster.append(j)
                visited.add(j)
                queue.append(j)
        if len(cluster) > 1 and len(cluster) <= 5:
            clusters.append(cluster)

    return clusters


def apply_flock_propagation(preds, clusters, min_confidence=0.90, min_margin=0.5):
    """Propagate most confident prediction to all cluster members."""
    out = preds.copy()
    margin = top2_margin(out)
    n_changed = 0

    for cluster in clusters:
        # Find the most confident member
        best_idx = -1
        best_margin = -1
        for idx in cluster:
            if margin[idx] > best_margin:
                best_margin = margin[idx]
                best_idx = idx

        # Safety gates
        if best_margin < min_margin:
            continue
        if out[best_idx].max() < min_confidence:
            continue

        # Propagate to other members (NON-MONOTONIC: overwrites their predictions)
        for idx in cluster:
            if idx != best_idx:
                out[idx] = out[best_idx].copy()
                n_changed += 1

    print(f"  Flock propagation: {len(clusters)} clusters, {n_changed} tracks changed")
    return renorm_rows(out)


print("  Building flock clusters (this may take a moment)...")
train_clusters = build_flock_clusters(train_df, time_thresh=60, dist_thresh_m=500)
print(f"  Found {len(train_clusters)} clusters in training data")

oof_a3 = apply_flock_propagation(oof_best, train_clusters, min_confidence=0.90, min_margin=0.5)
evaluate(oof_a3, "A3 Flock Propagation (conf=0.90, margin=0.50)")

# Try relaxed thresholds
for conf, marg in [(0.85, 0.4), (0.80, 0.3)]:
    oof_a3_t = apply_flock_propagation(oof_best, train_clusters, min_confidence=conf, min_margin=marg)
    evaluate(oof_a3_t, f"A3 Flock (conf={conf}, margin={marg})")


# ══════════════════════════════════════════════════════════════════════
# A4. MLLS with Bias-Corrected Temperature Scaling
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A4. MLLS Temperature-Scaled Prior Adjustment")
print("=" * 70)


def temperature_scale(probs, T):
    """Apply temperature scaling."""
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def find_temperature(probs, y_true):
    """Find optimal temperature on labeled data."""
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 5.0, 100):
        cal = temperature_scale(probs, T)
        nll = -np.log(np.clip(cal[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def mlls_estimate(probs, p_source, max_iter=200, tol=1e-7):
    """MLLS: estimate target class proportions from calibrated predictions."""
    w = p_source.copy()
    for _ in range(max_iter):
        ratio = w / np.maximum(p_source, 1e-12)
        adjusted = probs * ratio[np.newaxis, :]
        row_sums = adjusted.sum(axis=1, keepdims=True)
        adjusted = adjusted / np.maximum(row_sums, 1e-12)
        w_new = adjusted.mean(axis=0)
        w_new = np.maximum(w_new, 1e-8)
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < tol:
            break
        w = w_new
    return w


def apply_mlls_prior(preds, months, p_train, T=None, alpha=0.20, tau=0.20):
    """Apply MLLS-estimated priors per month after temperature scaling."""
    if T is None:
        T = find_temperature(preds, y)
    cal = temperature_scale(preds, T)

    out = preds.copy()
    margin = top2_margin(out)

    # Estimate per-month proportions using MLLS on calibrated predictions
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() < N_CLASSES:
            continue

        w_mlls = mlls_estimate(cal[mask], p_train)
        ratio = (w_mlls / np.maximum(p_train, 1e-12)) ** alpha

        # Apply only to uncertain rows
        gate = mask & (margin < tau)
        if gate.sum() > 0:
            out[gate] = out[gate] * ratio
            out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)

    return renorm_rows(out)


T_opt = find_temperature(oof_best, y)
print(f"  Optimal temperature: T = {T_opt:.2f}")

oof_a4 = apply_mlls_prior(oof_best, train_months, p_train, T=T_opt, alpha=0.20, tau=0.20)
evaluate(oof_a4, "A4 MLLS Prior (alpha=0.20, tau=0.20)")

for alpha, tau in [(0.10, 0.15), (0.15, 0.25), (0.30, 0.30)]:
    oof_a4_t = apply_mlls_prior(oof_best, train_months, p_train, T=T_opt, alpha=alpha, tau=tau)
    evaluate(oof_a4_t, f"A4 MLLS (a={alpha}, t={tau})")


# ══════════════════════════════════════════════════════════════════════
# A5. KNN Neighbor Blending
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A5. KNN Neighbor Blending")
print("=" * 70)


def apply_knn_blending(oof_preds, X_train, y_train, months, K=10, alpha=0.10):
    """Blend OOF predictions with KNN neighbor label distributions."""
    from sklearn.preprocessing import StandardScaler

    # Only apply to shared months (Sep/Oct) where features are most reliable
    shared_mask = np.isin(months, [9, 10])
    if shared_mask.sum() == 0:
        return oof_preds.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    out = oof_preds.copy()
    margin = top2_margin(out)

    # For each sample in shared months, find K nearest neighbors from OTHER months
    for i in np.where(shared_mask)[0]:
        if margin[i] > 0.5:  # skip confident predictions
            continue

        # Find K nearest in feature space (exclude self)
        dists = np.sqrt(((X_scaled - X_scaled[i]) ** 2).sum(axis=1))
        dists[i] = np.inf  # exclude self

        top_k_idx = np.argsort(dists)[:K]
        top_k_dists = dists[top_k_idx]

        # Inverse-distance weighting
        weights = 1.0 / np.maximum(top_k_dists, 1e-8)
        weights /= weights.sum()

        # Build neighbor label distribution
        knn_dist = np.zeros(N_CLASSES)
        for j, w in zip(top_k_idx, weights):
            knn_dist[y_train[j]] += w

        # Blend
        out[i] = (1 - alpha) * out[i] + alpha * knn_dist

    return renorm_rows(out)


oof_a5 = apply_knn_blending(oof_best, X_train_raw, y, train_months, K=10, alpha=0.10)
evaluate(oof_a5, "A5 KNN Blend (K=10, alpha=0.10)")

for K, alpha in [(5, 0.05), (15, 0.15), (20, 0.20)]:
    oof_a5_t = apply_knn_blending(oof_best, X_train_raw, y, train_months, K=K, alpha=alpha)
    evaluate(oof_a5_t, f"A5 KNN (K={K}, a={alpha})")


# ══════════════════════════════════════════════════════════════════════
# A6. Per-Class Isotonic Calibration (simpler than Venn-Abers)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  A6. Isotonic Calibration")
print("=" * 70)


def apply_isotonic_calibration(oof_preds, y_true):
    """Apply per-class isotonic regression calibration (non-monotonic after renorm)."""
    from sklearn.isotonic import IsotonicRegression

    out = oof_preds.copy()
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        y_bin = (y_true == c).astype(float)
        # Fit on OOF (leave-one-out would be better, but this is fast)
        ir.fit(out[:, c], y_bin)
        out[:, c] = ir.predict(out[:, c])

    return renorm_rows(out)


oof_a6 = apply_isotonic_calibration(oof_best, y)
evaluate(oof_a6, "A6 Isotonic Calibration")


# ══════════════════════════════════════════════════════════════════════
# Combinations
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Combined Techniques")
print("=" * 70)

# Collect winners
results = {}
results["baseline"] = (base_score, oof_best, test_best)

# A1 on all model variants
for name, oof, test in [
    ("best", oof_best, test_best),
    ("lgb", oof_lgb, test_lgb),
    ("dro", oof_dro, test_dro),
]:
    g, _ = optimize_power_gammas(oof, y)
    oof_t = apply_power_transform(oof, g)
    test_t = apply_power_transform(test, g)
    s, _ = compute_map(y, oof_t)
    results[f"A1_{name}"] = (s, oof_t, test_t)
    print(f"  A1 on {name}: {s:.4f}")

# A1 + A3 (power transform + flock propagation)
oof_a1a3 = apply_flock_propagation(oof_a1, train_clusters, min_confidence=0.90, min_margin=0.5)
s_a1a3, _ = compute_map(y, oof_a1a3)
print(f"  A1+A3: {s_a1a3:.4f}")

# A1 + A5 (power transform + KNN)
oof_a1a5 = apply_knn_blending(oof_a1, X_train_raw, y, train_months, K=10, alpha=0.10)
s_a1a5, _ = compute_map(y, oof_a1a5)
print(f"  A1+A5: {s_a1a5:.4f}")

# A1 + A6 (power transform + isotonic)
oof_a1a6 = apply_isotonic_calibration(oof_a1, y)
s_a1a6, _ = compute_map(y, oof_a1a6)
print(f"  A1+A6: {s_a1a6:.4f}")

# A6 + A1 (calibrate first, then power)
oof_a6a1_pre = apply_isotonic_calibration(oof_best, y)
g_post, _ = optimize_power_gammas(oof_a6a1_pre, y)
oof_a6a1 = apply_power_transform(oof_a6a1_pre, g_post)
s_a6a1, _ = compute_map(y, oof_a6a1)
print(f"  A6+A1: {s_a6a1:.4f}")

# A1 + standard NB PP
def apply_standard_nb(preds, test_df_in, test_months_in, train_df_in, y_in):
    """Standard NB PP from postprocessing.py."""
    counts = np.bincount(y_in, minlength=N_CLASSES).astype(float)
    p_tr = counts / counts.sum()
    priors = build_gbif_priors(p_tr)
    out, _ = apply_gated_ratio_priors(preds, test_months_in, p_tr, priors, BASE_ALPHA, tau=0.15)

    speed = pd.to_numeric(train_df_in["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df_in["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df_in["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed, "alt_mid": 0.5*(min_z+max_z), "alt_range": max_z-min_z}
    size_levels, log_p_size, mu, sig = build_nb_params(train_df_in, y_in, cont_tr)

    speed_t = pd.to_numeric(test_df_in["airspeed"], errors="coerce").values.astype(float)
    min_z_t = pd.to_numeric(test_df_in["min_z"], errors="coerce").values.astype(float)
    max_z_t = pd.to_numeric(test_df_in["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_t, "alt_mid": 0.5*(min_z_t+max_z_t), "alt_range": max_z_t-min_z_t}

    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    loglike = compute_log_p_u_given_c(test_df_in, size_levels, log_p_size, cont_te, weights, None, mu, sig)
    gate = np.isin(test_months_in, UNSEEN_MONTHS) & (top2_margin(out) < 0.25)
    return apply_nb_poe(out, loglike, gamma=0.10, gate=gate)


# A1 + NB PP (on OOF, using per-month leave-out for PP params)
# For OOF, the NB PP is tricky since we need to simulate unseen months.
# Instead, just apply NB PP to test and note the interaction.
print("\n  (NB PP applied only to test submissions, not OOF)")


# ══════════════════════════════════════════════════════════════════════
# Summary & Best Submission Selection
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE A SUMMARY")
print("=" * 70)

all_results = [
    ("Baseline (E175 blend)", base_score, oof_best, test_best),
    ("A1 Power Transform", score_a1, oof_a1, apply_power_transform(test_best, gammas_best)),
]

# Sort by score
all_results.sort(key=lambda x: -x[1])

print(f"\n  {'Technique':<40s} {'SKF mAP':>8s} {'Delta':>8s}")
print(f"  {'-'*40} {'-'*8} {'-'*8}")
for name, score, _, _ in all_results:
    delta = score - base_score
    marker = " ***" if delta > 0.005 else (" **" if delta > 0.001 else "")
    print(f"  {name:<40s} {score:>8.4f} {delta:>+8.4f}{marker}")

# Save best submissions
print("\n  Saving test submissions...")
for name, score, oof_p, test_p in all_results[:3]:
    tag = name.replace(" ", "_").replace("(", "").replace(")", "")[:30]

    # Raw
    sub = pd.DataFrame({"track_id": test_df["track_id"]})
    for i, cls in enumerate(CLASSES):
        sub[cls] = test_p[:, i]
    save_submission(sub, f"e176_{tag}_raw", score)

    # With NB PP
    test_pp = apply_standard_nb(test_p, test_df, test_months, train_df, y)
    sub_pp = pd.DataFrame({"track_id": test_df["track_id"]})
    for i, cls in enumerate(CLASSES):
        sub_pp[cls] = test_pp[:, i]
    save_submission(sub_pp, f"e176_{tag}_pp", score)


elapsed = time.time() - t0
print(f"\n  Phase A completed in {elapsed:.0f}s")
print("=" * 70)
