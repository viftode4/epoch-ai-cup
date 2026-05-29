"""E176 Phase A: LOMO evaluation of ALL post-processing techniques.

Phase A was only evaluated on SKF CV — never on LOMO.
A technique that hurts SKF could help LOMO if it improves month generalization.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.metrics import compute_map
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent

print("=" * 90)
print("  E176 Phase A: LOMO Evaluation of ALL Post-Processing Techniques")
print("=" * 90)

t0 = time.time()

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

# Load features for KNN
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
feat_cols = [c for c in train_feats.columns]
X_train_raw = np.nan_to_num(train_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def eval_skf_lomo(oof, name=""):
    """Evaluate both SKF and per-month LOMO."""
    skf, _ = compute_map(y, oof)
    lomo_scores = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            lomo_scores[m] = s
    lomo = np.mean(list(lomo_scores.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_scores.items()))
    print(f"  {name:<45s}  SKF={skf:.4f}  LOMO={lomo:.4f}  [{month_str}]")
    return skf, lomo


# ── Baseline ──
print("\n--- Baseline ---")
eval_skf_lomo(oof_best, "E175 best (baseline)")
eval_skf_lomo(oof_lgb, "E175 lgb")


# ══════════════════════════════════════════════════════════════════════
# A1. Per-Class Power Transform
# ══════════════════════════════════════════════════════════════════════

print("\n--- A1. Per-Class Power Transform ---")


def apply_power_transform(preds, gammas):
    out = np.clip(preds, 1e-12, None)
    for c in range(N_CLASSES):
        out[:, c] = out[:, c] ** gammas[c]
    return renorm_rows(out)


def optimize_gammas(oof, y_true, idx=None):
    if idx is None:
        idx = np.arange(len(y_true))
    best_gammas = np.ones(N_CLASSES)
    best_score, _ = compute_map(y_true[idx], oof[idx])
    for _ in range(3):
        improved = False
        for c in range(N_CLASSES):
            trial = best_gammas.copy()
            best_g = best_gammas[c]
            for g in np.linspace(0.3, 3.0, 30):
                trial[c] = g
                transformed = apply_power_transform(oof, trial)
                score, _ = compute_map(y_true[idx], transformed[idx])
                if score > best_score + 1e-6:
                    best_score = score
                    best_g = g
                    improved = True
            best_gammas[c] = best_g
        if not improved:
            break
    return best_gammas


# Non-CV (fit on all, eval on all)
gammas_full = optimize_gammas(oof_best, y)
oof_a1 = apply_power_transform(oof_best, gammas_full)
eval_skf_lomo(oof_a1, "A1 Power (non-CV, fit on all)")

# Per-month gammas (fit on 3 months, eval on held-out)
print("  A1 Per-month leave-one-out:")
oof_a1_lomo = oof_best.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    gammas_m = optimize_gammas(oof_best, y, np.where(mask_train)[0])
    oof_a1_lomo[mask_held] = apply_power_transform(oof_best[mask_held], gammas_m)
eval_skf_lomo(oof_a1_lomo, "A1 Power (LOMO: fit 3 months, eval 1)")

# On lgb too
gammas_lgb = optimize_gammas(oof_lgb, y)
oof_a1_lgb = apply_power_transform(oof_lgb, gammas_lgb)
eval_skf_lomo(oof_a1_lgb, "A1 Power on LGB (non-CV)")

oof_a1_lgb_lomo = oof_lgb.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    gammas_m = optimize_gammas(oof_lgb, y, np.where(mask_train)[0])
    oof_a1_lgb_lomo[mask_held] = apply_power_transform(oof_lgb[mask_held], gammas_m)
eval_skf_lomo(oof_a1_lgb_lomo, "A1 Power on LGB (LOMO)")


# ══════════════════════════════════════════════════════════════════════
# A2. Day-Level Prior Estimation
# ══════════════════════════════════════════════════════════════════════

print("\n--- A2. Day-Level Prior Estimation ---")


def apply_day_priors(preds, df, months, y_train_full, alpha=0.10):
    train_dates = pd.to_datetime(df["timestamp_start_radar_utc"]).dt.date
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values
    alt_mid = 0.5 * (min_z + max_z)
    sizes = df["radar_bird_size"].fillna("__UNK__")

    # Per-day class distributions
    day_class_dist = {}
    for date in sorted(train_dates.unique()):
        mask = train_dates == date
        if mask.sum() < 5:
            continue
        c = np.bincount(y_train_full[mask], minlength=N_CLASSES).astype(float)
        day_class_dist[date] = c / c.sum()

    # Per-day profiles
    day_profiles = {}
    for date in sorted(train_dates.unique()):
        mask = train_dates == date
        n = int(mask.sum())
        if n < 5:
            continue
        size_dist = sizes[mask].value_counts(normalize=True)
        size_vec = np.array([
            size_dist.get("Small bird", 0), size_dist.get("Medium bird", 0),
            size_dist.get("Large bird", 0), size_dist.get("Flock", 0),
        ])
        stats = np.array([
            np.nanmean(speed[mask]), np.nanstd(speed[mask]),
            np.nanmean(alt_mid[mask]), np.nanstd(alt_mid[mask]), float(n),
        ])
        day_profiles[date] = np.nan_to_num(np.concatenate([size_vec, stats]), nan=0.0)

    if not day_profiles:
        return preds.copy()

    train_dates_list = sorted(day_profiles.keys())
    train_profile_mat = np.array([day_profiles[d] for d in train_dates_list])
    norms = np.linalg.norm(train_profile_mat, axis=1, keepdims=True)
    train_profile_mat = train_profile_mat / np.maximum(norms, 1e-12)

    out = preds.copy()
    margin = top2_margin(out)

    for test_date in sorted(day_profiles.keys()):
        test_vec = day_profiles[test_date]
        test_vec = test_vec / max(np.linalg.norm(test_vec), 1e-12)
        sims = train_profile_mat @ test_vec
        top_k = np.argsort(-sims)[:3]

        w_prior = np.zeros(N_CLASSES)
        total_w = 0
        for idx in top_k:
            td = train_dates_list[idx]
            if td != test_date and td in day_class_dist:
                w = max(sims[idx], 0.01)
                w_prior += w * day_class_dist[td]
                total_w += w
        if total_w > 0:
            w_prior /= total_w
        else:
            continue

        test_day_mask = (pd.to_datetime(df["timestamp_start_radar_utc"]).dt.date == test_date)
        for i in np.where(test_day_mask)[0]:
            if margin[i] < 0.3:
                ratio = (w_prior / np.maximum(p_train, 1e-12)) ** alpha
                out[i] = out[i] * ratio
                out[i] /= max(out[i].sum(), 1e-12)

    return renorm_rows(out)


for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
    oof_a2 = apply_day_priors(oof_best, train_df, train_months, y, alpha=alpha)
    eval_skf_lomo(oof_a2, f"A2 Day Priors (alpha={alpha})")


# ══════════════════════════════════════════════════════════════════════
# A3. Max-Confidence Flock Propagation
# ══════════════════════════════════════════════════════════════════════

print("\n--- A3. Max-Confidence Flock Propagation ---")


def build_flock_clusters(df, time_thresh=60, dist_thresh_m=500):
    n = len(df)
    ts_unix = pd.to_datetime(df["timestamp_start_radar_utc"]).values.astype(np.int64) / 1e9
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
                if abs(ts_unix[curr] - ts_unix[j]) > time_thresh:
                    continue
                if sizes[curr] != sizes[j]:
                    continue
                dlat = (lats[curr] - lats[j]) * 111000
                dlon = (lons[curr] - lons[j]) * 67000
                if np.sqrt(dlat**2 + dlon**2) > dist_thresh_m:
                    continue
                cluster.append(j)
                visited.add(j)
                queue.append(j)
        if 1 < len(cluster) <= 5:
            clusters.append(cluster)
    return clusters


print("  Building flock clusters...")
train_clusters = build_flock_clusters(train_df)
print(f"  Found {len(train_clusters)} clusters")

for conf, marg in [(0.95, 0.6), (0.90, 0.5), (0.85, 0.4), (0.80, 0.3)]:
    out = oof_best.copy()
    margin = top2_margin(out)
    n_changed = 0
    for cluster in train_clusters:
        best_idx = max(cluster, key=lambda i: margin[i])
        if margin[best_idx] < marg or out[best_idx].max() < conf:
            continue
        for idx in cluster:
            if idx != best_idx:
                out[idx] = out[best_idx].copy()
                n_changed += 1
    out = renorm_rows(out)
    eval_skf_lomo(out, f"A3 Flock Prop (conf={conf}, marg={marg}, n={n_changed})")


# ══════════════════════════════════════════════════════════════════════
# A4. MLLS Temperature-Scaled Prior
# ══════════════════════════════════════════════════════════════════════

print("\n--- A4. MLLS Temperature-Scaled Prior ---")


def temperature_scale(probs, T):
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def find_temperature(probs, y_true):
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 5.0, 100):
        cal = temperature_scale(probs, T)
        nll = -np.log(np.clip(cal[np.arange(len(y_true)), y_true], 1e-8, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def mlls_estimate(probs, p_source, max_iter=200, tol=1e-7):
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


T_opt = find_temperature(oof_best, y)
print(f"  Temperature: T={T_opt:.2f}")

for alpha, tau in [(0.05, 0.15), (0.10, 0.15), (0.10, 0.25), (0.15, 0.25), (0.20, 0.20), (0.30, 0.30)]:
    cal = temperature_scale(oof_best, T_opt)
    out = oof_best.copy()
    margin = top2_margin(out)
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() < N_CLASSES:
            continue
        w_mlls = mlls_estimate(cal[mask], p_train)
        ratio = (w_mlls / np.maximum(p_train, 1e-12)) ** alpha
        gate = mask & (margin < tau)
        if gate.sum() > 0:
            out[gate] = out[gate] * ratio
            out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    out = renorm_rows(out)
    eval_skf_lomo(out, f"A4 MLLS (a={alpha}, t={tau})")


# ══════════════════════════════════════════════════════════════════════
# A5. KNN Neighbor Blending
# ══════════════════════════════════════════════════════════════════════

print("\n--- A5. KNN Neighbor Blending ---")


def apply_knn(oof_preds, X_data, y_data, K=15, alpha=0.15, max_margin=0.5):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    out = oof_preds.copy()
    margin = top2_margin(out)
    for i in range(len(out)):
        if margin[i] > max_margin:
            continue
        dists = np.sqrt(((X_scaled - X_scaled[i]) ** 2).sum(axis=1))
        dists[i] = np.inf
        top_k = np.argsort(dists)[:K]
        weights = 1.0 / np.maximum(dists[top_k], 1e-8)
        weights /= weights.sum()
        knn_dist = np.zeros(N_CLASSES)
        for j, w in zip(top_k, weights):
            knn_dist[y_data[j]] += w
        out[i] = (1 - alpha) * out[i] + alpha * knn_dist
    return renorm_rows(out)


for K, alpha in [(5, 0.05), (10, 0.10), (15, 0.15), (20, 0.20), (30, 0.25)]:
    oof_a5 = apply_knn(oof_best, X_train_raw, y, K=K, alpha=alpha)
    eval_skf_lomo(oof_a5, f"A5 KNN (K={K}, a={alpha})")


# ══════════════════════════════════════════════════════════════════════
# A6. Isotonic Calibration (non-CV and per-month CV)
# ══════════════════════════════════════════════════════════════════════

print("\n--- A6. Isotonic Calibration ---")

# Non-CV
oof_a6 = oof_best.copy()
for c in range(N_CLASSES):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_best[:, c], (y == c).astype(float))
    oof_a6[:, c] = ir.predict(oof_best[:, c])
oof_a6 = renorm_rows(oof_a6)
eval_skf_lomo(oof_a6, "A6 Isotonic (non-CV, fit on all)")

# Per-month leave-one-out isotonic
oof_a6_lomo = oof_best.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof_best[mask_train, c], (y[mask_train] == c).astype(float))
        oof_a6_lomo[mask_held, c] = ir.predict(oof_best[mask_held, c])
oof_a6_lomo = renorm_rows(oof_a6_lomo)
eval_skf_lomo(oof_a6_lomo, "A6 Isotonic (LOMO: fit 3 months, eval 1)")

# SKF-CV isotonic (for comparison)
oof_a6_cv = oof_best.copy()
skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in skf_cv.split(oof_best, y):
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof_best[tr_idx, c], (y[tr_idx] == c).astype(float))
        oof_a6_cv[va_idx, c] = ir.predict(oof_best[va_idx, c])
oof_a6_cv = renorm_rows(oof_a6_cv)
eval_skf_lomo(oof_a6_cv, "A6 Isotonic (5-fold SKF CV)")


# ══════════════════════════════════════════════════════════════════════
# Standard NB PP (reference)
# ══════════════════════════════════════════════════════════════════════

print("\n--- Standard NB PP (reference) ---")


def apply_nb_pp(preds, df, months, train_df_in, y_in, gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    counts_in = np.bincount(y_in, minlength=N_CLASSES).astype(float)
    p_tr = counts_in / counts_in.sum()
    priors = build_gbif_priors(p_tr)
    out, _ = apply_gated_ratio_priors(preds, months, p_tr, priors, BASE_ALPHA, tau=tau_prior)

    speed = pd.to_numeric(train_df_in["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df_in["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df_in["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed, "alt_mid": 0.5*(min_z+max_z), "alt_range": max_z-min_z}
    sl, lps, mu, sig = build_nb_params(train_df_in, y_in, cont_tr)

    speed_t = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z_t = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z_t = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_t, "alt_mid": 0.5*(min_z_t+max_z_t), "alt_range": max_z_t-min_z_t}
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(df, sl, lps, cont_te, w, None, mu, sig)
    gate = np.isin(months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)


# NB PP uses UNSEEN_MONTHS gating, so it only fires when month is in {2,5,12}
# On training data (months 1,4,9,10), the GBIF priors still fire for UNSEEN months
# To properly test NB PP on LOMO, we remap months
for gamma in [0.05, 0.10, 0.15, 0.20]:
    # Standard (no remapping — PP only fires on UNSEEN months, which aren't in train)
    oof_nb = apply_nb_pp(oof_best, train_df, train_months, train_df, y, gamma=gamma)
    eval_skf_lomo(oof_nb, f"NB PP gamma={gamma} (standard)")


# ══════════════════════════════════════════════════════════════════════
# COMBINATIONS
# ══════════════════════════════════════════════════════════════════════

print("\n--- Combinations ---")

# A1 LOMO + A5 KNN
oof_a1_lomo_a5 = apply_knn(oof_a1_lomo, X_train_raw, y, K=15, alpha=0.15)
eval_skf_lomo(oof_a1_lomo_a5, "A1(LOMO) + A5(K=15,a=0.15)")

# A6 LOMO + A5 KNN
oof_a6_lomo_a5 = apply_knn(oof_a6_lomo, X_train_raw, y, K=15, alpha=0.15)
eval_skf_lomo(oof_a6_lomo_a5, "A6(LOMO) + A5(K=15,a=0.15)")

# A5 KNN + A1 LOMO
oof_a5_base = apply_knn(oof_best, X_train_raw, y, K=15, alpha=0.15)
oof_a5_a1 = oof_a5_base.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    gammas_m = optimize_gammas(oof_a5_base, y, np.where(mask_train)[0])
    oof_a5_a1[mask_held] = apply_power_transform(oof_a5_base[mask_held], gammas_m)
eval_skf_lomo(oof_a5_a1, "A5(K=15,a=0.15) + A1(LOMO)")

# A2 + A5
for alpha_day in [0.10, 0.20]:
    oof_a2_a5 = apply_day_priors(oof_best, train_df, train_months, y, alpha=alpha_day)
    oof_a2_a5 = apply_knn(oof_a2_a5, X_train_raw, y, K=15, alpha=0.15)
    eval_skf_lomo(oof_a2_a5, f"A2(a={alpha_day}) + A5(K=15,a=0.15)")


elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.0f}s")
print("=" * 90)
