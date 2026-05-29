"""E176: Leakage Analysis — are stacking gains real or overfit?

Key question: GMM->Iso LOMO=0.6021 (+0.056) and LP LOMO=0.8585.
Are these real improvements or train-train leakage?

Tests:
1. GMM/Iso/KNN: fit on 3 months, apply to held-out month (true LOMO)
2. LP: check if train labels leak back through the graph
3. Compare non-CV vs true LOMO-CV for all stacking combos
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map
from src.postprocessing import N_CLASSES, renorm_rows, top2_margin

ROOT = Path(__file__).resolve().parent.parent

print("=" * 90)
print("  E176: Leakage Analysis — Are Stacking Gains Real?")
print("=" * 90)

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
feat_cols = [c for c in train_feats.columns if c in test_feats.columns]
X_train = np.nan_to_num(train_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

unique_months = sorted(set(train_months))


def lomo_score(oof):
    scores = {}
    for m in unique_months:
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            scores[m] = s
    return np.mean(list(scores.values())), scores


def print_result(name, skf, lomo_avg, lomo_dict):
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_dict.items()))
    print(f"  {name:<55s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]")


# ══════════════════════════════════════════════════════════════════════
# Baseline
# ══════════════════════════════════════════════════════════════════════
print("\n--- Baseline ---")
skf_base, _ = compute_map(y, oof_best)
lomo_base, lomo_base_d = lomo_score(oof_best)
print_result("E175 best", skf_base, lomo_base, lomo_base_d)


# ══════════════════════════════════════════════════════════════════════
# 1. TRUE LOMO-CV: fit on 3 months, apply to held-out month
#    This is the ONLY honest evaluation.
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TRUE LOMO-CV: Fit on 3 months, apply to held-out")
print("  (This is the only honest evaluation)")
print("=" * 90)


def true_lomo_isotonic(oof):
    """Fit isotonic on 3 months, predict on held-out. TRUE generalization."""
    out = oof.copy()
    for held in unique_months:
        mask_held = train_months == held
        mask_train = ~mask_held
        for c in range(N_CLASSES):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(oof[mask_train, c], (y[mask_train] == c).astype(float))
            out[mask_held, c] = ir.predict(oof[mask_held, c])
    return renorm_rows(out)


def true_lomo_gmm(oof, n_components=50, alpha=0.5):
    """Fit GMM on 3 months, predict on held-out. TRUE generalization."""
    out = oof.copy()
    for held in unique_months:
        mask_held = train_months == held
        mask_train = ~mask_held

        log_p_tr = np.log(np.clip(oof[mask_train], 1e-8, 1.0))
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                              random_state=42, max_iter=200)
        gmm.fit(log_p_tr)

        # Cluster distributions from training months only
        assignments = gmm.predict(log_p_tr)
        cluster_dists = np.zeros((n_components, N_CLASSES))
        for k in range(n_components):
            mask_k = assignments == k
            if mask_k.sum() > 0:
                cluster_dists[k] = np.bincount(y[mask_train][mask_k], minlength=N_CLASSES).astype(float)
                cluster_dists[k] /= max(cluster_dists[k].sum(), 1e-12)
            else:
                cluster_dists[k] = p_train

        # Apply to held-out
        log_p_held = np.log(np.clip(oof[mask_held], 1e-8, 1.0))
        resp = gmm.predict_proba(log_p_held)
        archetype = resp @ cluster_dists
        out[mask_held] = (1 - alpha) * oof[mask_held] + alpha * archetype

    return renorm_rows(out)


def true_lomo_knn(oof, X_data, K=15, alpha=0.15):
    """KNN using only OTHER months' labels. TRUE generalization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    out = oof.copy()
    margin = top2_margin(out)

    for held in unique_months:
        mask_held = train_months == held
        mask_train = ~mask_held
        X_tr = X_scaled[mask_train]
        y_tr = y[mask_train]

        for i in np.where(mask_held)[0]:
            if margin[i] > 0.5:
                continue
            dists = np.sqrt(((X_tr - X_scaled[i]) ** 2).sum(axis=1))
            top_k = np.argsort(dists)[:K]
            weights = 1.0 / np.maximum(dists[top_k], 1e-8)
            weights /= weights.sum()
            knn_dist = np.zeros(N_CLASSES)
            for j, w in zip(top_k, weights):
                knn_dist[y_tr[j]] += w
            out[i] = (1 - alpha) * out[i] + alpha * knn_dist

    return renorm_rows(out)


# Individual techniques — TRUE LOMO-CV
print("\n--- Individual (TRUE LOMO-CV) ---")

oof_iso_cv = true_lomo_isotonic(oof_best)
skf, _ = compute_map(y, oof_iso_cv)
lomo, lomo_d = lomo_score(oof_iso_cv)
print_result("Isotonic (TRUE LOMO-CV)", skf, lomo, lomo_d)

for n_comp in [20, 50]:
    for alpha in [0.3, 0.5]:
        oof_gmm_cv = true_lomo_gmm(oof_best, n_components=n_comp, alpha=alpha)
        skf, _ = compute_map(y, oof_gmm_cv)
        lomo, lomo_d = lomo_score(oof_gmm_cv)
        print_result(f"GMM K={n_comp} a={alpha} (TRUE LOMO-CV)", skf, lomo, lomo_d)

oof_knn_cv = true_lomo_knn(oof_best, X_train, K=15, alpha=0.15)
skf, _ = compute_map(y, oof_knn_cv)
lomo, lomo_d = lomo_score(oof_knn_cv)
print_result("KNN K=15 a=0.15 (TRUE LOMO-CV)", skf, lomo, lomo_d)

# Stacked — TRUE LOMO-CV
print("\n--- Stacked (TRUE LOMO-CV) ---")

# GMM -> Iso (both fitted on 3 months, applied to held-out)
oof_gmm_iso_cv = true_lomo_isotonic(true_lomo_gmm(oof_best, 50, 0.5))
skf, _ = compute_map(y, oof_gmm_iso_cv)
lomo, lomo_d = lomo_score(oof_gmm_iso_cv)
print_result("GMM->Iso (TRUE LOMO-CV)", skf, lomo, lomo_d)

# Iso -> GMM
oof_iso_gmm_cv = true_lomo_gmm(true_lomo_isotonic(oof_best), 50, 0.5)
skf, _ = compute_map(y, oof_iso_gmm_cv)
lomo, lomo_d = lomo_score(oof_iso_gmm_cv)
print_result("Iso->GMM (TRUE LOMO-CV)", skf, lomo, lomo_d)

# GMM -> Iso -> KNN
oof_3way_cv = true_lomo_knn(true_lomo_isotonic(true_lomo_gmm(oof_best, 50, 0.5)), X_train, K=15, alpha=0.15)
skf, _ = compute_map(y, oof_3way_cv)
lomo, lomo_d = lomo_score(oof_3way_cv)
print_result("GMM->Iso->KNN (TRUE LOMO-CV)", skf, lomo, lomo_d)

# Iso -> GMM -> KNN
oof_3way2_cv = true_lomo_knn(true_lomo_gmm(true_lomo_isotonic(oof_best), 50, 0.5), X_train, K=15, alpha=0.15)
skf, _ = compute_map(y, oof_3way2_cv)
lomo, lomo_d = lomo_score(oof_3way2_cv)
print_result("Iso->GMM->KNN (TRUE LOMO-CV)", skf, lomo, lomo_d)


# ══════════════════════════════════════════════════════════════════════
# 2. Label Propagation leakage check
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  Label Propagation Leakage Analysis")
print("=" * 90)

from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
X_all_pca = pca.fit_transform(np.vstack([X_train, X_test]))

# Test 1: LP on train only (no test data) — does it leak through train graph?
print("\n--- LP train-only (no test data in graph) ---")
for nn in [7, 15]:
    for lp_alpha in [0.1, 0.2]:
        # TRUE LOMO: fit LP on 3 months, predict held-out
        oof_lp_cv = oof_best.copy()
        for held in unique_months:
            mask_held = train_months == held
            mask_train = ~mask_held

            # Build graph with training months only + held-out as unlabeled
            X_lomo = X_all_pca[:len(y)]  # train only
            labels_lomo = y.copy()
            labels_lomo[mask_held] = -1  # mask held-out as unlabeled

            ls = LabelSpreading(kernel='knn', n_neighbors=nn, alpha=lp_alpha, max_iter=50)
            ls.fit(X_lomo, labels_lomo)
            lp_held = ls.label_distributions_[mask_held]

            # Blend
            for ba in [0.1, 0.2]:
                oof_lp_test = oof_best.copy()
                oof_lp_test[mask_held] = (1-ba) * oof_best[mask_held] + ba * renorm_rows(lp_held)
                # Only evaluate on this held month
                s, _ = compute_map(y[mask_held], renorm_rows(oof_lp_test[mask_held]))
                # Store for LOMO

        # Full LOMO evaluation
        oof_lp_full = oof_best.copy()
        for held in unique_months:
            mask_held = train_months == held
            labels_lomo = y.copy()
            labels_lomo[mask_held] = -1
            X_lomo = X_all_pca[:len(y)]
            ls = LabelSpreading(kernel='knn', n_neighbors=nn, alpha=lp_alpha, max_iter=50)
            ls.fit(X_lomo, labels_lomo)
            lp_held = ls.label_distributions_[mask_held]
            oof_lp_full[mask_held] = (1-0.2) * oof_best[mask_held] + 0.2 * renorm_rows(lp_held)

        oof_lp_full = renorm_rows(oof_lp_full)
        skf, _ = compute_map(y, oof_lp_full)
        lomo, lomo_d = lomo_score(oof_lp_full)
        print_result(f"LP train-only LOMO-CV (nn={nn},a={lp_alpha},b=0.2)", skf, lomo, lomo_d)


# Test 2: LP with test data in graph — TRUE LOMO (held-out month as unlabeled)
print("\n--- LP with test data (transductive, TRUE LOMO) ---")
for nn in [7, 15]:
    for lp_alpha in [0.1, 0.2]:
        oof_lp_trans = oof_best.copy()
        for held in unique_months:
            mask_held = train_months == held

            # Graph includes: labeled train (other months) + unlabeled test + unlabeled held-out
            labels_all = np.concatenate([y.copy(), np.full(len(test_df), -1)])
            labels_all[mask_held] = -1  # held-out month is unlabeled

            ls = LabelSpreading(kernel='knn', n_neighbors=nn, alpha=lp_alpha, max_iter=50)
            ls.fit(X_all_pca, labels_all)
            lp_held = ls.label_distributions_[:len(y)][mask_held]

            oof_lp_trans[mask_held] = (1-0.2) * oof_best[mask_held] + 0.2 * renorm_rows(lp_held)

        oof_lp_trans = renorm_rows(oof_lp_trans)
        skf, _ = compute_map(y, oof_lp_trans)
        lomo, lomo_d = lomo_score(oof_lp_trans)
        print_result(f"LP transductive LOMO-CV (nn={nn},a={lp_alpha},b=0.2)", skf, lomo, lomo_d)


# ══════════════════════════════════════════════════════════════════════
# 3. Comparison: non-CV vs TRUE LOMO-CV (overfit quantification)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  Non-CV vs TRUE LOMO-CV Comparison")
print("  (Gap = amount of overfit)")
print("=" * 90)

print(f"\n  {'Technique':<40s} {'Non-CV LOMO':>12s} {'TRUE CV LOMO':>12s} {'Overfit':>8s}")
print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8}")
print(f"  {'Baseline':<40s} {'0.5461':>12s} {'0.5461':>12s} {'0.000':>8s}")

# Recompute non-CV versions for comparison
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture

# Iso non-CV
oof_iso_ncv = oof_best.copy()
for c in range(N_CLASSES):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_best[:, c], (y == c).astype(float))
    oof_iso_ncv[:, c] = ir.predict(oof_best[:, c])
oof_iso_ncv = renorm_rows(oof_iso_ncv)
lomo_iso_ncv, _ = lomo_score(oof_iso_ncv)
lomo_iso_cv, _ = lomo_score(oof_iso_cv)
print(f"  {'Isotonic':<40s} {lomo_iso_ncv:>12.4f} {lomo_iso_cv:>12.4f} {lomo_iso_ncv-lomo_iso_cv:>+8.4f}")

# GMM non-CV
log_p = np.log(np.clip(oof_best, 1e-8, 1.0))
gmm = GaussianMixture(n_components=50, covariance_type="diag", random_state=42, max_iter=200)
gmm.fit(log_p)
assignments = gmm.predict(log_p)
cluster_dists = np.zeros((50, N_CLASSES))
for k in range(50):
    mask_k = assignments == k
    if mask_k.sum() > 0:
        cluster_dists[k] = np.bincount(y[mask_k], minlength=N_CLASSES).astype(float)
        cluster_dists[k] /= max(cluster_dists[k].sum(), 1e-12)
    else:
        cluster_dists[k] = p_train
resp = gmm.predict_proba(log_p)
archetype = resp @ cluster_dists
oof_gmm_ncv = renorm_rows(0.5 * oof_best + 0.5 * archetype)
lomo_gmm_ncv, _ = lomo_score(oof_gmm_ncv)
oof_gmm_cv_50 = true_lomo_gmm(oof_best, 50, 0.5)
lomo_gmm_cv, _ = lomo_score(oof_gmm_cv_50)
print(f"  {'GMM K=50 a=0.5':<40s} {lomo_gmm_ncv:>12.4f} {lomo_gmm_cv:>12.4f} {lomo_gmm_ncv-lomo_gmm_cv:>+8.4f}")

# KNN non-CV
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
oof_knn_ncv = oof_best.copy()
margin = top2_margin(oof_knn_ncv)
for i in range(len(oof_knn_ncv)):
    if margin[i] > 0.5:
        continue
    dists = np.sqrt(((X_scaled - X_scaled[i]) ** 2).sum(axis=1))
    dists[i] = np.inf
    top_k = np.argsort(dists)[:15]
    weights = 1.0 / np.maximum(dists[top_k], 1e-8)
    weights /= weights.sum()
    knn_dist = np.zeros(N_CLASSES)
    for j, w in zip(top_k, weights):
        knn_dist[y[j]] += w
    oof_knn_ncv[i] = 0.85 * oof_knn_ncv[i] + 0.15 * knn_dist
oof_knn_ncv = renorm_rows(oof_knn_ncv)
lomo_knn_ncv, _ = lomo_score(oof_knn_ncv)
lomo_knn_cv, _ = lomo_score(oof_knn_cv)
print(f"  {'KNN K=15 a=0.15':<40s} {lomo_knn_ncv:>12.4f} {lomo_knn_cv:>12.4f} {lomo_knn_ncv-lomo_knn_cv:>+8.4f}")

# GMM->Iso non-CV
oof_gmm_iso_ncv = oof_iso_ncv.copy()  # Iso is already applied to oof_best
# Need to re-do: first GMM(non-CV) then Iso(non-CV) on the GMM output
oof_gmm_then_iso_ncv = oof_gmm_ncv.copy()
for c in range(N_CLASSES):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_gmm_ncv[:, c], (y == c).astype(float))
    oof_gmm_then_iso_ncv[:, c] = ir.predict(oof_gmm_ncv[:, c])
oof_gmm_then_iso_ncv = renorm_rows(oof_gmm_then_iso_ncv)
lomo_gi_ncv, _ = lomo_score(oof_gmm_then_iso_ncv)
lomo_gi_cv, _ = lomo_score(oof_gmm_iso_cv)
print(f"  {'GMM->Iso':<40s} {lomo_gi_ncv:>12.4f} {lomo_gi_cv:>12.4f} {lomo_gi_ncv-lomo_gi_cv:>+8.4f}")

# GMM->Iso->KNN non-CV vs CV
lomo_gik_cv, _ = lomo_score(oof_3way_cv)
# Non-CV: GMM->Iso->KNN (all fit on all data)
oof_gik_ncv = oof_gmm_then_iso_ncv.copy()
margin_gik = top2_margin(oof_gik_ncv)
for i in range(len(oof_gik_ncv)):
    if margin_gik[i] > 0.5:
        continue
    dists = np.sqrt(((X_scaled - X_scaled[i]) ** 2).sum(axis=1))
    dists[i] = np.inf
    top_k = np.argsort(dists)[:15]
    weights = 1.0 / np.maximum(dists[top_k], 1e-8)
    weights /= weights.sum()
    knn_dist = np.zeros(N_CLASSES)
    for j, w in zip(top_k, weights):
        knn_dist[y[j]] += w
    oof_gik_ncv[i] = 0.85 * oof_gik_ncv[i] + 0.15 * knn_dist
oof_gik_ncv = renorm_rows(oof_gik_ncv)
lomo_gik_ncv, _ = lomo_score(oof_gik_ncv)
print(f"  {'GMM->Iso->KNN':<40s} {lomo_gik_ncv:>12.4f} {lomo_gik_cv:>12.4f} {lomo_gik_ncv-lomo_gik_cv:>+8.4f}")

print(f"\n  Note: 'Overfit' = non-CV LOMO minus TRUE LOMO-CV.")
print(f"  Positive = technique benefits from seeing held-out data.")
print(f"  Zero or small = technique generalizes across months.")

print(f"\nDone.")
