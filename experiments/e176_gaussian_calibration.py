"""E176: Gaussian/Mixture Calibration on OOF predictions.

Three approaches:
  1. Beta calibration (parametric isotonic, 2 params per class)
  2. GMM on prediction simplex (cluster archetypes + empirical correction)
  3. Gaussian discriminant on logits (class-conditional Gaussians + Bayesian update)

All evaluated on LOMO. Can use month-specific priors with month-invariant likelihoods.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train
from src.metrics import compute_map
from src.postprocessing import N_CLASSES, renorm_rows, top2_margin

ROOT = Path(__file__).resolve().parent.parent

print("=" * 90)
print("  E176: Gaussian / Mixture Calibration")
print("=" * 90)

t0 = time.time()

train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()


def eval_skf_lomo(oof, name=""):
    skf, _ = compute_map(y, oof)
    lomo = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            lomo[m] = s
    lomo_avg = np.mean(list(lomo.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo.items()))
    print(f"  {name:<55s}  SKF={skf:.4f}  LOMO={lomo_avg:.4f}  [{month_str}]")
    return skf, lomo_avg, lomo


print("\n--- Baseline ---")
eval_skf_lomo(oof_best, "E175 best")


# ══════════════════════════════════════════════════════════════════════
# 1. Beta Calibration (parametric, 2 params per class)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  1. Beta Calibration")
print("=" * 90)


def beta_calibrate_class(probs_c, y_bin, a_init=1.0, b_init=0.0):
    """Fit P(y=1|s) = sigmoid(a * logit(s) + b) via MLE."""
    logits = np.log(np.clip(probs_c, 1e-8, 1 - 1e-8))

    def nll(params):
        a, b = params
        z = a * logits + b
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p))

    result = minimize(nll, [a_init, b_init], method="Nelder-Mead", options={"maxiter": 500})
    return result.x


def apply_beta_calibration(preds, params_list):
    """Apply fitted beta calibration to predictions."""
    out = preds.copy()
    for c, (a, b) in enumerate(params_list):
        logits = np.log(np.clip(preds[:, c], 1e-8, 1 - 1e-8))
        z = a * logits + b
        out[:, c] = 1.0 / (1.0 + np.exp(-z))
    return renorm_rows(out)


# Non-CV beta calibration
params_beta = []
for c in range(N_CLASSES):
    y_bin = (y == c).astype(float)
    a, b = beta_calibrate_class(oof_best[:, c], y_bin)
    params_beta.append((a, b))
    print(f"  {CLASSES[c]:20s}: a={a:.3f}, b={b:.3f}")

oof_beta = apply_beta_calibration(oof_best, params_beta)
eval_skf_lomo(oof_beta, "Beta cal (non-CV, fit all)")

# LOMO beta calibration (fit on 3 months, eval on 1)
oof_beta_lomo = oof_best.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    params_m = []
    for c in range(N_CLASSES):
        y_bin = (y[mask_train] == c).astype(float)
        a, b = beta_calibrate_class(oof_best[mask_train, c], y_bin)
        params_m.append((a, b))
    oof_beta_lomo[mask_held] = apply_beta_calibration(oof_best[mask_held], params_m)
oof_beta_lomo = renorm_rows(oof_beta_lomo)
eval_skf_lomo(oof_beta_lomo, "Beta cal (LOMO: fit 3 months, eval 1)")

# Beta on lgb
params_beta_lgb = []
for c in range(N_CLASSES):
    y_bin = (y == c).astype(float)
    a, b = beta_calibrate_class(oof_lgb[:, c], y_bin)
    params_beta_lgb.append((a, b))
oof_beta_lgb = apply_beta_calibration(oof_lgb, params_beta_lgb)
eval_skf_lomo(oof_beta_lgb, "Beta cal on LGB (non-CV)")


# ══════════════════════════════════════════════════════════════════════
# 2. GMM on Prediction Simplex (archetype correction)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  2. GMM Archetype Correction")
print("=" * 90)


def gmm_archetype_correction(preds, y_true, n_components=20, alpha=0.3):
    """Cluster predictions, compute per-cluster true distribution, blend."""
    # Work in log-probability space for better GMM fitting
    log_preds = np.log(np.clip(preds, 1e-8, 1.0))

    gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                          random_state=42, max_iter=200)
    gmm.fit(log_preds)

    # Per-cluster empirical class distribution
    assignments = gmm.predict(log_preds)
    cluster_dists = np.zeros((n_components, N_CLASSES))
    for k in range(n_components):
        mask = assignments == k
        if mask.sum() > 0:
            cluster_dists[k] = np.bincount(y_true[mask], minlength=N_CLASSES).astype(float)
            cluster_dists[k] /= max(cluster_dists[k].sum(), 1e-12)
        else:
            cluster_dists[k] = counts / counts.sum()

    # Soft assignment + blend
    responsibilities = gmm.predict_proba(log_preds)  # (n, n_components)
    archetype_preds = responsibilities @ cluster_dists  # (n, N_CLASSES)

    # Blend: (1-alpha) * model + alpha * archetype
    out = (1 - alpha) * preds + alpha * archetype_preds
    return renorm_rows(out), gmm, cluster_dists


for n_comp in [10, 20, 30, 50]:
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        oof_gmm, _, _ = gmm_archetype_correction(oof_best, y, n_components=n_comp, alpha=alpha)
        _, lomo, _ = eval_skf_lomo(oof_gmm, f"GMM (K={n_comp}, alpha={alpha})")

# LOMO GMM (fit on 3 months, apply to held-out)
print("\n  GMM LOMO (fit on 3 months, eval on 1):")
for n_comp in [10, 20]:
    for alpha in [0.2, 0.3]:
        oof_gmm_lomo = oof_best.copy()
        for held_month in sorted(set(train_months)):
            mask_held = train_months == held_month
            mask_train = ~mask_held
            log_p = np.log(np.clip(oof_best[mask_train], 1e-8, 1.0))
            gmm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                                  random_state=42, max_iter=200)
            gmm.fit(log_p)

            # Cluster distributions from training months
            assignments = gmm.predict(log_p)
            cluster_dists = np.zeros((n_comp, N_CLASSES))
            for k in range(n_comp):
                mask_k = assignments == k
                if mask_k.sum() > 0:
                    cluster_dists[k] = np.bincount(y[mask_train][mask_k], minlength=N_CLASSES).astype(float)
                    cluster_dists[k] /= max(cluster_dists[k].sum(), 1e-12)
                else:
                    cluster_dists[k] = p_train

            # Apply to held-out month
            log_p_held = np.log(np.clip(oof_best[mask_held], 1e-8, 1.0))
            resp = gmm.predict_proba(log_p_held)
            archetype = resp @ cluster_dists
            oof_gmm_lomo[mask_held] = (1 - alpha) * oof_best[mask_held] + alpha * archetype

        oof_gmm_lomo = renorm_rows(oof_gmm_lomo)
        eval_skf_lomo(oof_gmm_lomo, f"GMM LOMO (K={n_comp}, a={alpha})")


# ══════════════════════════════════════════════════════════════════════
# 3. Gaussian Discriminant on Logits
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  3. Gaussian Discriminant on Logits")
print("=" * 90)


def gaussian_discriminant(preds, y_true, prior=None, regularize=0.01):
    """Fit class-conditional Gaussians on logits, recompute posteriors.

    P(c|x) ∝ N(logit(x) | mu_c, Sigma_c) * prior(c)

    Uses diagonal covariance (independent features assumption).
    """
    logits = np.log(np.clip(preds, 1e-8, 1.0))

    if prior is None:
        prior = np.bincount(y_true, minlength=N_CLASSES).astype(float)
        prior /= prior.sum()

    # Fit class-conditional Gaussians
    mu = np.zeros((N_CLASSES, N_CLASSES))
    sigma = np.zeros((N_CLASSES, N_CLASSES))

    for c in range(N_CLASSES):
        mask = y_true == c
        if mask.sum() >= 3:
            mu[c] = logits[mask].mean(axis=0)
            sigma[c] = logits[mask].std(axis=0) + regularize
        else:
            mu[c] = logits.mean(axis=0)
            sigma[c] = logits.std(axis=0) + regularize

    # Compute posteriors: P(c|x) ∝ prod_j N(x_j | mu_cj, sigma_cj) * prior_c
    log_posteriors = np.zeros_like(preds)
    for c in range(N_CLASSES):
        log_lik = -0.5 * np.sum(((logits - mu[c]) / sigma[c]) ** 2, axis=1)
        log_lik -= np.sum(np.log(sigma[c]))
        log_posteriors[:, c] = log_lik + np.log(prior[c] + 1e-12)

    # Normalize
    log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
    posteriors = np.exp(log_posteriors)
    return renorm_rows(posteriors), mu, sigma


# Non-CV
oof_gda, mu_gda, sig_gda = gaussian_discriminant(oof_best, y)
eval_skf_lomo(oof_gda, "GDA (non-CV, train prior)")

# LOMO GDA
oof_gda_lomo = oof_best.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held
    oof_gda_m, _, _ = gaussian_discriminant(oof_best[mask_train], y[mask_train])
    # Re-apply fitted model to held-out
    logits = np.log(np.clip(oof_best, 1e-8, 1.0))
    logits_held = logits[mask_held]

    # Refit on training months, apply to held-out
    mu_m = np.zeros((N_CLASSES, N_CLASSES))
    sigma_m = np.zeros((N_CLASSES, N_CLASSES))
    for c in range(N_CLASSES):
        mask_c = y[mask_train] == c
        if mask_c.sum() >= 3:
            mu_m[c] = logits[mask_train][mask_c].mean(axis=0)
            sigma_m[c] = logits[mask_train][mask_c].std(axis=0) + 0.01
        else:
            mu_m[c] = logits[mask_train].mean(axis=0)
            sigma_m[c] = logits[mask_train].std(axis=0) + 0.01

    prior_m = np.bincount(y[mask_train], minlength=N_CLASSES).astype(float)
    prior_m /= prior_m.sum()

    log_post = np.zeros((mask_held.sum(), N_CLASSES))
    for c in range(N_CLASSES):
        log_lik = -0.5 * np.sum(((logits_held - mu_m[c]) / sigma_m[c]) ** 2, axis=1)
        log_lik -= np.sum(np.log(sigma_m[c]))
        log_post[:, c] = log_lik + np.log(prior_m[c] + 1e-12)

    log_post -= log_post.max(axis=1, keepdims=True)
    oof_gda_lomo[mask_held] = np.exp(log_post)

oof_gda_lomo = renorm_rows(oof_gda_lomo)
eval_skf_lomo(oof_gda_lomo, "GDA (LOMO: fit 3 months, eval 1)")

# GDA with month-specific priors (the key idea: invariant likelihood + varying prior)
print("\n  GDA with month-specific priors:")
oof_gda_mprior = oof_best.copy()
# Fit Gaussians on ALL data (month-invariant)
logits_all = np.log(np.clip(oof_best, 1e-8, 1.0))
mu_all = np.zeros((N_CLASSES, N_CLASSES))
sigma_all = np.zeros((N_CLASSES, N_CLASSES))
for c in range(N_CLASSES):
    mask_c = y == c
    mu_all[c] = logits_all[mask_c].mean(axis=0)
    sigma_all[c] = logits_all[mask_c].std(axis=0) + 0.01

# Apply with per-month priors
for m in sorted(set(train_months)):
    mask_m = train_months == m
    if mask_m.sum() < 10:
        continue

    # Month-specific prior from training data
    prior_m = np.bincount(y[mask_m], minlength=N_CLASSES).astype(float)
    prior_m = np.maximum(prior_m, 0.5)  # smoothing
    prior_m /= prior_m.sum()

    logits_m = logits_all[mask_m]
    log_post = np.zeros((mask_m.sum(), N_CLASSES))
    for c in range(N_CLASSES):
        log_lik = -0.5 * np.sum(((logits_m - mu_all[c]) / sigma_all[c]) ** 2, axis=1)
        log_lik -= np.sum(np.log(sigma_all[c]))
        log_post[:, c] = log_lik + np.log(prior_m[c] + 1e-12)

    log_post -= log_post.max(axis=1, keepdims=True)
    oof_gda_mprior[mask_m] = np.exp(log_post)

oof_gda_mprior = renorm_rows(oof_gda_mprior)
eval_skf_lomo(oof_gda_mprior, "GDA month-specific priors (global lik, per-month prior)")

# LOMO version: fit Gaussians on 3 months, apply with held-out month's empirical prior
oof_gda_mprior_lomo = oof_best.copy()
for held_month in sorted(set(train_months)):
    mask_held = train_months == held_month
    mask_train = ~mask_held

    # Fit Gaussians on other 3 months
    mu_m = np.zeros((N_CLASSES, N_CLASSES))
    sigma_m = np.zeros((N_CLASSES, N_CLASSES))
    for c in range(N_CLASSES):
        mask_c = y[mask_train] == c
        if mask_c.sum() >= 3:
            mu_m[c] = logits_all[mask_train][mask_c].mean(axis=0)
            sigma_m[c] = logits_all[mask_train][mask_c].std(axis=0) + 0.01
        else:
            mu_m[c] = logits_all[mask_train].mean(axis=0)
            sigma_m[c] = logits_all[mask_train].std(axis=0) + 0.01

    # Use held-out month's ACTUAL prior (cheating — for upper bound)
    prior_held = np.bincount(y[mask_held], minlength=N_CLASSES).astype(float)
    prior_held = np.maximum(prior_held, 0.5)
    prior_held /= prior_held.sum()

    logits_held = logits_all[mask_held]
    log_post = np.zeros((mask_held.sum(), N_CLASSES))
    for c in range(N_CLASSES):
        log_lik = -0.5 * np.sum(((logits_held - mu_m[c]) / sigma_m[c]) ** 2, axis=1)
        log_lik -= np.sum(np.log(sigma_m[c]))
        log_post[:, c] = log_lik + np.log(prior_held[c] + 1e-12)

    log_post -= log_post.max(axis=1, keepdims=True)
    oof_gda_mprior_lomo[mask_held] = np.exp(log_post)

oof_gda_mprior_lomo = renorm_rows(oof_gda_mprior_lomo)
eval_skf_lomo(oof_gda_mprior_lomo, "GDA (LOMO, oracle month prior — upper bound)")


# ══════════════════════════════════════════════════════════════════════
# 4. Blending calibrated with original (hedge)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  4. Blending Calibrated with Original")
print("=" * 90)

for alpha in [0.1, 0.2, 0.3, 0.5]:
    # Beta blend
    blend_beta = (1-alpha) * oof_best + alpha * oof_beta
    eval_skf_lomo(renorm_rows(blend_beta), f"Blend: {1-alpha:.0%} orig + {alpha:.0%} beta")

print()
for alpha in [0.1, 0.2, 0.3, 0.5]:
    # GDA blend
    blend_gda = (1-alpha) * oof_best + alpha * oof_gda
    eval_skf_lomo(renorm_rows(blend_gda), f"Blend: {1-alpha:.0%} orig + {alpha:.0%} GDA")

print()
# Best GMM blend
for n_comp in [20]:
    for gmm_alpha in [0.2, 0.3]:
        oof_gmm, _, _ = gmm_archetype_correction(oof_best, y, n_components=n_comp, alpha=gmm_alpha)
        for blend_alpha in [0.3, 0.5]:
            blend = (1-blend_alpha) * oof_best + blend_alpha * oof_gmm
            eval_skf_lomo(renorm_rows(blend), f"Blend: {1-blend_alpha:.0%} orig + {blend_alpha:.0%} GMM(K={n_comp},a={gmm_alpha})")


elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.0f}s")
print("=" * 90)
