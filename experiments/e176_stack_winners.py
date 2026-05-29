"""E176: Stack ALL winning techniques + generate test submissions.

Tests all combinations and stacking orders of:
  - A6 Isotonic (LOMO +0.019)
  - GMM archetype correction (LOMO +0.017)
  - A5 KNN blending (LOMO +0.001)
  - GBDT diversity blend (LOMO +0.001)
  - TTA (test-time augmentation)
  - Per-class ensemble weight optimization
  - Label propagation (transductive)
  - MAPLS (Dirichlet-regularized MLLS)

Evaluates ALL on LOMO. Saves best submissions.
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows, top2_margin

ROOT = Path(__file__).resolve().parent.parent

print("=" * 90)
print("  E176: Stack ALL Winners + Submissions")
print("=" * 90)
t0 = time.time()

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

# Load predictions
oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))
test_best = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
test_lgb = renorm_rows(np.load(ROOT / "test_e175_lgb.npy").astype(np.float64))

# Load features for KNN
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
feat_cols = [c for c in train_feats.columns if c in test_feats.columns]
X_train = np.nan_to_num(train_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# Load Phase C GBDT if available
oof_gbdt = None
test_gbdt = None
if (ROOT / "oof_e176_C_extra_gbdt.npy").exists():
    oof_gbdt = np.load(ROOT / "oof_e176_C_extra_gbdt.npy")
    test_gbdt = np.load(ROOT / "test_e176_C_extra_gbdt.npy")


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
    print(f"  {name:<55s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]")
    return skf, lomo_avg


# ══════════════════════════════════════════════════════════════════════
# Individual techniques (OOF)
# ══════════════════════════════════════════════════════════════════════

print("\n--- Baseline ---")
eval_skf_lomo(oof_best, "E175 best")

# ── A6 Isotonic ──
def apply_isotonic_oof(oof, y_true):
    """Fit isotonic on full OOF, return calibrated OOF (non-CV)."""
    out = oof.copy()
    models = []
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof[:, c], (y_true == c).astype(float))
        out[:, c] = ir.predict(oof[:, c])
        models.append(ir)
    return renorm_rows(out), models

def apply_isotonic_test(test, models):
    out = test.copy()
    for c, ir in enumerate(models):
        out[:, c] = ir.predict(test[:, c])
    return renorm_rows(out)

print("\n--- A6 Isotonic ---")
oof_iso, iso_models = apply_isotonic_oof(oof_best, y)
eval_skf_lomo(oof_iso, "A6 Isotonic (non-CV)")

# ── GMM Archetype ──
def apply_gmm_oof(oof, y_true, n_components=50, alpha=0.5):
    log_p = np.log(np.clip(oof, 1e-8, 1.0))
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                          random_state=42, max_iter=200)
    gmm.fit(log_p)
    assignments = gmm.predict(log_p)
    cluster_dists = np.zeros((n_components, N_CLASSES))
    for k in range(n_components):
        mask = assignments == k
        if mask.sum() > 0:
            cluster_dists[k] = np.bincount(y_true[mask], minlength=N_CLASSES).astype(float)
            cluster_dists[k] /= max(cluster_dists[k].sum(), 1e-12)
        else:
            cluster_dists[k] = p_train
    resp = gmm.predict_proba(log_p)
    archetype = resp @ cluster_dists
    out = (1 - alpha) * oof + alpha * archetype
    return renorm_rows(out), gmm, cluster_dists

def apply_gmm_test(test, gmm, cluster_dists, alpha=0.5):
    log_p = np.log(np.clip(test, 1e-8, 1.0))
    resp = gmm.predict_proba(log_p)
    archetype = resp @ cluster_dists
    out = (1 - alpha) * test + alpha * archetype
    return renorm_rows(out)

print("\n--- GMM Archetype ---")
oof_gmm, gmm_model, gmm_dists = apply_gmm_oof(oof_best, y, n_components=50, alpha=0.5)
eval_skf_lomo(oof_gmm, "GMM (K=50, a=0.5)")

# ── A5 KNN ──
def apply_knn(preds, X_data, y_data, X_query=None, K=15, alpha=0.15):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    if X_query is not None:
        X_q_scaled = scaler.transform(X_query)
    else:
        X_q_scaled = X_scaled
    out = preds.copy()
    margin = top2_margin(out)
    for i in range(len(out)):
        if margin[i] > 0.5:
            continue
        dists = np.sqrt(((X_scaled - X_q_scaled[i]) ** 2).sum(axis=1))
        if X_query is None:
            dists[i] = np.inf
        top_k = np.argsort(dists)[:K]
        weights = 1.0 / np.maximum(dists[top_k], 1e-8)
        weights /= weights.sum()
        knn_dist = np.zeros(N_CLASSES)
        for j, w in zip(top_k, weights):
            knn_dist[y_data[j]] += w
        out[i] = (1 - alpha) * out[i] + alpha * knn_dist
    return renorm_rows(out)

print("\n--- A5 KNN ---")
oof_knn = apply_knn(oof_best, X_train, y, K=15, alpha=0.15)
eval_skf_lomo(oof_knn, "A5 KNN (K=15, a=0.15)")

# ── GBDT blend ──
if oof_gbdt is not None:
    print("\n--- GBDT diversity ---")
    oof_gbdt_blend = 0.9 * oof_best + 0.1 * renorm_rows(oof_gbdt)
    oof_gbdt_blend = renorm_rows(oof_gbdt_blend)
    eval_skf_lomo(oof_gbdt_blend, "E175 + GBDT@10%")

# ── Label Propagation (transductive) ──
print("\n--- Label Propagation ---")
try:
    from sklearn.semi_supervised import LabelSpreading
    # Use top 50 features (PCA or select) for better graph
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    X_all = np.vstack([X_train, X_test])
    X_all_pca = pca.fit_transform(X_all)

    all_labels = np.concatenate([y, np.full(len(test_df), -1)])

    for n_neighbors in [7, 15, 30]:
        for lp_alpha in [0.1, 0.2, 0.3]:
            ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors, alpha=lp_alpha, max_iter=50)
            ls.fit(X_all_pca, all_labels)
            lp_train = ls.label_distributions_[:len(train_df)]
            lp_test = ls.label_distributions_[len(train_df):]

            # Blend with model predictions
            for blend_alpha in [0.1, 0.2, 0.3]:
                oof_lp = (1 - blend_alpha) * oof_best + blend_alpha * renorm_rows(lp_train)
                _, lomo = eval_skf_lomo(renorm_rows(oof_lp),
                    f"LP(nn={n_neighbors},a={lp_alpha}) blend@{blend_alpha}")

    # Keep best LP config
    best_lp_lomo = -1
    best_lp_config = None
    for n_neighbors in [7, 15]:
        for lp_alpha in [0.1, 0.2]:
            ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors, alpha=lp_alpha, max_iter=50)
            ls.fit(X_all_pca, all_labels)
            lp_train = ls.label_distributions_[:len(train_df)]
            lp_test = ls.label_distributions_[len(train_df):]
            for ba in [0.1, 0.2]:
                oof_lp = renorm_rows((1-ba) * oof_best + ba * renorm_rows(lp_train))
                _, lomo = eval_skf_lomo.__wrapped__(oof_lp) if hasattr(eval_skf_lomo, '__wrapped__') else (None, None)
                lomo_scores = {}
                for m in sorted(set(train_months)):
                    mask = train_months == m
                    if mask.sum() >= 10:
                        s, _ = compute_map(y[mask], oof_lp[mask])
                        lomo_scores[m] = s
                lomo = np.mean(list(lomo_scores.values()))
                if lomo > best_lp_lomo:
                    best_lp_lomo = lomo
                    best_lp_config = (n_neighbors, lp_alpha, ba, lp_train, lp_test)

    if best_lp_config:
        nn, la, ba, lp_tr, lp_te = best_lp_config
        print(f"  Best LP: nn={nn}, alpha={la}, blend={ba}, LOMO={best_lp_lomo:.4f}")
except Exception as e:
    print(f"  Label Propagation failed: {e}")
    best_lp_config = None

# ── TTA (Test-Time Augmentation) ──
print("\n--- TTA (Test-Time Augmentation) ---")
# Can only evaluate on test, not OOF. Just note the concept.
print("  TTA applies to test predictions only (noise + average). Will apply in final submissions.")

# ── MAPLS (Dirichlet-regularized MLLS) ──
print("\n--- MAPLS ---")
def mapls_estimate(test_probs, p_source, alpha_prior=None, max_iter=200, tol=1e-7):
    K = len(p_source)
    if alpha_prior is None:
        alpha_prior = np.ones(K) * 2.0
    w = p_source.copy()
    for _ in range(max_iter):
        ratio = w / np.maximum(p_source, 1e-12)
        adjusted = test_probs * ratio[np.newaxis, :]
        row_sums = adjusted.sum(axis=1, keepdims=True)
        adjusted = adjusted / np.maximum(row_sums, 1e-12)
        N = len(test_probs)
        ml_counts = adjusted.sum(axis=0)
        w_new = (ml_counts + alpha_prior - 1) / (N + alpha_prior.sum() - K)
        w_new = np.maximum(w_new, 1e-8)
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < tol:
            break
        w = w_new
    return w

# Test MAPLS on OOF by remapping months
# (MAPLS is really for test — on OOF, it estimates OOF month proportions which are known)
# Just validate it doesn't crash and show estimated proportions
from src.postprocessing import build_gbif_priors
gbif_priors = build_gbif_priors(p_train)
print("  MAPLS estimated proportions per test month:")
for m in sorted(set(test_months)):
    mask = test_months == m
    if mask.sum() < N_CLASSES:
        continue
    # Use GBIF as Dirichlet prior
    gbif_m = gbif_priors.get(m, p_train)
    alpha_gbif = gbif_m * 50  # concentration parameter
    w_mapls = mapls_estimate(test_best[mask], p_train, alpha_prior=alpha_gbif)
    print(f"    Month {m:2d} (n={mask.sum()}): {' '.join(f'{CLASSES[c][:4]}={w_mapls[c]:.3f}' for c in range(N_CLASSES))}")


# ══════════════════════════════════════════════════════════════════════
# STACKING COMBINATIONS (OOF)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  STACKING COMBINATIONS")
print("=" * 90)

# All possible 2-way and 3-way stacks
print("\n--- 2-way stacks ---")

# Iso + GMM
oof_iso_gmm, _, _ = apply_gmm_oof(oof_iso, y, n_components=50, alpha=0.5)
eval_skf_lomo(oof_iso_gmm, "Iso -> GMM")

oof_gmm_iso, _ = apply_isotonic_oof(oof_gmm, y)
eval_skf_lomo(oof_gmm_iso, "GMM -> Iso")

# Iso + KNN
oof_iso_knn = apply_knn(oof_iso, X_train, y, K=15, alpha=0.15)
eval_skf_lomo(oof_iso_knn, "Iso -> KNN")

oof_knn_iso, _ = apply_isotonic_oof(oof_knn, y)
eval_skf_lomo(oof_knn_iso, "KNN -> Iso")

# GMM + KNN
oof_gmm_knn = apply_knn(oof_gmm, X_train, y, K=15, alpha=0.15)
eval_skf_lomo(oof_gmm_knn, "GMM -> KNN")

oof_knn_gmm, _, _ = apply_gmm_oof(oof_knn, y, n_components=50, alpha=0.5)
eval_skf_lomo(oof_knn_gmm, "KNN -> GMM")

# Blend Iso + GMM (average, not sequential)
for w in [0.3, 0.5, 0.7]:
    blend = w * oof_iso + (1-w) * oof_gmm
    eval_skf_lomo(renorm_rows(blend), f"Blend: {w:.0%} Iso + {1-w:.0%} GMM")

print("\n--- 3-way stacks ---")

# Iso -> GMM -> KNN
oof_3way_a = apply_knn(oof_iso_gmm, X_train, y, K=15, alpha=0.15)
eval_skf_lomo(oof_3way_a, "Iso -> GMM -> KNN")

# Iso -> KNN -> GMM
oof_iso_knn_gmm, _, _ = apply_gmm_oof(oof_iso_knn, y, n_components=50, alpha=0.5)
eval_skf_lomo(oof_iso_knn_gmm, "Iso -> KNN -> GMM")

# GMM -> Iso -> KNN
oof_gmm_iso_knn = apply_knn(oof_gmm_iso, X_train, y, K=15, alpha=0.15)
eval_skf_lomo(oof_gmm_iso_knn, "GMM -> Iso -> KNN")

# KNN -> Iso -> GMM
oof_knn_iso_gmm, _, _ = apply_gmm_oof(oof_knn_iso, y, n_components=50, alpha=0.5)
eval_skf_lomo(oof_knn_iso_gmm, "KNN -> Iso -> GMM")

# KNN -> GMM -> Iso
oof_knn_gmm_iso, _ = apply_isotonic_oof(oof_knn_gmm, y)
eval_skf_lomo(oof_knn_gmm_iso, "KNN -> GMM -> Iso")

# GMM -> KNN -> Iso
oof_gmm_knn_iso, _ = apply_isotonic_oof(oof_gmm_knn, y)
eval_skf_lomo(oof_gmm_knn_iso, "GMM -> KNN -> Iso")

# With GBDT diversity
if oof_gbdt is not None:
    print("\n--- With GBDT diversity base ---")
    oof_div = renorm_rows(0.9 * oof_best + 0.1 * renorm_rows(oof_gbdt))
    oof_div_iso, iso_div_models = apply_isotonic_oof(oof_div, y)
    eval_skf_lomo(oof_div_iso, "GBDT@10% -> Iso")
    oof_div_gmm, gmm_div, gmm_div_dists = apply_gmm_oof(oof_div, y, n_components=50, alpha=0.5)
    eval_skf_lomo(oof_div_gmm, "GBDT@10% -> GMM")
    oof_div_iso_gmm, _, _ = apply_gmm_oof(oof_div_iso, y, n_components=50, alpha=0.5)
    eval_skf_lomo(oof_div_iso_gmm, "GBDT@10% -> Iso -> GMM")

# With LGB blend
print("\n--- With LGB blend base ---")
for lgb_w in [0.3, 0.5]:
    base_blend = renorm_rows(lgb_w * oof_lgb + (1-lgb_w) * oof_best)
    base_iso, base_iso_models = apply_isotonic_oof(base_blend, y)
    eval_skf_lomo(base_iso, f"({lgb_w:.0%}lgb+{1-lgb_w:.0%}best) -> Iso")
    base_gmm, base_gmm_model, base_gmm_dists = apply_gmm_oof(base_blend, y, n_components=50, alpha=0.5)
    eval_skf_lomo(base_gmm, f"({lgb_w:.0%}lgb+{1-lgb_w:.0%}best) -> GMM")


# ══════════════════════════════════════════════════════════════════════
# Per-class ensemble weight optimization
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  Per-Class Ensemble Weight Optimization")
print("=" * 90)

def optimize_per_class_weights(oof_list, y_true, names):
    """Find optimal per-class weights for macro-mAP."""
    n_models = len(oof_list)
    best_weights = np.ones((n_models, N_CLASSES)) / n_models

    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(float)
        best_ap = -1
        best_w = np.ones(n_models) / n_models

        # Grid search for this class
        if n_models == 2:
            for w0 in np.arange(0, 1.05, 0.05):
                w1 = 1.0 - w0
                blended_c = w0 * oof_list[0][:, c] + w1 * oof_list[1][:, c]
                from sklearn.metrics import average_precision_score
                ap = average_precision_score(y_bin, blended_c)
                if ap > best_ap:
                    best_ap = ap
                    best_w = np.array([w0, w1])
        elif n_models == 3:
            for w0 in np.arange(0, 1.05, 0.1):
                for w1 in np.arange(0, 1.05 - w0, 0.1):
                    w2 = 1.0 - w0 - w1
                    if w2 < 0:
                        continue
                    blended_c = w0*oof_list[0][:,c] + w1*oof_list[1][:,c] + w2*oof_list[2][:,c]
                    from sklearn.metrics import average_precision_score
                    ap = average_precision_score(y_bin, blended_c)
                    if ap > best_ap:
                        best_ap = ap
                        best_w = np.array([w0, w1, w2])

        best_weights[:, c] = best_w
        print(f"  {CLASSES[c]:20s}: AP={best_ap:.3f}  weights={dict(zip(names, best_w.round(2)))}")

    # Build final blend
    final = np.zeros_like(oof_list[0])
    for c in range(N_CLASSES):
        for i in range(n_models):
            final[:, c] += best_weights[i, c] * oof_list[i][:, c]
    return renorm_rows(final), best_weights

# Optimize: E175 best vs LGB
print("\n  2-model per-class weights (best vs lgb):")
oof_pcw2, pcw2 = optimize_per_class_weights([oof_best, oof_lgb], y, ["best", "lgb"])
eval_skf_lomo(oof_pcw2, "Per-class weights (best+lgb)")

# Optimize with isotonic versions
print("\n  2-model per-class weights (iso_best vs iso_lgb):")
oof_iso_lgb, iso_lgb_models = apply_isotonic_oof(oof_lgb, y)
oof_pcw2_iso, _ = optimize_per_class_weights([oof_iso, oof_iso_lgb], y, ["iso_best", "iso_lgb"])
eval_skf_lomo(oof_pcw2_iso, "Per-class weights (iso_best+iso_lgb)")

# 3-model if GBDT available
if oof_gbdt is not None:
    print("\n  3-model per-class weights (best+lgb+gbdt):")
    oof_pcw3, pcw3 = optimize_per_class_weights(
        [oof_best, oof_lgb, renorm_rows(oof_gbdt)], y, ["best", "lgb", "gbdt"])
    eval_skf_lomo(oof_pcw3, "Per-class weights (best+lgb+gbdt)")


# ══════════════════════════════════════════════════════════════════════
# FINAL: Generate test submissions for top configs
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  GENERATING TEST SUBMISSIONS")
print("=" * 90)

# Apply same transforms to test predictions
test_iso = apply_isotonic_test(test_best, iso_models)
test_gmm = apply_gmm_test(test_best, gmm_model, gmm_dists, alpha=0.5)
test_knn = apply_knn(test_best, X_train, y, X_query=X_test, K=15, alpha=0.15)

# Stack combos on test
test_iso_gmm = apply_gmm_test(test_iso, gmm_model, gmm_dists, alpha=0.5)
test_iso_knn = apply_knn(test_iso, X_train, y, X_query=X_test, K=15, alpha=0.15)
test_gmm_knn = apply_knn(test_gmm, X_train, y, X_query=X_test, K=15, alpha=0.15)

# 3-way stacks on test
test_iso_gmm_knn = apply_knn(test_iso_gmm, X_train, y, X_query=X_test, K=15, alpha=0.15)

# Blends
test_iso_gmm_blend = renorm_rows(0.5 * test_iso + 0.5 * test_gmm)

# TTA on best configs
def apply_tta(test_preds, X_test, model_fn, n_aug=10, noise_scale=0.05):
    """Test-time augmentation: add noise, average predictions."""
    rng = np.random.RandomState(42)
    aug_preds = [test_preds.copy()]
    for _ in range(n_aug):
        X_noisy = X_test + rng.randn(*X_test.shape).astype(np.float32) * noise_scale * X_test.std(axis=0, keepdims=True)
        # Can't retrain here, but we can add noise to the predictions themselves
        # (probability-space TTA)
        noise = rng.randn(*test_preds.shape) * 0.01
        aug_preds.append(renorm_rows(np.clip(test_preds + noise, 1e-8, None)))
    return renorm_rows(np.mean(aug_preds, axis=0))

# Save submissions
submissions = {}

# Individual
submissions["e176_raw"] = (test_best, "raw baseline")
submissions["e176_iso"] = (test_iso, "isotonic")
submissions["e176_gmm"] = (test_gmm, "GMM K=50 a=0.5")
submissions["e176_knn"] = (test_knn, "KNN K=15 a=0.15")

# 2-way
submissions["e176_iso_gmm"] = (test_iso_gmm, "iso->gmm")
submissions["e176_iso_knn"] = (test_iso_knn, "iso->knn")
submissions["e176_gmm_knn"] = (test_gmm_knn, "gmm->knn")
submissions["e176_iso_gmm_blend"] = (test_iso_gmm_blend, "50% iso + 50% gmm")

# 3-way
submissions["e176_iso_gmm_knn"] = (test_iso_gmm_knn, "iso->gmm->knn")

# TTA variants
submissions["e176_iso_tta"] = (apply_tta(test_iso, X_test, None), "iso+tta")
submissions["e176_gmm_tta"] = (apply_tta(test_gmm, X_test, None), "gmm+tta")
submissions["e176_iso_gmm_knn_tta"] = (apply_tta(test_iso_gmm_knn, X_test, None), "iso->gmm->knn+tta")

# LP blend if available
if best_lp_config:
    nn, la, ba, lp_tr, lp_te = best_lp_config
    test_lp = renorm_rows((1-ba) * test_best + ba * renorm_rows(lp_te))
    submissions["e176_lp_blend"] = (test_lp, f"LP(nn={nn}) blend@{ba}")
    # LP + iso
    test_lp_iso = apply_isotonic_test(test_lp, iso_models)
    submissions["e176_lp_iso"] = (test_lp_iso, "LP->iso")

print(f"\nSaving {len(submissions)} submissions:")
for name, (preds, desc) in sorted(submissions.items()):
    # Get OOF score for filename (use SKF as reference)
    oof_equiv = oof_best  # default
    skf, _ = compute_map(y, oof_equiv)
    save_submission(renorm_rows(preds), name, cv_map=skf)
    print(f"  {name}: {desc}")

# Save key OOF/test for later use
np.save(ROOT / "oof_e176_iso.npy", oof_iso)
np.save(ROOT / "test_e176_iso.npy", test_iso)
np.save(ROOT / "oof_e176_gmm.npy", oof_gmm)
np.save(ROOT / "test_e176_gmm.npy", test_gmm)
np.save(ROOT / "oof_e176_iso_gmm_knn.npy", oof_3way_a)
np.save(ROOT / "test_e176_iso_gmm_knn.npy", test_iso_gmm_knn)

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.0f}s")
print("=" * 90)
