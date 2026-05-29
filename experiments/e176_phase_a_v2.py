"""E176 Phase A v2: Cross-Validated Post-Processing + Submission Generation.

From Phase A v1: A6 (Isotonic) + A1 (Power Transform) + A5 (KNN) are winners.
This script:
  1. Cross-validated isotonic calibration (honest evaluation)
  2. Cross-validated power transform (honest evaluation)
  3. Proper test submissions with all winning combos
  4. NB PP as final layer on test
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

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

print("=" * 70)
print("  E176 Phase A v2: Cross-Validated PP + Submissions")
print("=" * 70)

t0 = time.time()

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# Load E175 predictions
oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))
test_best = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
test_lgb = renorm_rows(np.load(ROOT / "test_e175_lgb.npy").astype(np.float64))

counts_train = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts_train / counts_train.sum()

base_score, base_pc = compute_map(y, oof_best)
print(f"\nBaseline E175 blend: SKF mAP = {base_score:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Cross-validated Isotonic Calibration (A6-CV)
# ══════════════════════════════════════════════════════════════════════

print("\n--- A6-CV: Cross-Validated Isotonic Calibration ---")


def cv_isotonic(oof, y_true, n_splits=5, seed=42):
    """Fit isotonic calibration in CV to get honest OOF estimates."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_cal = oof.copy()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof, y_true)):
        for c in range(N_CLASSES):
            ir = IsotonicRegression(out_of_bounds="clip")
            y_bin = (y_true == c).astype(float)
            ir.fit(oof[tr_idx, c], y_bin[tr_idx])
            oof_cal[va_idx, c] = ir.predict(oof[va_idx, c])

    return renorm_rows(oof_cal)


def fit_isotonic_full(oof, y_true):
    """Fit isotonic on full data, return list of models for test application."""
    models = []
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        y_bin = (y_true == c).astype(float)
        ir.fit(oof[:, c], y_bin)
        models.append(ir)
    return models


def apply_isotonic(preds, models):
    """Apply pre-fitted isotonic models."""
    out = preds.copy()
    for c, ir in enumerate(models):
        out[:, c] = ir.predict(preds[:, c])
    return renorm_rows(out)


# CV isotonic on best blend
oof_a6cv = cv_isotonic(oof_best, y)
s_a6cv, pc_a6cv = compute_map(y, oof_a6cv)
print(f"  A6-CV (best): SKF mAP = {s_a6cv:.4f} (delta: {s_a6cv - base_score:+.4f})")
for cls in CLASSES:
    d = pc_a6cv[cls] - base_pc[cls]
    if abs(d) > 0.005:
        print(f"    {cls}: {base_pc[cls]:.3f} -> {pc_a6cv[cls]:.3f} ({d:+.3f})")

# CV isotonic on lgb
oof_lgb_score, _ = compute_map(y, oof_lgb)
oof_a6cv_lgb = cv_isotonic(oof_lgb, y)
s_a6cv_lgb, _ = compute_map(y, oof_a6cv_lgb)
print(f"  A6-CV (lgb):  SKF mAP = {s_a6cv_lgb:.4f} (delta from lgb base: {s_a6cv_lgb - oof_lgb_score:+.4f})")

# Multi-seed isotonic for robustness
print("\n  Multi-seed CV isotonic:")
for seed in [42, 123, 456, 789, 2026]:
    oof_ms = cv_isotonic(oof_best, y, seed=seed)
    s_ms, _ = compute_map(y, oof_ms)
    print(f"    seed={seed}: {s_ms:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Cross-validated Power Transform (A1-CV)
# ══════════════════════════════════════════════════════════════════════

print("\n--- A1-CV: Cross-Validated Power Transform ---")


def apply_power_transform(preds, gammas):
    """Apply per-class power transform."""
    out = np.clip(preds, 1e-12, None)
    for c in range(N_CLASSES):
        out[:, c] = out[:, c] ** gammas[c]
    return renorm_rows(out)


def optimize_gammas_on_subset(oof, y_true, idx, gamma_range=(0.3, 3.0), n_grid=25, n_rounds=3):
    """Optimize gammas on a subset of data."""
    best_gammas = np.ones(N_CLASSES)
    best_score, _ = compute_map(y_true[idx], oof[idx])

    for round_i in range(n_rounds):
        improved = False
        for c in range(N_CLASSES):
            trial = best_gammas.copy()
            best_g = best_gammas[c]
            for g in np.linspace(gamma_range[0], gamma_range[1], n_grid):
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


def cv_power_transform(oof, y_true, n_splits=5, seed=42):
    """Cross-validated power transform: optimize gammas on train folds, apply to val."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pt = oof.copy()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof, y_true)):
        gammas = optimize_gammas_on_subset(oof, y_true, tr_idx)
        full_transformed = apply_power_transform(oof, gammas)
        oof_pt[va_idx] = full_transformed[va_idx]
        print(f"    Fold {fold}: gammas = {np.round(gammas, 1)}")

    return renorm_rows(oof_pt)


oof_a1cv = cv_power_transform(oof_best, y)
s_a1cv, pc_a1cv = compute_map(y, oof_a1cv)
print(f"  A1-CV (best): SKF mAP = {s_a1cv:.4f} (delta: {s_a1cv - base_score:+.4f})")


# ══════════════════════════════════════════════════════════════════════
# Combined: A6-CV + A1-CV
# ══════════════════════════════════════════════════════════════════════

print("\n--- Combined: A6-CV then A1-CV ---")

# First isotonic, then power transform
oof_a6cv_then_a1cv = cv_power_transform(oof_a6cv, y)
s_combo, pc_combo = compute_map(y, oof_a6cv_then_a1cv)
print(f"  A6-CV + A1-CV: SKF mAP = {s_combo:.4f} (delta: {s_combo - base_score:+.4f})")

# A5 KNN on top of A6-CV
print("\n--- A6-CV + A5 KNN ---")

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
feat_cols = [c for c in train_feats.columns if c in test_feats.columns]
X_train_raw = np.nan_to_num(train_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw = np.nan_to_num(test_feats[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def apply_knn_blending(oof_preds, X_data, y_data, months, K=15, alpha=0.15):
    """Blend predictions with KNN neighbor label distributions."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    out = oof_preds.copy()
    margin = top2_margin(out)

    n_changed = 0
    for i in range(len(out)):
        if margin[i] > 0.5:
            continue
        dists = np.sqrt(((X_scaled - X_scaled[i]) ** 2).sum(axis=1))
        dists[i] = np.inf
        top_k = np.argsort(dists)[:K]
        top_k_dists = dists[top_k]
        weights = 1.0 / np.maximum(top_k_dists, 1e-8)
        weights /= weights.sum()

        knn_dist = np.zeros(N_CLASSES)
        for j, w in zip(top_k, weights):
            knn_dist[y_data[j]] += w

        out[i] = (1 - alpha) * out[i] + alpha * knn_dist
        n_changed += 1

    print(f"    KNN modified {n_changed} tracks")
    return renorm_rows(out)


oof_a6cv_a5 = apply_knn_blending(oof_a6cv, X_train_raw, y, train_months, K=15, alpha=0.15)
s_a6cv_a5, _ = compute_map(y, oof_a6cv_a5)
print(f"  A6-CV + A5: SKF mAP = {s_a6cv_a5:.4f} (delta: {s_a6cv_a5 - base_score:+.4f})")

oof_a6cv_a5_a1 = cv_power_transform(oof_a6cv_a5, y)
s_a6cv_a5_a1, _ = compute_map(y, oof_a6cv_a5_a1)
print(f"  A6-CV + A5 + A1-CV: SKF mAP = {s_a6cv_a5_a1:.4f} (delta: {s_a6cv_a5_a1 - base_score:+.4f})")


# ══════════════════════════════════════════════════════════════════════
# Weighted Ensemble of Calibrated Predictions
# ══════════════════════════════════════════════════════════════════════

print("\n--- Ensemble Calibrated Models ---")

# Isotonic calibrate both best and lgb, then blend
oof_a6cv_lgb_full = cv_isotonic(oof_lgb, y)

for w_best in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_lgb = 1.0 - w_best
    oof_blend = w_best * oof_a6cv + w_lgb * oof_a6cv_lgb_full
    oof_blend = renorm_rows(oof_blend)
    s_blend, _ = compute_map(y, oof_blend)
    print(f"  {w_best:.0%} best + {w_lgb:.0%} lgb (both A6-CV): {s_blend:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Save Test Submissions
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Saving Test Submissions")
print("=" * 70)


def apply_nb_pp(preds, test_df_in, test_months_in, train_df_in, y_in, gamma=0.10, tau_nb=0.25):
    """Standard NB PP for test submissions."""
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
    gate = np.isin(test_months_in, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)


# Fit isotonic models on full training OOF
iso_models_best = fit_isotonic_full(oof_best, y)
iso_models_lgb = fit_isotonic_full(oof_lgb, y)

# Apply to test
test_iso_best = apply_isotonic(test_best, iso_models_best)
test_iso_lgb = apply_isotonic(test_lgb, iso_models_lgb)

# Optimize gammas on full OOF (for test application)
def optimize_gammas_full(oof, y_true, n_rounds=3):
    """Optimize gammas on full OOF."""
    best_gammas = np.ones(N_CLASSES)
    best_score, _ = compute_map(y_true, oof)
    for round_i in range(n_rounds):
        improved = False
        for c in range(N_CLASSES):
            trial = best_gammas.copy()
            best_g = best_gammas[c]
            for g in np.linspace(0.3, 3.0, 30):
                trial[c] = g
                transformed = apply_power_transform(oof, trial)
                score, _ = compute_map(y_true, transformed)
                if score > best_score + 1e-6:
                    best_score = score
                    best_g = g
                    improved = True
            best_gammas[c] = best_g
        if not improved:
            break
    return best_gammas


# Generate submissions
submissions = {}

# 1. Baseline (raw + PP)
submissions["e176_baseline_raw"] = (base_score, test_best)
submissions["e176_baseline_pp"] = (base_score, apply_nb_pp(test_best, test_df, test_months, train_df, y))

# 2. A6 isotonic on best (raw + PP)
submissions["e176_iso_best_raw"] = (s_a6cv, test_iso_best)
submissions["e176_iso_best_pp"] = (s_a6cv, apply_nb_pp(test_iso_best, test_df, test_months, train_df, y))

# 3. A6+A1: isotonic then power transform
gammas_iso = optimize_gammas_full(oof_a6cv, y)
print(f"  Gammas on A6-CV OOF: {dict(zip([c[:4] for c in CLASSES], np.round(gammas_iso, 2)))}")
test_iso_pt = apply_power_transform(test_iso_best, gammas_iso)
submissions["e176_iso_pt_raw"] = (s_combo, test_iso_pt)
submissions["e176_iso_pt_pp"] = (s_combo, apply_nb_pp(test_iso_pt, test_df, test_months, train_df, y))

# 4. A6 isotonic blend (best + lgb)
test_iso_blend = 0.5 * test_iso_best + 0.5 * test_iso_lgb
test_iso_blend = renorm_rows(test_iso_blend)
oof_iso_blend = 0.5 * oof_a6cv + 0.5 * oof_a6cv_lgb_full
s_iso_blend, _ = compute_map(y, renorm_rows(oof_iso_blend))
submissions["e176_iso_blend_raw"] = (s_iso_blend, test_iso_blend)
submissions["e176_iso_blend_pp"] = (s_iso_blend, apply_nb_pp(test_iso_blend, test_df, test_months, train_df, y))

# 5. A1 power transform only (on raw best)
gammas_raw = optimize_gammas_full(oof_best, y)
test_pt_raw = apply_power_transform(test_best, gammas_raw)
s_a1_raw, _ = compute_map(y, apply_power_transform(oof_best, gammas_raw))
submissions["e176_pt_only_raw"] = (s_a1_raw, test_pt_raw)
submissions["e176_pt_only_pp"] = (s_a1_raw, apply_nb_pp(test_pt_raw, test_df, test_months, train_df, y))

# 6. A6 isotonic on best + A5 KNN on test (with test neighbors from train)
# For test, we can't do KNN the same way (no labels).
# Instead, use train neighbors to influence test predictions.
def apply_knn_test(test_preds, X_test, X_train, y_train, K=15, alpha=0.10):
    """Blend test predictions with KNN from training data."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    out = test_preds.copy()
    margin = top2_margin(out)
    n_changed = 0

    for i in range(len(out)):
        if margin[i] > 0.5:
            continue
        dists = np.sqrt(((X_tr_s - X_te_s[i]) ** 2).sum(axis=1))
        top_k = np.argsort(dists)[:K]
        top_k_dists = dists[top_k]
        weights = 1.0 / np.maximum(top_k_dists, 1e-8)
        weights /= weights.sum()

        knn_dist = np.zeros(N_CLASSES)
        for j, w in zip(top_k, weights):
            knn_dist[y_train[j]] += w

        out[i] = (1 - alpha) * out[i] + alpha * knn_dist
        n_changed += 1

    return renorm_rows(out)


test_iso_knn = apply_knn_test(test_iso_best, X_test_raw, X_train_raw, y, K=15, alpha=0.10)
submissions["e176_iso_knn_raw"] = (s_a6cv_a5, test_iso_knn)
submissions["e176_iso_knn_pp"] = (s_a6cv_a5, apply_nb_pp(test_iso_knn, test_df, test_months, train_df, y))

# Save all
print(f"\nSaving {len(submissions)} submissions:")
for name, (score, preds) in sorted(submissions.items()):
    save_submission(preds, name, score)

# Save OOF predictions for Phase B/C to build on
np.save(ROOT / "oof_e176_iso.npy", oof_a6cv)
np.save(ROOT / "test_e176_iso.npy", test_iso_best)
print(f"\nSaved OOF/test isotonic predictions for later phases")

elapsed = time.time() - t0
print(f"\nPhase A v2 completed in {elapsed:.0f}s")

# Final summary
print("\n" + "=" * 70)
print("  FINAL PHASE A SUMMARY (Cross-Validated)")
print("=" * 70)
print(f"\n  {'Method':<45s} {'CV mAP':>7s} {'Delta':>8s}")
print(f"  {'-'*45} {'-'*7} {'-'*8}")

summary = [
    ("Baseline (E175 blend)", base_score),
    ("A6-CV Isotonic (best)", s_a6cv),
    ("A6-CV Isotonic (lgb)", s_a6cv_lgb),
    ("A1-CV Power Transform", s_a1cv),
    ("A6-CV + A1-CV Combined", s_combo),
    ("A6-CV + A5 KNN", s_a6cv_a5),
    ("A6-CV + A5 + A1-CV", s_a6cv_a5_a1),
    ("A6-CV Blend (50/50 best+lgb)", s_iso_blend),
]
summary.sort(key=lambda x: -x[1])
for name, score in summary:
    delta = score - base_score
    marker = " ***" if delta > 0.01 else (" **" if delta > 0.005 else "")
    print(f"  {name:<45s} {score:>7.4f} {delta:>+8.4f}{marker}")

print("\n" + "=" * 70)
