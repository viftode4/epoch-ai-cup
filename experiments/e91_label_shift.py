"""E91: BBSE/MLLS Label Shift Correction.

Estimate test class proportions per month using confusion-matrix methods,
then reweight predictions. Uses E90 intrinsic-feature OOF to build the
confusion matrix (satisfies P(X|Y) invariance assumption).

Theory:
  BBSE (Black Box Shift Estimation, Lipton et al. 2018):
    Given confusion matrix C from OOF and avg test predictions mu,
    solve C^T * w = mu to recover test class proportions w.

  MLLS (Maximum Likelihood Label Shift, Alexandari et al. 2020):
    EM-based alternative. E-step: reweight using current w estimate.
    M-step: update w to maximize likelihood. More robust for small samples.

Validation:
  LOMO: for each held-out month, pretend it's "test" and estimate its
  class proportions. Compare estimated vs actual (ground truth available).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42

UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)


def renorm_rows(pred):
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred):
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


# ====================================================================
# BBSE: Black Box Shift Estimation
# ====================================================================

def build_confusion_matrix(y_true, y_pred_proba, n_classes=N_CLASSES):
    """Build soft confusion matrix C[i,j] = P(predict j | true class i).

    Uses soft predictions (probabilities) rather than hard argmax to get
    a smoother, more informative confusion matrix.
    """
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            C[c] = y_pred_proba[mask].mean(axis=0)
    return C


def bbse_estimate(C, mu_test, epsilon=1e-4):
    """BBSE: solve C^T * w = mu_test for class proportions w.

    Args:
        C: confusion matrix [K, K], C[i,j] = P(predict j | true i)
        mu_test: average prediction vector on test data [K]
        epsilon: regularization for diagonal

    Returns:
        w: estimated class proportions [K], clipped non-negative, normalized
    """
    K = C.shape[0]
    # Regularize: add epsilon to diagonal for stability
    C_reg = C.copy()
    C_reg += epsilon * np.eye(K)
    # Renormalize rows
    C_reg = C_reg / C_reg.sum(axis=1, keepdims=True)

    # Solve C^T * w = mu_test
    try:
        w = np.linalg.solve(C_reg.T, mu_test)
    except np.linalg.LinAlgError:
        # Fall back to least-squares if singular
        w, _, _, _ = np.linalg.lstsq(C_reg.T, mu_test, rcond=None)

    # Clip and normalize
    w = np.maximum(w, 1e-8)
    w = w / w.sum()
    return w


def mlls_estimate(y_pred_proba, pi_train, n_iter=100, tol=1e-8):
    """MLLS: EM-based label shift estimation (Alexandari et al. 2020).

    Args:
        y_pred_proba: test predictions [N, K]
        pi_train: training class proportions [K]
        n_iter: max EM iterations
        tol: convergence tolerance

    Returns:
        w: estimated class proportions [K]
    """
    K = y_pred_proba.shape[1]
    N = y_pred_proba.shape[0]

    # Initialize with training proportions
    w = pi_train.copy()

    for it in range(n_iter):
        w_old = w.copy()

        # E-step: compute importance weights
        # ratio[c] = w[c] / pi_train[c]
        ratio = w / np.maximum(pi_train, 1e-12)

        # Reweight predictions: q_{i,c} = p_{i,c} * ratio[c] / sum_k(p_{i,k} * ratio[k])
        weighted = y_pred_proba * ratio[None, :]
        normalizer = weighted.sum(axis=1, keepdims=True)
        q = weighted / np.maximum(normalizer, 1e-12)

        # M-step: update w
        w = q.mean(axis=0)

        # Normalize
        w = np.maximum(w, 1e-8)
        w = w / w.sum()

        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def apply_label_shift_correction(predictions, test_months_arr, w_per_month,
                                 pi_train, tau=None):
    """Apply label shift correction to predictions.

    For each test sample in month m:
      q_{i,c} = p_{i,c} * (w_m[c] / pi_train[c])
      then renormalize.

    Optionally gate by uncertainty (only correct when margin < tau).
    """
    out = predictions.copy()
    for month, w_m in w_per_month.items():
        mask_m = test_months_arr == month
        if tau is not None:
            margin = top2_margin(out)
            mask_m = mask_m & (margin < tau)
        if mask_m.sum() == 0:
            continue

        ratio = w_m / np.maximum(pi_train, 1e-12)
        out[mask_m] = out[mask_m] * ratio[None, :]
        out[mask_m] = out[mask_m] / out[mask_m].sum(axis=1, keepdims=True)

    return renorm_rows(out)


# ====================================================================
# NB PHYSICS CORRECTION (from E87)
# ====================================================================

def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_nb_params(train_df):
    """E75-style NB: size + speed + alt_mid + alt_range."""
    LAPLACE = 1.0
    MIN_SIGMA = 0.50
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    size_idx = (
        train_df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}

    K, S = len(CLASSES), len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu, sig = {}, {}
    for feat, x in feats.items():
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        gm, gs = float(np.nanmean(x)), float(np.nanstd(x))
        if not np.isfinite(gs) or gs < MIN_SIGMA:
            gs = MIN_SIGMA
        for c in range(K):
            xc = x[y == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat], sig[feat] = mu_f, sig_f

    return size_levels, log_p_size, mu, sig


def compute_nb_factors(df, size_levels, log_p_size, mu, sig):
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
    loglik = log_p_size[:, size_idx].T
    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


# ====================================================================
# GBIF PRIORS (from E87)
# ====================================================================

def build_gbif_priors(p_train):
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(len(CLASSES))
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals
    priors = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


# ====================================================================
print("=" * 70, flush=True)
print("E91: BBSE/MLLS LABEL SHIFT CORRECTION".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unique_train_months = sorted(np.unique(train_months))

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
pi_train = counts / counts.sum()

print(f"  Train: {len(y)} samples, months: {unique_train_months}", flush=True)
print(f"  Test: {len(test_months)} samples", flush=True)
print(f"  Train priors: {dict(zip(CLASSES, np.round(pi_train, 4)))}", flush=True)

# -- Load E90 predictions ----------------------------------------------
# Try to load E90 artifacts; fall back to E79 if not available
try:
    oof_e90 = np.load(ROOT / "oof_e90.npy")
    test_e90 = np.load(ROOT / "test_e90.npy")
    oof_e90_lomo = np.load(ROOT / "oof_e90_lomo.npy")
    base_name = "E90 (intrinsic_25)"
    print(f"\n  Loaded E90 intrinsic model predictions", flush=True)
except FileNotFoundError:
    print("\n  WARNING: E90 artifacts not found, falling back to E79", flush=True)
    oof_e90 = np.load(ROOT / "oof_e79.npy")
    test_e90 = np.load(ROOT / "test_e79.npy")
    oof_e90_lomo = oof_e90  # E79 only has SKF OOF
    base_name = "E79 (fallback)"

# Also load E79 for comparison
oof_e79 = np.load(ROOT / "oof_e79.npy")
test_e79 = np.load(ROOT / "test_e79.npy")

print(f"  OOF shape: {oof_e90.shape}, Test shape: {test_e90.shape}", flush=True)

# -- NB params (for pipeline stage 3) ----------------------------------
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors_test, ok_test = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

# -- GBIF priors -------------------------------------------------------
gbif_priors = build_gbif_priors(pi_train)

# ====================================================================
# PART 1: LOMO VALIDATION OF BBSE/MLLS
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("PART 1: LOMO VALIDATION".center(70), flush=True)
print(f"{'='*70}", flush=True)

print("""
  For each train month m (held out):
    1. Build confusion matrix from OOF of remaining months
    2. Apply BBSE/MLLS to estimate class proportions of month m
    3. Compare estimated proportions to actual (ground truth)
    4. Measure mAP after reweighting
""", flush=True)

lomo_results = {"raw": [], "bbse": [], "mlls": [], "bbse_gated": [], "mlls_gated": []}

for month_held in unique_train_months:
    print(f"\n  --- Hold-out month {month_held} ---", flush=True)
    va_idx = np.where(train_months == month_held)[0]
    tr_idx = np.where(train_months != month_held)[0]

    y_tr, y_va = y[tr_idx], y[va_idx]
    oof_tr = oof_e90_lomo[tr_idx]
    oof_va = oof_e90_lomo[va_idx]

    # Actual class proportions for held-out month
    actual_counts = np.bincount(y_va, minlength=N_CLASSES).astype(float)
    actual_props = actual_counts / actual_counts.sum()

    # Training proportions (from remaining months)
    tr_counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    pi_tr = tr_counts / tr_counts.sum()

    # Raw LOMO score (no correction)
    raw_map, raw_per = compute_map(y_va, oof_va)
    lomo_results["raw"].append(raw_map)

    # Build confusion matrix from training OOF
    C = build_confusion_matrix(y_tr, oof_tr)

    # Average prediction on held-out month
    mu_va = oof_va.mean(axis=0)

    # BBSE estimation
    w_bbse = bbse_estimate(C, mu_va, epsilon=1e-3)

    # MLLS estimation
    w_mlls = mlls_estimate(oof_va, pi_tr, n_iter=200)

    # Print comparison
    print(f"    {'Class':15s} {'Actual':>8s} {'Train':>8s} {'BBSE':>8s} {'MLLS':>8s}", flush=True)
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:15s} {actual_props[i]:8.4f} {pi_tr[i]:8.4f} "
              f"{w_bbse[i]:8.4f} {w_mlls[i]:8.4f}", flush=True)

    # Compute L1 error of proportion estimates
    l1_train = np.abs(pi_tr - actual_props).sum()
    l1_bbse = np.abs(w_bbse - actual_props).sum()
    l1_mlls = np.abs(w_mlls - actual_props).sum()
    print(f"    L1 error: train={l1_train:.4f} BBSE={l1_bbse:.4f} MLLS={l1_mlls:.4f}", flush=True)

    # Apply corrections and measure mAP
    months_va = train_months[va_idx]

    # BBSE correction
    w_per_month_bbse = {month_held: w_bbse}
    corrected_bbse = apply_label_shift_correction(
        oof_va, months_va, w_per_month_bbse, pi_tr)
    bbse_map, _ = compute_map(y_va, corrected_bbse)
    lomo_results["bbse"].append(bbse_map)

    # MLLS correction
    w_per_month_mlls = {month_held: w_mlls}
    corrected_mlls = apply_label_shift_correction(
        oof_va, months_va, w_per_month_mlls, pi_tr)
    mlls_map, _ = compute_map(y_va, corrected_mlls)
    lomo_results["mlls"].append(mlls_map)

    # Gated BBSE (tau=0.30)
    corrected_bbse_g = apply_label_shift_correction(
        oof_va, months_va, w_per_month_bbse, pi_tr, tau=0.30)
    bbse_g_map, _ = compute_map(y_va, corrected_bbse_g)
    lomo_results["bbse_gated"].append(bbse_g_map)

    # Gated MLLS (tau=0.30)
    corrected_mlls_g = apply_label_shift_correction(
        oof_va, months_va, w_per_month_mlls, pi_tr, tau=0.30)
    mlls_g_map, _ = compute_map(y_va, corrected_mlls_g)
    lomo_results["mlls_gated"].append(mlls_g_map)

    print(f"    mAP: raw={raw_map:.4f} BBSE={bbse_map:.4f} MLLS={mlls_map:.4f} "
          f"BBSE_g={bbse_g_map:.4f} MLLS_g={mlls_g_map:.4f}", flush=True)

# LOMO summary
print(f"\n  {'='*60}", flush=True)
print(f"  LOMO SUMMARY", flush=True)
print(f"  {'='*60}", flush=True)
for method, scores in lomo_results.items():
    avg = np.mean(scores)
    print(f"    {method:15s}: {avg:.4f}  (per-month: {[f'{s:.4f}' for s in scores]})", flush=True)

base_lomo_avg = np.mean(lomo_results["raw"])
for method, scores in lomo_results.items():
    if method != "raw":
        delta = np.mean(scores) - base_lomo_avg
        print(f"    {method:15s} delta: {delta:+.4f}", flush=True)


# ====================================================================
# PART 2: ESTIMATE TEST CLASS PROPORTIONS
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("PART 2: TEST CLASS PROPORTION ESTIMATION".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Build confusion matrix from FULL training OOF
C_full = build_confusion_matrix(y, oof_e90)
print(f"\n  Confusion matrix (soft, from SKF OOF):", flush=True)
print(f"  {'':15s}", end="", flush=True)
for cls in CLASSES:
    print(f" {cls[:6]:>7s}", end="")
print()
for i, cls in enumerate(CLASSES):
    print(f"  {cls:15s}", end="", flush=True)
    for j in range(N_CLASSES):
        print(f" {C_full[i,j]:7.3f}", end="")
    print()

# Condition number check
cond = np.linalg.cond(C_full)
print(f"\n  Condition number of C: {cond:.1f}", flush=True)
if cond > 100:
    print(f"  WARNING: Ill-conditioned! BBSE may be unreliable.", flush=True)

# Estimate per-month test proportions
test_month_values = sorted(np.unique(test_months))
print(f"\n  Test months: {test_month_values}", flush=True)

w_bbse_per_month = {}
w_mlls_per_month = {}

for month in test_month_values:
    mask_m = test_months == month
    n_m = mask_m.sum()
    test_preds_m = test_e90[mask_m]
    mu_m = test_preds_m.mean(axis=0)

    # BBSE
    w_bbse = bbse_estimate(C_full, mu_m, epsilon=1e-3)
    w_bbse_per_month[month] = w_bbse

    # MLLS
    w_mlls = mlls_estimate(test_preds_m, pi_train, n_iter=200)
    w_mlls_per_month[month] = w_mlls

    print(f"\n  Month {month} (n={n_m}):", flush=True)
    print(f"    {'Class':15s} {'Train':>8s} {'BBSE':>8s} {'MLLS':>8s} {'GBIF':>8s}", flush=True)
    for i, cls in enumerate(CLASSES):
        gbif_p = gbif_priors.get(month, pi_train)[i] if month in gbif_priors else pi_train[i]
        print(f"    {cls:15s} {pi_train[i]:8.4f} {w_bbse[i]:8.4f} "
              f"{w_mlls[i]:8.4f} {gbif_p:8.4f}", flush=True)


# ====================================================================
# PART 3: APPLY CORRECTIONS AND GENERATE SUBMISSIONS
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("PART 3: GENERATE SUBMISSIONS".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Strategy matrix: base model x correction method x gating
bases = {
    "e90": test_e90,
    "e79": test_e79,
}

corrections = {
    "bbse": w_bbse_per_month,
    "mlls": w_mlls_per_month,
}

taus = {
    "ungated": None,
    "tau030": 0.30,
    "tau050": 0.50,
}

# Also define a "unseen-only" variant that only corrects months 2, 5, 12
corrections_unseen = {}
for cname, w_dict in corrections.items():
    unseen_only = {m: w for m, w in w_dict.items() if m in UNSEEN_MONTHS}
    corrections_unseen[f"{cname}_unseen"] = unseen_only

all_corrections = {**corrections, **corrections_unseen}

# Track best submissions
best_submissions = []

for base_name_key, test_pred in bases.items():
    for corr_name, w_dict in all_corrections.items():
        for tau_name, tau_val in taus.items():
            corrected = apply_label_shift_correction(
                test_pred, test_months, w_dict, pi_train, tau=tau_val
            )

            # Count flips
            flips = int((test_pred.argmax(1) != corrected.argmax(1)).sum())
            unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
            unseen_flips = int(
                ((test_pred.argmax(1) != corrected.argmax(1)) & unseen_mask).sum()
            )

            label = f"e91_{base_name_key}_{corr_name}_{tau_name}"
            print(f"  {label}: {flips} flips ({unseen_flips} unseen)", flush=True)

            best_submissions.append((label, corrected, flips, unseen_flips))

# ====================================================================
# PART 4: COMBINED PIPELINE (Label Shift + NB Physics)
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("PART 4: LABEL SHIFT + NB PHYSICS PIPELINE".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Best NB params from E86/E87 LOMO validation
NB_GAMMAS = [0.10, 0.30, 0.50]
NB_TAU = 0.30

for base_name_key, test_pred in bases.items():
    for corr_name in ["bbse_unseen", "mlls_unseen"]:
        w_dict = all_corrections[corr_name]

        for ls_tau_name, ls_tau in [("ungated", None), ("tau030", 0.30)]:
            # Stage 1: Label shift correction
            after_ls = apply_label_shift_correction(
                test_pred, test_months, w_dict, pi_train, tau=ls_tau
            )

            for gamma in NB_GAMMAS:
                # Stage 2: NB physics (unseen months, gated)
                out = after_ls.copy()
                unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
                margin = top2_margin(out)
                nb_gate = unseen_mask & ok_test & (margin < NB_TAU)

                if nb_gate.any():
                    out[nb_gate] = out[nb_gate] * (factors_test[nb_gate] ** gamma)
                    out = renorm_rows(out)

                flips = int((test_pred.argmax(1) != out.argmax(1)).sum())
                unseen_flips = int(
                    ((test_pred.argmax(1) != out.argmax(1)) & unseen_mask).sum()
                )

                label = f"e91_{base_name_key}_{corr_name}_{ls_tau_name}_nb{gamma:.2f}"
                print(f"  {label}: {flips} flips ({unseen_flips} unseen)", flush=True)
                best_submissions.append((label, out, flips, unseen_flips))


# ====================================================================
# PART 5: SAVE TOP SUBMISSIONS
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("PART 5: SAVING SUBMISSIONS".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Save a curated selection (not all combos)
save_labels = [
    # Pure label shift on E90
    "e91_e90_bbse_unseen_ungated",
    "e91_e90_bbse_unseen_tau030",
    "e91_e90_mlls_unseen_ungated",
    "e91_e90_mlls_unseen_tau030",
    # Pure label shift on E79
    "e91_e79_bbse_unseen_ungated",
    "e91_e79_bbse_unseen_tau030",
    "e91_e79_mlls_unseen_ungated",
    "e91_e79_mlls_unseen_tau030",
    # Combined: label shift + NB on E79 (best base per LB)
    "e91_e79_bbse_unseen_ungated_nb0.30",
    "e91_e79_bbse_unseen_tau030_nb0.30",
    "e91_e79_mlls_unseen_ungated_nb0.30",
    "e91_e79_mlls_unseen_tau030_nb0.30",
    "e91_e79_bbse_unseen_ungated_nb0.50",
    "e91_e79_mlls_unseen_ungated_nb0.50",
    # Combined on E90
    "e91_e90_bbse_unseen_ungated_nb0.30",
    "e91_e90_mlls_unseen_ungated_nb0.30",
]

saved = 0
for label, preds, flips, unseen_flips in best_submissions:
    if label in save_labels:
        save_submission(preds, label)
        saved += 1

print(f"\n  Saved {saved} submissions", flush=True)

# Save estimated proportions for E92 (individual files for reliable loading)
np.save(ROOT / "e91_pi_train.npy", pi_train)
np.save(ROOT / "e91_confusion_matrix.npy", C_full)
for month in test_month_values:
    np.save(ROOT / f"e91_bbse_m{month}.npy", w_bbse_per_month[month])
    np.save(ROOT / f"e91_mlls_m{month}.npy", w_mlls_per_month[month])
print(f"  Saved proportion estimates for E92", flush=True)


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("SUMMARY".center(70), flush=True)
print(f"{'='*70}", flush=True)

print(f"""
  LOMO Validation:
    Raw baseline:    {np.mean(lomo_results['raw']):.4f}
    BBSE:            {np.mean(lomo_results['bbse']):.4f}  ({np.mean(lomo_results['bbse']) - base_lomo_avg:+.4f})
    MLLS:            {np.mean(lomo_results['mlls']):.4f}  ({np.mean(lomo_results['mlls']) - base_lomo_avg:+.4f})
    BBSE gated:      {np.mean(lomo_results['bbse_gated']):.4f}  ({np.mean(lomo_results['bbse_gated']) - base_lomo_avg:+.4f})
    MLLS gated:      {np.mean(lomo_results['mlls_gated']):.4f}  ({np.mean(lomo_results['mlls_gated']) - base_lomo_avg:+.4f})

  Key outputs:
    - Per-month test proportions (BBSE + MLLS)
    - {saved} curated submissions
    - Proportion estimates saved for E92

  Upload priority:
    1. e91_e79_mlls_unseen_tau030_nb0.30  (MLLS + NB on proven E79 base)
    2. e91_e79_bbse_unseen_tau030_nb0.30  (BBSE + NB on E79)
    3. e91_e79_mlls_unseen_ungated_nb0.50 (stronger NB)
    4. e91_e90_mlls_unseen_tau030         (pure label shift on intrinsic model)
""", flush=True)

print("Done.", flush=True)
