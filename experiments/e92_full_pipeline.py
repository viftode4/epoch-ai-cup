"""E92: Full Pipeline — Intrinsic Model + Label Shift + NB Physics.

Combines the best of all approaches into one principled pipeline:
  Stage 1: Base model predictions (E90 intrinsic or E79 full)
  Stage 2: BBSE/MLLS per-month label shift correction (from E91)
  Stage 3: GBIF prior tilt (E67-style, uncertainty-gated)
  Stage 4: NB physics correction (E75-style, uncertainty-gated)

Also tests hybrid: E79 base + E90-derived BBSE proportions.
Generates comprehensive submission grid for Kaggle LB validation.
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


def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


# ====================================================================
# BUILD NB PARAMS
# ====================================================================

def build_nb_params(train_df):
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
# BUILD GBIF PRIORS
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
# LABEL SHIFT CORRECTION
# ====================================================================

def build_confusion_matrix(y_true, y_pred_proba, n_classes=N_CLASSES):
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            C[c] = y_pred_proba[mask].mean(axis=0)
    return C


def bbse_estimate(C, mu_test, epsilon=1e-4):
    K = C.shape[0]
    C_reg = C.copy()
    C_reg += epsilon * np.eye(K)
    C_reg = C_reg / C_reg.sum(axis=1, keepdims=True)
    try:
        w = np.linalg.solve(C_reg.T, mu_test)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(C_reg.T, mu_test, rcond=None)
    w = np.maximum(w, 1e-8)
    return w / w.sum()


def mlls_estimate(y_pred_proba, pi_train, n_iter=200, tol=1e-8):
    w = pi_train.copy()
    for _ in range(n_iter):
        w_old = w.copy()
        ratio = w / np.maximum(pi_train, 1e-12)
        weighted = y_pred_proba * ratio[None, :]
        normalizer = weighted.sum(axis=1, keepdims=True)
        q = weighted / np.maximum(normalizer, 1e-12)
        w = q.mean(axis=0)
        w = np.maximum(w, 1e-8)
        w = w / w.sum()
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


# ====================================================================
# FULL PIPELINE
# ====================================================================

def full_pipeline(base_preds, test_months_arr, pi_train, gbif_priors,
                  w_per_month, nb_factors, nb_ok,
                  # Label shift params
                  ls_months="unseen",     # "unseen", "all", or "none"
                  ls_tau=None,            # gate threshold for label shift
                  # GBIF prior params
                  gbif_alpha=None,        # dict of {month: alpha} or None
                  gbif_tau=0.15,          # gate threshold for GBIF
                  # NB physics params
                  nb_gamma=0.30,          # NB exponent
                  nb_tau=0.30,            # NB gate threshold
                  nb_months="unseen",     # "unseen", "all"
                  ):
    """Apply the full post-processing pipeline.

    Order: Label Shift -> GBIF Priors -> NB Physics
    """
    out = base_preds.copy()
    unseen_mask = np.isin(test_months_arr, UNSEEN_MONTHS)

    # --- Stage 1: Label shift correction ---
    if ls_months != "none" and w_per_month:
        margin = top2_margin(out)
        for month, w_m in w_per_month.items():
            if ls_months == "unseen" and month not in UNSEEN_MONTHS:
                continue
            mask_m = test_months_arr == month
            if ls_tau is not None:
                mask_m = mask_m & (margin < ls_tau)
            if mask_m.sum() == 0:
                continue
            ratio = w_m / np.maximum(pi_train, 1e-12)
            out[mask_m] = out[mask_m] * ratio[None, :]
            out[mask_m] = out[mask_m] / out[mask_m].sum(axis=1, keepdims=True)
        out = renorm_rows(out)

    # --- Stage 2: GBIF prior tilt ---
    if gbif_alpha:
        margin = top2_margin(out)
        for month, alpha in gbif_alpha.items():
            mask_m = test_months_arr == month
            gate = mask_m & (margin < gbif_tau)
            if gate.sum() == 0:
                continue
            ratio = (gbif_priors[month] / np.maximum(pi_train, 1e-12)) ** alpha
            out[gate] = out[gate] * ratio
            out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        out = renorm_rows(out)

    # --- Stage 3: NB physics ---
    if nb_gamma > 0:
        margin = top2_margin(out)
        if nb_months == "unseen":
            month_mask = unseen_mask
        else:
            month_mask = np.ones(len(test_months_arr), dtype=bool)
        nb_gate = month_mask & nb_ok & (margin < nb_tau)
        if nb_gate.any():
            out[nb_gate] = out[nb_gate] * (nb_factors[nb_gate] ** nb_gamma)
            out = renorm_rows(out)

    return out


# ====================================================================
print("=" * 70, flush=True)
print("E92: FULL PIPELINE".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
pi_train = counts / counts.sum()

# NB params
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
nb_factors, nb_ok = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

# GBIF priors
gbif_priors = build_gbif_priors(pi_train)

# -- Load base predictions ----------------------------------------------
print("\nLoading base predictions...", flush=True)
test_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))
oof_e79 = np.load(ROOT / "oof_e79.npy")

has_e90 = (ROOT / "test_e90.npy").exists()
if has_e90:
    test_e90 = renorm_rows(np.load(ROOT / "test_e90.npy").astype(float))
    oof_e90 = np.load(ROOT / "oof_e90.npy")
    print(f"  E90 loaded: {test_e90.shape}", flush=True)
else:
    print(f"  WARNING: E90 not found, using E79 for all configs", flush=True)

print(f"  E79 loaded: {test_e79.shape}", flush=True)

# -- Compute label shift proportions ------------------------------------
print("\nComputing label shift proportions...", flush=True)

# Build confusion matrix from the appropriate OOF
# Hybrid approach: use E90 confusion matrix (intrinsic, satisfies P(X|Y))
# even when correcting E79 predictions (the confusion matrix should come
# from the model where the invariance assumption holds best)

if has_e90:
    C_e90 = build_confusion_matrix(y, oof_e90)
    print(f"  Confusion matrix from E90 (intrinsic model)", flush=True)
else:
    C_e90 = build_confusion_matrix(y, oof_e79)
    print(f"  Confusion matrix from E79 (fallback)", flush=True)

# Also build one from E79 for comparison
C_e79 = build_confusion_matrix(y, oof_e79)

test_month_values = sorted(np.unique(test_months))

# Estimate proportions using BOTH confusion matrices
w_bbse = {}  # from E90 confusion matrix
w_mlls = {}
w_bbse_e79 = {}  # from E79 confusion matrix
w_mlls_e79 = {}

for month in test_month_values:
    mask_m = test_months == month
    n_m = mask_m.sum()

    # E90-based estimation (theoretically better: P(X|Y) invariant)
    if has_e90:
        test_preds_m = test_e90[mask_m]
        mu_m = test_preds_m.mean(axis=0)
        w_bbse[month] = bbse_estimate(C_e90, mu_m, epsilon=1e-3)
        w_mlls[month] = mlls_estimate(test_preds_m, pi_train, n_iter=200)

    # E79-based estimation (has weather features, violates P(X|Y))
    test_preds_m_79 = test_e79[mask_m]
    mu_m_79 = test_preds_m_79.mean(axis=0)
    w_bbse_e79[month] = bbse_estimate(C_e79, mu_m_79, epsilon=1e-3)
    w_mlls_e79[month] = mlls_estimate(test_preds_m_79, pi_train, n_iter=200)

    print(f"\n  Month {month} (n={n_m}):", flush=True)
    print(f"    {'Class':15s} {'Train':>8s}", end="", flush=True)
    if has_e90:
        print(f" {'BBSE90':>8s} {'MLLS90':>8s}", end="")
    print(f" {'BBSE79':>8s} {'MLLS79':>8s}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:15s} {pi_train[i]:8.4f}", end="", flush=True)
        if has_e90:
            print(f" {w_bbse[month][i]:8.4f} {w_mlls[month][i]:8.4f}", end="")
        print(f" {w_bbse_e79[month][i]:8.4f} {w_mlls_e79[month][i]:8.4f}")

# ====================================================================
# GENERATE SUBMISSION GRID
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("GENERATING SUBMISSIONS".center(70), flush=True)
print(f"{'='*70}", flush=True)

# GBIF alpha configs (from E54/E67 best)
GBIF_CONFIGS = {
    "no_gbif": None,
    "gbif_e54": {2: 0.22, 5: 0.12, 12: 0.24},
}

submissions = []


def make_sub(label, preds, base):
    """Compute stats and save."""
    unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
    flips = int((base.argmax(1) != preds.argmax(1)).sum())
    unseen_flips = int(((base.argmax(1) != preds.argmax(1)) & unseen_mask).sum())
    shared_flips = flips - unseen_flips
    submissions.append((label, preds, flips, unseen_flips, shared_flips))
    print(f"  {label}: flips={flips} (unseen={unseen_flips}, shared={shared_flips})", flush=True)


# ── Config A: E79 base + E87-style PP (reference) ────────────────
print("\n--- A: E79 + E87 reference (no label shift) ---", flush=True)
for gamma in [0.10, 0.30, 0.50]:
    preds = full_pipeline(
        test_e79, test_months, pi_train, gbif_priors,
        w_per_month={}, nb_factors=nb_factors, nb_ok=nb_ok,
        ls_months="none",
        gbif_alpha={2: 0.22, 5: 0.12, 12: 0.24}, gbif_tau=0.15,
        nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
    )
    make_sub(f"e92_A_e79_e87ref_g{gamma:.2f}", preds, test_e79)

# ── Config B: E79 base + MLLS label shift (E79 confusion) ────────
print("\n--- B: E79 + MLLS (E79 confusion matrix) ---", flush=True)
for ls_tau in [None, 0.30]:
    for gamma in [0.0, 0.30, 0.50]:
        tau_label = "ungated" if ls_tau is None else f"tau{ls_tau:.2f}"
        preds = full_pipeline(
            test_e79, test_months, pi_train, gbif_priors,
            w_per_month={m: w_mlls_e79[m] for m in UNSEEN_MONTHS if m in w_mlls_e79},
            nb_factors=nb_factors, nb_ok=nb_ok,
            ls_months="unseen", ls_tau=ls_tau,
            gbif_alpha=None,  # No GBIF -- label shift replaces it
            nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
        )
        make_sub(f"e92_B_e79_mlls79_{tau_label}_nb{gamma:.2f}", preds, test_e79)

# ── Config C: E79 base + MLLS label shift + GBIF ─────────────────
print("\n--- C: E79 + MLLS + GBIF ---", flush=True)
for gamma in [0.30, 0.50]:
    preds = full_pipeline(
        test_e79, test_months, pi_train, gbif_priors,
        w_per_month={m: w_mlls_e79[m] for m in UNSEEN_MONTHS if m in w_mlls_e79},
        nb_factors=nb_factors, nb_ok=nb_ok,
        ls_months="unseen", ls_tau=0.30,
        gbif_alpha={2: 0.22, 5: 0.12, 12: 0.24}, gbif_tau=0.15,
        nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
    )
    make_sub(f"e92_C_e79_mlls79_gbif_nb{gamma:.2f}", preds, test_e79)

# ── Config D: E79 base + BBSE label shift ─────────────────────────
print("\n--- D: E79 + BBSE ---", flush=True)
for ls_tau in [None, 0.30]:
    for gamma in [0.30, 0.50]:
        tau_label = "ungated" if ls_tau is None else f"tau{ls_tau:.2f}"
        preds = full_pipeline(
            test_e79, test_months, pi_train, gbif_priors,
            w_per_month={m: w_bbse_e79[m] for m in UNSEEN_MONTHS if m in w_bbse_e79},
            nb_factors=nb_factors, nb_ok=nb_ok,
            ls_months="unseen", ls_tau=ls_tau,
            gbif_alpha=None,
            nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
        )
        make_sub(f"e92_D_e79_bbse79_{tau_label}_nb{gamma:.2f}", preds, test_e79)

if has_e90:
    # ── Config E: E90 base + MLLS (E90 confusion, principled) ─────
    print("\n--- E: E90 + MLLS (principled) ---", flush=True)
    for ls_tau in [None, 0.30]:
        for gamma in [0.0, 0.30, 0.50]:
            tau_label = "ungated" if ls_tau is None else f"tau{ls_tau:.2f}"
            preds = full_pipeline(
                test_e90, test_months, pi_train, gbif_priors,
                w_per_month={m: w_mlls[m] for m in UNSEEN_MONTHS if m in w_mlls},
                nb_factors=nb_factors, nb_ok=nb_ok,
                ls_months="unseen", ls_tau=ls_tau,
                gbif_alpha=None,
                nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
            )
            make_sub(f"e92_E_e90_mlls90_{tau_label}_nb{gamma:.2f}", preds, test_e90)

    # ── Config F: HYBRID — E79 base + E90-derived MLLS proportions ──
    # Key idea: E79 has better raw predictions (weather helps ranking),
    # but E90's confusion matrix satisfies P(X|Y) invariance better.
    print("\n--- F: HYBRID (E79 preds + E90 proportions) ---", flush=True)
    for ls_tau in [None, 0.30]:
        for gamma in [0.30, 0.50]:
            tau_label = "ungated" if ls_tau is None else f"tau{ls_tau:.2f}"
            preds = full_pipeline(
                test_e79, test_months, pi_train, gbif_priors,
                w_per_month={m: w_mlls[m] for m in UNSEEN_MONTHS if m in w_mlls},
                nb_factors=nb_factors, nb_ok=nb_ok,
                ls_months="unseen", ls_tau=ls_tau,
                gbif_alpha=None,
                nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
            )
            make_sub(f"e92_F_e79_mlls90_{tau_label}_nb{gamma:.2f}", preds, test_e79)

    # ── Config G: HYBRID + GBIF ──────────────────────────────────────
    print("\n--- G: HYBRID + GBIF ---", flush=True)
    for gamma in [0.30, 0.50]:
        preds = full_pipeline(
            test_e79, test_months, pi_train, gbif_priors,
            w_per_month={m: w_mlls[m] for m in UNSEEN_MONTHS if m in w_mlls},
            nb_factors=nb_factors, nb_ok=nb_ok,
            ls_months="unseen", ls_tau=0.30,
            gbif_alpha={2: 0.22, 5: 0.12, 12: 0.24}, gbif_tau=0.15,
            nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
        )
        make_sub(f"e92_G_e79_mlls90_gbif_nb{gamma:.2f}", preds, test_e79)

    # ── Config H: All-month label shift (including shared) ───────────
    print("\n--- H: All-month label shift ---", flush=True)
    for gamma in [0.30]:
        preds = full_pipeline(
            test_e79, test_months, pi_train, gbif_priors,
            w_per_month=w_mlls,  # All months
            nb_factors=nb_factors, nb_ok=nb_ok,
            ls_months="all", ls_tau=0.30,
            gbif_alpha=None,
            nb_gamma=gamma, nb_tau=0.30, nb_months="unseen",
        )
        make_sub(f"e92_H_e79_mlls_allmonth_nb{gamma:.2f}", preds, test_e79)


# ====================================================================
# SAVE CURATED SUBMISSIONS
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("SAVING CURATED SUBMISSIONS".center(70), flush=True)
print(f"{'='*70}", flush=True)

# Priority submissions for Kaggle upload
priority_labels = [
    # Reference: E87-style (no label shift)
    "e92_A_e79_e87ref_g0.30",
    "e92_A_e79_e87ref_g0.50",
    # MLLS on E79 (most promising -- data-driven priors)
    "e92_B_e79_mlls79_ungated_nb0.30",
    "e92_B_e79_mlls79_tau0.30_nb0.30",
    "e92_B_e79_mlls79_ungated_nb0.50",
    # MLLS + GBIF combined
    "e92_C_e79_mlls79_gbif_nb0.30",
    # BBSE alternatives
    "e92_D_e79_bbse79_ungated_nb0.30",
    "e92_D_e79_bbse79_tau0.30_nb0.30",
]

# Add E90-based if available
if has_e90:
    priority_labels += [
        "e92_E_e90_mlls90_ungated_nb0.30",
        "e92_E_e90_mlls90_tau0.30_nb0.30",
        "e92_F_e79_mlls90_ungated_nb0.30",
        "e92_F_e79_mlls90_tau0.30_nb0.30",
        "e92_G_e79_mlls90_gbif_nb0.30",
        "e92_H_e79_mlls_allmonth_nb0.30",
    ]

saved = 0
for label, preds, flips, unseen_flips, shared_flips in submissions:
    if label in priority_labels:
        save_submission(preds, label)
        saved += 1

print(f"\n  Saved {saved} priority submissions", flush=True)

# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*70}", flush=True)
print("SUMMARY AND UPLOAD PRIORITY".center(70), flush=True)
print(f"{'='*70}", flush=True)

print(f"""
  Submission grid: {len(submissions)} total configs, {saved} saved.

  Upload priority (test on Kaggle LB):

  Tier 1 — Most likely to improve (data-driven prior correction):
    1. e92_B_e79_mlls79_ungated_nb0.30  — MLLS replaces GBIF priors
    2. e92_B_e79_mlls79_tau0.30_nb0.30  — Same but gated
    3. e92_C_e79_mlls79_gbif_nb0.30     — MLLS + GBIF combined

  Tier 2 — Principled (if E90 available):
    4. e92_F_e79_mlls90_ungated_nb0.30  — HYBRID: E79 preds + E90 proportions
    5. e92_G_e79_mlls90_gbif_nb0.30     — HYBRID + GBIF
    6. e92_E_e90_mlls90_ungated_nb0.30  — Pure intrinsic model + MLLS

  Tier 3 — BBSE alternatives:
    7. e92_D_e79_bbse79_ungated_nb0.30

  Reference:
    - e92_A_e79_e87ref_g0.30            — E87-style (no label shift)
    - e92_A_e79_e87ref_g0.50            — E87-style stronger gamma

  Hypothesis:
    - If MLLS works, Tier 1 submissions should beat E79 raw (0.59 LB)
    - If HYBRID works, E79 base + E90 proportions is theoretically best
    - BBSE may be unstable for small months (Feb=176, Dec=133)
""", flush=True)

print("Done.", flush=True)
