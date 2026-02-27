"""E87: Apply LOMO-validated PP strategies to E79 test predictions.

LOMO validation (E86) proved:
  - gamma=0.30-0.50 >> gamma=0.10 (current E75)  [+0.012 vs +0.002 LOMO]
  - Gated NB > Ungated NB
  - Pure NB is catastrophic
  - GBIF alone is weak but helps month 1 (proxy for Feb)

Best validated strategy:
  GBIF priors (alpha=0.22, tau=0.15) + NB (gamma=0.30-0.50, tau=0.30)

This script generates the FINAL validated submissions for Kaggle upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent

UNSEEN_MONTHS = (2, 5, 12)
LAPLACE = 1.0
MIN_SIGMA = 0.50


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


def build_nb_params(train_df):
    """E75-style NB: size + speed + alt_mid + alt_range."""
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
print("=" * 70, flush=True)
print("E87: VALIDATED PP ON E79 BASE".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data ---------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# NB params
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors, ok_feats = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

# Load E79 base predictions
base_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))
print(f"  E79 base: {base_e79.shape}", flush=True)
print(f"  Unseen: {unseen_mask.sum()} ({100*unseen_mask.mean():.1f}%)", flush=True)

# -- GBIF prior config (fixed, E54/E67 best) --------------------------
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15


def apply_pp(base, gamma, tau_nb, label):
    """Apply GBIF + NB post-processing pipeline."""
    out = base.copy()

    # Stage 1: GBIF priors (unseen months only)
    margin = top2_margin(out)
    changed_gbif = 0
    for month, alpha in BASE_ALPHA.items():
        mask_m = test_months == month
        gate = mask_m & (margin < TAU_PRIOR)
        if gate.any():
            ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
            out[gate] = out[gate] * ratio
            out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
            changed_gbif += int(gate.sum())
    out = renorm_rows(out)

    # Stage 2: NB physics (unseen months, margin-gated)
    margin2 = top2_margin(out)
    nb_gate = unseen_mask & ok_feats & (margin2 < tau_nb)
    changed_nb = int(nb_gate.sum())
    if nb_gate.any():
        out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** gamma)
        out = renorm_rows(out)

    # Stats
    flips = int(((base.argmax(1) != out.argmax(1)) & unseen_mask).sum())
    top_pred = out.argmax(axis=1)
    base_pred = base.argmax(axis=1)

    print(f"\n  {label}:", flush=True)
    print(f"    GBIF changed: {changed_gbif}", flush=True)
    print(f"    NB gated: {changed_nb}, flips: {flips}", flush=True)

    # Class count changes
    for i, cls in enumerate(CLASSES):
        diff = int((top_pred == i).sum()) - int((base_pred == i).sum())
        if abs(diff) >= 3:
            print(f"    {cls}: {diff:+d}", flush=True)

    save_submission(out, label, cv_map=None)
    return out


# ====================================================================
#  VALIDATED CONFIGURATIONS (from E86 LOMO)
# ====================================================================
print("\n--- LOMO-validated configurations ---", flush=True)

# 1. Best validated: gamma=0.50, tau=0.30 (LOMO +0.0126)
apply_pp(base_e79, gamma=0.50, tau_nb=0.30,
         label="e87_v1_g050_t030")

# 2. Second best: gamma=0.30, tau=0.30 (LOMO +0.0115)
apply_pp(base_e79, gamma=0.30, tau_nb=0.30,
         label="e87_v2_g030_t030")

# 3. Third best: gamma=0.30, tau=0.50 (LOMO +0.0115, wider gate)
apply_pp(base_e79, gamma=0.30, tau_nb=0.50,
         label="e87_v3_g030_t050")

# 4. Intermediate: gamma=0.20, tau=0.30 (LOMO +0.0054)
apply_pp(base_e79, gamma=0.20, tau_nb=0.30,
         label="e87_v4_g020_t030")

# 5. Current E75 config for reference: gamma=0.10, tau=0.30 (LOMO +0.0020)
apply_pp(base_e79, gamma=0.10, tau_nb=0.30,
         label="e87_v5_g010_t030_e75config")

# ====================================================================
#  ALSO: Test with STRONGER GBIF for month 2 (Jan proxy showed +0.05)
# ====================================================================
print("\n--- Stronger GBIF for February (Jan proxy helped) ---", flush=True)

# Save originals with boosted Feb GBIF
for gbif_feb_scale in [1.5, 2.0]:
    for gamma in [0.30, 0.50]:
        out = base_e79.copy()

        # Stage 1: GBIF with boosted Feb alpha
        scaled_alpha = BASE_ALPHA.copy()
        scaled_alpha[2] = BASE_ALPHA[2] * gbif_feb_scale  # Boost Feb
        margin = top2_margin(out)
        for month, alpha in scaled_alpha.items():
            mask_m = test_months == month
            gate = mask_m & (margin < TAU_PRIOR)
            if gate.any():
                ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
                out[gate] = out[gate] * ratio
                out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        out = renorm_rows(out)

        # Stage 2: NB
        margin2 = top2_margin(out)
        nb_gate = unseen_mask & ok_feats & (margin2 < 0.30)
        if nb_gate.any():
            out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** gamma)
            out = renorm_rows(out)

        flips = int(((base_e79.argmax(1) != out.argmax(1)) & unseen_mask).sum())
        label = f"e87_gbifFeb{gbif_feb_scale:.1f}_g{gamma:.2f}"
        print(f"  {label}: flips={flips}", flush=True)
        save_submission(out, label, cv_map=None)

# ====================================================================
#  SUMMARY
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  UPLOAD PRIORITY (LOMO-validated)".center(70), flush=True)
print("=" * 70, flush=True)
print("""
  1. e87_v1_g050_t030 -- LOMO winner (+0.0126). Strongest correction.
  2. e87_v2_g030_t030 -- 2nd best (+0.0115). More conservative.
  3. e87_v5_g010_t030_e75config -- Current E75 config on E79 base.
     (If this = 0.59 again, confirms PP is redundant for E79)
  4. e87_gbifFeb2.0_g0.30 -- Boosted Feb GBIF. Jan proxy helped +0.05.
  5. e87_v4_g020_t030 -- Intermediate gamma for bracketing.

  Reasoning:
  - #1 and #2 test VALIDATED stronger gamma (the key finding)
  - #3 tells us if E75 PP helps E79 at all
  - #4 tests boosted Feb GBIF (validated on month 1 proxy)
  - #5 provides interpolation data between #2 and #3
""", flush=True)

print("Done.", flush=True)
