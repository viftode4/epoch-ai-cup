"""E80: Enhanced NB evidence with RCS features.

Extends E75's NB correction by adding rcs_mean and rcs_range to the
Naive Bayes evidence. RCS separates Clutter (-13.8 dB) from birds
(-24 to -30 dB), and rcs_range captures flapping patterns.

Runs on both E50 (old base) and E79 (pruned base) for comparison.

Pipeline:
  test_e50/e79.npy -> E67 gated GBIF priors (unseen months) -> enhanced NB
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent

UNSEEN_MONTHS = (2, 5, 12)

# Prior stage (fixed best-known from E54/E67)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# NB stage grid
TAU_NB_GRID = [0.25, 0.30]
GAMMA_GRID = [0.06, 0.08, 0.10]

LAPLACE = 1.0
MIN_SIGMA = 0.50


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
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


def apply_gated_ratio_priors(
    preds, months, p_train, priors, alpha_map, tau
):
    out = preds.copy()
    margin = top2_margin(out)
    changed = 0
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


def extract_rcs_from_trajectory(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract rcs_mean and rcs_range from raw EWKB trajectory."""
    rcs_means = np.zeros(len(df), dtype=float)
    rcs_ranges = np.zeros(len(df), dtype=float)
    for i, hex_str in enumerate(df["trajectory"].values):
        try:
            pts = parse_ewkb_4d(hex_str)
            rcs_vals = np.array([p[3] for p in pts])
            rcs_means[i] = np.mean(rcs_vals)
            rcs_ranges[i] = np.ptp(rcs_vals)
        except Exception:
            rcs_means[i] = np.nan
            rcs_ranges[i] = np.nan
    return rcs_means, rcs_ranges


def build_nb_params_enhanced(train_df: pd.DataFrame):
    """Build NB params with size + speed + alt_mid + alt_range + rcs_mean + rcs_range."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    size_idx = (
        train_df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
    )

    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    # Extract RCS from trajectory
    rcs_means, rcs_ranges = extract_rcs_from_trajectory(train_df)

    feats = {
        "speed": speed,
        "alt_mid": alt_mid,
        "alt_range": alt_range,
        "rcs_mean": rcs_means,
        "rcs_range": rcs_ranges,
    }

    K = len(CLASSES)
    S = len(size_levels)

    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu = {}
    sig = {}
    for feat, x in feats.items():
        mu_f = np.zeros(K, dtype=float)
        sig_f = np.zeros(K, dtype=float)
        global_mu = float(np.nanmean(x))
        global_sig = float(np.nanstd(x))
        if not np.isfinite(global_sig) or global_sig < MIN_SIGMA:
            global_sig = MIN_SIGMA
        for c in range(K):
            xc = x[y == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c] = global_mu
                sig_f[c] = global_sig
        mu[feat] = mu_f
        sig[feat] = sig_f

    return size_levels, log_p_size, mu, sig


def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def compute_nb_factors_enhanced(df, size_levels, log_p_size, mu, sig):
    """Compute NB factors including RCS evidence."""
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    rcs_means, rcs_ranges = extract_rcs_from_trajectory(df)

    ok = (np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
          & np.isfinite(rcs_means) & np.isfinite(rcs_ranges))

    loglik = log_p_size[:, size_idx].T  # (N,K)
    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])
        loglik[ok] += log_gaussian(rcs_means[ok], mu["rcs_mean"], sig["rcs_mean"])
        loglik[ok] += log_gaussian(rcs_ranges[ok], mu["rcs_range"], sig["rcs_range"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


# Also keep the original E75 NB for comparison
def build_nb_params_e75(train_df):
    """Original E75 NB params: size + speed + alt_mid + alt_range (no RCS)."""
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


def compute_nb_factors_e75(df, size_levels, log_p_size, mu, sig):
    """Original E75 NB factors (no RCS)."""
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
print("E80 NB + RCS EVIDENCE".center(70), flush=True)
print("=" * 70, flush=True)

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

# Build enhanced NB params (with RCS)
print("\nBuilding enhanced NB params (with RCS)...", flush=True)
size_levels_enh, log_p_size_enh, mu_enh, sig_enh = build_nb_params_enhanced(train_df)
print("  RCS mean per class:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:15s}: mu={mu_enh['rcs_mean'][i]:+.1f} sig={sig_enh['rcs_mean'][i]:.1f}", flush=True)

# Build original E75 NB params (no RCS) for comparison
size_levels_75, log_p_size_75, mu_75, sig_75 = build_nb_params_e75(train_df)

# Compute factors for test
print("\nComputing NB factors for test...", flush=True)
factors_enh, ok_enh = compute_nb_factors_enhanced(test_df, size_levels_enh, log_p_size_enh, mu_enh, sig_enh)
factors_75, ok_75 = compute_nb_factors_e75(test_df, size_levels_75, log_p_size_75, mu_75, sig_75)
print(f"  Enhanced: {ok_enh.sum()}/{len(ok_enh)} samples with valid RCS+alt+speed", flush=True)
print(f"  E75 orig: {ok_75.sum()}/{len(ok_75)} samples with valid alt+speed", flush=True)


def per_month_counts(mask):
    parts = []
    for m in UNSEEN_MONTHS:
        mm = test_months == m
        parts.append(f"m{m}:{int((mask & mm).sum())}")
    return " ".join(parts)


# -- Process each base model ----------------------------------------
bases = {}
base_e50_path = ROOT / "test_e50.npy"
base_e79_path = ROOT / "test_e79.npy"

for name, path in [("E50", base_e50_path), ("E79", base_e79_path)]:
    if path.exists():
        try:
            b = np.load(path, allow_pickle=True)
            b = np.array(b, dtype=float)
            if b.size > 100:
                bases[name] = renorm_rows(b)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}", flush=True)

if not bases:
    print("\nERROR: No base models found. Run E50 or E79 first.", flush=True)
    sys.exit(1)

print(f"\nBase models available: {list(bases.keys())}", flush=True)

for base_name, base in bases.items():
    print(f"\n{'='*60}", flush=True)
    print(f"  Base: {base_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Apply gated priors
    pred0, changed_prior = apply_gated_ratio_priors(
        base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    print(f"  Prior stage: changed={changed_prior}", flush=True)
    margin0 = top2_margin(pred0)

    # -- Enhanced NB (with RCS) --
    print(f"\n  --- Enhanced NB (size+speed+alt+RCS) ---", flush=True)
    for tau_nb in TAU_NB_GRID:
        for gamma in GAMMA_GRID:
            gate = unseen_mask & ok_enh & (margin0 < tau_nb)
            out = pred0.copy()
            out[gate] = out[gate] * (factors_enh[gate] ** gamma)
            out = renorm_rows(out)

            top_flip = int(((pred0.argmax(1) != out.argmax(1)) & unseen_mask).sum())
            print(
                f"    tau_nb={tau_nb:.2f} gamma={gamma:.2f} "
                f"gated={int(gate.sum())} ({per_month_counts(gate)}) "
                f"flips={top_flip}",
                flush=True,
            )
            save_submission(
                out,
                f"e80_{base_name.lower()}_rcs_tau{tau_nb:.2f}_g{gamma:.2f}",
                cv_map=None,
            )

    # -- Original E75 NB (no RCS) for comparison --
    print(f"\n  --- Original NB (size+speed+alt, no RCS) ---", flush=True)
    for tau_nb in [0.30]:
        for gamma in [0.10]:
            gate = unseen_mask & ok_75 & (margin0 < tau_nb)
            out = pred0.copy()
            out[gate] = out[gate] * (factors_75[gate] ** gamma)
            out = renorm_rows(out)
            top_flip = int(((pred0.argmax(1) != out.argmax(1)) & unseen_mask).sum())
            print(
                f"    tau_nb={tau_nb:.2f} gamma={gamma:.2f} "
                f"gated={int(gate.sum())} ({per_month_counts(gate)}) "
                f"flips={top_flip} (E75 baseline)",
                flush=True,
            )
            save_submission(
                out,
                f"e80_{base_name.lower()}_e75nb_tau{tau_nb:.2f}_g{gamma:.2f}",
                cv_map=None,
            )

print("\nDone.", flush=True)
