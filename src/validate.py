"""Importance-weighted mAP validation for post-processing experiments.

APPROACH
--------
Temperature scaling + MLLS label shift estimation + importance-weighted mAP.

1. Temperature-scale OOF predictions (calibration fix for tree models)
2. MLLS on calibrated test predictions -> estimate true class proportions per month
3. For each fold, importance-weight the OOF mAP using MLLS estimates
4. Combine into a single estimated LB score

WHY THIS IS BETTER THAN LOMO
------------------------------
LOMO evaluates on training months (Jan/Apr) as proxies for test months (Feb/May).
But it uses RAW class frequencies from the proxy fold, ignoring that Feb/May have
completely different species mixes. LOMO correlation with LB ~ 0.40 (nearly noise).

Importance weighting fixes this: we reweight Jan samples to match the ESTIMATED
Feb class distribution (via MLLS). A Jan fold with 5% Geese gets reweighted to
match Feb's MLLS-estimated 10% Geese. This makes the proxy evaluation more
representative of what the LB actually measures.

USAGE
------
    from src.validate import eval_pp, default_nb_pp

    def my_pp(preds, test_df, test_months, train_df, y):
        return modified_preds

    result = eval_pp(my_pp)          # full report + estimated LB
    score  = result['estimated_lb']  # single number to optimize

    # Quick mode for hyperparameter sweeps:
    result = eval_pp(my_pp, verbose=False)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

from .data import CLASSES, load_test, load_train
from .metrics import compute_map
from .postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

N_CLASSES = len(CLASSES)

# Test month distribution (from competition data analysis)
TEST_MONTH_WEIGHTS = {9: 0.244, 10: 0.429, 2: 0.094, 5: 0.162, 12: 0.071}

# Training month -> which test months it proxies
FOLD_TO_PROXY = {
    9:  [9],       # Sep -> Sep (shared)
    10: [10],      # Oct -> Oct (shared)
    1:  [2, 12],   # Jan -> Feb + Dec (winter)
    4:  [5],       # Apr -> May (spring)
}

# Known (IW-mAP, actual LB) pairs for NB PP calibration.
# Only NB PP variants are used (GBIF-only priors don't differentiate in IW-mAP).
# To recompute: python experiments/calibrate.py
# Fit quality: RMSE = 0.006 across 6 known LB submissions.
LB_CALIBRATION = [
    (0.6818, 0.59),   # E79 raw (no PP)
    (0.6787, 0.59),   # E75/E96 NB gamma=0.10 tau=0.25
    (0.6787, 0.59),   # E78/E75 NB gamma=0.10 tau=0.30
    (0.6786, 0.58),   # E73 NB gamma=0.12 tau=0.25
    (0.6780, 0.58),   # E74 NB gamma=0.14 tau=0.30
    (0.6718, 0.54),   # E87 NB gamma=0.50 tau=0.30
]

# Cache for expensive one-time computations
_cache: dict = {}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probability matrix."""
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def _load_calibration_pairs() -> list[tuple[float, float]]:
    """Load calibration pairs, trying CSV file first, falling back to hardcoded.

    Only includes rows where use_in_fit == 'yes' (or column is absent).
    """
    csv_path = ROOT / "data" / "lb_calibration.csv"
    if csv_path.exists():
        import csv
        pairs = []
        with open(csv_path) as f:
            lines = [l for l in f if not l.strip().startswith("#")]
            reader = csv.DictReader(lines)
            for row in reader:
                iw = row.get("iw_map", "").strip()
                lb = row.get("actual_lb", "").strip()
                use = row.get("use_in_fit", "yes").strip().lower()
                if iw and lb and use == "yes":
                    try:
                        pairs.append((float(iw), float(lb)))
                    except ValueError:
                        pass
        if len(pairs) >= 2:
            return pairs
    return list(LB_CALIBRATION)


def _calibrate_lb(iw_map: float) -> float | None:
    """Map raw IW-mAP to calibrated LB estimate using known datapoints.

    Loads from data/lb_calibration.csv if available, otherwise uses
    hardcoded LB_CALIBRATION. Fits a linear regression. Returns None if
    fewer than 2 calibration points are available.
    """
    pairs = _load_calibration_pairs()
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    # Least-squares linear fit: lb = slope * iw + intercept
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope * iw_map + intercept)


def _find_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Find T that minimizes NLL on labeled OOF data."""
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 5.0, 100):
        cal = _temperature_scale(probs, T)
        nll = -np.log(np.clip(cal[np.arange(len(y)), y], 1e-8, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


# ---------------------------------------------------------------------------
# MLLS (Maximum Likelihood Label Shift)
# ---------------------------------------------------------------------------

def _mlls_estimate(
    probs: np.ndarray, p_source: np.ndarray,
    max_iter: int = 200, tol: float = 1e-7,
) -> np.ndarray:
    """Estimate target class proportions from calibrated predictions via EM.

    E-step: reweight predictions by w[c] / p_source[c]
    M-step: w = mean of reweighted predictions
    """
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


# ---------------------------------------------------------------------------
# Importance-weighted macro mAP
# ---------------------------------------------------------------------------

def _iw_macro_map(
    y: np.ndarray, preds: np.ndarray,
    w_target: np.ndarray, p_fold: np.ndarray,
    max_weight: float = 10.0,
) -> float:
    """Compute importance-weighted macro mAP.

    Reweights samples so the evaluation distribution matches w_target
    instead of the fold's natural distribution p_fold.
    Weights are clipped at max_weight to prevent extreme values when
    source and target distributions diverge heavily.
    """
    from sklearn.metrics import average_precision_score

    # Per-sample importance weight: w_target[class] / p_fold[class]
    safe_p = np.maximum(p_fold, 1e-12)
    sample_w = (w_target / safe_p)[y]
    sample_w = np.clip(sample_w, 0, max_weight)
    sample_w = sample_w / sample_w.mean()  # normalize for numerical stability

    y_bin = np.zeros((len(y), N_CLASSES), dtype=int)
    y_bin[np.arange(len(y)), y] = 1

    aps = []
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue  # class not in this fold, skip
        ap = average_precision_score(
            y_bin[:, c], preds[:, c], sample_weight=sample_w,
        )
        aps.append(ap)
    return float(np.mean(aps))


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _load_train_data():
    if "train" in _cache:
        return _cache["train"]
    train_df = load_train()
    y = np.asarray(
        pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int
    )
    months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    result = (train_df, y, months)
    _cache["train"] = result
    return result


def _load_oof() -> tuple[np.ndarray, str]:
    if "oof" in _cache:
        return _cache["oof"]
    for name, path in [
        ("LOMO (honest)", ROOT / "oof_lomo.npy"),
        ("SKF E79", ROOT / "oof_e79.npy"),
        ("SKF E50", ROOT / "oof_e50.npy"),
    ]:
        if path.exists():
            result = (renorm_rows(np.load(path).astype(float)), name)
            _cache["oof"] = result
            return result
    raise FileNotFoundError("No OOF found (oof_lomo.npy / oof_e79.npy / oof_e50.npy)")


def _load_test_base():
    if "test" in _cache:
        return _cache["test"]
    for path in [ROOT / "test_e79.npy", ROOT / "test_e50.npy"]:
        if path.exists():
            test_preds = renorm_rows(np.load(path).astype(float))
            test_df = load_test()
            test_months = pd.to_datetime(
                test_df["timestamp_start_radar_utc"]
            ).dt.month.values
            result = (test_preds, test_df, test_months)
            _cache["test"] = result
            return result
    return None


def _get_mlls_weights() -> dict[int, np.ndarray]:
    """Compute GBIF-regularized MLLS class proportions per test month (cached).

    Blends MLLS estimates with GBIF biological priors using Bayesian shrinkage:
        w_final = beta * w_mlls + (1-beta) * w_gbif
        beta = n_samples / (n_samples + PRIOR_STRENGTH)

    Small months (Dec n=133) get pulled toward GBIF (beta~0.57).
    Large months (Oct n=803) stay mostly MLLS (beta~0.89).
    """
    if "mlls" in _cache:
        return _cache["mlls"]

    train_df, y, months = _load_train_data()
    oof, _ = _load_oof()
    test_data = _load_test_base()

    # Training class proportions
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    _cache["p_train"] = p_train

    # Temperature scaling (fit on OOF)
    T = _find_temperature(oof, y)
    _cache["T"] = T

    # GBIF biological priors per month
    gbif_priors = build_gbif_priors(p_train)

    if test_data is None:
        # No test data: fall back to GBIF priors
        _cache["mlls"] = gbif_priors
        return gbif_priors

    test_preds, _, test_months = test_data
    test_cal = _temperature_scale(test_preds, T)

    # MLLS per test month, regularized with GBIF
    PRIOR_STRENGTH = 100  # effective sample size of GBIF prior
    w = {}
    for m in sorted(set(test_months)):
        mask = test_months == m
        n = int(mask.sum())

        if n >= N_CLASSES:
            w_mlls = _mlls_estimate(test_cal[mask], p_train)
        else:
            w_mlls = p_train.copy()

        # Bayesian shrinkage toward GBIF
        w_gbif = gbif_priors.get(m, p_train)
        beta = n / (n + PRIOR_STRENGTH)
        w_blend = beta * w_mlls + (1 - beta) * w_gbif
        w_blend = np.maximum(w_blend, 1e-8)
        w_blend /= w_blend.sum()
        w[m] = w_blend

    _cache["mlls"] = w
    _cache["mlls_raw"] = {
        m: _mlls_estimate(test_cal[test_months == m], p_train)
        if (test_months == m).sum() >= N_CLASSES else p_train.copy()
        for m in sorted(set(test_months))
    }
    return w


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def eval_pp(
    pp_fn,
    verbose: bool = True,
) -> dict:
    """Evaluate PP using importance-weighted mAP estimation.

    Args:
        pp_fn: callable(preds, test_df, test_months, train_df, y) -> preds
        verbose: print the report

    Returns:
        dict with:
            estimated_lb:  weighted IW-mAP across all folds (main optimization target)
            estimated_delta: estimated_lb - baseline estimated_lb
            unseen_iw_map: IW-mAP for unseen months only (Feb/May/Dec proxy)
            shared_delta:  raw mAP change on Sep+Oct (safety gate)
            shared_pass:   True if shared months don't degrade
            pct_changed:   % of test predictions shifted >5%
            recommendation: SUBMIT / SKIP / REJECT
            per_fold:      detailed per-fold results
            mlls:          MLLS-estimated class proportions per month
    """
    train_df, y, months = _load_train_data()
    oof, oof_mode = _load_oof()
    w_test = _get_mlls_weights()
    p_train = _cache["p_train"]
    T = _cache["T"]

    # -------------------------------------------------------------------
    # Per-fold evaluation
    # -------------------------------------------------------------------
    fold_results = {}
    for held_out, proxy_months in FOLD_TO_PROXY.items():
        va_mask = months == held_out
        tr_mask = ~va_mask

        va_oof = oof[va_mask]
        va_y = y[va_mask]
        va_df = train_df[va_mask].reset_index(drop=True)
        tr_df = train_df[tr_mask].reset_index(drop=True)
        tr_y = y[tr_mask]

        # Fold class distribution
        p_fold = np.bincount(va_y, minlength=N_CLASSES).astype(float)
        p_fold = np.maximum(p_fold / p_fold.sum(), 1e-12)

        # Blend MLLS weights for the proxy test months
        w_target = np.zeros(N_CLASSES)
        total_tw = 0.0
        for pm in proxy_months:
            if pm in w_test:
                tw = TEST_MONTH_WEIGHTS.get(pm, 0.05)
                w_target += w_test[pm] * tw
                total_tw += tw
        if total_tw > 0:
            w_target /= total_tw
        else:
            w_target = p_fold.copy()

        # Remap month so PP gating fires on proxy folds
        remapped_m = np.full(va_mask.sum(), proxy_months[0], dtype=int)

        # Apply PP
        after_preds = pp_fn(va_oof, va_df, remapped_m, tr_df, tr_y)

        # IW-mAP (the main metric)
        before_iw = _iw_macro_map(va_y, va_oof, w_target, p_fold)
        after_iw = _iw_macro_map(va_y, after_preds, w_target, p_fold)

        # Raw mAP (for shared-month safety check)
        before_raw, _ = compute_map(va_y, va_oof)
        after_raw, _ = compute_map(va_y, after_preds)

        # Test weight for this fold
        test_w = sum(TEST_MONTH_WEIGHTS.get(pm, 0) for pm in proxy_months)

        fold_results[held_out] = {
            "proxy": proxy_months,
            "test_weight": test_w,
            "before_iw": float(before_iw),
            "after_iw": float(after_iw),
            "delta_iw": float(after_iw - before_iw),
            "before_raw": float(before_raw),
            "after_raw": float(after_raw),
            "delta_raw": float(after_raw - before_raw),
            "n": int(va_mask.sum()),
            "shared": held_out in (9, 10),
        }

    # -------------------------------------------------------------------
    # Combined estimates
    # -------------------------------------------------------------------
    lb_before = sum(r["test_weight"] * r["before_iw"] for r in fold_results.values())
    lb_after = sum(r["test_weight"] * r["after_iw"] for r in fold_results.values())

    # Shared-month safety gate (raw mAP, not IW)
    shared_w = sum(r["test_weight"] for r in fold_results.values() if r["shared"])
    shared_before = sum(
        r["test_weight"] * r["before_raw"]
        for r in fold_results.values() if r["shared"]
    ) / shared_w
    shared_after = sum(
        r["test_weight"] * r["after_raw"]
        for r in fold_results.values() if r["shared"]
    ) / shared_w
    shared_delta = shared_after - shared_before
    shared_pass = shared_delta >= -0.002

    # Unseen-month-only IW-mAP (what PP actually affects)
    unseen_w = sum(r["test_weight"] for r in fold_results.values() if not r["shared"])
    unseen_before = sum(
        r["test_weight"] * r["before_iw"]
        for r in fold_results.values() if not r["shared"]
    ) / max(unseen_w, 1e-12)
    unseen_after = sum(
        r["test_weight"] * r["after_iw"]
        for r in fold_results.values() if not r["shared"]
    ) / max(unseen_w, 1e-12)

    # -------------------------------------------------------------------
    # Conservatism check (on actual test predictions)
    # -------------------------------------------------------------------
    pct_changed_5 = None
    pct_changed_10 = None
    test_data = _load_test_base()
    if test_data is not None:
        test_base, test_df, test_months = test_data
        new_test = pp_fn(test_base, test_df, test_months, train_df, y)
        shift = np.abs(new_test - test_base)
        pct_changed_5 = float((shift.max(axis=1) > 0.05).mean() * 100)
        pct_changed_10 = float((shift.max(axis=1) > 0.10).mean() * 100)

    # -------------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------------
    if not shared_pass:
        recommendation = "REJECT (shared months degraded)"
    elif pct_changed_5 is not None and pct_changed_5 > 30:
        recommendation = "RISKY (large prediction shift)"
    elif lb_after > lb_before + 0.005:
        recommendation = "SUBMIT (estimated improvement)"
    elif lb_after >= lb_before - 0.010:
        # IW-mAP noise is ~0.01, so deltas within [-0.01, +0.005] are neutral
        recommendation = "SUBMIT (within noise, safe to try)"
    else:
        recommendation = "SKIP (estimated regression)"

    # LB calibration (linear correction from known IW-mAP -> actual LB pairs)
    cal_lb_after = _calibrate_lb(lb_after)
    cal_lb_before = _calibrate_lb(lb_before)
    cal_delta = (cal_lb_after - cal_lb_before) if (cal_lb_after and cal_lb_before) else None

    results = {
        "estimated_lb": round(lb_after, 4),
        "estimated_lb_before": round(lb_before, 4),
        "estimated_delta": round(lb_after - lb_before, 4),
        "calibrated_lb": round(cal_lb_after, 3) if cal_lb_after else None,
        "calibrated_lb_before": round(cal_lb_before, 3) if cal_lb_before else None,
        "calibrated_delta": round(cal_delta, 3) if cal_delta else None,
        "unseen_before": round(unseen_before, 4),
        "unseen_after": round(unseen_after, 4),
        "unseen_delta": round(unseen_after - unseen_before, 4),
        "shared_delta": round(shared_delta, 4),
        "shared_pass": shared_pass,
        "pct_changed": round(pct_changed_5, 1) if pct_changed_5 is not None else None,
        "temperature": round(T, 3),
        "recommendation": recommendation,
        "per_fold": fold_results,
        "mlls": {m: w.tolist() for m, w in w_test.items()},
    }

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    if verbose:
        _print_report(
            results, oof_mode, T, w_test, p_train, fold_results,
            lb_before, lb_after,
            cal_lb_before, cal_lb_after, cal_delta,
            unseen_before, unseen_after,
            shared_before, shared_after, shared_delta, shared_pass,
            pct_changed_5, pct_changed_10,
            recommendation,
        )

    return results


def _print_report(
    results, oof_mode, T, w_test, p_train, fold_results,
    lb_before, lb_after,
    cal_lb_before, cal_lb_after, cal_delta,
    unseen_before, unseen_after,
    shared_before, shared_after, shared_delta, shared_pass,
    pct_changed_5, pct_changed_10,
    recommendation,
):
    W = 70
    print("\n" + "=" * W, flush=True)
    print("  IW-mAP VALIDATION REPORT".center(W), flush=True)
    print(f"  OOF: {oof_mode} | Temperature: T={T:.2f}".center(W), flush=True)
    print("=" * W, flush=True)

    # MLLS + GBIF estimates
    abbr = [c[:4] for c in CLASSES]
    print(f"\n  Estimated class proportions (MLLS + GBIF regularization):", flush=True)
    header = "           " + "  ".join(f"{a:>5}" for a in abbr)
    print(header, flush=True)
    print(f"  Train    " + "  ".join(f"{p:5.2f}" for p in p_train), flush=True)
    month_names = {2: "Feb", 5: "May", 9: "Sep", 10: "Oct", 12: "Dec"}
    for m in [9, 10, 2, 5, 12]:
        if m in w_test:
            lbl = month_names.get(m, f"M{m:02d}")
            vals = "  ".join(f"{w:5.2f}" for w in w_test[m])
            marker = " *" if m in (2, 5, 12) else ""
            print(f"  {lbl:<7}  {vals}{marker}", flush=True)
    print(f"  (* = unseen, MLLS blended with GBIF biological priors)", flush=True)

    # Per-fold breakdown
    print(f"\n  PER-FOLD IW-mAP:", flush=True)
    print(
        f"  {'Fold':>5}  {'Proxy':>10}  {'Weight':>6}"
        f"  {'Before':>7}  {'After':>7}  {'Delta':>7}  {'N':>5}",
        flush=True,
    )
    proxy_labels = {
        9: "Sep", 10: "Oct", 1: "Feb+Dec", 4: "May",
    }
    for m in [9, 10, 1, 4]:
        r = fold_results[m]
        lbl = proxy_labels[m]
        shared_tag = "" if not r["shared"] else " (shared)"
        print(
            f"  M{m:02d}    {lbl:>10}  {r['test_weight']:>6.3f}"
            f"  {r['before_iw']:>7.4f}  {r['after_iw']:>7.4f}"
            f"  {r['delta_iw']:>+7.4f}  {r['n']:>5}{shared_tag}",
            flush=True,
        )

    # Combined estimates
    print(f"\n  ESTIMATED LB:", flush=True)
    if cal_lb_after is not None:
        print(f"    Calibrated:  {cal_lb_before:.3f} -> {cal_lb_after:.3f}  (delta: {cal_delta:+.3f})", flush=True)
        n_cal = len(_load_calibration_pairs())
        print(f"    (calibrated from {n_cal} known LB datapoints)", flush=True)
    print(f"    Raw IW-mAP:  {lb_before:.4f} -> {lb_after:.4f}  (delta: {lb_after-lb_before:+.4f})", flush=True)
    print(f"    Unseen only: {unseen_before:.4f} -> {unseen_after:.4f}  (delta: {unseen_after-unseen_before:+.4f})", flush=True)

    # Safety signals
    print(f"\n  SAFETY:", flush=True)
    gate = "PASS" if shared_pass else "FAIL"
    print(f"    Shared months (Sep+Oct): {shared_delta:+.4f} [{gate}]", flush=True)
    if pct_changed_5 is not None:
        risk = "safe" if pct_changed_5 < 20 else ("elevated" if pct_changed_5 < 35 else "HIGH")
        print(f"    Conservatism: {pct_changed_5:.1f}% shifted >5%, {pct_changed_10:.1f}% shifted >10% [{risk}]", flush=True)

    print(f"\n  >> {recommendation}", flush=True)
    print("=" * W, flush=True)


# ---------------------------------------------------------------------------
# Reference PP (conservative tabular-only baseline, matches E96 recipe)
# ---------------------------------------------------------------------------

def default_nb_pp(
    preds: np.ndarray,
    test_df: pd.DataFrame,
    test_months: np.ndarray,
    train_df: pd.DataFrame,
    y: np.ndarray,
    tau_prior: float = 0.15,
    tau_nb: float = 0.25,
    gamma: float = 0.10,
) -> np.ndarray:
    """Reference NB post-processing (tabular channels, gamma=0.10, LB ~0.59)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    out, _ = apply_gated_ratio_priors(
        preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior
    )

    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed, "alt_mid": 0.5*(min_z+max_z), "alt_range": max_z-min_z}
    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_t = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_t = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_t = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_t, "alt_mid": 0.5*(min_z_t+max_z_t), "alt_range": max_z_t-min_z_t}

    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig
    )
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)


if __name__ == "__main__":
    print("Running IW-mAP validation with default NB PP...", flush=True)
    result = eval_pp(default_nb_pp)
    if result["calibrated_lb"] is not None:
        print(f"\nCalibrated LB: {result['calibrated_lb']}", flush=True)
    print(f"Raw IW-mAP:    {result['estimated_lb']}", flush=True)
