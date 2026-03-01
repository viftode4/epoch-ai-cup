"""Build calibration curve from Kaggle submissions using eval_pp.

Usage:
    python experiments/calibrate.py           # compute and show calibration
    python experiments/calibrate.py --fit     # show per-point residuals

Approach: for each Kaggle submission with known LB, build a PP function
that replicates the config (parsed from filename), then compute IW-mAP
via eval_pp. Only works for submissions that are PP on E50/E79 base.
"""
from __future__ import annotations

import re
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES
from src.postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    extract_heading_ac1,
)
from src.validate import eval_pp, default_nb_pp, _cache

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
CAL_FILE = ROOT / "data" / "lb_calibration.csv"


def fetch_kaggle_submissions() -> list[dict]:
    """Fetch all submissions and LB scores from Kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "ai-cup-2026-performance"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
    except Exception:
        return []
    subs = []
    for line in result.stdout.strip().split("\n")[2:]:
        parts = line.split()
        if len(parts) >= 4 and "COMPLETE" in line:
            fn = parts[0]
            lb = parts[-1]
            try:
                subs.append({"filename": fn, "lb": float(lb)})
            except ValueError:
                pass
    return subs


# ---------------------------------------------------------------------------
# PP function builders
# ---------------------------------------------------------------------------

def _no_pp(preds, test_df, test_months, train_df, y):
    return preds.copy()


def _gbif_only(preds, test_df, test_months, train_df, y, tau_prior=0.15):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)
    return out


def _nb_pp(preds, test_df, test_months, train_df, y,
           gamma=0.10, tau_nb=0.25, tau_prior=0.15, w_alt_range=0.5,
           unseen_only=True):
    """Standard tabular NB PP (speed + alt_mid + alt_range)."""
    return default_nb_pp(preds, test_df, test_months, train_df, y,
                         tau_prior=tau_prior, tau_nb=tau_nb, gamma=gamma)


def _nb_shared(preds, test_df, test_months, train_df, y,
               gamma_u=0.10, tau_nb_u=0.25, gamma_s=0.08, tau_nb_s=0.10):
    """NB PP on both unseen AND shared months."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=0.15)

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
    loglike = compute_log_p_u_given_c(test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig)

    gate_u = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb_u)
    out = apply_nb_poe(out, loglike, gamma=gamma_u, gate=gate_u)
    gate_s = ~np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb_s)
    out = apply_nb_poe(out, loglike, gamma=gamma_s, gate=gate_s)
    return out


def _nb_heading(preds, test_df, test_months, train_df, y,
                gamma=0.10, tau_nb=0.25):
    """NB PP with heading_R + rcs_ac1 channels."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=0.15)

    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    heading_r_tr, ac1_tr, traj_ok_tr = extract_heading_ac1(train_df)
    cont_tr = {
        "speed": speed, "alt_mid": 0.5*(min_z+max_z), "alt_range": max_z-min_z,
        "heading_R": heading_r_tr, "rcs_ac1": ac1_tr,
    }
    ok_tr = {"heading_R": traj_ok_tr, "rcs_ac1": traj_ok_tr}
    min_sigma = {"speed": 0.50, "alt_mid": 0.50, "alt_range": 0.50, "heading_R": 0.10, "rcs_ac1": 0.10}
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, cont_tr, ok_masks=ok_tr, min_sigma=min_sigma
    )

    speed_t = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_t = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_t = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    heading_r_te, ac1_te, traj_ok_te = extract_heading_ac1(test_df)
    cont_te = {
        "speed": speed_t, "alt_mid": 0.5*(min_z_t+max_z_t), "alt_range": max_z_t-min_z_t,
        "heading_R": heading_r_te, "rcs_ac1": ac1_te,
    }
    ok_te = {"heading_R": traj_ok_te, "rcs_ac1": traj_ok_te}
    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5, "heading_R": 1.0, "rcs_ac1": 1.0}
    loglike = compute_log_p_u_given_c(test_df, size_levels, log_p_size, cont_te, weights, ok_te, mu, sig)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)


# ---------------------------------------------------------------------------
# Match submission filenames to PP functions
# ---------------------------------------------------------------------------

def match_submission_to_pp(fn: str) -> tuple[callable, str] | None:
    """Parse submission filename and return (pp_function, description).

    Returns None for submissions that use different base models or
    non-replicable approaches (TabPFN, blends, pseudo-labels, etc.)
    """
    # Skip non-PP submissions (different models, blends)
    skip_prefixes = [
        "e42_", "e48_", "e49_", "e62_", "e63_", "e65_",
        "e69_", "e71_winter", "e71_pergroup", "e72_", "e82_",
        "e83_", "e92_", "submission_", "e75_framework",
    ]
    for prefix in skip_prefixes:
        if fn.startswith(prefix):
            return None

    # E87: gamma=0.50, tau=0.30 (on E79 base)
    if fn.startswith("e87_v1_g050"):
        return partial(_nb_pp, gamma=0.50, tau_nb=0.30), "E87 g=0.50 tau=0.30"

    # E81: shared month correction
    if fn.startswith("e81_nbalt_shared"):
        m = re.search(r"tau(\d+\.\d+)_g(\d+\.\d+)", fn)
        if m:
            return partial(_nb_shared, gamma_s=float(m.group(2)), tau_nb_s=float(m.group(1))), f"E81 shared"
        return partial(_nb_shared), "E81 shared"

    # E96+: heading+AC1 submissions
    if "heading_ac1" in fn or "heading" in fn.lower():
        m_g = re.search(r"_g(\d+\.\d+)", fn)
        m_t = re.search(r"tau(\d+\.\d+)", fn)
        g = float(m_g.group(1)) if m_g else 0.10
        t = float(m_t.group(1)) if m_t else 0.25
        return partial(_nb_heading, gamma=g, tau_nb=t), f"heading g={g} tau={t}"

    # E97-E101: various evidence extensions (use heading PP as closest match)
    for prefix in ["e97_", "e98_", "e99_", "e100_", "e101_"]:
        if fn.startswith(prefix):
            m_g = re.search(r"_g(\d+\.\d+)", fn)
            g = float(m_g.group(1)) if m_g else 0.10
            return partial(_nb_heading, gamma=g), f"{prefix[:-1]} approx heading g={g}"

    # E94/E95: discriminative PoE (approximate with NB)
    if fn.startswith("e94_") or fn.startswith("e95_"):
        m_l = re.search(r"lam(\d+\.\d+)", fn)
        lam = float(m_l.group(1)) if m_l else 0.20
        return partial(_nb_pp, gamma=lam, tau_nb=0.30), f"E94/95 approx g={lam}"

    # E73-E80: NB physics variants
    # Extract gamma and tau from filename
    m_g = re.search(r"_g(\d+\.\d+)", fn)
    m_t = re.search(r"tau(\d+\.\d+)", fn)
    if m_g:
        g = float(m_g.group(1))
        t = float(m_t.group(1)) if m_t else 0.25
        return partial(_nb_pp, gamma=g, tau_nb=t), f"NB g={g} tau={t}"

    # E54-E58, E67, E70: GBIF-only priors (no NB evidence)
    if any(fn.startswith(p) for p in ["e54_", "e55_", "e56_", "e57_", "e58_", "e67_", "e70_"]):
        return _gbif_only, "GBIF priors only"

    return None


def main():
    show_fit = "--fit" in sys.argv

    # Fetch from Kaggle
    print("Fetching Kaggle submissions...", flush=True)
    subs = fetch_kaggle_submissions()
    if not subs:
        print("  ERROR: No submissions found", flush=True)
        return
    print(f"  Found {len(subs)} submissions\n", flush=True)

    # Compute IW-mAP for each matched submission
    print(f"  {'LB':>5} {'IW-mAP':>7} {'Cal.LB':>7}  Description", flush=True)
    print(f"  {'-'*60}", flush=True)

    results = []
    heading_computed = False

    for sub in subs:
        fn = sub["filename"]
        lb = sub["lb"]

        match = match_submission_to_pp(fn)
        if match is None:
            continue

        pp_fn, desc = match

        # Skip heading PP runs after the first (slow due to trajectory parsing)
        is_heading = "heading" in desc or "approx heading" in desc or desc.startswith("E94") or desc.startswith("E81")
        if is_heading and heading_computed:
            # Reuse cached heading IW-mAP (trajectory extraction is expensive)
            # The heading PP gives ~same IW-mAP for similar gamma values
            pass
        else:
            _cache.clear()

        r = eval_pp(pp_fn, verbose=False)
        iw = r["estimated_lb"]
        cal = r.get("calibrated_lb")
        cal_str = f"{cal:.3f}" if cal else "  N/A"

        print(f"  {lb:>5.2f} {iw:>7.4f} {cal_str:>7}  {desc}  ({fn[:50]})", flush=True)
        results.append((iw, lb, fn, desc))

        if is_heading:
            heading_computed = True

    # Fit calibration
    if len(results) < 2:
        print("\nToo few results for calibration", flush=True)
        return

    x = np.array([r[0] for r in results])
    y_vals = np.array([r[1] for r in results])
    slope, intercept = np.polyfit(x, y_vals, 1)
    preds = slope * x + intercept
    residuals = y_vals - preds
    rmse = float(np.sqrt((residuals**2).mean()))
    mae = float(np.abs(residuals).mean())

    print(f"\nCalibration ({len(results)} points):", flush=True)
    print(f"  Cal.LB = {slope:.2f} * IW-mAP + ({intercept:.4f})", flush=True)
    print(f"  RMSE = {rmse:.4f}, MAE = {mae:.4f}", flush=True)

    if show_fit:
        print(f"\n  {'IW-mAP':>8} {'Actual':>7} {'Pred':>7} {'Error':>7}  Config", flush=True)
        for (iw, lb, fn, desc), pred in sorted(zip(results, preds), key=lambda t: t[0][0]):
            err = lb - pred
            print(f"  {iw:>8.4f} {lb:>7.2f} {pred:>7.3f} {err:>+7.3f}  {desc}", flush=True)

    # Deduplicate
    from collections import defaultdict
    groups = defaultdict(list)
    for iw, lb, fn, desc in results:
        groups[round(iw, 4)].append(lb)

    deduped = sorted(
        [(iw, float(np.median(lbs))) for iw, lbs in groups.items()],
        reverse=True,
    )

    print(f"\n  Deduplicated ({len(deduped)} unique IW-mAP values):", flush=True)
    print(f"  LB_CALIBRATION = [", flush=True)
    for iw, lb in deduped:
        print(f"      ({iw:.4f}, {lb:.2f}),", flush=True)
    print(f"  ]", flush=True)

    # Save CSV
    with open(CAL_FILE, "w", newline="") as f:
        f.write("config_name,iw_map,actual_lb,base,notes\n")
        for iw, lb, fn, desc in results:
            name = fn.replace(".csv", "")
            f.write(f"{name},{iw:.4f},{lb:.2f},auto,{desc}\n")
    print(f"\n  Saved to {CAL_FILE}", flush=True)


if __name__ == "__main__":
    main()
