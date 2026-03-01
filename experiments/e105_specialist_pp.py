"""E105: Stronger NB PP with heading_R + AC1 evidence.

E93 LOMO analysis showed:
  - gamma=0.50 with heading_R + rcs_ac1: LOMO +0.016 (best PP ever)
  - gamma=0.10 (current E75): only +0.002 (way too conservative)
  - heading_R adds +0.004 on top of AC1

This experiment evaluates the stronger config through IW-mAP validation
and sweeps gamma/tau to find the optimal PP strength.

Uses E79 base predictions.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.submission import save_submission
from src.postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    extract_heading_ac1,
)
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# Evidence weights (from E93 analysis)
WEIGHTS = {
    "speed":     1.00,
    "alt_mid":   1.00,
    "alt_range": 0.50,
    "heading_R": 1.00,
    "rcs_ac1":   1.00,
}
MIN_SIGMA = {
    "speed":     0.50,
    "alt_mid":   0.50,
    "alt_range": 0.50,
    "heading_R": 0.10,
    "rcs_ac1":   0.10,
}


def build_pp(tau_prior, tau_nb, gamma):
    """Create a PP function with given hyperparameters."""
    def pp(preds, test_df, test_months, train_df, y):
        counts = np.bincount(y, minlength=N_CLASSES).astype(float)
        p_train = counts / counts.sum()
        priors = build_gbif_priors(p_train)

        out, _ = apply_gated_ratio_priors(
            preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior,
        )

        # Tabular channels
        speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
        minz_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
        maxz_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)

        speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
        minz_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
        maxz_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)

        # Trajectory channels
        heading_r_tr, ac1_tr, ok_tr = extract_heading_ac1(train_df)
        heading_r_te, ac1_te, ok_te = extract_heading_ac1(test_df)

        cont_tr = {
            "speed": speed_tr,
            "alt_mid": 0.5 * (minz_tr + maxz_tr),
            "alt_range": maxz_tr - minz_tr,
            "heading_R": heading_r_tr,
            "rcs_ac1": ac1_tr,
        }
        ok_masks_tr = {"heading_R": ok_tr, "rcs_ac1": ok_tr}

        cont_te = {
            "speed": speed_te,
            "alt_mid": 0.5 * (minz_te + maxz_te),
            "alt_range": maxz_te - minz_te,
            "heading_R": heading_r_te,
            "rcs_ac1": ac1_te,
        }
        ok_masks_te = {"heading_R": ok_te, "rcs_ac1": ok_te}

        size_levels, log_p_size, mu, sig = build_nb_params(
            train_df, y, cont_tr, ok_masks=ok_masks_tr, min_sigma=MIN_SIGMA,
        )
        loglike = compute_log_p_u_given_c(
            test_df, size_levels, log_p_size, cont_te, WEIGHTS, ok_masks_te, mu, sig,
        )
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
        return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)

    return pp


def main():
    print("=" * 70, flush=True)
    print("E105 STRONGER NB PP (heading_R + AC1)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.values.astype(int)

    test_base = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))

    configs = [
        # (name, tau_prior, tau_nb, gamma)
        ("A_g010", 0.15, 0.25, 0.10),  # E75 reference
        ("B_g020", 0.15, 0.25, 0.20),
        ("C_g030", 0.15, 0.25, 0.30),
        ("D_g050", 0.15, 0.25, 0.50),  # E93 best LOMO config
        ("E_g030_wide", 0.15, 0.35, 0.30),
        ("F_g050_wide", 0.15, 0.35, 0.50),
    ]

    # Need to extract evidence once
    print("\nExtracting train evidence...", flush=True)
    heading_r_tr, ac1_tr, ok_tr = extract_heading_ac1(train_df)
    print("Extracting test evidence...", flush=True)
    heading_r_te, ac1_te, ok_te = extract_heading_ac1(test_df)

    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    minz_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    maxz_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    minz_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    maxz_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)

    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    cont_tr = {
        "speed": speed_tr, "alt_mid": 0.5*(minz_tr+maxz_tr),
        "alt_range": maxz_tr-minz_tr, "heading_R": heading_r_tr, "rcs_ac1": ac1_tr,
    }
    ok_masks_tr = {"heading_R": ok_tr, "rcs_ac1": ok_tr}
    cont_te = {
        "speed": speed_te, "alt_mid": 0.5*(minz_te+maxz_te),
        "alt_range": maxz_te-minz_te, "heading_R": heading_r_te, "rcs_ac1": ac1_te,
    }
    ok_masks_te = {"heading_R": ok_te, "rcs_ac1": ok_te}

    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, cont_tr, ok_masks=ok_masks_tr, min_sigma=MIN_SIGMA,
    )
    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, cont_te, WEIGHTS, ok_masks_te, mu, sig,
    )

    for name, tau_prior, tau_nb, gamma in configs:
        print(f"\n--- Config {name}: tau_prior={tau_prior} tau_nb={tau_nb} gamma={gamma} ---", flush=True)
        out, _ = apply_gated_ratio_priors(
            test_base, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior,
        )
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
        out = apply_nb_poe(out, loglike, gamma=gamma, gate=gate)
        print(f"  Gate: {int(gate.sum())} rows", flush=True)
        sub_name = f"e105_{name}"
        save_submission(out, sub_name, cv_map=None)

    print("\nDone.", flush=True)


def local_eval():
    """Evaluate via IW-mAP."""
    from src.validate import eval_pp

    configs = [
        ("A_g010", 0.15, 0.25, 0.10),
        ("B_g020", 0.15, 0.25, 0.20),
        ("C_g030", 0.15, 0.25, 0.30),
        ("D_g050", 0.15, 0.25, 0.50),
        ("E_g030_wide", 0.15, 0.35, 0.30),
        ("F_g050_wide", 0.15, 0.35, 0.50),
    ]

    for name, tau_prior, tau_nb, gamma in configs:
        print(f"\n{'='*70}", flush=True)
        print(f"  Config {name}: tau_prior={tau_prior} tau_nb={tau_nb} gamma={gamma}", flush=True)
        print(f"{'='*70}", flush=True)

        result = eval_pp(build_pp(tau_prior, tau_nb, gamma))
        cal_lb = result.get("calibrated_lb", "N/A")
        iw_map = result.get("estimated_lb", "N/A")
        delta = result.get("estimated_delta", "N/A")
        print(f"  >> Cal.LB={cal_lb} IW-mAP={iw_map} delta={delta}", flush=True)


if __name__ == "__main__":
    if "--eval" in sys.argv:
        local_eval()
    else:
        main()
