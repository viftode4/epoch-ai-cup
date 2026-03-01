"""E103 TEMPLATE — starting point for all future post-processing experiments.

Copy this file, rename it, and only edit the sections marked with ★.
The canonical 3-stage pipeline (priors → evidence → PoE) lives in src/postprocessing.py.
A new experiment is ~80-150 lines; boilerplate is zero.

Pipeline recap:
  Stage 1  apply_gated_ratio_priors  — GBIF month priors on unseen months (fixed)
  Stage 2  YOUR evidence channels    — ★ define arrays here
  Stage 3  apply_nb_poe             — gated PoE update (fixed)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.submission import save_submission  # noqa: E402
from src.postprocessing import (  # noqa: E402
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    extract_heading_ac1,
)

ROOT = Path(__file__).resolve().parent.parent

# ★ --- Hyperparameters (edit these) ------------------------------------------
TAU_PRIOR = 0.15   # gate for GBIF prior stage (keep fixed unless experimenting)
TAU_NB    = 0.25   # gate for evidence stage (uncertain rows only)
GAMMA     = 0.10   # evidence strength

# ★ Per-channel minimum sigma (prevents over-sharp likelihoods)
MIN_SIGMA = {
    "speed":     0.50,
    "alt_mid":   0.50,
    "alt_range": 0.50,
    "heading_R": 0.10,
    "rcs_ac1":   0.10,
    # add your new channels here
}

# ★ Per-channel evidence weights (0 = disabled)
WEIGHTS = {
    "speed":     1.00,
    "alt_mid":   1.00,
    "alt_range": 0.50,
    "heading_R": 1.00,
    "rcs_ac1":   1.00,
    # add your new channels here
}
# -----------------------------------------------------------------------------


def extract_evidence(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """★ Extract your evidence channel arrays from raw DataFrame.

    Returns:
        cont_channels: name -> (n,) float array of feature values
        ok_masks:      name -> (n,) bool validity mask  (omit key = use np.isfinite)
    """
    # --- baseline channels (always include) ---
    speed     = pd.to_numeric(df["airspeed"],  errors="coerce").values.astype(float)
    min_z     = pd.to_numeric(df["min_z"],     errors="coerce").values.astype(float)
    max_z     = pd.to_numeric(df["max_z"],     errors="coerce").values.astype(float)
    alt_mid   = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    # --- trajectory-derived channels ---
    heading_r, rcs_ac1, traj_ok = extract_heading_ac1(df)

    cont_channels = {
        "speed":     speed,
        "alt_mid":   alt_mid,
        "alt_range": alt_range,
        "heading_R": heading_r,
        "rcs_ac1":   rcs_ac1,
        # ★ add your new channel arrays here
    }
    ok_masks = {
        "heading_R": traj_ok,
        "rcs_ac1":   traj_ok,
        # ★ add validity masks for channels that can be missing
    }
    return cont_channels, ok_masks


def main() -> None:
    print("=" * 70, flush=True)
    print("E103 TEMPLATE".center(70), flush=True)  # ★ update name
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df  = load_test()

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months  = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    y      = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.values.astype(int)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    p_train = counts / counts.sum()

    # Load base predictions (★ swap to oof_e79/test_e79 if needed)
    oof_base  = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

    # Stage 1: GBIF priors
    priors = build_gbif_priors(p_train)
    test_p, n_changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR,
    )
    print(f"\nPriors: tau={TAU_PRIOR}, changed={n_changed} rows", flush=True)

    # Stage 2: extract evidence
    print("\nExtracting train evidence...", flush=True)
    train_ch, train_ok = extract_evidence(train_df)

    print("\nExtracting test evidence...", flush=True)
    test_ch, test_ok = extract_evidence(test_df)

    # Stage 3: NB PoE — ★ add variant loop here if sweeping hyperparameters
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, train_ch, ok_masks=train_ok, min_sigma=MIN_SIGMA,
    )

    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_p) < TAU_NB)
    print(f"Evidence gate: {int(gate.sum())} rows (tau_nb={TAU_NB})", flush=True)

    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, test_ch, WEIGHTS, test_ok, mu, sig,
    )
    out = apply_nb_poe(test_p, loglike, gamma=GAMMA, gate=gate)

    # ★ Update experiment name
    name = f"e103_template_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}"
    save_submission(out, name, cv_map=None)
    print("\nDone.", flush=True)


def local_eval() -> None:
    """Fast local validation before submitting (~30s, no GPU, no retraining).

    Usage: python experiments/e103_template.py --eval

    Compares this experiment's PP against the E96 reference baseline.
    Decision rule:
      - Signal 1: Shared months delta >= -0.002  (Sep+Oct must not degrade)
      - Signal 2: L1 vs conservative baseline < 0.020  (not too aggressive)
      - Signal 3: Predictions shifted >5% < 20%  (conservatism check)
    """
    import pandas as pd
    from src.validate import eval_pp

    def my_pp(preds, test_df, test_months, train_df, y):
        counts   = np.bincount(y, minlength=len(CLASSES)).astype(float)
        p_train  = counts / counts.sum()
        priors   = build_gbif_priors(p_train)
        out, _   = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)

        # Tabular channels only (skip GPU trajectory extraction for speed)
        speed_tr  = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
        minz_tr   = pd.to_numeric(train_df["min_z"],    errors="coerce").values.astype(float)
        maxz_tr   = pd.to_numeric(train_df["max_z"],    errors="coerce").values.astype(float)
        cont_tr   = {"speed": speed_tr, "alt_mid": 0.5*(minz_tr+maxz_tr), "alt_range": maxz_tr-minz_tr}
        size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr, min_sigma=MIN_SIGMA)

        speed_te  = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
        minz_te   = pd.to_numeric(test_df["min_z"],    errors="coerce").values.astype(float)
        maxz_te   = pd.to_numeric(test_df["max_z"],    errors="coerce").values.astype(float)
        cont_te   = {"speed": speed_te, "alt_mid": 0.5*(minz_te+maxz_te), "alt_range": maxz_te-minz_te}

        loglike = compute_log_p_u_given_c(test_df, size_levels, log_p_size, cont_te, WEIGHTS, None, mu, sig)
        gate    = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_p := out) < TAU_NB)
        return apply_nb_poe(out, loglike, gamma=GAMMA, gate=gate)

    eval_pp(my_pp)


if __name__ == "__main__":
    import sys
    if "--eval" in sys.argv:
        local_eval()
    else:
        main()
