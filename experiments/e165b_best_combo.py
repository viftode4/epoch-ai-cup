"""E165b: Best MLLS+PP combo from E165 sweep.

Best configs from E165:
  - D_stronger+MLLS_then_PP: calLB=0.612 (MLLS first, then standard PP)
  - E_aggressive+PP_then_MLLS: calLB=0.620 (PP first, then aggressive MLLS)

Generate both as submission files.
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
)
from src.validate import _mlls_estimate, eval_pp

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)


def apply_mlls_calibration(preds, months, p_train, alpha_map, tau=0.35):
    out = preds.copy()
    margin = top2_margin(out)
    for month, alpha in alpha_map.items():
        if alpha == 0:
            continue
        mask_m = months == month
        if mask_m.sum() < N_CLASSES:
            continue
        w_mlls = _mlls_estimate(out[mask_m], p_train)
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (w_mlls / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def make_nb_pp(preds, test_df, test_months, train_df, y,
               tau_prior=0.15, tau_nb=0.25, gamma=0.10):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)

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
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)


def main():
    print("=" * 70, flush=True)
    print("E165b: BEST MLLS+PP COMBOS".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()

    test_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))

    # Config D: MLLS first, then PP
    print("\n--- D_stronger + MLLS_then_PP ---", flush=True)
    alpha_d = {2: 0.15, 5: 0.12, 9: 0.0, 10: 0.0, 12: 0.12}
    out_d = apply_mlls_calibration(test_e79, test_months, p_train, alpha_d, tau=0.35)
    out_d = make_nb_pp(out_d, test_df, test_months, train_df, y)
    save_submission(out_d, "e165b_D_mlls_then_pp")

    # Config E: PP first, then MLLS
    print("\n--- E_aggressive + PP_then_MLLS ---", flush=True)
    alpha_e = {2: 0.20, 5: 0.15, 9: 0.0, 10: 0.0, 12: 0.15}
    out_e = make_nb_pp(test_e79, test_df, test_months, train_df, y)
    out_e = apply_mlls_calibration(out_e, test_months, p_train, alpha_e, tau=0.35)
    save_submission(out_e, "e165b_E_pp_then_mlls")

    # D standalone (no PP, just MLLS)
    print("\n--- D_stronger standalone ---", flush=True)
    out_d_raw = apply_mlls_calibration(test_e79, test_months, p_train, alpha_d, tau=0.35)
    save_submission(out_d_raw, "e165b_D_mlls_only")

    # Validate all via eval_pp
    print("\n--- IW-mAP Validation ---", flush=True)
    for name, alpha_map, tau, order in [
        ("D_mlls_then_pp", alpha_d, 0.35, "mlls_first"),
        ("E_pp_then_mlls", alpha_e, 0.35, "pp_first"),
        ("D_mlls_only", alpha_d, 0.35, "mlls_only"),
    ]:
        def make_pp(am=alpha_map, t=tau, o=order):
            def pp_fn(preds, test_df_v, test_months_v, train_df_v, y_v):
                counts_v = np.bincount(y_v, minlength=N_CLASSES).astype(float)
                p_tr_v = counts_v / counts_v.sum()
                if o == "mlls_only":
                    return apply_mlls_calibration(preds, test_months_v, p_tr_v, am, tau=t)
                elif o == "mlls_first":
                    out_v = apply_mlls_calibration(preds, test_months_v, p_tr_v, am, tau=t)
                    return make_nb_pp(out_v, test_df_v, test_months_v, train_df_v, y_v)
                else:
                    out_v = make_nb_pp(preds, test_df_v, test_months_v, train_df_v, y_v)
                    return apply_mlls_calibration(out_v, test_months_v, p_tr_v, am, tau=t)
            return pp_fn

        result = eval_pp(make_pp(), verbose=True)
        print(f"\n  {name}: calLB={result.get('calibrated_lb', 'N/A')}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
