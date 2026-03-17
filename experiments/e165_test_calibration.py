"""E165: Test-Time Label Calibration.

Layer MLLS-based class proportion calibration on top of E79 predictions.
No retraining needed - pure post-processing on existing predictions.

Approach:
  1. Start from E79 test predictions
  2. Temperature scale using OOF-fitted T
  3. Per-month MLLS class proportion estimation
  4. Apply Bayes ratio adjustment: p_cal = p_raw * (w_mlls / w_train)^alpha
  5. Grid search alpha per month
  6. Compare: raw -> +MLLS -> +PP -> +PP+MLLS

Also test composition with existing NB PP pipeline.
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
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)
from src.validate import (
    _temperature_scale, _find_temperature, _mlls_estimate,
    eval_pp,
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# Test month distribution (from competition data analysis)
TEST_MONTH_WEIGHTS = {9: 0.244, 10: 0.429, 2: 0.094, 5: 0.162, 12: 0.071}


def apply_mlls_calibration(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    alpha_map: dict[int, float],
    tau: float = 0.30,
) -> np.ndarray:
    """Apply MLLS-based Bayes ratio calibration per month.

    For each month m:
        p_cal[i] = p_raw[i] * (w_mlls[m] / p_train)^alpha_m
    Only apply to uncertain rows (top2_margin < tau).
    """
    out = preds.copy()
    margin = top2_margin(out)
    n_changed = 0

    for month, alpha in alpha_map.items():
        if alpha == 0:
            continue
        mask_m = months == month
        if mask_m.sum() < N_CLASSES:
            continue

        # Estimate class proportions for this month via MLLS
        w_mlls = _mlls_estimate(out[mask_m], p_train)

        # Apply ratio tilt to uncertain rows
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue

        ratio = (w_mlls / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        n_changed += int(gate.sum())

    return renorm_rows(out), n_changed


def make_nb_pp(tau_prior=0.15, tau_nb=0.25, gamma=0.10):
    """Create standard NB PP function."""
    def pp_fn(preds, test_df, test_months, train_df, y):
        counts = np.bincount(y, minlength=N_CLASSES).astype(float)
        p_train = counts / counts.sum()
        priors = build_gbif_priors(p_train)

        out, _ = apply_gated_ratio_priors(
            preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior,
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
            test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig,
        )
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
        return apply_nb_poe(out, loglike, gamma=gamma, gate=gate)
    return pp_fn


def main():
    print("=" * 70, flush=True)
    print("E165 TEST-TIME LABEL CALIBRATION".center(70), flush=True)
    print("=" * 70, flush=True)

    # -- Load data ---
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()

    # Load base predictions
    oof_e79 = renorm_rows(np.load(ROOT / "oof_e79.npy").astype(float))
    test_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))

    # Temperature scaling
    T = _find_temperature(oof_e79, y)
    print(f"\nTemperature: T={T:.3f}", flush=True)

    test_cal = _temperature_scale(test_e79, T)

    # MLLS class proportion estimation per month
    print("\nMLLS class proportion estimates per month:", flush=True)
    abbr = [c[:4] for c in CLASSES]
    print(f"  {'':>8}" + "  ".join(f"{a:>5}" for a in abbr), flush=True)
    print(f"  {'Train':>8}" + "  ".join(f"{p:5.3f}" for p in p_train), flush=True)

    mlls_per_month = {}
    for m in sorted(np.unique(test_months)):
        mask = test_months == m
        w_mlls = _mlls_estimate(test_cal[mask], p_train)
        mlls_per_month[m] = w_mlls
        month_names = {2: "Feb", 5: "May", 9: "Sep", 10: "Oct", 12: "Dec"}
        lbl = month_names.get(m, f"M{m:02d}")
        print(f"  {lbl:>8}" + "  ".join(f"{w:5.3f}" for w in w_mlls), flush=True)

    # ============================================================
    # Sweep alpha values for MLLS calibration
    # ============================================================
    print("\n--- Alpha sweep for MLLS calibration ---", flush=True)
    print("  (Evaluating via IW-mAP validation system)", flush=True)

    alpha_configs = [
        # Conservative per-month alphas
        {"name": "A_conservative", "alphas": {2: 0.05, 5: 0.05, 9: 0.02, 10: 0.02, 12: 0.03}},
        {"name": "B_moderate", "alphas": {2: 0.10, 5: 0.08, 9: 0.03, 10: 0.03, 12: 0.05}},
        {"name": "C_unseen_only", "alphas": {2: 0.10, 5: 0.08, 9: 0.0, 10: 0.0, 12: 0.08}},
        {"name": "D_stronger_unseen", "alphas": {2: 0.15, 5: 0.12, 9: 0.0, 10: 0.0, 12: 0.12}},
        {"name": "E_aggressive_unseen", "alphas": {2: 0.20, 5: 0.15, 9: 0.0, 10: 0.0, 12: 0.15}},
        # Uniform
        {"name": "F_uniform_005", "alphas": {2: 0.05, 5: 0.05, 9: 0.05, 10: 0.05, 12: 0.05}},
        {"name": "G_uniform_010", "alphas": {2: 0.10, 5: 0.10, 9: 0.10, 10: 0.10, 12: 0.10}},
    ]

    for tau in [0.25, 0.35, 0.50]:
        print(f"\n  tau = {tau}:", flush=True)
        for config in alpha_configs:
            name = config["name"]
            alpha_map = config["alphas"]

            def make_pp(am=alpha_map, t=tau):
                def pp_fn(preds, test_df, test_months, train_df, y_tr):
                    counts_tr = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
                    p_tr = counts_tr / counts_tr.sum()
                    out, _ = apply_mlls_calibration(preds, test_months, p_tr, am, tau=t)
                    return out
                return pp_fn

            result = eval_pp(make_pp(), verbose=False)
            cal_lb = result.get("calibrated_lb", "N/A")
            iw = result["estimated_lb"]
            delta = result["estimated_delta"]
            shared = result["shared_delta"]
            rec = result["recommendation"]
            print(f"    {name:<25s}  IW={iw:.4f} d={delta:+.4f}  "
                  f"shared={shared:+.4f}  calLB={cal_lb}  {rec}", flush=True)

    # ============================================================
    # Composition: MLLS + NB PP
    # ============================================================
    print("\n--- Composition: MLLS calibration + NB PP ---", flush=True)

    # Best MLLS configs (pick top from sweep above, or test all)
    mlls_configs = [
        ("C_unseen", {2: 0.10, 5: 0.08, 9: 0.0, 10: 0.0, 12: 0.08}, 0.35),
        ("D_stronger", {2: 0.15, 5: 0.12, 9: 0.0, 10: 0.0, 12: 0.12}, 0.35),
        ("E_aggressive", {2: 0.20, 5: 0.15, 9: 0.0, 10: 0.0, 12: 0.15}, 0.35),
    ]

    nb_pp = make_nb_pp(tau_prior=0.15, tau_nb=0.25, gamma=0.10)

    for mlls_name, alpha_map, tau in mlls_configs:
        for order_name, apply_order in [
            ("MLLS_then_PP", "mlls_first"),
            ("PP_then_MLLS", "pp_first"),
        ]:
            def make_combined(am=alpha_map, t=tau, order=apply_order):
                def pp_fn(preds, test_df, test_months, train_df, y_tr):
                    counts_tr = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
                    p_tr = counts_tr / counts_tr.sum()

                    if order == "mlls_first":
                        # MLLS calibration first, then NB PP
                        out, _ = apply_mlls_calibration(preds, test_months, p_tr, am, tau=t)
                        out = nb_pp(out, test_df, test_months, train_df, y_tr)
                    else:
                        # NB PP first, then MLLS calibration
                        out = nb_pp(preds, test_df, test_months, train_df, y_tr)
                        out, _ = apply_mlls_calibration(out, test_months, p_tr, am, tau=t)
                    return out
                return pp_fn

            result = eval_pp(make_combined(), verbose=False)
            cal_lb = result.get("calibrated_lb", "N/A")
            iw = result["estimated_lb"]
            delta = result["estimated_delta"]
            shared = result["shared_delta"]
            print(f"  {mlls_name}+{order_name:<15s}  IW={iw:.4f} d={delta:+.4f}  "
                  f"shared={shared:+.4f}  calLB={cal_lb}  {result['recommendation']}", flush=True)

    # ============================================================
    # Save best variants as submissions
    # ============================================================
    print("\n--- Generating submission files ---", flush=True)

    # Generate raw + MLLS variants on actual test predictions
    for mlls_name, alpha_map, tau in [
        ("C_unseen", {2: 0.10, 5: 0.08, 9: 0.0, 10: 0.0, 12: 0.08}, 0.35),
        ("D_stronger", {2: 0.15, 5: 0.12, 9: 0.0, 10: 0.0, 12: 0.12}, 0.35),
    ]:
        out, n_changed = apply_mlls_calibration(test_e79, test_months, p_train, alpha_map, tau=tau)
        save_submission(out, f"e165_{mlls_name}_tau{tau:.2f}")
        print(f"  {mlls_name}: changed {n_changed} rows", flush=True)

    # MLLS + PP composition
    priors = build_gbif_priors(p_train)

    for mlls_name, alpha_map, tau in [
        ("C_unseen", {2: 0.10, 5: 0.08, 9: 0.0, 10: 0.0, 12: 0.08}, 0.35),
    ]:
        # Apply MLLS first, then standard PP
        out, _ = apply_mlls_calibration(test_e79, test_months, p_train, alpha_map, tau=tau)

        # Apply GBIF priors
        out, _ = apply_gated_ratio_priors(out, test_months, p_train, priors, BASE_ALPHA, tau=0.15)

        # Apply NB evidence
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
            test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig,
        )
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < 0.25)
        out = apply_nb_poe(out, loglike, gamma=0.10, gate=gate)

        save_submission(out, f"e165_{mlls_name}_plus_pp")
        print(f"  {mlls_name}+PP saved", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
