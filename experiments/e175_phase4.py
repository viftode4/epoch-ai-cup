"""E175 Phase 4: PP + ensemble + submissions.

Takes the 4 model OOF/test predictions from Phase 3 and:
1. Applies NB PP to probability-based models (CB, DRO)
2. Tests rank-power ensembles
3. Tests blending PP'd and raw predictions
4. Saves best submission variants
"""

import sys
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# Load Phase 3 predictions
oof_lgb = np.load(ROOT / "oof_e175_lgb.npy")
oof_cb = np.load(ROOT / "oof_e175_cb.npy")
oof_dro = np.load(ROOT / "oof_e175_dro.npy")
oof_xgb = np.load(ROOT / "oof_e175_xgb.npy")
test_lgb = np.load(ROOT / "test_e175_lgb.npy")
test_cb = np.load(ROOT / "test_e175_cb.npy")
test_dro = np.load(ROOT / "test_e175_dro.npy")
test_xgb = np.load(ROOT / "test_e175_xgb.npy")

print("Loaded all Phase 3 predictions.")


def eval_oof(oof, label):
    skf, pc = compute_map(y, oof)
    lomo = {}
    for held in sorted(set(months)):
        mask = months == held
        lm, _ = compute_map(y[mask], oof[mask])
        lomo[held] = lm
    lomo_avg = np.mean(list(lomo.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo.items()))
    print(f"  [{label}] SKF={skf:.4f}  LOMO={lomo_avg:.4f}  ({month_str})")
    return skf, lomo_avg, lomo, pc


def apply_nb_pp(preds, gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    """Apply standard 3-channel NB PP to test predictions."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)

    sp = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    mz = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    xz = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    ct = {"speed": sp, "alt_mid": 0.5 * (mz + xz), "alt_range": xz - mz}
    sl, lps, mu, sig = build_nb_params(train_df, y, ct)

    sp_t = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    mz_t = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    xz_t = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    ct_t = {"speed": sp_t, "alt_mid": 0.5 * (mz_t + xz_t), "alt_range": xz_t - mz_t}
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(test_df, sl, lps, ct_t, w, None, mu, sig)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)


def apply_nb_pp_oof(preds, gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    """Apply NB PP to OOF predictions (for evaluation)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, months, p_train, priors, BASE_ALPHA, tau=tau_prior)

    sp = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    mz = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    xz = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    ct = {"speed": sp, "alt_mid": 0.5 * (mz + xz), "alt_range": xz - mz}
    sl, lps, mu, sig = build_nb_params(train_df, y, ct)
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(train_df, sl, lps, ct, w, None, mu, sig)
    gate = np.isin(months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)


def rank_power_blend(preds_list, weights, power=1.5):
    """Rank-based ensemble with power averaging."""
    n = preds_list[0].shape[0]
    nc = preds_list[0].shape[1]
    out = np.zeros((n, nc))
    for c in range(nc):
        for p, w in zip(preds_list, weights):
            r = rankdata(p[:, c]) / n
            out[:, c] += w * (r ** power)
    return out


print("=" * 70)
print("  PHASE 4: PP + ENSEMBLE + SUBMISSIONS")
print("=" * 70)

# === 1. Raw model baselines (from Phase 3) ===
print("\n--- RAW BASELINES ---")
eval_oof(oof_lgb, "LGB DART raw")
eval_oof(oof_dro, "CB DRO raw")
eval_oof(0.5 * oof_lgb + 0.5 * oof_dro, "50/50 raw")

# === 2. PP on OOF (test which models benefit from PP) ===
print("\n--- PP ON OOF (gamma sweep) ---")
best_pp_lomo = 0
best_pp_config = None
for gamma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for tau in [0.20, 0.25, 0.30]:
        # PP on the best ensemble (50/50 LGB+DRO)
        oof_blend = 0.5 * oof_lgb + 0.5 * oof_dro
        oof_pp = apply_nb_pp_oof(oof_blend, gamma=gamma, tau_nb=tau)
        skf, pc = compute_map(y, oof_pp)
        lomo_vals = []
        for held in sorted(set(months)):
            mask = months == held
            lm, _ = compute_map(y[mask], oof_pp[mask])
            lomo_vals.append(lm)
        la = np.mean(lomo_vals)
        if la > best_pp_lomo:
            best_pp_lomo = la
            best_pp_config = (gamma, tau)
            best_pp_skf = skf
            print(f"  NEW BEST: g={gamma} t={tau} -> SKF={skf:.4f} LOMO={la:.4f}")

print(f"\n  Best PP config: gamma={best_pp_config[0]}, tau={best_pp_config[1]}")
print(f"  SKF={best_pp_skf:.4f}, LOMO={best_pp_lomo:.4f}")

# === 3. PP on individual models ===
print("\n--- PP ON INDIVIDUAL MODELS ---")
g, t = best_pp_config
oof_lgb_pp = apply_nb_pp_oof(oof_lgb, gamma=g, tau_nb=t)
oof_dro_pp = apply_nb_pp_oof(oof_dro, gamma=g, tau_nb=t)
oof_cb_pp = apply_nb_pp_oof(oof_cb, gamma=g, tau_nb=t)

eval_oof(oof_lgb_pp, "LGB DART + PP")
eval_oof(oof_dro_pp, "CB DRO + PP")
eval_oof(oof_cb_pp, "CB Balanced + PP")

# === 4. Ensemble sweep with PP'd predictions ===
print("\n--- ENSEMBLE SWEEP (raw + PP'd models) ---")
candidates = {
    "lgb": oof_lgb,
    "lgb_pp": oof_lgb_pp,
    "dro": oof_dro,
    "dro_pp": oof_dro_pp,
    "cb_pp": oof_cb_pp,
    "xgb": oof_xgb,
}

best_final_lomo = 0
best_final_config = None

# Try all pairs and triples
import itertools
keys = list(candidates.keys())
for n_models in [2, 3]:
    for combo in itertools.combinations(keys, n_models):
        # Try equal weights and a few skewed options
        if n_models == 2:
            weight_options = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (0.8, 0.2), (0.2, 0.8)]
        else:
            weight_options = [
                (0.5, 0.3, 0.2), (0.5, 0.2, 0.3), (0.4, 0.4, 0.2),
                (0.6, 0.2, 0.2), (0.33, 0.33, 0.34),
            ]
        for ws in weight_options:
            oof_e = sum(w * candidates[k] for w, k in zip(ws, combo))
            lvals = []
            for held in sorted(set(months)):
                mask = months == held
                lm, _ = compute_map(y[mask], oof_e[mask])
                lvals.append(lm)
            la = np.mean(lvals)
            if la > best_final_lomo:
                s, _ = compute_map(y, oof_e)
                best_final_lomo = la
                best_final_skf = s
                best_final_config = (combo, ws)

combo, ws = best_final_config
config_str = " + ".join(f"{w:.1f}*{k}" for w, k in zip(ws, combo))
print(f"  Best: {config_str}")
print(f"  SKF={best_final_skf:.4f}  LOMO={best_final_lomo:.4f}")

# Evaluate best
oof_final = sum(w * candidates[k] for w, k in zip(ws, combo))
eval_oof(oof_final, "BEST FINAL")

# === 5. Rank-power ensemble ===
print("\n--- RANK-POWER ENSEMBLE ---")
for power in [1.0, 1.25, 1.5, 2.0]:
    oof_rp = rank_power_blend([oof_lgb, oof_dro], [0.5, 0.5], power=power)
    skf_rp, _ = compute_map(y, oof_rp)
    lvals = []
    for held in sorted(set(months)):
        mask = months == held
        lm, _ = compute_map(y[mask], oof_rp[mask])
        lvals.append(lm)
    la = np.mean(lvals)
    print(f"  power={power}: SKF={skf_rp:.4f} LOMO={la:.4f}")

# === 6. Generate test submissions ===
print("\n--- GENERATING SUBMISSIONS ---")
g, t = best_pp_config

# Build test predictions for each variant
test_blend_raw = 0.5 * test_lgb + 0.5 * test_dro
test_blend_pp = apply_nb_pp(test_blend_raw, gamma=g, tau_nb=t)

test_lgb_pp = apply_nb_pp(test_lgb, gamma=g, tau_nb=t)
test_dro_pp = apply_nb_pp(test_dro, gamma=g, tau_nb=t)
test_cb_pp = apply_nb_pp(test_cb, gamma=g, tau_nb=t)

# Best ensemble from sweep
test_candidates = {
    "lgb": test_lgb, "lgb_pp": test_lgb_pp,
    "dro": test_dro, "dro_pp": test_dro_pp,
    "cb_pp": test_cb_pp, "xgb": test_xgb,
}
combo, ws = best_final_config
test_final = sum(w * test_candidates[k] for w, k in zip(ws, combo))

# Save all variants
skf_blend, _ = compute_map(y, 0.5 * oof_lgb + 0.5 * oof_dro)
save_submission(test_lgb, "e175_lgb_dart_raw", cv_map=round(compute_map(y, oof_lgb)[0], 4))
save_submission(test_lgb_pp, "e175_lgb_dart_pp", cv_map=round(compute_map(y, oof_lgb)[0], 4))
save_submission(test_blend_raw, "e175_blend_raw", cv_map=round(skf_blend, 4))
save_submission(test_blend_pp, "e175_blend_pp", cv_map=round(skf_blend, 4))
save_submission(test_final, "e175_best_final", cv_map=round(best_final_skf, 4))
save_submission(test_dro_pp, "e175_dro_pp", cv_map=round(compute_map(y, oof_dro)[0], 4))

# === SUMMARY ===
print(f"\n{'='*70}")
print(f"  PHASE 4 SUMMARY")
print(f"{'='*70}")
print(f"  {'Variant':35s} {'SKF':>7s} {'LOMO':>7s}")

variants = [
    ("LGB DART raw", compute_map(y, oof_lgb)[0],
     np.mean([compute_map(y[months==m], oof_lgb[months==m])[0] for m in sorted(set(months)) if (months==m).sum()>=10])),
    ("50/50 LGB+DRO raw", skf_blend,
     np.mean([compute_map(y[months==m], (0.5*oof_lgb+0.5*oof_dro)[months==m])[0] for m in sorted(set(months)) if (months==m).sum()>=10])),
    ("50/50 LGB+DRO + PP", best_pp_skf, best_pp_lomo),
    (f"Best ensemble ({config_str})", best_final_skf, best_final_lomo),
]
for name, skf, lomo in variants:
    print(f"  {name:35s} {skf:7.4f} {lomo:7.4f}")

print(f"\n  PP config: gamma={g}, tau={t}")
print(f"  Best ensemble: {config_str}")
print(f"\n  Reference: E170 SKF=0.7013 LOMO=0.5141, E79 LB=0.59")
print(f"{'='*70}")
