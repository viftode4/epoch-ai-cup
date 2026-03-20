"""E167: Improved Post-Processing with External Data Evidence.

E166 proved: adding external features to the tree model HURTS LOMO (feature dilution).
Instead, use external data as NB evidence channels in post-processing.

Strategy:
  1. Use E79 base predictions (oof_e79.npy, test_e79.npy) -- unchanged
  2. Fix GBIF priors (corvid-separated species-level proportions)
  3. Add new NB evidence channels from external data:
     - tidal_phase, tidal_hours (Waders, gravitational, month-invariant)
     - rain_occurring (Ducks vs Pigeons, month-invariant)
     - boundary_layer_height (BoP thermal soaring)
     - sea_surface_temperature (Cormorant fishing)
     - dist_to_water_m (Ducks/Cormorants, spatial)
     - dist_to_grassland_m (Geese, spatial)
     - true_airspeed (ERA5 wind correction, Alerstam priors)
  4. Evaluate with eval_pp (calibrated LB estimate)
  5. Gamma/tau sweep for optimal parameters

Key fix: pre-load all external data as columns of train_df/test_df so they
get correctly sliced by eval_pp's fold logic.
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
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin, log_gaussian,
    apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# Improved GBIF priors (corvid-separated)
# ======================================================================

def build_improved_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    """Use species-level GBIF proportions with corvid separation."""
    props_path = ROOT / "data" / "gbif_class_monthly_proportions.csv"
    if not props_path.exists():
        from src.postprocessing import build_gbif_priors
        return build_gbif_priors(p_train)

    props = pd.read_csv(props_path, index_col=0)
    priors = {}
    for month in range(1, 13):
        col = f"month_{month}"
        if col not in props.columns:
            priors[month] = p_train.copy()
            continue
        si = np.ones(N_CLASSES)
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                si[i] = 1.0
            elif cls in props.index:
                month_val = float(props.loc[cls, col])
                year_mean = float(props.loc[cls, [f"month_{m}" for m in range(1, 13)]].mean())
                si[i] = month_val / year_mean if year_mean > 1e-8 else 1.0
        raw = np.maximum(p_train * si, 1e-8)
        priors[month] = raw / raw.sum()
    return priors


# ======================================================================
# Pre-load external data into DataFrames
# ======================================================================

def augment_df_with_external(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Add external data columns to DataFrame. Row-aligned by index position."""
    ext_map = {
        "tidal": ["hours_since_high_tide", "tidal_phase"],
        "water": ["dist_to_water_m"],
        "visibility": ["rain_occurring"],
        "altitude_winds": ["boundary_layer_height", "wind_at_bird_alt"],
        "landuse": ["dist_to_grassland_m"],
        "marine": ["sea_surface_temperature"],
        "pressure": ["pressure_trend_3h"],
    }
    for csv_name, cols in ext_map.items():
        path = ROOT / "data" / f"{split}_{csv_name}.csv"
        if not path.exists():
            continue
        ext = pd.read_csv(path)
        for col in cols:
            if col in ext.columns:
                df[f"ext_{col}"] = pd.to_numeric(ext[col], errors="coerce").fillna(0).values
    # Derived: true airspeed
    if "ext_wind_at_bird_alt" in df.columns:
        airspeed = pd.to_numeric(df["airspeed"], errors="coerce").fillna(0).values
        df["ext_true_airspeed"] = airspeed - df["ext_wind_at_bird_alt"].values
    return df


# ======================================================================
# PP function factory — uses DataFrame columns (fold-safe)
# ======================================================================

def make_pp_fn(
    channels: list[str],
    gamma: float = 0.10,
    tau_prior: float = 0.15,
    tau_nb: float = 0.25,
    use_improved_gbif: bool = True,
):
    """Create a PP function for eval_pp.

    channels: list of channel names. Standard channels (speed, alt_mid, alt_range)
    are always included. Additional channels map to ext_ columns in the DataFrame.
    """
    # Map channel names to DataFrame column names
    ch_to_col = {
        "tidal_phase": "ext_tidal_phase",
        "tidal_hours": "ext_hours_since_high_tide",
        "rain": "ext_rain_occurring",
        "blh": "ext_boundary_layer_height",
        "sst": "ext_sea_surface_temperature",
        "water_dist": "ext_dist_to_water_m",
        "grassland_dist": "ext_dist_to_grassland_m",
        "true_airspeed": "ext_true_airspeed",
        "pressure_trend": "ext_pressure_trend_3h",
    }
    ch_weights = {
        "tidal_phase": 0.5, "tidal_hours": 0.4, "rain": 0.5,
        "blh": 0.3, "sst": 0.3, "water_dist": 0.3,
        "grassland_dist": 0.3, "true_airspeed": 0.5,
        "pressure_trend": 0.2,
    }
    ch_min_sigma = {
        "tidal_phase": 0.10, "tidal_hours": 0.5, "rain": 0.10,
        "blh": 50.0, "sst": 1.0, "water_dist": 50.0,
        "grassland_dist": 100.0, "true_airspeed": 1.0,
        "pressure_trend": 0.5,
    }

    def pp_fn(preds, test_df, test_months, train_df, y_local):
        counts_l = np.bincount(y_local, minlength=N_CLASSES).astype(float)
        p_train_l = counts_l / counts_l.sum()

        # Stage 1: GBIF priors
        if use_improved_gbif:
            priors_l = build_improved_gbif_priors(p_train_l)
        else:
            from src.postprocessing import build_gbif_priors
            priors_l = build_gbif_priors(p_train_l)

        out, _ = apply_gated_ratio_priors(
            preds, test_months, p_train_l, priors_l, BASE_ALPHA, tau=tau_prior
        )

        # Stage 2: Build NB parameters from train_df columns
        speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
        min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
        max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
        cont_tr = {
            "speed": speed_tr,
            "alt_mid": 0.5 * (min_z_tr + max_z_tr),
            "alt_range": max_z_tr - min_z_tr,
        }
        weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
        min_sig = {}

        # Add external channels from DataFrame columns
        for ch in channels:
            col = ch_to_col.get(ch)
            if col and col in train_df.columns:
                cont_tr[ch] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).values.astype(float)
                weights[ch] = ch_weights.get(ch, 0.3)
                min_sig[ch] = ch_min_sigma.get(ch, 0.5)

        sl, lps, mu_l, sig_l = build_nb_params(
            train_df, y_local, cont_tr, min_sigma=min_sig
        )

        # Stage 3: Test channels
        speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
        min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
        max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
        cont_te = {
            "speed": speed_te,
            "alt_mid": 0.5 * (min_z_te + max_z_te),
            "alt_range": max_z_te - min_z_te,
        }
        for ch in channels:
            col = ch_to_col.get(ch)
            if col and col in test_df.columns:
                cont_te[ch] = pd.to_numeric(test_df[col], errors="coerce").fillna(0).values.astype(float)

        ll = compute_log_p_u_given_c(
            test_df, sl, lps, cont_te, weights, None, mu_l, sig_l
        )
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
        return apply_nb_poe(out, ll, gamma=gamma, gate=gate)

    return pp_fn


# ======================================================================
# MAIN
# ======================================================================

print("=" * 70, flush=True)
print("E167: IMPROVED PP WITH EXTERNAL DATA EVIDENCE".center(70), flush=True)
print("=" * 70, flush=True)

# Load and augment data
train_df = load_train()
test_df = load_test()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

# Pre-load external data into DataFrames
print("Loading external data into DataFrames...", flush=True)
train_df = augment_df_with_external(train_df, "train")
test_df = augment_df_with_external(test_df, "test")
ext_cols = [c for c in train_df.columns if c.startswith("ext_")]
print(f"  Added {len(ext_cols)} external columns: {ext_cols}", flush=True)

# Check base predictions exist
for pred_name in ["oof_e79.npy", "test_e79.npy"]:
    if not (ROOT / pred_name).exists():
        print(f"  ERROR: {pred_name} not found", flush=True)
        sys.exit(1)

test_base = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))

# Monkey-patch src.validate to use our augmented DataFrames
import src.validate as _val
_val._cache.clear()
_val._cache["train"] = (train_df, y,
    pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values)

from src.validate import eval_pp

# ── Exp 1: Baseline vs Improved GBIF priors ────────────────────────
print("\n--- Exp 1: GBIF Priors Comparison ---", flush=True)

print("\n  [A] Original PP (reference):", flush=True)
pp_orig = make_pp_fn(channels=[], use_improved_gbif=False)
r_orig = eval_pp(pp_orig, verbose=True)

print("\n  [B] Improved GBIF priors:", flush=True)
pp_igbif = make_pp_fn(channels=[], use_improved_gbif=True)
r_igbif = eval_pp(pp_igbif, verbose=True)

# ── Exp 2: Individual channel contributions ─────────────────────────
print("\n--- Exp 2: Individual Channel Contributions ---", flush=True)

channel_options = [
    "tidal_phase", "tidal_hours", "rain", "blh", "sst",
    "water_dist", "grassland_dist", "true_airspeed", "pressure_trend",
]

ref_iw = r_igbif["estimated_lb"]
channel_results = {}
for ch in channel_options:
    pp = make_pp_fn(channels=[ch], use_improved_gbif=True, gamma=0.10, tau_nb=0.25)
    r = eval_pp(pp, verbose=False)
    delta = r["estimated_lb"] - ref_iw
    cal = r.get("calibrated_lb")
    cal_str = f"{cal:.3f}" if cal else "N/A"
    channel_results[ch] = r
    print(f"  + {ch:20s}: IW-mAP={r['estimated_lb']:.4f} (delta={delta:+.4f}) cal_LB={cal_str}", flush=True)

# Sort by contribution
sorted_ch = sorted(channel_results, key=lambda k: channel_results[k]["estimated_lb"], reverse=True)
helpful = [ch for ch in sorted_ch if channel_results[ch]["estimated_lb"] >= ref_iw - 0.001]
print(f"\n  Helpful channels (>= baseline - 0.001): {helpful}", flush=True)

# ── Exp 3: Combined channels ───────────────────────────────────────
print("\n--- Exp 3: Combined Channels ---", flush=True)

combo_results = {}
for n_ch in range(1, min(len(helpful) + 1, 7)):
    channels = helpful[:n_ch]
    label = "+".join(channels)
    pp = make_pp_fn(channels=channels, use_improved_gbif=True, gamma=0.10, tau_nb=0.25)
    r = eval_pp(pp, verbose=False)
    cal = r.get("calibrated_lb")
    cal_str = f"{cal:.3f}" if cal else "N/A"
    combo_results[label] = r
    print(f"  [{n_ch} ch] {label}: IW-mAP={r['estimated_lb']:.4f} cal_LB={cal_str}", flush=True)

# ── Exp 4: Gamma/tau sweep on best combo ────────────────────────────
print("\n--- Exp 4: Gamma/Tau Sweep ---", flush=True)

best_combo = max(combo_results, key=lambda k: combo_results[k]["estimated_lb"])
best_channels = best_combo.split("+")
print(f"  Sweeping on channels: {best_channels}", flush=True)

sweep_results = {}
for gamma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    for tau in [0.20, 0.25, 0.30, 0.35]:
        pp = make_pp_fn(
            channels=best_channels, use_improved_gbif=True,
            gamma=gamma, tau_nb=tau,
        )
        r = eval_pp(pp, verbose=False)
        key = f"g={gamma:.2f}_t={tau:.2f}"
        sweep_results[key] = r

# Print sweep table
print(f"\n  {'Config':20s} {'IW-mAP':>8s} {'Cal LB':>8s} {'Unseen D':>8s} {'Shared D':>8s} {'Rec':>12s}", flush=True)
print("  " + "-" * 72, flush=True)
for key in sorted(sweep_results, key=lambda k: sweep_results[k]["estimated_lb"], reverse=True)[:15]:
    r = sweep_results[key]
    cal = f"{r['calibrated_lb']:.3f}" if r.get("calibrated_lb") else "N/A"
    rec = r.get("recommendation", "")[:12]
    unseen_d = r.get("unseen_after", 0) - r.get("unseen_before", 0)
    print(f"  {key:20s} {r['estimated_lb']:>8.4f} {cal:>8s} {unseen_d:>+8.4f} {r.get('shared_delta', 0):>+8.4f} {rec:>12s}", flush=True)

# Best config
best_key = max(sweep_results, key=lambda k: sweep_results[k]["estimated_lb"])
best_r = sweep_results[best_key]
print(f"\n  BEST: {best_key}", flush=True)
print(f"    IW-mAP:      {best_r['estimated_lb']:.4f}", flush=True)
print(f"    Calibrated:   {best_r.get('calibrated_lb', 'N/A')}", flush=True)
print(f"    Rec:          {best_r.get('recommendation', 'N/A')}", flush=True)

# ── Generate submissions ────────────────────────────────────────────
print("\n--- Generating Submissions ---", flush=True)

parts = best_key.replace("g=", "").replace("t=", "").split("_")
best_gamma = float(parts[0])
best_tau = float(parts[1])

# Apply best PP to test predictions
priors = build_improved_gbif_priors(p_train)
test_pp, n_ch = apply_gated_ratio_priors(
    test_base.copy(), test_months, p_train, priors, BASE_ALPHA, tau=0.15
)

# Build NB evidence from FULL training data
speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
cont_tr = {"speed": speed_tr, "alt_mid": 0.5*(min_z_tr+max_z_tr), "alt_range": max_z_tr-min_z_tr}
weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
min_sig_map = {
    "tidal_phase": 0.10, "tidal_hours": 0.5, "rain": 0.10,
    "blh": 50.0, "sst": 1.0, "water_dist": 50.0,
    "grassland_dist": 100.0, "true_airspeed": 1.0,
    "pressure_trend": 0.5,
}
ch_weight_map = {
    "tidal_phase": 0.5, "tidal_hours": 0.4, "rain": 0.5,
    "blh": 0.3, "sst": 0.3, "water_dist": 0.3,
    "grassland_dist": 0.3, "true_airspeed": 0.5,
    "pressure_trend": 0.2,
}
ch_col_map = {
    "tidal_phase": "ext_tidal_phase", "tidal_hours": "ext_hours_since_high_tide",
    "rain": "ext_rain_occurring", "blh": "ext_boundary_layer_height",
    "sst": "ext_sea_surface_temperature", "water_dist": "ext_dist_to_water_m",
    "grassland_dist": "ext_dist_to_grassland_m", "true_airspeed": "ext_true_airspeed",
    "pressure_trend": "ext_pressure_trend_3h",
}
min_sig = {}
for ch in best_channels:
    col = ch_col_map.get(ch)
    if col and col in train_df.columns:
        cont_tr[ch] = train_df[col].values.astype(float)
        weights[ch] = ch_weight_map.get(ch, 0.3)
        min_sig[ch] = min_sig_map.get(ch, 0.5)

sl, lps, mu_f, sig_f = build_nb_params(train_df, y, cont_tr, min_sigma=min_sig)

speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
cont_te = {"speed": speed_te, "alt_mid": 0.5*(min_z_te+max_z_te), "alt_range": max_z_te-min_z_te}
for ch in best_channels:
    col = ch_col_map.get(ch)
    if col and col in test_df.columns:
        cont_te[ch] = test_df[col].values.astype(float)

ll_f = compute_log_p_u_given_c(test_df, sl, lps, cont_te, weights, None, mu_f, sig_f)
gate_f = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < best_tau)
test_final = apply_nb_poe(test_pp, ll_f, gamma=best_gamma, gate=gate_f)

save_submission(test_final, f"e167_pp_g{best_gamma:.2f}_t{best_tau:.2f}",
                cv_map=best_r.get("calibrated_lb") or best_r["estimated_lb"])

# Conservative version
if best_gamma != 0.10 or best_tau != 0.25:
    gate_c = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < 0.25)
    test_cons = apply_nb_poe(test_pp, ll_f, gamma=0.10, gate=gate_c)
    save_submission(test_cons, "e167_conservative", cv_map=None)

# Full eval_pp report
print("\n--- Final Report ---", flush=True)
pp_final = make_pp_fn(
    channels=best_channels, use_improved_gbif=True,
    gamma=best_gamma, tau_nb=best_tau,
)
eval_pp(pp_final, verbose=True)

# Summary
print("\n" + "=" * 70, flush=True)
print("E167 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  Original PP cal LB: {r_orig.get('calibrated_lb', 'N/A')}", flush=True)
print(f"  Best PP cal LB:     {best_r.get('calibrated_lb', 'N/A')}", flush=True)
print(f"  Best channels:      {best_channels}", flush=True)
print(f"  Best config:        gamma={best_gamma:.2f}, tau={best_tau:.2f}", flush=True)
print(f"  Recommendation:     {best_r.get('recommendation', 'N/A')}", flush=True)
print("=" * 70, flush=True)
print("\nDone.", flush=True)
