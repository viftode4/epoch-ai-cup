"""Shared post-processing pipeline for NB evidence experiments (E73+).

All recent experiments follow the same 3-stage pipeline:
  1. Load base predictions -> apply_gated_ratio_priors (GBIF month priors, unseen only)
  2. Extract evidence channels (trajectory-derived or tabular)
  3. apply_nb_poe (gated Product-of-Experts evidence update)

This module centralises the ~250 lines of boilerplate that was copy-pasted across
every post-processing experiment. A new experiment only needs to define its evidence
channels (dicts of arrays) and hyperparameters, then call these functions.

Usage pattern
-------------
    from src.postprocessing import (
        UNSEEN_MONTHS, SIZE_LEVELS,
        renorm_rows, top2_margin,
        build_gbif_priors, apply_gated_ratio_priors,
        log_gaussian, build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
        extract_heading_ac1,
    )

    # Stage 1: GBIF priors (identical across all experiments)
    priors = build_gbif_priors(p_train)
    test_p, changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )

    # Stage 2: define your evidence channels (experiment-specific)
    cont_channels = {"speed": speed, "alt_mid": alt_mid, "heading_R": heading_r}
    ok_masks      = {"heading_R": heading_ok}   # omit keys that are always valid
    weights       = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5, "heading_R": 1.0}
    min_sigma     = {"heading_R": 0.10}          # override per-channel; default 0.50

    # Stage 3: NB PoE
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, train_channels, ok_masks_train, min_sigma
    )
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_p) < TAU_NB)
    loglike = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, test_channels, weights, ok_masks_test, mu, sig
    )
    out = apply_nb_poe(test_p, loglike, gamma=GAMMA, gate=gate)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# Months present in test but absent from train (hence "unseen").
UNSEEN_MONTHS: tuple[int, ...] = (2, 5, 12)

# Canonical radar_bird_size category ordering (consistent across all experiments).
SIZE_LEVELS: list[str] = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]

# Best-known GBIF alpha values for each unseen month (tuned historically).
BASE_ALPHA: dict[int, float] = {2: 0.22, 5: 0.12, 12: 0.24}


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def renorm_rows(pred: np.ndarray) -> np.ndarray:
    """Row-normalise probability matrix to sum to 1."""
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    """Return top-1 minus top-2 probability for each row (uncertainty proxy)."""
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Evaluate log N(x | mu[c], sigma[c]) for each class c.

    Args:
        x:     (n,)   observed values
        mu:    (K,)   per-class means
        sigma: (K,)   per-class std devs (>0)

    Returns:
        (n, K) log-likelihood matrix (constant terms omitted per channel)
    """
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


# ---------------------------------------------------------------------------
# GBIF prior stage (Stage 1)
# ---------------------------------------------------------------------------

def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    """Build per-month class-prior vectors from GBIF observation counts.

    Priors are ratio-scaled from the training marginal: p(y|month) ∝ p_train * SI(month).
    Clutter is held fixed at SI=1 (no ecological prior for radar artefacts).

    Args:
        p_train: (K,) training class marginal probabilities

    Returns:
        dict mapping month int (1..12) -> (K,) normalised prior vector
    """
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si: dict[int, np.ndarray] = {}
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

    priors: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_gated_ratio_priors(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
    tau: float,
) -> tuple[np.ndarray, int]:
    """Apply month-aware ratio-tilt to uncertain rows (Stage 1 of the pipeline).

    For each unseen month m, rows where top2_margin < tau are updated:
        p_new ∝ p_old * (prior[m] / p_train)^alpha_m

    Args:
        preds:     (n, K) base predictions
        months:    (n,)   integer month per track
        p_train:   (K,)   training marginal
        priors:    month -> (K,) from build_gbif_priors
        alpha_map: month -> alpha strength (0 = skip)
        tau:       uncertainty gate threshold on top2_margin

    Returns:
        (updated predictions (n, K), number of rows changed)
    """
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


# ---------------------------------------------------------------------------
# NB parameter estimation (Stage 2 / 3)
# ---------------------------------------------------------------------------

def build_nb_params(
    df: pd.DataFrame,
    y: np.ndarray,
    cont_channels: dict[str, np.ndarray],
    ok_masks: dict[str, np.ndarray] | None = None,
    min_sigma: dict[str, float] | None = None,
    default_min_sigma: float = 0.50,
    laplace: float = 1.0,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Estimate diagonal Gaussian NB parameters from training data.

    Handles both tabular evidence (speed, alt_mid, alt_range) and trajectory-derived
    channels (heading_R, rcs_ac1, etc.) through a generic dict interface.

    Args:
        df:            raw train DataFrame (for radar_bird_size)
        y:             (n,) integer class labels
        cont_channels: name -> (n,) array of continuous feature values
        ok_masks:      name -> (n,) bool mask; channels without an entry are masked
                       by np.isfinite(value). Pass None to use isfinite for all.
        min_sigma:     name -> minimum std dev (prevents over-sharp likelihoods).
                       Keys not present use default_min_sigma.
        default_min_sigma: fallback min sigma for channels not in min_sigma
        laplace:       Laplace smoothing count for the size categorical

    Returns:
        (size_levels, log_p_size (K, S), mu dict, sig dict)
        size_levels: ordered size category strings
        log_p_size:  (K, S) log P(size|class) with Laplace smoothing
        mu:          name -> (K,) per-class mean for each continuous channel
        sig:         name -> (K,) per-class std dev for each continuous channel
    """
    if ok_masks is None:
        ok_masks = {}
    if min_sigma is None:
        min_sigma = {}

    size_to_idx = {s: i for i, s in enumerate(SIZE_LEVELS)}
    size_idx = (
        df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )

    K, S = N_CLASSES, len(SIZE_LEVELS)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + laplace) / np.clip(counts_c[:, None] + laplace * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu: dict[str, np.ndarray] = {}
    sig: dict[str, np.ndarray] = {}
    for feat_name, x in cont_channels.items():
        min_s = min_sigma.get(feat_name, default_min_sigma)
        ok = ok_masks.get(feat_name, np.isfinite(x))
        x_use = np.where(ok, x, np.nan)

        gm = float(np.nanmean(x_use))
        gs = float(np.nanstd(x_use))
        if not np.isfinite(gs) or gs < min_s:
            gs = min_s

        mu_f = np.full(K, gm, dtype=float)
        sig_f = np.full(K, gs, dtype=float)
        for c in range(K):
            xc = x_use[y == c]
            ok_c = np.isfinite(xc)
            if ok_c.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > min_s else min_s

        mu[feat_name] = mu_f
        sig[feat_name] = sig_f

    return SIZE_LEVELS, log_p_size, mu, sig


def compute_log_p_u_given_c(
    df: pd.DataFrame,
    size_levels: list[str],
    log_p_size: np.ndarray,
    cont_channels: dict[str, np.ndarray],
    weights: dict[str, float],
    ok_masks: dict[str, np.ndarray] | None,
    mu: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-row log P(u|y) = log P(size|y) + sum_k w_k * log N(x_k | mu_k, sig_k).

    Args:
        df:            test DataFrame (for radar_bird_size)
        size_levels:   from build_nb_params
        log_p_size:    (K, S) from build_nb_params
        cont_channels: name -> (n,) test feature arrays
        weights:       name -> scalar weight (0 = skip channel)
        ok_masks:      name -> (n,) validity mask; None = use isfinite per channel
        mu:            from build_nb_params
        sig:           from build_nb_params

    Returns:
        (n, K) log-likelihood matrix
    """
    if ok_masks is None:
        ok_masks = {}

    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )
    loglike = log_p_size[:, size_idx].T.copy()  # (n, K)

    for feat_name, x in cont_channels.items():
        w = weights.get(feat_name, 1.0)
        if w == 0 or feat_name not in mu:
            continue
        ok = ok_masks.get(feat_name, np.isfinite(x))
        if ok.sum() == 0:
            continue
        lg = log_gaussian(np.where(np.isfinite(x), x, 0.0), mu[feat_name], sig[feat_name])
        loglike[ok] += w * lg[ok]

    return loglike


def apply_nb_poe(
    base: np.ndarray,
    log_p_u_given_c: np.ndarray,
    gamma: float,
    gate: np.ndarray,
) -> np.ndarray:
    """Apply gated Product-of-Experts evidence update.

    p_new[i] ∝ p_old[i] * exp(gamma * log P(u_i|y))  for rows where gate[i] is True.

    Args:
        base:            (n, K) predictions after the priors stage
        log_p_u_given_c: (n, K) from compute_log_p_u_given_c
        gamma:           evidence strength (0 = no update)
        gate:            (n,) bool mask of rows to update

    Returns:
        (n, K) updated and row-normalised predictions
    """
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)
    ll = log_p_u_given_c[gate]
    ll = ll - ll.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


# ---------------------------------------------------------------------------
# Trajectory evidence extractors (shared across experiments)
# ---------------------------------------------------------------------------

def apply_specialist_corrections(
    preds: np.ndarray,
    df: pd.DataFrame,
    months: np.ndarray,
    train_df: pd.DataFrame,
    y: np.ndarray,
    tau_prior: float = 0.15,
    gamma_clutter: float = 0.50,
    gamma_bop: float = 0.30,
    gamma_rescue: float = 0.20,
    tau_nb: float = 0.25,
) -> np.ndarray:
    """Per-class specialist corrections for unseen months.

    Unlike the generic NB PoE which applies the same evidence update to all
    classes, this applies class-specific correction strategies:

    1. Clutter gate: RCS > -18 dB AND Large bird -> boost Clutter
    2. BoP gate: slow flight + high curvature -> boost BoP
    3. Pigeon/Duck rescue: size + altitude + speed classifier for unseen months
    4. Size-stratified GBIF priors (per radar_bird_size category)

    Args:
        preds:     (n, K) base predictions
        df:        test DataFrame with raw columns
        months:    (n,) integer months
        train_df:  training DataFrame
        y:         (n_train,) integer class labels
        tau_prior: gate threshold for GBIF priors
        gamma_clutter: Clutter correction strength
        gamma_bop: BoP correction strength
        gamma_rescue: Pigeon/Duck rescue strength
        tau_nb:    uncertainty gate threshold

    Returns:
        (n, K) corrected predictions
    """
    # Training class proportions
    counts_train = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts_train / counts_train.sum()

    # Stage 1: Standard GBIF priors (unchanged)
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, months, p_train, priors, BASE_ALPHA, tau=tau_prior)

    # Identify unseen-month uncertain rows
    margin = top2_margin(out)
    unseen_mask = np.isin(months, UNSEEN_MONTHS)
    uncertain = unseen_mask & (margin < tau_nb)

    if uncertain.sum() == 0:
        return renorm_rows(out)

    n = len(df)

    # Extract tabular features for corrections
    rcs_mean = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float) * 0  # placeholder
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)

    # Get radar_bird_size
    size_str = df["radar_bird_size"].fillna("__UNK__").values

    # -- Class indices --
    clutter_idx = CLASSES.index("Clutter")
    bop_idx = CLASSES.index("Birds of Prey")
    pigeon_idx = CLASSES.index("Pigeons")
    duck_idx = CLASSES.index("Ducks")
    gull_idx = CLASSES.index("Gulls")
    geese_idx = CLASSES.index("Geese")
    songbird_idx = CLASSES.index("Songbirds")

    # -- Build class-conditional statistics from training data --
    tr_speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    tr_minz = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    tr_maxz = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    tr_altmid = 0.5 * (tr_minz + tr_maxz)
    tr_size = train_df["radar_bird_size"].fillna("__UNK__").values

    # -- Correction 1: Size-stratified priors --
    # Different priors per radar_bird_size category
    size_categories = ["Small bird", "Medium bird", "Large bird", "Flock"]
    size_priors = {}
    for sc in size_categories:
        sc_mask = tr_size == sc
        if sc_mask.sum() > 10:
            sc_counts = np.bincount(y[sc_mask], minlength=N_CLASSES).astype(float)
            sc_prior = sc_counts / sc_counts.sum()
        else:
            sc_prior = p_train.copy()
        size_priors[sc] = sc_prior

    # Apply size-stratified prior adjustment for uncertain unseen rows
    for i in range(n):
        if not uncertain[i]:
            continue
        sc = str(size_str[i])
        if sc in size_priors:
            prior = size_priors[sc]
            # Soft ratio tilt (similar to GBIF but per-size)
            ratio = (prior / np.maximum(p_train, 1e-12)) ** gamma_rescue
            out[i] = out[i] * ratio
            out[i] = out[i] / max(out[i].sum(), 1e-12)

    out = renorm_rows(out)

    # -- Correction 2: Clutter boost for Large + high RCS --
    # Clutter has RCS=-13.8 dB (birds: -24 to -30 dB)
    # We can't directly access per-track mean RCS from the tabular columns,
    # but we can use size as a proxy: Clutter is 67.9% Large bird
    for i in range(n):
        if not uncertain[i]:
            continue
        is_large = (str(size_str[i]) == "Large bird")
        is_slow = (speed[i] < 10.0 if np.isfinite(speed[i]) else False)
        is_low = (alt_mid[i] < 50.0 if np.isfinite(alt_mid[i]) else False)

        if is_large and is_slow:
            # Large + slow = likely Clutter
            boost = np.ones(N_CLASSES)
            boost[clutter_idx] = 1.0 + gamma_clutter * 3.0  # strong boost
            out[i] = out[i] * boost
            out[i] = out[i] / max(out[i].sum(), 1e-12)

    out = renorm_rows(out)

    # -- Correction 3: BoP boost for slow + Small bird --
    # BoP: 93.5% Small bird, mean speed 11.8 m/s, soaring behavior
    for i in range(n):
        if not uncertain[i]:
            continue
        is_small = (str(size_str[i]) == "Small bird")
        is_slow = (speed[i] < 13.0 if np.isfinite(speed[i]) else False)
        is_high = (alt_mid[i] > 100.0 if np.isfinite(alt_mid[i]) else False)

        if is_small and is_slow and is_high:
            # Small + slow + high altitude = likely BoP (soaring)
            boost = np.ones(N_CLASSES)
            boost[bop_idx] = 1.0 + gamma_bop * 2.0
            out[i] = out[i] * boost
            out[i] = out[i] / max(out[i].sum(), 1e-12)

    out = renorm_rows(out)

    # -- Standard NB evidence channels (tabular) as final layer --
    cont_tr = {"speed": tr_speed, "alt_mid": tr_altmid, "alt_range": tr_maxz - tr_minz}
    size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_te = speed
    cont_te = {"speed": speed_te, "alt_mid": alt_mid, "alt_range": max_z - min_z}
    weights = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}

    loglike = compute_log_p_u_given_c(df, size_levels, log_p_size, cont_te, weights, None, mu, sig)

    # Apply NB PoE with moderate gamma
    gate = uncertain  # already computed above
    out = apply_nb_poe(out, loglike, gamma=0.10, gate=gate)

    return out


def extract_heading_ac1(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract heading consistency (R) and RCS lag-1 autocorrelation from trajectories.

    heading_R (0..1): circular resultant length of step headings.
        High = straight flight (Geese, Pigeons, Cormorants).
        Low  = circling/erratic (BoP, Clutter).
    rcs_ac1 (~-1..1): lag-1 autocorrelation of RCS time series.
        High = regular wingbeat pattern (flappers).
        Low  = irregular / gliders.

    Args:
        df: DataFrame with 'trajectory' and 'trajectory_time' columns

    Returns:
        heading_r: (n,) array (0.0 where invalid)
        rcs_ac1:   (n,) array (0.0 where invalid)
        ok:        (n,) bool mask -- True where both values are finite (>= 6 points)
    """
    n = len(df)
    heading_r = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  heading/ac1 extraction: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 6:
                continue

            rcs = np.array([p[3] for p in pts], dtype=float)
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            rcs_c = rcs - float(np.mean(rcs))
            var_rcs = float(np.var(rcs_c))
            if var_rcs > 1e-12:
                ac1_val = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs)
                if np.isfinite(ac1_val):
                    ac1[i] = ac1_val

            parse_trajectory_time(row["trajectory_time"])  # validate parse
            dx = np.diff(lons) * 67000.0
            dy = np.diff(lats) * 111000.0
            headings = np.arctan2(dy, dx)
            if len(headings) > 1:
                R = float(np.sqrt(np.mean(np.sin(headings)) ** 2 + np.mean(np.cos(headings)) ** 2))
                if np.isfinite(R):
                    heading_r[i] = R
        except Exception:
            continue

    ok = np.isfinite(heading_r) & np.isfinite(ac1)
    heading_r = np.where(np.isfinite(heading_r), heading_r, 0.0)
    ac1 = np.where(np.isfinite(ac1), ac1, 0.0)
    print(f"  heading/ac1 valid: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)
    return heading_r, ac1, ok


# ---------------------------------------------------------------------------
# External evidence channel loaders (for NB PP — NOT base model features)
# ---------------------------------------------------------------------------

def _load_pp_csv(name: str, split: str) -> pd.DataFrame:
    """Load an aligned external CSV for PP evidence. Returns empty DF if missing."""
    path = ROOT / "data" / f"{split}_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_pp_evidence(
    df: pd.DataFrame,
    split: str,
) -> dict[str, np.ndarray]:
    """Load all available external evidence channels for NB post-processing.

    Returns dict of channel_name -> (n,) float arrays.  Channels missing
    from disk are silently skipped.

    These channels are used as EVIDENCE ONLY (PP), never as base-model features.
    """
    n = len(df)
    channels: dict[str, np.ndarray] = {}

    # --- Insect activity index (Clutter discrimination) ---
    insect = _load_pp_csv("insect", split)
    if not insect.empty and "insect_activity_index" in insect.columns:
        channels["insect_activity"] = insect["insect_activity_index"].values[:n].astype(float)

    # --- Visibility / fog / rain (Clutter + Duck) ---
    vis = _load_pp_csv("visibility", split)
    if not vis.empty:
        if "visibility_km" in vis.columns:
            channels["visibility_km"] = vis["visibility_km"].values[:n].astype(float)
        if "fog" in vis.columns:
            channels["fog"] = vis["fog"].values[:n].astype(float)
        if "rain_occurring" in vis.columns:
            channels["rain"] = vis["rain_occurring"].values[:n].astype(float)

    # --- Marine data (Cormorant/Gull/Wader habitat) ---
    marine = _load_pp_csv("marine", split)
    if not marine.empty:
        if "wave_height" in marine.columns:
            channels["wave_height"] = marine["wave_height"].values[:n].astype(float)
        if "sea_surface_temperature" in marine.columns:
            channels["sst"] = marine["sea_surface_temperature"].values[:n].astype(float)

    # --- Wind shear 80-180m (BoP thermal proxy) ---
    alt_winds = _load_pp_csv("altitude_winds", split)
    if not alt_winds.empty:
        if "wind_shear_80_180" in alt_winds.columns:
            channels["wind_shear"] = alt_winds["wind_shear_80_180"].values[:n].astype(float)
        if "direct_radiation" in alt_winds.columns:
            channels["direct_radiation"] = alt_winds["direct_radiation"].values[:n].astype(float)

    # --- Photoperiod change rate (migration trigger) ---
    photo = _load_pp_csv("photoperiod", split)
    if not photo.empty:
        if "daylength_change_rate" in photo.columns:
            channels["daylength_change"] = photo["daylength_change_rate"].values[:n].astype(float)

    # --- Natura2000 distance (Wader concentration) ---
    natura = _load_pp_csv("natura2000", split)
    if not natura.empty:
        if "dist_to_natura2000_m" in natura.columns:
            channels["natura2000_dist"] = natura["dist_to_natura2000_m"].values[:n].astype(float)

    # --- CAPE normalized (BoP thermals) ---
    cape = _load_pp_csv("cape", split)
    if not cape.empty:
        if "cape_normalized" in cape.columns:
            channels["cape_norm"] = cape["cape_normalized"].values[:n].astype(float)

    # --- Crepuscular index (Duck/Pigeon discrimination) ---
    solar = _load_pp_csv("solar", split)
    if not solar.empty:
        if "hours_since_sunrise" in solar.columns and "daylight_hours" in solar.columns:
            hrs = solar["hours_since_sunrise"].values[:n].astype(float)
            daylight = solar["daylight_hours"].values[:n].astype(float)
            hrs_since_sunset = daylight - hrs
            channels["crepuscular"] = (
                (hrs < 1.0) | (hrs_since_sunset < 1.0)
            ).astype(float)

    # --- True airspeed (wind-corrected) ---
    era5 = _load_pp_csv("era5_winds", split)
    if not era5.empty:
        airspeed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
        if "era5_wind_10m" in era5.columns and "era5_wind_dir_at_alt" in era5.columns:
            wind_speed = era5["era5_wind_10m"].values[:n].astype(float)
            wind_dir = era5["era5_wind_dir_at_alt"].values[:n].astype(float)
            # Compute track heading from trajectory
            headings = _extract_track_headings(df)
            wind_from_math = np.pi / 2.0 - np.radians(wind_dir)
            headwind = wind_speed * np.cos(wind_from_math - headings)
            true_airspeed = airspeed - headwind
            channels["true_airspeed"] = np.where(
                np.isfinite(true_airspeed), true_airspeed, airspeed
            )

            # --- Insect/wind match: speed ratio + heading difference ---
            avg_speed = airspeed
            speed_wind_ratio = np.where(
                wind_speed > 0.5,
                avg_speed / np.maximum(wind_speed, 0.5),
                np.nan,
            )
            # Heading difference (circular, 0-pi)
            heading_wind_diff = np.abs(headings - wind_from_math)
            heading_wind_diff = np.minimum(heading_wind_diff, 2 * np.pi - heading_wind_diff)
            # Composite: 1.0 when speed matches wind AND heading matches wind direction
            wind_match = np.where(
                np.isfinite(speed_wind_ratio),
                np.exp(-((speed_wind_ratio - 1.0) ** 2) / 0.5)
                * np.exp(-(heading_wind_diff ** 2) / 0.5),
                0.0,
            )
            channels["insect_wind_match"] = wind_match

    loaded = [k for k in channels.keys()]
    print(f"  PP evidence loaded: {len(loaded)} channels ({', '.join(loaded)})", flush=True)
    return channels


def _extract_track_headings(df: pd.DataFrame) -> np.ndarray:
    """Extract overall track heading (radians, math convention) for each row."""
    n = len(df)
    headings = np.zeros(n, dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) >= 2:
                lons = [p[0] for p in pts]
                lats = [p[1] for p in pts]
                dx = (lons[-1] - lons[0]) * 67000.0
                dy = (lats[-1] - lats[0]) * 111000.0
                headings[i] = np.arctan2(dy, dx)
        except Exception:
            pass
    return headings


# Alerstam 2007 literature airspeed priors (mean, std in m/s per class)
ALERSTAM_AIRSPEED: dict[str, tuple[float, float]] = {
    "Birds of Prey": (10.8, 2.4),
    "Cormorants": (14.4, 1.5),
    "Ducks": (15.6, 2.4),
    "Geese": (17.2, 2.5),
    "Gulls": (12.4, 2.2),
    "Pigeons": (15.2, 2.5),
    "Songbirds": (13.1, 2.2),
    "Waders": (14.9, 2.2),
}


def get_alerstam_speed_params() -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) arrays for the speed channel using Alerstam 2007 priors.

    These literature-derived priors don't overfit to training months, improving
    generalization for unseen months in post-processing.

    Returns:
        mu: (K,) per-class mean airspeed
        sigma: (K,) per-class std airspeed
    """
    mu = np.array([ALERSTAM_AIRSPEED.get(c, (13.0, 3.0))[0] for c in CLASSES])
    sigma = np.array([ALERSTAM_AIRSPEED.get(c, (13.0, 3.0))[1] for c in CLASSES])
    # Clutter doesn't have a literature value — use wide prior
    clutter_idx = CLASSES.index("Clutter")
    mu[clutter_idx] = 5.0   # Clutter/insects tend to be slow
    sigma[clutter_idx] = 5.0  # Wide uncertainty
    return mu, sigma
