"""E93: Enhanced NB Post-Processing + Clean Ablation Grid.

Adds two new physics evidence channels to the NB post-processing:
  1. RCS Autocorrelation (AC1) -- wingbeat modulation, month-INVARIANT
     Cormorants 0.50-0.80, others 0.20-0.40.
  2. Heading Consistency (Circular R) -- straight-line vs circling
     R~1 migrants, R~0 soaring BoP / erratic Clutter.

These go into NB (product-of-experts), NOT into tree features.
LOMO-validated before any submission.

Ablation grid (8 variants):
  A: size+speed+alt (original) gamma=0.30 unseen  (E87 v2 ref)
  B: size+speed+alt (original) gamma=0.50 unseen  (E87 v1 ref)
  C: + AC1                     gamma=0.30 unseen
  D: + AC1                     gamma=0.50 unseen
  E: + AC1 + heading           gamma=0.30 unseen
  F: + AC1 + heading           gamma=0.50 unseen
  G: + AC1                     gamma=0.30 ALL months
  H: + AC1 + heading           gamma=0.30 ALL months
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

UNSEEN_MONTHS = (2, 5, 12)
LAPLACE = 1.0

# Per-feature MIN_SIGMA: AC1 and heading_R have scale 0-1 with std ~0.10-0.30
# Using 0.50 would destroy separation for these.
MIN_SIGMA = {
    "speed": 0.50,
    "alt_mid": 0.50,
    "alt_range": 0.50,
    "rcs_ac1": 0.10,
    "heading_R": 0.10,
}
DEFAULT_MIN_SIGMA = 0.50


# ======================================================================
# EVIDENCE EXTRACTION
# ======================================================================

def extract_nb_evidence(df):
    """Extract AC1 and heading_R from raw trajectories.

    Returns:
        ac1: array of RCS autocorrelation lag-1 values
        heading_r: array of heading consistency (circular R) values
        ok: boolean mask -- True where both values are valid
    """
    n = len(df)
    ac1 = np.full(n, np.nan)
    heading_r = np.full(n, np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            npts = len(pts)
            if npts < 6:
                continue

            rcs = np.array([p[3] for p in pts])
            lons = np.array([p[0] for p in pts])
            lats = np.array([p[1] for p in pts])

            # RCS Autocorrelation (lag-1)
            rcs_c = rcs - np.mean(rcs)
            var_rcs = np.var(rcs)
            if var_rcs > 1e-12:
                ac1_val = np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs
                if np.isfinite(ac1_val):
                    ac1[i] = float(ac1_val)

            # Heading consistency (circular resultant length R)
            times = parse_trajectory_time(row["trajectory_time"])
            dt = np.maximum(np.diff(times), 0.001)
            dx = np.diff(lons) * 67000
            dy = np.diff(lats) * 111000
            headings = np.arctan2(dy, dx)
            if len(headings) > 1:
                R = np.sqrt(np.mean(np.sin(headings))**2 +
                            np.mean(np.cos(headings))**2)
                if np.isfinite(R):
                    heading_r[i] = float(R)

        except Exception:
            continue

    ok = np.isfinite(ac1) & np.isfinite(heading_r)
    # Fill NaN with global means for the ok computation downstream
    ac1_clean = np.where(np.isfinite(ac1), ac1, 0.0)
    heading_r_clean = np.where(np.isfinite(heading_r), heading_r, 0.0)

    print(f"  Evidence valid: {ok.sum()}/{n} ({100*ok.mean():.1f}%)", flush=True)
    return ac1_clean, heading_r_clean, ok


# ======================================================================
# ENHANCED NB BUILDER
# ======================================================================

def renorm_rows(pred):
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred):
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_enhanced_nb_params(df, y_arr, ac1, heading_r, ok,
                             use_ac1=True, use_heading=True):
    """Build NB params: size + speed + alt + optionally AC1/heading."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
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

    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}
    if use_ac1:
        feats["rcs_ac1"] = ac1
    if use_heading:
        feats["heading_R"] = heading_r

    K, S = N_CLASSES, len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y_arr == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu, sig = {}, {}
    for feat_name, x in feats.items():
        min_s = MIN_SIGMA.get(feat_name, DEFAULT_MIN_SIGMA)
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        # For AC1/heading, only use OK samples
        if feat_name in ("rcs_ac1", "heading_R"):
            x_use = np.where(ok, x, np.nan)
        else:
            x_use = x
        gm = float(np.nanmean(x_use))
        gs = float(np.nanstd(x_use))
        if not np.isfinite(gs) or gs < min_s:
            gs = min_s
        for c in range(K):
            xc = x_use[y_arr == c]
            ok_c = np.isfinite(xc)
            if ok_c.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > min_s else min_s
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat_name], sig[feat_name] = mu_f, sig_f

    return size_levels, log_p_size, mu, sig


def compute_enhanced_nb_factors(df, size_levels, log_p_size, mu, sig,
                                ac1, heading_r, evidence_ok,
                                use_ac1=True, use_heading=True):
    """Compute NB likelihood factors for each sample."""
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

    ok_base = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
    loglik = log_p_size[:, size_idx].T.copy()
    if ok_base.any():
        loglik[ok_base] += log_gaussian(speed[ok_base], mu["speed"], sig["speed"])
        loglik[ok_base] += log_gaussian(alt_mid[ok_base], mu["alt_mid"], sig["alt_mid"])
        loglik[ok_base] += log_gaussian(alt_range[ok_base], mu["alt_range"], sig["alt_range"])

    # Add AC1 channel (only for samples with valid evidence)
    if use_ac1 and "rcs_ac1" in mu:
        ac1_ok = ok_base & evidence_ok
        if ac1_ok.any():
            loglik[ac1_ok] += log_gaussian(ac1[ac1_ok], mu["rcs_ac1"], sig["rcs_ac1"])

    # Add heading channel
    if use_heading and "heading_R" in mu:
        head_ok = ok_base & evidence_ok
        if head_ok.any():
            loglik[head_ok] += log_gaussian(heading_r[head_ok],
                                            mu["heading_R"], sig["heading_R"])

    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok_base


# ======================================================================
# GBIF PRIORS (same as E87)
# ======================================================================

def build_gbif_priors(p_train):
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


# ======================================================================
# WEATHER/SOLAR HELPERS (same as E86/E88)
# ======================================================================

def add_weather_solar(df_feats, split="train"):
    weather = pd.read_csv(ROOT / "data" / f"{split}_weather.csv")
    solar = pd.read_csv(ROOT / "data" / f"{split}_solar.csv")
    for col in weather.columns:
        df_feats[f"wx_{col}"] = weather[col].values
    for col in solar.columns:
        df_feats[f"sol_{col}"] = solar[col].values
    return df_feats


# ======================================================================
# MAIN
# ======================================================================
print("=" * 70, flush=True)
print("E93: ENHANCED NB POST-PROCESSING".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
unique_months = sorted(np.unique(train_months))
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train_full = counts / counts.sum()

print(f"  Train months: {unique_months}", flush=True)
print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

# -- Extract evidence from raw trajectories ---------------------------
print("\nExtracting NB evidence (train)...", flush=True)
train_ac1, train_heading, train_ev_ok = extract_nb_evidence(train_df)

print("\nExtracting NB evidence (test)...", flush=True)
test_ac1, test_heading, test_ev_ok = extract_nb_evidence(test_df)

# -- Diagnostic prints ------------------------------------------------
print("\n" + "=" * 70, flush=True)
print("  EVIDENCE DIAGNOSTICS".center(70), flush=True)
print("=" * 70, flush=True)

print(f"\n  {'Class':20s} {'AC1 mean':>10s} {'AC1 std':>10s} {'Head_R mean':>10s} {'Head_R std':>10s} {'N':>6s}", flush=True)
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}", flush=True)
for c in range(N_CLASSES):
    mask = (y == c) & train_ev_ok
    n_c = int(mask.sum())
    if n_c > 0:
        ac1_m = np.mean(train_ac1[mask])
        ac1_s = np.std(train_ac1[mask])
        hr_m = np.mean(train_heading[mask])
        hr_s = np.std(train_heading[mask])
        print(f"  {CLASSES[c]:20s} {ac1_m:10.4f} {ac1_s:10.4f} {hr_m:10.4f} {hr_s:10.4f} {n_c:6d}", flush=True)
    else:
        print(f"  {CLASSES[c]:20s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s} {n_c:6d}", flush=True)

# -- Build features for LOMO tree training ----------------------------
print("\nBuilding features for LOMO trees...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
train_feats = add_weather_solar(train_feats, "train")
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
X_all = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X_all.shape[1]}", flush=True)

# -- Class weights (same as E79) --------------------------------------
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- GBIF priors (full train) ----------------------------------------
gbif_priors = build_gbif_priors(p_train_full)

# GBIF config (from E54/E67)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15


# ======================================================================
# ABLATION VARIANT DEFINITIONS
# ======================================================================

VARIANTS = {
    "A": {"use_ac1": False, "use_heading": False, "gamma": 0.30, "scope": "unseen",
           "desc": "E87 v2 ref (original NB, g=0.30)"},
    "B": {"use_ac1": False, "use_heading": False, "gamma": 0.50, "scope": "unseen",
           "desc": "E87 v1 ref (original NB, g=0.50)"},
    "C": {"use_ac1": True, "use_heading": False, "gamma": 0.30, "scope": "unseen",
           "desc": "+AC1, g=0.30"},
    "D": {"use_ac1": True, "use_heading": False, "gamma": 0.50, "scope": "unseen",
           "desc": "+AC1, g=0.50"},
    "E": {"use_ac1": True, "use_heading": True, "gamma": 0.30, "scope": "unseen",
           "desc": "+AC1+heading, g=0.30"},
    "F": {"use_ac1": True, "use_heading": True, "gamma": 0.50, "scope": "unseen",
           "desc": "+AC1+heading, g=0.50"},
    "G": {"use_ac1": True, "use_heading": False, "gamma": 0.30, "scope": "all",
           "desc": "+AC1, g=0.30, ALL months"},
    "H": {"use_ac1": True, "use_heading": True, "gamma": 0.30, "scope": "all",
           "desc": "+AC1+heading, g=0.30, ALL months"},
}


# ======================================================================
# LOMO VALIDATION
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("  LOMO VALIDATION".center(70), flush=True)
print("=" * 70, flush=True)

# Store per-variant, per-fold results
lomo_results = {}  # variant_id -> list of (month, raw_mAP, pp_mAP)
lomo_per_class = {}  # variant_id -> list of per_class dicts

for fold_i, held_out_month in enumerate(unique_months):
    va_idx = np.where(train_months == held_out_month)[0]
    tr_idx = np.where(train_months != held_out_month)[0]

    print(f"\n--- LOMO fold {fold_i+1}/{len(unique_months)}: held-out month={held_out_month} "
          f"(train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)

    # Class distribution in validation
    va_classes = np.bincount(y[va_idx], minlength=N_CLASSES)
    absent = [CLASSES[i] for i in range(N_CLASSES) if va_classes[i] == 0]
    if absent:
        print(f"  ABSENT classes: {', '.join(absent)}", flush=True)

    # -- Train LGB + XGB + CB ensemble --------------------------------
    X_tr, y_tr = X_all[tr_idx], y[tr_idx]
    X_va, y_va = X_all[va_idx], y[va_idx]

    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    oof_lgb = lgb.predict_proba(X_va)

    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb = xgb.predict_proba(X_va)

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb = cb.predict_proba(X_va)

    # Ensemble (E79 weights: LGB=0.50, XGB=0.40, CB=0.10)
    raw_preds = renorm_rows(0.50 * oof_lgb + 0.40 * oof_xgb + 0.10 * oof_cb)

    # Baseline mAP
    raw_mAP = float(np.mean([
        average_precision_score((y_va == c).astype(int), raw_preds[:, c])
        for c in range(N_CLASSES) if (y_va == c).sum() > 0
    ]))
    print(f"  Raw ensemble mAP: {raw_mAP:.4f}", flush=True)

    # -- Build NB params from TRAIN fold for each variant config ------
    # We need separate NB params for each variant (AC1/heading inclusion)
    train_fold_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_fold_df = train_df.iloc[va_idx].reset_index(drop=True)

    tr_counts = np.bincount(y[tr_idx], minlength=N_CLASSES).astype(float)
    p_train_fold = tr_counts / tr_counts.sum()
    fold_priors = build_gbif_priors(p_train_fold)

    # -- Evaluate each variant ----------------------------------------
    for vid, vcfg in VARIANTS.items():
        # Build NB params with appropriate feature set
        sl, lps, mu, sig = build_enhanced_nb_params(
            train_fold_df, y[tr_idx],
            train_ac1[tr_idx], train_heading[tr_idx], train_ev_ok[tr_idx],
            use_ac1=vcfg["use_ac1"], use_heading=vcfg["use_heading"],
        )

        # Compute factors for validation fold
        factors, ok_feats = compute_enhanced_nb_factors(
            val_fold_df, sl, lps, mu, sig,
            train_ac1[va_idx], train_heading[va_idx], train_ev_ok[va_idx],
            use_ac1=vcfg["use_ac1"], use_heading=vcfg["use_heading"],
        )

        # Apply PP pipeline
        out = raw_preds.copy()

        # Stage 1: GBIF priors (held-out month = "unseen")
        margin = top2_margin(out)
        alpha_val = BASE_ALPHA.get(held_out_month, 0.22)
        gbif_gate = margin < TAU_PRIOR
        if gbif_gate.any():
            ratio = (fold_priors[held_out_month] / np.maximum(p_train_fold, 1e-12)) ** alpha_val
            out[gbif_gate] = out[gbif_gate] * ratio
            out = renorm_rows(out)

        # Stage 2: NB physics (scope-dependent)
        apply_nb = True
        if vcfg["scope"] == "unseen":
            # In LOMO, held-out month is always "unseen"
            apply_nb = True
        # For "all" scope, also always apply in LOMO (every fold is held-out)

        if apply_nb:
            margin2 = top2_margin(out)
            nb_gate = ok_feats & (margin2 < 0.30)
            if nb_gate.any():
                out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** vcfg["gamma"])
                out = renorm_rows(out)

        # Compute mAP after PP
        pp_aps = []
        pp_per = {}
        for c in range(N_CLASSES):
            y_bin = (y_va == c).astype(int)
            if y_bin.sum() > 0:
                ap = average_precision_score(y_bin, out[:, c])
                pp_aps.append(ap)
                pp_per[CLASSES[c]] = ap
            else:
                pp_per[CLASSES[c]] = float("nan")
        pp_mAP = float(np.mean([v for v in pp_aps if np.isfinite(v)])) if pp_aps else 0.0

        if vid not in lomo_results:
            lomo_results[vid] = []
            lomo_per_class[vid] = []
        lomo_results[vid].append({
            "month": held_out_month, "raw_mAP": raw_mAP,
            "pp_mAP": pp_mAP, "n_val": len(va_idx),
        })
        lomo_per_class[vid].append(pp_per)

    # Also store raw baseline
    if "RAW" not in lomo_results:
        lomo_results["RAW"] = []
        lomo_per_class["RAW"] = []
    raw_per = {}
    for c in range(N_CLASSES):
        y_bin = (y_va == c).astype(int)
        if y_bin.sum() > 0:
            raw_per[CLASSES[c]] = average_precision_score(y_bin, raw_preds[:, c])
        else:
            raw_per[CLASSES[c]] = float("nan")
    lomo_results["RAW"].append({
        "month": held_out_month, "raw_mAP": raw_mAP,
        "pp_mAP": raw_mAP, "n_val": len(va_idx),
    })
    lomo_per_class["RAW"].append(raw_per)


# ======================================================================
# AGGREGATE LOMO RESULTS
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("  AGGREGATE LOMO RESULTS".center(70), flush=True)
print("=" * 70, flush=True)

# Weighted-average mAP across folds
agg = {}
for vid in ["RAW"] + list(VARIANTS.keys()):
    folds = lomo_results[vid]
    total_n = sum(f["n_val"] for f in folds)
    wt_mAP = sum(f["pp_mAP"] * f["n_val"] for f in folds) / total_n
    simple_mAP = np.mean([f["pp_mAP"] for f in folds])
    raw_simple = np.mean([f["raw_mAP"] for f in folds])
    agg[vid] = {"wt_mAP": wt_mAP, "simple_mAP": simple_mAP, "raw_mAP": raw_simple}

raw_baseline = agg["RAW"]["wt_mAP"]

print(f"\n  {'ID':>4s}  {'Description':40s}  {'wt mAP':>8s}  {'delta':>8s}", flush=True)
print(f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*8}", flush=True)

# Print RAW first
print(f"  {'RAW':>4s}  {'No PP (baseline)':40s}  {raw_baseline:.4f}  {'+0.0000':>8s}", flush=True)

# Print variants sorted by wt_mAP
ranking = sorted(VARIANTS.keys(), key=lambda v: -agg[v]["wt_mAP"])
for vid in ranking:
    desc = VARIANTS[vid]["desc"]
    delta = agg[vid]["wt_mAP"] - raw_baseline
    marker = " ***" if delta > 0.005 else ""
    print(f"  {vid:>4s}  {desc:40s}  {agg[vid]['wt_mAP']:.4f}  {delta:+.4f}{marker}", flush=True)

# Per-month breakdown for top variants
print(f"\n  Per-month breakdown:", flush=True)
print(f"  {'ID':>4s}", end="", flush=True)
for m in unique_months:
    print(f"  {'M'+str(m):>8s}", end="", flush=True)
print(flush=True)

for vid in ["RAW"] + ranking[:4]:
    print(f"  {vid:>4s}", end="", flush=True)
    for fold in lomo_results[vid]:
        print(f"  {fold['pp_mAP']:8.4f}", end="", flush=True)
    print(flush=True)

# Per-class comparison: RAW vs best variant
best_vid = ranking[0]
print(f"\n  Per-class: RAW vs {best_vid} ({VARIANTS[best_vid]['desc']})", flush=True)
print(f"  {'Class':20s} {'RAW AP':>8s} {'Best AP':>8s} {'Delta':>8s}", flush=True)
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}", flush=True)

for cls in CLASSES:
    raw_vals = [f.get(cls, float("nan")) for f in lomo_per_class["RAW"]]
    best_vals = [f.get(cls, float("nan")) for f in lomo_per_class[best_vid]]
    raw_avg = np.nanmean(raw_vals)
    best_avg = np.nanmean(best_vals)
    delta = best_avg - raw_avg
    print(f"  {cls:20s} {raw_avg:8.4f} {best_avg:8.4f} {delta:+8.4f}", flush=True)

# Ablation questions answered
print(f"\n  Ablation answers:", flush=True)
d_A = agg["A"]["wt_mAP"] - raw_baseline
d_B = agg["B"]["wt_mAP"] - raw_baseline
d_C = agg["C"]["wt_mAP"] - raw_baseline
d_D = agg["D"]["wt_mAP"] - raw_baseline
d_E = agg["E"]["wt_mAP"] - raw_baseline
d_F = agg["F"]["wt_mAP"] - raw_baseline
d_G = agg["G"]["wt_mAP"] - raw_baseline
d_H = agg["H"]["wt_mAP"] - raw_baseline
print(f"  A vs B (gamma 0.30 vs 0.50):  {d_A:+.4f} vs {d_B:+.4f}", flush=True)
print(f"  C vs A (AC1 at g=0.30):       {d_C:+.4f} vs {d_A:+.4f}  delta={d_C-d_A:+.4f}", flush=True)
print(f"  E vs C (heading on top AC1):   {d_E:+.4f} vs {d_C:+.4f}  delta={d_E-d_C:+.4f}", flush=True)
print(f"  D vs B (AC1 at g=0.50):        {d_D:+.4f} vs {d_B:+.4f}  delta={d_D-d_B:+.4f}", flush=True)
print(f"  G vs C (all vs unseen months): {d_G:+.4f} vs {d_C:+.4f}  delta={d_G-d_C:+.4f}", flush=True)


# ======================================================================
# TEST SUBMISSIONS (using E79 base + enhanced NB)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("  TEST SUBMISSIONS".center(70), flush=True)
print("=" * 70, flush=True)

# Load E79 base test predictions
base_e79 = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
print(f"\n  E79 base: {base_e79.shape}", flush=True)
print(f"  Unseen: {unseen_mask.sum()} ({100*unseen_mask.mean():.1f}%)", flush=True)

# Build NB params from FULL training set
for vid, vcfg in VARIANTS.items():
    sl, lps, mu, sig = build_enhanced_nb_params(
        train_df, y,
        train_ac1, train_heading, train_ev_ok,
        use_ac1=vcfg["use_ac1"], use_heading=vcfg["use_heading"],
    )

    factors, ok_feats = compute_enhanced_nb_factors(
        test_df, sl, lps, mu, sig,
        test_ac1, test_heading, test_ev_ok,
        use_ac1=vcfg["use_ac1"], use_heading=vcfg["use_heading"],
    )

    out = base_e79.copy()

    # Stage 1: GBIF priors (unseen months only, always)
    margin = top2_margin(out)
    changed_gbif = 0
    for month, alpha in BASE_ALPHA.items():
        mask_m = test_months == month
        gate = mask_m & (margin < TAU_PRIOR)
        if gate.any():
            ratio = (gbif_priors[month] / np.maximum(p_train_full, 1e-12)) ** alpha
            out[gate] = out[gate] * ratio
            out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
            changed_gbif += int(gate.sum())
    out = renorm_rows(out)

    # Stage 2: NB physics (scope-dependent)
    if vcfg["scope"] == "unseen":
        nb_scope_mask = unseen_mask
    else:
        nb_scope_mask = np.ones(len(test_df), dtype=bool)

    margin2 = top2_margin(out)
    nb_gate = nb_scope_mask & ok_feats & (margin2 < 0.30)
    changed_nb = int(nb_gate.sum())
    if nb_gate.any():
        out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** vcfg["gamma"])
        out = renorm_rows(out)

    # Stats
    flips = int(((base_e79.argmax(1) != out.argmax(1)) & unseen_mask).sum())
    flips_shared = int(((base_e79.argmax(1) != out.argmax(1)) & ~unseen_mask).sum())
    total_flips = flips + flips_shared

    label = f"e93_{vid}_{vcfg['desc'].split(',')[0].replace(' ', '_').replace('+', '_')}"
    # Sanitize label
    label = label.replace("(", "").replace(")", "").replace("__", "_")
    if len(label) > 60:
        label = f"e93_{vid}_g{vcfg['gamma']:.2f}"

    print(f"\n  {vid}: {vcfg['desc']}", flush=True)
    print(f"    GBIF changed: {changed_gbif}", flush=True)
    print(f"    NB gated: {changed_nb}, flips unseen: {flips}, shared: {flips_shared}, total: {total_flips}", flush=True)

    # Sanity check: if >100 total flips, warn
    if total_flips > 100:
        print(f"    WARNING: {total_flips} flips seems high -- check if too aggressive", flush=True)

    # Class count changes
    top_pred = out.argmax(axis=1)
    base_pred = base_e79.argmax(axis=1)
    for i, cls in enumerate(CLASSES):
        diff = int((top_pred == i).sum()) - int((base_pred == i).sum())
        if abs(diff) >= 3:
            print(f"    {cls}: {diff:+d}", flush=True)

    save_submission(out, f"e93_{vid}_g{vcfg['gamma']:.2f}", cv_map=None)


# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("  E93 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)

print(f"\n  LOMO baseline (raw): {raw_baseline:.4f}", flush=True)
print(f"\n  Best variant: {ranking[0]} ({VARIANTS[ranking[0]]['desc']})", flush=True)
print(f"    LOMO delta: {agg[ranking[0]]['wt_mAP'] - raw_baseline:+.4f}", flush=True)
print(f"\n  Runner-up:    {ranking[1]} ({VARIANTS[ranking[1]]['desc']})", flush=True)
print(f"    LOMO delta: {agg[ranking[1]]['wt_mAP'] - raw_baseline:+.4f}", flush=True)

print(f"""
  UPLOAD PRIORITY:
  1. e93_{ranking[0]}_g{VARIANTS[ranking[0]]['gamma']:.2f} -- LOMO winner
  2. e93_{ranking[1]}_g{VARIANTS[ranking[1]]['gamma']:.2f} -- runner-up
  3. e93_B_g0.50 -- E87 v1 reference (gamma=0.50 untested on LB)
  4. e93_A_g0.30 -- E87 v2 reference (gamma=0.30)
  5. Best all-month variant if LOMO improved
""", flush=True)

print("Done.", flush=True)
