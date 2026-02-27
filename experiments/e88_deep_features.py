"""E88: Deep Radar Physics + Time Series Features.

Add the best novel features from analysis_deep_radar_features.py and
analysis_timeseries_features.py to E79's 36 features. Validate with LOMO.

Novel feature categories:
  A. Radar physics (from extract_radar_physics_features):
     - Detection gap rate, wing loading proxy, soaring score, etc.
  B. Time series canonical features:
     - Sample entropy, autocorrelation, spectral entropy per channel
  C. Deep physics features computed inline:
     - Heading consistency, thermal score, aspect-angle-corrected RCS

We test:
  1. E79 baseline (36 features) -- LOMO reference
  2. E79 + best physics features
  3. E79 + best TS features
  4. E79 + physics + TS (full combo)
  5. Also compute SKF for the best variant -> submit if promising
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.features import ALL_TEMPORAL, build_features, haversine
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42
RADAR_LAT, RADAR_LON = 53.44, 6.83

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]


# ======================================================================
# TIME SERIES FEATURE HELPERS
# ======================================================================

def ts_autocorrelation(x, lag):
    n = len(x)
    if n <= lag + 1:
        return 0.0
    xc = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    ac = np.mean(xc[:n-lag] * xc[lag:]) / var
    return float(ac) if np.isfinite(ac) else 0.0


def ts_first_zero_acf(x, max_lag=20):
    n = len(x)
    max_lag = min(max_lag, n // 3)
    if max_lag < 2:
        return 0
    prev = ts_autocorrelation(x, 1)
    for lag in range(2, max_lag + 1):
        curr = ts_autocorrelation(x, lag)
        if prev > 0 and curr <= 0:
            return lag
        prev = curr
    return max_lag


def ts_sample_entropy(x, m=2, r_frac=0.2, max_n=80):
    if len(x) > max_n:
        idx = np.linspace(0, len(x)-1, max_n).astype(int)
        x = x[idx]
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_frac * np.std(x)
    if r < 1e-12:
        return 0.0
    def count_matches(dim):
        templates = np.array([x[i:i+dim] for i in range(n - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count
    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0:
        return 0.0
    return float(-np.log(max(A, 1) / max(B, 1)))


def ts_zero_crossing_rate(x):
    if len(x) < 2:
        return 0.0
    xc = x - np.mean(x)
    return float(np.sum(xc[:-1] * xc[1:] < 0) / (len(x) - 1))


def ts_spectral_entropy(x, fs=1.0):
    n = len(x)
    if n < 8:
        return 0.0
    nperseg = min(n, 64)
    _, psd = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    psd = psd[1:]
    if psd.sum() < 1e-12:
        return 0.0
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]
    ent = -np.sum(psd_norm * np.log(psd_norm))
    max_ent = np.log(len(psd_norm))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def ts_turning_points(x):
    if len(x) < 3:
        return 0.0
    peaks = 0
    for i in range(1, len(x) - 1):
        if (x[i] > x[i-1] and x[i] > x[i+1]) or (x[i] < x[i-1] and x[i] < x[i+1]):
            peaks += 1
    return float(peaks / (len(x) - 2))


# ======================================================================
# EXTRACT ALL NEW FEATURES FOR ONE TRACK
# ======================================================================

def extract_novel_features(row):
    """Extract physics + time series features for one track."""
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    n = len(pts)

    feat = {}
    if n < 6:
        return feat

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    dt = np.maximum(np.diff(times), 0.001)
    duration = times[-1] - times[0]
    median_dt = np.median(dt)

    dx = np.diff(lons) * 67000
    dy = np.diff(lats) * 111000
    speeds = np.sqrt(dx**2 + dy**2) / dt
    headings = np.arctan2(dy, dx)

    # ===== PHYSICS FEATURES =====

    # 1. Detection gaps (body size proxy)
    gap_threshold = max(median_dt * 1.5, 1.5)
    is_gap = dt > gap_threshold
    feat["nf_gap_fraction"] = float(is_gap.mean())
    feat["nf_gap_rate_per_min"] = float(is_gap.sum() / max(duration/60, 0.01))

    # 2. Wing loading proxy
    rcs_linear = 10.0 ** (np.mean(rcs) / 10.0)
    feat["nf_wing_loading"] = float(np.mean(speeds)**2 / max(rcs_linear, 1e-10))

    # 3. Heading consistency (circular R)
    if len(headings) > 1:
        R = np.sqrt(np.sin(headings).mean()**2 + np.cos(headings).mean()**2)
        feat["nf_heading_consistency"] = float(R)
    else:
        feat["nf_heading_consistency"] = 0.0

    # 4. 3D soaring score + thermal score
    if len(headings) > 2:
        bc = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))
        ar = np.diff(alts[:-1]) / dt[:-1]
        ml = min(len(bc), len(ar))
        feat["nf_soaring_score"] = float(((bc[:ml] > 0.1) & (ar[:ml] > 0.5)).mean())
        feat["nf_thermal_score"] = float(((bc[:ml] > 0.2) & (ar[:ml] > 1.0)).mean())
    else:
        feat["nf_soaring_score"] = 0.0
        feat["nf_thermal_score"] = 0.0

    # 5. Aspect-angle-corrected RCS
    dx_r = (lons - RADAR_LON) * 67000
    dy_r = (lats - RADAR_LAT) * 111000
    angle_to_radar = np.arctan2(dy_r, dx_r)
    bird_heading = np.concatenate([headings, [headings[-1]]])
    aspect = (bird_heading - angle_to_radar + np.pi) % (2*np.pi) - np.pi
    abs_aspect = np.abs(aspect)
    broadside = (abs_aspect > np.pi/4) & (abs_aspect < 3*np.pi/4)
    headtail = ~broadside
    if broadside.sum() > 0 and headtail.sum() > 0:
        feat["nf_rcs_aspect_diff"] = float(rcs[broadside].mean() - rcs[headtail].mean())
    else:
        feat["nf_rcs_aspect_diff"] = 0.0

    # 6. Radar distance std
    dist_r = np.sqrt(dx_r**2 + dy_r**2)
    feat["nf_radar_dist_std"] = float(dist_r.std())

    # 7. Altitude profile
    ad = np.diff(alts)
    feat["nf_frac_climbing"] = float((ad > 0.5).mean())
    feat["nf_frac_level"] = float((np.abs(ad) <= 0.5).mean())
    feat["nf_alt_net_rate"] = float((alts[-1] - alts[0]) / max(duration, 0.01))

    # 8. Speed persistence
    if len(speeds) > 5 and np.std(speeds) > 1e-6:
        sc = speeds - np.mean(speeds)
        feat["nf_speed_ac1"] = float(np.mean(sc[:-1]*sc[1:]) / max(np.var(speeds), 1e-10))
    else:
        feat["nf_speed_ac1"] = 0.0

    # ===== TIME SERIES FEATURES (per channel) =====

    # RCS channel
    feat["nf_rcs_sampen"] = ts_sample_entropy(rcs)
    feat["nf_rcs_ac1"] = ts_autocorrelation(rcs, 1)
    feat["nf_rcs_first_zero"] = float(ts_first_zero_acf(rcs))
    feat["nf_rcs_spec_ent"] = ts_spectral_entropy(rcs)

    # Altitude channel
    feat["nf_alt_sampen"] = ts_sample_entropy(alts)
    feat["nf_alt_ac5"] = ts_autocorrelation(alts, min(5, n-2))
    feat["nf_alt_first_zero"] = float(ts_first_zero_acf(alts))
    feat["nf_alt_zcr"] = ts_zero_crossing_rate(alts)
    feat["nf_alt_skewness"] = float(skew(alts, nan_policy='omit'))

    # Speed channel
    feat["nf_spd_sampen"] = ts_sample_entropy(speeds)
    feat["nf_spd_ac1"] = ts_autocorrelation(speeds, 1)
    feat["nf_spd_ac3"] = ts_autocorrelation(speeds, min(3, len(speeds)-2))
    feat["nf_spd_skewness"] = float(skew(speeds, nan_policy='omit'))
    feat["nf_spd_kurtosis"] = float(kurtosis(speeds, nan_policy='omit'))

    # Bearing channel
    if len(headings) > 2:
        bc_arr = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
        feat["nf_brg_sampen"] = ts_sample_entropy(bc_arr)
        feat["nf_brg_zcr"] = ts_zero_crossing_rate(bc_arr)
        feat["nf_brg_kurtosis"] = float(kurtosis(bc_arr, nan_policy='omit'))
    else:
        feat["nf_brg_sampen"] = 0.0
        feat["nf_brg_zcr"] = 0.0
        feat["nf_brg_kurtosis"] = 0.0

    # Cross-channel
    rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])
    if np.std(rcs_mid) > 1e-6 and np.std(speeds) > 1e-6:
        cc = np.corrcoef(rcs_mid, speeds)[0, 1]
        feat["nf_rcs_speed_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["nf_rcs_speed_corr"] = 0.0

    # Clean
    for k in feat:
        if not np.isfinite(feat[k]):
            feat[k] = 0.0

    return feat


# ======================================================================
# HELPERS
# ======================================================================

def add_weather_solar(train_feats, test_feats):
    train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
    test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
    for col in train_weather.columns:
        train_feats[f"wx_{col}"] = train_weather[col].values
        test_feats[f"wx_{col}"] = test_weather[col].values
    train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
    test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
    for col in train_solar.columns:
        train_feats[f"sol_{col}"] = train_solar[col].values
        test_feats[f"sol_{col}"] = test_solar[col].values
    return train_feats, test_feats


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


# ======================================================================
print("=" * 70, flush=True)
print("E88: DEEP FEATURES EXPERIMENT".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data ---------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# -- Build E79 base features -------------------------------------------
print("\nBuilding E79 base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

available = [f for f in KEEP_FEATURES if f in train_feats.columns]
X_base_train = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0)
X_base_test = test_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"  Base features: {X_base_train.shape[1]}", flush=True)

# -- Extract novel features for train + test ---------------------------
print("\nExtracting novel features (train)...", flush=True)
novel_train = []
for idx, (_, row) in enumerate(train_df.iterrows()):
    if idx % 500 == 0:
        print(f"  {idx}/{len(train_df)}", flush=True)
    novel_train.append(extract_novel_features(row))
print(f"  {len(train_df)}/{len(train_df)} done", flush=True)
novel_train_df = pd.DataFrame(novel_train).replace([np.inf, -np.inf], np.nan).fillna(0)

print("Extracting novel features (test)...", flush=True)
novel_test = []
for idx, (_, row) in enumerate(test_df.iterrows()):
    if idx % 500 == 0:
        print(f"  {idx}/{len(test_df)}", flush=True)
    novel_test.append(extract_novel_features(row))
print(f"  {len(test_df)}/{len(test_df)} done", flush=True)
novel_test_df = pd.DataFrame(novel_test).replace([np.inf, -np.inf], np.nan).fillna(0)

novel_cols = list(novel_train_df.columns)
print(f"  Novel features: {len(novel_cols)}", flush=True)

# Identify physics vs TS subsets
physics_cols = [c for c in novel_cols if not any(
    c.startswith(f"nf_{ch}_") for ch in ["rcs_sampen", "rcs_ac1", "rcs_first_zero",
    "rcs_spec_ent", "alt_sampen", "alt_ac5", "alt_first_zero", "alt_zcr", "alt_skewness",
    "spd_sampen", "spd_ac1", "spd_ac3", "spd_skewness", "spd_kurtosis",
    "brg_sampen", "brg_zcr", "brg_kurtosis", "rcs_speed_corr"]
)]
# Actually let me split more carefully
physics_cols = [c for c in novel_cols if c in [
    "nf_gap_fraction", "nf_gap_rate_per_min", "nf_wing_loading",
    "nf_heading_consistency", "nf_soaring_score", "nf_thermal_score",
    "nf_rcs_aspect_diff", "nf_radar_dist_std",
    "nf_frac_climbing", "nf_frac_level", "nf_alt_net_rate",
    "nf_speed_ac1",
]]
ts_cols = [c for c in novel_cols if c not in physics_cols]

print(f"  Physics features: {len(physics_cols)}", flush=True)
print(f"  Time series features: {len(ts_cols)}", flush=True)
print(f"  Physics: {physics_cols}", flush=True)
print(f"  TS: {ts_cols}", flush=True)

# -- Build feature variants --------------------------------------------
variants = {
    "base_36": (X_base_train.values, X_base_test.values),
    "base+physics": (
        np.hstack([X_base_train.values, novel_train_df[physics_cols].values]),
        np.hstack([X_base_test.values, novel_test_df[physics_cols].values]),
    ),
    "base+ts": (
        np.hstack([X_base_train.values, novel_train_df[ts_cols].values]),
        np.hstack([X_base_test.values, novel_test_df[ts_cols].values]),
    ),
    "base+all": (
        np.hstack([X_base_train.values, novel_train_df.values]),
        np.hstack([X_base_test.values, novel_test_df.values]),
    ),
}

# Feature names for each variant
var_names = {
    "base_36": list(available),
    "base+physics": list(available) + physics_cols,
    "base+ts": list(available) + ts_cols,
    "base+all": list(available) + novel_cols,
}


# ======================================================================
# LOMO VALIDATION
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("LOMO VALIDATION".center(70), flush=True)
print("=" * 70, flush=True)

# Effective number weights
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

lomo_results = {}

for var_name, (X_tr, X_te) in variants.items():
    X_tr = X_tr.astype(np.float32)
    print(f"\n--- {var_name} ({X_tr.shape[1]} features) ---", flush=True)

    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]

        # LGB
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X_tr[tr_idx], y[tr_idx], eval_set=[(X_tr[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X_tr[va_idx])

        # CB
        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED,
            verbose=0, early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X_tr[tr_idx], y[tr_idx], eval_set=(X_tr[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X_tr[va_idx])

    # Find best weight on LOMO OOF
    best_w_lgb = 0.5
    best_lomo = -1
    for w in np.arange(0.0, 1.05, 0.1):
        oof_ens = w * oof_lgb + (1-w) * oof_cb
        m, _ = compute_map(y, oof_ens)
        if m > best_lomo:
            best_lomo = m
            best_w_lgb = w

    oof_ens = best_w_lgb * oof_lgb + (1 - best_w_lgb) * oof_cb
    lomo_map, per_class = compute_map(y, oof_ens)
    lomo_results[var_name] = (lomo_map, per_class, best_w_lgb)

    lgb_map, _ = compute_map(y, oof_lgb)
    cb_map, _ = compute_map(y, oof_cb)
    print(f"  LGB LOMO: {lgb_map:.4f}", flush=True)
    print(f"  CB  LOMO: {cb_map:.4f}", flush=True)
    print(f"  Ens LOMO: {lomo_map:.4f} (w_lgb={best_w_lgb:.1f})", flush=True)
    print_results(lomo_map, per_class, label=f"{var_name} LOMO")


# ======================================================================
# COMPARISON
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("LOMO COMPARISON".center(70), flush=True)
print("=" * 70, flush=True)

base_lomo = lomo_results["base_36"][0]
print(f"\n  {'Variant':20s} {'LOMO':>8s} {'Delta':>8s} {'N_feat':>8s}", flush=True)
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}", flush=True)
for var_name in ["base_36", "base+physics", "base+ts", "base+all"]:
    lomo, per, w = lomo_results[var_name]
    n_feat = variants[var_name][0].shape[1]
    delta = lomo - base_lomo
    print(f"  {var_name:20s} {lomo:8.4f} {delta:+8.4f} {n_feat:8d}", flush=True)

# Per-class comparison for the best variant
best_var = max(lomo_results, key=lambda k: lomo_results[k][0])
print(f"\n  Best variant: {best_var}", flush=True)

base_per = lomo_results["base_36"][1]
best_per = lomo_results[best_var][1]
print(f"\n  {'Class':20s} {'Base':>8s} {'Best':>8s} {'Delta':>8s}", flush=True)
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}", flush=True)
for i, cls in enumerate(CLASSES):
    b = base_per[i]
    n = best_per[i]
    print(f"  {cls:20s} {b:8.4f} {n:8.4f} {n-b:+8.4f}", flush=True)


# ======================================================================
# IF LOMO IMPROVED: also run SKF for submission
# ======================================================================
if lomo_results[best_var][0] > base_lomo + 0.001:
    print(f"\n{'='*70}", flush=True)
    print(f"LOMO improved! Running SKF for {best_var}".center(70), flush=True)
    print(f"{'='*70}", flush=True)

    X_tr, X_te = variants[best_var]
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb = np.zeros((len(X_te), N_CLASSES), dtype=np.float64)
    test_cb = np.zeros((len(X_te), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        print(f"  SKF Fold {fold_i+1}/5", flush=True)
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X_tr[tr_idx], y[tr_idx], eval_set=[(X_tr[va_idx], y[va_idx])])
        oof_lgb[va_idx] = lgb.predict_proba(X_tr[va_idx])
        test_lgb += lgb.predict_proba(X_te) / 5

        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED,
            verbose=0, early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X_tr[tr_idx], y[tr_idx], eval_set=(X_tr[va_idx], y[va_idx]), verbose=0)
        oof_cb[va_idx] = cb.predict_proba(X_tr[va_idx])
        test_cb += cb.predict_proba(X_te) / 5

    w = lomo_results[best_var][2]
    oof_ens = w * oof_lgb + (1-w) * oof_cb
    test_ens = w * test_lgb + (1-w) * test_cb

    skf_map, skf_per = compute_map(y, oof_ens)
    print_results(skf_map, skf_per, label=f"E88 {best_var} SKF")

    np.save(ROOT / "oof_e88.npy", oof_ens)
    np.save(ROOT / "test_e88.npy", test_ens)
    save_submission(test_ens, f"e88_{best_var}", cv_map=skf_map)
else:
    print("\n  LOMO did not improve enough. No SKF run.", flush=True)
    print("  The 36 pruned features remain optimal.", flush=True)


print("\nDone.", flush=True)
