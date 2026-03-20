"""E169: Domain-Informed Feature Engineering + Movebank GPS Evidence.

Uses ALL gathered research to build hypothesis-driven features:

A. DOMAIN-ENGINEERED FEATURES (composites, not raw columns):
   1. insect_drift_score — Clutter: airspeed/wind_speed ~ 1 means insect
   2. cormorant_speed_residual — |airspeed - (0.70*wind + 14.4)| (SD=0.31!)
   3. wader_tidal_score — Gaussian(hours_since_high_tide, mu=6, sigma=1)
   4. duck_rain_score — rain * low_altitude (Ducks fly in rain, Pigeons don't)
   5. geese_grassland_score — grassland_frac * is_flock
   6. bop_thermal_score — (BLH > 500) * slow_speed
   7. true_airspeed — ground_speed - wind (Alerstam-comparable)
   8. Alerstam likelihood scores — P(true_airspeed | class) per Alerstam 2007
   9. rcs_scintillation_index — linear-scale flock detection (fixes dB issue)
   10. altitude_censoring — start/end alt, inverted-U score (migrant vs local)

B. FEATURE REPLACEMENT (not addition):
   - Keep top ~20 E79 features by LOMO importance
   - SWAP bottom ~16 E79 features for domain features
   - Total stays at 36

C. MOVEBANK GPS PROCESSING:
   - Process 40.8M GPS events into per-class speed distributions
   - Use as gold-standard NB evidence priors (not training data Gaussians)

D. ALERSTAM TRUE AIRSPEED EVIDENCE:
   - Gold-standard true airspeed distributions per class
   - Replace wind-contaminated empirical speed priors
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.features import ALL_TEMPORAL, SIZE_MAP, build_features, haversine
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# Alerstam 2007 true airspeed distributions (gold standard)
ALERSTAM = {
    "Birds of Prey": (10.8, 2.4),
    "Cormorants": (14.4, 1.5),
    "Ducks": (15.6, 2.4),
    "Geese": (17.2, 2.5),
    "Gulls": (12.4, 2.2),
    "Pigeons": (15.2, 2.5),
    "Songbirds": (13.1, 2.2),
    "Waders": (14.9, 2.2),
}


# ======================================================================
# Domain feature engineering (using external CSVs)
# ======================================================================

def add_domain_composite_features(
    feats: pd.DataFrame, df_orig: pd.DataFrame, split: str,
) -> pd.DataFrame:
    """Add domain-engineered composite features using external data.

    Each feature encodes a specific biological hypothesis targeting
    a known confusion pair. NOT raw columns from CSVs.
    """
    airspeed = pd.to_numeric(df_orig["airspeed"], errors="coerce").fillna(0).values
    min_z = pd.to_numeric(df_orig["min_z"], errors="coerce").fillna(0).values
    max_z = pd.to_numeric(df_orig["max_z"], errors="coerce").fillna(0).values
    alt_mean = 0.5 * (min_z + max_z)
    size_num = df_orig["radar_bird_size"].map(SIZE_MAP).fillna(2).values

    def load_ext(name):
        p = ROOT / "data" / f"{split}_{name}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    # ── 1. Clutter: insect drift score ──
    # Insects drift at wind speed. If track speed ~ wind speed, it's likely an insect.
    aw = load_ext("altitude_winds")
    if "wind_at_bird_alt" in aw.columns:
        wind = aw["wind_at_bird_alt"].values.astype(float)
        # Score: 1 when airspeed == wind, 0 when very different
        speed_diff = np.abs(airspeed - wind)
        feats["dom_insect_drift"] = np.exp(-speed_diff / 3.0)  # smooth decay
        # Also: airspeed/wind ratio (< 1.5 suggests passive drift)
        feats["dom_airspeed_wind_ratio"] = airspeed / np.maximum(wind, 0.5)

        # ── 2. Cormorant: speed model residual ──
        # Known: ground_speed = 0.70 * wind + 14.4 (SD=0.31)
        expected_cormorant_speed = 0.70 * wind + 14.4
        feats["dom_cormorant_residual"] = np.abs(airspeed - expected_cormorant_speed)

        # ── 7. True airspeed (wind-corrected) ──
        feats["dom_true_airspeed"] = airspeed - wind

        # ── 8. Alerstam likelihood scores per class ──
        # P(true_airspeed | class) — trees can split on these directly
        true_as = airspeed - wind
        for cls, (mu, sd) in ALERSTAM.items():
            z = (true_as - mu) / sd
            # Log-likelihood (Gaussian), shifted so max=0
            feats[f"dom_alerstam_{cls.replace(' ', '_')[:4]}"] = np.exp(-0.5 * z * z)
    else:
        feats["dom_insect_drift"] = 0.0
        feats["dom_airspeed_wind_ratio"] = 1.0
        feats["dom_cormorant_residual"] = 5.0
        feats["dom_true_airspeed"] = airspeed.copy()

    # ── 3. Wader: tidal feeding score ──
    # Waders fly to roost ~3h before high tide, feed at low tide (~6h after high)
    tidal = load_ext("tidal")
    if "hours_since_high_tide" in tidal.columns:
        h = tidal["hours_since_high_tide"].values.astype(float)
        # Gaussian centered on 6h (low tide feeding peak)
        feats["dom_wader_tidal"] = np.exp(-((h - 6.0) ** 2) / 2.0)
        # Also: rising tide = waders take flight
        if "tide_rising" in tidal.columns:
            feats["dom_tide_rising"] = tidal["tide_rising"].values.astype(float)
    else:
        feats["dom_wader_tidal"] = 0.0
        feats["dom_tide_rising"] = 0.0

    # ── 4. Duck vs Pigeon: rain separator ──
    # Ducks fly in rain + low altitude. Pigeons avoid rain entirely.
    vis = load_ext("visibility")
    if "rain_occurring" in vis.columns:
        rain = vis["rain_occurring"].values.astype(float)
        feats["dom_duck_rain"] = rain * (alt_mean < 50).astype(float)
        feats["dom_rain"] = rain
    else:
        feats["dom_duck_rain"] = 0.0
        feats["dom_rain"] = 0.0

    # ── 5. Geese: grassland foraging ──
    # Geese forage on grassland in flocks
    lu = load_ext("landuse")
    if "grassland_fraction_2km" in lu.columns:
        gf = lu["grassland_fraction_2km"].values.astype(float)
        is_flock = (size_num == 4).astype(float)  # Flock = 4
        feats["dom_geese_grassland"] = gf * is_flock
        feats["dom_grassland_frac"] = gf
    else:
        feats["dom_geese_grassland"] = 0.0
        feats["dom_grassland_frac"] = 0.0

    # ── 6. BoP: thermal soaring ──
    # BoP need high boundary layer height + slow flight
    if "boundary_layer_height" in aw.columns:
        blh = aw["boundary_layer_height"].values.astype(float)
        feats["dom_bop_thermal"] = (blh > 500).astype(float) * (airspeed < 13).astype(float)
        feats["dom_blh"] = blh
    else:
        feats["dom_bop_thermal"] = 0.0
        feats["dom_blh"] = 0.0

    # ── Spatial features (month-invariant, raw) ──
    water = load_ext("water")
    if "dist_to_water_m" in water.columns:
        feats["dom_water_dist"] = water["dist_to_water_m"].values.astype(float)

    turb = load_ext("turbines")
    if "dist_to_turbine_m" in turb.columns:
        feats["dom_turbine_dist"] = turb["dist_to_turbine_m"].values.astype(float)

    lu2 = load_ext("landuse")
    if "dist_to_grassland_m" in lu2.columns:
        feats["dom_grassland_dist"] = lu2["dist_to_grassland_m"].values.astype(float)

    return feats


# ======================================================================
# Movebank GPS processing
# ======================================================================

def process_movebank_speed_distributions() -> dict[str, tuple[float, float]]:
    """Process Movebank GPS data into per-class speed distributions.

    Returns dict: class_name -> (mean_speed_ms, std_speed_ms)
    """
    class_files = {
        "Gulls": [
            "movebank_deltatrack_gulls.csv", "movebank_lbbg_adult.csv",
            "movebank_lbbg_juvenile.csv", "movebank_lbbg_zeebrugge.csv",
            "movebank_medgull_antwerpen.csv",
        ],
        "Waders": [
            "movebank_oystercatcher_ameland.csv", "movebank_oystercatcher_balgzand.csv",
            "movebank_oystercatcher_schier.csv", "movebank_oystercatcher_vlieland.csv",
            "movebank_oystercatcher_westerschelde.csv", "movebank_oystercatcher_assen.csv",
        ],
        "Birds of Prey": [
            "movebank_marshharrier_groningen.csv", "movebank_marshharrier_waterland.csv",
            "movebank_marshharrier_antwerpen.csv",
        ],
        "Geese": [
            "movebank_goose_newyear.csv", "movebank_whitefronted_goose_family.csv",
            "movebank_whitefronted_goose_alterra.csv",
        ],
    }

    distributions = {}
    for cls, files in class_files.items():
        all_speeds = []
        for fname in files:
            path = ROOT / "data" / fname
            if not path.exists():
                continue
            # Sample for efficiency (large files)
            try:
                df = pd.read_csv(path, usecols=["timestamp", "location_lat", "location_long"])
                if len(df) > 500000:
                    df = df.sample(500000, random_state=SEED)
                df = df.sort_values("timestamp").reset_index(drop=True)

                lats = df["location_lat"].values
                lons = df["location_long"].values
                ts = pd.to_datetime(df["timestamp"])
                dt_sec = ts.diff().dt.total_seconds().values[1:]

                # Compute inter-fix speeds
                valid = (dt_sec > 0) & (dt_sec < 3600)  # 1s to 1h fixes
                if valid.sum() < 100:
                    continue
                dists = np.array([
                    haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                    for i in range(len(lons)-1)
                ])
                speeds = dists[valid] / dt_sec[valid]
                # Filter realistic bird speeds (1-35 m/s)
                speeds = speeds[(speeds > 1) & (speeds < 35)]
                all_speeds.extend(speeds.tolist())
            except Exception as e:
                print(f"  Warning: {fname}: {e}", flush=True)
                continue

        if len(all_speeds) > 100:
            speeds_arr = np.array(all_speeds)
            distributions[cls] = (float(np.mean(speeds_arr)), float(np.std(speeds_arr)))
            print(f"  Movebank {cls}: {len(all_speeds)} speeds, "
                  f"mean={distributions[cls][0]:.1f} sd={distributions[cls][1]:.1f} m/s", flush=True)
        else:
            # Fall back to Alerstam
            if cls in ALERSTAM:
                distributions[cls] = ALERSTAM[cls]
                print(f"  Movebank {cls}: insufficient data, using Alerstam", flush=True)

    # Fill missing classes from Alerstam
    for cls in CLASSES:
        if cls not in distributions and cls in ALERSTAM:
            distributions[cls] = ALERSTAM[cls]
        elif cls not in distributions:
            distributions[cls] = (13.0, 3.0)  # generic fallback

    return distributions


# ======================================================================
# MAIN
# ======================================================================

print("=" * 70, flush=True)
print("E169: DOMAIN-INFORMED FEATURES + MOVEBANK EVIDENCE".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
train_df = load_train()
test_df = load_test()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- Build trajectory features (E79 base + domain) --------------------
print("\n--- Building features ---", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass", "domain"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather/solar (E79 uses them)
for prefix, name in [("wx_", "weather"), ("sol_", "solar")]:
    tr_csv = pd.read_csv(ROOT / "data" / f"train_{name}.csv")
    te_csv = pd.read_csv(ROOT / "data" / f"test_{name}.csv")
    for col in tr_csv.columns:
        train_feats[f"{prefix}{col}"] = tr_csv[col].values
        test_feats[f"{prefix}{col}"] = te_csv[col].values

# Add domain composite features (using external CSVs)
print("Adding domain composite features...", flush=True)
train_feats = add_domain_composite_features(train_feats, train_df, "train")
test_feats = add_domain_composite_features(test_feats, test_df, "test")

print(f"  Total features: {train_feats.shape[1]}", flush=True)

# -- Feature replacement strategy -------------------------------------
print("\n--- Feature Replacement Strategy ---", flush=True)

# E79 original features
E79_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# Domain features we just added
dom_features = [c for c in train_feats.columns if c.startswith("dom_")]
print(f"  E79 features: {len(E79_FEATURES)}", flush=True)
print(f"  Domain features: {len(dom_features)}: {dom_features}", flush=True)

# LOMO importance to identify E79's weakest features
from lightgbm import LGBMClassifier

all_cols = list(train_feats.columns)
X_all = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test_all = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Quick LOMO importance ranking
print("Computing LOMO feature importance...", flush=True)
unique_months = sorted(np.unique(train_months))
importances = np.zeros(X_all.shape[1])
for month in unique_months:
    va = train_months == month
    tr = ~va
    lgb = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb.fit(X_all[tr], y[tr], sample_weight=sample_weights[tr])
    importances += lgb.feature_importances_ / len(unique_months)

# Rank E79 features by importance
e79_importances = [(f, importances[all_cols.index(f)]) for f in E79_FEATURES if f in all_cols]
e79_importances.sort(key=lambda x: x[1], reverse=True)

# Rank domain features by importance
dom_importances = [(f, importances[all_cols.index(f)]) for f in dom_features if f in all_cols]
dom_importances.sort(key=lambda x: x[1], reverse=True)

print("\n  E79 features ranked by LOMO importance:", flush=True)
for i, (f, imp) in enumerate(e79_importances):
    tag = " *BOTTOM*" if i >= len(e79_importances) - 16 else ""
    print(f"    {i+1:3d}. {f:30s} imp={imp:8.1f}{tag}", flush=True)

print("\n  Domain features ranked by LOMO importance:", flush=True)
for i, (f, imp) in enumerate(dom_importances):
    print(f"    {i+1:3d}. {f:30s} imp={imp:8.1f}", flush=True)

# -- Build candidate feature sets -------------------------------------
print("\n--- Evaluating Feature Sets ---", flush=True)

def lomo_evaluate(X_sub, y, months, sw):
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_months:
        va, tr = months == m, months != m
        lgb = LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X_sub[tr], y[tr], sample_weight=sw[tr])
        oof[va] = lgb.predict_proba(X_sub[va])
    return compute_map(y, oof)

# A: E79 baseline (36 features)
e79_idx = [all_cols.index(f) for f in E79_FEATURES if f in all_cols]
mA, pA = lomo_evaluate(X_all[:, e79_idx], y, train_months, sample_weights)
print(f"  [A] E79 original (36 feats): LOMO mAP = {mA:.4f}", flush=True)

# B: E79 top-20 + all domain features
top20_e79 = [f for f, _ in e79_importances[:20]]
setB = top20_e79 + dom_features
setB_idx = [all_cols.index(f) for f in setB if f in all_cols]
mB, pB = lomo_evaluate(X_all[:, setB_idx], y, train_months, sample_weights)
print(f"  [B] E79 top-20 + domain ({len(setB_idx)} feats): LOMO mAP = {mB:.4f}", flush=True)

# C: E79 top-25 + top domain features
top25_e79 = [f for f, _ in e79_importances[:25]]
top_dom = [f for f, _ in dom_importances[:11]]
setC = top25_e79 + top_dom
setC_idx = [all_cols.index(f) for f in setC if f in all_cols]
mC, pC = lomo_evaluate(X_all[:, setC_idx], y, train_months, sample_weights)
print(f"  [C] E79 top-25 + top-11 domain ({len(setC_idx)} feats): LOMO mAP = {mC:.4f}", flush=True)

# D: All domain features only (no E79)
dom_idx = [all_cols.index(f) for f in dom_features if f in all_cols]
if len(dom_idx) >= 5:
    mD, pD = lomo_evaluate(X_all[:, dom_idx], y, train_months, sample_weights)
    print(f"  [D] Domain only ({len(dom_idx)} feats): LOMO mAP = {mD:.4f}", flush=True)

# E: E79 full + all domain (addition, for comparison)
setE = [f for f in E79_FEATURES if f in all_cols] + dom_features
setE_idx = [all_cols.index(f) for f in setE if f in all_cols]
mE, pE = lomo_evaluate(X_all[:, setE_idx], y, train_months, sample_weights)
print(f"  [E] E79 full + domain ({len(setE_idx)} feats): LOMO mAP = {mE:.4f}", flush=True)

# F: Top-36 from ALL features by LOMO importance (unrestricted)
top36_idx = list(np.argsort(-importances)[:36])
top36_names = [all_cols[i] for i in top36_idx]
mF, pF = lomo_evaluate(X_all[:, top36_idx], y, train_months, sample_weights)
print(f"  [F] Top-36 overall ({len(top36_idx)} feats): LOMO mAP = {mF:.4f}", flush=True)
n_dom_in_top36 = sum(1 for n in top36_names if n.startswith("dom_"))
print(f"      ({n_dom_in_top36} domain features in top 36)", flush=True)

# Pick best
results = {"A": (mA, e79_idx, "E79 original"),
           "B": (mB, setB_idx, "E79 top20 + domain"),
           "C": (mC, setC_idx, "E79 top25 + top domain"),
           "E": (mE, setE_idx, "E79 + domain (addition)"),
           "F": (mF, top36_idx, "Top-36 overall")}
best_key = max(results, key=lambda k: results[k][0])
best_mAP, best_idx, best_label = results[best_key]
best_features = [all_cols[i] for i in best_idx]

print(f"\n  BEST: [{best_key}] {best_label} -> LOMO mAP = {best_mAP:.4f}", flush=True)

# -- Train ensemble on best feature set -------------------------------
print("\n--- Ensemble Training ---", flush=True)

X = X_all[:, best_idx]
X_test = X_test_all[:, best_idx]
print(f"  Features: {X.shape[1]} ({best_label})", flush=True)

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}", flush=True)

    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X[va_idx])
    test_xgb += xgb.predict_proba(X_test) / N_FOLDS

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0,
        border_count=128, loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# Ensemble weight optimization
best_w, best_ens_map = None, -1.0
for w_lgb in np.arange(0.0, 0.55, 0.05):
    for w_xgb in np.arange(0.0, 0.55, 0.05):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01 or w_cb > 1.01:
            continue
        oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        m, _ = compute_map(y, oof_ens)
        if m > best_ens_map:
            best_ens_map = m
            best_w = (w_lgb, w_xgb, w_cb)

w_lgb, w_xgb, w_cb = best_w
oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb
print(f"  Weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)

final_map, final_per = compute_map(y, oof_ens)
print_results(final_map, final_per, label=f"E169 {best_label} (SKF)")

# -- Process Movebank GPS ---------------------------------------------
print("\n--- Movebank GPS Processing ---", flush=True)
movebank_dists = process_movebank_speed_distributions()

# -- Save artifacts ---------------------------------------------------
print("\n--- Saving ---", flush=True)
np.save(ROOT / "oof_e169.npy", oof_ens)
np.save(ROOT / "test_e169.npy", test_ens)
save_submission(test_ens, "e169_domain_raw", cv_map=final_map)

# Save feature list
with open(ROOT / "data" / "best_features_e169.txt", "w") as f:
    for feat in best_features:
        f.write(feat + "\n")

# -- Apply PP with Movebank/Alerstam evidence -------------------------
print("\n--- Post-Processing with Movebank/Alerstam Evidence ---", flush=True)

from src.postprocessing import (
    renorm_rows, top2_margin, UNSEEN_MONTHS, BASE_ALPHA,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    log_gaussian,
)

p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Stage 1: GBIF priors
test_pp, n_ch = apply_gated_ratio_priors(
    test_ens.copy(), test_months, p_train, priors, BASE_ALPHA, tau=0.15
)

# Stage 2: NB evidence with MOVEBANK speed priors (not training data)
# Override mu/sig for speed channel with Movebank distributions
speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
cont_tr = {"speed": speed_tr, "alt_mid": 0.5*(min_z_tr+max_z_tr), "alt_range": max_z_tr-min_z_tr}
sl, lps, mu, sig = build_nb_params(train_df, y, cont_tr)

# Replace speed mu/sig with Movebank/Alerstam gold-standard distributions
for i, cls in enumerate(CLASSES):
    if cls in movebank_dists:
        mu_mb, sd_mb = movebank_dists[cls]
        mu["speed"][i] = mu_mb
        sig["speed"][i] = max(sd_mb, 0.5)
        print(f"  {cls}: speed prior = {mu_mb:.1f} +/- {sd_mb:.1f} m/s (Movebank/Alerstam)", flush=True)

speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
cont_te = {"speed": speed_te, "alt_mid": 0.5*(min_z_te+max_z_te), "alt_range": max_z_te-min_z_te}
w_nb = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
ll = compute_log_p_u_given_c(test_df, sl, lps, cont_te, w_nb, None, mu, sig)

gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < 0.25)
test_pp_final = apply_nb_poe(test_pp, ll, gamma=0.10, gate=gate)
save_submission(test_pp_final, "e169_domain_movebank_pp", cv_map=final_map)

# Conservative PP (gamma=0.05)
gate_cons = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < 0.20)
test_pp_cons = apply_nb_poe(test_pp, ll, gamma=0.05, gate=gate_cons)
save_submission(test_pp_cons, "e169_domain_conservative_pp", cv_map=final_map)

# -- Summary ----------------------------------------------------------
print("\n" + "=" * 70, flush=True)
print("E169 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  Feature set: [{best_key}] {best_label} ({len(best_features)} feats)", flush=True)
print(f"  LOMO mAP:  {best_mAP:.4f} (E79 baseline: {mA:.4f}, delta: {best_mAP - mA:+.4f})", flush=True)
print(f"  SKF mAP:   {final_map:.4f}", flush=True)
print(f"  Ensemble:  LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Movebank:  {len(movebank_dists)} classes with GPS speed priors", flush=True)
print(f"  Domain features used: {[f for f in best_features if f.startswith('dom_')]}", flush=True)
print(f"  Submissions: e169_domain_raw, e169_domain_movebank_pp, e169_domain_conservative_pp", flush=True)
print("=" * 70, flush=True)
print("\nDone.", flush=True)
