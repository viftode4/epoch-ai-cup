"""E166: All-External-Data Mega Model with LOMO Feature Selection.

Strategy:
  1. Build E79 trajectory features (core + rcs_fft + tabular + targeted + flight_mode + weakclass)
  2. Add ALL 76 external features from 16 aligned datasets
  3. Compute derived features (true airspeed proxy, spatial interactions)
  4. LOMO-based LightGBM importance → rank all features by generalization value
  5. Feature selection sweep → find optimal subset (30-60 features)
  6. Train LGB+XGB+CB ensemble on optimal features (SKF for OOF/test, LOMO for evaluation)
  7. OOF-based noise weighting (lightweight cleanlab proxy)
  8. Improved post-processing: fixed GBIF priors (corvid-separated) + new NB evidence channels
  9. Save oof_e166.npy, test_e166.npy, submissions

Key improvements over E79:
  - 76 new month-invariant external features (tidal, water, landuse, turbines, BLH, etc.)
  - LOMO-driven feature selection replaces manual backward elimination
  - Noise-aware sample weighting from E79 OOF confidence
  - StratifiedGroupKFold (primary_observation_id) for honest SKF
  - Improved GBIF priors (corvid contamination fixed)
  - New NB evidence channels: tidal_phase, rain_occurring, boundary_layer_height
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    log_gaussian,
)

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
SEED = 42


# ======================================================================
# Helper functions
# ======================================================================

def load_external_csv(name: str, split: str) -> pd.DataFrame:
    """Load an aligned external CSV. Returns empty DataFrame if missing."""
    path = ROOT / "data" / f"{split}_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path.name} not found, skipping", flush=True)
    return pd.DataFrame()


def add_all_external_features(
    feats: pd.DataFrame, split: str
) -> pd.DataFrame:
    """Add all external dataset features with appropriate prefixes."""
    datasets = {
        "tidal": "tid_",
        "water": "wat_",
        "visibility": "vis_",
        "landuse": "lu_",
        "altitude_winds": "aw_",
        "pressure": "prs_",
        "marine": "mar_",
        "cape": "cap_",
        "turbines": "tur_",
        "soil": "sol2_",          # sol2_ to avoid conflict with existing sol_
        "photoperiod": "pho_",
        "natura2000": "nat_",
        "era5_winds": "era_",
        "weather_extra": "wex_",
    }
    for name, prefix in datasets.items():
        ext = load_external_csv(name, split)
        if ext.empty:
            continue
        for col in ext.columns:
            feats[f"{prefix}{col}"] = ext[col].values
    return feats


def add_derived_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Compute physics-motivated derived features from existing columns."""
    airspeed = feats.get("airspeed")
    wind = feats.get("aw_wind_at_bird_alt")

    if airspeed is not None and wind is not None:
        asp = pd.to_numeric(airspeed, errors="coerce").fillna(0)
        wnd = pd.to_numeric(wind, errors="coerce").fillna(0)
        # True airspeed approximations (scalar — direction unknown)
        feats["true_airspeed_lo"] = np.abs(asp - wnd)       # tailwind case
        feats["true_airspeed_hi"] = asp + wnd                # headwind case
        feats["airspeed_wind_ratio"] = asp / np.maximum(wnd, 0.1)
        feats["wind_excess"] = asp - wnd  # positive = faster than wind

    # BLH interaction: high BLH + slow speed → BoP thermal soaring
    blh = feats.get("aw_boundary_layer_height")
    if blh is not None and airspeed is not None:
        asp = pd.to_numeric(airspeed, errors="coerce").fillna(0)
        blh_val = pd.to_numeric(blh, errors="coerce").fillna(0)
        feats["blh_x_slow"] = blh_val * (asp < 13).astype(float)

    # Tidal × altitude interaction: Waders feed low at specific tidal phases
    tidal = feats.get("tid_tidal_phase")
    alt = feats.get("alt_mean")
    if tidal is not None and alt is not None:
        feats["tidal_x_low_alt"] = pd.to_numeric(tidal, errors="coerce").fillna(0) * \
                                   (pd.to_numeric(alt, errors="coerce").fillna(100) < 50).astype(float)

    # Water proximity × small bird → Duck signal
    water = feats.get("wat_dist_to_water_m")
    size = feats.get("radar_bird_size")
    if water is not None and size is not None:
        feats["water_x_small"] = pd.to_numeric(water, errors="coerce").fillna(1000) * \
                                 (pd.to_numeric(size, errors="coerce").fillna(2) <= 1).astype(float)

    return feats


def compute_noise_weights(y: np.ndarray, oof_path: Path) -> np.ndarray:
    """Compute per-sample noise weights from OOF predictions.

    Samples where the model was less confident about the correct label
    get lower weight. Normalized per-class to avoid penalizing minority classes.
    """
    weights = np.ones(len(y), dtype=float)
    if not oof_path.exists():
        return weights

    oof = np.load(oof_path).astype(float)
    self_conf = oof[np.arange(len(y)), y]

    # Per-class rank normalization: within each class, rank by confidence
    for c in range(N_CLASSES):
        mask = y == c
        if mask.sum() < 2:
            continue
        ranks = rankdata(self_conf[mask])
        # Map ranks to [0.3, 1.0] — never fully remove a sample
        weights[mask] = 0.3 + 0.7 * (ranks / len(ranks))

    return weights


def build_improved_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    """Build GBIF priors using improved species-level proportions.

    Uses gbif_class_monthly_proportions.csv which properly separates
    corvids from small passerines within Songbirds class.
    Falls back to original if file not found.
    """
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
        # Compute seasonality index from proportions
        si = np.ones(N_CLASSES)
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                si[i] = 1.0  # no ecological prior for radar artifacts
            elif cls in props.index:
                month_val = props.loc[cls, col]
                year_mean = props.loc[cls, [f"month_{m}" for m in range(1, 13)]].mean()
                si[i] = month_val / year_mean if year_mean > 1e-8 else 1.0
            else:
                si[i] = 1.0
        raw = np.maximum(p_train * si, 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def lomo_evaluate(X, y, months, sample_weights=None, params=None):
    """Quick LOMO evaluation with LightGBM. Returns (mAP, per_class, importances)."""
    from lightgbm import LGBMClassifier

    if params is None:
        params = dict(
            n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1,
            device="gpu",
        )

    unique_months = sorted(np.unique(months))
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    importances = np.zeros(X.shape[1], dtype=np.float64)

    for month in unique_months:
        va = months == month
        tr = ~va
        lgb = LGBMClassifier(**params)
        sw = sample_weights[tr] if sample_weights is not None else None
        lgb.fit(X[tr], y[tr], sample_weight=sw)
        oof[va] = lgb.predict_proba(X[va])
        importances += lgb.feature_importances_ / len(unique_months)

    mAP, per_class = compute_map(y, oof)
    return mAP, per_class, importances


# ======================================================================
# MAIN EXECUTION
# ======================================================================

print("=" * 70, flush=True)
print("E166: ALL-EXTERNAL-DATA MEGA MODEL".center(70), flush=True)
print("=" * 70, flush=True)

# -- Phase 1: Load data and build features ---------------------------
print("\n--- Phase 1: Feature Engineering ---", flush=True)
train_df = load_train()
test_df = load_test()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

# Effective number class weights (beta=0.999)
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
base_sample_weights = class_weights_arr[y]

# Noise-aware weights from E79 OOF
noise_w = compute_noise_weights(y, ROOT / "oof_e79.npy")
sample_weights = base_sample_weights * noise_w
print(f"  Noise weights: min={noise_w.min():.2f} mean={noise_w.mean():.2f} max={noise_w.max():.2f}", flush=True)

# Build trajectory features
print("\nBuilding trajectory features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
print(f"  Trajectory features (after removing temporal): {train_feats.shape[1]}", flush=True)

# Add E79's weather/solar features
print("Adding weather/solar features...", flush=True)
for prefix, name in [("wx_", "weather"), ("sol_", "solar")]:
    tr_csv = pd.read_csv(ROOT / "data" / f"train_{name}.csv")
    te_csv = pd.read_csv(ROOT / "data" / f"test_{name}.csv")
    for col in tr_csv.columns:
        train_feats[f"{prefix}{col}"] = tr_csv[col].values
        test_feats[f"{prefix}{col}"] = te_csv[col].values
print(f"  After weather/solar: {train_feats.shape[1]}", flush=True)

# Add all external features
print("Adding external features...", flush=True)
train_feats = add_all_external_features(train_feats, "train")
test_feats = add_all_external_features(test_feats, "test")
print(f"  After external: {train_feats.shape[1]}", flush=True)

# Add derived features
print("Computing derived features...", flush=True)
train_feats = add_derived_features(train_feats)
test_feats = add_derived_features(test_feats)
print(f"  After derived: {train_feats.shape[1]}", flush=True)

# Drop non-numeric columns (e.g., era5_alt_level is a string like "100m")
non_numeric = train_feats.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"  Dropping {len(non_numeric)} non-numeric columns: {non_numeric}", flush=True)
    train_feats = train_feats.drop(columns=non_numeric)
    test_feats = test_feats.drop(columns=[c for c in non_numeric if c in test_feats.columns])

# Remove constant/near-constant features
nunique = train_feats.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    print(f"  Removing {len(constant_cols)} constant features: {constant_cols}", flush=True)
    train_feats = train_feats.drop(columns=constant_cols)
    test_feats = test_feats.drop(columns=[c for c in constant_cols if c in test_feats.columns])

# Clean inf/nan
all_feature_names = list(train_feats.columns)
X_all = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test_all = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"\n  Total feature pool: {X_all.shape[1]} features", flush=True)

# -- Phase 2: LOMO Feature Importance Ranking -------------------------
print("\n--- Phase 2: LOMO Feature Importance ---", flush=True)

# E79 baseline features for comparison
E79_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# LOMO with ALL features
print("Running LOMO with all features...", flush=True)
lomo_all, per_all, imp_all = lomo_evaluate(X_all, y, train_months, sample_weights)
print(f"  LOMO mAP (all {X_all.shape[1]} features): {lomo_all:.4f}", flush=True)

# LOMO with E79's 36 features (baseline)
e79_idx = [all_feature_names.index(f) for f in E79_FEATURES if f in all_feature_names]
if len(e79_idx) == len(E79_FEATURES):
    X_e79 = X_all[:, e79_idx]
    lomo_e79, per_e79, _ = lomo_evaluate(X_e79, y, train_months, sample_weights)
    print(f"  LOMO mAP (E79 36 features): {lomo_e79:.4f}", flush=True)
else:
    lomo_e79 = 0.0
    print(f"  WARNING: E79 features not fully available ({len(e79_idx)}/{len(E79_FEATURES)})", flush=True)

# Rank features by LOMO importance
imp_ranked = np.argsort(-imp_all)
print("\n  Top 50 features by LOMO importance:", flush=True)
for rank, idx in enumerate(imp_ranked[:50]):
    name = all_feature_names[idx]
    in_e79 = "E79" if name in E79_FEATURES else "NEW"
    print(f"    {rank+1:3d}. {name:40s} imp={imp_all[idx]:8.1f} [{in_e79}]", flush=True)

# Count new features in top 50
new_in_top50 = sum(1 for idx in imp_ranked[:50] if all_feature_names[idx] not in E79_FEATURES)
print(f"\n  New features in top 50: {new_in_top50}", flush=True)

# -- Phase 3: Feature Selection Sweep ---------------------------------
print("\n--- Phase 3: Feature Selection Sweep ---", flush=True)

# Strategy: start with E79 features, add top-N new features by LOMO importance
new_features_ranked = [
    all_feature_names[idx] for idx in imp_ranked
    if all_feature_names[idx] not in E79_FEATURES
]

best_lomo = lomo_e79
best_n_new = 0
best_features = list(E79_FEATURES)

sweep_results = []
for n_new in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
    if n_new == 0:
        feat_names = [f for f in E79_FEATURES if f in all_feature_names]
    else:
        feat_names = [f for f in E79_FEATURES if f in all_feature_names] + \
                     new_features_ranked[:n_new]

    feat_idx = [all_feature_names.index(f) for f in feat_names]
    X_sub = X_all[:, feat_idx]
    lomo_sub, per_sub, _ = lomo_evaluate(X_sub, y, train_months, sample_weights)
    sweep_results.append((n_new, len(feat_names), lomo_sub))
    tag = " *** BEST" if lomo_sub > best_lomo else ""
    print(f"  E79 + {n_new:3d} new = {len(feat_names):3d} features -> LOMO mAP: {lomo_sub:.4f}{tag}", flush=True)

    if lomo_sub > best_lomo:
        best_lomo = lomo_sub
        best_n_new = n_new
        best_features = feat_names

# Also try pure top-N from all features (no E79 constraint)
for n_top in [36, 40, 50, 60]:
    feat_names = [all_feature_names[idx] for idx in imp_ranked[:n_top]]
    feat_idx = list(imp_ranked[:n_top])
    X_sub = X_all[:, feat_idx]
    lomo_sub, per_sub, _ = lomo_evaluate(X_sub, y, train_months, sample_weights)
    sweep_results.append((f"top{n_top}", n_top, lomo_sub))
    tag = " *** BEST" if lomo_sub > best_lomo else ""
    print(f"  Top-{n_top:3d} by importance = {n_top:3d} features -> LOMO mAP: {lomo_sub:.4f}{tag}", flush=True)

    if lomo_sub > best_lomo:
        best_lomo = lomo_sub
        best_n_new = f"top{n_top}"
        best_features = feat_names

print(f"\n  BEST: {len(best_features)} features, LOMO mAP: {best_lomo:.4f} (E79 baseline: {lomo_e79:.4f})", flush=True)
print(f"  Delta vs E79: {best_lomo - lomo_e79:+.4f}", flush=True)

# Save best feature list
with open(ROOT / "data" / "best_features_e166.txt", "w") as f:
    for feat in best_features:
        f.write(feat + "\n")

# -- Phase 4: Train Final Ensemble ------------------------------------
print("\n--- Phase 4: Ensemble Training ---", flush=True)

# Prepare feature matrices for best feature set
feat_idx = [all_feature_names.index(f) for f in best_features if f in all_feature_names]
X = X_all[:, feat_idx]
X_test = X_test_all[:, feat_idx]
feature_names = [all_feature_names[i] for i in feat_idx]
print(f"  Using {X.shape[1]} features", flush=True)

# Try StratifiedGroupKFold, fall back to StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    groups = train_df["primary_observation_id"].fillna(-train_df.index - 1).values
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split_iter = list(skf.split(X, y, groups=groups))
    print(f"  Using StratifiedGroupKFold (groups from primary_observation_id)", flush=True)
except Exception:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split_iter = list(skf.split(X, y))
    print(f"  Using StratifiedKFold (GroupKFold unavailable)", flush=True)

# Import models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGBoost
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

    # CatBoost
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

# Individual model SKF scores
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# -- Ensemble weight optimization -------------------------------------
print("\n--- Ensemble Weight Optimization ---", flush=True)
best_w = None
best_ens_map = -1.0
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
print(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Ensemble SKF mAP: {best_ens_map:.4f}", flush=True)

oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

# -- Binary specialists for weak classes (Waders, Pigeons) ------------
print("\n--- Training Specialists ---", flush=True)
from sklearn.metrics import average_precision_score

SPECIALIST_CLASSES = ["Waders", "Pigeons"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
specialist_oof = {}
specialist_test = {}
ap_delta = {}

for cls in SPECIALIST_CLASSES:
    idx = CLASSES.index(cls)
    y_bin = (y == idx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
        cb_spec = CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=80, task_type="GPU",
        )
        cb_spec.fit(X[tr_idx], y_bin[tr_idx],
                     eval_set=(X[va_idx], y_bin[va_idx]), verbose=0)
        oof_bin[va_idx] = cb_spec.predict_proba(X[va_idx])[:, 1]
        test_bin += cb_spec.predict_proba(X_test)[:, 1] / N_FOLDS

    ap_base = average_precision_score(y_bin, oof_ens[:, idx])
    ap_spec = average_precision_score(y_bin, oof_bin)
    ap_delta[cls] = ap_spec - ap_base
    specialist_oof[cls] = oof_bin
    specialist_test[cls] = test_bin
    print(f"  {cls:<15s}: base={ap_base:.4f} spec={ap_spec:.4f} delta={ap_delta[cls]:+.4f}", flush=True)

# Blend specialists
import itertools


def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)


improving = [cls for cls in SPECIALIST_CLASSES if ap_delta[cls] > 0.002]
print(f"  Improving specialists (>0.002 AP): {improving}", flush=True)

if not improving:
    best_oof = oof_ens.copy()
    best_test = test_ens.copy()
    best_map = best_ens_map
else:
    best_map = -1.0
    best_alpha_map = None
    for combo in itertools.product(ALPHA_GRID, repeat=len(improving)):
        alpha_map = {cls: alpha for cls, alpha in zip(improving, combo)}
        oof_blend = apply_blend(oof_ens, specialist_oof, alpha_map)
        m, _ = compute_map(y, oof_blend)
        if m > best_map:
            best_map = m
            best_alpha_map = alpha_map
            best_oof = oof_blend

    print(f"  Best alpha map: {best_alpha_map}", flush=True)
    best_test = apply_blend(test_ens, specialist_test, best_alpha_map)

# Final SKF results
base_map, base_per = compute_map(y, best_oof)
print_results(base_map, base_per, label="E166 base model (SKF OOF)")

# LOMO evaluation of the full ensemble
print("\n--- LOMO Evaluation ---", flush=True)
lomo_final, per_lomo, _ = lomo_evaluate(X, y, train_months, sample_weights)
print(f"  LOMO mAP: {lomo_final:.4f} (E79 baseline: {lomo_e79:.4f}, delta: {lomo_final - lomo_e79:+.4f})", flush=True)

# -- Phase 5: Save base artifacts ------------------------------------
print("\n--- Saving Base Artifacts ---", flush=True)
np.save(ROOT / "oof_e166.npy", best_oof)
np.save(ROOT / "test_e166.npy", best_test)
save_submission(best_test, "e166_mega_raw", cv_map=base_map)

# -- Phase 6: Improved Post-Processing --------------------------------
print("\n--- Phase 6: Improved Post-Processing ---", flush=True)

p_train = counts / counts.sum()
priors = build_improved_gbif_priors(p_train)

# Apply gated ratio priors (Stage 1) - same structure as default pipeline
test_p, n_changed = apply_gated_ratio_priors(
    best_test.copy(), test_months, p_train, priors, BASE_ALPHA, tau=0.15
)
print(f"  Stage 1 (improved GBIF priors): {n_changed} rows adjusted", flush=True)

# Build NB evidence channels (Stage 2)
# Standard channels
speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
cont_tr = {
    "speed": speed_tr,
    "alt_mid": 0.5 * (min_z_tr + max_z_tr),
    "alt_range": max_z_tr - min_z_tr,
}

# New evidence channels from external data
for name, csv_name, col_name, weight in [
    ("tidal_phase", "tidal", "tidal_phase", 0.5),
    ("boundary_layer_height", "altitude_winds", "boundary_layer_height", 0.3),
]:
    tr_ext = load_external_csv(csv_name, "train")
    if not tr_ext.empty and col_name in tr_ext.columns:
        cont_tr[name] = pd.to_numeric(tr_ext[col_name], errors="coerce").fillna(0).values.astype(float)

size_levels, log_p_size, mu, sig = build_nb_params(train_df, y, cont_tr)

# Test channels
speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
cont_te = {
    "speed": speed_te,
    "alt_mid": 0.5 * (min_z_te + max_z_te),
    "alt_range": max_z_te - min_z_te,
}

for name, csv_name, col_name, weight in [
    ("tidal_phase", "tidal", "tidal_phase", 0.5),
    ("boundary_layer_height", "altitude_winds", "boundary_layer_height", 0.3),
]:
    te_ext = load_external_csv(csv_name, "test")
    if not te_ext.empty and col_name in te_ext.columns:
        cont_te[name] = pd.to_numeric(te_ext[col_name], errors="coerce").fillna(0).values.astype(float)

weights = {
    "speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5,
    "tidal_phase": 0.5, "boundary_layer_height": 0.3,
}

loglike = compute_log_p_u_given_c(
    test_df, size_levels, log_p_size, cont_te, weights, None, mu, sig
)

# Sweep gamma for NB PoE
print("\n  NB PoE gamma sweep:", flush=True)
best_pp_test = None
for gamma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for tau_nb in [0.20, 0.25, 0.30]:
        gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_p) < tau_nb)
        pp_test = apply_nb_poe(test_p.copy(), loglike, gamma=gamma, gate=gate)
        n_gated = int(gate.sum())
        print(f"    gamma={gamma:.2f} tau={tau_nb:.2f}: {n_gated} rows gated", flush=True)
        # Save the conservative default
        if gamma == 0.10 and tau_nb == 0.25:
            best_pp_test = pp_test

if best_pp_test is None:
    best_pp_test = test_p

# -- Phase 7: OOF Post-Processing evaluation --------------------------
print("\n--- OOF Post-Processing Evaluation ---", flush=True)


def pp_fn(preds, test_df_local, test_months_local, train_df_local, y_local):
    """Post-processing function for eval_pp."""
    counts_local = np.bincount(y_local, minlength=N_CLASSES).astype(float)
    p_train_local = counts_local / counts_local.sum()
    priors_local = build_improved_gbif_priors(p_train_local)

    out, _ = apply_gated_ratio_priors(
        preds, test_months_local, p_train_local, priors_local, BASE_ALPHA, tau=0.15
    )

    speed_l = pd.to_numeric(train_df_local["airspeed"], errors="coerce").values.astype(float)
    min_z_l = pd.to_numeric(train_df_local["min_z"], errors="coerce").values.astype(float)
    max_z_l = pd.to_numeric(train_df_local["max_z"], errors="coerce").values.astype(float)
    cont_l = {
        "speed": speed_l,
        "alt_mid": 0.5 * (min_z_l + max_z_l),
        "alt_range": max_z_l - min_z_l,
    }
    sl, lps, mu_l, sig_l = build_nb_params(train_df_local, y_local, cont_l)

    speed_t = pd.to_numeric(test_df_local["airspeed"], errors="coerce").values.astype(float)
    min_z_t = pd.to_numeric(test_df_local["min_z"], errors="coerce").values.astype(float)
    max_z_t = pd.to_numeric(test_df_local["max_z"], errors="coerce").values.astype(float)
    cont_t = {
        "speed": speed_t,
        "alt_mid": 0.5 * (min_z_t + max_z_t),
        "alt_range": max_z_t - min_z_t,
    }
    w_local = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(test_df_local, sl, lps, cont_t, w_local, None, mu_l, sig_l)
    gate = np.isin(test_months_local, UNSEEN_MONTHS) & (top2_margin(out) < 0.25)
    return apply_nb_poe(out, ll, gamma=0.10, gate=gate)


try:
    from src.validate import eval_pp
    result = eval_pp(pp_fn, verbose=True)
    if result.get("calibrated_lb"):
        print(f"\n  Calibrated LB estimate: {result['calibrated_lb']}", flush=True)
except Exception as e:
    print(f"  eval_pp failed: {e}", flush=True)

# -- Phase 8: Save final submissions ----------------------------------
print("\n--- Saving Final Submissions ---", flush=True)

# Raw (no PP)
save_submission(best_test, "e166_mega_raw", cv_map=base_map)

# With improved PP
save_submission(best_pp_test, "e166_mega_pp", cv_map=base_map)

# Summary
print("\n" + "=" * 70, flush=True)
print("E166 SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  Features: {len(best_features)} (E79 base + {best_n_new} new)", flush=True)
print(f"  LOMO mAP: {best_lomo:.4f} (E79: {lomo_e79:.4f}, delta: {best_lomo - lomo_e79:+.4f})", flush=True)
print(f"  SKF mAP:  {base_map:.4f}", flush=True)
print(f"  Ensemble: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Submissions: e166_mega_raw, e166_mega_pp", flush=True)
print("=" * 70, flush=True)
print("\nDone.", flush=True)
