"""E37: Species Trait Features from AVONET + Bruderer Wingbeat Data

Add features that encode "how close is this radar track to the known traits of each species?"

Data sources:
- AVONET (Tobias et al. 2022): body mass, wing length, hand-wing index per species
- Bruderer et al. 2010 (Ibis): wingbeat frequency measured by radar, 150+ species
- Alerstam et al. 2007, Pennycuick 2001: typical flight speeds

For each sample, compute per-class distance features:
- |observed_airspeed - typical_speed(class)| / typical_speed(class)
- |observed_rcs_mean - expected_rcs(class)| based on body mass -> RCS relationship
- |rcs_peak_freq - wingbeat_freq(class)| / wingbeat_freq(class)

Also add class-level trait values as features the model can use to associate
radar observations with species characteristics.

Builds on E36-B (114 base + 10 GBIF = 124 features).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

W_LGB = 0.33
W_XGB = 0.33
W_CB = 0.34

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}


def train_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, fold_label):
    """Train LGB+XGB+CB on a single fold. Returns (oof_pred, test_pred)."""
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    mdl_lgb = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb = mdl_lgb.predict(X_va)
    test_lgb = mdl_lgb.predict(X_test) if X_test is not None else None

    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
    mdl_xgb = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=2000,
                         evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb = mdl_xgb.predict(dval_xgb)
    test_xgb = mdl_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    if X_test is not None:
        test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb
    else:
        test_ens = None

    fold_map, _ = compute_map(y_va, oof_ens)
    print(f"  {fold_label}: mAP={fold_map:.4f} (n={len(y_va)})", flush=True)
    return oof_ens, test_ens


# ======================================================================
# Step 1: Load trait data
# ======================================================================
print("=" * 60, flush=True)
print("E37 SPECIES TRAIT FEATURES", flush=True)
print("=" * 60, flush=True)

# Group-level traits from AVONET
traits_df = pd.read_csv(ROOT / "data" / "group_traits_avonet.csv")
wb_df = pd.read_csv(ROOT / "data" / "group_wingbeat.csv")
speed_df = pd.read_csv(ROOT / "data" / "group_flight_speed.csv")

# Build trait lookup: {class_name: {trait: value}}
trait_lookup = {}
for _, row in traits_df.iterrows():
    grp = row["group"]
    trait_lookup[grp] = {
        "mass_g": row["Mass"],
        "wing_mm": row["Wing.Length"],
        "hwi": row["Hand-Wing.Index"],
        "tail_mm": row.get("Tail.Length", 0),
        "tarsus_mm": row.get("Tarsus.Length", 0),
    }

for _, row in wb_df.iterrows():
    grp = row["group"]
    if grp not in trait_lookup:
        trait_lookup[grp] = {}
    trait_lookup[grp]["wingbeat_hz"] = row["wingbeat_hz"]
    trait_lookup[grp]["wb_min"] = row["wb_min"]
    trait_lookup[grp]["wb_max"] = row["wb_max"]

for _, row in speed_df.iterrows():
    grp = row["group"]
    if grp not in trait_lookup:
        trait_lookup[grp] = {}
    trait_lookup[grp]["flight_speed_ms"] = row["flight_speed_ms"]

# Print trait table
print("\n  Species trait lookup:", flush=True)
print(f"  {'Class':<15s} {'Mass_g':>7s} {'Wing':>5s} {'HWI':>5s} {'WB_Hz':>6s} {'Speed':>6s}", flush=True)
for cls in CLASSES:
    t = trait_lookup.get(cls, {})
    print(f"  {cls:<15s} {t.get('mass_g',0):>7.0f} {t.get('wing_mm',0):>5.0f} {t.get('hwi',0):>5.1f} "
          f"{t.get('wingbeat_hz',0):>6.1f} {t.get('flight_speed_ms',0):>6.1f}", flush=True)

# Body mass to expected RCS relationship (empirical from radar ornithology literature)
# RCS_dBsm ~ 10 * log10(mass_kg^0.7) is a rough approximation
# But our RCS is in dBm2 and varies with aspect angle, so we use relative scaling
# Normalize: log10(mass) centered and scaled
mass_arr = np.array([trait_lookup.get(cls, {}).get("mass_g", 100) for cls in CLASSES])
log_mass = np.log10(mass_arr + 1)
log_mass_mean = log_mass.mean()
log_mass_std = log_mass.std()

print(f"\n  Body mass -> log scale: mean={log_mass_mean:.3f}, std={log_mass_std:.3f}", flush=True)

# ======================================================================
# Step 2: Load data and build features
# ======================================================================
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)
p_train = counts.astype(float) / counts.sum()

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# Months for GBIF features
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

# Build base features (same as E36-B: all feature sets, remove temporal, add GBIF)
print("Building base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add GBIF seasonal features (same as E36-B)
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_counts = gbif[cls].values
            class_mean = class_counts.mean()
            if class_mean > 0:
                si[i] = row[cls] / class_mean
            else:
                si[i] = 1.0
    gbif_si[m] = si

for i, cls in enumerate(CLASSES):
    col_name = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col_name] = [gbif_si[m][i] for m in train_months]
    test_feats[col_name] = [gbif_si[m][i] for m in test_months]

gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.array([row[cls] for cls in CLASSES])
    probs = np.maximum(probs, 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))

train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

print(f"  E36-B features: {len(train_feats.columns)}", flush=True)

# ======================================================================
# Step 3: Add species trait features
# ======================================================================
print("\nAdding species trait features...", flush=True)

# --- Feature Set 1: Per-class trait values (9 classes x 3 key traits = 27 features) ---
# These let the model associate radar observations with species morphology
for i, cls in enumerate(CLASSES):
    t = trait_lookup.get(cls, {})
    # Body mass (log-scaled)
    train_feats[f"trait_logmass_{cls.lower().replace(' ', '_')}"] = np.log10(t.get("mass_g", 100) + 1)
    test_feats[f"trait_logmass_{cls.lower().replace(' ', '_')}"] = np.log10(t.get("mass_g", 100) + 1)
    # Wingbeat frequency
    train_feats[f"trait_wb_{cls.lower().replace(' ', '_')}"] = t.get("wingbeat_hz", 0)
    test_feats[f"trait_wb_{cls.lower().replace(' ', '_')}"] = t.get("wingbeat_hz", 0)
    # Typical flight speed
    train_feats[f"trait_speed_{cls.lower().replace(' ', '_')}"] = t.get("flight_speed_ms", 0)
    test_feats[f"trait_speed_{cls.lower().replace(' ', '_')}"] = t.get("flight_speed_ms", 0)

# --- Feature Set 2: Distance features (per-class "match scores") ---
# How well does this track match each class's expected traits?

# Speed match: |observed - expected| / expected
for i, cls in enumerate(CLASSES):
    expected_speed = trait_lookup.get(cls, {}).get("flight_speed_ms", 11)
    if expected_speed > 0:
        train_feats[f"speed_match_{cls.lower().replace(' ', '_')}"] = (
            np.abs(train_feats["airspeed"].values - expected_speed) / expected_speed
        )
        test_feats[f"speed_match_{cls.lower().replace(' ', '_')}"] = (
            np.abs(test_feats["airspeed"].values - expected_speed) / expected_speed
        )
    else:
        # Clutter: match score = airspeed itself (low speed = more likely clutter)
        train_feats[f"speed_match_{cls.lower().replace(' ', '_')}"] = train_feats["airspeed"].values
        test_feats[f"speed_match_{cls.lower().replace(' ', '_')}"] = test_feats["airspeed"].values

# RCS match: compare observed RCS to expected from body mass
# Rough relationship: larger birds have higher (less negative) RCS
# RCS_expected ~ -30 + 10 * log10(mass_kg / 0.1)  (very rough)
for i, cls in enumerate(CLASSES):
    mass_g = trait_lookup.get(cls, {}).get("mass_g", 100)
    if cls == "Clutter":
        expected_rcs = -13.8  # Clutter has distinctive high RCS
    else:
        expected_rcs = -30 + 10 * np.log10(mass_g / 100 + 0.01)
    train_feats[f"rcs_match_{cls.lower().replace(' ', '_')}"] = (
        np.abs(train_feats["rcs_mean"].values - expected_rcs)
    )
    test_feats[f"rcs_match_{cls.lower().replace(' ', '_')}"] = (
        np.abs(test_feats["rcs_mean"].values - expected_rcs)
    )

# Wingbeat match: compare RCS oscillation frequency to known wingbeat
# We have rcs_peak_freq from FFT features -- compare to expected wingbeat_hz
for i, cls in enumerate(CLASSES):
    wb_hz = trait_lookup.get(cls, {}).get("wingbeat_hz", 0)
    if wb_hz > 0 and "rcs_peak_freq" in train_feats.columns:
        train_feats[f"wb_match_{cls.lower().replace(' ', '_')}"] = (
            np.abs(train_feats["rcs_peak_freq"].values - wb_hz) / wb_hz
        )
        test_feats[f"wb_match_{cls.lower().replace(' ', '_')}"] = (
            np.abs(test_feats["rcs_peak_freq"].values - wb_hz) / wb_hz
        )
    else:
        train_feats[f"wb_match_{cls.lower().replace(' ', '_')}"] = 0.0
        test_feats[f"wb_match_{cls.lower().replace(' ', '_')}"] = 0.0

# --- Feature Set 3: Best-match features (which class does this track match best?) ---
# Instead of 9 per-class features, compute summary stats

# Speed: closest-class index + distance to closest
speed_expected = np.array([trait_lookup.get(cls, {}).get("flight_speed_ms", 11) for cls in CLASSES])
speed_expected[CLASSES.index("Clutter")] = 0.0  # Clutter has no real speed

for df_feats, label in [(train_feats, "train"), (test_feats, "test")]:
    airspeed = df_feats["airspeed"].values
    dists = np.abs(airspeed[:, None] - speed_expected[None, :])
    df_feats["speed_best_class"] = np.argmin(dists, axis=1)
    df_feats["speed_best_dist"] = np.min(dists, axis=1)
    df_feats["speed_2nd_dist"] = np.partition(dists, 1, axis=1)[:, 1]
    df_feats["speed_ambiguity"] = df_feats["speed_best_dist"] / (df_feats["speed_2nd_dist"] + 0.01)

# Mass-RCS: best match
rcs_expected = np.array([
    -30 + 10 * np.log10(trait_lookup.get(cls, {}).get("mass_g", 100) / 100 + 0.01)
    if cls != "Clutter" else -13.8
    for cls in CLASSES
])

for df_feats, label in [(train_feats, "train"), (test_feats, "test")]:
    rcs_mean = df_feats["rcs_mean"].values
    dists = np.abs(rcs_mean[:, None] - rcs_expected[None, :])
    df_feats["rcs_best_class"] = np.argmin(dists, axis=1)
    df_feats["rcs_best_dist"] = np.min(dists, axis=1)

# Hand-wing index as flight style indicator
# High HWI = long-distance migrant, fast pointed wings (ducks, waders, gulls)
# Low HWI = resident, broad wings (BoP, songbirds)
hwi_arr = np.array([trait_lookup.get(cls, {}).get("hwi", 40) for cls in CLASSES])
print(f"  HWI range: {hwi_arr.min():.1f} (Songbirds) - {hwi_arr.max():.1f} (Gulls)", flush=True)

# Clean up inf/nan
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

trait_feat_names = [c for c in train_feats.columns if c.startswith(("trait_", "speed_match_", "rcs_match_", "wb_match_", "speed_best", "speed_2nd", "speed_ambi", "rcs_best"))]
print(f"  Added {len(trait_feat_names)} species trait features", flush=True)
print(f"  Total features: {len(train_feats.columns)}", flush=True)

# ======================================================================
# Step 4: Train -- Run 3 configs to isolate effects
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("TRAINING", flush=True)
print("=" * 60, flush=True)

# Config A: E36-B baseline (GBIF features only, no traits) -- for comparison
e36b_cols = [c for c in train_feats.columns if not c.startswith(("trait_", "speed_match_", "rcs_match_", "wb_match_", "speed_best", "speed_2nd", "speed_ambi", "rcs_best"))]

# Config B: E36-B + trait distance features only (speed_match, rcs_match, wb_match, best features)
trait_dist_cols = [c for c in train_feats.columns if c.startswith(("speed_match_", "rcs_match_", "wb_match_", "speed_best", "speed_2nd", "speed_ambi", "rcs_best"))]
e37_dist_cols = e36b_cols + trait_dist_cols

# Config C: E36-B + ALL trait features (values + distances)
e37_all_cols = list(train_feats.columns)

configs = {
    "E36-B (baseline)": e36b_cols,
    "E37-dist (distances only)": e37_dist_cols,
    "E37-all (values+distances)": e37_all_cols,
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for config_name, cols in configs.items():
    print(f"\n--- {config_name}: {len(cols)} features ---", flush=True)
    X = train_feats[cols].values.astype(np.float32)
    X_test_cfg = test_feats[cols].values.astype(np.float32)
    fn = list(cols)

    oof = np.zeros((len(y), N_CLASSES))
    test_pred = np.zeros((len(X_test_cfg), N_CLASSES))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        oof_fold, test_fold = train_fold(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            sample_weights[tr_idx], X_test_cfg, fn,
            f"Fold {fold_idx}",
        )
        oof[va_idx] = oof_fold
        test_pred += test_fold / 5

    map_val, per_val = compute_map(y, oof)
    results[config_name] = {"map": map_val, "per": per_val, "oof": oof, "test": test_pred}
    print(f"\n  {config_name}: mAP = {map_val:.4f}", flush=True)

# ======================================================================
# Step 5: Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E37 SUMMARY", flush=True)
print("=" * 60, flush=True)

baseline_map = results["E36-B (baseline)"]["map"]
print(f"\n  {'Config':<30s} {'Features':>8s} {'CV mAP':>8s} {'Delta':>8s}", flush=True)
print(f"  {'-'*54}", flush=True)
for name, res in results.items():
    n_feats = len(configs[name])
    delta = res["map"] - baseline_map
    delta_str = f"{delta:+.4f}" if name != "E36-B (baseline)" else "---"
    print(f"  {name:<30s} {n_feats:>8d} {res['map']:>8.4f} {delta_str:>8s}", flush=True)

# Per-class comparison
print(f"\n  Per-class breakdown:", flush=True)
header = f"  {'Class':<15s}"
for name in results:
    short = name.split("(")[0].strip()
    header += f" {short:>8s}"
print(header, flush=True)

for cls in CLASSES:
    line = f"  {cls:<15s}"
    for name, res in results.items():
        ap = res["per"].get(cls, 0)
        line += f" {ap:>8.4f}"
    print(line, flush=True)

# Pick best config
best_name = max(results, key=lambda k: results[k]["map"])
best = results[best_name]
print(f"\n  Best config: {best_name} (mAP={best['map']:.4f})", flush=True)

# Save predictions for best config
np.save(ROOT / "oof_e37.npy", best["oof"])
np.save(ROOT / "test_e37.npy", best["test"])
print(f"  Saved: oof_e37.npy, test_e37.npy", flush=True)

save_submission(best["test"], f"e37_traits", cv_map=best["map"])

# Test class distributions
print(f"\n  Test class distributions (argmax):", flush=True)
print(f"  {'Class':<15s}", end="", flush=True)
for name in results:
    short = name.split("(")[0].strip()
    print(f" {short:>8s}", end="")
print(flush=True)
for i, cls in enumerate(CLASSES):
    line = f"  {cls:<15s}"
    for name, res in results.items():
        dist = np.bincount(res["test"].argmax(axis=1), minlength=N_CLASSES)
        line += f" {dist[i]:>8d}"
    print(line, flush=True)

print("\nDone!", flush=True)
