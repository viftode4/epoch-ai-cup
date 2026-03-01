"""E38: Weather + Solar Features (LOMO-focused evaluation)

Adds per-sample weather features (KNMI station 277) and solar position
features to the E32 baseline. These are physically causal features that
vary per-sample and should help generalization to unseen months.

Weather features (11): wind_speed, wind_gust, wind_u, wind_v, temp_c,
  dewpoint_c, sunshine_hrs, radiation, precip_dur, precip_mm, humidity
Solar features (4): solar_elevation, daylight_hours, hours_since_sunrise,
  daylight_fraction

Total new: 15 features. Base: 114. Grand total: 129.

PRIMARY EVALUATION: LOMO (E32 LOMO = 0.3317)
Secondary: SKF 5-fold (E32 SKF = 0.6808)

Configs tested:
  A: E32 base only (114 feats) -- reference
  B: E32 + weather (125 feats)
  C: E32 + solar (118 feats)
  D: E32 + weather + solar (129 feats)
  E: E32 + weather + solar + GBIF (139 feats) -- full stack
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
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999
W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34

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


def train_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label):
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb = m_lgb.predict(X_va)
    test_lgb = m_lgb.predict(X_test) if X_test is not None else None

    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn),
                       2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                       early_stopping_rounds=80, verbose_eval=0)
    oof_xgb = m_xgb.predict(xgb.DMatrix(X_va, feature_names=fn))
    test_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    cb = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    test_ens = (W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb) if X_test is not None else None

    m, _ = compute_map(y_va, oof)
    print(f"  {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_ens


# ======================================================================
# Load data
# ======================================================================
print("=" * 60, flush=True)
print("E38 WEATHER + SOLAR FEATURES", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# Build base features
print("\nBuilding base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
base_cols = list(train_feats.columns)
print(f"  Base features: {len(base_cols)}", flush=True)

# Load weather features
print("Loading weather features...", flush=True)
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
weather_cols = list(train_weather.columns)

for col in weather_cols:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

# Load solar features
print("Loading solar features...", flush=True)
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
solar_cols = list(train_solar.columns)

for col in solar_cols:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

# Load GBIF features
print("Loading GBIF features...", flush=True)
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
            si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    gbif_si[m] = si

for i, cls in enumerate(CLASSES):
    col = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col] = [gbif_si[m][i] for m in train_months]
    test_feats[col] = [gbif_si[m][i] for m in test_months]

gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

# Clean
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

# Define feature sets
wx_feat_cols = [c for c in train_feats.columns if c.startswith("wx_")]
sol_feat_cols = [c for c in train_feats.columns if c.startswith("sol_")]
gbif_feat_cols = [c for c in train_feats.columns if c.startswith("gbif_si_") or c == "month_gbif_diversity"]

print(f"\n  Weather features ({len(wx_feat_cols)}): {wx_feat_cols}", flush=True)
print(f"  Solar features ({len(sol_feat_cols)}): {sol_feat_cols}", flush=True)
print(f"  GBIF features ({len(gbif_feat_cols)}): {len(gbif_feat_cols)}", flush=True)

# Configs
configs = {
    "A: E32 base": base_cols,
    "B: +weather": base_cols + wx_feat_cols,
    "C: +solar": base_cols + sol_feat_cols,
    "D: +wx+solar": base_cols + wx_feat_cols + sol_feat_cols,
    "E: +wx+sol+gbif": base_cols + wx_feat_cols + sol_feat_cols + gbif_feat_cols,
}

# ======================================================================
# LOMO evaluation (primary)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("LOMO EVALUATION (Leave-One-Month-Out) -- PRIMARY", flush=True)
print("=" * 60, flush=True)

lomo_results = {}
for cfg_name, cols in configs.items():
    print(f"\n--- {cfg_name}: {len(cols)} features ---", flush=True)
    X = train_feats[cols].values.astype(np.float32)
    fn = list(cols)

    oof_lomo = np.zeros((len(y), N_CLASSES))
    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]
        oof_fold, _ = train_fold(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            sample_weights[tr_idx], None, fn, f"LOMO Month {m}",
        )
        oof_lomo[va_idx] = oof_fold

    lomo_map, lomo_per = compute_map(y, oof_lomo)
    lomo_results[cfg_name] = {"map": lomo_map, "per": lomo_per}
    print(f"\n  {cfg_name} LOMO: {lomo_map:.4f}", flush=True)

# LOMO summary
print("\n" + "=" * 60, flush=True)
print("LOMO SUMMARY", flush=True)
print("=" * 60, flush=True)

base_lomo = lomo_results["A: E32 base"]["map"]
print(f"\n  {'Config':<25s} {'Feats':>5s} {'LOMO':>7s} {'Delta':>7s}", flush=True)
print(f"  {'-'*44}", flush=True)
for name, res in lomo_results.items():
    delta = res["map"] - base_lomo
    d_str = f"{delta:+.4f}" if name != "A: E32 base" else "---"
    n_feats = len(configs[name])
    print(f"  {name:<25s} {n_feats:>5d} {res['map']:>7.4f} {d_str:>7s}", flush=True)

# Per-class LOMO comparison
print(f"\n  Per-class LOMO:", flush=True)
header = f"  {'Class':<15s}"
for name in lomo_results:
    short = name.split(":")[0]
    header += f" {short:>7s}"
print(header, flush=True)
for cls in CLASSES:
    line = f"  {cls:<15s}"
    for name, res in lomo_results.items():
        ap = res["per"].get(cls, 0)
        line += f" {ap:>7.4f}"
    print(line, flush=True)

# ======================================================================
# SKF evaluation + test predictions for best LOMO config
# ======================================================================
# Find best LOMO config
best_lomo_name = max(lomo_results, key=lambda k: lomo_results[k]["map"])
best_lomo_cols = configs[best_lomo_name]
print(f"\n\nBest LOMO config: {best_lomo_name} ({lomo_results[best_lomo_name]['map']:.4f})", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"SKF EVALUATION + TEST PREDICTIONS for {best_lomo_name}", flush=True)
print("=" * 60, flush=True)

X = train_feats[best_lomo_cols].values.astype(np.float32)
X_test = test_feats[best_lomo_cols].values.astype(np.float32)
fn = list(best_lomo_cols)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.zeros((len(y), N_CLASSES))
test_pred = np.zeros((len(X_test), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    oof_fold, test_fold = train_fold(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test, fn, f"SKF Fold {fold_idx}",
    )
    oof_skf[va_idx] = oof_fold
    test_pred += test_fold / 5

skf_map, skf_per = compute_map(y, oof_skf)
best_lomo_map = lomo_results[best_lomo_name]["map"]

print(f"\n  SKF CV mAP: {skf_map:.4f}", flush=True)
print(f"  LOMO mAP:   {best_lomo_map:.4f}", flush=True)
print(f"  Gap:        {skf_map - best_lomo_map:.4f}", flush=True)

print(f"\n  Per-class SKF:", flush=True)
print(f"  {'Class':<15s} {'SKF':>7s} {'LOMO':>7s} {'Gap':>7s}", flush=True)
for cls in CLASSES:
    s = skf_per.get(cls, 0)
    l = lomo_results[best_lomo_name]["per"].get(cls, 0)
    print(f"  {cls:<15s} {s:>7.4f} {l:>7.4f} {s-l:>+7.4f}", flush=True)

# Test distribution
print(f"\n  Test class distribution (argmax):", flush=True)
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {dist[i]}", flush=True)

# Save
np.save(ROOT / "oof_e38.npy", oof_skf)
np.save(ROOT / "test_e38.npy", test_pred)
save_submission(test_pred, "e38_weather_solar", cv_map=skf_map)

print("\nDone!", flush=True)
