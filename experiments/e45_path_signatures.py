"""E45: Path Signature Features

Path signatures are mathematically invariant to time reparameterization --
a bird's signature is the same whether sampled fast or slow, in January or
October. This directly addresses our temporal distribution shift problem.

Channels: altitude (normalized), RCS (standardized), speed (normalized),
          cumulative bearing change (normalized).

Configs:
  A: E38 base (139 feats) -- reference
  B: E38 + signatures depth 2 lead-lag (139 + 73 = 212 feats)
  C: E38 + signatures depth 3 no lead-lag (139 + 85 = 224 feats)
  D: E38 + physics(E44) + signatures depth 2 LL (139 + 24 + 73 = 236 feats)

PRIMARY EVALUATION: LOMO
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
print("E45 PATH SIGNATURE FEATURES", flush=True)
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

# ======================================================================
# Build features -- multiple configs
# ======================================================================

# Config B: base + sig depth 2 lead-lag
print("\nBuilding features (base + sig d2 LL)...", flush=True)
feat_sets_b = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
               "weakclass", "path_signature"]
train_feats_b = build_features(train_df, feature_sets=feat_sets_b, sig_depth=2, sig_lead_lag=True)
test_feats_b = build_features(test_df, feature_sets=feat_sets_b, sig_depth=2, sig_lead_lag=True)

# Config C: base + sig depth 3 no LL
print("\nBuilding features (base + sig d3 no LL)...", flush=True)
feat_sets_c = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
               "weakclass", "path_signature"]
train_feats_c = build_features(train_df, feature_sets=feat_sets_c, sig_depth=3, sig_lead_lag=False)
test_feats_c = build_features(test_df, feature_sets=feat_sets_c, sig_depth=3, sig_lead_lag=False)

# Config D: base + physics + sig d2 LL
print("\nBuilding features (base + physics + sig d2 LL)...", flush=True)
feat_sets_d = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
               "weakclass", "flight_physics", "path_signature"]
train_feats_d = build_features(train_df, feature_sets=feat_sets_d, sig_depth=2, sig_lead_lag=True)
test_feats_d = build_features(test_df, feature_sets=feat_sets_d, sig_depth=2, sig_lead_lag=True)

# Remove temporal leaks from all
for tf, testf in [(train_feats_b, test_feats_b), (train_feats_c, test_feats_c),
                  (train_feats_d, test_feats_d)]:
    keep = [c for c in tf.columns if c not in ALL_TEMPORAL]
    for col in list(tf.columns):
        if col not in keep:
            tf.drop(columns=[col], inplace=True)
            testf.drop(columns=[col], inplace=True)

# Config A: base only (extract from B by removing sig cols)
sig_cols_b = [c for c in train_feats_b.columns if c.startswith("sig_")]
base_cols = [c for c in train_feats_b.columns if not c.startswith("sig_")]
sig_cols_c = [c for c in train_feats_c.columns if c.startswith("sig_")]
phys_cols = [c for c in train_feats_d.columns if c.startswith("phys_")]
sig_cols_d = [c for c in train_feats_d.columns if c.startswith("sig_")]

print(f"\n  Base features: {len(base_cols)}", flush=True)
print(f"  Sig d2 LL features: {len(sig_cols_b)}", flush=True)
print(f"  Sig d3 noLL features: {len(sig_cols_c)}", flush=True)
print(f"  Physics features: {len(phys_cols)}", flush=True)

# Add weather + solar + GBIF (same as E38)
print("Loading weather + solar + GBIF features...", flush=True)
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")

for tf, testf in [(train_feats_b, test_feats_b), (train_feats_c, test_feats_c),
                  (train_feats_d, test_feats_d)]:
    for col in train_weather.columns:
        tf[f"wx_{col}"] = train_weather[col].values
        testf[f"wx_{col}"] = test_weather[col].values
    for col in train_solar.columns:
        tf[f"sol_{col}"] = train_solar[col].values
        testf[f"sol_{col}"] = test_solar[col].values

    # GBIF
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
        tf[col] = [gbif_si[m][i] for m in train_months]
        testf[col] = [gbif_si[m][i] for m in test_months]

    month_entropy = {}
    for _, row in gbif_priors_df.iterrows():
        m = int(row["month"])
        probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
        month_entropy[m] = -np.sum(probs * np.log(probs))
    tf["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
    testf["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

    # Clean
    for c in tf.columns:
        tf[c] = pd.to_numeric(tf[c], errors='coerce').fillna(0)
        testf[c] = pd.to_numeric(testf[c], errors='coerce').fillna(0)

# Recompute column lists after adding external
wx_cols = [c for c in train_feats_b.columns if c.startswith("wx_")]
sol_cols = [c for c in train_feats_b.columns if c.startswith("sol_")]
gbif_cols = [c for c in train_feats_b.columns if c.startswith("gbif_si_") or c == "month_gbif_diversity"]
external_cols = wx_cols + sol_cols + gbif_cols

e38_base = base_cols + external_cols

configs = {
    "A: E38 base": (train_feats_b, test_feats_b, e38_base),
    "B: +sig d2 LL": (train_feats_b, test_feats_b, e38_base + sig_cols_b),
    "C: +sig d3 noLL": (train_feats_c, test_feats_c, e38_base + sig_cols_c),
    "D: +phys+sig d2": (train_feats_d, test_feats_d,
                        [c for c in train_feats_d.columns if not c.startswith("sig_")] +
                        external_cols + sig_cols_d),
}

for name, (tf, _, cols) in configs.items():
    # Validate all cols exist
    missing = [c for c in cols if c not in tf.columns]
    if missing:
        print(f"  WARNING {name}: {len(missing)} missing cols: {missing[:5]}", flush=True)
    print(f"  {name}: {len(cols)} features", flush=True)

# ======================================================================
# LOMO evaluation (primary)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("LOMO EVALUATION (Leave-One-Month-Out) -- PRIMARY", flush=True)
print("=" * 60, flush=True)

lomo_results = {}
for cfg_name, (tf, testf, cols) in configs.items():
    # Filter to valid cols only
    valid_cols = [c for c in cols if c in tf.columns]
    print(f"\n--- {cfg_name}: {len(valid_cols)} features ---", flush=True)
    X = tf[valid_cols].values.astype(np.float32)
    fn = list(valid_cols)

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

base_lomo = lomo_results["A: E38 base"]["map"]
print(f"\n  {'Config':<25s} {'Feats':>5s} {'LOMO':>7s} {'Delta':>7s}", flush=True)
print(f"  {'-'*44}", flush=True)
for name, res in lomo_results.items():
    delta = res["map"] - base_lomo
    d_str = f"{delta:+.4f}" if name != "A: E38 base" else "---"
    _, _, cols = configs[name]
    valid = [c for c in cols if c in configs[name][0].columns]
    print(f"  {name:<25s} {len(valid):>5d} {res['map']:>7.4f} {d_str:>7s}", flush=True)

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
# SKF + test preds for best config
# ======================================================================
best_name = max(lomo_results, key=lambda k: lomo_results[k]["map"])
best_tf, best_testf, best_cols_raw = configs[best_name]
best_cols = [c for c in best_cols_raw if c in best_tf.columns]
print(f"\n\nBest LOMO config: {best_name} ({lomo_results[best_name]['map']:.4f})", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"SKF EVALUATION + TEST PREDICTIONS for {best_name}", flush=True)
print("=" * 60, flush=True)

X = best_tf[best_cols].values.astype(np.float32)
X_test = best_testf[best_cols].values.astype(np.float32)
fn = list(best_cols)

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
best_lomo_map = lomo_results[best_name]["map"]

print(f"\n  SKF CV mAP: {skf_map:.4f}", flush=True)
print(f"  LOMO mAP:   {best_lomo_map:.4f}", flush=True)
print(f"  Gap:        {skf_map - best_lomo_map:.4f}", flush=True)

print(f"\n  Per-class SKF:", flush=True)
print(f"  {'Class':<15s} {'SKF':>7s} {'LOMO':>7s} {'Gap':>7s}", flush=True)
for cls in CLASSES:
    s = skf_per.get(cls, 0)
    l = lomo_results[best_name]["per"].get(cls, 0)
    print(f"  {cls:<15s} {s:>7.4f} {l:>7.4f} {s-l:>+7.4f}", flush=True)

# Test distribution
print(f"\n  Test class distribution (argmax):", flush=True)
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {dist[i]}", flush=True)

# Save
np.save(ROOT / "oof_e45.npy", oof_skf)
np.save(ROOT / "test_e45.npy", test_pred)
save_submission(test_pred, "e45_path_signatures", cv_map=skf_map)

print("\nDone!", flush=True)
