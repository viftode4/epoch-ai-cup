"""E59: Col de la Croix 1988 Transfer Learning

Train an auxiliary model on the Col de la Croix 1988 dataset (8 coarse bird classes)
using basic kinematic features (altitude, speed, airspeed, vertical speed).
Then use this model to generate 8 probability features for the competition dataset.
Evaluate the impact on LOMO using the E38 (weather+solar+gbif) baseline.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "cpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cpu", "tree_method": "hist",
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
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="CPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    test_ens = (W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb) if X_test is not None else None

    m, _ = compute_map(y_va, oof)
    print(f"  {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_ens

print("=" * 60, flush=True)
print("E59 COL DE LA CROIX TRANSFER LEARNING", flush=True)
print("=" * 60, flush=True)

# 1. Load and train auxiliary model on Col de la Croix 1988
print("\nLoading Col de la Croix 1988 dataset...", flush=True)
c88 = pd.read_csv(ROOT / "data" / "other_datasets" / "Col de la Croix 1988.csv", sep=";", encoding="latin1", skiprows=24)

# FieldClass mapping (1-8 are birds, 9 is unknown)
c88["FieldClass"] = pd.to_numeric(c88["FieldClass"], errors="coerce")
c88 = c88[c88["FieldClass"].notna()]
c88 = c88[c88["FieldClass"] < 9]

c88["Z"] = pd.to_numeric(c88["Z"], errors="coerce")
c88["Vg"] = pd.to_numeric(c88["Vg"], errors="coerce")
c88["Va"] = pd.to_numeric(c88["Va"], errors="coerce")
c88["Vz"] = pd.to_numeric(c88["Vz"], errors="coerce")

c88 = c88.dropna(subset=["Z", "Vg", "Va", "Vz", "FieldClass"]).reset_index(drop=True)

X_c88 = pd.DataFrame()
X_c88["Z"] = c88["Z"].values
X_c88["Vg_m"] = c88["Vg"].values / 100.0
X_c88["Va_m"] = c88["Va"].values / 100.0
X_c88["Vz_m"] = c88["Vz"].values / 100.0

y_c88 = c88["FieldClass"].astype(int).values - 1

print(f"Training auxiliary XGBoost model on {len(X_c88)} samples from Col de la Croix...", flush=True)
X_c88_tr, X_c88_va, y_c88_tr, y_c88_va = train_test_split(X_c88.values, y_c88, test_size=0.1, random_state=42, stratify=y_c88)

dtrain_c88 = xgb.DMatrix(X_c88_tr, label=y_c88_tr)
dval_c88 = xgb.DMatrix(X_c88_va, label=y_c88_va)
c88_params = {
    "objective": "multi:softprob", "num_class": 8,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 5, "seed": 42, "nthread": -1, "verbosity": 0,
}
model_c88 = xgb.train(c88_params, dtrain_c88, 1000, evals=[(dval_c88, "val")], early_stopping_rounds=50, verbose_eval=100)


# 2. Build competition features
print("\nLoading competition data...", flush=True)
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

print("Building base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Extract C88 equivalent features and predict probabilities
print("Extracting C88 probabilities for competition data...", flush=True)
comp_c88_train = pd.DataFrame()
comp_c88_train["Z"] = train_feats["alt_mean"].values
comp_c88_train["Vg_m"] = train_feats["avg_ground_speed"].values
comp_c88_train["Va_m"] = train_feats["airspeed"].values
comp_c88_train["Vz_m"] = train_feats["alt_rate_mean"].values

comp_c88_test = pd.DataFrame()
comp_c88_test["Z"] = test_feats["alt_mean"].values
comp_c88_test["Vg_m"] = test_feats["avg_ground_speed"].values
comp_c88_test["Va_m"] = test_feats["airspeed"].values
comp_c88_test["Vz_m"] = test_feats["alt_rate_mean"].values

train_c88_probs = model_c88.predict(xgb.DMatrix(comp_c88_train.values))
test_c88_probs = model_c88.predict(xgb.DMatrix(comp_c88_test.values))

c88_cols = []
for i in range(8):
    col_name = f"c88_prob_class{i+1}"
    train_feats[col_name] = train_c88_probs[:, i]
    test_feats[col_name] = test_c88_probs[:, i]
    c88_cols.append(col_name)

# 3. Add Weather, Solar, GBIF (E38 full config)
print("Loading weather, solar, GBIF features...", flush=True)
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

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_mean = gbif[cls].values.mean()
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

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

wx_feat_cols = [c for c in train_feats.columns if c.startswith("wx_")]
sol_feat_cols = [c for c in train_feats.columns if c.startswith("sol_")]
gbif_feat_cols = [c for c in train_feats.columns if c.startswith("gbif_si_") or c == "month_gbif_diversity"]
base_cols = [c for c in train_feats.columns if c not in wx_feat_cols + sol_feat_cols + gbif_feat_cols + c88_cols]

e38_cols = base_cols + wx_feat_cols + sol_feat_cols + gbif_feat_cols
e59_cols = e38_cols + c88_cols

configs = {
    "A: E38 config E (139)": e38_cols,
    "B: +C88 probs (147)": e59_cols,
}

print("\n" + "=" * 60, flush=True)
print("LOMO EVALUATION", flush=True)
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

print("\n" + "=" * 60, flush=True)
print("LOMO SUMMARY", flush=True)
print("=" * 60, flush=True)

base_lomo = lomo_results["A: E38 config E (139)"]["map"]
print(f"\n  {'Config':<25s} {'Feats':>5s} {'LOMO':>7s} {'Delta':>7s}", flush=True)
print(f"  {'-'*44}", flush=True)
for name, res in lomo_results.items():
    delta = res["map"] - base_lomo
    d_str = f"{delta:+.4f}" if name != "A: E38 config E (139)" else "---"
    n_feats = len(configs[name])
    print(f"  {name:<25s} {n_feats:>5d} {res['map']:>7.4f} {d_str:>7s}", flush=True)

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

print("\nDone!", flush=True)
