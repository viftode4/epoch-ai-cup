"""E60: Optuna Hyperparameter Tuning

Tunes LightGBM hyperparameters on the E38 feature set to maximize macro mAP.
Evaluates using 5-fold StratifiedKFold (temporal features removed).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import optuna
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

print("=" * 60, flush=True)
print("E60 OPTUNA HYPERPARAMETER TUNING (LGBM)", flush=True)
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

# Build base features
print("\nBuilding base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Load weather, solar, GBIF (E38 full stack)
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

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"Total features: {len(fn)}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(skf.split(X, y))

def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
        "device": "cpu"
    }

    oof_preds = np.zeros((len(y), N_CLASSES))

    for tr_idx, va_idx in cv_splits:
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx], feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=fn, reference=dtrain)
        
        mdl = lgb.train(params, dtrain, 1000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[va_idx] = mdl.predict(X[va_idx])

    mAP, _ = compute_map(y, oof_preds)
    return mAP

print("\nStarting Optuna hyperparameter tuning for 30 trials...", flush=True)
study = optuna.create_study(direction="maximize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest trial:")
print(f"  mAP: {study.best_value:.4f}")
print("  Params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

best_params = study.best_params
best_params.update({
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
    "device": "cpu"
})

print("\nRetraining with best parameters on 5 folds to generate submission...", flush=True)
oof_final = np.zeros((len(y), N_CLASSES))
test_final = np.zeros((len(X_test), N_CLASSES))

for tr_idx, va_idx in cv_splits:
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx], feature_name=fn)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=fn, reference=dtrain)
    
    mdl = lgb.train(best_params, dtrain, 1500, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_final[va_idx] = mdl.predict(X[va_idx])
    test_final += mdl.predict(X_test) / 5

final_mAP, _ = compute_map(y, oof_final)
print(f"\nFinal SKF CV mAP with tuned params: {final_mAP:.4f}", flush=True)

save_submission(test_final, "e60_lgbm_tuned", cv_map=final_mAP)
print("Saved e60_lgbm_tuned submission.", flush=True)
