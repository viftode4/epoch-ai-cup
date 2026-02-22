"""E62: Soft Pseudo-labeling

Uses the predictions from the best model (E54 winter tilt, LB 0.56)
as soft targets for the test set. The test set is appended to the train
set by replicating each test sample N_CLASSES times, with the sample_weight
equal to the predicted probability.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
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
print("E62 SOFT PSEUDO-LABELING", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y_train = le.transform(train_df["bird_group"])
counts = np.bincount(y_train, minlength=N_CLASSES)

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights_train = np.array([class_w[yi] for yi in y_train])

# Load features
print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

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

X_train = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)

# Load E54 Winter Tilt Predictions as soft labels
sub_path = ROOT / "submissions" / "e54_e50_winter_tilt_m2_0.22_m5_0.12_m12_0.24_20260218_2229.csv"
sub_df = pd.read_csv(sub_path)
soft_preds = sub_df[CLASSES].values

# Construct augmented training set from test data
# We replicate each test sample for any class probability > 0.05
print("\nGenerating soft pseudo-labels from test data...", flush=True)
pseudo_X = []
pseudo_y = []
pseudo_weights = []

PROB_THRESHOLD = 0.05
PSEUDO_WEIGHT_MULTIPLIER = 0.5  # Downweight pseudo labels relative to real data

for i in range(len(X_test)):
    probs = soft_preds[i]
    for c in range(N_CLASSES):
        if probs[c] > PROB_THRESHOLD:
            pseudo_X.append(X_test[i])
            pseudo_y.append(c)
            # Use class weight * prob * multiplier
            pseudo_weights.append(class_w[c] * probs[c] * PSEUDO_WEIGHT_MULTIPLIER)

pseudo_X = np.array(pseudo_X)
pseudo_y = np.array(pseudo_y)
pseudo_weights = np.array(pseudo_weights)
print(f"Added {len(pseudo_X)} soft pseudo-labeled samples.", flush=True)

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "cpu",
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y_train), N_CLASSES))
test_preds = np.zeros((len(X_test), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nTraining Fold {fold_idx} with soft pseudo-labels...", flush=True)
    
    # Combine real train data with all pseudo data
    X_tr_fold = np.vstack([X_train[tr_idx], pseudo_X])
    y_tr_fold = np.concatenate([y_train[tr_idx], pseudo_y])
    w_tr_fold = np.concatenate([sample_weights_train[tr_idx], pseudo_weights])
    
    dtrain = lgb.Dataset(X_tr_fold, label=y_tr_fold, weight=w_tr_fold, feature_name=fn)
    dval = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], feature_name=fn, reference=dtrain)
    
    mdl = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
    
    oof_preds[va_idx] = mdl.predict(X_train[va_idx])
    test_preds += mdl.predict(X_test) / 5

mAP, per_class = compute_map(y_train, oof_preds)
print(f"\nSoft Pseudo-labeling SKF CV mAP: {mAP:.4f}", flush=True)
for cls in CLASSES:
    print(f"  {cls:<15s}: {per_class[cls]:.4f}")

save_submission(test_preds, "e62_soft_pseudolabels", cv_map=mAP)
print("\nDone!", flush=True)
