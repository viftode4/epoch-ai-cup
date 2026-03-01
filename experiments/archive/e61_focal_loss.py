"""E61: Custom Focal Loss for LightGBM

Implements a custom multi-class Focal Loss function for LightGBM to
better handle the extreme class imbalance (e.g., Cormorants vs Gulls)
without manually bleeding predictions via post-processing.
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

# Custom Focal Loss for LightGBM
def focal_loss_lgb_eval(preds, train_data):
    y_true = train_data.get_label().astype(int)
    preds = preds.reshape(N_CLASSES, -1).T
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    p = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    p_true = p[np.arange(len(y_true)), y_true]
    loss = -np.mean((1 - p_true)**2 * np.log(np.maximum(p_true, 1e-15)))
    return 'focal_loss', loss, False

def focal_loss_lgb(preds, train_data):
    gamma = 2.0
    y_true = train_data.get_label().astype(int)
    preds = preds.reshape(N_CLASSES, -1).T
    
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    p = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    
    p_true = p[np.arange(len(y_true)), y_true]
    mod = (1 - p_true) ** gamma
    
    grad = (p - y_onehot) * mod[:, None]
    hess = p * (1 - p) * mod[:, None] # Approximation
    
    return grad.T.flatten(), hess.T.flatten()

print("=" * 60, flush=True)
print("E61 FOCAL LOSS LIGHTGBM", flush=True)
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

print("Building base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), N_CLASSES))
test_preds = np.zeros((len(X_test), N_CLASSES))

LGB_PARAMS = {
    # In this LightGBM version, custom objective is passed via params["objective"] callable.
    "objective": focal_loss_lgb,
    "learning_rate": 0.05,
    "num_leaves": 47,
    "max_depth": 7,
    "min_child_samples": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
    "num_class": N_CLASSES,
}

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining Fold {fold_idx} with Focal Loss...", flush=True)
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx], feature_name=fn)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=fn, reference=dtrain)
    
    mdl = lgb.train(LGB_PARAMS, dtrain, num_boost_round=1500, valid_sets=[dval],
                    feval=focal_loss_lgb_eval,
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # For custom objective, predictions are raw logits, need softmax
    raw_oof = mdl.predict(X[va_idx], raw_score=True)
    exp_oof = np.exp(raw_oof - np.max(raw_oof, axis=1, keepdims=True))
    oof_preds[va_idx] = exp_oof / np.sum(exp_oof, axis=1, keepdims=True)
    
    raw_test = mdl.predict(X_test, raw_score=True)
    exp_test = np.exp(raw_test - np.max(raw_test, axis=1, keepdims=True))
    test_preds += (exp_test / np.sum(exp_test, axis=1, keepdims=True)) / 5

mAP, per_class = compute_map(y, oof_preds)
print(f"\nFocal Loss SKF CV mAP: {mAP:.4f}", flush=True)
for cls in CLASSES:
    print(f"  {cls:<15s}: {per_class[cls]:.4f}")

save_submission(test_preds, "e61_lgbm_focal_loss", cv_map=mAP)
print("\nDone!", flush=True)
