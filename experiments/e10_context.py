"""
E10: Recovery & Texture.

1. REVERT: Remove 'Context' features (e09) which caused leakage/shift (0.61 -> 0.40).
2. ADD: 'Texture' features to distinguish Pigeons (continuous) vs Songbirds (bounding).
   - 'Bounding Flight' Index: Standard deviation of rolling RCS variance.
   - Altitude Oscillation: Songbirds drop altitude when bounding (wings folded).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import (
    extract_core_features, extract_rcs_fft_features,
    extract_wingbeat_features, extract_shape_features,
    extract_flight_mode_features, add_tabular_features,
    add_biological_time_features,
)
from src.metrics import compute_map, print_results
from src.submission import save_submission

# ============================================================
# 1. NEW: TEXTURE & BOUNDING FLIGHT FEATURES
# ============================================================
def extract_texture_features(trajectory, time_seq):
    """
    Captures the 'rhythm' of the flight to find Songbirds.
    """
    pts = parse_ewkb_4d(trajectory)
    times = parse_trajectory_time(time_seq)
    
    if len(pts) < 10: return {}
    
    traj = np.array(pts)
    rcs = traj[:, 3]
    alts = traj[:, 2]
    
    # 1. RCS Texture (Bounding Flight Detector)
    # Songbirds: [Flap, Flap, Pause, Pause] -> Variance changes.
    # Pigeons:   [Flap, Flap, Flap, Flap]   -> Variance constant.
    
    # Calculate Variance in small windows (e.g., 0.5s or 10 samples)
    window = min(len(rcs)//2, 10)
    if window < 2: return {}
    
    rolling_var = pd.Series(rcs).rolling(window).var().dropna()
    
    # "Texture Stability": If high, it's bounding (Songbird). If low, it's constant (Pigeon/Gull).
    var_of_vars = rolling_var.std()
    mean_of_vars = rolling_var.mean() + 1e-9
    texture_cv = var_of_vars / mean_of_vars
    
    # 2. Altitude Texture (Bounding Drop)
    # When Songbirds bound (wings folded), they drop slightly in altitude.
    # We look for high frequency altitude "wiggle".
    alt_diffs = np.diff(alts)
    alt_jerk = np.diff(alt_diffs) # 2nd derivative
    
    return {
        "txt_rcs_var_stability": var_of_vars, # High for Songbirds
        "txt_rcs_texture_cv": texture_cv,     # Normalized bounding index
        "txt_alt_jerk_mean": np.mean(np.abs(alt_jerk)) if len(alt_jerk)>0 else 0,
        "txt_alt_entropy": float(pd.Series(alts).nunique()) / len(alts) # "Complexity" of path
    }

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def extract_spectrogram_features(trajectory, time_seq):
    # (Restored from E08 - working version)
    pts = parse_ewkb_4d(trajectory)
    times = parse_trajectory_time(time_seq)
    if len(pts) < 10 or len(times) < 10: return {}
    traj = np.array(pts)
    rcs = traj[:, 3]
    try:
        duration = times[-1] - times[0]
        if duration <= 0: return {}
        fs = 20.0
        n_points = int(duration * fs)
        if n_points < 16: return {}
        t_new = np.linspace(times[0], times[-1], n_points)
        f_interp = interp1d(times, rcs, kind='linear', fill_value="extrapolate")
        rcs_detrend = f_interp(t_new) - np.mean(rcs)
        f, t, Sxx = spectrogram(rcs_detrend, fs=fs, nperseg=min(32, len(rcs_detrend)))
        total_E = np.sum(Sxx, axis=0)
        mean_E = np.mean(total_E) + 1e-9
        return {
            "spec_burstiness": np.std(total_E) / mean_E,
            "spec_mean_E": mean_E,
            "band_high_rel": np.sum(Sxx[f>=6,:]) / (np.sum(Sxx) + 1e-9)
        }
    except: return {}

def extract_all(df):
    rows = []
    print(f"Extracting features for {len(df)} tracks...")
    for i, (_, r) in enumerate(df.iterrows()):
        feats = {}
        # Core & E08 features (Proven Good)
        feats.update(extract_core_features(r.trajectory, r.trajectory_time))
        feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
        feats.update(extract_wingbeat_features(r.trajectory, r.trajectory_time))
        feats.update(extract_shape_features(r.trajectory, r.trajectory_time))
        feats.update(extract_flight_mode_features(r.trajectory, r.trajectory_time))
        feats.update(extract_spectrogram_features(r.trajectory, r.trajectory_time))
        
        # New E10 Texture features
        feats.update(extract_texture_features(r.trajectory, r.trajectory_time))
        
        rows.append(feats)
        if (i+1)%500==0: print(f"  {i+1}/{len(df)}")
    
    feat_df = pd.DataFrame(rows)
    feat_df = add_tabular_features(feat_df, df)
    feat_df = add_biological_time_features(feat_df, df)
    
    # NO CONTEXT FEATURES! (They caused the drop)
    return feat_df

# ============================================================
# 3. PIPELINE
# ============================================================
train = load_train()
test = load_test()

train_feats = extract_all(train)
test_feats = extract_all(test)

DROP = [
    "hour", "month", "dayofweek", "time_of_day", 
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_pigeon_window", "lon_mean", "lat_mean", "lon_std", "lat_std", "spatial_spread",
    "timestamp_start_radar_utc", "track_id", "ts_temp"
]
train_feats = train_feats.drop(columns=[c for c in DROP if c in train_feats.columns], errors='ignore')
test_feats = test_feats.drop(columns=[c for c in DROP if c in test_feats.columns], errors='ignore')

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================
# 4. TRAINING
# ============================================================
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train["bird_group"])
groups = train["primary_observation_id"].values
X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# Weights
counts = np.bincount(y, minlength=len(CLASSES))
weights = len(y) / (len(CLASSES) * counts)
sample_weights = np.array([weights[yi] for yi in y])

# CV
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(CLASSES)))
test_preds = np.zeros((len(X_test), len(CLASSES)))

print("Training Ensemble (Texture/Recovery)...")

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]
    
    # LGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    model = lgb.train({
        "objective": "multiclass", "num_class": len(CLASSES), "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "subsample": 0.7,
        "colsample_bytree": 0.6, "reg_lambda": 3, "verbose": -1, "seed": 42,
        "is_unbalance": True
    }, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    oof_preds[va_idx] += model.predict(X_va) * 0.4
    test_preds += model.predict(X_test) * 0.4 / 5
    
    # XGB
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    model_xgb = xgb.train({
        "objective": "multi:softprob", "num_class": len(CLASSES), "eval_metric": "mlogloss",
        "eta": 0.03, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.6, "seed": 42, "verbosity": 0
    }, dtrain_xgb, num_boost_round=2000, evals=[(dval_xgb, "val")], early_stopping_rounds=100, verbose_eval=False)
    
    oof_preds[va_idx] += model_xgb.predict(dval_xgb) * 0.3
    test_preds += model_xgb.predict(xgb.DMatrix(X_test, feature_names=feature_names)) * 0.3 / 5
    
    # CB
    model_cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=5, loss_function="MultiClass",
        random_seed=42, verbose=0, allow_writing_files=False, auto_class_weights="Balanced"
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    
    oof_preds[va_idx] += model_cb.predict_proba(X_va) * 0.3
    test_preds += model_cb.predict_proba(X_test) * 0.3 / 5
    
    print(f"Fold {fold} done.")

map_score, per_class = compute_map(y, oof_preds)
print_results(map_score, per_class, "E10 Texture (No Context)")

save_submission(test_preds, "e10_texture", cv_map=map_score)
print("Done.")