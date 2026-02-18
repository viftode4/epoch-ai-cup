"""
E16: Absolute Frequency Fix.

CRITICAL FIX:
- Previous versions normalized wingbeat freq by Nyquist (sampling rate).
- This destroyed the biological signal (Hz) because radar sampling rates vary.
- We now extract ABSOLUTE Hz frequencies to separate:
  - Geese/Cormorants (~3-4 Hz)
  - Ducks/Pigeons (~5-8 Hz)
  - Songbirds (~10+ Hz or bursts)
  - Gulls/BoP (<1 Hz or DC)

Features:
- wb_freq_hz: Dominant frequency in Hz.
- wb_power_bands: Energy in 0-2, 2-4, 4-8, 8+ Hz.
- wb_freq_stability: Std dev of frequency over time (Cormorants = Stable).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.signal import spectrogram, welch
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

from src.data import load_train, load_test, parse_ewkb_4d, parse_trajectory_time, ROOT
from src.features import (
    extract_core_features, extract_rcs_fft_features,
    extract_shape_features, extract_flight_mode_features, 
    add_tabular_features, add_biological_time_features,
)

CLASSES = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

# ============================================================
# 1. NEW: ABSOLUTE FREQUENCY EXTRACTOR
# ============================================================
def extract_absolute_wingbeat(trajectory, time_seq):
    """
    Extracts wingbeat features in ABSOLUTE Hertz (Hz).
    """
    pts = parse_ewkb_4d(trajectory)
    times = parse_trajectory_time(time_seq)
    if len(pts) < 10: return {}
    
    traj = np.array(pts)
    rcs = traj[:, 3]
    
    # 1. Interpolate to fixed 20Hz (Nyquist = 10Hz)
    # This ensures 5Hz is always 5Hz
    try:
        duration = times[-1] - times[0]
        if duration <= 0: return {}
        fs = 20.0 
        n_points = int(duration * fs)
        if n_points < 32: return {} # Need enough points for FFT
        
        t_new = np.linspace(times[0], times[-1], n_points)
        f_interp = interp1d(times, rcs, kind='linear', fill_value="extrapolate")
        rcs_const = f_interp(t_new)
        rcs_ac = rcs_const - np.mean(rcs_const)
        
        # 2. Welch's Periodogram (PSD)
        freqs, psd = welch(rcs_ac, fs=fs, nperseg=min(64, len(rcs_ac)))
        
        # 3. Absolute Bands
        # 0-1 Hz: Gliding / Thermal (Gulls, BoP)
        # 1-3 Hz: Slow Flap (Herons? Large Geese?)
        # 3-6 Hz: Medium Flap (Cormorants, Geese, Ducks)
        # 6-10 Hz: Fast Flap (Pigeons, Small Ducks, Waders)
        
        def get_band_power(f_min, f_max):
            mask = (freqs >= f_min) & (freqs < f_max)
            return np.sum(psd[mask])
            
        total_p = np.sum(psd) + 1e-9
        
        p_glide = get_band_power(0, 1)
        p_slow  = get_band_power(1, 3)
        p_med   = get_band_power(3, 6)
        p_fast  = get_band_power(6, 10)
        
        # 4. Dominant Frequency (Hz)
        peak_idx = np.argmax(psd)
        peak_hz = freqs[peak_idx]
        
        # 5. Frequency Stability (Spectrogram approach)
        # Split into small windows, get peak freq of each, calculate std
        f_spec, t_spec, Sxx = spectrogram(rcs_ac, fs=fs, nperseg=32, noverlap=16)
        # Find peak freq at each time step
        peak_freqs_over_time = f_spec[np.argmax(Sxx, axis=0)]
        freq_stability = np.std(peak_freqs_over_time) # Low = Stable (Cormorant), High = Burst (Songbird)
        
        return {
            "wb_hz_peak": peak_hz,
            "wb_hz_stability": freq_stability,
            "wb_band_0_1Hz": p_glide / total_p,
            "wb_band_1_3Hz": p_slow / total_p,
            "wb_band_3_6Hz": p_med / total_p,
            "wb_band_6_10Hz": p_fast / total_p,
            "wb_energy_total": np.log1p(total_p)
        }
        
    except Exception:
        return {}

# ============================================================
# 2. FEATURE PIPELINE
# ============================================================
def extract_linearity_features(trajectory, time_seq):
    # (Restored from E15 - Good for Cormorants)
    pts = parse_ewkb_4d(trajectory)
    if len(pts) < 3: return {}
    coords = np.array([p[:2] for p in pts])
    start, end = coords[0], coords[-1]
    vec_line = end - start
    len_line = np.linalg.norm(vec_line)
    if len_line < 1e-6: return {"lin_error_mean": 0}
    vec_line_norm = vec_line / len_line
    vec_points = coords - start
    projections = np.dot(vec_points, vec_line_norm)
    closest_points = np.outer(projections, vec_line_norm)
    errors = np.linalg.norm(vec_points - closest_points, axis=1)
    return {"lin_error_mean": np.mean(errors / (len_line + 1e-6))}

def extract_all(df):
    rows = []
    print(f"Extracting features for {len(df)} tracks...")
    for i, (_, r) in enumerate(df.iterrows()):
        feats = {}
        # Core & Shape (proven)
        feats.update(extract_core_features(r.trajectory, r.trajectory_time))
        feats.update(extract_shape_features(r.trajectory, r.trajectory_time))
        feats.update(extract_flight_mode_features(r.trajectory, r.trajectory_time))
        
        # New Absolute Frequency (Replaces old wingbeat)
        feats.update(extract_absolute_wingbeat(r.trajectory, r.trajectory_time))
        
        # Linearity (Cormorants)
        feats.update(extract_linearity_features(r.trajectory, r.trajectory_time))
        
        rows.append(feats)
        if (i+1)%500==0: print(f"  {i+1}/{len(df)}")
    
    feat_df = pd.DataFrame(rows)
    feat_df = add_tabular_features(feat_df, df)
    feat_df = add_biological_time_features(feat_df, df)
    return feat_df

# ============================================================
# 3. EXECUTION
# ============================================================
train = load_train()
test = load_test()

train_feats = extract_all(train)
test_feats = extract_all(test)

DROP = [
    "hour", "month", "dayofweek", "time_of_day", 
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_pigeon_window", "lon_mean", "lat_mean", "lon_std", "lat_std", "spatial_spread",
    "timestamp_start_radar_utc", "track_id"
]
train_feats = train_feats.drop(columns=[c for c in DROP if c in train_feats.columns], errors='ignore')
test_feats = test_feats.drop(columns=[c for c in DROP if c in test_feats.columns], errors='ignore')
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

# OVERSAMPLING (Keep E15 logic)
y = train["bird_group"].map(CLASS_MAP).values
groups = train["primary_observation_id"].values
X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(CLASSES)))
test_preds = np.zeros((len(X_test), len(CLASSES)))

print("\nTraining Ensemble (Absolute Hz + Oversampling)...")

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    
    # Oversampling multipliers
    multipliers = {1: 4, 6: 3, 7: 3, 3: 2} # Cormorants, BoP, Waders, Ducks
    X_res, y_res = [X_tr], [y_tr]
    for c, m in multipliers.items():
        mask = (y_tr == c)
        if mask.sum() > 0:
            X_res.append(np.tile(X_tr[mask], (m-1, 1)))
            y_res.append(np.tile(y_tr[mask], (m-1,)))
    X_tr_final = np.vstack(X_res)
    y_tr_final = np.concatenate(y_res)
    perm = np.random.permutation(len(X_tr_final))
    X_tr_final, y_tr_final = X_tr_final[perm], y_tr_final[perm]
    
    # Recalc weights
    counts = np.bincount(y_tr_final, minlength=len(CLASSES))
    weights = len(y_tr_final) / (len(CLASSES) * counts)
    w_tr_final = np.array([weights[yi] for yi in y_tr_final])
    
    # LGBM
    dtrain = lgb.Dataset(X_tr_final, label=y_tr_final, weight=w_tr_final, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    model = lgb.train({
        "objective": "multiclass", "num_class": len(CLASSES), "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "subsample": 0.7,
        "colsample_bytree": 0.6, "reg_lambda": 3, "verbose": -1, "seed": 42
    }, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    oof_preds[va_idx] += model.predict(X_va) * 0.4
    test_preds += model.predict(X_test) * 0.4 / 5
    
    # XGB
    dtrain_xgb = xgb.DMatrix(X_tr_final, label=y_tr_final, weight=w_tr_final, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    model_xgb = xgb.train({
        "objective": "multi:softprob", "num_class": len(CLASSES), "eval_metric": "mlogloss",
        "eta": 0.03, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.6, "seed": 42, "verbosity": 0
    }, dtrain_xgb, num_boost_round=2000, evals=[(dval_xgb, "val")], early_stopping_rounds=100, verbose_eval=False)
    
    oof_preds[va_idx] += model_xgb.predict(dval_xgb) * 0.3
    test_preds += model_xgb.predict(xgb.DMatrix(X_test, feature_names=feature_names)) * 0.3 / 5
    
    # CB
    model_cb = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=5, loss_function="MultiClass",
                                  random_seed=42, verbose=0, allow_writing_files=False)
    model_cb.fit(X_tr_final, y_tr_final, eval_set=(X_va, y_va), verbose=0)
    oof_preds[va_idx] += model_cb.predict_proba(X_va) * 0.3
    test_preds += model_cb.predict_proba(X_test) * 0.3 / 5
    
    print(f"Fold {fold} done.")

def compute_map_local(y_true, y_pred):
    n_classes = len(CLASSES)
    y_onehot = np.eye(n_classes)[y_true]
    per_class = {}
    for c in range(n_classes):
        if y_onehot[:, c].sum() > 0:
            per_class[CLASSES[c]] = average_precision_score(y_onehot[:, c], y_pred[:, c])
        else: per_class[CLASSES[c]] = 0.0
    return np.mean(list(per_class.values())), per_class

def save_submission_local(test_preds, name, cv_map=None):
    from datetime import datetime
    test = load_test()
    sub = pd.DataFrame({"track_id": test["track_id"]})
    for i, cls in enumerate(CLASSES): sub[cls] = test_preds[:, i]
    submissions_dir = ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    score_str = f"_{cv_map:.4f}" if cv_map is not None else ""
    path = submissions_dir / f"{name}{score_str}_{ts}.csv"
    sub.to_csv(path, index=False)
    print(f"Saved: {path.name}")

map_score, per_class = compute_map_local(y, oof_preds)
print(f"\nOverall mAP: {map_score:.4f}")
for cls in CLASSES: print(f"  {cls:15s}: {per_class[cls]:.4f}")

save_submission_local(test_preds, "e16_abs_freq", cv_map=map_score)