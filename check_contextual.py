import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from src.data import CLASSES, load_train
from src.features import build_features
from src.metrics import compute_map

train_df = load_train()
y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

PRUNED_FEATURES = [
    'airspeed', 'radar_bird_size', 'min_z', 'max_z', 'duration', 'track_length',
    'z_mean', 'z_std', 'rcs_mean', 'rcs_std', 'rcs_max', 'rcs_min',
    'speed_mean', 'speed_std', 'speed_max', 'speed_min', 'accel_mean', 'accel_std',
    'accel_max', 'accel_min', 'turn_mean', 'turn_std', 'turn_max', 'turn_min',
    'sinuosity', 'rcs_range', 'rcs_q25', 'rcs_q75', 'rcs_iqr', 'rcs_skew', 'rcs_kurtosis',
    'rcs_trend', 'rcs_p2p', 'rcs_smoothness', 'rcs_ac1', 'rcs_ac2'
]
X_df = build_features(train_df, feature_sets=["core", "tabular", "rcs_fft"])
keep_cols = [c for c in PRUNED_FEATURES if c in X_df.columns]
X_base = X_df[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Build Contextual Features
train_df["dt"] = pd.to_datetime(train_df["timestamp_start_radar_utc"])
time_s = train_df["dt"].astype(int).values / 10**6

speed = pd.to_numeric(train_df["airspeed"], errors="coerce").fillna(10.0).values
alt = pd.to_numeric(train_df["min_z"], errors="coerce").fillna(0.0).values
size_map = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}
size = train_df["radar_bird_size"].map(size_map).fillna(2).values

n_neighbors = np.zeros(len(train_df))
neighbor_mean_speed = np.zeros(len(train_df))
neighbor_mean_alt = np.zeros(len(train_df))
neighbor_mean_size = np.zeros(len(train_df))

for i in range(len(train_df)):
    # window: +/- 60 seconds
    mask = (np.abs(time_s - time_s[i]) <= 60)
    mask[i] = False # exclude self
    
    n_neighbors[i] = mask.sum()
    if mask.sum() > 0:
        neighbor_mean_speed[i] = np.mean(speed[mask])
        neighbor_mean_alt[i] = np.mean(alt[mask])
        neighbor_mean_size[i] = np.mean(size[mask])
    else:
        neighbor_mean_speed[i] = speed[i]
        neighbor_mean_alt[i] = alt[i]
        neighbor_mean_size[i] = size[i]

X_ctx = pd.DataFrame({
    "n_neighbors": n_neighbors,
    "neighbor_mean_speed": neighbor_mean_speed,
    "neighbor_mean_alt": neighbor_mean_alt,
    "neighbor_mean_size": neighbor_mean_size,
    "speed_diff": speed - neighbor_mean_speed,
    "alt_diff": alt - neighbor_mean_alt,
    "size_diff": size - neighbor_mean_size
})

X_full = pd.concat([X_base, X_ctx], axis=1)

if 'radar_bird_size' in X_full.columns and X_full['radar_bird_size'].dtype == 'object':
    X_full = pd.get_dummies(X_full, columns=['radar_bird_size'])

X = X_full.to_numpy(dtype=np.float32)

unique_months = sorted(set(train_months))
oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)

for m in unique_months:
    tr_idx = np.where(train_months != m)[0]
    va_idx = np.where(train_months == m)[0]
    
    lgb = LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES),
        learning_rate=0.02, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    )
    lgb.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = lgb.predict_proba(X[va_idx])

m_ctx, _ = compute_map(y, oof)
print(f"Contextual Features LOMO mAP: {m_ctx:.4f}")

# Compare to base
X_b = X_base.copy()
if 'radar_bird_size' in X_b.columns and X_b['radar_bird_size'].dtype == 'object':
    X_b = pd.get_dummies(X_b, columns=['radar_bird_size'])
X_b = X_b.to_numpy(dtype=np.float32)

oof_b = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
for m in unique_months:
    tr_idx = np.where(train_months != m)[0]
    va_idx = np.where(train_months == m)[0]
    lgb = LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES),
        learning_rate=0.02, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    )
    lgb.fit(X_b[tr_idx], y[tr_idx])
    oof_b[va_idx] = lgb.predict_proba(X_b[va_idx])

m_b, _ = compute_map(y, oof_b)
print(f"Base Features LOMO mAP: {m_b:.4f}")
print(f"Delta: {m_ctx - m_b:+.4f}")

