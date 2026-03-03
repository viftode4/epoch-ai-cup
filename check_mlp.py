import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
X_df = X_df[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

if 'radar_bird_size' in X_df.columns and X_df['radar_bird_size'].dtype == 'object':
    X_df = pd.get_dummies(X_df, columns=['radar_bird_size'])

X = X_df.to_numpy(dtype=np.float32)

unique_months = sorted(set(train_months))
oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)

for m in unique_months:
    tr_idx = np.where(train_months != m)[0]
    va_idx = np.where(train_months == m)[0]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_tr, y[tr_idx])
    oof[va_idx] = model.predict_proba(X_va)

m_mlp, _ = compute_map(y, oof)
print(f"MLP LOMO mAP: {m_mlp:.4f}")

oof_e50 = np.load("oof_e50.npy")
ens = 0.8 * oof_e50 + 0.2 * oof
m_ens, _ = compute_map(y, ens)
print(f"Ensemble (80% Tree + 20% MLP) LOMO mAP: {m_ens:.4f} (Delta: {m_ens - 0.3625:+.4f})")
