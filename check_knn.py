import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from src.data import CLASSES, load_train
from src.features import build_features
from src.metrics import compute_map

train_df = load_train()
y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes

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

oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, va_idx in skf.split(X, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(X_tr, y[tr_idx])
    oof[va_idx] = knn.predict_proba(X_va)

m_knn, _ = compute_map(y, oof)
print(f"kNN SKF mAP: {m_knn:.4f}")

oof_e50 = np.load("oof_e50.npy")
ens = 0.8 * oof_e50 + 0.2 * oof
m_ens, _ = compute_map(y, ens)
print(f"Ensemble (80% Tree + 20% kNN) SKF mAP: {m_ens:.4f}")
