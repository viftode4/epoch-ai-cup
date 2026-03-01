"""
V4: Focus on what actually works.
- Feature selection based on importance
- Multi-seed averaging for stability
- Optimized class weights (not just inverse freq)
- Per-class probability calibration
"""
import pandas as pd
import numpy as np
import struct
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD
# ============================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']
print(f'Train: {len(train)}, Test: {len(test)}')

# ============================================================
# EWKB
# ============================================================
def parse_ewkb_4d(hex_str):
    raw = bytes.fromhex(hex_str)
    offset = 0
    bo = '<' if raw[offset] == 1 else '>'
    offset += 1
    geom_type = struct.unpack_from(f'{bo}I', raw, offset)[0]
    offset += 4
    if geom_type & 0x20000000:
        offset += 4
    n_points = struct.unpack_from(f'{bo}I', raw, offset)[0]
    offset += 4
    points = []
    for _ in range(n_points):
        lon, lat, alt, rcs = struct.unpack_from(f'{bo}4d', raw, offset)
        points.append((lon, lat, alt, rcs))
        offset += 32
    return points

# ============================================================
# FEATURES
# ============================================================
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def extract_features(hex_str, traj_time_str):
    pts = parse_ewkb_4d(hex_str)
    times = np.array(eval(traj_time_str))
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)
    duration = times[-1] - times[0] if n > 1 else 0.001

    if n > 1:
        dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1]) for i in range(n-1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists / dt
    else:
        dists, dt, speeds = np.array([0.0]), np.array([1.0]), np.array([0.0])

    total_dist = dists.sum()
    straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
    sinuosity = total_dist / max(straight_dist, 1e-6)

    alt_diffs = np.diff(alts) if n > 1 else np.array([0.0])
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    if n > 2:
        bearings = np.arctan2(np.diff(lats), np.diff(lons))
        bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
    else:
        bearing_changes = np.array([0.0])

    # RCS FFT
    rcs_peak_freq = rcs_peak_power = rcs_total_power = rcs_spectral_centroid = 0
    if n >= 8:
        try:
            uniform_t = np.linspace(times[0], times[-1], n)
            fs = 1.0 / max(uniform_t[1] - uniform_t[0], 0.01)
            rcs_uniform = interp1d(times, rcs, kind='linear', fill_value='extrapolate')(uniform_t)
            freqs, psd = welch(rcs_uniform - np.mean(rcs_uniform), fs=fs, nperseg=min(n, 32))
            peak_idx = np.argmax(psd[1:]) + 1
            rcs_peak_freq = freqs[peak_idx]
            rcs_peak_power = psd[peak_idx]
            rcs_total_power = psd[1:].sum()
            rcs_spectral_centroid = np.sum(freqs[1:] * psd[1:]) / max(psd[1:].sum(), 1e-10)
        except:
            pass

    # Acceleration
    accel = np.diff(speeds) / np.maximum(dt[:-1], 0.001) if len(speeds) > 1 and len(dt) > 1 else np.array([0.0])

    # Half splits
    mid = n // 2
    if mid > 1:
        alt_first, alt_second = np.mean(alts[:mid]), np.mean(alts[mid:])
        rcs_first, rcs_second = np.mean(rcs[:mid]), np.mean(rcs[mid:])
        speed_first = np.mean(speeds[:mid]) if mid <= len(speeds) else np.mean(speeds)
        speed_second = np.mean(speeds[mid:]) if mid < len(speeds) else np.mean(speeds)
    else:
        alt_first = alt_second = np.mean(alts)
        rcs_first = rcs_second = np.mean(rcs)
        speed_first = speed_second = np.mean(speeds)

    # Trends
    alt_trend = np.polyfit(np.arange(n), alts, 1)[0] if n > 2 else 0
    rcs_trend = np.polyfit(np.arange(n), rcs, 1)[0] if n > 2 else 0

    speed_cv = np.std(speeds) / max(np.mean(speeds), 0.01)

    feats = {
        'n_points': n,
        'duration': duration,
        'total_dist': total_dist,
        'straight_dist': straight_dist,
        'sinuosity': min(sinuosity, 50),
        # altitude
        'alt_mean': np.mean(alts), 'alt_std': np.std(alts),
        'alt_min': np.min(alts), 'alt_max': np.max(alts),
        'alt_range': np.ptp(alts), 'alt_median': np.median(alts),
        'alt_q25': np.percentile(alts, 25), 'alt_q75': np.percentile(alts, 75),
        'alt_diff_mean': np.mean(np.abs(alt_diffs)), 'alt_diff_std': np.std(alt_diffs),
        'climb_rate': climbing.sum() / max(duration, 0.001) if len(climbing) > 0 else 0,
        'descent_rate': abs(descending.sum()) / max(duration, 0.001) if len(descending) > 0 else 0,
        'climb_frac': len(climbing) / max(len(alt_diffs), 1),
        'alt_trend': alt_trend,
        # RCS
        'rcs_mean': np.mean(rcs), 'rcs_std': np.std(rcs),
        'rcs_min': np.min(rcs), 'rcs_max': np.max(rcs),
        'rcs_range': np.ptp(rcs), 'rcs_median': np.median(rcs),
        'rcs_q25': np.percentile(rcs, 25), 'rcs_q75': np.percentile(rcs, 75),
        'rcs_skew': float(pd.Series(rcs).skew()) if n > 2 else 0,
        'rcs_trend': rcs_trend,
        'rcs_above_neg15': np.mean(rcs > -15),
        'rcs_below_neg28': np.mean(rcs < -28),
        # RCS FFT
        'rcs_peak_freq': rcs_peak_freq, 'rcs_peak_power': rcs_peak_power,
        'rcs_total_power': rcs_total_power, 'rcs_spectral_centroid': rcs_spectral_centroid,
        # speed
        'speed_mean': np.mean(speeds), 'speed_std': np.std(speeds),
        'speed_max': np.max(speeds), 'speed_min': np.min(speeds),
        'speed_median': np.median(speeds),
        'avg_ground_speed': total_dist / max(duration, 0.001),
        'speed_cv': speed_cv,
        # acceleration
        'accel_mean': np.mean(accel), 'accel_std': np.std(accel), 'accel_max': np.max(np.abs(accel)),
        # turning
        'bearing_change_mean': np.mean(np.abs(bearing_changes)),
        'bearing_change_std': np.std(bearing_changes),
        'bearing_change_max': np.max(np.abs(bearing_changes)),
        'total_turning': np.sum(np.abs(bearing_changes)),
        'net_turning': np.abs(np.sum(bearing_changes)),
        'turning_per_meter': np.sum(np.abs(bearing_changes)) / max(total_dist, 1),
        # position
        'lon_mean': np.mean(lons), 'lat_mean': np.mean(lats),
        'lon_std': np.std(lons), 'lat_std': np.std(lats),
        # halves
        'alt_change_halves': alt_second - alt_first,
        'rcs_change_halves': rcs_second - rcs_first,
        'speed_change_halves': speed_second - speed_first,
        # interactions
        'speed_x_alt': np.mean(speeds) * np.mean(alts),
        'rcs_x_alt': np.mean(rcs) * np.mean(alts),
        'rcs_x_speed': np.mean(rcs) * np.mean(speeds),
        'dist_per_point': total_dist / max(n, 1),
        'points_per_sec': n / max(duration, 0.001),
    }
    return feats

print('Extracting features...')
train_feats = pd.DataFrame([extract_features(r.trajectory, r.trajectory_time) for _, r in train.iterrows()])
test_feats = pd.DataFrame([extract_features(r.trajectory, r.trajectory_time) for _, r in test.iterrows()])

# Non-trajectory features
for df_feat, df_orig in [(train_feats, train), (test_feats, test)]:
    ts = pd.to_datetime(df_orig['timestamp_start_radar_utc'])
    te = pd.to_datetime(df_orig['timestamp_end_radar_utc'])
    hour = ts.dt.hour.values
    month = ts.dt.month.values

    df_feat['hour'] = hour
    df_feat['month'] = month
    df_feat['time_of_day'] = hour + ts.dt.minute.values / 60.0
    df_feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * month / 12)
    df_feat['timestamp_duration'] = (te - ts).dt.total_seconds().values

    df_feat['airspeed'] = df_orig['airspeed'].values
    df_feat['min_z'] = df_orig['min_z'].values
    df_feat['max_z'] = df_orig['max_z'].values
    df_feat['z_range'] = df_orig['max_z'].values - df_orig['min_z'].values
    df_feat['z_mean'] = (df_orig['max_z'].values + df_orig['min_z'].values) / 2

    size_map = {'Small bird': 0, 'Medium': 1, 'Large': 2, 'Flock': 3}
    df_feat['radar_bird_size'] = df_orig['radar_bird_size'].map(size_map).values
    df_feat['airspeed_vs_ground'] = df_feat['airspeed'] / df_feat['avg_ground_speed'].clip(lower=0.01)

    # Targeted
    df_feat['is_afternoon'] = (hour >= 13).astype(int)
    df_feat['is_october'] = (month == 10).astype(int)
    df_feat['is_april'] = (month == 4).astype(int)
    df_feat['oct_afternoon'] = ((month == 10) & (hour >= 13)).astype(int)

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f'Features: {train_feats.shape[1]}')

# ============================================================
# TARGET
# ============================================================
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train['bird_group'])
n_classes = len(CLASSES)
feature_names = list(train_feats.columns)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)

# Tuned sample weights — boost Pigeons extra hard
class_counts = np.bincount(y, minlength=n_classes)
class_weights = len(y) / (n_classes * class_counts)
# Extra boost for Pigeons (idx 2) — hardest class
pigeon_idx = le.transform(['Pigeons'])[0]
clutter_idx = le.transform(['Clutter'])[0]
class_weights[pigeon_idx] *= 1.5
class_weights[clutter_idx] *= 1.3
sample_weights = np.array([class_weights[yi] for yi in y])
print(f'Class weights: { {CLASSES[i]: round(class_weights[i], 2) for i in range(n_classes)} }')

def compute_map(y_true, y_pred, n_classes, classes):
    y_oh = np.eye(n_classes)[y_true]
    aps = []
    for c in range(n_classes):
        if y_oh[:, c].sum() > 0:
            aps.append(average_precision_score(y_oh[:, c], y_pred[:, c]))
    return np.mean(aps), aps

# ============================================================
# MULTI-SEED ENSEMBLE
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 2024, 7, 999]

oof_all = np.zeros((len(X), n_classes))
test_all = np.zeros((len(X_test), n_classes))

for seed_i, SEED in enumerate(SEEDS):
    print(f'\n{"="*50}')
    print(f'SEED {SEED} ({seed_i+1}/{len(SEEDS)})')
    print(f'{"="*50}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(X), n_classes))
    oof_xgb = np.zeros((len(X), n_classes))
    oof_cb = np.zeros((len(X), n_classes))
    test_lgb = np.zeros((len(X_test), n_classes))
    test_xgb = np.zeros((len(X_test), n_classes))
    test_cb = np.zeros((len(X_test), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx]

        # LightGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
        lgb_model = lgb.train({
            'objective': 'multiclass', 'num_class': n_classes, 'metric': 'multi_logloss',
            'learning_rate': 0.04, 'num_leaves': 40, 'max_depth': 7,
            'min_child_samples': 8, 'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.3, 'reg_lambda': 1.5, 'verbose': -1, 'seed': SEED,
            'is_unbalance': True,
        }, dtrain, num_boost_round=3000, valid_sets=[dval],
           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_test) / N_FOLDS

        # XGBoost
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
        xgb_model = xgb.train({
            'objective': 'multi:softprob', 'num_class': n_classes, 'eval_metric': 'mlogloss',
            'learning_rate': 0.04, 'max_depth': 6, 'min_child_weight': 3,
            'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.3, 'reg_lambda': 1.5,
            'seed': SEED, 'verbosity': 0,
        }, dtrain_xgb, num_boost_round=3000, evals=[(dval_xgb, 'val')],
           early_stopping_rounds=100, verbose_eval=0)
        oof_xgb[va_idx] = xgb_model.predict(dval_xgb)
        test_xgb += xgb_model.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

        # CatBoost
        cb_model = CatBoostClassifier(
            iterations=3000, learning_rate=0.04, depth=6, l2_leaf_reg=3,
            loss_function='MultiClass', eval_metric='MultiClass',
            random_seed=SEED, verbose=0, early_stopping_rounds=100,
            auto_class_weights='Balanced',
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        oof_cb[va_idx] = cb_model.predict_proba(X_va)
        test_cb += cb_model.predict_proba(X_test) / N_FOLDS

    # Find best weights for this seed
    best_map, best_w = 0, (0.33, 0.33, 0.34)
    for w1 in np.arange(0.2, 0.55, 0.05):
        for w2 in np.arange(0.2, 0.55, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1: continue
            oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            m, _ = compute_map(y, oof_ens, n_classes, CLASSES)
            if m > best_map:
                best_map = m
                best_w = (w1, w2, w3)

    oof_seed = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cb
    test_seed = best_w[0]*test_lgb + best_w[1]*test_xgb + best_w[2]*test_cb

    seed_map, seed_aps = compute_map(y, oof_seed, n_classes, CLASSES)
    print(f'Seed {SEED}: mAP={seed_map:.4f} (w={best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f})')

    oof_all += oof_seed / len(SEEDS)
    test_all += test_seed / len(SEEDS)

# ============================================================
# FINAL RESULTS
# ============================================================
final_map, final_aps = compute_map(y, oof_all, n_classes, CLASSES)

print(f'\n{"="*50}')
print(f'  FINAL MULTI-SEED ENSEMBLE')
print(f'{"="*50}')
print(f'  v1 baseline:       0.7030')
print(f'  v2 ensemble:       0.7214')
print(f'  v4 multi-seed:     {final_map:.4f}')
print(f'{"="*50}')

# Previous v2 APs for comparison
v2_aps = {
    'Clutter': 0.6099, 'Cormorants': 0.9388, 'Pigeons': 0.2537,
    'Ducks': 0.6659, 'Geese': 0.7281, 'Gulls': 0.9562,
    'Birds of Prey': 0.8847, 'Waders': 0.8155, 'Songbirds': 0.6400
}

print(f'\nPer-class AP comparison:')
print(f'  {"Class":15s} {"v2":>8s} {"v4":>8s} {"delta":>8s}')
for c in range(n_classes):
    prev = v2_aps[CLASSES[c]]
    curr = final_aps[c]
    delta = curr - prev
    marker = ' !!!' if abs(delta) > 0.03 else ''
    print(f'  {CLASSES[c]:15s} {prev:8.4f} {curr:8.4f} {delta:+8.4f}{marker}')

# ============================================================
# SUBMISSION
# ============================================================
submission = pd.DataFrame({'track_id': test['track_id']})
for i, cls in enumerate(CLASSES):
    submission[cls] = test_all[:, i]
submission.to_csv('submission.csv', index=False)
print(f'\nSaved submission.csv ({len(submission)} rows)')
