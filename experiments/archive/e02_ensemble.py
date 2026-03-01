import pandas as pd
import numpy as np
import struct
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']

print(f'Train: {len(train)}, Test: {len(test)}')
print(f'\nClass distribution:\n{train.bird_group.value_counts()}')
print(f'\nClass %:\n{(train.bird_group.value_counts(normalize=True)*100).round(1)}')

# ============================================================
# 2. DATA EXPLORATION & CLEANING
# ============================================================
# Check for missing values
print(f'\n--- Missing values (train) ---')
missing = train.isnull().sum()
print(missing[missing > 0] if missing.any() else 'None')

print(f'\n--- Missing values (test) ---')
missing_t = test.isnull().sum()
print(missing_t[missing_t > 0] if missing_t.any() else 'None')

# Check for duplicate track_ids
print(f'\nDuplicate track_ids in train: {train.track_id.duplicated().sum()}')
print(f'Duplicate track_ids in test: {test.track_id.duplicated().sum()}')

# Basic stats on numeric columns
print(f'\n--- Airspeed stats ---')
print(train.groupby('bird_group')['airspeed'].describe().round(1).to_string())

print(f'\n--- Radar bird size distribution per class ---')
print(pd.crosstab(train['bird_group'], train['radar_bird_size'], normalize='index').round(2).to_string())

# ============================================================
# 3. EWKB PARSING
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
# 4. FEATURE ENGINEERING (ENRICHED)
# ============================================================
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def extract_features(hex_str, traj_time_str):
    pts = parse_ewkb_4d(hex_str)
    times = eval(traj_time_str)

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    times = np.array(times)

    n = len(pts)
    duration = times[-1] - times[0] if n > 1 else 0.001

    # --- Segment distances ---
    if n > 1:
        dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1]) for i in range(n-1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists / dt
    else:
        dists = np.array([0.0])
        dt = np.array([1.0])
        speeds = np.array([0.0])

    total_dist = dists.sum()
    straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1]) if n > 1 else 0
    sinuosity = total_dist / max(straight_dist, 1e-6)

    # --- Altitude ---
    alt_diffs = np.diff(alts) if n > 1 else np.array([0.0])
    climbing = alt_diffs[alt_diffs > 0]
    descending = alt_diffs[alt_diffs < 0]

    # --- Bearings & turning ---
    if n > 2:
        dy = np.diff(lats)
        dx = np.diff(lons)
        bearings = np.arctan2(dy, dx)
        bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
    else:
        bearing_changes = np.array([0.0])

    # --- RCS FFT (wingbeat proxy) ---
    rcs_fft_feats = {}
    if n >= 8:
        # Interpolate RCS to uniform time grid for FFT
        from scipy.interpolate import interp1d
        from scipy.signal import welch
        uniform_t = np.linspace(times[0], times[-1], n)
        dt_uniform = uniform_t[1] - uniform_t[0] if n > 1 else 1.0
        fs = 1.0 / max(dt_uniform, 0.01)
        try:
            interp_fn = interp1d(times, rcs, kind='linear', fill_value='extrapolate')
            rcs_uniform = interp_fn(uniform_t)
            rcs_detrended = rcs_uniform - np.mean(rcs_uniform)
            freqs, psd = welch(rcs_detrended, fs=fs, nperseg=min(n, 32))
            peak_freq_idx = np.argmax(psd[1:]) + 1  # skip DC
            rcs_fft_feats['rcs_peak_freq'] = freqs[peak_freq_idx]
            rcs_fft_feats['rcs_peak_power'] = psd[peak_freq_idx]
            rcs_fft_feats['rcs_total_power'] = psd[1:].sum()
            rcs_fft_feats['rcs_spectral_centroid'] = np.sum(freqs[1:] * psd[1:]) / max(psd[1:].sum(), 1e-10)
        except:
            rcs_fft_feats = {'rcs_peak_freq': 0, 'rcs_peak_power': 0, 'rcs_total_power': 0, 'rcs_spectral_centroid': 0}
    else:
        rcs_fft_feats = {'rcs_peak_freq': 0, 'rcs_peak_power': 0, 'rcs_total_power': 0, 'rcs_spectral_centroid': 0}

    # --- Acceleration ---
    if len(speeds) > 1:
        accel = np.diff(speeds) / np.maximum(dt[:-1], 0.001) if len(dt) > 1 else np.array([0.0])
    else:
        accel = np.array([0.0])

    # --- First half vs second half (flight pattern change) ---
    mid = n // 2
    if mid > 1:
        first_half_dist = sum(dists[:mid])
        second_half_dist = sum(dists[mid:])
        dist_ratio = first_half_dist / max(second_half_dist, 1e-6)
        alt_first = np.mean(alts[:mid])
        alt_second = np.mean(alts[mid:])
        rcs_first = np.mean(rcs[:mid])
        rcs_second = np.mean(rcs[mid:])
    else:
        dist_ratio = 1.0
        alt_first = alt_second = np.mean(alts)
        rcs_first = rcs_second = np.mean(rcs)

    # --- Quarter splits (finer than halves) ---
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4
    if q1 > 0 and q3 < n:
        speed_q1 = np.mean(speeds[:q1]) if q1 > 0 else 0
        speed_q4 = np.mean(speeds[q3:]) if q3 < len(speeds) else 0
        alt_trend = np.polyfit(np.arange(n), alts, 1)[0] if n > 2 else 0
        rcs_trend = np.polyfit(np.arange(n), rcs, 1)[0] if n > 2 else 0
    else:
        speed_q1 = speed_q4 = 0
        alt_trend = rcs_trend = 0

    # --- RCS thresholds (Clutter has RCS >> birds) ---
    rcs_above_neg15 = np.mean(rcs > -15)  # clutter indicator
    rcs_above_neg20 = np.mean(rcs > -20)  # large bird / clutter
    rcs_below_neg28 = np.mean(rcs < -28)  # small bird indicator

    # --- Altitude variability patterns ---
    if n > 4:
        # rolling altitude std (flight stability)
        win = min(5, n)
        alt_rolling_std = np.std([np.std(alts[max(0,i-win):i+1]) for i in range(win, n)])
    else:
        alt_rolling_std = 0

    # --- Speed variability (erratic vs steady flight) ---
    speed_cv = np.std(speeds) / max(np.mean(speeds), 0.01)  # coefficient of variation

    # --- Percentile features ---
    feats = {
        'n_points': n,
        'duration': duration,
        'total_dist': total_dist,
        'straight_dist': straight_dist,
        'sinuosity': sinuosity,
        'sinuosity_clipped': min(sinuosity, 50),
        # altitude
        'alt_mean': np.mean(alts),
        'alt_std': np.std(alts),
        'alt_min': np.min(alts),
        'alt_max': np.max(alts),
        'alt_range': np.ptp(alts),
        'alt_median': np.median(alts),
        'alt_q25': np.percentile(alts, 25),
        'alt_q75': np.percentile(alts, 75),
        'alt_iqr': np.percentile(alts, 75) - np.percentile(alts, 25),
        'alt_diff_mean': np.mean(np.abs(alt_diffs)),
        'alt_diff_std': np.std(alt_diffs),
        'climb_rate': climbing.sum() / max(duration, 0.001) if len(climbing) > 0 else 0,
        'descent_rate': abs(descending.sum()) / max(duration, 0.001) if len(descending) > 0 else 0,
        'climb_frac': len(climbing) / max(len(alt_diffs), 1),
        'alt_trend': alt_trend,
        'alt_rolling_std': alt_rolling_std,
        # RCS
        'rcs_mean': np.mean(rcs),
        'rcs_std': np.std(rcs),
        'rcs_min': np.min(rcs),
        'rcs_max': np.max(rcs),
        'rcs_range': np.ptp(rcs),
        'rcs_median': np.median(rcs),
        'rcs_q25': np.percentile(rcs, 25),
        'rcs_q75': np.percentile(rcs, 75),
        'rcs_iqr': np.percentile(rcs, 75) - np.percentile(rcs, 25),
        'rcs_skew': float(pd.Series(rcs).skew()) if n > 2 else 0,
        'rcs_trend': rcs_trend,
        # RCS thresholds (clutter vs birds)
        'rcs_above_neg15': rcs_above_neg15,
        'rcs_above_neg20': rcs_above_neg20,
        'rcs_below_neg28': rcs_below_neg28,
        # speed
        'speed_mean': np.mean(speeds),
        'speed_std': np.std(speeds),
        'speed_max': np.max(speeds),
        'speed_min': np.min(speeds),
        'speed_median': np.median(speeds),
        'avg_ground_speed': total_dist / max(duration, 0.001),
        'speed_cv': speed_cv,
        'speed_q1': speed_q1,
        'speed_q4': speed_q4,
        'speed_change': speed_q4 - speed_q1,
        # acceleration
        'accel_mean': np.mean(accel),
        'accel_std': np.std(accel),
        'accel_max': np.max(np.abs(accel)),
        # turning
        'bearing_change_mean': np.mean(np.abs(bearing_changes)),
        'bearing_change_std': np.std(bearing_changes),
        'bearing_change_max': np.max(np.abs(bearing_changes)),
        'total_turning': np.sum(np.abs(bearing_changes)),
        'net_turning': np.abs(np.sum(bearing_changes)),
        'turning_per_meter': np.sum(np.abs(bearing_changes)) / max(total_dist, 1),
        # position
        'lon_mean': np.mean(lons),
        'lat_mean': np.mean(lats),
        'lon_std': np.std(lons),
        'lat_std': np.std(lats),
        'spatial_spread': np.std(lons) + np.std(lats),
        'lon_range': np.ptp(lons),
        'lat_range': np.ptp(lats),
        # first vs second half
        'dist_ratio_halves': dist_ratio,
        'alt_change_halves': alt_second - alt_first,
        'rcs_change_halves': rcs_second - rcs_first,
        # interactions
        'speed_x_alt': np.mean(speeds) * np.mean(alts),
        'rcs_x_alt': np.mean(rcs) * np.mean(alts),
        'dist_per_point': total_dist / max(n, 1),
        'duration_x_speed': duration * np.mean(speeds),
        'rcs_x_speed': np.mean(rcs) * np.mean(speeds),
        'alt_x_duration': np.mean(alts) * duration,
        # short track indicators (Pigeons + Clutter have short tracks)
        'is_short_track': int(n < 20),
        'is_very_short': int(duration < 15),
        'points_per_sec': n / max(duration, 0.001),
    }
    feats.update(rcs_fft_feats)
    return feats

print('Extracting train features...')
train_feats = pd.DataFrame([extract_features(r.trajectory, r.trajectory_time) for _, r in train.iterrows()])
print('Extracting test features...')
test_feats = pd.DataFrame([extract_features(r.trajectory, r.trajectory_time) for _, r in test.iterrows()])

# Add non-trajectory features
for df_feat, df_orig in [(train_feats, train), (test_feats, test)]:
    ts = pd.to_datetime(df_orig['timestamp_start_radar_utc'])
    te = pd.to_datetime(df_orig['timestamp_end_radar_utc'])
    df_feat['hour'] = ts.dt.hour.values
    df_feat['month'] = ts.dt.month.values
    df_feat['dayofweek'] = ts.dt.dayofweek.values
    df_feat['minute'] = ts.dt.minute.values
    df_feat['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour.values / 24).astype(np.float32)
    df_feat['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour.values / 24).astype(np.float32)
    df_feat['month_sin'] = np.sin(2 * np.pi * ts.dt.month.values / 12).astype(np.float32)
    df_feat['month_cos'] = np.cos(2 * np.pi * ts.dt.month.values / 12).astype(np.float32)
    df_feat['timestamp_duration'] = (te - ts).dt.total_seconds().values

    df_feat['airspeed'] = df_orig['airspeed'].values
    df_feat['min_z'] = df_orig['min_z'].values
    df_feat['max_z'] = df_orig['max_z'].values
    df_feat['z_range'] = df_orig['max_z'].values - df_orig['min_z'].values
    df_feat['z_mean'] = (df_orig['max_z'].values + df_orig['min_z'].values) / 2

    size_map = {'Small bird': 0, 'Medium': 1, 'Large': 2, 'Flock': 3}
    df_feat['radar_bird_size'] = df_orig['radar_bird_size'].map(size_map).values

    # airspeed vs computed ground speed ratio
    df_feat['airspeed_vs_ground'] = df_feat['airspeed'] / df_feat['avg_ground_speed'].clip(lower=0.01)

    # --- TARGETED FEATURES FOR WEAK CLASSES ---

    # Finer time bins (Pigeons peak at 14:00 — very distinctive)
    hour = ts.dt.hour.values
    df_feat['is_afternoon'] = (hour >= 13).astype(int)  # Pigeons
    df_feat['is_early_morning'] = (hour < 8).astype(int)  # Geese
    df_feat['hour_bin_3h'] = (hour // 3).astype(int)  # 3-hour bins
    df_feat['time_of_day'] = hour + ts.dt.minute.values / 60.0  # continuous

    # Month-based features (Clutter=April, Pigeons=Oct)
    month = ts.dt.month.values
    df_feat['is_april'] = (month == 4).astype(int)  # Clutter peak
    df_feat['is_october'] = (month == 10).astype(int)  # Pigeon/migration peak
    df_feat['is_migration'] = ((month >= 9) & (month <= 11)).astype(int)  # autumn migration
    df_feat['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)  # spring

    # Month x hour interaction (Pigeons = October + afternoon)
    df_feat['month_x_hour'] = month * 100 + hour  # unique combo
    df_feat['oct_afternoon'] = ((month == 10) & (hour >= 13)).astype(int)  # Pigeon signal

    # Size x speed interactions (Clutter = Large + fast)
    df_feat['size_x_airspeed'] = df_feat['radar_bird_size'] * df_feat['airspeed']
    df_feat['size_x_rcs'] = df_feat['radar_bird_size'] * df_feat['rcs_mean']
    df_feat['size_x_alt'] = df_feat['radar_bird_size'] * df_feat['alt_mean']

    # Airspeed bins (helps separate Pigeons/Ducks/Clutter high-speed group from Gulls/BoP)
    df_feat['airspeed_high'] = (df_feat['airspeed'] > 17).astype(int)
    df_feat['airspeed_low'] = (df_feat['airspeed'] < 12).astype(int)

    # One-hot radar bird size (trees can split better on these)
    for sz_name, sz_val in [('Small bird', 0), ('Medium', 1), ('Large', 2), ('Flock', 3)]:
        df_feat[f'is_{sz_name.replace(" ", "_").lower()}'] = (df_feat['radar_bird_size'] == sz_val).astype(int)

    # Duration bins (short = Clutter/Pigeons)
    df_feat['duration_short'] = (df_feat['duration'] < 25).astype(int)
    df_feat['duration_long'] = (df_feat['duration'] > 60).astype(int)

# Handle any inf/nan
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f'\nFinal feature count: {train_feats.shape[1]}')

# ============================================================
# 5. PREPARE TARGET
# ============================================================
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train['bird_group'])
n_classes = len(CLASSES)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# Class weights (inverse frequency)
class_counts = np.bincount(y, minlength=n_classes)
class_weights = len(y) / (n_classes * class_counts)
sample_weights_train = np.array([class_weights[yi] for yi in y])
print(f'\nClass weights: {dict(zip(CLASSES, class_weights.round(2)))}')

# ============================================================
# 6. TRAIN 3-MODEL ENSEMBLE
# ============================================================
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), n_classes))
oof_xgb = np.zeros((len(X), n_classes))
oof_cb = np.zeros((len(X), n_classes))
test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_cb = np.zeros((len(X_test), n_classes))

lgb_params = {
    'objective': 'multiclass',
    'num_class': n_classes,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 47,
    'max_depth': 7,
    'min_child_samples': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'is_unbalance': True,
}

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'seed': 42,
    'nthread': -1,
    'verbosity': 0,
}

def compute_map(y_true, y_pred, n_classes, classes):
    y_onehot = np.eye(n_classes)[y_true]
    aps = []
    for c in range(n_classes):
        if y_onehot[:, c].sum() > 0:
            aps.append(average_precision_score(y_onehot[:, c], y_pred[:, c]))
    return np.mean(aps), aps

print('\n' + '='*60)
print('TRAINING LIGHTGBM')
print('='*60)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights_train[tr_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)

    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])

    oof_lgb[va_idx] = model.predict(X_va)
    test_lgb += model.predict(X_test) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_lgb[va_idx], n_classes, CLASSES)
    print(f'  Fold {fold}: mAP = {fold_map:.4f}')

lgb_map, lgb_aps = compute_map(y, oof_lgb, n_classes, CLASSES)
print(f'LightGBM CV mAP: {lgb_map:.4f}')

print('\n' + '='*60)
print('TRAINING XGBOOST')
print('='*60)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights_train[tr_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)

    model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                      evals=[(dval, 'val')], early_stopping_rounds=80, verbose_eval=200)

    oof_xgb[va_idx] = model.predict(dval)
    test_xgb += model.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_xgb[va_idx], n_classes, CLASSES)
    print(f'  Fold {fold}: mAP = {fold_map:.4f}')

xgb_map, xgb_aps = compute_map(y, oof_xgb, n_classes, CLASSES)
print(f'XGBoost CV mAP: {xgb_map:.4f}')

print('\n' + '='*60)
print('TRAINING CATBOOST')
print('='*60)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights_train[tr_idx]

    cb_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='MultiClass', eval_metric='MultiClass',
        random_seed=42, verbose=0, early_stopping_rounds=80,
        auto_class_weights='Balanced',
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

    oof_cb[va_idx] = cb_model.predict_proba(X_va)
    test_cb += cb_model.predict_proba(X_test) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_cb[va_idx], n_classes, CLASSES)
    print(f'  Fold {fold}: mAP = {fold_map:.4f}')

cb_map, cb_aps = compute_map(y, oof_cb, n_classes, CLASSES)
print(f'CatBoost CV mAP: {cb_map:.4f}')

# ============================================================
# 7. ENSEMBLE
# ============================================================
print('\n' + '='*60)
print('ENSEMBLE RESULTS')
print('='*60)

# Try different ensemble weights
best_map = 0
best_w = None
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        ens_map, _ = compute_map(y, oof_ens, n_classes, CLASSES)
        if ens_map > best_map:
            best_map = ens_map
            best_w = (w1, w2, w3)

print(f'Best ensemble weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}')

# Final ensemble
oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

final_map, final_aps = compute_map(y, oof_final, n_classes, CLASSES)

print(f'\n{"="*40}')
print(f'  MODEL COMPARISON (CV mAP)')
print(f'{"="*40}')
print(f'  Baseline (v1):          0.7030')
print(f'  Ensemble (v2):          0.7214')
print(f'  LightGBM (v3):          {lgb_map:.4f}')
print(f'  XGBoost (v3):            {xgb_map:.4f}')
print(f'  CatBoost (v3):           {cb_map:.4f}')
print(f'  Ensemble (v3):           {final_map:.4f}')
print(f'{"="*40}')

print(f'\nPer-class AP (Ensemble):')
for c in range(n_classes):
    marker = ' <-- weak' if final_aps[c] < 0.6 else ''
    print(f'  {CLASSES[c]:15s}: {final_aps[c]:.4f}{marker}')

# ============================================================
# 8. SUBMISSION
# ============================================================
submission = pd.DataFrame({'track_id': test['track_id']})
for i, cls in enumerate(CLASSES):
    submission[cls] = test_final[:, i]

submission.to_csv('submission.csv', index=False)
print(f'\nSaved submission.csv ({len(submission)} rows)')
print(submission.head())
