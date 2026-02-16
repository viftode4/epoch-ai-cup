"""Cormorant detection: DTW kNN + MiniRocket binary on LOMO.

Tests raw time series approaches for Cormorant-vs-rest classification.
- DTW kNN: RCS-only and multi-channel (RCS + alt + speed)
- MiniRocket binary: 3-channel fixed-length time series

Evaluated on LOMO (4 folds: leave out months 1, 4, 9, 10).
Baseline: tabular top-50 features ensemble AP = 0.1568
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")  # = 2

# ======================================================================
# Load data and extract raw time series
# ======================================================================
print("=" * 60, flush=True)
print("CORMORANT DETECTION: DTW + MiniRocket BINARY", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

# Extract months for LOMO
ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months = ts.dt.month.values
unique_months = sorted(np.unique(months))
print(f"  Months: {unique_months}", flush=True)
print(f"  Cormorants: {y_bin.sum()}/{ len(y_bin)}", flush=True)

# Parse raw trajectories
print("\nExtracting raw time series...", flush=True)
rcs_series = []
alt_series = []
speed_series = []
time_series = []

for i, row in train_df.iterrows():
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])

    rcs = np.array([p[3] for p in pts])
    alt = np.array([p[2] for p in pts])

    # Compute instantaneous speed from lat/lon
    if len(pts) > 1 and len(times) > 1:
        dt = np.diff(times)
        dt[dt == 0] = 1e-6
        dlat = np.diff([p[1] for p in pts])
        dlon = np.diff([p[0] for p in pts])
        dalt = np.diff(alt)
        # Approximate distance in meters (at ~53N, 1 deg lat ~ 111km, 1 deg lon ~ 67km)
        dx = dlon * 67000
        dy = dlat * 111000
        dist = np.sqrt(dx**2 + dy**2 + dalt**2)
        spd = dist / dt
        spd = np.concatenate([[spd[0]], spd])  # repeat first to match length
    else:
        spd = np.array([row["airspeed"]] * len(pts))

    rcs_series.append(rcs)
    alt_series.append(alt)
    speed_series.append(spd)
    time_series.append(times)

print(f"  Extracted {len(rcs_series)} time series", flush=True)
print(f"  Length range: [{min(len(s) for s in rcs_series)}, {max(len(s) for s in rcs_series)}]", flush=True)

# ======================================================================
# Resample to fixed length for MiniRocket
# ======================================================================
FIXED_LEN = 64

def resample_to_fixed(series, target_len=FIXED_LEN):
    """Resample a 1D series to fixed length using linear interpolation."""
    n = len(series)
    if n == target_len:
        return series.copy()
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, series)

print(f"\nResampling to fixed length {FIXED_LEN}...", flush=True)
X_rcs = np.array([resample_to_fixed(s) for s in rcs_series])
X_alt = np.array([resample_to_fixed(s) for s in alt_series])
X_spd = np.array([resample_to_fixed(s) for s in speed_series])

# Normalize each channel per-sample (zero-mean, unit-var)
def normalize_per_sample(X):
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mu) / std

X_rcs_norm = normalize_per_sample(X_rcs)
X_alt_norm = normalize_per_sample(X_alt)
X_spd_norm = normalize_per_sample(X_spd)

# Multi-channel: (n_samples, n_channels, seq_len) for MiniRocket
X_multi = np.stack([X_rcs_norm, X_alt_norm, X_spd_norm], axis=1)
print(f"  X_multi shape: {X_multi.shape}", flush=True)

# ======================================================================
# LOMO evaluation helper
# ======================================================================
def lomo_evaluate(predict_fn, name):
    """Run LOMO and return Cormorant AP."""
    oof_scores = np.full(len(y_bin), np.nan)

    for held_month in unique_months:
        val_mask = months == held_month
        train_mask = ~val_mask

        n_corm_train = y_bin[train_mask].sum()
        n_corm_val = y_bin[val_mask].sum()

        if n_corm_val == 0:
            continue

        scores = predict_fn(train_mask, val_mask)
        oof_scores[val_mask] = scores

        ap = average_precision_score(y_bin[val_mask], scores)
        print(f"    M{held_month:2d}: train={train_mask.sum()} "
              f"(corm={n_corm_train}), val={val_mask.sum()} "
              f"(corm={n_corm_val}), AP={ap:.4f}", flush=True)

    valid = ~np.isnan(oof_scores)
    if valid.sum() > 0 and y_bin[valid].sum() > 0:
        overall_ap = average_precision_score(y_bin[valid], oof_scores[valid])
    else:
        overall_ap = 0.0
    print(f"  {name} LOMO AP: {overall_ap:.4f}", flush=True)
    return overall_ap, oof_scores

# ======================================================================
# 1. DTW kNN (RCS only)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("1. DTW kNN (RCS only)", flush=True)
print("=" * 60, flush=True)

try:
    from dtaidistance import dtw

    def dtw_knn_rcs(train_mask, val_mask, k=5):
        """kNN with DTW distance on RCS series."""
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        corm_train = train_idx[y_bin[train_idx] == 1]

        scores = np.zeros(val_mask.sum())
        for j, vi in enumerate(val_idx):
            # Distance to all cormorant training samples
            dists = []
            for ci in corm_train:
                d = dtw.distance_fast(
                    rcs_series[vi].astype(np.float64),
                    rcs_series[ci].astype(np.float64)
                )
                dists.append(d)

            if len(dists) == 0:
                scores[j] = 0.0
                continue

            dists = np.array(dists)
            # Score = inverse of mean distance to k nearest cormorants
            k_actual = min(k, len(dists))
            top_k = np.sort(dists)[:k_actual]
            scores[j] = 1.0 / (1.0 + np.mean(top_k))

        return scores

    for k in [3, 5, 10]:
        print(f"\n  k={k}:", flush=True)
        ap, _ = lomo_evaluate(
            lambda tr, va, _k=k: dtw_knn_rcs(tr, va, _k),
            f"DTW-kNN-RCS(k={k})"
        )

except ImportError:
    print("  dtaidistance not available, skipping DTW", flush=True)

# ======================================================================
# 2. DTW kNN (multi-channel: RCS + alt + speed)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("2. DTW kNN (multi-channel)", flush=True)
print("=" * 60, flush=True)

try:
    from dtaidistance import dtw_ndim

    def dtw_knn_multi(train_mask, val_mask, k=5):
        """kNN with multi-dim DTW on (RCS, alt, speed)."""
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        corm_train = train_idx[y_bin[train_idx] == 1]

        scores = np.zeros(val_mask.sum())
        for j, vi in enumerate(val_idx):
            # Build multi-dim arrays: (seq_len, n_dims)
            s_val = np.column_stack([
                rcs_series[vi], alt_series[vi], speed_series[vi]
            ]).astype(np.float64)

            dists = []
            for ci in corm_train:
                s_train = np.column_stack([
                    rcs_series[ci], alt_series[ci], speed_series[ci]
                ]).astype(np.float64)
                d = dtw_ndim.distance_fast(s_val, s_train)
                dists.append(d)

            if len(dists) == 0:
                scores[j] = 0.0
                continue

            dists = np.array(dists)
            k_actual = min(k, len(dists))
            top_k = np.sort(dists)[:k_actual]
            scores[j] = 1.0 / (1.0 + np.mean(top_k))

        return scores

    for k in [3, 5]:
        print(f"\n  k={k}:", flush=True)
        ap, _ = lomo_evaluate(
            lambda tr, va, _k=k: dtw_knn_multi(tr, va, _k),
            f"DTW-kNN-Multi(k={k})"
        )

except ImportError:
    print("  dtaidistance ndim not available, skipping", flush=True)

# ======================================================================
# 3. MiniRocket binary (3-channel)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("3. MiniRocket Binary (3-channel)", flush=True)
print("=" * 60, flush=True)

try:
    from aeon.transformations.collection.convolution_based import MiniRocket

    def minirocket_logreg(train_mask, val_mask):
        """MiniRocket + LogisticRegression binary."""
        X_tr = X_multi[train_mask]
        X_va = X_multi[val_mask]
        y_tr = y_bin[train_mask]

        mr = MiniRocket(n_kernels=10000, random_state=42)
        mr.fit(X_tr)
        Z_tr = mr.transform(X_tr)
        Z_va = mr.transform(X_va)

        # Scale
        sc = StandardScaler()
        Z_tr = sc.fit_transform(Z_tr)
        Z_va = sc.transform(Z_va)

        # Logistic regression with balanced weights
        lr = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0],
            class_weight="balanced",
            max_iter=5000,
            cv=3,
            random_state=42,
            scoring="average_precision"
        )
        lr.fit(Z_tr, y_tr)
        proba = lr.predict_proba(Z_va)[:, 1]
        return proba

    print("\n  MiniRocket + LogReg:", flush=True)
    ap_mr_lr, oof_mr_lr = lomo_evaluate(minirocket_logreg, "MiniRocket-LogReg")

    def minirocket_ridge(train_mask, val_mask):
        """MiniRocket + Ridge binary (convert decision to score)."""
        X_tr = X_multi[train_mask]
        X_va = X_multi[val_mask]
        y_tr = y_bin[train_mask]

        mr = MiniRocket(n_kernels=10000, random_state=42)
        mr.fit(X_tr)
        Z_tr = mr.transform(X_tr)
        Z_va = mr.transform(X_va)

        sc = StandardScaler()
        Z_tr = sc.fit_transform(Z_tr)
        Z_va = sc.transform(Z_va)

        # Ridge with balanced weights
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y_tr)

        ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(Z_tr, y_tr, sample_weight=sw)
        # Decision function as score
        scores = ridge.decision_function(Z_va)
        return scores

    print("\n  MiniRocket + Ridge:", flush=True)
    ap_mr_ridge, oof_mr_ridge = lomo_evaluate(minirocket_ridge, "MiniRocket-Ridge")

except ImportError as e:
    print(f"  aeon/MiniRocket not available: {e}", flush=True)

# ======================================================================
# 4. Euclidean kNN on resampled fixed-length (fast baseline)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("4. Euclidean kNN on fixed-length (fast baseline)", flush=True)
print("=" * 60, flush=True)

def euclidean_knn(train_mask, val_mask, k=5):
    """kNN with Euclidean distance on fixed-length multi-channel."""
    X_flat_tr = X_multi[train_mask].reshape(train_mask.sum(), -1)
    X_flat_va = X_multi[val_mask].reshape(val_mask.sum(), -1)
    y_tr = y_bin[train_mask]

    sc = StandardScaler()
    X_flat_tr = sc.fit_transform(X_flat_tr)
    X_flat_va = sc.transform(X_flat_va)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    knn.fit(X_flat_tr, y_tr)
    proba = knn.predict_proba(X_flat_va)
    # Make sure class 1 column exists
    if proba.shape[1] == 2:
        return proba[:, 1]
    else:
        return proba[:, 0] if knn.classes_[0] == 1 else np.zeros(val_mask.sum())

for k in [3, 5, 10, 20]:
    print(f"\n  k={k}:", flush=True)
    ap, _ = lomo_evaluate(
        lambda tr, va, _k=k: euclidean_knn(tr, va, _k),
        f"Euclidean-kNN(k={k})"
    )

# ======================================================================
# 5. RCS autocorrelation features + simple classifier
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("5. RCS autocorrelation + LogReg", flush=True)
print("=" * 60, flush=True)

# Build autocorrelation features (lags 1-20)
MAX_LAG = 20
X_acf = np.zeros((len(rcs_series), MAX_LAG))
for i, rcs in enumerate(rcs_series):
    if len(rcs) < MAX_LAG + 1:
        n = len(rcs)
    else:
        n = len(rcs)
    rcs_centered = rcs - rcs.mean()
    var = np.var(rcs)
    if var < 1e-10:
        continue
    for lag in range(1, min(MAX_LAG + 1, n)):
        X_acf[i, lag - 1] = np.corrcoef(rcs_centered[:-lag], rcs_centered[lag:])[0, 1]

# Also add track length and mean RCS
X_acf_plus = np.column_stack([
    X_acf,
    np.array([len(s) for s in rcs_series]),  # track length
    np.array([s.mean() for s in rcs_series]),  # mean RCS
    np.array([s.std() for s in rcs_series]),  # RCS std
])

def acf_logreg(train_mask, val_mask):
    """Autocorrelation features + LogReg."""
    X_tr = X_acf_plus[train_mask]
    X_va = X_acf_plus[val_mask]
    y_tr = y_bin[train_mask]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)

    lr = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        class_weight="balanced",
        max_iter=5000,
        cv=3,
        random_state=42,
    )
    lr.fit(X_tr, y_tr)
    return lr.predict_proba(X_va)[:, 1]

print("\n  ACF + LogReg:", flush=True)
ap_acf, _ = lomo_evaluate(acf_logreg, "ACF-LogReg")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY: Cormorant LOMO AP", flush=True)
print("=" * 60, flush=True)
print(f"  Tabular top-50 ensemble baseline:  0.1568", flush=True)
print(f"  (Results above show per-method LOMO APs)", flush=True)
print("\nDone!", flush=True)
