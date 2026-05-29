"""E180: Spatial features — WHERE each bird flies relative to radar and coastline.

Hypothesis: Different species occupy different spatial regions around Eemshaven.
  - Gulls: over sea (north of radar)
  - Waders: along coastline
  - BoP: over land (south of radar)
  - Cormorants: linear coastal routes

These spatial features are month-invariant (geography doesn't change with season).

New features (8):
  1. mean_bearing      — bearing (deg) from radar to track centroid
  2. mean_distance_km  — distance from radar to track centroid
  3. bearing_std       — spread of per-point bearings from radar (circling vs linear)
  4. north_fraction    — fraction of track points north of radar (over sea)
  5. track_bearing     — already in v3, reused
  6. dist_to_coast     — simple lat proxy (lat - 53.44, positive = sea)
  7. over_water        — already in v3, reused
  8. radial_velocity   — speed component toward/away from radar

Pipeline: Load cached v3 features (100 selected) + compute 6 new spatial features
          from raw trajectories + train LGB GBDT (5 seeds, SGKF, LOMO eval).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
N_SEEDS = 5

# Radar location: Eemshaven, Netherlands
RADAR_LON = 6.835
RADAR_LAT = 53.438
# Approximate coastline latitude (north = Wadden Sea, south = land)
COAST_LAT = 53.44


# ══════════════════════════════════════════════════════════════════════
# Spatial Feature Extraction
# ══════════════════════════════════════════════════════════════════════

def _haversine_np(lon1, lat1, lon2, lat2):
    """Haversine distance in meters (vectorized for scalars or arrays)."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lon1, lat1, lon2, lat2):
    """Bearing from (lon1, lat1) to (lon2, lat2) in degrees [0, 360)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return bearing % 360


def extract_spatial_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract 6 new spatial features from a single track.

    All features are month-invariant: they depend on geography, not season.
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    n = len(pts)

    # ── Centroid ──
    lon_c = np.mean(lons)
    lat_c = np.mean(lats)

    # 1. mean_bearing: bearing from radar to track centroid
    mean_bearing = _bearing_deg(RADAR_LON, RADAR_LAT, lon_c, lat_c)

    # 2. mean_distance_km: distance from radar to track centroid
    mean_distance_km = _haversine_np(RADAR_LON, RADAR_LAT, lon_c, lat_c) / 1000.0

    # 3. bearing_std: circular std of per-point bearings from radar
    if n >= 2:
        point_bearings_rad = np.radians(
            np.array([_bearing_deg(RADAR_LON, RADAR_LAT, lons[i], lats[i]) for i in range(n)])
        )
        # Circular standard deviation: sqrt(-2 * ln(R)), R = mean resultant length
        sin_sum = np.sum(np.sin(point_bearings_rad))
        cos_sum = np.sum(np.cos(point_bearings_rad))
        R_len = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / n
        R_len = np.clip(R_len, 1e-10, 1.0)
        bearing_std = np.degrees(np.sqrt(-2.0 * np.log(R_len)))
    else:
        bearing_std = 0.0

    # 4. north_fraction: fraction of track points north of radar
    north_fraction = float(np.mean(lats > RADAR_LAT))

    # 5. dist_to_coast: simple latitude proxy (positive = over sea, negative = over land)
    dist_to_coast = (lat_c - COAST_LAT) * 111.0  # rough km conversion

    # 6. radial_velocity: speed component toward/away from radar
    #    Positive = moving away from radar, negative = approaching
    if n >= 2:
        dists_to_radar = np.array([
            _haversine_np(RADAR_LON, RADAR_LAT, lons[i], lats[i])
            for i in range(n)
        ])
        dt = np.diff(times)
        dt = np.maximum(dt, 0.5)  # avoid division by zero / glitches
        radial_v = np.diff(dists_to_radar) / dt  # m/s, positive = receding
        radial_velocity = float(np.median(radial_v))
    else:
        radial_velocity = 0.0

    return {
        "sp_mean_bearing": mean_bearing,
        "sp_mean_distance_km": mean_distance_km,
        "sp_bearing_std": bearing_std,
        "sp_north_fraction": north_fraction,
        "sp_dist_to_coast_km": dist_to_coast,
        "sp_radial_velocity": radial_velocity,
    }


def compute_spatial_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial features for all rows in a DataFrame."""
    rows = []
    n = len(df)
    for idx, (_, r) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    Spatial features: {idx}/{n}", flush=True)
        rows.append(extract_spatial_features(r.trajectory, r.trajectory_time))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Load cached v3 features (100 selected) + compute new spatial features."""
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    # Load cached v3 features
    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    # Load stability-selected features (100 from E175)
    selected = [
        l.strip()
        for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines()
        if l.strip()
    ]
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
    print(f"  Base features: {len(selected)} (from E175 selection)")

    # Compute NEW spatial features from raw trajectories
    print("  Computing spatial features for train...", flush=True)
    t0 = time.time()
    sp_train = compute_spatial_features_batch(train_df)
    print(f"    Train spatial done in {time.time()-t0:.1f}s", flush=True)

    print("  Computing spatial features for test...", flush=True)
    t0 = time.time()
    sp_test = compute_spatial_features_batch(test_df)
    print(f"    Test spatial done in {time.time()-t0:.1f}s", flush=True)

    # Combine: selected v3 features + new spatial features
    spatial_cols = list(sp_train.columns)
    print(f"  New spatial features: {spatial_cols}")

    X_train_base = train_feats[selected].values.astype(np.float32)
    X_test_base = test_feats[selected].values.astype(np.float32)
    X_train_sp = sp_train.values.astype(np.float32)
    X_test_sp = sp_test.values.astype(np.float32)

    X_train = np.hstack([X_train_base, X_train_sp])
    X_test = np.hstack([X_test_base, X_test_sp])

    all_cols = selected + spatial_cols

    # Handle inf/nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Final shape: train={X_train.shape}, test={X_test.shape}")
    return (train_df, test_df, y, train_months, test_months, groups,
            X_train, X_test, all_cols)


# ══════════════════════════════════════════════════════════════════════
# Training: LGB GBDT (fast, 5 seeds)
# ══════════════════════════════════════════════════════════════════════

def train_lgb_gbdt(X_train, y, train_months, X_test, groups, feature_names, n_seeds=N_SEEDS):
    """Train LightGBM GBDT with StratifiedGroupKFold, multi-seed averaging.

    Returns OOF predictions, test predictions, and per-seed LOMO scores.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    params = {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    importances = np.zeros(X_train.shape[1])

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)

        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params, random_state=42 + seed + fold)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            oof_seed[val_idx] = model.predict_proba(X_va)
            test_seed += model.predict_proba(X_test) / N_FOLDS
            importances += model.feature_importances_

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed

        oof_map, _ = compute_map(y, oof_seed)
        elapsed = time.time() - t_seed
        print(f"    Seed {seed+1}/{n_seeds}: SKF mAP={oof_map:.4f} ({elapsed:.1f}s)", flush=True)

    # Average across seeds
    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)

    # Normalize importances
    importances = importances / (n_seeds * N_FOLDS)

    return oof_mean, test_mean, importances


# ══════════════════════════════════════════════════════════════════════
# LOMO Evaluation
# ══════════════════════════════════════════════════════════════════════

def eval_lomo(y, oof, train_months):
    """Evaluate with true Leave-One-Month-Out (per-month held-out mAP)."""
    lomo_maps = {}
    per_class_per_month = {}
    for held in sorted(set(train_months)):
        mask = train_months == held
        if mask.sum() >= 10:
            lm, pc = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
            per_class_per_month[held] = pc
    lomo_avg = float(np.mean(list(lomo_maps.values())))
    return lomo_avg, lomo_maps, per_class_per_month


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 70)
    print("  E180: SPATIAL FEATURES")
    print("  Where each bird flies relative to radar & coastline")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/4] Loading data + computing spatial features...", flush=True)
    (train_df, test_df, y, train_months, test_months, groups,
     X_train, X_test, feature_cols) = load_data()

    # ── Train LGB GBDT ──
    print(f"\n[2/4] Training LGB GBDT ({N_SEEDS} seeds × {N_FOLDS} folds)...", flush=True)
    oof, test_preds, importances = train_lgb_gbdt(
        X_train, y, train_months, X_test, groups, feature_cols, N_SEEDS,
    )

    # ── SKF evaluation ──
    print("\n[3/4] Evaluation...", flush=True)
    skf_map, skf_pc = compute_map(y, oof)
    print_results(skf_map, skf_pc, "E180 Spatial — SKF (5-seed avg)")

    # ── LOMO evaluation ──
    lomo_avg, lomo_maps, lomo_pc = eval_lomo(y, oof, train_months)
    print(f"\n{'='*50}")
    print(f"  E180 Spatial — LOMO")
    print(f"{'='*50}")
    print(f"\n  Overall LOMO mAP: {lomo_avg:.4f}")
    month_str = "  ".join(f"M{m}={v:.4f}" for m, v in sorted(lomo_maps.items()))
    print(f"  Per-month: {month_str}")

    # Per-class LOMO for each month
    print(f"\n  Per-class LOMO breakdown:")
    for m in sorted(lomo_pc.keys()):
        pc = lomo_pc[m]
        worst = min(pc.values(), key=lambda x: x)
        weak = [f"{c[:4]}={v:.3f}" for c, v in pc.items() if v < 0.5]
        print(f"    M{m}: mAP={lomo_maps[m]:.4f}  weak=[{', '.join(weak)}]")

    # ── Feature importance for NEW spatial features ──
    print(f"\n  Feature importance (new spatial features):")
    sp_start = len(feature_cols) - 6  # last 6 are spatial
    for i in range(sp_start, len(feature_cols)):
        print(f"    {feature_cols[i]:25s}: {importances[i]:8.1f}")

    # Top 15 features overall
    print(f"\n  Top 15 features by importance:")
    idx_sorted = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx_sorted[:15]):
        marker = " **NEW**" if i >= sp_start else ""
        print(f"    {rank+1:2d}. {feature_cols[i]:25s}: {importances[i]:8.1f}{marker}")

    # ── Save outputs ──
    print(f"\n[4/4] Saving outputs...", flush=True)
    np.save(ROOT / "oof_e180_spatial.npy", oof)
    np.save(ROOT / "test_e180_spatial.npy", test_preds)
    print(f"  Saved: oof_e180_spatial.npy, test_e180_spatial.npy")

    save_submission(test_preds, "e180_spatial_raw", cv_map=skf_map)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  E180 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Features:   {len(feature_cols)} ({len(feature_cols)-6} base + 6 spatial)")
    print(f"  SKF mAP:    {skf_map:.4f}")
    print(f"  LOMO mAP:   {lomo_avg:.4f}")
    print(f"  Per-month:  {month_str}")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
