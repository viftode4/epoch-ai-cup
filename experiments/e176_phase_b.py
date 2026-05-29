"""E176 Phase B: New Features + Retrain with LOMO evaluation.

New features added directly to feature matrix (no cache rebuild needed):
  B1. Flock intensity (temporal+spatial neighbor count)
  B2. Glide ratio (horizontal_dist / altitude_loss)
  B3. Session-relative time
  B4. Wing loading proxy from RCS
  B5. (skip rotation-invariant sigs — needs iisignature, low priority)
  B6. Observer comment confidence weights

Evaluates with BOTH SKF and LOMO. Only LOMO improvements matter.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5

print("=" * 70)
print("  E176 Phase B: New Features + Retrain")
print("=" * 70)

t0 = time.time()

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load existing features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

counts = np.bincount(y, minlength=N_CLASSES).astype(float)


def eval_skf_lomo(oof, name=""):
    """Evaluate both SKF and LOMO."""
    skf, skf_pc = compute_map(y, oof)
    lomo_scores = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            lomo_scores[m] = s
    lomo = np.mean(list(lomo_scores.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_scores.items()))
    print(f"  {name}: SKF={skf:.4f} LOMO={lomo:.4f} gap={skf-lomo:.4f}")
    print(f"    months: {month_str}")
    return skf, lomo


# ══════════════════════════════════════════════════════════════════════
# B1. Flock Intensity Features
# ══════════════════════════════════════════════════════════════════════

print("\n--- B1. Flock Intensity Features ---")


def compute_flock_intensity(df, time_window=60, dist_thresh_m=500):
    """Count temporal+spatial neighbors per track (proxy for n_birds_observed)."""
    n = len(df)
    timestamps = pd.to_datetime(df["timestamp_start_radar_utc"])
    ts_unix = timestamps.values.astype(np.int64) / 1e9
    sizes = df["radar_bird_size"].fillna("__UNK__").values

    # Extract centroids
    lons = np.zeros(n)
    lats = np.zeros(n)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if pts:
                lons[i] = np.mean([p[0] for p in pts])
                lats[i] = np.mean([p[1] for p in pts])
        except Exception:
            pass

    # For each track, count neighbors within time+distance window
    flock_count = np.zeros(n)
    flock_small = np.zeros(n)
    flock_medium = np.zeros(n)
    flock_large = np.zeros(n)
    flock_flock = np.zeros(n)

    # Sort by timestamp for efficient window search
    order = np.argsort(ts_unix)
    ts_sorted = ts_unix[order]

    for ii in range(n):
        i = order[ii]
        # Find time window neighbors
        t_lo = ts_unix[i] - time_window
        t_hi = ts_unix[i] + time_window
        lo_idx = np.searchsorted(ts_sorted, t_lo)
        hi_idx = np.searchsorted(ts_sorted, t_hi, side='right')

        neighbors_in_time = order[lo_idx:hi_idx]
        count = 0
        for j in neighbors_in_time:
            if j == i:
                continue
            dlat = (lats[i] - lats[j]) * 111000
            dlon = (lons[i] - lons[j]) * 67000
            dist = np.sqrt(dlat**2 + dlon**2)
            if dist <= dist_thresh_m:
                count += 1
                s = sizes[j]
                if s == "Small bird":
                    flock_small[i] += 1
                elif s == "Medium bird":
                    flock_medium[i] += 1
                elif s == "Large bird":
                    flock_large[i] += 1
                elif s == "Flock":
                    flock_flock[i] += 1

        flock_count[i] = count
        if ii % 500 == 0:
            print(f"    Progress: {ii}/{n}", flush=True)

    return {
        "flock_count": flock_count,
        "flock_small_neighbors": flock_small,
        "flock_medium_neighbors": flock_medium,
        "flock_large_neighbors": flock_large,
        "flock_flock_neighbors": flock_flock,
    }


print("  Computing flock intensity for train...")
flock_train = compute_flock_intensity(train_df)
print("  Computing flock intensity for test...")
flock_test = compute_flock_intensity(test_df)


# ══════════════════════════════════════════════════════════════════════
# B2. Glide Ratio Feature
# ══════════════════════════════════════════════════════════════════════

print("\n--- B2. Glide Ratio Feature ---")


def compute_glide_ratio(df):
    """Compute L/D = horizontal_distance / altitude_loss during descending segments.

    BoP 10-20, Gulls 8-12, Songbirds 4-8.
    """
    n = len(df)
    glide_ratio = np.full(n, np.nan)
    glide_frac = np.zeros(n)  # fraction of track that's gliding

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"    Progress: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 3:
                continue
            lons = np.array([p[0] for p in pts])
            lats = np.array([p[1] for p in pts])
            alts = np.array([p[2] for p in pts])

            # Find descending segments (altitude decreasing)
            dalt = np.diff(alts)
            descending = dalt < -0.5  # at least 0.5m drop

            if descending.sum() < 2:
                continue

            # Horizontal distance during descending segments
            dx = np.diff(lons) * 67000  # approx meters
            dy = np.diff(lats) * 111000
            h_dist = np.sqrt(dx**2 + dy**2)

            desc_h = h_dist[descending].sum()
            desc_v = np.abs(dalt[descending]).sum()

            if desc_v > 1.0:  # at least 1m total descent
                glide_ratio[i] = desc_h / desc_v
            glide_frac[i] = descending.sum() / len(dalt)
        except Exception:
            continue

    glide_ratio = np.where(np.isfinite(glide_ratio), glide_ratio, 0.0)
    return {"glide_ratio": glide_ratio, "glide_frac": glide_frac}


print("  Computing glide ratio for train...")
glide_train = compute_glide_ratio(train_df)
print("  Computing glide ratio for test...")
glide_test = compute_glide_ratio(test_df)


# ══════════════════════════════════════════════════════════════════════
# B3. Session-Relative Time
# ══════════════════════════════════════════════════════════════════════

print("\n--- B3. Session-Relative Time ---")


def compute_session_time(df):
    """Hours since start of observation session (month-invariant)."""
    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    dates = ts.dt.date
    hours_in_day = ts.dt.hour + ts.dt.minute / 60.0

    # Group by date, compute hours since first track that day
    session_time = np.zeros(len(df))
    for date in dates.unique():
        mask = dates == date
        day_ts = ts[mask]
        first = day_ts.min()
        session_time[mask.values] = (day_ts - first).dt.total_seconds().values / 3600.0

    return {"session_relative_hours": session_time, "hour_of_day": hours_in_day.values}


sess_train = compute_session_time(train_df)
sess_test = compute_session_time(test_df)


# ══════════════════════════════════════════════════════════════════════
# B4. Wing Loading Proxy from RCS
# ══════════════════════════════════════════════════════════════════════

print("\n--- B4. Wing Loading Proxy ---")


def compute_wing_loading(df):
    """mass_proxy = (10^(rcs_dB/10))^1.5, wing_loading = mass^0.28 * 25.3"""
    n = len(df)
    mass_proxy = np.zeros(n)
    wing_loading = np.zeros(n)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 2:
                continue
            rcs_dB = np.array([p[3] for p in pts])
            mean_rcs_dB = np.mean(rcs_dB)
            # Convert dB to linear RCS (m²)
            rcs_linear = 10 ** (mean_rcs_dB / 10.0)
            # Allometric scaling: mass ∝ RCS^1.5 (Eastwood 1967)
            mass = max(rcs_linear ** 1.5, 1e-6)
            mass_proxy[i] = mass
            # Wing loading: W/S ∝ mass^0.28 (allometric)
            wing_loading[i] = mass ** 0.28 * 25.3
        except Exception:
            continue

    return {"mass_proxy": mass_proxy, "wing_loading_proxy": wing_loading}


print("  Computing wing loading for train...")
wl_train = compute_wing_loading(train_df)
print("  Computing wing loading for test...")
wl_test = compute_wing_loading(test_df)


# ══════════════════════════════════════════════════════════════════════
# B6. Observer Comment Confidence Weights
# ══════════════════════════════════════════════════════════════════════

print("\n--- B6. Observer Comment Weights ---")

# Check for confirming comments
if "observer_comment" in train_df.columns:
    comments = train_df["observer_comment"].fillna("").str.lower()
    # Patterns that confirm species identity
    confirm_patterns = {
        "courtship": 1.5,
        "soaring": 1.3,
        "thermal": 1.3,
        "hovering": 1.3,
        "kestrel": 1.5,
        "sparrowhawk": 1.5,
        "cormorant": 1.5,
        "pigeon": 1.5,
        "goose": 1.5,
        "duck": 1.5,
        "wader": 1.5,
        "starling": 1.5,
        "flock": 1.2,
    }
    sample_weights = np.ones(len(train_df))
    n_boosted = 0
    for pattern, weight in confirm_patterns.items():
        mask = comments.str.contains(pattern, na=False)
        sample_weights[mask] = np.maximum(sample_weights[mask], weight)
        n_boosted += mask.sum()
    print(f"  {n_boosted} tracks with confirming comments (weights up to 1.5x)")
else:
    sample_weights = np.ones(len(train_df))
    print("  No observer_comment column")


# ══════════════════════════════════════════════════════════════════════
# Combine features and retrain
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Building augmented feature matrix")
print("=" * 70)

# Start with E175 selected features
X_train_base = train_feats[selected].values.astype(np.float32)
X_test_base = test_feats[selected].values.astype(np.float32)

# Add new features
new_feats_train = []
new_feats_test = []
new_names = []

for feat_dict_tr, feat_dict_te in [
    (flock_train, flock_test),
    (glide_train, glide_test),
    (sess_train, sess_test),
    (wl_train, wl_test),
]:
    for k in feat_dict_tr:
        new_feats_train.append(feat_dict_tr[k])
        new_feats_test.append(feat_dict_te[k])
        new_names.append(k)

new_train = np.column_stack(new_feats_train).astype(np.float32)
new_test = np.column_stack(new_feats_test).astype(np.float32)

X_train_aug = np.hstack([X_train_base, new_train])
X_test_aug = np.hstack([X_test_base, new_test])

X_train_aug = np.nan_to_num(X_train_aug, nan=0.0, posinf=0.0, neginf=0.0)
X_test_aug = np.nan_to_num(X_test_aug, nan=0.0, posinf=0.0, neginf=0.0)

all_feature_names = selected + new_names
print(f"  Base features: {len(selected)}, New features: {len(new_names)}, Total: {len(all_feature_names)}")
print(f"  New features: {new_names}")


# ══════════════════════════════════════════════════════════════════════
# Train: LGB DART (same as E175, but with new features + comment weights)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Training LGB DART with augmented features")
print("=" * 70)

N_SEEDS = 5


def train_lgb_dart(X_tr, X_te, y, groups, months, sw=None, n_seeds=N_SEEDS, tag=""):
    """Train LGB DART multiclass with SGKF, return OOF and test predictions."""
    n_train, n_test = X_tr.shape[0], X_te.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_tr, y, groups)):
            import time as _time
            _t_fold = _time.time()
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                boosting_type="dart",
                n_estimators=1500,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                colsample_bytree=0.6,
                subsample=0.7,
                drop_rate=0.15,
                is_unbalance=True,
                verbosity=-1,
                random_state=42 + seed + fold,
                n_jobs=-1,
            )
            fit_kwargs = {
                "eval_set": [(X_tr[va_idx], y[va_idx])],
                "callbacks": [lgb.early_stopping(100, verbose=False)],
            }
            if sw is not None:
                fit_kwargs["sample_weight"] = sw[tr_idx]

            model.fit(X_tr[tr_idx], y[tr_idx], **fit_kwargs)
            oof_seed[va_idx] = model.predict_proba(X_tr[va_idx])
            test_seed += model.predict_proba(X_te) / N_FOLDS
            print(f"    {tag} seed={seed+1} fold={fold+1} ({_time.time()-_t_fold:.0f}s)", flush=True)

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1}: {s:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    return oof_mean, test_mean


# 1. Baseline: original features, no comment weights
print("\n[1] Original features (baseline retrain):")
oof_base, test_base = train_lgb_dart(X_train_base, X_test_base, y, groups, train_months, tag="base")
eval_skf_lomo(oof_base, "Original features")

# 2. Augmented features, no comment weights
print("\n[2] Augmented features (+B1-B4):")
oof_aug, test_aug = train_lgb_dart(X_train_aug, X_test_aug, y, groups, train_months, tag="aug")
eval_skf_lomo(oof_aug, "Augmented features")

# 3. Augmented features + comment weights (B6)
print("\n[3] Augmented features + comment weights:")
oof_aug_w, test_aug_w = train_lgb_dart(X_train_aug, X_test_aug, y, groups, train_months, sw=sample_weights, tag="aug_w")
eval_skf_lomo(oof_aug_w, "Aug + comment weights")

# 4. Just new features (ablation: do they help individually?)
print("\n[4] New features only (ablation):")
oof_new, test_new = train_lgb_dart(new_train, new_test, y, groups, train_months, tag="new_only")
eval_skf_lomo(oof_new, "New features only")


# ══════════════════════════════════════════════════════════════════════
# Blend augmented with E175
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Blending augmented models with E175")
print("=" * 70)

oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
eval_skf_lomo(oof_e175, "E175 baseline")

for name, oof, test in [
    ("aug", oof_aug, test_aug),
    ("aug_w", oof_aug_w, test_aug_w),
]:
    for alpha in [0.2, 0.3, 0.4, 0.5]:
        blend = (1-alpha) * oof_e175 + alpha * renorm_rows(oof)
        blend = renorm_rows(blend)
        skf, lomo = eval_skf_lomo(blend, f"E175+{name}@{alpha}")


# ══════════════════════════════════════════════════════════════════════
# Feature importance for new features
# ══════════════════════════════════════════════════════════════════════

print("\n--- New Feature Importance (from last fold) ---")
# Quick single-model to check importance
model = lgb.LGBMClassifier(
    objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    is_unbalance=True, verbosity=-1, n_jobs=-1,
)
model.fit(X_train_aug, y)
importances = model.feature_importances_
for i, name in enumerate(all_feature_names):
    if name in new_names:
        print(f"  {name:30s}: importance = {importances[i]}")


# ══════════════════════════════════════════════════════════════════════
# Save best submissions
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Saving submissions")
print("=" * 70)

# Save augmented model
skf_aug, _ = compute_map(y, oof_aug)
save_submission(renorm_rows(test_aug), "e176_phase_b_aug", cv_map=skf_aug)
np.save(ROOT / "oof_e176_phase_b_aug.npy", oof_aug)
np.save(ROOT / "test_e176_phase_b_aug.npy", test_aug)

# Save best blend
best_blend_test = 0.6 * test_e175 + 0.4 * renorm_rows(test_aug)
best_blend_oof = 0.6 * oof_e175 + 0.4 * renorm_rows(oof_aug)
skf_bl, _ = compute_map(y, renorm_rows(best_blend_oof))
save_submission(renorm_rows(best_blend_test), "e176_phase_b_blend", cv_map=skf_bl)

elapsed = time.time() - t0
print(f"\nPhase B completed in {elapsed:.0f}s")
print("=" * 70)
