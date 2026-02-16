"""Test 6 new domain-informed features for Cormorant detection.

A. Aspect angle RCS profile (RCS vs angle to radar)
B. Zero-glide detection (flap ratio, any glide segments?)
C. Low altitude commuting (fraction of track < 10m)
D. Spatial corridor features (heading, distance from radar)
E. Speed constancy (coefficient of variation)
F. RCS amplitude pattern (deep wingbeat regularity)

Evaluate each feature's discriminative power and combine with existing features.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import build_features, ALL_TEMPORAL

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

ts_train = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months_train = ts_train.dt.month.values

print("=" * 60, flush=True)
print("NEW DOMAIN-INFORMED FEATURES FOR CORMORANTS", flush=True)
print("=" * 60, flush=True)

# ======================================================================
# Estimate radar position (center of all trajectories)
# ======================================================================
print("\nEstimating radar position...", flush=True)
all_lats = []
all_lons = []
for _, row in train_df.iterrows():
    pts = parse_ewkb_4d(row["trajectory"])
    all_lats.extend([p[1] for p in pts])
    all_lons.extend([p[0] for p in pts])
# Radar is likely at the densest point - use median
radar_lat = np.median(all_lats)
radar_lon = np.median(all_lons)
print(f"  Estimated radar: {radar_lat:.6f}N, {radar_lon:.6f}E", flush=True)

# ======================================================================
# Extract new features for a dataframe
# ======================================================================
def extract_new_features(df):
    """Extract all 6 new feature sets from trajectories."""
    records = []

    for _, row in df.iterrows():
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])
        n = len(pts)

        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs = np.array([p[3] for p in pts])

        # Derived: speed, heading, bearing change per step
        if n > 1:
            dt = np.diff(times)
            dt[dt == 0] = 1e-6
            dx = np.diff(lons) * 67000  # meters (at 53N)
            dy = np.diff(lats) * 111000
            dalt = np.diff(alts)
            horiz_dist = np.sqrt(dx**2 + dy**2)
            speed = horiz_dist / dt
            headings = np.arctan2(dy, dx)  # radians, 0=E, pi/2=N
        else:
            speed = np.array([row["airspeed"]])
            headings = np.array([0.0])

        feat = {}

        # ==========================================================
        # A. ASPECT ANGLE RCS PROFILE
        # ==========================================================
        # Angle from radar to bird at each point
        dx_radar = (lons - radar_lon) * 67000
        dy_radar = (lats - radar_lat) * 111000
        angle_to_radar = np.arctan2(dy_radar, dx_radar)  # direction FROM radar TO bird

        if n > 1:
            # Bird's heading at each point (use forward diff, repeat last)
            bird_heading = np.concatenate([headings, [headings[-1]]])

            # Aspect angle = difference between bird heading and angle to radar
            aspect = bird_heading - angle_to_radar
            aspect = (aspect + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

            # How much does RCS vary with aspect angle?
            # Split into broadside (|aspect| near pi/2) vs head/tail (|aspect| near 0 or pi)
            abs_aspect = np.abs(aspect)
            broadside_mask = (abs_aspect > np.pi/4) & (abs_aspect < 3*np.pi/4)
            headtail_mask = ~broadside_mask

            if broadside_mask.sum() > 0 and headtail_mask.sum() > 0:
                rcs_broadside = rcs[broadside_mask].mean()
                rcs_headtail = rcs[headtail_mask].mean()
                feat["rcs_aspect_diff"] = rcs_broadside - rcs_headtail
                feat["rcs_broadside_mean"] = rcs_broadside
                feat["rcs_headtail_mean"] = rcs_headtail
            else:
                feat["rcs_aspect_diff"] = 0.0
                feat["rcs_broadside_mean"] = rcs.mean()
                feat["rcs_headtail_mean"] = rcs.mean()

            # RCS variation explained by aspect angle (correlation)
            feat["rcs_aspect_corr"] = abs(np.corrcoef(abs_aspect, rcs)[0, 1]) if n > 2 else 0.0

            # Aspect angle range (how much angle changes during track)
            feat["aspect_range"] = abs_aspect.max() - abs_aspect.min()

            # Distance to radar (mean and variation)
            dist_to_radar = np.sqrt(dx_radar**2 + dy_radar**2)
            feat["radar_dist_mean"] = dist_to_radar.mean()
            feat["radar_dist_std"] = dist_to_radar.std()
        else:
            feat["rcs_aspect_diff"] = 0.0
            feat["rcs_broadside_mean"] = rcs.mean()
            feat["rcs_headtail_mean"] = rcs.mean()
            feat["rcs_aspect_corr"] = 0.0
            feat["aspect_range"] = 0.0
            feat["radar_dist_mean"] = 0.0
            feat["radar_dist_std"] = 0.0

        # ==========================================================
        # B. ZERO-GLIDE DETECTION
        # ==========================================================
        if n > 3:
            # Glide = altitude decreasing + speed stable or increasing
            # Flap = altitude changing + RCS oscillating
            # Simple: look at altitude change rate
            alt_diff = np.diff(alts)
            descending = (alt_diff < -0.5).sum() / max(len(alt_diff), 1)
            ascending = (alt_diff > 0.5).sum() / max(len(alt_diff), 1)
            level = 1.0 - descending - ascending

            feat["frac_level_flight"] = level
            feat["frac_descending"] = descending
            feat["frac_ascending"] = ascending

            # Any sustained glide (3+ consecutive descending steps)?
            consecutive_desc = 0
            max_desc = 0
            for ad in alt_diff:
                if ad < -0.5:
                    consecutive_desc += 1
                    max_desc = max(max_desc, consecutive_desc)
                else:
                    consecutive_desc = 0
            feat["max_consecutive_descent"] = max_desc
            feat["has_glide_segment"] = 1.0 if max_desc >= 3 else 0.0

            # Pure powered flight score: level + no long glides
            feat["powered_flight_score"] = level * (1.0 - min(max_desc / 10.0, 1.0))
        else:
            feat["frac_level_flight"] = 0.0
            feat["frac_descending"] = 0.0
            feat["frac_ascending"] = 0.0
            feat["max_consecutive_descent"] = 0
            feat["has_glide_segment"] = 0.0
            feat["powered_flight_score"] = 0.0

        # ==========================================================
        # C. LOW ALTITUDE COMMUTING
        # ==========================================================
        feat["frac_below_5m"] = (alts < 5).mean()
        feat["frac_below_10m"] = (alts < 10).mean()
        feat["frac_below_20m"] = (alts < 20).mean()
        feat["min_altitude"] = alts.min()
        feat["alt_start"] = alts[0]
        feat["alt_end"] = alts[-1]
        feat["alt_change"] = alts[-1] - alts[0]  # net altitude change
        feat["alt_stability"] = 1.0 / (1.0 + np.std(alts)) if n > 1 else 0.0

        # ==========================================================
        # D. SPATIAL CORRIDOR FEATURES
        # ==========================================================
        if n > 1:
            # Overall flight heading (start to end)
            dx_total = (lons[-1] - lons[0]) * 67000
            dy_total = (lats[-1] - lats[0]) * 111000
            overall_heading = np.degrees(np.arctan2(dy_total, dx_total))
            feat["flight_heading_deg"] = overall_heading
            feat["heading_sin"] = np.sin(np.radians(overall_heading))
            feat["heading_cos"] = np.cos(np.radians(overall_heading))

            # Is heading NW/W/SW? (Cormorant commuting direction)
            # NW/W/SW = heading between 180 and 360 (or -180 to 0)
            feat["heading_westward"] = 1.0 if -180 <= overall_heading <= 0 else 0.0

            # Heading consistency: std of instantaneous headings
            if len(headings) > 1:
                # Circular std
                heading_sin = np.sin(headings)
                heading_cos = np.cos(headings)
                R = np.sqrt(heading_sin.mean()**2 + heading_cos.mean()**2)
                feat["heading_consistency"] = R  # 1.0 = perfectly straight
            else:
                feat["heading_consistency"] = 0.0

            # Distance and direction from radar at midpoint
            mid = n // 2
            feat["mid_dist_radar"] = np.sqrt(
                ((lons[mid] - radar_lon) * 67000)**2 +
                ((lats[mid] - radar_lat) * 111000)**2
            )
        else:
            feat["flight_heading_deg"] = 0.0
            feat["heading_sin"] = 0.0
            feat["heading_cos"] = 0.0
            feat["heading_westward"] = 0.0
            feat["heading_consistency"] = 0.0
            feat["mid_dist_radar"] = 0.0

        # ==========================================================
        # E. SPEED CONSTANCY
        # ==========================================================
        if len(speed) > 2:
            speed_mean = speed.mean()
            speed_std = speed.std()
            feat["speed_cv"] = speed_std / max(speed_mean, 1e-6)  # coeff of variation
            feat["speed_iqr"] = np.percentile(speed, 75) - np.percentile(speed, 25)
            feat["speed_range_ratio"] = (speed.max() - speed.min()) / max(speed_mean, 1e-6)

            # Is speed constant throughout? (low CV + no trend)
            half = len(speed) // 2
            speed_first = speed[:half].mean()
            speed_second = speed[half:].mean()
            feat["speed_trend"] = (speed_second - speed_first) / max(speed_mean, 1e-6)

            # Acceleration profile
            accel = np.diff(speed)
            feat["accel_mean"] = np.abs(accel).mean()
            feat["accel_std"] = accel.std()
            feat["frac_constant_speed"] = (np.abs(accel) < 2.0).mean()  # <2 m/s^2 change
        else:
            feat["speed_cv"] = 0.0
            feat["speed_iqr"] = 0.0
            feat["speed_range_ratio"] = 0.0
            feat["speed_trend"] = 0.0
            feat["accel_mean"] = 0.0
            feat["accel_std"] = 0.0
            feat["frac_constant_speed"] = 0.0

        # ==========================================================
        # F. RCS AMPLITUDE PATTERN (deep wingbeat regularity)
        # ==========================================================
        if n > 5:
            rcs_diff = np.diff(rcs)
            feat["rcs_swing"] = rcs.max() - rcs.min()
            feat["rcs_iqr"] = np.percentile(rcs, 75) - np.percentile(rcs, 25)

            # Regularity: autocorrelation of RCS at lags 1-5
            rcs_centered = rcs - rcs.mean()
            rcs_var = np.var(rcs)
            if rcs_var > 1e-10:
                for lag in [1, 2, 3, 4, 5]:
                    if lag < n:
                        acf = np.corrcoef(rcs_centered[:-lag], rcs_centered[lag:])[0, 1]
                    else:
                        acf = 0.0
                    feat[f"rcs_acf_lag{lag}"] = acf

                # Persistence: how many lags stay positive?
                persistence = 0
                for lag in range(1, min(n, 11)):
                    acf = np.corrcoef(rcs_centered[:-lag], rcs_centered[lag:])[0, 1]
                    if acf > 0:
                        persistence += 1
                    else:
                        break
                feat["rcs_acf_persistence"] = persistence
            else:
                for lag in [1, 2, 3, 4, 5]:
                    feat[f"rcs_acf_lag{lag}"] = 0.0
                feat["rcs_acf_persistence"] = 0

            # Sign changes in RCS derivative (oscillation frequency proxy)
            sign_changes = np.sum(np.diff(np.sign(rcs_diff)) != 0)
            feat["rcs_sign_changes_per_sec"] = sign_changes / max(times[-1], 1)

            # Deep wingbeat: large regular swings
            feat["rcs_diff_mean_abs"] = np.abs(rcs_diff).mean()
            feat["rcs_diff_std"] = rcs_diff.std()
        else:
            feat["rcs_swing"] = 0.0
            feat["rcs_iqr"] = 0.0
            for lag in [1, 2, 3, 4, 5]:
                feat[f"rcs_acf_lag{lag}"] = 0.0
            feat["rcs_acf_persistence"] = 0
            feat["rcs_sign_changes_per_sec"] = 0.0
            feat["rcs_diff_mean_abs"] = 0.0
            feat["rcs_diff_std"] = 0.0

        records.append(feat)

    return pd.DataFrame(records)

# ======================================================================
# Extract for train
# ======================================================================
print("\nExtracting new features (train)...", flush=True)
new_feats_train = extract_new_features(train_df)
print(f"  New features: {new_feats_train.shape[1]}", flush=True)
print(f"  Columns: {list(new_feats_train.columns)}", flush=True)

# Clean
new_feats_train = new_feats_train.replace([np.inf, -np.inf], np.nan).fillna(0)

# ======================================================================
# Discriminative power of each new feature
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("DISCRIMINATIVE POWER: NEW FEATURES", flush=True)
print("=" * 60, flush=True)

corm_mask = y_bin == 1
print(f"\n  {'Feature':<30s} {'t-stat':>8s} {'p-value':>10s} "
      f"{'Corm_med':>10s} {'Rest_med':>10s}", flush=True)
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}", flush=True)

feature_groups = {
    "A_aspect": ["rcs_aspect_diff", "rcs_broadside_mean", "rcs_headtail_mean",
                  "rcs_aspect_corr", "aspect_range", "radar_dist_mean", "radar_dist_std"],
    "B_glide": ["frac_level_flight", "frac_descending", "frac_ascending",
                 "max_consecutive_descent", "has_glide_segment", "powered_flight_score"],
    "C_low_alt": ["frac_below_5m", "frac_below_10m", "frac_below_20m",
                   "min_altitude", "alt_start", "alt_end", "alt_change", "alt_stability"],
    "D_corridor": ["flight_heading_deg", "heading_sin", "heading_cos",
                    "heading_westward", "heading_consistency", "mid_dist_radar"],
    "E_speed": ["speed_cv", "speed_iqr", "speed_range_ratio", "speed_trend",
                 "accel_mean", "accel_std", "frac_constant_speed"],
    "F_rcs_amp": ["rcs_swing", "rcs_iqr", "rcs_acf_lag1", "rcs_acf_lag2",
                   "rcs_acf_lag3", "rcs_acf_lag4", "rcs_acf_lag5",
                   "rcs_acf_persistence", "rcs_sign_changes_per_sec",
                   "rcs_diff_mean_abs", "rcs_diff_std"],
}

best_per_group = {}
for group, feats in feature_groups.items():
    print(f"\n  --- {group} ---", flush=True)
    best_t = 0
    best_feat = ""
    for feat in feats:
        if feat not in new_feats_train.columns:
            continue
        c_vals = new_feats_train.loc[corm_mask, feat].values
        r_vals = new_feats_train.loc[~corm_mask, feat].values
        t, p = ttest_ind(c_vals, r_vals, equal_var=False)
        c_med = np.median(c_vals)
        r_med = np.median(r_vals)
        marker = " ***" if abs(t) > 3 else " **" if abs(t) > 2 else ""
        print(f"  {feat:<30s} {t:>+8.2f} {p:>10.4f} {c_med:>10.4f} {r_med:>10.4f}{marker}", flush=True)
        if abs(t) > best_t:
            best_t = abs(t)
            best_feat = feat
    best_per_group[group] = (best_feat, best_t)

print(f"\n  Best feature per group:", flush=True)
for group, (feat, t) in sorted(best_per_group.items(), key=lambda x: -x[1][1]):
    print(f"    {group}: {feat} (|t|={t:.2f})", flush=True)

# ======================================================================
# SKF evaluation: new features alone and combined
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SKF EVALUATION: NEW FEATURES", flush=True)
print("=" * 60, flush=True)

from catboost import CatBoostClassifier

# Build existing tabular features
print("\n  Building tabular features...", flush=True)
X_tab = build_features(train_df)
drop_cols = [c for c in ALL_TEMPORAL if c in X_tab.columns]
X_tab = X_tab.drop(columns=drop_cols)
X_tab = X_tab.replace([np.inf, -np.inf], np.nan).fillna(0)

# Top-50 existing
t_stats_existing = []
for col in X_tab.columns:
    t, p = ttest_ind(X_tab.loc[corm_mask, col], X_tab.loc[~corm_mask, col], equal_var=False)
    t_stats_existing.append((col, abs(t)))
t_stats_existing.sort(key=lambda x: -x[1])
top50_existing = [x[0] for x in t_stats_existing[:50]]

X_tab50 = X_tab[top50_existing].values
X_new = new_feats_train.values
X_combined = np.hstack([X_tab50, X_new])
X_all_features = np.hstack([X_tab.values, X_new])

# Also: new features grouped
X_groups = {}
for group, feats in feature_groups.items():
    cols = [f for f in feats if f in new_feats_train.columns]
    X_groups[group] = new_feats_train[cols].values

def eval_skf(X, name):
    """5-fold SKF evaluation for binary Cormorant detection."""
    oof = np.full(len(y_bin), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y_bin):
        n_pos = y_bin[tr_idx].sum()
        n_neg = len(tr_idx) - n_pos
        cb = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg/max(n_pos,1),
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X[tr_idx], y_bin[tr_idx])
        oof[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
    ap = average_precision_score(y_bin, oof)
    return ap

def eval_lomo(X, name):
    """LOMO evaluation."""
    oof = np.full(len(y_bin), np.nan)
    for held_month in sorted(np.unique(months_train)):
        val_mask = months_train == held_month
        train_mask = ~val_mask
        if y_bin[val_mask].sum() == 0:
            continue
        n_pos = y_bin[train_mask].sum()
        n_neg = train_mask.sum() - n_pos
        cb = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg/max(n_pos,1),
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X[train_mask], y_bin[train_mask])
        oof[val_mask] = cb.predict_proba(X[val_mask])[:, 1]
    valid = ~np.isnan(oof)
    return average_precision_score(y_bin[valid], oof[valid]) if y_bin[valid].sum() > 0 else 0.0

print(f"\n  {'Feature Set':<40s} {'SKF AP':>8s} {'LOMO AP':>8s}", flush=True)
print(f"  {'-'*40} {'-'*8} {'-'*8}", flush=True)

# Individual groups
for group, X_g in X_groups.items():
    skf_ap = eval_skf(X_g, group)
    lomo_ap = eval_lomo(X_g, group)
    print(f"  {group:<40s} {skf_ap:>8.4f} {lomo_ap:>8.4f}", flush=True)

# Combined sets
sets = {
    "all_new_features": X_new,
    "tabular_top50 (baseline)": X_tab50,
    "tabular_top50 + all_new": X_combined,
    "all_tabular + all_new": X_all_features,
}
for name, X_s in sets.items():
    skf_ap = eval_skf(X_s, name)
    lomo_ap = eval_lomo(X_s, name)
    print(f"  {name:<40s} {skf_ap:>8.4f} {lomo_ap:>8.4f}", flush=True)

# ======================================================================
# Best new features + top existing: cherry-picked combo
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("CHERRY-PICKED: BEST NEW + BEST EXISTING", flush=True)
print("=" * 60, flush=True)

# Rank ALL features (existing + new) by t-stat
all_feat_df = pd.concat([X_tab, new_feats_train], axis=1)
all_t = []
for col in all_feat_df.columns:
    t, p = ttest_ind(all_feat_df.loc[corm_mask, col], all_feat_df.loc[~corm_mask, col], equal_var=False)
    all_t.append((col, abs(t), p))
all_t.sort(key=lambda x: -x[1])

print(f"\n  Top 20 features (existing + new combined):", flush=True)
new_feat_names = set(new_feats_train.columns)
for i, (feat, t, p) in enumerate(all_t[:20]):
    is_new = " [NEW]" if feat in new_feat_names else ""
    print(f"    {i+1:2d}. {feat:<35s} |t|={t:.2f}{is_new}", flush=True)

# Top 50 from combined ranking
top50_combined = [x[0] for x in all_t[:50]]
n_new_in_top50 = sum(1 for f in top50_combined if f in new_feat_names)
print(f"\n  New features in top 50: {n_new_in_top50}", flush=True)

X_top50_combined = all_feat_df[top50_combined].values
skf_ap = eval_skf(X_top50_combined, "top50_combined")
lomo_ap = eval_lomo(X_top50_combined, "top50_combined")
print(f"  Top-50 combined SKF AP: {skf_ap:.4f}, LOMO AP: {lomo_ap:.4f}", flush=True)

# ======================================================================
# Full 9-class evaluation: do new features help the multiclass model?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("9-CLASS MULTICLASS: DO NEW FEATURES HELP?", flush=True)
print("=" * 60, flush=True)

from src.metrics import compute_map

def effective_number_weights(labels, n_classes, beta=0.999):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    eff[eff == 0] = 1.0
    weights = 1.0 / eff
    weights = weights / weights.sum() * n_classes
    return weights

for feat_name, X_feat in [("tabular_only", X_tab.values),
                           ("tabular + new_feats", np.hstack([X_tab.values, X_new]))]:
    oof = np.zeros((len(y), len(CLASSES)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y)):
        ew = effective_number_weights(y[tr_idx], len(CLASSES))
        sw = ew[y[tr_idx]]

        cb = CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X_feat[tr_idx], y[tr_idx], sample_weight=sw)
        oof[va_idx] = cb.predict_proba(X_feat[va_idx])

    map_score, per_class = compute_map(y, oof)
    print(f"\n  {feat_name}: mAP = {map_score:.4f}", flush=True)
    for ci, cls in enumerate(CLASSES):
        marker = " <--" if cls == "Cormorants" else ""
        print(f"    {cls:<18s}: {per_class[ci]:.4f}{marker}", flush=True)

print("\nDone!", flush=True)
