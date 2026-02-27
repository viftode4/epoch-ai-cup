"""Analysis: Deep radar-bird physics features.

Explore genuinely novel feature concepts that go beyond aggregate statistics:
1. Detection gap rate (body size proxy independent of RCS)
2. Wing loading proxy (speed^2 / RCS_linear)
3. RCS-speed cross-correlation (wing morphology)
4. 3D soaring score (thermal signature for BoP)
5. Speed-altitude phase portrait (trajectory shape in 2D)
6. Aspect-angle-corrected RCS (separate body from wings)
7. Altitude profile shape (motif-based classification)
8. Speed persistence (metronomic vs erratic)
9. Flight power proxy

For each feature: compute per-class distributions, effect sizes (Cohen's d),
and test whether it helps separate our PROBLEM classes:
  - Cormorants vs Gulls (17.5% accuracy)
  - Birds of Prey fast vs slow (speed > 12 m/s confused with Gulls)
  - Ducks vs Pigeons (overlap)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, parse_ewkb_4d, parse_trajectory_time

# Estimated radar position (Eemshaven wind farm)
RADAR_LAT = 53.44
RADAR_LON = 6.83


def extract_deep_features(row):
    """Extract ALL deep radar physics features for one track."""
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    n = len(pts)

    feat = {}

    if n < 4:
        return feat

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    dt = np.diff(times)
    dt = np.maximum(dt, 0.001)
    duration = times[-1] - times[0]
    median_dt = np.median(dt)

    # Horizontal distances and speeds
    dx = np.diff(lons) * 67000  # meters at ~53N
    dy = np.diff(lats) * 111000
    horiz_dist = np.sqrt(dx**2 + dy**2)
    speeds = horiz_dist / dt
    headings = np.arctan2(dy, dx)

    mean_speed = np.mean(speeds)
    rcs_mean = np.mean(rcs)

    # ===================================================================
    # 1. DETECTION GAP FEATURES (body size proxy)
    # ===================================================================
    gap_threshold = max(median_dt * 1.5, 1.5)
    is_gap = dt > gap_threshold
    feat["gap_fraction"] = float(is_gap.mean())
    feat["max_gap_s"] = float(dt.max())
    feat["n_gaps"] = int(is_gap.sum())
    feat["gap_rate_per_min"] = float(is_gap.sum() / max(duration/60, 0.01))

    # ===================================================================
    # 2. WING LOADING PROXY (speed^2 / RCS_linear)
    # ===================================================================
    rcs_linear = 10.0 ** (rcs_mean / 10.0)
    feat["wing_loading_proxy"] = mean_speed**2 / max(rcs_linear, 1e-10)
    # Also: speed per unit body size
    feat["speed_per_rcs"] = mean_speed / max(rcs_linear**0.5, 1e-10)

    # ===================================================================
    # 3. RCS-SPEED CROSS-CORRELATION (wing morphology)
    # ===================================================================
    rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])
    if np.std(speeds) > 1e-6 and np.std(rcs_mid) > 1e-6:
        cc = np.corrcoef(rcs_mid, speeds)[0, 1]
        feat["rcs_speed_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["rcs_speed_corr"] = 0.0

    # ===================================================================
    # 4. 3D SOARING SCORE (thermal signature)
    # ===================================================================
    if len(headings) > 1:
        bearing_changes = np.abs(
            np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
        )
        alt_rates = np.diff(alts[:-1]) / dt[:-1]
        min_len = min(len(bearing_changes), len(alt_rates))
        # Soaring = turning (>0.1 rad/step) AND climbing (>0.5 m/s)
        is_soaring = (bearing_changes[:min_len] > 0.1) & (alt_rates[:min_len] > 0.5)
        feat["soaring_score"] = float(is_soaring.mean())
        # Also: thermal_climb = fraction climbing > 1 m/s while turning > 0.2 rad
        is_thermal = (bearing_changes[:min_len] > 0.2) & (alt_rates[:min_len] > 1.0)
        feat["thermal_score"] = float(is_thermal.mean())
    else:
        feat["soaring_score"] = 0.0
        feat["thermal_score"] = 0.0

    # ===================================================================
    # 5. SPEED-ALTITUDE PHASE PORTRAIT
    # ===================================================================
    # Map each timestep to a (speed, altitude) point. Analyze the shape.
    # Use segment midpoints for both.
    alt_mid = 0.5 * (alts[:-1] + alts[1:])
    if len(speeds) > 5:
        # Normalize to unit scale for shape analysis
        sp_norm = (speeds - speeds.mean()) / max(speeds.std(), 1e-6)
        al_norm = (alt_mid - alt_mid.mean()) / max(alt_mid.std(), 1e-6)

        # a) Phase space area (convex hull proxy): std of product
        feat["phase_area"] = float(np.std(sp_norm * al_norm))

        # b) Phase space correlation: does speed co-vary with altitude?
        if np.std(speeds) > 1e-6 and np.std(alt_mid) > 1e-6:
            cc = np.corrcoef(speeds, alt_mid)[0, 1]
            feat["speed_alt_corr"] = float(cc) if np.isfinite(cc) else 0.0
        else:
            feat["speed_alt_corr"] = 0.0

        # c) Trajectory complexity: total path length in normalized phase space
        d_sp = np.diff(sp_norm)
        d_al = np.diff(al_norm)
        phase_path_len = np.sum(np.sqrt(d_sp**2 + d_al**2))
        feat["phase_path_length"] = float(phase_path_len / max(len(speeds)-1, 1))

        # d) Number of loops/reversals in phase space
        # Approximate as direction changes in speed
        speed_diffs = np.diff(speeds)
        speed_reversals = np.sum(speed_diffs[:-1] * speed_diffs[1:] < 0)
        feat["speed_reversals_per_min"] = float(speed_reversals / max(duration/60, 0.01))
    else:
        feat["phase_area"] = 0.0
        feat["speed_alt_corr"] = 0.0
        feat["phase_path_length"] = 0.0
        feat["speed_reversals_per_min"] = 0.0

    # ===================================================================
    # 6. ASPECT-ANGLE-CORRECTED RCS
    # ===================================================================
    # Compute angle from radar to bird at each point
    dx_radar = (lons - RADAR_LON) * 67000
    dy_radar = (lats - RADAR_LAT) * 111000
    angle_to_radar = np.arctan2(dy_radar, dx_radar)

    # Bird heading at each point
    bird_heading = np.concatenate([headings, [headings[-1]]])

    # Aspect angle = bird heading - angle_to_radar
    aspect = bird_heading - angle_to_radar
    aspect = (aspect + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]
    abs_aspect = np.abs(aspect)

    # RCS at broadside vs head-on
    broadside = (abs_aspect > np.pi/4) & (abs_aspect < 3*np.pi/4)
    headtail = ~broadside
    if broadside.sum() > 0 and headtail.sum() > 0:
        feat["rcs_aspect_diff"] = float(rcs[broadside].mean() - rcs[headtail].mean())
    else:
        feat["rcs_aspect_diff"] = 0.0

    # How much does aspect angle explain RCS variation?
    if n > 3 and np.std(rcs) > 1e-6 and np.std(abs_aspect) > 1e-6:
        cc = np.corrcoef(abs_aspect, rcs)[0, 1]
        feat["rcs_aspect_corr"] = float(abs(cc)) if np.isfinite(cc) else 0.0
    else:
        feat["rcs_aspect_corr"] = 0.0

    # RCS residual after removing aspect effect (body-only size)
    if n > 5 and np.std(abs_aspect) > 1e-6:
        # Linear regression: RCS = a * abs_aspect + b + residual
        slope, intercept, _, _, _ = stats.linregress(abs_aspect, rcs)
        rcs_residual = rcs - (slope * abs_aspect + intercept)
        feat["rcs_body_only_mean"] = float(np.mean(rcs_residual))
        feat["rcs_body_only_std"] = float(np.std(rcs_residual))
    else:
        feat["rcs_body_only_mean"] = 0.0
        feat["rcs_body_only_std"] = 0.0

    # Distance from radar
    dist_to_radar = np.sqrt(dx_radar**2 + dy_radar**2)
    feat["radar_dist_mean"] = float(dist_to_radar.mean())
    feat["radar_dist_std"] = float(dist_to_radar.std())

    # ===================================================================
    # 7. ALTITUDE PROFILE SHAPE
    # ===================================================================
    # Categorize the altitude profile into flight modes
    alt_diffs = np.diff(alts)
    frac_climbing = float((alt_diffs > 0.5).mean())
    frac_descending = float((alt_diffs < -0.5).mean())
    frac_level = 1.0 - frac_climbing - frac_descending

    feat["frac_climbing"] = frac_climbing
    feat["frac_descending"] = frac_descending
    feat["frac_level"] = frac_level

    # Bounding flight index: altitude zero-crossing rate
    alt_detrended = alts - np.linspace(alts[0], alts[-1], n)
    if n > 3:
        zcr = np.sum(np.diff(np.sign(alt_detrended)) != 0) / (n - 1)
        feat["alt_zcr"] = float(zcr)
    else:
        feat["alt_zcr"] = 0.0

    # Net altitude change: migration vs local flight
    feat["alt_net_change"] = float(alts[-1] - alts[0])
    feat["alt_net_rate"] = float((alts[-1] - alts[0]) / max(duration, 0.01))

    # Max consecutive climb (thermal detection)
    max_climb_run = 0
    run = 0
    for d in alt_diffs:
        if d > 0.3:
            run += 1
            max_climb_run = max(max_climb_run, run)
        else:
            run = 0
    feat["max_climb_run"] = max_climb_run

    # ===================================================================
    # 8. SPEED PERSISTENCE
    # ===================================================================
    if len(speeds) > 5 and np.std(speeds) > 1e-6:
        sp_c = speeds - np.mean(speeds)
        sp_var = np.var(speeds)
        # Lag-1 autocorrelation of speed
        ac1 = float(np.mean(sp_c[:-1] * sp_c[1:]) / max(sp_var, 1e-10))
        feat["speed_ac1"] = ac1 if np.isfinite(ac1) else 0.0
        # Speed coefficient of variation (inverse of consistency)
        feat["speed_cv"] = float(np.std(speeds) / max(np.mean(speeds), 1e-6))
    else:
        feat["speed_ac1"] = 0.0
        feat["speed_cv"] = 0.0

    # ===================================================================
    # 9. FLIGHT POWER PROXY
    # ===================================================================
    # Aerodynamic power ~ speed^3 (parasite drag) + climb_rate * weight
    alt_gain = float(np.sum(np.maximum(np.diff(alts), 0)))
    if duration > 0:
        feat["power_proxy"] = mean_speed**3 + 9.81 * alt_gain / max(duration, 1)
    else:
        feat["power_proxy"] = 0.0

    # ===================================================================
    # 10. VERTICAL/HORIZONTAL ACCELERATION RATIO
    # ===================================================================
    if len(speeds) > 3:
        h_accel = np.abs(np.diff(speeds) / dt[:-1]) if len(dt) > 1 else np.array([0.0])
        v_accel = np.abs(np.diff(alts[:-1]) / dt[:-1]) if len(dt) > 1 else np.array([0.0])
        min_len = min(len(h_accel), len(v_accel))
        h_accel = h_accel[:min_len]
        v_accel = v_accel[:min_len]
        h_mean = float(np.mean(h_accel))
        v_mean = float(np.mean(v_accel))
        feat["vh_accel_ratio"] = v_mean / max(h_mean, 1e-6)
        feat["accel_3d_magnitude"] = float(np.mean(np.sqrt(h_accel**2 + v_accel**2)))
    else:
        feat["vh_accel_ratio"] = 0.0
        feat["accel_3d_magnitude"] = 0.0

    # ===================================================================
    # 11. HEADING CONSISTENCY (circular statistics)
    # ===================================================================
    if len(headings) > 1:
        h_sin = np.sin(headings)
        h_cos = np.cos(headings)
        R = np.sqrt(h_sin.mean()**2 + h_cos.mean()**2)
        feat["heading_consistency"] = float(R)
    else:
        feat["heading_consistency"] = 0.0

    # ===================================================================
    # 12. RCS TEMPORAL DERIVATIVE (rate of RCS change)
    # ===================================================================
    if n > 2:
        rcs_rate = np.abs(np.diff(rcs) / dt)
        feat["rcs_rate_mean"] = float(np.mean(rcs_rate))
        feat["rcs_rate_std"] = float(np.std(rcs_rate))
    else:
        feat["rcs_rate_mean"] = 0.0
        feat["rcs_rate_std"] = 0.0

    return feat


# ======================================================================
# MAIN ANALYSIS
# ======================================================================
print("=" * 70, flush=True)
print("DEEP RADAR-BIRD PHYSICS FEATURE ANALYSIS".center(70), flush=True)
print("=" * 70, flush=True)

print("\nLoading train data...", flush=True)
train_df = load_train()

print("Extracting deep features...", flush=True)
records = []
for idx, (_, row) in enumerate(train_df.iterrows()):
    if idx % 500 == 0:
        print(f"  {idx}/{len(train_df)}", flush=True)
    feat = extract_deep_features(row)
    feat["bird_group"] = row["bird_group"]
    feat["track_id"] = row["track_id"]
    records.append(feat)
print(f"  {len(train_df)}/{len(train_df)} done", flush=True)

df = pd.DataFrame(records)
feat_cols = [c for c in df.columns if c not in ("bird_group", "track_id")]
print(f"\nExtracted {len(feat_cols)} deep features", flush=True)

# ======================================================================
# A. PER-CLASS DISTRIBUTIONS
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("A. PER-CLASS MEAN VALUES".center(70), flush=True)
print("=" * 70, flush=True)

# Show per-class means for all features
means = df.groupby("bird_group")[feat_cols].mean()
# Reorder by CLASSES
means = means.reindex([c for c in CLASSES if c in means.index])

for feat_name in feat_cols:
    vals = means[feat_name]
    # Only print features with meaningful variation
    if vals.std() > 1e-6:
        # Sort by value for this feature
        sorted_vals = vals.sort_values(ascending=False)
        top3 = ", ".join([f"{c[:4]}={v:.3f}" for c, v in sorted_vals.head(3).items()])
        bot3 = ", ".join([f"{c[:4]}={v:.3f}" for c, v in sorted_vals.tail(3).items()])
        print(f"  {feat_name:30s} HIGH: {top3}  |  LOW: {bot3}", flush=True)


# ======================================================================
# B. COHEN'S D: KEY CLASS PAIRS
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("B. COHEN'S D FOR PROBLEM CLASS PAIRS".center(70), flush=True)
print("=" * 70, flush=True)

pairs = [
    ("Cormorants", "Gulls"),
    ("Birds of Prey", "Gulls"),
    ("Ducks", "Pigeons"),
    ("Ducks", "Songbirds"),
    ("Geese", "Gulls"),
    ("Clutter", "Gulls"),
    ("Pigeons", "Songbirds"),
    ("Waders", "Gulls"),
]

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled < 1e-10:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled

for c1, c2 in pairs:
    print(f"\n  {c1} vs {c2}:", flush=True)
    g1 = df[df["bird_group"] == c1]
    g2 = df[df["bird_group"] == c2]

    scores = []
    for feat_name in feat_cols:
        v1 = g1[feat_name].dropna().values
        v2 = g2[feat_name].dropna().values
        d = cohens_d(v1, v2)
        scores.append((feat_name, abs(d), d))

    # Sort by absolute effect size
    scores.sort(key=lambda x: -x[1])
    for fname, abs_d, d in scores[:8]:
        m1 = g1[fname].mean()
        m2 = g2[fname].mean()
        bar = "+" * min(int(abs_d * 10), 30)
        direction = "^" if d > 0 else "v"
        print(f"    {fname:30s} d={d:+.3f} {bar}  ({c1[:4]}={m1:.3f}, {c2[:4]}={m2:.3f})", flush=True)


# ======================================================================
# C. CORMORANT DEEP DIVE
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("C. CORMORANTS: WHAT MAKES THEM UNIQUE?".center(70), flush=True)
print("=" * 70, flush=True)

corm = df[df["bird_group"] == "Cormorants"]
gull = df[df["bird_group"] == "Gulls"]
rest = df[df["bird_group"] != "Cormorants"]

print(f"\n  N: Cormorants={len(corm)}, Gulls={len(gull)}, All others={len(rest)}", flush=True)
print("\n  Features where Cormorants differ MOST from Gulls:", flush=True)

corm_vs_gull = []
for feat_name in feat_cols:
    v1 = corm[feat_name].dropna().values
    v2 = gull[feat_name].dropna().values
    d = cohens_d(v1, v2)
    corm_vs_gull.append((feat_name, abs(d), d, v1.mean(), v2.mean()))

corm_vs_gull.sort(key=lambda x: -x[1])
for fname, abs_d, d, mc, mg in corm_vs_gull[:15]:
    print(f"    {fname:30s} d={d:+.3f}  (Corm={mc:.3f}, Gull={mg:.3f})", flush=True)

# What about Cormorants vs ALL other classes?
print("\n  Features where Cormorants are MOST distinctive overall:", flush=True)
corm_vs_rest = []
for feat_name in feat_cols:
    v1 = corm[feat_name].dropna().values
    v2 = rest[feat_name].dropna().values
    d = cohens_d(v1, v2)
    corm_vs_rest.append((feat_name, abs(d), d, v1.mean(), v2.mean()))
corm_vs_rest.sort(key=lambda x: -x[1])
for fname, abs_d, d, mc, mr in corm_vs_rest[:10]:
    print(f"    {fname:30s} d={d:+.3f}  (Corm={mc:.3f}, Rest={mr:.3f})", flush=True)


# ======================================================================
# D. BIRDS OF PREY: FAST vs SLOW
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("D. BIRDS OF PREY: FAST (>12 m/s) vs SLOW".center(70), flush=True)
print("=" * 70, flush=True)

bop = df[df["bird_group"] == "Birds of Prey"].copy()
# Use the airspeed from the original train data
bop_merged = bop.merge(
    train_df[["track_id", "airspeed"]].rename(columns={"airspeed": "orig_airspeed"}),
    on="track_id"
)
bop_merged["orig_airspeed"] = pd.to_numeric(bop_merged["orig_airspeed"], errors="coerce")
bop_fast = bop_merged[bop_merged["orig_airspeed"] > 12]
bop_slow = bop_merged[bop_merged["orig_airspeed"] <= 12]

print(f"\n  BoP total: {len(bop)}, Fast (>12 m/s): {len(bop_fast)}, Slow: {len(bop_slow)}", flush=True)

if len(bop_fast) > 3 and len(bop_slow) > 3:
    print("\n  Features separating fast BoP from slow BoP:", flush=True)
    bop_scores = []
    for feat_name in feat_cols:
        v1 = bop_fast[feat_name].dropna().values
        v2 = bop_slow[feat_name].dropna().values
        d = cohens_d(v1, v2)
        bop_scores.append((feat_name, abs(d), d, v1.mean(), v2.mean()))
    bop_scores.sort(key=lambda x: -x[1])
    for fname, abs_d, d, mf, ms in bop_scores[:10]:
        print(f"    {fname:30s} d={d:+.3f}  (Fast={mf:.3f}, Slow={ms:.3f})", flush=True)


# ======================================================================
# E. WHICH FEATURES ARE MONTH-INVARIANT?
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("E. TEMPORAL STABILITY OF DEEP FEATURES".center(70), flush=True)
print("=" * 70, flush=True)

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
df["month"] = train_months

# For each feature, compute within-class correlation with month
# Low correlation = month-invariant = good for generalization
print("\n  Features MOST invariant to month (within-class, good for transfer):", flush=True)

month_corrs = []
for feat_name in feat_cols:
    # For each class, compute correlation with month, then average
    class_corrs = []
    for cls in CLASSES:
        mask = df["bird_group"] == cls
        if mask.sum() > 10:
            x = df.loc[mask, feat_name].values
            m = df.loc[mask, "month"].values
            if np.std(x) > 1e-6 and np.std(m) > 1e-6:
                cc = abs(np.corrcoef(x, m)[0, 1])
                if np.isfinite(cc):
                    class_corrs.append(cc)
    avg_corr = np.mean(class_corrs) if class_corrs else 1.0
    month_corrs.append((feat_name, avg_corr))

month_corrs.sort(key=lambda x: x[1])
print("\n  MOST month-invariant (best for generalization):", flush=True)
for fname, corr in month_corrs[:15]:
    print(f"    {fname:30s} avg |corr| with month = {corr:.4f}", flush=True)

print("\n  MOST month-dependent (risky for generalization):", flush=True)
for fname, corr in month_corrs[-10:]:
    print(f"    {fname:30s} avg |corr| with month = {corr:.4f}", flush=True)


# ======================================================================
# F. COMBINED SCORE: DISCRIMINATIVE + MONTH-INVARIANT
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("F. BEST FEATURES: DISCRIMINATIVE & MONTH-INVARIANT".center(70), flush=True)
print("=" * 70, flush=True)

# Score each feature: high discriminative power + low month dependence
# Discriminative = average absolute Cohen's d across problem pairs
disc_scores = {}
for feat_name in feat_cols:
    pair_ds = []
    for c1, c2 in pairs:
        g1 = df[df["bird_group"] == c1][feat_name].dropna().values
        g2 = df[df["bird_group"] == c2][feat_name].dropna().values
        d = abs(cohens_d(g1, g2))
        pair_ds.append(d)
    disc_scores[feat_name] = np.mean(pair_ds)

month_corr_dict = dict(month_corrs)

print("\n  Rank by: discriminative_power * (1 - month_correlation)", flush=True)
combined = []
for feat_name in feat_cols:
    disc = disc_scores.get(feat_name, 0)
    mcorr = month_corr_dict.get(feat_name, 1.0)
    score = disc * (1.0 - mcorr)
    combined.append((feat_name, score, disc, mcorr))

combined.sort(key=lambda x: -x[1])
print(f"\n  {'Feature':30s} {'Score':>8s} {'Discrim':>8s} {'Mo.corr':>8s}", flush=True)
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}", flush=True)
for fname, score, disc, mcorr in combined[:20]:
    print(f"  {fname:30s} {score:8.4f} {disc:8.4f} {mcorr:8.4f}", flush=True)


# ======================================================================
# G. SUMMARY
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("G. KEY FINDINGS & RECOMMENDATIONS".center(70), flush=True)
print("=" * 70, flush=True)

# Top 5 features by combined score
top5 = [x[0] for x in combined[:5]]
print(f"\n  TOP 5 deep features (discriminative + month-invariant):", flush=True)
for i, fname in enumerate(top5, 1):
    disc = disc_scores[fname]
    mcorr = month_corr_dict[fname]
    print(f"    {i}. {fname}: discrim={disc:.3f}, month_corr={mcorr:.3f}", flush=True)

# What separates Cormorants?
print(f"\n  Best features for Cormorants vs Gulls:", flush=True)
for fname, abs_d, d, mc, mg in corm_vs_gull[:5]:
    mcorr = month_corr_dict.get(fname, 0)
    print(f"    {fname}: d={d:+.3f}, month_corr={mcorr:.3f}", flush=True)

print("\nDone.", flush=True)
