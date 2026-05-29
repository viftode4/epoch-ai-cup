"""Proper physics-based Cormorant analysis using REAL radar ornithology knowledge.

Key findings from literature:
1. CORNER REFLECTOR EFFECT: During flapping, wings+body form corner reflector -> 10 dB RCS boost.
   During gliding, this effect DISAPPEARS -> 0 dB boost. (IEEE 2019)
   At 1 Hz: continuous flappers (Cormorant) should have CONSISTENTLY HIGH RCS modulation.
   Flap-gliders (Gull) should alternate between high and low modulation.

2. Cormorant wingbeat ~3-4 Hz. At 1 Hz sampling -> aliased. But the MODULATION DEPTH
   is still visible: high modulation every sample (continuous flap) vs intermittent (flap-glide).

3. Cormorants have HIGHEST flight cost -> shortest flights (mean 92s), can't soar/glide.

4. The radar classifies bird_size from RCS. A Cormorant appearing as "Small bird" means
   the radar saw it at an unfavorable aspect angle or during a specific wing phase.
   The KEY: a "Small bird" that has DEEP RCS modulation (10dB swings) is likely a large bird
   at unfavorable aspect, NOT actually a small bird.

NEW APPROACH: Model the CORNER REFLECTOR effect directly.
- Compute per-point: is this point a "corner reflector peak" (local RCS maximum)?
- Cormorants: peaks at EVERY beat cycle = uniformly distributed peaks
- Gulls: peaks only during flapping phases, gaps during glides
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

train = load_train()
test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
CORM = 2; y_corm = (y == CORM).astype(int)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("Computing physics-based features from radar ornithology literature...", flush=True)
t0 = time.time()

# Approximate radar position
all_pts = [parse_ewkb_4d(row['trajectory'])[0] for _, row in train.head(100).iterrows()]
radar_lon = np.median([p[0] for p in all_pts])
radar_lat = np.median([p[1] for p in all_pts])

def compute_physics_features(row):
    pts = parse_ewkb_4d(row['trajectory'])
    times = parse_trajectory_time(row['trajectory_time'])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    n = len(pts)
    f = {}

    if n < 6:
        return {k: 0 for k in [
            'corner_reflector_consistency', 'rcs_modulation_depth_consistency',
            'rcs_peak_regularity', 'rcs_peak_spacing_cv', 'rcs_trough_spacing_cv',
            'modulation_depth_mean', 'modulation_depth_cv',
            'aspect_angle_mean', 'aspect_angle_range', 'rcs_vs_aspect_corr',
            'glide_gap_fraction', 'max_glide_gap', 'flap_burst_regularity',
            'size_inconsistency', 'rcs_for_declared_size',
            'alt_gain_total', 'alt_gain_rate', 'energy_proxy',
            'flight_efficiency', 'combined_corm_score',
        ]}

    dists = np.array([haversine(lons[j], lats[j], lons[j+1], lats[j+1]) for j in range(n-1)])
    dt = np.maximum(np.diff(times), 0.001)
    speeds = dists / dt
    dur = times[-1] - times[0]

    # ═══ CORNER REFLECTOR ANALYSIS ═══
    # At each point, compute local RCS modulation (how much RCS changes)
    # Corner reflector effect: during flap, RCS swings by ~10dB
    # During glide, RCS is stable (modulation < 2dB)
    rcs_diff = np.abs(np.diff(rcs))

    # Define "active modulation" = |dRCS| > 2 dB (corner reflector active)
    # Define "quiet" = |dRCS| < 1 dB (possible glide, no corner reflector)
    active = rcs_diff > 2.0
    quiet = rcs_diff < 1.0

    # 1. Corner reflector consistency: fraction of time modulation is active
    f['corner_reflector_consistency'] = np.mean(active)

    # 2. Modulation depth consistency: how UNIFORM is the modulation depth?
    # For continuous flappers: modulation depth should be CONSISTENT
    # For flap-gliders: modulation depth alternates between high (flap) and near-zero (glide)
    if n >= 8:
        window = 5
        local_mod = np.array([np.mean(rcs_diff[j:j+window]) for j in range(len(rcs_diff) - window + 1)])
        f['rcs_modulation_depth_consistency'] = 1.0 - (np.std(local_mod) / max(np.mean(local_mod), 0.001))
    else:
        f['rcs_modulation_depth_consistency'] = 0

    # 3. RCS peak analysis: find local maxima in RCS signal
    # Peaks = moments when wing is at maximum corner reflector angle
    peaks = []
    troughs = []
    for j in range(1, n-1):
        if rcs[j] > rcs[j-1] and rcs[j] > rcs[j+1]:
            peaks.append(j)
        elif rcs[j] < rcs[j-1] and rcs[j] < rcs[j+1]:
            troughs.append(j)

    # Peak spacing regularity (uniform for continuous flappers)
    if len(peaks) >= 3:
        peak_spacings = np.diff(peaks)
        f['rcs_peak_regularity'] = 1.0 - (np.std(peak_spacings) / max(np.mean(peak_spacings), 0.001))
        f['rcs_peak_spacing_cv'] = np.std(peak_spacings) / max(np.mean(peak_spacings), 0.001)
    else:
        f['rcs_peak_regularity'] = 0
        f['rcs_peak_spacing_cv'] = 0

    if len(troughs) >= 3:
        trough_spacings = np.diff(troughs)
        f['rcs_trough_spacing_cv'] = np.std(trough_spacings) / max(np.mean(trough_spacings), 0.001)
    else:
        f['rcs_trough_spacing_cv'] = 0

    # Modulation depth: peak-to-trough amplitude
    if len(peaks) > 0 and len(troughs) > 0:
        mod_depths = []
        for p in peaks:
            # Find nearest trough
            nearest_t = troughs[np.argmin(np.abs(np.array(troughs) - p))]
            mod_depths.append(abs(rcs[p] - rcs[nearest_t]))
        mod_depths = np.array(mod_depths)
        f['modulation_depth_mean'] = np.mean(mod_depths)
        f['modulation_depth_cv'] = np.std(mod_depths) / max(np.mean(mod_depths), 0.001)
    else:
        f['modulation_depth_mean'] = 0
        f['modulation_depth_cv'] = 0

    # ═══ ASPECT ANGLE ANALYSIS ═══
    # Bird heading from trajectory
    bird_heading = np.arctan2(np.diff(lons), np.diff(lats))
    # Bearing from radar to bird
    bearing_to_bird = np.arctan2(lons[:-1] - radar_lon, lats[:-1] - radar_lat)
    # Aspect angle = difference (0 = flying toward radar, pi/2 = broadside)
    aspect = bird_heading - bearing_to_bird
    aspect = (aspect + np.pi) % (2*np.pi) - np.pi

    f['aspect_angle_mean'] = np.mean(np.abs(aspect))
    f['aspect_angle_range'] = np.ptp(np.abs(aspect))

    # RCS vs aspect angle correlation
    # Corner reflector effect: RCS should peak at specific aspect angles
    min_l = min(len(rcs) - 1, len(aspect))
    rcs_seg = 0.5 * (rcs[:-1] + rcs[1:])[:min_l]
    if min_l > 3:
        corr = np.corrcoef(np.abs(aspect[:min_l]), rcs_seg)[0, 1]
        f['rcs_vs_aspect_corr'] = corr if not np.isnan(corr) else 0
    else:
        f['rcs_vs_aspect_corr'] = 0

    # ═══ GLIDE GAP DETECTION ═══
    # Find consecutive "quiet" periods (potential glides)
    quiet_runs = []
    current = 0
    for q in quiet:
        if q:
            current += 1
        else:
            if current > 0:
                quiet_runs.append(current)
            current = 0
    if current > 0:
        quiet_runs.append(current)

    if quiet_runs:
        f['glide_gap_fraction'] = sum(quiet_runs) / len(rcs_diff)
        f['max_glide_gap'] = max(quiet_runs)
    else:
        f['glide_gap_fraction'] = 0
        f['max_glide_gap'] = 0

    # Flap burst regularity: spacing between active modulation bursts
    active_starts = []
    in_active = False
    for j, a in enumerate(active):
        if a and not in_active:
            active_starts.append(j)
            in_active = True
        elif not a:
            in_active = False
    if len(active_starts) >= 3:
        burst_spacings = np.diff(active_starts)
        f['flap_burst_regularity'] = 1.0 - np.std(burst_spacings) / max(np.mean(burst_spacings), 0.001)
    else:
        f['flap_burst_regularity'] = 0

    # ═══ SIZE INCONSISTENCY ═══
    # "Small bird" with high RCS modulation = likely large bird at bad aspect angle
    size_map = {'Small bird': 1, 'Medium bird': 2, 'Large bird': 3, 'Flock': 4}
    declared_size = size_map.get(row['radar_bird_size'], 2)
    # RCS modulation depth suggests actual bird size
    implied_size = 1 + min(3, f['modulation_depth_mean'] / 5.0)  # rough mapping
    f['size_inconsistency'] = abs(implied_size - declared_size)
    f['rcs_for_declared_size'] = np.mean(rcs) - [-35, -30, -25, -20][declared_size - 1]  # RCS relative to expected for size

    # ═══ ENERGY / FLIGHT COST ═══
    # Cormorants: highest flight cost -> specific speed-altitude relationship
    f['alt_gain_total'] = alts[-1] - alts[0]
    f['alt_gain_rate'] = f['alt_gain_total'] / max(dur, 0.001)
    # Energy proxy: speed^3 * duration (proportional to flight energy)
    f['energy_proxy'] = np.mean(speeds ** 3) if len(speeds) > 0 else 0
    # Flight efficiency: distance / energy
    total_dist = np.sum(dists)
    f['flight_efficiency'] = total_dist / max(f['energy_proxy'] * dur, 0.001)

    # ═══ COMBINED CORMORANT SCORE ═══
    # Weight the physics signals
    f['combined_corm_score'] = (
        f['corner_reflector_consistency'] * 2.0 +  # continuous flapping
        f['rcs_modulation_depth_consistency'] * 2.0 +  # uniform modulation
        (1.0 - f['glide_gap_fraction']) * 1.5 +  # no glide gaps
        f['size_inconsistency'] * 1.0 +  # RCS-size mismatch
        f['rcs_peak_regularity'] * 1.0  # regular peaks
    ) / 7.5

    return f

# Compute for all training samples
feats_list = []
for i, (_, row) in enumerate(train.iterrows()):
    if i % 500 == 0:
        print(f"  {i}/2601", flush=True)
    feats_list.append(compute_physics_features(row))

df = pd.DataFrame(feats_list)
print(f"\n{len(df.columns)} features in {time.time()-t0:.0f}s\n", flush=True)

# ═══ RANK ALL FEATURES ═══
print("=" * 90, flush=True)
print("PHYSICS-BASED FEATURE RANKING (Cormorant detection)", flush=True)
print("=" * 90, flush=True)

GULL_IDX = CLASSES.index("Gulls")
results = []
for col in df.columns:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    vals = np.clip(vals, -1e6, 1e6)
    try:
        ap_p = average_precision_score(y_corm, vals)
        ap_n = average_precision_score(y_corm, -vals)
        if ap_n > ap_p: ap = ap_n; d = "lower"
        else: ap = ap_p; d = "higher"
        auc = roc_auc_score(y_corm, vals if d == "higher" else -vals)
    except: ap = 0.015; auc = 0.5; d = "?"
    cm = np.median(vals[y == CORM])
    gm = np.median(vals[y == GULL_IDX])
    results.append((col, ap, auc, d, cm, gm))

results.sort(key=lambda x: -x[1])
print(f"\n{'Feature':40s} {'AP':>7s} {'AUC':>6s} {'Dir':>7s} {'Corm':>10s} {'Gull':>10s}")
print("-" * 90)
for col, ap, auc, d, cm, gm in results:
    m = "***" if ap > 0.06 else "**" if ap > 0.04 else "*" if ap > 0.025 else ""
    print(f"  {col:40s} {ap:7.4f} {auc:6.3f} {d:>7s} {cm:10.4f} {gm:10.4f} {m}")

# ═══ PER-CLASS AUC ═══
print(f"\n{'Feature':40s}", end="")
for c in CLASSES: print(f" {c[:6]:>7s}", end="")
print()
print("-" * (40 + 8*9))
for col, ap, _, d, _, _ in results[:12]:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    vals = np.clip(vals, -1e6, 1e6)
    if d == "lower": vals = -vals
    print(f"  {col:40s}", end="")
    for c in range(len(CLASSES)):
        try:
            a = roc_auc_score((y==c).astype(int), vals)
            if a < 0.5: a = 1-a
            print(f" {a:7.3f}", end="")
        except: print(f" {'N/A':>7s}", end="")
    print()

# ═══ COMBINED: add physics features to existing and test ═══
print(f"\n\n--- Combined: physics + existing features in 9-class LGB ---", flush=True)
import lightgbm as lgb

existing = pd.read_pickle("G:/Projects/epoch-ai-cup/data/_cached_train_features_v3.pkl")
X_existing = np.nan_to_num(existing.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_physics = np.nan_to_num(df.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_physics = np.clip(X_physics, -1e6, 1e6)
X_combined = np.column_stack([X_existing, X_physics])

clf_base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                                random_state=42, verbose=-1, n_jobs=1)
probs_base = cross_val_predict(clf_base, X_existing, y, cv=sgkf, method="predict_proba", groups=groups)
map_base, pc_base = compute_map(y, probs_base)

clf_comb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                                random_state=42, verbose=-1, n_jobs=1)
probs_comb = cross_val_predict(clf_comb, X_combined, y, cv=sgkf, method="predict_proba", groups=groups)
map_comb, pc_comb = compute_map(y, probs_comb)

from src.metrics import compute_map
print(f"  Baseline (327 feat):  mAP={map_base:.4f} Corm={pc_base['Cormorants']:.4f} Wader={pc_base['Waders']:.4f}")
print(f"  + Physics ({len(df.columns)} new): mAP={map_comb:.4f} Corm={pc_comb['Cormorants']:.4f} Wader={pc_comb['Waders']:.4f}")
print(f"  Delta: mAP={map_comb-map_base:+.4f} Corm={pc_comb['Cormorants']-pc_base['Cormorants']:+.4f}")

# Also per-class comparison
print(f"\n  {'Class':15s} {'Base':>8s} {'+ Physics':>10s} {'Delta':>8s}")
for cls in CLASSES:
    d = pc_comb[cls] - pc_base[cls]
    m = " ***" if abs(d) > 0.02 else " **" if abs(d) > 0.01 else ""
    print(f"  {cls:15s} {pc_base[cls]:8.4f} {pc_comb[cls]:10.4f} {d:+8.4f}{m}")

print("\nDone.", flush=True)
