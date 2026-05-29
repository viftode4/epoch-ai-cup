"""E181: Radar Physics Features — Tier 1 + Key Tier 2 from research catalog.

20 new month-invariant features targeting weak classes:
  Cormorants: commuter_score, persistence_length, echo_shape_type
  Waders: rcs_periodicity, undulation_index, flap_glide_ratio
  BoP: torsion, vertical_straightness, altitude_speed_corr
  Clutter: rcs_spectral_entropy, permutation_entropy, nakagami_m
  General: displacement_rate, speed_per_alt, turning_kurtosis,
           track_regularity, rcs_modulation_index, fractal_dim,
           convex_hull_ratio, bounding_score

Plus spatial features (radial_velocity, mean_distance_km, bearing).
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
MONTHS = [1, 4, 9, 10]
RADAR_LON, RADAR_LAT = 6.835, 53.438

print("=" * 90)
print("  E181: Radar Physics Features (20 new + 3 spatial)")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)
t0 = time.time()

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values


def extract_physics_features(row):
    """Extract all 23 new physics features from a single track."""
    feats = {}
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])
    except Exception:
        return {k: 0.0 for k in _FEATURE_NAMES}

    n = len(pts)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs_dB = np.array([p[3] for p in pts])
    rcs_lin = 10 ** (rcs_dB / 10.0)

    # Meters relative to centroid
    x = (lons - lons.mean()) * 67000
    z = (lats - lats.mean()) * 111000
    dt = np.maximum(np.diff(times), 0.001)

    # Speed, heading
    dx = np.diff(x); dz = np.diff(z)
    dist_seg = np.sqrt(dx**2 + dz**2)
    speeds = dist_seg / dt
    headings = np.arctan2(dz, dx)

    # ── 1. Vertical straightness ──
    total_alt_change = np.sum(np.abs(np.diff(alts)))
    net_alt_change = abs(alts[-1] - alts[0]) if n > 1 else 0
    feats["vertical_straightness"] = net_alt_change / max(total_alt_change, 1e-6)

    # ── 2. RCS spectral entropy ──
    if n >= 8:
        from numpy.fft import rfft
        rcs_centered = rcs_lin - rcs_lin.mean()
        psd = np.abs(rfft(rcs_centered))[1:] ** 2
        psd_norm = psd / max(psd.sum(), 1e-12)
        psd_norm = np.clip(psd_norm, 1e-12, None)
        feats["rcs_spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm)))
    else:
        feats["rcs_spectral_entropy"] = 0.0

    # ── 3. Permutation entropy of RCS (order 3) ──
    if n >= 6:
        m = 3
        patterns = {}
        for i in range(n - m + 1):
            pat = tuple(np.argsort(rcs_dB[i:i+m]))
            patterns[pat] = patterns.get(pat, 0) + 1
        total = sum(patterns.values())
        probs = np.array([c / total for c in patterns.values()])
        feats["permutation_entropy_rcs"] = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(6))
    else:
        feats["permutation_entropy_rcs"] = 1.0

    # ── 4. Undulation index ──
    if n >= 5:
        t_norm = np.linspace(0, 1, n)
        slope, intercept = np.polyfit(t_norm, alts, 1)
        alt_detrended = alts - (slope * t_norm + intercept)
        feats["undulation_index"] = float(np.std(alt_detrended) / max(np.mean(speeds) if len(speeds) > 0 else 1, 0.1))
        # Zero crossings of detrended altitude
        zc = np.sum(np.diff(np.sign(alt_detrended)) != 0)
        duration = times[-1] - times[0] if times[-1] > times[0] else 1
        feats["undulation_freq"] = float(zc / max(duration, 1))
    else:
        feats["undulation_index"] = 0.0
        feats["undulation_freq"] = 0.0

    # ── 5. RCS periodicity score ──
    if n >= 8:
        rcs_c = rcs_dB - rcs_dB.mean()
        var = float(np.var(rcs_dB))
        if var > 1e-12:
            ac = np.correlate(rcs_c, rcs_c, mode="full")[n-1:]
            ac = ac / (var * n)
            ac_tail = ac[2:min(n, 20)]
            if len(ac_tail) > 0:
                feats["rcs_periodicity"] = float(np.max(ac_tail) - np.mean(ac_tail))
            else:
                feats["rcs_periodicity"] = 0.0
        else:
            feats["rcs_periodicity"] = 0.0
    else:
        feats["rcs_periodicity"] = 0.0

    # ── 6. Altitude-speed correlation ──
    if n > 2 and len(speeds) > 2:
        alt_for_corr = alts[:-1] if len(alts) > len(speeds) else alts[:len(speeds)]
        if len(alt_for_corr) == len(speeds) and np.std(alt_for_corr) > 0 and np.std(speeds) > 0:
            feats["alt_speed_corr"] = float(np.corrcoef(alt_for_corr, speeds)[0, 1])
        else:
            feats["alt_speed_corr"] = 0.0
    else:
        feats["alt_speed_corr"] = 0.0

    # ── 7. Speed per altitude ──
    feats["speed_per_alt"] = float(np.median(speeds) / max(np.mean(alts), 1)) if len(speeds) > 0 else 0.0

    # ── 8. Displacement rate ──
    total_disp = np.sqrt((x[-1] - x[0])**2 + (z[-1] - z[0])**2) if n > 1 else 0
    duration = max(times[-1] - times[0], 1) if n > 1 else 1
    feats["displacement_rate"] = float(total_disp / duration)

    # ── 9. Nakagami m-parameter ──
    rcs_amp = np.sqrt(rcs_lin)
    var_amp = float(np.var(rcs_amp))
    feats["nakagami_m"] = float(np.mean(rcs_amp)**2 / max(var_amp, 1e-12))

    # ── 10. RCS modulation index ──
    if rcs_lin.max() + rcs_lin.min() > 1e-12:
        feats["rcs_modulation_index"] = float((rcs_lin.max() - rcs_lin.min()) / (rcs_lin.max() + rcs_lin.min()))
    else:
        feats["rcs_modulation_index"] = 0.0

    # ── 11. Turning angle kurtosis ──
    if len(headings) > 3:
        dheading = np.diff(headings)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
        from scipy.stats import kurtosis
        feats["turning_kurtosis"] = float(kurtosis(dheading, fisher=True))
    else:
        feats["turning_kurtosis"] = 0.0

    # ── 12. Track sampling regularity ──
    if n > 2:
        feats["track_regularity"] = float(np.std(dt) / max(np.mean(dt), 0.01))
    else:
        feats["track_regularity"] = 0.0

    # ── 13. Echo shape type (BirdScan-style) ──
    if n >= 8:
        rcs_c = rcs_dB - rcs_dB.mean()
        var = float(np.var(rcs_dB))
        if var > 1e-12:
            ac = np.correlate(rcs_c, rcs_c, mode="full")[n-1:]
            ac = ac / (var * n)
            ac2 = float(ac[2]) if len(ac) > 2 else 0
            ac3 = float(ac[3]) if len(ac) > 3 else 0
            ac4 = float(ac[4]) if len(ac) > 4 else 0
            if ac2 > 0.3 and ac3 > 0.2:
                feats["echo_shape"] = 2.0  # continuous wingbeat
            elif ac2 < 0.1 and ac4 > 0.15:
                feats["echo_shape"] = 1.0  # intermittent
            else:
                feats["echo_shape"] = 0.0  # none/gliding
        else:
            feats["echo_shape"] = 0.0
    else:
        feats["echo_shape"] = 0.0

    # ── 14. Torsion (3D path twist) ──
    if n >= 5:
        # Velocity vectors
        v = np.column_stack([np.diff(x), np.diff(z), np.diff(alts)])
        # Acceleration
        a = np.diff(v, axis=0)
        # Jerk
        if len(a) >= 2:
            j = np.diff(a, axis=0)
            torsions = []
            for i in range(min(len(j), len(v)-2)):
                cross_va = np.cross(v[i], a[i])
                cross_norm_sq = np.dot(cross_va, cross_va)
                if cross_norm_sq > 1e-12:
                    tau = np.dot(cross_va, j[i]) / cross_norm_sq
                    if np.isfinite(tau):
                        torsions.append(tau)
            feats["torsion_median"] = float(np.median(torsions)) if torsions else 0.0
            feats["torsion_std"] = float(np.std(torsions)) if torsions else 0.0
        else:
            feats["torsion_median"] = 0.0
            feats["torsion_std"] = 0.0
    else:
        feats["torsion_median"] = 0.0
        feats["torsion_std"] = 0.0

    # ── 15. Fractal dimension (dividers method) ──
    if n >= 20:
        path_xy = np.column_stack([x, z])
        total_path = np.sum(dist_seg)
        divider_lengths = [5, 10, 20, 50, 100, 200]
        log_N = []; log_L = []
        for L in divider_lengths:
            if L > total_path * 0.8:
                continue
            steps = 0; pos = 0; current = 0
            while current < n - 1:
                remaining = L
                while remaining > 0 and current < n - 1:
                    seg_len = dist_seg[current] if current < len(dist_seg) else 0
                    if seg_len <= remaining:
                        remaining -= seg_len
                        current += 1
                    else:
                        current += 1
                        break
                steps += 1
            if steps > 1:
                log_N.append(np.log(steps))
                log_L.append(np.log(L))
        if len(log_N) >= 3:
            slope = np.polyfit(log_L, log_N, 1)[0]
            feats["fractal_dim"] = float(1 - slope)
        else:
            feats["fractal_dim"] = 1.0
    else:
        feats["fractal_dim"] = 1.0

    # ── 16. Convex hull ratio ──
    if n >= 4:
        try:
            points_2d = np.column_stack([x, z])
            # Check not collinear
            if np.linalg.matrix_rank(points_2d - points_2d.mean(axis=0)) >= 2:
                hull = ConvexHull(points_2d)
                total_path_len = np.sum(dist_seg) if len(dist_seg) > 0 else 1
                feats["hull_ratio"] = float(hull.area / max(total_path_len**2, 1e-6))
            else:
                feats["hull_ratio"] = 0.0
        except Exception:
            feats["hull_ratio"] = 0.0
    else:
        feats["hull_ratio"] = 0.0

    # ── 17. Bounding flight score ──
    alt_curv = float(np.mean(np.abs(np.diff(alts, n=2)))) if n >= 3 else 0.0
    speed_cv = float(np.std(speeds) / max(np.mean(speeds), 0.1)) if len(speeds) > 1 else 0.0
    rcs_ac1 = 0.0
    if n >= 4:
        rcs_c = rcs_dB - rcs_dB.mean()
        var = float(np.var(rcs_dB))
        if var > 1e-12:
            rcs_ac1 = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var)
    feats["bounding_score"] = float(alt_curv * speed_cv * max(1 - rcs_ac1, 0))

    # ── 18. Commuter score (Cormorant detector) ──
    straightness_2d = total_disp / max(np.sum(dist_seg), 1e-6) if len(dist_seg) > 0 else 0
    heading_std = float(np.std(headings)) if len(headings) > 1 else 1.0
    feats["commuter_score"] = float(straightness_2d * (1 - speed_cv) / max(heading_std, 0.01))

    # ── 19. Flap-glide ratio (windowed RCS variance) ──
    if n >= 10:
        window = 5
        vars_w = []
        for i in range(0, n - window + 1, 2):
            w = rcs_lin[i:i+window]
            vars_w.append(np.var(w))
        if vars_w:
            median_var = np.median(vars_w)
            flapping = sum(1 for v in vars_w if v > 2 * median_var)
            feats["flap_glide_ratio"] = float(flapping / max(len(vars_w), 1))
        else:
            feats["flap_glide_ratio"] = 0.5
    else:
        feats["flap_glide_ratio"] = 0.5

    # ── 20. RCS spectral flatness ──
    if n >= 8:
        from numpy.fft import rfft
        rcs_centered = rcs_lin - rcs_lin.mean()
        psd = np.abs(rfft(rcs_centered))[1:] ** 2
        psd = np.clip(psd, 1e-12, None)
        geo_mean = np.exp(np.mean(np.log(psd)))
        arith_mean = np.mean(psd)
        feats["rcs_spectral_flatness"] = float(geo_mean / max(arith_mean, 1e-12))
    else:
        feats["rcs_spectral_flatness"] = 1.0

    # ── SPATIAL: 3 features ──
    mean_lon = float(np.mean(lons))
    mean_lat = float(np.mean(lats))
    dx_radar = (mean_lon - RADAR_LON) * 67000
    dz_radar = (mean_lat - RADAR_LAT) * 111000
    dist_km = np.sqrt(dx_radar**2 + dz_radar**2) / 1000
    bearing = float(np.degrees(np.arctan2(dx_radar, dz_radar))) % 360

    feats["sp_distance_km"] = dist_km
    feats["sp_bearing"] = bearing
    # Radial velocity: speed component toward/away from radar
    if n > 1:
        track_dx = x[-1] - x[0]
        track_dz = z[-1] - z[0]
        track_dist = np.sqrt(track_dx**2 + track_dz**2)
        if track_dist > 1:
            radar_dir = np.array([dx_radar, dz_radar])
            radar_dir = radar_dir / max(np.linalg.norm(radar_dir), 1e-6)
            track_dir = np.array([track_dx, track_dz]) / track_dist
            feats["sp_radial_velocity"] = float(np.dot(track_dir, radar_dir) * np.median(speeds))
        else:
            feats["sp_radial_velocity"] = 0.0
    else:
        feats["sp_radial_velocity"] = 0.0

    # Replace any NaN/Inf
    for k in feats:
        if not np.isfinite(feats[k]):
            feats[k] = 0.0

    return feats


_FEATURE_NAMES = [
    "vertical_straightness", "rcs_spectral_entropy", "permutation_entropy_rcs",
    "undulation_index", "undulation_freq", "rcs_periodicity", "alt_speed_corr",
    "speed_per_alt", "displacement_rate", "nakagami_m", "rcs_modulation_index",
    "turning_kurtosis", "track_regularity", "echo_shape", "torsion_median",
    "torsion_std", "fractal_dim", "hull_ratio", "bounding_score", "commuter_score",
    "flap_glide_ratio", "rcs_spectral_flatness",
    "sp_distance_km", "sp_bearing", "sp_radial_velocity",
]


# ── Extract features ──
print(f"  Extracting {len(_FEATURE_NAMES)} new features...", flush=True)


def extract_all(df, label=""):
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append(extract_physics_features(row))
        if (i + 1) % 500 == 0:
            print(f"    {label} {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


t_ext = time.time()
new_train = extract_all(train_df, "Train")
new_test = extract_all(test_df, "Test")
print(f"  Extraction: {time.time()-t_ext:.0f}s, {new_train.shape[1]} features", flush=True)

# ── Combine with base 100 features ──
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

X_base_tr = train_feats[selected].values.astype(np.float32)
X_base_te = test_feats[selected].values.astype(np.float32)
X_new_tr = new_train.values.astype(np.float32)
X_new_te = new_test.values.astype(np.float32)

X_train = np.hstack([X_base_tr, X_new_tr])
X_test = np.hstack([X_base_te, X_new_te])
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
all_names = selected + list(new_train.columns)

print(f"  Total features: {X_train.shape[1]} ({len(selected)} base + {X_new_tr.shape[1]} new)")


def true_lomo(oof, name=""):
    skf, skf_pc = compute_map(y, oof)
    scores = {}
    for m in MONTHS:
        mask = train_months == m
        s, _ = compute_map(y[mask], oof[mask])
        scores[m] = s
    lomo = np.mean(list(scores.values()))
    ms = " ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()))
    print(f"  {name:<50s} SKF={skf:.4f} LOMO={lomo:.4f} [{ms}]", flush=True)
    return skf, lomo, skf_pc


# ── Train LGB GBDT (fast, 5 seeds) ──
import lightgbm as lgb

print(f"\n  Training LGB GBDT (5 seeds)...", flush=True)
N_SEEDS = 5
oof_all = np.zeros((N_SEEDS, len(y), N_CLASSES))
test_all = np.zeros((N_SEEDS, len(test_df), N_CLASSES))

for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+seed)
    oof_s = np.zeros((len(y), N_CLASSES))
    test_s = np.zeros((len(test_df), N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            is_unbalance=True, verbosity=-1, random_state=42+seed+fold, n_jobs=-1,
        )
        m.fit(X_train[tr], y[tr], eval_set=[(X_train[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_s[va] = m.predict_proba(X_train[va])
        test_s += m.predict_proba(X_test) / 5
    oof_all[seed] = oof_s
    test_all[seed] = test_s
    s, _ = compute_map(y, oof_s)
    print(f"    Seed {seed+1}: {s:.4f}", flush=True)

oof_aug = np.mean(oof_all, axis=0)
test_aug = np.mean(test_all, axis=0)

# Also train on base features only for fair comparison
print(f"\n  Training base features only (5 seeds)...", flush=True)
oof_base_all = np.zeros((N_SEEDS, len(y), N_CLASSES))
for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+seed)
    oof_s = np.zeros((len(y), N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_base_tr, y, groups)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            is_unbalance=True, verbosity=-1, random_state=42+seed+fold, n_jobs=-1,
        )
        m.fit(np.nan_to_num(X_base_tr[tr], nan=0.0), y[tr],
              eval_set=[(np.nan_to_num(X_base_tr[va], nan=0.0), y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_s[va] = m.predict_proba(np.nan_to_num(X_base_tr[va], nan=0.0))
    oof_base_all[seed] = oof_s
    s, _ = compute_map(y, oof_s)
    print(f"    Seed {seed+1}: {s:.4f}", flush=True)
oof_base = np.mean(oof_base_all, axis=0)


# ── Results ──
print(f"\n{'='*90}")
print("  RESULTS")
print(f"{'='*90}")

oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))

_, _, pc_base = true_lomo(oof_base, "Base 100 features (GBDT)")
_, _, pc_aug = true_lomo(oof_aug, "Base + 25 physics features (GBDT)")
true_lomo(oof_e175, "E175 best (reference)")

# Per-class comparison
print(f"\n  Per-class AP (base vs augmented):")
print(f"  {'Class':<20s} {'Base':>6s} {'Aug':>6s} {'Delta':>7s}")
for cls in CLASSES:
    d = pc_aug[cls] - pc_base[cls]
    marker = " +" if d > 0.005 else (" -" if d < -0.005 else "  ")
    print(f"  {cls:<20s} {pc_base[cls]:>6.3f} {pc_aug[cls]:>6.3f} {d:>+7.3f}{marker}")

# Feature importance for new features
print(f"\n  New Feature Importance:")
m_imp = lgb.LGBMClassifier(
    objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    is_unbalance=True, verbosity=-1, n_jobs=-1,
)
m_imp.fit(X_train, y)
imp = m_imp.feature_importances_
new_imp = [(all_names[i], imp[i]) for i in range(len(selected), len(all_names))]
new_imp.sort(key=lambda x: -x[1])
for name, importance in new_imp:
    rank = sorted(imp, reverse=True).index(importance) + 1
    print(f"  {name:<30s}: {importance:>6.0f} (rank #{rank}/{len(all_names)})")

# Blend with E175
print(f"\n  Blends with E175:")
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
    blend = renorm_rows((1-alpha) * oof_e175 + alpha * renorm_rows(oof_aug))
    true_lomo(blend, f"E175 + physics@{alpha}")

# Save
np.save(ROOT / "oof_e181_physics.npy", oof_aug)
np.save(ROOT / "test_e181_physics.npy", test_aug)
save_submission(renorm_rows(test_aug), "e181_physics_raw", cv_map=compute_map(y, oof_aug)[0])

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed/60:.1f} min")
print(f"{'='*90}")
