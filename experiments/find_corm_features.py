"""Find features that ACTUALLY separate Cormorants from everything else.
Compute creative trajectory features and rank by Cormorant separation."""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from sklearn.metrics import average_precision_score, roc_auc_score

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
CORM = 2; y_corm = (y == CORM).astype(int)

print("Computing creative trajectory features...", flush=True)
t0 = time.time()

feats = []
for i, (_, row) in enumerate(train.iterrows()):
    if i % 500 == 0: print(f"  {i}/2601", flush=True)
    pts = parse_ewkb_4d(row['trajectory'])
    times = parse_trajectory_time(row['trajectory_time'])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    rcs_lin = 10 ** (rcs / 10.0)
    n = len(pts)
    f = {}

    if n > 4:
        dists = np.array([haversine(lons[j], lats[j], lons[j+1], lats[j+1]) for j in range(n-1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists / dt
        climb = np.diff(alts) / dt
        dur = times[-1] - times[0]

        # ── ALTITUDE ──
        f['alt_std'] = np.std(alts)
        f['alt_flat_frac_5m'] = np.mean(np.abs(alts - np.mean(alts)) < 5)
        f['alt_flat_frac_10m'] = np.mean(np.abs(alts - np.mean(alts)) < 10)
        f['alt_residual'] = np.std(alts - np.polyval(np.polyfit(np.arange(n), alts, 1), np.arange(n))) if n > 3 else 0
        f['max_alt_jump'] = np.max(np.abs(np.diff(alts)))
        f['climb_std'] = np.std(climb)
        f['climb_abs_mean'] = np.mean(np.abs(climb))

        # ── SPEED ──
        f['speed_mean'] = np.mean(speeds)
        f['speed_std'] = np.std(speeds)
        f['speed_cv'] = np.std(speeds) / max(np.mean(speeds), 0.001)
        med_spd = np.median(speeds)
        f['speed_steady_20pct'] = np.mean(np.abs(speeds - med_spd) < med_spd * 0.2) if med_spd > 1 else 0
        f['speed_steady_10pct'] = np.mean(np.abs(speeds - med_spd) < med_spd * 0.1) if med_spd > 1 else 0
        if len(speeds) > 2:
            accel = np.diff(speeds)
            f['accel_std'] = np.std(accel)
            f['jerk_std'] = np.std(np.diff(accel)) if len(accel) > 1 else 0
        else:
            f['accel_std'] = 0; f['jerk_std'] = 0

        # ── COMBINED STEADINESS ──
        f['speed_x_alt_std'] = f['speed_std'] * f['alt_std']
        f['combined_steady'] = f['speed_steady_20pct'] * f['alt_flat_frac_10m']
        f['flight_smoothness'] = 1.0 / (1 + f['accel_std'] + f['climb_std'])

        # ── 3D TRAJECTORY SHAPE ──
        total_3d = np.sum(np.sqrt(dists**2 + np.diff(alts)**2))
        disp_2d = haversine(lons[0], lats[0], lons[-1], lats[-1])
        disp_3d = np.sqrt(disp_2d**2 + (alts[-1] - alts[0])**2)
        f['straight_2d'] = disp_2d / max(np.sum(dists), 0.001)
        f['straight_3d'] = disp_3d / max(total_3d, 0.001)

        # Bearing analysis
        dlons = np.diff(lons); dlats = np.diff(lats)
        bearings = np.arctan2(dlons, dlats)
        if len(bearings) > 1:
            db = np.diff(bearings)
            db = (db + np.pi) % (2 * np.pi) - np.pi
            f['bearing_std'] = np.std(db)
            f['bearing_abs_mean'] = np.mean(np.abs(db))
            f['total_turn'] = np.sum(np.abs(db))
            f['turn_per_meter'] = f['total_turn'] / max(np.sum(dists), 0.001)
            f['turn_per_sec'] = f['total_turn'] / max(dur, 0.001)
            # Torsion: combined horizontal + vertical twisting
            min_l = min(len(db), len(climb) - 1)
            if min_l > 0:
                f['torsion'] = np.mean(np.sqrt(db[:min_l]**2 + np.diff(climb[:min_l+1])[:min_l]**2))
            else:
                f['torsion'] = 0
        else:
            f['bearing_std'] = 0; f['bearing_abs_mean'] = 0
            f['total_turn'] = 0; f['turn_per_meter'] = 0; f['turn_per_sec'] = 0; f['torsion'] = 0

        # ── RCS BEHAVIOR (not magnitude!) ──
        rcs_c = rcs - np.mean(rcs)
        rv = np.var(rcs)
        for lag in [1, 2, 3, 4, 5]:
            if n > lag + 1 and rv > 0:
                f[f'rcs_ac{lag}'] = np.sum(rcs_c[:-lag] * rcs_c[lag:]) / (rv * (n - lag))
            else:
                f[f'rcs_ac{lag}'] = 0

        rcs_diff = np.diff(rcs)
        f['rcs_diff_std'] = np.std(rcs_diff)
        f['rcs_diff_abs_mean'] = np.mean(np.abs(rcs_diff))
        if len(rcs_diff) > 1:
            f['rcs_reversal_rate'] = np.mean(np.diff(np.sign(rcs_diff)) != 0)
        else:
            f['rcs_reversal_rate'] = 0

        # RCS regularity: how periodic is the signal?
        signs = np.sign(rcs_c)
        runs = []
        current = 1
        for j in range(1, len(signs)):
            if signs[j] == signs[j-1]:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        runs = np.array(runs)
        f['rcs_run_mean'] = np.mean(runs)
        f['rcs_run_std'] = np.std(runs)
        f['rcs_run_regularity'] = np.std(runs) / max(np.mean(runs), 0.001)

        # RCS on linear scale
        f['rcs_lin_cv'] = np.std(rcs_lin) / max(np.mean(rcs_lin), 1e-10)
        f['scintillation'] = np.var(rcs_lin) / max(np.mean(rcs_lin)**2, 1e-20)
        f['k_factor'] = np.mean(rcs_lin)**2 / max(2 * np.var(rcs_lin), 1e-20)

        # RCS envelope stability in windows
        if n >= 10:
            win_ranges = [np.ptp(rcs[j:j+10]) for j in range(n - 9)]
            f['rcs_envelope_cv'] = np.std(win_ranges) / max(np.mean(win_ranges), 0.001)
        else:
            f['rcs_envelope_cv'] = 0

        # ── COUPLING ──
        min_l = min(len(speeds), len(climb))
        if min_l > 3:
            c_val = np.corrcoef(speeds[:min_l], climb[:min_l])[0, 1]
            f['speed_climb_corr'] = c_val if not np.isnan(c_val) else 0
        else:
            f['speed_climb_corr'] = 0

        # RCS-speed coupling
        min_l2 = min(n - 1, len(speeds))
        rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])[:min_l2]
        if min_l2 > 3:
            c_val = np.corrcoef(rcs_mid, speeds[:min_l2])[0, 1]
            f['rcs_speed_corr'] = c_val if not np.isnan(c_val) else 0
        else:
            f['rcs_speed_corr'] = 0

        # Cruise flap detection: speed stable + RCS varying = continuous powered flight
        if min_l2 > 3:
            spd_stable = np.abs(np.diff(speeds[:min_l2])) < 1.5
            rcs_active = np.abs(np.diff(rcs_mid)) > 1.5
            n_seg = min(len(spd_stable), len(rcs_active))
            f['cruise_flap_frac'] = np.mean(spd_stable[:n_seg] & rcs_active[:n_seg])
            f['glide_frac'] = np.mean(spd_stable[:n_seg] & ~rcs_active[:n_seg])
        else:
            f['cruise_flap_frac'] = 0; f['glide_frac'] = 0

        # ── COMPOSITE SCORES ──
        f['commuter_score'] = f['straight_2d'] * f['speed_steady_20pct'] * min(dur / 60, 1.0)
        f['powered_straight'] = f['straight_2d'] * f['cruise_flap_frac']
        f['steady_commuter'] = f['flight_smoothness'] * f['straight_3d']
        f['n_points'] = n
        f['duration'] = dur

    else:
        for key in ['alt_std', 'alt_flat_frac_5m', 'alt_flat_frac_10m', 'alt_residual',
                     'max_alt_jump', 'climb_std', 'climb_abs_mean',
                     'speed_mean', 'speed_std', 'speed_cv', 'speed_steady_20pct', 'speed_steady_10pct',
                     'accel_std', 'jerk_std', 'speed_x_alt_std', 'combined_steady', 'flight_smoothness',
                     'straight_2d', 'straight_3d', 'bearing_std', 'bearing_abs_mean',
                     'total_turn', 'turn_per_meter', 'turn_per_sec', 'torsion',
                     'rcs_ac1', 'rcs_ac2', 'rcs_ac3', 'rcs_ac4', 'rcs_ac5',
                     'rcs_diff_std', 'rcs_diff_abs_mean', 'rcs_reversal_rate',
                     'rcs_run_mean', 'rcs_run_std', 'rcs_run_regularity',
                     'rcs_lin_cv', 'scintillation', 'k_factor', 'rcs_envelope_cv',
                     'speed_climb_corr', 'rcs_speed_corr', 'cruise_flap_frac', 'glide_frac',
                     'commuter_score', 'powered_straight', 'steady_commuter',
                     'n_points', 'duration']:
            f[key] = 0
    feats.append(f)

df = pd.DataFrame(feats)
print(f"\nComputed {len(df.columns)} features in {time.time()-t0:.0f}s\n", flush=True)

# ── RANK BY CORMORANT SEPARATION ──
print("=" * 100)
print("CORMORANT SEPARATION RANKING")
print("=" * 100)

results = []
for col in df.columns:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    try:
        ap_pos = average_precision_score(y_corm, vals)
        ap_neg = average_precision_score(y_corm, -vals)
        if ap_neg > ap_pos:
            ap = ap_neg; direction = "lower"
        else:
            ap = ap_pos; direction = "higher"
        auc = roc_auc_score(y_corm, vals if direction == "higher" else -vals)
    except:
        ap = 0.015; auc = 0.5; direction = "?"

    c_med = np.median(vals[y == CORM])
    g_med = np.median(vals[y == 5])
    results.append((col, ap, auc, direction, c_med, g_med))

results.sort(key=lambda x: -x[1])

print(f"\n{'Feature':30s} {'AP':>7s} {'AUC':>6s} {'Dir':>7s} {'Corm_med':>10s} {'Gull_med':>10s}")
print("-" * 80)
for col, ap, auc, d, cm, gm in results:
    marker = "***" if ap > 0.06 else "**" if ap > 0.04 else "*" if ap > 0.025 else ""
    print(f"  {col:30s} {ap:7.4f} {auc:6.3f} {d:>7s} {cm:10.4f} {gm:10.4f} {marker}")

# ── PER-CLASS AUC for top features ──
print(f"\n\n{'Feature':30s}", end="")
for cls in CLASSES:
    print(f" {cls[:6]:>7s}", end="")
print()
print("-" * (30 + 8 * 9))
for col, ap, _, d, _, _ in results[:15]:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    if d == "lower": vals = -vals
    print(f"  {col:30s}", end="")
    for c in range(len(CLASSES)):
        try:
            a = roc_auc_score((y == c).astype(int), vals)
            if a < 0.5: a = 1 - a
            print(f" {a:7.3f}", end="")
        except:
            print(f" {'N/A':>7s}", end="")
    print()

print("\nDone.", flush=True)
