"""Analysis: micro-pattern features from raw radar signals."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train
from src.features import parse_ewkb_4d, parse_trajectory_time
from sklearn.preprocessing import LabelEncoder

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
oof = np.load("oof_e162b.npy")
pred_labels = np.argmax(oof, axis=1)

class_data = {cls: {"correct": [], "wrong": [], "all": []} for cls in CLASSES}

for idx, r in train_df.iterrows():
    pts = parse_ewkb_4d(r.trajectory)
    times = parse_trajectory_time(r.trajectory_time)
    n = len(pts)
    if n < 5:
        continue
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    lats = np.array([p[1] for p in pts])
    lons = np.array([p[0] for p in pts])
    dt = np.diff(times); dt = np.maximum(dt, 0.01)
    dlat = np.diff(lats)*111000; dlon = np.diff(lons)*67000
    dists = np.sqrt(dlat**2+dlon**2)
    speeds = dists/dt
    bearings = np.arctan2(dlat, dlon)
    dbearing = np.diff(bearings) if len(bearings)>1 else np.array([0])
    dbearing = np.arctan2(np.sin(dbearing), np.cos(dbearing))
    vert_speed = np.diff(alts)/dt
    duration = times[-1]-times[0]

    # 1. Speed dip/surge fraction (local anomalies)
    if len(speeds) > 5:
        win = 5
        local_avg = np.convolve(speeds, np.ones(win)/win, mode="same")
        speed_dip_frac = float(np.mean(speeds < 0.7 * local_avg))
        speed_surge_frac = float(np.mean(speeds > 1.3 * local_avg))
    else:
        speed_dip_frac = 0.0; speed_surge_frac = 0.0

    # 2. RCS modulation rate (zero-crossings per second)
    rcs_centered = rcs - np.mean(rcs)
    rcs_zero_cross = np.sum(np.diff(np.sign(rcs_centered)) != 0)
    rcs_mod_rate = float(rcs_zero_cross / (duration + 1e-10))

    # 3. Altitude stability
    alt_stable_frac = float(np.mean(np.abs(alts - np.mean(alts)) < 5))

    # 4. Segment flight modes
    if len(speeds) > 3 and len(dbearing) > 0:
        turn_segs = np.abs(dbearing) > np.radians(15)
        turn_frac = float(np.mean(turn_segs))
        min_l2 = min(len(speeds)-1, len(vert_speed)-1)
        if min_l2 > 0:
            speed_dec = np.diff(speeds[:min_l2+1]) < 0
            alt_dec = vert_speed[:min_l2+1] < -0.5
            glide_frac = float(np.mean(speed_dec[:min_l2] & alt_dec[:min_l2]))
            speed_inc = np.diff(speeds[:min_l2+1]) >= 0
            alt_up = vert_speed[:min_l2+1] >= -0.5
            powered_frac = float(np.mean(speed_inc[:min_l2] & alt_up[:min_l2]))
        else:
            glide_frac = 0.0; powered_frac = 0.0
    else:
        turn_frac = 0.0; glide_frac = 0.0; powered_frac = 0.0

    # 5. Bounding flight (speed sign alternation rate)
    if len(speeds) > 4:
        speed_sign = np.sign(np.diff(speeds))
        bounding = float(np.mean(np.diff(speed_sign) != 0))
    else:
        bounding = 0.0

    # 6. RCS half ratio (body orientation shift)
    half = n // 2
    rcs_half_ratio = float(np.std(rcs[:half]) / (np.std(rcs[half:]) + 1e-10))

    # 7. Altitude R^2 (how linear is the altitude profile)
    if n > 3:
        t_norm = np.linspace(0, 1, n)
        coeffs = np.polyfit(t_norm, alts, 1)
        alt_pred = np.polyval(coeffs, t_norm)
        ss_res = np.sum((alts - alt_pred)**2)
        ss_tot = np.sum((alts - np.mean(alts))**2)
        alt_r2 = float(max(0, 1 - ss_res/(ss_tot + 1e-10)))
    else:
        alt_r2 = 0.0

    # 8. Turn-speed correlation (slow down when turning?)
    if len(speeds) > 2 and len(dbearing) > 0:
        min_l3 = min(len(speeds)-1, len(dbearing))
        c = np.corrcoef(np.abs(dbearing[:min_l3]), speeds[1:min_l3+1])[0,1]
        turn_speed_corr = float(c) if np.isfinite(c) else 0.0
    else:
        turn_speed_corr = 0.0

    # 9. RCS periodicity (max ACF lag 2-5)
    if n > 8:
        rcs_c = rcs - rcs.mean()
        acf_vals = []
        for lag in range(2, min(6, n//2)):
            c = np.corrcoef(rcs_c[:-lag], rcs_c[lag:])[0,1]
            if np.isfinite(c):
                acf_vals.append(c)
        rcs_periodicity = float(max(acf_vals)) if acf_vals else 0.0
    else:
        rcs_periodicity = 0.0

    # 10. Speed quantile ratio (bimodal speed detection)
    if len(speeds) > 3:
        p10 = np.percentile(speeds, 10)
        p90 = np.percentile(speeds, 90)
        speed_q_ratio = float(p90 / (p10 + 0.1))
    else:
        speed_q_ratio = 1.0

    # 11. Max consecutive altitude direction run
    if len(vert_speed) > 2:
        vs_sign = np.sign(vert_speed)
        boundaries = np.where(np.diff(vs_sign) != 0)[0]
        if len(boundaries) > 0:
            runs = np.diff(np.concatenate([[0], boundaries, [len(vs_sign)]]))
            max_alt_run = float(runs.max() / len(vs_sign))
        else:
            max_alt_run = 1.0
    else:
        max_alt_run = 0.0

    # 12. RCS local range CV (consistency of RCS variability)
    if n > 6:
        seg_size = 5
        rcs_local_ranges = [np.ptp(rcs[j:j+seg_size]) for j in range(0, n-seg_size, seg_size)]
        rcs_local_range_cv = float(np.std(rcs_local_ranges) / (np.mean(rcs_local_ranges) + 1e-10))
    else:
        rcs_local_range_cv = 0.0

    # 13. High speed at low altitude (duck/pigeon signature)
    high_speed_low_alt = float(np.mean((speeds > 15) & (alts[1:] < 30)))

    # 14. Speed acceleration symmetry (are accelerations and decelerations equal?)
    if len(speeds) > 2:
        accel = np.diff(speeds)
        pos_accel = accel[accel > 0]
        neg_accel = accel[accel < 0]
        if len(pos_accel) > 0 and len(neg_accel) > 0:
            accel_symmetry = float(np.mean(pos_accel) / (-np.mean(neg_accel) + 1e-10))
        else:
            accel_symmetry = 1.0
    else:
        accel_symmetry = 1.0

    # 15. Vertical oscillation frequency (how often does altitude change direction)
    if len(vert_speed) > 3:
        vs_sign = np.sign(vert_speed)
        vs_changes = np.sum(np.diff(vs_sign) != 0)
        vert_osc_freq = float(vs_changes / (duration + 1e-10))
    else:
        vert_osc_freq = 0.0

    # 16. RCS during turns vs straight (body aspect angle effect)
    if len(dbearing) > 2:
        turning = np.abs(dbearing) > np.radians(10)
        straight = ~turning
        min_l4 = min(len(turning), n-2)
        rcs_seg = rcs[1:min_l4+1]
        if turning[:min_l4].sum() > 0 and straight[:min_l4].sum() > 0:
            rcs_turn = np.mean(rcs_seg[turning[:min_l4]])
            rcs_straight = np.mean(rcs_seg[straight[:min_l4]])
            rcs_turn_diff = float(rcs_turn - rcs_straight)
        else:
            rcs_turn_diff = 0.0
    else:
        rcs_turn_diff = 0.0

    feats = {
        "speed_dip_frac": speed_dip_frac,
        "speed_surge_frac": speed_surge_frac,
        "rcs_mod_rate": rcs_mod_rate,
        "alt_stable_frac": alt_stable_frac,
        "turn_frac_15deg": turn_frac,
        "glide_frac": glide_frac,
        "powered_frac": powered_frac,
        "bounding_rate": bounding,
        "rcs_half_ratio": rcs_half_ratio,
        "alt_r2": alt_r2,
        "turn_speed_corr": turn_speed_corr,
        "rcs_periodicity": rcs_periodicity,
        "speed_q_ratio": speed_q_ratio,
        "max_alt_run": max_alt_run,
        "rcs_local_range_cv": rcs_local_range_cv,
        "high_speed_low_alt": high_speed_low_alt,
        "accel_symmetry": accel_symmetry,
        "vert_osc_freq": vert_osc_freq,
        "rcs_turn_diff": rcs_turn_diff,
    }

    cidx = le.transform([r.bird_group])[0]
    cls = r.bird_group
    is_correct = pred_labels[idx] == cidx
    class_data[cls]["all"].append(feats)
    if is_correct:
        class_data[cls]["correct"].append(feats)
    else:
        class_data[cls]["wrong"].append(feats)

feat_keys = list(class_data[CLASSES[0]]["all"][0].keys())

# Build global arrays
all_feats = {k: [] for k in feat_keys}
all_labels = []
for cls in CLASSES:
    for row in class_data[cls]["all"]:
        for k in feat_keys:
            all_feats[k].append(row[k])
        all_labels.append(cls)
all_labels = np.array(all_labels)
for k in feat_keys:
    all_feats[k] = np.array(all_feats[k])
    all_feats[k] = np.where(np.isfinite(all_feats[k]), all_feats[k], 0)

print("=== MICRO-PATTERN FEATURES: Cohen d per class ===")
print()
header = f"{'Feature':>22s}"
for cls in CLASSES:
    header += f"  {cls[:8]:>8s}"
print(header)
print("-"*len(header))

for k in feat_keys:
    line = f"{k:>22s}"
    for cls in CLASSES:
        mask = all_labels == cls
        vc = all_feats[k][mask]
        vr = all_feats[k][~mask]
        pooled = np.sqrt((vc.std()**2 + vr.std()**2)/2)
        d = (vc.mean()-vr.mean())/pooled if pooled>1e-10 else 0
        star = "*" if abs(d)>0.5 else " "
        line += f"  {d:+7.3f}{star}"
    print(line)

print()
print("=== CORRECT vs WRONG: what separates the hard cases? ===")
print()
for cls in ["Cormorants", "Birds of Prey", "Waders", "Ducks"]:
    c = class_data[cls]["correct"]
    w = class_data[cls]["wrong"]
    if len(c) < 3 or len(w) < 3:
        continue
    print(f"{cls}: {len(c)} correct, {len(w)} wrong")
    for k in feat_keys:
        vc = np.array([r[k] for r in c])
        vw = np.array([r[k] for r in w])
        mc, mw = np.nanmean(vc), np.nanmean(vw)
        pooled = np.sqrt((np.nanstd(vc)**2 + np.nanstd(vw)**2)/2)
        d = (mc-mw)/pooled if pooled>1e-10 else 0
        if abs(d) > 0.25:
            print(f"  {k:>22s}: correct={mc:.4f} wrong={mw:.4f} d={d:+.3f}")
    print()

print("=== MISCLASSIFIED-AS-GULLS vs ACTUAL GULLS ===")
print("(what makes them NOT gulls even though model thinks they are?)")
print()
gull_feats = class_data["Gulls"]["all"]
for cls in ["Cormorants", "Birds of Prey", "Waders", "Ducks"]:
    cidx = CLASSES.index(cls)
    gull_idx = CLASSES.index("Gulls")
    mask_wrong_gull = (y == cidx) & (pred_labels == gull_idx)
    n_wrong = mask_wrong_gull.sum()
    if n_wrong < 3:
        continue

    print(f"{cls} misclassified as Gulls (n={n_wrong}) vs actual Gulls (n={len(gull_feats)}):")
    for k in feat_keys:
        gv = np.array([r[k] for r in gull_feats])
        wv = np.array([r[k] for r in class_data[cls]["wrong"]])
        if len(wv) < 3:
            continue
        mg = np.nanmean(gv); mw = np.nanmean(wv)
        pooled = np.sqrt((np.nanstd(gv)**2 + np.nanstd(wv)**2)/2)
        d = (mw - mg) / pooled if pooled > 1e-10 else 0
        if abs(d) > 0.2:
            print(f"  {k:>22s}: gulls={mg:.4f} mis-{cls[:6]}={mw:.4f} d={d:+.3f}")
    print()
