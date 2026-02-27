"""Deep outlier analysis: examine individual misclassified samples.

Not aggregates. Actual samples. What do they look like? Why does
the model get them wrong? Are there fixable patterns?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, parse_ewkb_4d
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)


# -- Load data --------------------------------------------------------
print("Loading...", flush=True)
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
oof = np.load(ROOT / "oof_e79.npy")

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

# Per-sample predictions
pred_class = oof.argmax(axis=1)
pred_conf = oof.max(axis=1)
correct = pred_class == y

# Per-sample margin (confidence gap)
sorted_probs = np.sort(oof, axis=1)[:, ::-1]
margin = sorted_probs[:, 0] - sorted_probs[:, 1]

# Second-best prediction
second_best = np.argsort(-oof, axis=1)[:, 1]

print(f"  Overall accuracy: {correct.mean():.3f} ({correct.sum()}/{len(y)})", flush=True)
print(f"  Errors: {(~correct).sum()}", flush=True)


# ====================================================================
#  1. CONFIDENT WRONG PREDICTIONS
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  1. MOST CONFIDENTLY WRONG PREDICTIONS".center(70), flush=True)
print("=" * 70, flush=True)
print("  (Model is SURE but WRONG -- these reveal systematic bias)", flush=True)

wrong_mask = ~correct
wrong_idx = np.where(wrong_mask)[0]
wrong_conf = pred_conf[wrong_mask]
wrong_sorted = wrong_idx[np.argsort(-wrong_conf)]

print(f"\n  {'Conf':>5s}  {'Margin':>6s}  {'True':15s}  {'Pred':15s}  {'2nd':15s}  "
      f"{'Speed':>6s}  {'MinZ':>6s}  {'MaxZ':>6s}  {'Size':12s}  {'Month':>5s}  {'ID'}", flush=True)
print("-" * 140, flush=True)

for i in range(min(50, len(wrong_sorted))):
    idx = wrong_sorted[i]
    row = train_df.iloc[idx]
    true_cls = CLASSES[y[idx]]
    pred_cls = CLASSES[pred_class[idx]]
    sec_cls = CLASSES[second_best[idx]]
    conf = pred_conf[idx]
    marg = margin[idx]
    speed = row.get("airspeed", np.nan)
    min_z = row.get("min_z", np.nan)
    max_z = row.get("max_z", np.nan)
    size = str(row.get("radar_bird_size", "?"))[:12]
    month = train_months[idx]
    tid = row.get("track_id", "?")

    print(f"  {conf:5.3f}  {marg:6.3f}  {true_cls:15s}  {pred_cls:15s}  {sec_cls:15s}  "
          f"{speed:6.1f}  {min_z:6.0f}  {max_z:6.0f}  {size:12s}  M{month:>2d}    {tid}", flush=True)


# ====================================================================
#  2. PER-CLASS ERROR PROFILES
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  2. ERROR PROFILE PER CLASS".center(70), flush=True)
print("=" * 70, flush=True)

for c in range(N_CLASSES):
    cls = CLASSES[c]
    true_mask = y == c
    n_total = true_mask.sum()
    n_correct = (true_mask & correct).sum()
    n_wrong = n_total - n_correct

    if n_wrong == 0:
        continue

    # Where do errors go?
    error_idx = np.where(true_mask & ~correct)[0]
    error_preds = pred_class[error_idx]
    destinations = np.bincount(error_preds, minlength=N_CLASSES)

    # Confidence of correct vs wrong
    correct_conf = pred_conf[true_mask & correct]
    wrong_conf = pred_conf[true_mask & ~correct]

    print(f"\n  --- {cls} ({n_correct}/{n_total} correct, {n_wrong} errors) ---", flush=True)
    print(f"  Correct confidence: {correct_conf.mean():.3f} (median {np.median(correct_conf):.3f})", flush=True)
    print(f"  Wrong confidence:   {wrong_conf.mean():.3f} (median {np.median(wrong_conf):.3f})", flush=True)

    print(f"  Errors go to:", flush=True)
    for dest_c in np.argsort(-destinations):
        if destinations[dest_c] > 0:
            pct = 100 * destinations[dest_c] / n_wrong
            print(f"    -> {CLASSES[dest_c]:15s}: {destinations[dest_c]:3d} ({pct:4.1f}%)", flush=True)

    # Feature profile of wrong samples
    wrong_df = train_df.iloc[error_idx]
    correct_idx = np.where(true_mask & correct)[0]
    correct_df = train_df.iloc[correct_idx]

    features = {
        "airspeed": "airspeed",
        "min_z": "min_z",
        "max_z": "max_z",
    }
    print(f"  Feature comparison (correct vs wrong):", flush=True)
    for fname, col in features.items():
        c_vals = pd.to_numeric(correct_df[col], errors="coerce").dropna()
        w_vals = pd.to_numeric(wrong_df[col], errors="coerce").dropna()
        if len(c_vals) > 0 and len(w_vals) > 0:
            print(f"    {fname:12s}: correct={c_vals.mean():7.1f}+/-{c_vals.std():5.1f}, "
                  f"wrong={w_vals.mean():7.1f}+/-{w_vals.std():5.1f}", flush=True)

    # Month distribution of errors
    error_months = train_months[error_idx]
    month_counts = {}
    for m in sorted(np.unique(train_months)):
        n_in_month = (true_mask & (train_months == m)).sum()
        n_err_in_month = (error_months == m).sum()
        if n_in_month > 0:
            month_counts[m] = (n_err_in_month, n_in_month, n_err_in_month / n_in_month)
    print(f"  Errors per month:", flush=True)
    for m, (nerr, ntot, rate) in month_counts.items():
        bar = "#" * int(rate * 20)
        print(f"    M{m:2d}: {nerr:3d}/{ntot:3d} ({100*rate:5.1f}%) {bar}", flush=True)


# ====================================================================
#  3. TRAJECTORY DEEP DIVE ON WORST SAMPLES
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  3. TRAJECTORY ANALYSIS OF HARDEST SAMPLES".center(70), flush=True)
print("=" * 70, flush=True)
print("  Extract actual flight characteristics from raw EWKB trajectories", flush=True)

def analyze_trajectory(hex_str, time_str):
    """Extract detailed flight characteristics from raw trajectory."""
    try:
        pts = parse_ewkb_4d(hex_str)
        if len(pts) < 3:
            return None

        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs = np.array([p[3] for p in pts])

        # Parse times
        import json
        times = np.array(json.loads(time_str)) if isinstance(time_str, str) else np.array(time_str)
        dt = np.diff(times)
        dt = np.clip(dt, 0.1, None)

        # Speeds between consecutive points
        dlat = np.diff(lats) * 111320
        dlon = np.diff(lons) * 111320 * np.cos(np.radians(lats[:-1]))
        dalt = np.diff(alts)
        dist_2d = np.sqrt(dlat**2 + dlon**2)
        speed_2d = dist_2d / dt

        # RCS characteristics
        rcs_mean = np.mean(rcs)
        rcs_std = np.std(rcs)
        rcs_range = np.ptp(rcs)
        # Autocorrelation lag 1
        if len(rcs) > 2:
            rcs_centered = rcs - rcs_mean
            ac1 = np.corrcoef(rcs_centered[:-1], rcs_centered[1:])[0, 1]
        else:
            ac1 = 0.0

        # Flight pattern
        total_dist = dist_2d.sum()
        straight_line = np.sqrt((dlat.sum())**2 + (dlon.sum())**2)
        sinuosity = total_dist / max(straight_line, 1.0)

        # Altitude pattern
        alt_changes = np.diff(alts)
        ascending_frac = (alt_changes > 0).mean() if len(alt_changes) > 0 else 0.5
        alt_variability = np.std(alts)

        # Speed pattern
        speed_mean = np.mean(speed_2d) if len(speed_2d) > 0 else 0
        speed_cv = np.std(speed_2d) / max(speed_mean, 0.1) if len(speed_2d) > 0 else 0

        # Bearing changes
        if len(dlat) >= 2:
            bearings = np.arctan2(dlon, dlat)
            bearing_changes = np.abs(np.diff(bearings))
            bearing_changes = np.minimum(bearing_changes, 2*np.pi - bearing_changes)
            mean_turn = np.mean(bearing_changes)
        else:
            mean_turn = 0.0

        return {
            "n_pts": len(pts),
            "duration": times[-1] - times[0] if len(times) > 1 else 0,
            "rcs_mean": rcs_mean,
            "rcs_std": rcs_std,
            "rcs_range": rcs_range,
            "rcs_ac1": ac1 if np.isfinite(ac1) else 0.0,
            "alt_mean": np.mean(alts),
            "alt_std": alt_variability,
            "alt_ascending_frac": ascending_frac,
            "sinuosity": sinuosity,
            "speed_mean": speed_mean,
            "speed_cv": speed_cv,
            "mean_turn": np.degrees(mean_turn),
        }
    except Exception:
        return None


# Analyze the top 30 most confidently wrong predictions
print("\nExtracting trajectory features for top-30 confident errors...", flush=True)
top30_wrong = wrong_sorted[:30]

for i, idx in enumerate(top30_wrong):
    row = train_df.iloc[idx]
    true_cls = CLASSES[y[idx]]
    pred_cls = CLASSES[pred_class[idx]]
    conf = pred_conf[idx]

    traj_info = analyze_trajectory(row["trajectory"], row["trajectory_time"])
    if traj_info is None:
        continue

    if i < 15:  # Print details for top 15
        print(f"\n  #{i+1} [TRUE={true_cls}, PRED={pred_cls}, conf={conf:.3f}]", flush=True)
        print(f"    Track: {traj_info['n_pts']} pts, {traj_info['duration']:.0f}s", flush=True)
        print(f"    RCS:   mean={traj_info['rcs_mean']:.1f} dB, std={traj_info['rcs_std']:.1f}, "
              f"range={traj_info['rcs_range']:.1f}, AC1={traj_info['rcs_ac1']:.2f}", flush=True)
        print(f"    Alt:   mean={traj_info['alt_mean']:.0f}m, std={traj_info['alt_std']:.1f}, "
              f"ascending={traj_info['alt_ascending_frac']:.2f}", flush=True)
        print(f"    Speed: mean={traj_info['speed_mean']:.1f} m/s, CV={traj_info['speed_cv']:.2f}", flush=True)
        print(f"    Path:  sinuosity={traj_info['sinuosity']:.2f}, "
              f"mean_turn={traj_info['mean_turn']:.1f} deg", flush=True)

        # Compare with class centroids
        print(f"    Typical {true_cls}:", flush=True)
        true_c = le.transform([true_cls])[0]
        true_mask_c = y == true_c
        true_df = train_df[true_mask_c]
        speed_vals = pd.to_numeric(true_df["airspeed"], errors="coerce")
        minz_vals = pd.to_numeric(true_df["min_z"], errors="coerce")
        maxz_vals = pd.to_numeric(true_df["max_z"], errors="coerce")
        print(f"      airspeed={speed_vals.mean():.1f}, "
              f"min_z={minz_vals.mean():.0f}, max_z={maxz_vals.mean():.0f}", flush=True)

        pred_c = le.transform([pred_cls])[0]
        pred_mask_c = y == pred_c
        pred_df = train_df[pred_mask_c]
        speed_vals2 = pd.to_numeric(pred_df["airspeed"], errors="coerce")
        minz_vals2 = pd.to_numeric(pred_df["min_z"], errors="coerce")
        maxz_vals2 = pd.to_numeric(pred_df["max_z"], errors="coerce")
        print(f"    Typical {pred_cls}:", flush=True)
        print(f"      airspeed={speed_vals2.mean():.1f}, "
              f"min_z={minz_vals2.mean():.0f}, max_z={maxz_vals2.mean():.0f}", flush=True)


# ====================================================================
#  4. CORMORANTS DEEP DIVE (biggest headroom)
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  4. CORMORANTS DEEP DIVE (AP=0.34, headroom=0.66)".center(70), flush=True)
print("=" * 70, flush=True)

corm_idx_true = np.where(y == CLASSES.index("Cormorants"))[0]
print(f"  Total Cormorants: {len(corm_idx_true)}", flush=True)

for idx in corm_idx_true:
    row = train_df.iloc[idx]
    pred_cls = CLASSES[pred_class[idx]]
    conf = pred_conf[idx]
    p_corm = oof[idx, CLASSES.index("Cormorants")]
    is_correct = pred_class[idx] == y[idx]

    traj_info = analyze_trajectory(row["trajectory"], row["trajectory_time"])
    if traj_info is None:
        traj_info = {"rcs_mean": np.nan, "rcs_ac1": np.nan, "sinuosity": np.nan,
                     "speed_mean": np.nan, "n_pts": np.nan, "alt_mean": np.nan}

    status = "OK" if is_correct else "WRONG"
    month = train_months[idx]

    print(f"  [{status:5s}] M{month:2d} pred={pred_cls:15s} conf={conf:.3f} P(Corm)={p_corm:.3f} | "
          f"RCS={traj_info['rcs_mean']:+5.1f} AC1={traj_info['rcs_ac1']:.2f} "
          f"sin={traj_info['sinuosity']:.2f} spd={traj_info['speed_mean']:5.1f} "
          f"n={traj_info['n_pts']} alt={traj_info['alt_mean']:.0f}m "
          f"size={str(row.get('radar_bird_size', '?'))}", flush=True)


# ====================================================================
#  5. WHAT THE PP CHANGES: Sample-level before/after
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  5. PP IMPACT: WHICH SAMPLES CHANGE AND ARE THEY RIGHT?".center(70), flush=True)
print("=" * 70, flush=True)

# Simulate PP with gamma=0.50, tau=0.30 on OOF
# For each LOMO fold, apply PP to the held-out month

from sklearn.preprocessing import LabelEncoder as LE2

size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]

def build_nb_simple(df_train, y_train):
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = df_train["radar_bird_size"].fillna("__UNK__").map(
        lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    speed = pd.to_numeric(df_train["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df_train["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df_train["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    K, S = N_CLASSES, len(size_levels)
    counts_cs = np.zeros((K, S))
    counts_c = np.zeros(K)
    for c in range(K):
        mask = y_train == c
        counts_c[c] = mask.sum()
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)
    p_size = (counts_cs + 1.0) / np.clip(counts_c[:, None] + S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))

    mu, sig = {}, {}
    for name, x in [("speed", speed), ("alt_mid", alt_mid), ("alt_range", alt_range)]:
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        gm, gs = float(np.nanmean(x)), max(float(np.nanstd(x)), 0.5)
        for c in range(K):
            xc = x[y_train == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = np.nanmean(xc)
                sig_f[c] = max(np.nanstd(xc), 0.5)
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[name], sig[name] = mu_f, sig_f
    return log_p_size, mu, sig


def nb_factors(df_val, log_p_size, mu, sig):
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = df_val["radar_bird_size"].fillna("__UNK__").map(
        lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    speed = pd.to_numeric(df_val["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df_val["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df_val["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)

    loglik = log_p_size[:, size_idx].T
    if ok.any():
        for name, x in [("speed", speed), ("alt_mid", alt_mid), ("alt_range", alt_range)]:
            z = (x[ok, None] - mu[name][None, :]) / sig[name][None, :]
            loglik[ok] += -0.5 * z * z - np.log(sig[name][None, :])
    loglik -= loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


unique_months = sorted(np.unique(train_months))

pp_correct_before = 0
pp_correct_after = 0
pp_flipped_right = 0  # was wrong, now right
pp_flipped_wrong = 0  # was right, now wrong
pp_stayed_wrong = 0   # was wrong, still wrong but different
flip_details = []

for held_month in unique_months:
    va_idx = np.where(train_months == held_month)[0]
    tr_idx = np.where(train_months != held_month)[0]

    # Build NB from training fold
    log_ps, mu_nb, sig_nb = build_nb_simple(train_df.iloc[tr_idx].reset_index(drop=True), y[tr_idx])
    facs, ok = nb_factors(train_df.iloc[va_idx].reset_index(drop=True), log_ps, mu_nb, sig_nb)

    # Get OOF predictions for this fold
    preds = oof[va_idx].copy()
    preds = np.clip(preds, 1e-12, None)
    preds /= preds.sum(axis=1, keepdims=True)

    # Apply PP (gamma=0.50, tau=0.30)
    margin_va = np.sort(preds, axis=1)[:, -1] - np.sort(preds, axis=1)[:, -2]
    gate = ok & (margin_va < 0.30)

    pp_preds = preds.copy()
    if gate.any():
        pp_preds[gate] = pp_preds[gate] * (facs[gate] ** 0.50)
        pp_preds[gate] /= np.clip(pp_preds[gate].sum(axis=1, keepdims=True), 1e-12, None)

    # Compare before/after
    pred_before = preds.argmax(axis=1)
    pred_after = pp_preds.argmax(axis=1)
    y_va = y[va_idx]

    flipped = pred_before != pred_after
    for local_i in np.where(flipped)[0]:
        global_i = va_idx[local_i]
        before_cls = CLASSES[pred_before[local_i]]
        after_cls = CLASSES[pred_after[local_i]]
        true_cls = CLASSES[y_va[local_i]]
        was_right = pred_before[local_i] == y_va[local_i]
        now_right = pred_after[local_i] == y_va[local_i]

        if was_right and not now_right:
            pp_flipped_wrong += 1
            direction = "BAD"
        elif not was_right and now_right:
            pp_flipped_right += 1
            direction = "GOOD"
        elif not was_right and not now_right:
            pp_stayed_wrong += 1
            direction = "SIDE"
        else:
            direction = "???"

        flip_details.append({
            "idx": global_i, "month": held_month,
            "true": true_cls, "before": before_cls, "after": after_cls,
            "direction": direction,
            "conf_before": preds[local_i].max(),
            "conf_after": pp_preds[local_i].max(),
        })

print(f"  Total flips across LOMO folds: {len(flip_details)}", flush=True)
print(f"  GOOD flips (wrong -> right): {pp_flipped_right}", flush=True)
print(f"  BAD flips  (right -> wrong): {pp_flipped_wrong}", flush=True)
print(f"  SIDE flips (wrong -> diff wrong): {pp_stayed_wrong}", flush=True)
print(f"  Net: {pp_flipped_right - pp_flipped_wrong:+d}", flush=True)

print(f"\n  --- Individual flips ---", flush=True)
print(f"  {'Dir':5s}  {'True':15s}  {'Before':15s}  {'After':15s}  "
      f"{'Conf_B':>6s}  {'Conf_A':>6s}  {'Month':>5s}", flush=True)
print("-" * 85, flush=True)

for fd in sorted(flip_details, key=lambda x: x["direction"]):
    print(f"  {fd['direction']:5s}  {fd['true']:15s}  {fd['before']:15s}  {fd['after']:15s}  "
          f"{fd['conf_before']:6.3f}  {fd['conf_after']:6.3f}  M{fd['month']:2d}", flush=True)


# ====================================================================
#  6. THE "IMPOSSIBLE" SAMPLES: correctly classified with very low conf
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  6. LOW-CONFIDENCE CORRECT PREDICTIONS (vulnerable)".center(70), flush=True)
print("=" * 70, flush=True)
print("  These are BARELY correct -- any perturbation could flip them", flush=True)

correct_idx = np.where(correct)[0]
correct_conf_vals = pred_conf[correct_idx]
vulnerable = correct_idx[np.argsort(correct_conf_vals)][:30]

print(f"\n  {'Conf':>5s}  {'Margin':>6s}  {'True':15s}  {'2nd guess':15s}  "
      f"{'Speed':>6s}  {'MinZ':>6s}  {'MaxZ':>6s}  {'Month':>5s}", flush=True)
print("-" * 100, flush=True)

for idx in vulnerable:
    row = train_df.iloc[idx]
    true_cls = CLASSES[y[idx]]
    sec_cls = CLASSES[second_best[idx]]
    conf = pred_conf[idx]
    marg = margin[idx]
    speed = row.get("airspeed", np.nan)
    min_z = row.get("min_z", np.nan)
    max_z = row.get("max_z", np.nan)
    month = train_months[idx]

    try:
        speed_v = float(speed)
        minz_v = float(min_z)
        maxz_v = float(max_z)
    except (ValueError, TypeError):
        speed_v = minz_v = maxz_v = float("nan")

    print(f"  {conf:5.3f}  {marg:6.3f}  {true_cls:15s}  {sec_cls:15s}  "
          f"{speed_v:6.1f}  {minz_v:6.0f}  {maxz_v:6.0f}  M{month:>2d}", flush=True)


print("\nDone.", flush=True)
