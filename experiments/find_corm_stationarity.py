"""Test STATIONARITY and BIOMECHANICS features for Cormorant detection.
Key insight: Cormorants have the highest flight cost of any bird -> MUST flap continuously.
Their RCS signal should be STATIONARY (same flapping process always).
Gulls switch between flap and glide -> NON-STATIONARY RCS."""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from sklearn.metrics import average_precision_score, roc_auc_score

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
CORM = 2; y_corm = (y == CORM).astype(int)

print("Computing stationarity + biomechanics features...", flush=True)
t0 = time.time()
feats = []

for i, (_, row) in enumerate(train.iterrows()):
    if i % 500 == 0: print(f"  {i}/2601", flush=True)
    pts = parse_ewkb_4d(row['trajectory'])
    times = parse_trajectory_time(row['trajectory_time'])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    n = len(pts)
    f = {}

    if n > 8:
        dists = np.array([haversine(lons[j], lats[j], lons[j+1], lats[j+1]) for j in range(n-1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists / dt
        dur = times[-1] - times[0]

        # ── RCS STATIONARITY ──
        for w in [3, 5, 8]:
            if n >= w + 2:
                local_vars = np.array([np.var(rcs[j:j+w]) for j in range(n - w + 1)])
                f[f"rcs_var_stability_w{w}"] = np.std(local_vars) / max(np.mean(local_vars), 0.001)
                # Fraction of windows with above-median variance
                med_lv = np.median(local_vars)
                f[f"rcs_high_var_frac_w{w}"] = np.mean(local_vars > med_lv * 1.5) if med_lv > 0 else 0
            else:
                f[f"rcs_var_stability_w{w}"] = 0
                f[f"rcs_high_var_frac_w{w}"] = 0

        # Half-track RCS variance ratio (stationarity proxy)
        half = n // 2
        v1 = np.var(rcs[:half]); v2 = np.var(rcs[half:])
        f["rcs_var_ratio"] = min(v1, v2) / max(max(v1, v2), 0.001)

        # ── ALTITUDE REVERSALS ──
        alt_diff = np.diff(alts)
        alt_clean = np.where(np.abs(alt_diff) < 0.5, 0, alt_diff)
        signs = np.sign(alt_clean)
        signs_nz = signs[signs != 0]
        if len(signs_nz) > 1:
            rev = np.sum(np.diff(signs_nz) != 0)
            f["alt_reversals_per_pt"] = rev / n
        else:
            f["alt_reversals_per_pt"] = 0

        # ── SPEED-RCS COUPLING ──
        min_l = min(len(speeds), n - 1)
        rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])[:min_l]
        spd = speeds[:min_l]
        if min_l > 3:
            sd = np.abs(np.diff(spd))
            rd = np.abs(np.diff(rcs_mid))
            ns = min(len(sd), len(rd))
            f["rcs_only_vary"] = np.mean((sd[:ns] < 0.5) & (rd[:ns] > 1.0))
            f["spd_only_vary"] = np.mean((sd[:ns] > 0.5) & (rd[:ns] < 1.0))
            f["neither_vary"] = np.mean((sd[:ns] < 0.5) & (rd[:ns] < 1.0))
        else:
            f["rcs_only_vary"] = 0; f["spd_only_vary"] = 0; f["neither_vary"] = 0

        # ── ALTITUDE HOLD ──
        if n >= 10:
            rm = np.convolve(alts, np.ones(10)/10, mode="valid")
            pad = np.concatenate([np.full(4, rm[0]), rm, np.full(5, rm[-1])])[:n]
            f["alt_hold_3m"] = np.mean(np.abs(alts - pad) < 3)
            f["alt_hold_5m"] = np.mean(np.abs(alts - pad) < 5)
        else:
            f["alt_hold_3m"] = 0; f["alt_hold_5m"] = 0

        # ── SPEED HOLD ──
        med_s = np.median(speeds)
        if med_s > 1:
            f["speed_hold_10pct"] = np.mean(np.abs(speeds - med_s) < med_s * 0.1)
            f["speed_hold_15pct"] = np.mean(np.abs(speeds - med_s) < med_s * 0.15)
            f["speed_max_dev"] = np.max(np.abs(speeds - med_s)) / med_s
        else:
            f["speed_hold_10pct"] = 0; f["speed_hold_15pct"] = 0; f["speed_max_dev"] = 0

        # ── HALF-TRACK SPEED CONSISTENCY ──
        h = len(speeds) // 2
        if h > 2:
            f["speed_half_consistency"] = 1 - abs(np.mean(speeds[:h]) - np.mean(speeds[h:])) / max(np.mean(speeds), 0.001)
        else:
            f["speed_half_consistency"] = 0

        # ── COMPOSITE: "continuous powered level flight" ──
        f["powered_level"] = f["speed_hold_15pct"] * f["alt_hold_5m"] * (1 - f["alt_reversals_per_pt"])
        f["pure_commute"] = f["powered_level"] * min(dur / 60, 1.0)  # longer = more commuting-like

        # ── TAKEOFF/LANDING TURBULENCE ──
        if n > 10:
            f["takeoff_accel"] = np.std(speeds[:5]) / max(np.std(speeds[5:min(n-1,20)]), 0.001)
            f["landing_accel"] = np.std(speeds[-5:]) / max(np.std(speeds[max(0,len(speeds)-20):-5]), 0.001)
        else:
            f["takeoff_accel"] = 0; f["landing_accel"] = 0

        # ── GLIDE DETECTION ──
        # Segments where speed drops AND altitude drops simultaneously (potential glide)
        if len(speeds) > 2:
            spd_diff = np.diff(speeds)
            alt_d = np.diff(alts[:len(speeds)])
            n_sg = min(len(spd_diff), len(alt_d))
            f["glide_segments"] = np.mean((spd_diff[:n_sg] < -0.3) & (alt_d[:n_sg] < -0.3))
            # Powered climb: speed stable + altitude increasing
            f["powered_climb"] = np.mean((np.abs(spd_diff[:n_sg]) < 1.0) & (alt_d[:n_sg] > 0.3))
        else:
            f["glide_segments"] = 0; f["powered_climb"] = 0

        f["duration"] = dur
    else:
        for k in ["rcs_var_stability_w3","rcs_var_stability_w5","rcs_var_stability_w8",
                   "rcs_high_var_frac_w3","rcs_high_var_frac_w5","rcs_high_var_frac_w8",
                   "rcs_var_ratio","alt_reversals_per_pt",
                   "rcs_only_vary","spd_only_vary","neither_vary",
                   "alt_hold_3m","alt_hold_5m","speed_hold_10pct","speed_hold_15pct",
                   "speed_max_dev","speed_half_consistency",
                   "powered_level","pure_commute","takeoff_accel","landing_accel",
                   "glide_segments","powered_climb","duration"]:
            f[k] = 0
    feats.append(f)

df = pd.DataFrame(feats)
print(f"\n{len(df.columns)} features in {time.time()-t0:.0f}s\n", flush=True)

# ── RANK ──
print("=" * 90)
print("STATIONARITY + BIOMECHANICS RANKING")
print("=" * 90)

results = []
for col in df.columns:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    try:
        ap_p = average_precision_score(y_corm, vals)
        ap_n = average_precision_score(y_corm, -vals)
        if ap_n > ap_p: ap = ap_n; d = "lower"
        else: ap = ap_p; d = "higher"
        auc = roc_auc_score(y_corm, vals if d == "higher" else -vals)
    except:
        ap = 0.015; auc = 0.5; d = "?"
    cm = np.median(vals[y == CORM])
    gm = np.median(vals[y == 5])
    # Per-class AUC
    cls_aucs = {}
    for ci in range(len(CLASSES)):
        try:
            a = roc_auc_score((y == ci).astype(int), vals if d == "higher" else -vals)
            if a < 0.5: a = 1 - a
            cls_aucs[CLASSES[ci]] = a
        except:
            cls_aucs[CLASSES[ci]] = 0.5
    results.append((col, ap, auc, d, cm, gm, cls_aucs))

results.sort(key=lambda x: -x[1])

print(f"\n{'Feature':30s} {'AP':>7s} {'AUC':>6s} {'Dir':>7s} {'Corm':>10s} {'Gull':>10s}")
print("-" * 80)
for col, ap, auc, d, cm, gm, _ in results:
    m = "***" if ap > 0.06 else "**" if ap > 0.04 else "*" if ap > 0.025 else ""
    print(f"  {col:30s} {ap:7.4f} {auc:6.3f} {d:>7s} {cm:10.4f} {gm:10.4f} {m}")

# Per-class AUC for top 10
print(f"\n{'Feature':30s}", end="")
for c in CLASSES:
    print(f" {c[:6]:>7s}", end="")
print()
print("-" * (30 + 8 * 9))
for col, ap, _, d, _, _, cls_aucs in results[:10]:
    print(f"  {col:30s}", end="")
    for c in CLASSES:
        print(f" {cls_aucs[c]:7.3f}", end="")
    print()

# Show Cormorant vs Gull distributions for top 3
print("\n\nDISTRIBUTION DETAIL (top 3 features):")
for col, ap, auc, d, cm, gm, _ in results[:3]:
    vals = np.nan_to_num(df[col].values, nan=0, posinf=0, neginf=0)
    corm_vals = vals[y == CORM]
    gull_vals = vals[y == 5]
    print(f"\n  {col} (AP={ap:.4f}, AUC={auc:.3f}, {d}):")
    print(f"    Corm: mean={np.mean(corm_vals):.4f} std={np.std(corm_vals):.4f} "
          f"[p10={np.percentile(corm_vals,10):.4f}, p90={np.percentile(corm_vals,90):.4f}]")
    print(f"    Gull: mean={np.mean(gull_vals):.4f} std={np.std(gull_vals):.4f} "
          f"[p10={np.percentile(gull_vals,10):.4f}, p90={np.percentile(gull_vals,90):.4f}]")
    overlap = np.mean((corm_vals >= np.percentile(gull_vals, 10)) &
                       (corm_vals <= np.percentile(gull_vals, 90)))
    print(f"    Overlap: {overlap*100:.0f}% of Cormorants within Gull 10-90 range")

print("\nDone.", flush=True)
