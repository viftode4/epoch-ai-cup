"""Systematic diagnostic analysis -- finding ACTIONABLE insights.

NOT repeating: confusion matrix, feature importance, adversarial validation,
Cormorant deep-dives (6 prior scripts), basic per-class stats.

NEW questions answered:
  A. Post-processing dissection: per-class AP at each pipeline stage
  B. "Best achievable" ceiling: what if we perfectly fix class X?
  C. Subgroup analysis: within confused pairs, WHAT separates correct/wrong?
  D. Cross-month transferability per class
  E. E75 vs E79 Waders divergence: why do both = 0.59?
  F. Feature interactions the model might miss
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, load_test, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# -- Load everything --
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

oof_e79 = np.load(ROOT / "oof_e79.npy", allow_pickle=True).astype(float)
test_e79 = np.load(ROOT / "test_e79.npy", allow_pickle=True).astype(float)

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# =====================================================================
# A. POST-PROCESSING DISSECTION
# =====================================================================
print("=" * 70, flush=True)
print("A. POST-PROCESSING PIPELINE DISSECTION".center(70), flush=True)
print("=" * 70, flush=True)
print("\nApplying E75 pipeline stage by stage on OOF to see per-class AP change.\n", flush=True)

# Reconstruct E75 pipeline stages
UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15
TAU_NB = 0.30
GAMMA = 0.10
LAPLACE = 1.0
MIN_SIGMA = 0.50

def renorm_rows(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)

def top2_margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]

def log_gaussian(x, mu, sigma):
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])

# Build NB params
def build_nb():
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    yy = le.transform(train_df["bird_group"])
    size_idx = train_df["radar_bird_size"].fillna("__UNK__").map(
        lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}
    K, S = N_CLASSES, len(size_levels)
    counts_cs = np.zeros((K, S))
    counts_c = np.zeros(K)
    for c in range(K):
        mask = yy == c
        counts_c[c] = mask.sum()
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)
    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))
    mu, sig = {}, {}
    for feat, x in feats.items():
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        gm, gs = float(np.nanmean(x)), max(float(np.nanstd(x)), MIN_SIGMA)
        for c in range(K):
            xc = x[yy == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sig_f[c] = max(float(np.nanstd(xc)), MIN_SIGMA)
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat], sig[feat] = mu_f, sig_f
    return size_levels, log_p_size, mu, sig

def compute_nb(df, size_levels, log_p_size, mu, sig):
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = df["radar_bird_size"].fillna("__UNK__").map(
        lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
    loglik = log_p_size[:, size_idx].T
    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])
    loglik -= loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok

# GBIF priors
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()
si = {}
for _, row in gbif.iterrows():
    month = int(row["month"])
    vals = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            vals[i] = 1.0
        else:
            class_mean = gbif[cls].values.mean()
            vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    si[month] = vals
priors = {}
for month in range(1, 13):
    raw = np.maximum(p_train * si[month], 1e-8)
    priors[month] = raw / raw.sum()

size_levels, log_p_size, mu, sig = build_nb()
nb_factors, ok_nb = compute_nb(train_df, size_levels, log_p_size, mu, sig)

# Stage 0: Raw OOF
stage0 = oof_e79.copy()
m0, p0 = compute_map(y, stage0)

# Stage 1: GBIF gated priors (unseen months only in train: months 1,4 proxy)
# Actually on train OOF, unseen months don't exist. But we apply to ALL months
# to see the effect. Let's apply to months that WOULD be unseen-like.
# Better: apply exactly as the pipeline does and measure per-class AP.
stage1 = stage0.copy()
margin = top2_margin(stage1)
prior_changed = np.zeros(len(y), dtype=bool)
for month, alpha in BASE_ALPHA.items():
    mask_m = train_months == month
    if mask_m.sum() == 0:
        continue
    gate = mask_m & (margin < TAU_PRIOR)
    if gate.sum() == 0:
        continue
    ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
    stage1[gate] = stage1[gate] * ratio
    stage1[gate] /= np.clip(stage1[gate].sum(axis=1, keepdims=True), 1e-12, None)
    prior_changed |= gate
stage1 = renorm_rows(stage1)
m1, p1 = compute_map(y, stage1)

# Stage 2: NB physics (unseen months = none in train, but we apply to all)
stage2 = stage1.copy()
margin2 = top2_margin(stage2)
# Apply to ALL months to see effect
gate_nb = ok_nb & (margin2 < TAU_NB)
nb_changed = np.zeros(len(y), dtype=bool)
if gate_nb.any():
    stage2[gate_nb] = stage2[gate_nb] * (nb_factors[gate_nb] ** GAMMA)
    stage2 = renorm_rows(stage2)
    nb_changed = gate_nb
m2, p2 = compute_map(y, stage2)

print(f"{'Stage':30s} {'mAP':>7s} {'delta':>7s}  " +
      "  ".join(f"{c[:5]:>6s}" for c in CLASSES), flush=True)
print("-" * 120, flush=True)

for label, m, p in [("0: Raw E79 OOF", m0, p0),
                     ("1: + GBIF priors (gated)", m1, p1),
                     ("2: + NB physics (gated)", m2, p2)]:
    delta = m - m0
    per = "  ".join(f"{p[c]:6.4f}" for c in CLASSES)
    print(f"  {label:28s} {m:7.4f} {delta:+7.4f}  {per}", flush=True)

# Show which samples changed
print(f"\n  GBIF priors changed: {int(prior_changed.sum())} samples", flush=True)
print(f"  NB physics changed:  {int(nb_changed.sum())} samples", flush=True)

# Per-class delta from raw to post-processed
print(f"\n  Per-class AP delta (stage2 - stage0):", flush=True)
for c in CLASSES:
    d = p2[c] - p0[c]
    direction = "+" if d > 0 else ""
    print(f"    {c:15s}: {direction}{d:.4f} ({p0[c]:.4f} -> {p2[c]:.4f})", flush=True)

# =====================================================================
# B. "BEST ACHIEVABLE" CEILING ANALYSIS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("B. CEILING ANALYSIS: WHAT IF WE PERFECTLY FIX ONE CLASS?".center(70), flush=True)
print("=" * 70, flush=True)
print("\nIf we could set one class's AP to 1.0, what would overall mAP be?\n", flush=True)

for c in CLASSES:
    simulated = dict(p0)
    simulated[c] = 1.0
    sim_map = np.mean(list(simulated.values()))
    gain = sim_map - m0
    print(f"  Perfect {c:15s}: mAP={sim_map:.4f} (gain={gain:+.4f}, "
          f"current AP={p0[c]:.4f}, headroom={1.0 - p0[c]:.4f})", flush=True)

print(f"\n  Current mAP: {m0:.4f}", flush=True)
print(f"  Perfect ALL: {1.0:.4f}", flush=True)
print(f"\n  Most impactful to improve (headroom / 9):", flush=True)
ranked = sorted(CLASSES, key=lambda c: 1.0 - p0[c], reverse=True)
for c in ranked:
    headroom = 1.0 - p0[c]
    print(f"    {c:15s}: headroom={headroom:.4f}, "
          f"max mAP gain={headroom / 9:.4f}", flush=True)

# =====================================================================
# C. SUBGROUP ANALYSIS: WITHIN CONFUSED PAIRS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("C. SUBGROUP ANALYSIS WITHIN CONFUSED CLASS PAIRS".center(70), flush=True)
print("=" * 70, flush=True)

# Extract trajectory-level features for deeper analysis
print("\nExtracting per-track features...", flush=True)
track_feats = []
for idx in range(len(train_df)):
    row = train_df.iloc[idx]
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])
        rcs = np.array([p[3] for p in pts])
        alts = np.array([p[2] for p in pts])
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        n = len(pts)
        duration = max(times[-1] - times[0], 0.001) if n > 1 else 0.001

        # RCS autocorrelation
        rc = rcs - rcs.mean()
        var = np.var(rc)
        ac1 = np.correlate(rc[:-1], rc[1:])[0] / (var * (n - 1)) if var > 1e-8 and n > 2 else 0

        # Altitude change pattern
        alt_ascending = np.sum(np.diff(alts) > 0) / max(n - 1, 1) if n > 1 else 0.5

        # Speed from trajectory
        if n > 1:
            from math import radians, sin, cos, asin, sqrt
            dists = []
            for i in range(n - 1):
                R = 6371000
                lon1, lat1 = radians(lons[i]), radians(lats[i])
                lon2, lat2 = radians(lons[i+1]), radians(lats[i+1])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                dists.append(R * 2 * asin(sqrt(a)))
            dists = np.array(dists)
            dt = np.maximum(np.diff(times), 0.001)
            speeds = dists / dt
            speed_cv = np.std(speeds) / (np.mean(speeds) + 0.01) if len(speeds) > 1 else 0
        else:
            speed_cv = 0

        # Sinuosity: total distance / straight-line distance
        if n > 2:
            total_dist = sum(dists)
            straight = dists[0]  # placeholder
            R = 6371000
            lon1, lat1 = radians(lons[0]), radians(lats[0])
            lon2, lat2 = radians(lons[-1]), radians(lats[-1])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            straight = R * 2 * asin(sqrt(max(a, 0)))
            sinuosity = total_dist / max(straight, 1.0)
        else:
            sinuosity = 1.0

        track_feats.append({
            "n_pts": n, "duration": duration,
            "rcs_mean": rcs.mean(), "rcs_std": rcs.std(),
            "rcs_ac1": ac1,
            "alt_mean": alts.mean(), "alt_std": alts.std(),
            "alt_ascending_frac": alt_ascending,
            "speed_cv": speed_cv,
            "sinuosity": sinuosity,
            "lat_mean": lats.mean(), "lon_mean": lons.mean(),
        })
    except Exception:
        track_feats.append({k: np.nan for k in [
            "n_pts", "duration", "rcs_mean", "rcs_std", "rcs_ac1",
            "alt_mean", "alt_std", "alt_ascending_frac", "speed_cv",
            "sinuosity", "lat_mean", "lon_mean"]})

tf = pd.DataFrame(track_feats)
tf["airspeed"] = pd.to_numeric(train_df["airspeed"], errors="coerce")
tf["min_z"] = pd.to_numeric(train_df["min_z"], errors="coerce")
tf["max_z"] = pd.to_numeric(train_df["max_z"], errors="coerce")
tf["size"] = train_df["radar_bird_size"]
tf["month"] = train_months
tf["true_class"] = [CLASSES[yi] for yi in y]
tf["pred_class"] = [CLASSES[yi] for yi in oof_e79.argmax(1)]
tf["correct"] = (oof_e79.argmax(1) == y)
tf["confidence"] = oof_e79.max(1)
tf["margin"] = top2_margin(oof_e79)

# For the top confused pairs, find what separates correct from wrong
confused_pairs = [
    ("Waders", "Gulls", 37),
    ("Birds of Prey", "Gulls", 35),
    ("Cormorants", "Gulls", 18),
    ("Birds of Prey", "Songbirds", 15),
    ("Pigeons", "Songbirds", 14),
    ("Songbirds", "Gulls", 54),
]

analysis_features = ["airspeed", "min_z", "max_z", "rcs_mean", "rcs_std", "rcs_ac1",
                      "alt_mean", "alt_std", "n_pts", "duration", "speed_cv",
                      "sinuosity", "alt_ascending_frac"]

for true_cls, pred_cls, n_confused in confused_pairs:
    true_idx = CLASSES.index(true_cls)
    pred_idx = CLASSES.index(pred_cls)

    # Correctly classified true_cls
    correct_mask = (y == true_idx) & (oof_e79.argmax(1) == true_idx)
    # Misclassified true_cls -> pred_cls
    wrong_mask = (y == true_idx) & (oof_e79.argmax(1) == pred_idx)
    # Actual pred_cls samples (for comparison)
    actual_pred_mask = (y == pred_idx)

    n_c = int(correct_mask.sum())
    n_w = int(wrong_mask.sum())

    print(f"\n  --- {true_cls} -> {pred_cls} ({n_confused} confused) ---", flush=True)
    print(f"  Correct {true_cls}: {n_c}, Wrong (-> {pred_cls}): {n_w}, "
          f"Actual {pred_cls}: {int(actual_pred_mask.sum())}", flush=True)

    if n_w < 3:
        print(f"  Too few wrong samples to analyze.", flush=True)
        continue

    print(f"\n  {'Feature':20s} {'Correct':>12s} {'Wrong->'+pred_cls[:4]:>12s} "
          f"{'Actual '+pred_cls[:4]:>12s} {'Separable?':>12s}", flush=True)
    print("  " + "-" * 72, flush=True)

    for feat in analysis_features:
        v_c = tf.loc[correct_mask, feat].dropna()
        v_w = tf.loc[wrong_mask, feat].dropna()
        v_a = tf.loc[actual_pred_mask, feat].dropna()

        if len(v_c) < 3 or len(v_w) < 3:
            continue

        mean_c = v_c.mean()
        mean_w = v_w.mean()
        mean_a = v_a.mean()

        # Effect size (Cohen's d between correct and wrong)
        pooled_std = np.sqrt((v_c.std()**2 + v_w.std()**2) / 2)
        if pooled_std > 1e-6:
            d = abs(mean_c - mean_w) / pooled_std
        else:
            d = 0

        # Does the wrong group look more like the actual pred class?
        wrong_closer_to_pred = abs(mean_w - mean_a) < abs(mean_c - mean_a)

        sep = "***" if d > 0.8 else "**" if d > 0.5 else "*" if d > 0.3 else ""
        arrow = " (->pred)" if wrong_closer_to_pred and d > 0.3 else ""

        print(f"  {feat:20s} {mean_c:12.3f} {mean_w:12.3f} {mean_a:12.3f} "
              f"d={d:.2f} {sep}{arrow}", flush=True)

# =====================================================================
# D. CROSS-MONTH TRANSFERABILITY PER CLASS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("D. CROSS-MONTH TRANSFERABILITY".center(70), flush=True)
print("=" * 70, flush=True)
print("\nFor each class: how much does accuracy drop when evaluated on a", flush=True)
print("month it was NOT trained on (LOMO proxy for unseen months)?\n", flush=True)

# Use LOMO-like analysis: for each month, compute per-class metrics
# Train months are [1,4,9,10]. For each held-out month, see which classes transfer.
print(f"{'Class':15s} {'M1_acc':>7s} {'M4_acc':>7s} {'M9_acc':>7s} {'M10_acc':>7s} "
      f"{'Best':>7s} {'Worst':>7s} {'Gap':>7s}", flush=True)
print("-" * 75, flush=True)

for c in range(N_CLASSES):
    accs = {}
    for month in sorted(np.unique(train_months)):
        mask = (train_months == month) & (y == c)
        n = int(mask.sum())
        if n == 0:
            accs[month] = None
            continue
        correct = int((oof_e79[mask].argmax(1) == c).sum())
        accs[month] = correct / n

    vals = [f"{accs[m]:.3f}" if accs[m] is not None else "  N/A" for m in [1, 4, 9, 10]]
    valid_accs = [v for v in accs.values() if v is not None]
    best = max(valid_accs) if valid_accs else 0
    worst = min(valid_accs) if valid_accs else 0
    gap = best - worst

    print(f"  {CLASSES[c]:15s} {'  '.join(vals)}   {best:.3f}   {worst:.3f}   {gap:.3f}", flush=True)

# =====================================================================
# E. E75 vs E79 WADERS DIVERGENCE
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("E. E75 vs E79 WADERS DIVERGENCE ON TEST".center(70), flush=True)
print("=" * 70, flush=True)

e75_file = ROOT / "submissions" / "e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv"
if e75_file.exists():
    e75_df = pd.read_csv(e75_file)
    e75_probs = e75_df[CLASSES].values

    # E79 submission with same post-processing
    # Use raw test_e79 and apply post-processing ourselves
    e79_raw = test_e79.copy()

    top_e75 = e75_probs.argmax(1)
    top_e79 = e79_raw.argmax(1)

    # E75 predicts way more Waders. Where?
    wader_idx = CLASSES.index("Waders")
    e75_waders = (top_e75 == wader_idx)
    e79_waders = (top_e79 == wader_idx)

    print(f"\n  E75 predicts {int(e75_waders.sum())} Waders, E79 predicts {int(e79_waders.sum())}", flush=True)

    # Where are the extra E75 Waders?
    extra_e75_waders = e75_waders & ~e79_waders
    print(f"  Extra E75 Waders (not in E79): {int(extra_e75_waders.sum())}", flush=True)

    # What does E79 call these extra Waders?
    print(f"  E79 calls E75's extra Waders:", flush=True)
    for c in range(N_CLASSES):
        n = int(((top_e79 == c) & extra_e75_waders).sum())
        if n > 0:
            print(f"    {CLASSES[c]:15s}: {n}", flush=True)

    # Per-month breakdown of extra Waders
    print(f"\n  Extra E75 Waders per month:", flush=True)
    for month in sorted(np.unique(test_months)):
        mask = (test_months == month) & extra_e75_waders
        n = int(mask.sum())
        n_total = int((test_months == month).sum())
        print(f"    Month {month:2d}: {n:3d}/{n_total} ({n/n_total*100:.1f}%)", flush=True)

    # Analyze the features of these extra Waders
    print(f"\n  Feature profile of E75's extra Waders vs actual Waders in train:", flush=True)
    train_waders = train_df[y == wader_idx]
    extra_w_df = test_df[extra_e75_waders]

    for col in ["airspeed", "min_z", "max_z"]:
        tr_v = pd.to_numeric(train_waders[col], errors="coerce").dropna()
        te_v = pd.to_numeric(extra_w_df[col], errors="coerce").dropna()
        if len(te_v) > 0:
            print(f"    {col:15s}: train_waders={tr_v.mean():.1f}+/-{tr_v.std():.1f}, "
                  f"extra_e75_waders={te_v.mean():.1f}+/-{te_v.std():.1f}", flush=True)

    print(f"  Size distribution:", flush=True)
    print(f"    Train waders: {dict(train_waders['radar_bird_size'].value_counts())}", flush=True)
    print(f"    Extra E75 waders: {dict(extra_w_df['radar_bird_size'].value_counts())}", flush=True)

    # Key question: what's E75's Wader probability vs E79's on these samples?
    print(f"\n  E75 vs E79 Wader probability on the extra Wader samples:", flush=True)
    e75_wader_prob = e75_probs[extra_e75_waders, wader_idx]
    e79_wader_prob = e79_raw[extra_e75_waders, wader_idx]
    print(f"    E75 P(Wader): mean={e75_wader_prob.mean():.3f}, "
          f"median={np.median(e75_wader_prob):.3f}", flush=True)
    print(f"    E79 P(Wader): mean={e79_wader_prob.mean():.3f}, "
          f"median={np.median(e79_wader_prob):.3f}", flush=True)
    print(f"    E75 P(Wader) > 0.5: {int((e75_wader_prob > 0.5).sum())}", flush=True)
    print(f"    E79 P(Wader) > 0.5: {int((e79_wader_prob > 0.5).sum())}", flush=True)

    # What's E75's confidence on Waders overall?
    e75_top = e75_probs.max(1)
    e79_top = e79_raw.max(1)
    print(f"\n  Overall confidence comparison:", flush=True)
    print(f"    E75: mean_conf={e75_top.mean():.3f}", flush=True)
    print(f"    E79: mean_conf={e79_top.mean():.3f}", flush=True)

    # Per-class prediction counts
    print(f"\n  Per-class test prediction counts:", flush=True)
    print(f"  {'Class':15s} {'E75':>6s} {'E79':>6s} {'Diff':>6s}", flush=True)
    for c in range(N_CLASSES):
        n75 = int((top_e75 == c).sum())
        n79 = int((top_e79 == c).sum())
        diff = n75 - n79
        print(f"  {CLASSES[c]:15s} {n75:6d} {n79:6d} {diff:+6d}", flush=True)

else:
    print("  E75 submission not found.", flush=True)

# =====================================================================
# F. FEATURE INTERACTIONS THE MODEL MIGHT MISS
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("F. FEATURE INTERACTIONS & MISSED PATTERNS".center(70), flush=True)
print("=" * 70, flush=True)

# Look for 2-feature interaction rules that separate confused classes
# Focus on the biggest confusion: Waders misclassified as Gulls
print("\n  --- Waders vs Gulls: 2D decision boundary analysis ---", flush=True)
wader_correct = (y == CLASSES.index("Waders")) & (oof_e79.argmax(1) == CLASSES.index("Waders"))
wader_as_gull = (y == CLASSES.index("Waders")) & (oof_e79.argmax(1) == CLASSES.index("Gulls"))
actual_gulls = (y == CLASSES.index("Gulls"))

feat_pairs = [
    ("airspeed", "max_z"),
    ("airspeed", "rcs_mean"),
    ("min_z", "rcs_ac1"),
    ("rcs_mean", "sinuosity"),
    ("alt_ascending_frac", "speed_cv"),
    ("n_pts", "rcs_ac1"),
]

for f1, f2 in feat_pairs:
    v1_wc = tf.loc[wader_correct, f1].dropna()
    v2_wc = tf.loc[wader_correct, f2].dropna()
    v1_wg = tf.loc[wader_as_gull, f1].dropna()
    v2_wg = tf.loc[wader_as_gull, f2].dropna()
    v1_g = tf.loc[actual_gulls, f1].dropna()
    v2_g = tf.loc[actual_gulls, f2].dropna()

    if len(v1_wc) < 3 or len(v1_wg) < 3:
        continue

    # Find a simple rule: if f1 > threshold AND f2 > threshold -> Wader
    # Try the medians of correctly classified Waders
    t1 = v1_wc.median()
    t2 = v2_wc.median()

    # How many Wader->Gull errors would this rule recover?
    recoverable = 0
    for idx in tf[wader_as_gull].index:
        v1 = tf.loc[idx, f1]
        v2 = tf.loc[idx, f2]
        if pd.notna(v1) and pd.notna(v2) and v1 > t1 and v2 > t2:
            recoverable += 1

    # How many actual Gulls would this falsely catch?
    false_catches = 0
    for idx in tf[actual_gulls].index[:200]:  # sample for speed
        v1 = tf.loc[idx, f1]
        v2 = tf.loc[idx, f2]
        if pd.notna(v1) and pd.notna(v2) and v1 > t1 and v2 > t2:
            false_catches += 1
    false_catches = int(false_catches * len(tf[actual_gulls]) / 200)

    print(f"  Rule: {f1}>{t1:.2f} AND {f2}>{t2:.2f}", flush=True)
    print(f"    Recovers {recoverable}/{int(wader_as_gull.sum())} Wader errors, "
          f"false-catches ~{false_catches}/{int(actual_gulls.sum())} Gulls", flush=True)

# =====================================================================
# G. THE KEY QUESTION: WHERE IS 0.10 mAP HIDING?
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print("G. WHERE IS THE MISSING 0.10 mAP?".center(70), flush=True)
print("=" * 70, flush=True)
print("\nCurrent SKF OOF mAP = 0.7736, LB = 0.59.", flush=True)
print("Gap = 0.18. Where does it come from?\n", flush=True)

# Estimate per-class LB AP assuming the gap is proportional to
# class vulnerability to month shift (measured by LOMO vs SKF)
# We know: Month 10 alone = 0.80 mAP (best month)
# Month 9 = 0.52 mAP
# Month 1 = 0.53 mAP
# Month 4 = 0.57 mAP

# Simulate LB-like mAP by reweighting months to match test distribution
# Test: Oct=42.9%, Sep=24.4%, May=16.2%, Feb=9.4%, Dec=7.1%
# We can only compute for shared months (Sep, Oct)
# For unseen, assume performance similar to the worst training month

print("  Per-class AP on shared months (from OOF):", flush=True)
shared_mask = np.isin(train_months, [9, 10])
if shared_mask.sum() > 0:
    m_shared, p_shared = compute_map(y[shared_mask], oof_e79[shared_mask])
    print(f"  Shared months mAP (Sep+Oct on OOF): {m_shared:.4f}", flush=True)
    for c in CLASSES:
        print(f"    {c:15s}: {p_shared[c]:.4f}", flush=True)

# What would happen if unseen months have X% of shared-month AP?
print(f"\n  Simulated LB = 0.67 * shared_mAP + 0.33 * unseen_mAP:", flush=True)
for unseen_frac in [0.2, 0.3, 0.4, 0.5, 0.6]:
    simulated_unseen = {c: p_shared[c] * unseen_frac for c in CLASSES}
    sim_lb = 0.67 * m_shared + 0.33 * np.mean(list(simulated_unseen.values()))
    print(f"    unseen={unseen_frac:.0%} of shared: LB ~ {sim_lb:.3f}", flush=True)

# What fraction would give us LB=0.59?
# 0.67 * m_shared + 0.33 * x = 0.59
# x = (0.59 - 0.67 * m_shared) / 0.33
x_needed = (0.59 - 0.67 * m_shared) / 0.33
print(f"\n  To get LB=0.59: unseen mAP must be ~ {x_needed:.3f}", flush=True)
print(f"  That's {x_needed/m_shared:.1%} of shared-month performance.", flush=True)

x_target = (0.65 - 0.67 * m_shared) / 0.33
print(f"  To get LB=0.65: unseen mAP must be ~ {x_target:.3f}", flush=True)
print(f"  To get LB=0.70: unseen mAP must be ~ {(0.70 - 0.67 * m_shared) / 0.33:.3f}", flush=True)

print("\n\nDone.", flush=True)
