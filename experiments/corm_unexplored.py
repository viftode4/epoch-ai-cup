"""Unexplored angles for Cormorant detection.
1. TabPFN vs LGB: per-Cormorant comparison — what does TabPFN see?
2. Three confusion types analyzed separately
3. Observation context / co-occurrence
4. Test data cluster structure
5. FFT power spectrum of RCS
6. Multi-resolution consistency (first half vs second half)
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from src.metrics import compute_map
from sklearn.metrics import average_precision_score, roc_auc_score

train = load_train()
test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
CORM = 2; GULL = 5
corm_idx = np.where(y == CORM)[0]

# Load existing predictions
oof_tabpfn = np.load("oof_e183_tabpfn.npy")
oof_e175 = np.load("oof_e175_best.npy")
from scipy.special import softmax
oof_e175_prob = softmax(oof_e175, axis=1)
test_tabpfn = np.load("test_e183_tabpfn.npy")

# ═══════════════════════════════════════════════════════════════
print("=" * 90, flush=True)
print("1. TabPFN vs E175: WHAT DOES TabPFN SEE THAT E175 MISSES?", flush=True)
print("=" * 90, flush=True)

# Rank each Cormorant by both models
for i, idx in enumerate(corm_idx):
    p_tab = oof_tabpfn[idx, CORM]
    p_e175 = oof_e175_prob[idx, CORM]
    rank_tab = (oof_tabpfn[:, CORM] > p_tab).sum()
    rank_e175 = (oof_e175_prob[:, CORM] > p_e175).sum()

    # Where they DISAGREE most
    diff = rank_e175 - rank_tab  # positive = TabPFN ranks better

    pts = parse_ewkb_4d(train.iloc[idx]["trajectory"])
    times = parse_trajectory_time(train.iloc[idx]["trajectory_time"])
    rcs = np.array([p[3] for p in pts])
    alts = np.array([p[2] for p in pts])
    n = len(pts)

    if n > 2:
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        dists = np.array([haversine(lons[j],lats[j],lons[j+1],lats[j+1]) for j in range(n-1)])
        dt = np.maximum(np.diff(times), 0.001)
        speeds = dists/dt
        speed_cv = np.std(speeds)/max(np.mean(speeds), 0.001)
    else:
        speed_cv = 0

    label = "TabPFN WINS" if diff > 50 else "E175 WINS" if diff < -50 else "SIMILAR"
    print(f"  Corm#{i+1:2d}: TabPFN rank={rank_tab:4d} E175 rank={rank_e175:4d} diff={diff:+5d} [{label}] "
          f"n={n} rcs={np.mean(rcs):.1f} spd_cv={speed_cv:.2f} size={train.iloc[idx]['radar_bird_size']} m={months[idx]}", flush=True)

# Summary: which Cormorant PROPERTIES predict TabPFN success?
tabpfn_ranks = np.array([(oof_tabpfn[:, CORM] > oof_tabpfn[idx, CORM]).sum() for idx in corm_idx])
e175_ranks = np.array([(oof_e175_prob[:, CORM] > oof_e175_prob[idx, CORM]).sum() for idx in corm_idx])
improvement = e175_ranks - tabpfn_ranks  # positive = TabPFN better

# What features correlate with TabPFN improvement?
feats = pd.read_pickle("data/_cached_train_features_v3.pkl")
print(f"\n  Features that predict WHERE TabPFN improves over E175:", flush=True)
corm_feats = feats.iloc[corm_idx]
for col in feats.columns:
    vals = corm_feats[col].values
    if np.std(vals) > 0:
        corr = np.corrcoef(vals, improvement)[0, 1]
        if abs(corr) > 0.3 and not np.isnan(corr):
            print(f"    {col:35s} r={corr:+.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("2. THREE CONFUSION TYPES — SEPARATE ANALYSIS", flush=True)
print("=" * 90, flush=True)

# Split Cormorants by what E175 predicts them as
e175_pred = oof_e175_prob[corm_idx].argmax(axis=1)
confusion_groups = {
    "Correct (Corm)": corm_idx[e175_pred == CORM],
    "Confused as Gulls": corm_idx[e175_pred == GULL],
    "Confused as Songbirds": corm_idx[e175_pred == CLASSES.index("Songbirds")],
    "Confused as Geese": corm_idx[e175_pred == CLASSES.index("Geese")],
    "Confused as Other": corm_idx[~np.isin(e175_pred, [CORM, GULL, CLASSES.index("Songbirds"), CLASSES.index("Geese")])],
}

for group_name, indices in confusion_groups.items():
    if len(indices) == 0:
        continue
    print(f"\n  --- {group_name} (n={len(indices)}) ---", flush=True)
    sub = train.iloc[indices]

    # Key characteristics
    speeds = []; track_lens = []; rcs_means = []; sizes = []; alt_means = []
    for _, row in sub.iterrows():
        pts = parse_ewkb_4d(row["trajectory"])
        rcs = np.array([p[3] for p in pts])
        alts = np.array([p[2] for p in pts])
        rcs_means.append(np.mean(rcs))
        alt_means.append(np.mean(alts))
        track_lens.append(len(pts))
        sizes.append(row["radar_bird_size"])
        speeds.append(row["airspeed"])

    print(f"    speed: {np.mean(speeds):.1f} m/s (range {np.min(speeds):.1f}-{np.max(speeds):.1f})")
    print(f"    track_len: {np.mean(track_lens):.0f} pts (range {np.min(track_lens)}-{np.max(track_lens)})")
    print(f"    rcs: {np.mean(rcs_means):.1f} dB (range {np.min(rcs_means):.1f} to {np.max(rcs_means):.1f})")
    print(f"    alt: {np.mean(alt_means):.0f} m")
    print(f"    sizes: {pd.Series(sizes).value_counts().to_dict()}")
    print(f"    months: {pd.Series(months[indices]).value_counts().to_dict()}")

    # Does TabPFN fix any of these?
    tab_pred = oof_tabpfn[indices].argmax(axis=1)
    fixed = (tab_pred == CORM) & (e175_pred[np.isin(corm_idx, indices)] != CORM) if len(indices) > 0 else []
    # Simpler: just count how many TabPFN gets right in this group
    tab_correct = np.sum(tab_pred == CORM)
    e175_correct = np.sum(e175_pred[np.isin(corm_idx, indices)] == CORM) if group_name != "Correct (Corm)" else len(indices)
    print(f"    E175 correct: {e175_correct}/{len(indices)}, TabPFN correct: {tab_correct}/{len(indices)}")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("3. OBSERVATION CONTEXT — CO-OCCURRENCE", flush=True)
print("=" * 90, flush=True)

# Do Cormorants share primary_observation_id with other Cormorants?
corm_groups = groups[corm_idx]
unique_corm_groups = np.unique(corm_groups)
print(f"  {len(corm_idx)} Cormorants in {len(unique_corm_groups)} observation groups", flush=True)

# For each group, how many tracks and what classes?
for grp in unique_corm_groups:
    grp_mask = groups == grp
    grp_classes = y[grp_mask]
    n_corm = (grp_classes == CORM).sum()
    n_total = len(grp_classes)
    if n_total > 1:
        other_classes = [CLASSES[c] for c in grp_classes if c != CORM]
        print(f"    Group {grp}: {n_total} tracks ({n_corm} Corm, others: {other_classes})")

# Temporal proximity: are Cormorants observed near each other in time?
corm_times = pd.to_datetime(train.iloc[corm_idx]["timestamp_start_radar_utc"]).values
corm_times_sorted = np.sort(corm_times)
time_gaps = np.diff(corm_times_sorted).astype("timedelta64[s]").astype(float)
print(f"\n  Time gaps between consecutive Cormorant observations:")
print(f"    <60s:  {(time_gaps < 60).sum()} pairs")
print(f"    <5min: {(time_gaps < 300).sum()} pairs")
print(f"    <30min: {(time_gaps < 1800).sum()} pairs")
print(f"    <1hr: {(time_gaps < 3600).sum()} pairs")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("4. TEST DATA: DO CORMORANT-LIKE SAMPLES CLUSTER?", flush=True)
print("=" * 90, flush=True)

# Load test features
test_feats = pd.read_pickle("data/_cached_test_features_v3.pkl")
X_test = np.nan_to_num(test_feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_train = np.nan_to_num(feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)

# Find test samples most similar to Cormorants
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

scaler = StandardScaler()
X_all_s = scaler.fit_transform(np.vstack([X_train, X_test]))
X_train_s = X_all_s[:len(X_train)]
X_test_s = X_all_s[len(X_train):]

# Cormorant centroid in train
corm_centroid = X_train_s[y == CORM].mean(axis=0)

# Distance of each test sample to Cormorant centroid
test_dist_to_corm = np.linalg.norm(X_test_s - corm_centroid, axis=1)

# Top 30 closest test samples
top30 = np.argsort(test_dist_to_corm)[:30]
test_months = pd.to_datetime(test["timestamp_start_radar_utc"]).dt.month.values

print(f"  Top 30 test samples closest to Cormorant centroid:", flush=True)
print(f"  TabPFN P(Corm) distribution for these:", flush=True)
for rank, idx in enumerate(top30):
    p_corm = test_tabpfn[idx, CORM]
    p_top = test_tabpfn[idx].max()
    top_cls = CLASSES[test_tabpfn[idx].argmax()]
    dist = test_dist_to_corm[idx]
    m = test_months[idx]
    size = test.iloc[idx]["radar_bird_size"]
    print(f"    #{rank+1}: dist={dist:.2f} P(Corm)={p_corm:.4f} top={top_cls}({p_top:.3f}) month={m} size={size}")

# How many test samples are "in Cormorant territory"?
# Define as: closer to Cormorant centroid than median Cormorant-to-centroid distance
corm_dists = np.linalg.norm(X_train_s[y == CORM] - corm_centroid, axis=1)
threshold = np.median(corm_dists)
n_test_in_territory = (test_dist_to_corm < threshold).sum()
print(f"\n  Test samples within Cormorant territory (< median Corm distance {threshold:.2f}):")
print(f"    {n_test_in_territory} / {len(test)} ({n_test_in_territory/len(test)*100:.1f}%)")
territory_months = test_months[test_dist_to_corm < threshold]
print(f"    By month: {pd.Series(territory_months).value_counts().sort_index().to_dict()}")

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("5. FFT POWER SPECTRUM OF RCS", flush=True)
print("=" * 90, flush=True)

from scipy.fft import rfft, rfftfreq

def rcs_spectrum_features(row):
    pts = parse_ewkb_4d(row["trajectory"])
    rcs = np.array([p[3] for p in pts])
    n = len(rcs)
    if n < 10:
        return {f"fft_{k}": 0 for k in ["peak_freq", "peak_power_ratio", "spectral_entropy",
                "low_freq_power", "mid_freq_power", "high_freq_power", "spectral_flatness"]}

    # Detrend
    rcs_dt = rcs - np.polyval(np.polyfit(np.arange(n), rcs, 1), np.arange(n))

    # FFT
    fft_vals = np.abs(rfft(rcs_dt))
    freqs = rfftfreq(n, d=1.0)  # 1 Hz sampling

    # Skip DC
    fft_vals = fft_vals[1:]
    freqs = freqs[1:]

    if len(fft_vals) == 0:
        return {f"fft_{k}": 0 for k in ["peak_freq", "peak_power_ratio", "spectral_entropy",
                "low_freq_power", "mid_freq_power", "high_freq_power", "spectral_flatness"]}

    psd = fft_vals ** 2
    psd_norm = psd / max(psd.sum(), 1e-10)

    f = {}
    # Peak frequency
    peak_idx = np.argmax(fft_vals)
    f["fft_peak_freq"] = freqs[peak_idx]
    f["fft_peak_power_ratio"] = fft_vals[peak_idx] / max(np.mean(fft_vals), 1e-10)

    # Spectral entropy (flat spectrum = high entropy = random; peaked = low entropy = periodic)
    f["fft_spectral_entropy"] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # Band powers (fraction in low/mid/high frequency)
    low = freqs < 0.1
    mid = (freqs >= 0.1) & (freqs < 0.3)
    high = freqs >= 0.3
    total = psd.sum()
    f["fft_low_freq_power"] = psd[low].sum() / max(total, 1e-10) if low.any() else 0
    f["fft_mid_freq_power"] = psd[mid].sum() / max(total, 1e-10) if mid.any() else 0
    f["fft_high_freq_power"] = psd[high].sum() / max(total, 1e-10) if high.any() else 0

    # Spectral flatness (geometric mean / arithmetic mean of PSD)
    # Flat spectrum (noise) -> 1.0; peaked spectrum (tonal) -> 0.0
    log_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    f["fft_spectral_flatness"] = log_mean / max(np.mean(psd), 1e-10)

    return f

print("  Computing FFT features...", flush=True)
fft_feats = [rcs_spectrum_features(row) for _, row in train.iterrows()]
fft_df = pd.DataFrame(fft_feats)

y_corm = (y == CORM).astype(int)
print(f"\n  FFT feature ranking for Cormorant detection:")
for col in fft_df.columns:
    vals = np.nan_to_num(fft_df[col].values, nan=0, posinf=0, neginf=0)
    try:
        ap_p = average_precision_score(y_corm, vals)
        ap_n = average_precision_score(y_corm, -vals)
        ap = max(ap_p, ap_n)
        auc = roc_auc_score(y_corm, vals if ap_p >= ap_n else -vals)
        cm = np.median(vals[y == CORM])
        gm = np.median(vals[y == GULL])
        print(f"    {col:30s} AP={ap:.4f} AUC={auc:.3f} Corm={cm:.4f} Gull={gm:.4f}")
    except:
        pass

# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90, flush=True)
print("6. MULTI-RESOLUTION: FIRST HALF vs SECOND HALF", flush=True)
print("=" * 90, flush=True)

def half_comparison(row):
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    rcs = np.array([p[3] for p in pts])
    alts = np.array([p[2] for p in pts])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    n = len(pts)
    if n < 10:
        return {k: 0 for k in ["half_rcs_diff", "half_speed_diff", "half_alt_diff",
                "half_rcs_var_ratio", "half_consistency"]}

    h = n // 2
    dists = np.array([haversine(lons[j],lats[j],lons[j+1],lats[j+1]) for j in range(n-1)])
    dt = np.maximum(np.diff(times), 0.001)
    speeds = dists / dt
    hs = len(speeds) // 2

    f = {}
    # Mean differences between halves (Cormorants should be CONSISTENT)
    f["half_rcs_diff"] = abs(np.mean(rcs[:h]) - np.mean(rcs[h:]))
    f["half_speed_diff"] = abs(np.mean(speeds[:hs]) - np.mean(speeds[hs:])) if hs > 0 else 0
    f["half_alt_diff"] = abs(np.mean(alts[:h]) - np.mean(alts[h:]))

    # Variance ratio between halves
    v1 = np.var(rcs[:h]); v2 = np.var(rcs[h:])
    f["half_rcs_var_ratio"] = min(v1, v2) / max(max(v1, v2), 0.001)

    # Combined consistency (all halves similar = consistent flight = Cormorant)
    f["half_consistency"] = 1.0 / (1 + f["half_rcs_diff"] + f["half_speed_diff"] + f["half_alt_diff"])

    return f

print("  Computing half-comparison features...", flush=True)
half_feats = [half_comparison(row) for _, row in train.iterrows()]
half_df = pd.DataFrame(half_feats)

print(f"\n  Half-comparison feature ranking:")
for col in half_df.columns:
    vals = np.nan_to_num(half_df[col].values, nan=0, posinf=0, neginf=0)
    try:
        ap_p = average_precision_score(y_corm, vals)
        ap_n = average_precision_score(y_corm, -vals)
        ap = max(ap_p, ap_n)
        auc = roc_auc_score(y_corm, vals if ap_p >= ap_n else -vals)
        cm = np.median(vals[y == CORM])
        gm = np.median(vals[y == GULL])
        print(f"    {col:25s} AP={ap:.4f} AUC={auc:.3f} Corm={cm:.4f} Gull={gm:.4f}")
    except:
        pass

print("\nDone.", flush=True)
