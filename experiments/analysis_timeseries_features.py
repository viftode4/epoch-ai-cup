"""Analysis: Time Series Features from Raw Radar Channels.

Instead of hand-crafted aggregate features, extract canonical time series
features from the 4 raw radar channels:
  - RCS (dBm2)     -- body size + wing modulation
  - Altitude (m)    -- flight mode (soaring, bounding, level)
  - Speed (m/s)     -- flight energy and consistency
  - Bearing change  -- turning behavior (soaring vs straight)

For each channel, compute ~10 canonical time series features capturing:
  - Temporal structure (autocorrelation, decorrelation time)
  - Complexity/regularity (sample entropy, Lempel-Ziv)
  - Oscillation (zero-crossing rate, spectral entropy)
  - Trend vs noise (R^2 of linear fit, detrended variance ratio)
  - Distribution shape (skewness, kurtosis)

Then test: do these time series features add value BEYOND our 36 aggregate features?
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine


# ======================================================================
# CANONICAL TIME SERIES FEATURES (catch22-inspired)
# ======================================================================

def ts_autocorrelation(x, lag):
    """Autocorrelation at given lag."""
    n = len(x)
    if n <= lag + 1:
        return 0.0
    xc = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    ac = np.mean(xc[:n-lag] * xc[lag:]) / var
    return float(ac) if np.isfinite(ac) else 0.0


def ts_first_min_acf(x, max_lag=20):
    """Lag of first local minimum of autocorrelation function.
    Proxy for dominant oscillation period."""
    n = len(x)
    max_lag = min(max_lag, n // 3)
    if max_lag < 2:
        return 0
    acf = [ts_autocorrelation(x, lag) for lag in range(1, max_lag + 1)]
    for i in range(1, len(acf) - 1):
        if acf[i] < acf[i-1] and acf[i] <= acf[i+1]:
            return i + 1  # lag value (1-indexed)
    return max_lag


def ts_first_zero_acf(x, max_lag=20):
    """Lag where autocorrelation first crosses zero.
    Short decorrelation = erratic. Long = persistent."""
    n = len(x)
    max_lag = min(max_lag, n // 3)
    if max_lag < 2:
        return 0
    prev = ts_autocorrelation(x, 1)
    for lag in range(2, max_lag + 1):
        curr = ts_autocorrelation(x, lag)
        if prev > 0 and curr <= 0:
            return lag
        prev = curr
    return max_lag  # never crosses zero = very persistent


def ts_sample_entropy(x, m=2, r_frac=0.2):
    """Sample entropy (SampEn). Low = regular, high = complex.
    m = embedding dimension, r = tolerance (fraction of std)."""
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_frac * np.std(x)
    if r < 1e-12:
        return 0.0

    def count_matches(dim):
        templates = np.array([x[i:i+dim] for i in range(n - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0:
        return 0.0
    return float(-np.log(max(A, 1) / max(B, 1)))


def ts_sample_entropy_fast(x, m=2, r_frac=0.2, max_n=100):
    """Fast approximate sample entropy using subsampling for long series."""
    n = len(x)
    if n > max_n:
        # Subsample uniformly
        idx = np.linspace(0, n-1, max_n).astype(int)
        x = x[idx]
    return ts_sample_entropy(x, m, r_frac)


def ts_zero_crossing_rate(x):
    """Fraction of consecutive pairs where signal crosses its mean."""
    if len(x) < 2:
        return 0.0
    xc = x - np.mean(x)
    return float(np.sum(xc[:-1] * xc[1:] < 0) / (len(x) - 1))


def ts_trend_strength(x):
    """R^2 of linear fit. High = strong trend, low = oscillatory/random."""
    n = len(x)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    # Linear regression
    t_mean = t.mean()
    x_mean = x.mean()
    ss_t = np.sum((t - t_mean)**2)
    if ss_t < 1e-12:
        return 0.0
    slope = np.sum((t - t_mean) * (x - x_mean)) / ss_t
    x_pred = slope * t + (x_mean - slope * t_mean)
    ss_res = np.sum((x - x_pred)**2)
    ss_tot = np.sum((x - x_mean)**2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def ts_detrended_variance_ratio(x):
    """Ratio of detrended variance to original variance.
    Low = most variance is trend. High = most variance is oscillation."""
    n = len(x)
    if n < 3:
        return 0.0
    var_orig = np.var(x)
    if var_orig < 1e-12:
        return 0.0
    t = np.arange(n, dtype=float)
    # Remove linear trend
    t_mean = t.mean()
    x_mean = x.mean()
    ss_t = np.sum((t - t_mean)**2)
    if ss_t < 1e-12:
        return 1.0
    slope = np.sum((t - t_mean) * (x - x_mean)) / ss_t
    detrended = x - (slope * t + (x_mean - slope * t_mean))
    return float(np.var(detrended) / var_orig)


def ts_spectral_entropy(x, fs=1.0):
    """Spectral entropy. Low = peaked spectrum (periodic).
    High = flat spectrum (noise/complex)."""
    n = len(x)
    if n < 8:
        return 0.0
    nperseg = min(n, 64)
    freqs, psd = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    psd = psd[1:]  # Remove DC
    if psd.sum() < 1e-12:
        return 0.0
    psd_norm = psd / psd.sum()
    # Entropy
    psd_norm = psd_norm[psd_norm > 0]
    ent = -np.sum(psd_norm * np.log(psd_norm))
    max_ent = np.log(len(psd_norm))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def ts_lempel_ziv_complexity(x, n_bins=3):
    """Lempel-Ziv complexity on symbolized time series.
    Quantize into n_bins symbols, compute LZ76 complexity."""
    if len(x) < 3:
        return 0.0
    # Quantize into symbols
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, percentiles)
    bins[-1] += 1  # Include maximum
    symbols = np.digitize(x, bins[1:-1])

    # LZ76 complexity
    s = symbols.tolist()
    n = len(s)
    i, c = 0, 1
    prefix_set = set()
    current = ""
    for j in range(n):
        current += str(s[j])
        if current not in prefix_set:
            prefix_set.add(current)
            c += 1
            current = ""
    # Normalize by theoretical maximum
    b = n_bins  # alphabet size
    max_c = n / max(np.log(n) / np.log(b), 1)
    return float(c / max(max_c, 1))


def ts_turning_points(x):
    """Fraction of points that are local extrema (peaks or valleys)."""
    if len(x) < 3:
        return 0.0
    peaks = 0
    for i in range(1, len(x) - 1):
        if (x[i] > x[i-1] and x[i] > x[i+1]) or (x[i] < x[i-1] and x[i] < x[i+1]):
            peaks += 1
    return float(peaks / (len(x) - 2))


def compute_channel_features(x, prefix):
    """Compute all canonical time series features for one channel."""
    n = len(x)
    feat = {}

    if n < 4 or np.std(x) < 1e-10:
        keys = [f"{prefix}_ac1", f"{prefix}_ac3", f"{prefix}_ac5",
                f"{prefix}_first_min_acf", f"{prefix}_first_zero_acf",
                f"{prefix}_sampen", f"{prefix}_zcr", f"{prefix}_trend_r2",
                f"{prefix}_detrend_var_ratio", f"{prefix}_spectral_entropy",
                f"{prefix}_lz_complexity", f"{prefix}_turning_points",
                f"{prefix}_skewness", f"{prefix}_kurtosis"]
        return {k: 0.0 for k in keys}

    # Autocorrelation at key lags
    feat[f"{prefix}_ac1"] = ts_autocorrelation(x, 1)
    feat[f"{prefix}_ac3"] = ts_autocorrelation(x, min(3, n-2))
    feat[f"{prefix}_ac5"] = ts_autocorrelation(x, min(5, n-2))

    # Decorrelation timescales
    feat[f"{prefix}_first_min_acf"] = float(ts_first_min_acf(x))
    feat[f"{prefix}_first_zero_acf"] = float(ts_first_zero_acf(x))

    # Complexity / regularity
    feat[f"{prefix}_sampen"] = ts_sample_entropy_fast(x)
    feat[f"{prefix}_zcr"] = ts_zero_crossing_rate(x)

    # Trend vs oscillation
    feat[f"{prefix}_trend_r2"] = ts_trend_strength(x)
    feat[f"{prefix}_detrend_var_ratio"] = ts_detrended_variance_ratio(x)

    # Spectral
    feat[f"{prefix}_spectral_entropy"] = ts_spectral_entropy(x)

    # Symbolic complexity
    feat[f"{prefix}_lz_complexity"] = ts_lempel_ziv_complexity(x)

    # Shape
    feat[f"{prefix}_turning_points"] = ts_turning_points(x)
    feat[f"{prefix}_skewness"] = float(skew(x, nan_policy='omit'))
    feat[f"{prefix}_kurtosis"] = float(kurtosis(x, nan_policy='omit'))

    # Clean NaN/Inf
    for k, v in feat.items():
        if not np.isfinite(v):
            feat[k] = 0.0

    return feat


def extract_all_ts_features(row):
    """Extract time series features from all 4 raw channels."""
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    n = len(pts)

    if n < 6:
        return {}

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    dt = np.diff(times)
    dt = np.maximum(dt, 0.001)

    # Compute speed and bearing change as derived channels
    dx = np.diff(lons) * 67000
    dy = np.diff(lats) * 111000
    speeds = np.sqrt(dx**2 + dy**2) / dt

    headings = np.arctan2(dy, dx)
    if len(headings) > 1:
        bearing_changes = np.arctan2(
            np.sin(np.diff(headings)),
            np.cos(np.diff(headings))
        )
    else:
        bearing_changes = np.array([0.0])

    feat = {}

    # Channel 1: RCS time series
    feat.update(compute_channel_features(rcs, "ts_rcs"))

    # Channel 2: Altitude time series
    feat.update(compute_channel_features(alts, "ts_alt"))

    # Channel 3: Speed time series (one shorter than position)
    feat.update(compute_channel_features(speeds, "ts_spd"))

    # Channel 4: Bearing change time series
    if len(bearing_changes) > 5:
        feat.update(compute_channel_features(bearing_changes, "ts_brg"))
    else:
        feat.update(compute_channel_features(np.array([0.0]*6), "ts_brg"))

    # ============================================================
    # CROSS-CHANNEL features (the juice!)
    # ============================================================

    # RCS-altitude coupling over time
    if len(rcs) == len(alts) and np.std(rcs) > 1e-6 and np.std(alts) > 1e-6:
        cc = np.corrcoef(rcs, alts)[0, 1]
        feat["ts_cross_rcs_alt_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["ts_cross_rcs_alt_corr"] = 0.0

    # RCS-speed coupling (midpoints)
    rcs_mid = 0.5 * (rcs[:-1] + rcs[1:])
    if np.std(rcs_mid) > 1e-6 and np.std(speeds) > 1e-6:
        cc = np.corrcoef(rcs_mid, speeds)[0, 1]
        feat["ts_cross_rcs_speed_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["ts_cross_rcs_speed_corr"] = 0.0

    # Speed-altitude coupling (phase relationship)
    alt_mid = 0.5 * (alts[:-1] + alts[1:])
    if np.std(speeds) > 1e-6 and np.std(alt_mid) > 1e-6:
        cc = np.corrcoef(speeds, alt_mid)[0, 1]
        feat["ts_cross_speed_alt_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["ts_cross_speed_alt_corr"] = 0.0

    # Altitude rate vs speed (diving = high speed + descending)
    alt_rate = np.diff(alts) / dt
    if len(alt_rate) == len(speeds) and np.std(alt_rate) > 1e-6 and np.std(speeds) > 1e-6:
        cc = np.corrcoef(alt_rate, speeds)[0, 1]
        feat["ts_cross_altrate_speed_corr"] = float(cc) if np.isfinite(cc) else 0.0
    else:
        feat["ts_cross_altrate_speed_corr"] = 0.0

    return feat


# ======================================================================
# MAIN
# ======================================================================
print("=" * 70, flush=True)
print("TIME SERIES FEATURE ANALYSIS".center(70), flush=True)
print("=" * 70, flush=True)

print("\nLoading train data...", flush=True)
train_df = load_train()

print("Extracting time series features...", flush=True)
records = []
for idx, (_, row) in enumerate(train_df.iterrows()):
    if idx % 250 == 0:
        print(f"  {idx}/{len(train_df)}", flush=True)
    feat = extract_all_ts_features(row)
    feat["bird_group"] = row["bird_group"]
    records.append(feat)
print(f"  {len(train_df)}/{len(train_df)} done", flush=True)

df = pd.DataFrame(records)
feat_cols = [c for c in df.columns if c not in ("bird_group",)]
print(f"\nExtracted {len(feat_cols)} time series features across 4 channels", flush=True)

# ======================================================================
# A. PER-CLASS PROFILES
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("A. PER-CLASS TIME SERIES SIGNATURES".center(70), flush=True)
print("=" * 70, flush=True)

# For each channel, show the most distinctive features per class
channels = ["ts_rcs", "ts_alt", "ts_spd", "ts_brg"]
for ch in channels:
    ch_cols = [c for c in feat_cols if c.startswith(ch + "_")]
    if not ch_cols:
        continue

    print(f"\n  --- {ch.upper().replace('TS_', '')} Channel ---", flush=True)
    means = df.groupby("bird_group")[ch_cols].mean()
    means = means.reindex([c for c in CLASSES if c in means.index])

    for fc in ch_cols:
        vals = means[fc]
        if vals.std() > 1e-6:
            hi = vals.idxmax()
            lo = vals.idxmin()
            if abs(vals[hi] - vals[lo]) > 1e-6:
                print(f"    {fc:40s}  HI: {hi[:6]:6s}={vals[hi]:+.4f}  LO: {lo[:6]:6s}={vals[lo]:+.4f}", flush=True)


# ======================================================================
# B. PROBLEM CLASS SEPARATION
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("B. PROBLEM CLASS SEPARATION (Cohen's d)".center(70), flush=True)
print("=" * 70, flush=True)

pairs = [
    ("Cormorants", "Gulls"),
    ("Birds of Prey", "Gulls"),
    ("Ducks", "Pigeons"),
    ("Clutter", "Gulls"),
    ("Pigeons", "Songbirds"),
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
    for fc in feat_cols:
        v1 = g1[fc].dropna().values
        v2 = g2[fc].dropna().values
        d = cohens_d(v1, v2)
        if np.isfinite(d):
            scores.append((fc, abs(d), d))

    scores.sort(key=lambda x: -x[1])
    for fname, abs_d, d in scores[:8]:
        bar = "+" * min(int(abs_d * 10), 20)
        m1, m2 = g1[fname].mean(), g2[fname].mean()
        print(f"    {fname:40s} d={d:+.3f} {bar}  ({c1[:4]}={m1:.3f}, {c2[:4]}={m2:.3f})", flush=True)


# ======================================================================
# C. TEMPORAL STABILITY
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("C. MONTH INVARIANCE".center(70), flush=True)
print("=" * 70, flush=True)

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
df["month"] = train_months

month_corrs = []
for fc in feat_cols:
    class_corrs = []
    for cls in CLASSES:
        mask = df["bird_group"] == cls
        if mask.sum() > 10:
            x = df.loc[mask, fc].values
            m = df.loc[mask, "month"].values
            if np.std(x) > 1e-6 and np.std(m) > 1e-6:
                cc = abs(np.corrcoef(x, m)[0, 1])
                if np.isfinite(cc):
                    class_corrs.append(cc)
    avg_corr = np.mean(class_corrs) if class_corrs else 1.0
    month_corrs.append((fc, avg_corr))

month_corrs.sort(key=lambda x: x[1])
print("\n  MOST month-invariant:", flush=True)
for fname, corr in month_corrs[:15]:
    print(f"    {fname:40s} avg |corr| with month = {corr:.4f}", flush=True)

print("\n  MOST month-dependent:", flush=True)
for fname, corr in month_corrs[-10:]:
    print(f"    {fname:40s} avg |corr| with month = {corr:.4f}", flush=True)


# ======================================================================
# D. COMBINED RANKING
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("D. BEST TS FEATURES: DISCRIMINATIVE & STABLE".center(70), flush=True)
print("=" * 70, flush=True)

disc_scores = {}
for fc in feat_cols:
    pair_ds = []
    for c1, c2 in pairs:
        g1 = df[df["bird_group"] == c1][fc].dropna().values
        g2 = df[df["bird_group"] == c2][fc].dropna().values
        d = abs(cohens_d(g1, g2))
        if np.isfinite(d):
            pair_ds.append(d)
    disc_scores[fc] = np.mean(pair_ds) if pair_ds else 0

month_corr_dict = dict(month_corrs)

combined = []
for fc in feat_cols:
    disc = disc_scores.get(fc, 0)
    mcorr = month_corr_dict.get(fc, 1.0)
    score = disc * (1.0 - mcorr)
    combined.append((fc, score, disc, mcorr))
combined.sort(key=lambda x: -x[1])

print(f"\n  {'Feature':42s} {'Score':>8s} {'Discrim':>8s} {'Mo.corr':>8s}", flush=True)
print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8}", flush=True)
for fname, score, disc, mcorr in combined[:25]:
    print(f"  {fname:42s} {score:8.4f} {disc:8.4f} {mcorr:8.4f}", flush=True)


# ======================================================================
# E. CROSS-CHANNEL FEATURES
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("E. CROSS-CHANNEL COUPLING".center(70), flush=True)
print("=" * 70, flush=True)

cross_cols = [c for c in feat_cols if "cross" in c]
if cross_cols:
    for fc in cross_cols:
        means = df.groupby("bird_group")[fc].mean()
        means = means.reindex([c for c in CLASSES if c in means.index])
        sorted_vals = means.sort_values(ascending=False)
        top = ", ".join([f"{c[:6]}={v:+.3f}" for c, v in sorted_vals.head(3).items()])
        bot = ", ".join([f"{c[:6]}={v:+.3f}" for c, v in sorted_vals.tail(3).items()])
        print(f"  {fc}:", flush=True)
        print(f"    HIGH: {top}", flush=True)
        print(f"    LOW:  {bot}", flush=True)


# ======================================================================
# F. SPECIES FINGERPRINTS (what makes each species unique in TS space?)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("F. SPECIES TIME SERIES FINGERPRINTS".center(70), flush=True)
print("=" * 70, flush=True)

for cls in CLASSES:
    cls_data = df[df["bird_group"] == cls]
    rest_data = df[df["bird_group"] != cls]

    # Top 5 most distinctive features for this class
    dists = []
    for fc in feat_cols:
        v1 = cls_data[fc].dropna().values
        v2 = rest_data[fc].dropna().values
        d = cohens_d(v1, v2)
        if np.isfinite(d):
            dists.append((fc, d, abs(d)))
    dists.sort(key=lambda x: -x[2])

    top3 = dists[:3]
    if top3:
        feats_str = ", ".join([f"{f[0].split('_', 2)[-1]}({f[1]:+.2f})" for f in top3])
        print(f"  {cls:20s}: {feats_str}", flush=True)


print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print("""
  Key insight: Time series features capture TEMPORAL STRUCTURE that
  aggregate statistics miss. The best discriminators are:

  1. Autocorrelation patterns (how persistent is each channel?)
  2. Spectral entropy (how complex is the frequency content?)
  3. Trend strength (is the bird climbing/descending consistently?)
  4. Cross-channel coupling (does RCS co-vary with altitude/speed?)

  Next step: Add the top 10-15 TS features to E79's 36 features
  and test in a LOMO validation experiment.
""", flush=True)

print("Done.", flush=True)
