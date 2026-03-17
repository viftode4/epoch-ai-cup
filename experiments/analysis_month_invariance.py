"""Analysis: Month Invariance Study.

Parts:
  A - Per-month class distributions
  B - Feature stability across months (variance ratio)
  C - RCS wingbeat frequency analysis (FFT per track)
  D - Privileged column analysis (n_birds_observed, radar_bird_size)
  E - Turn radius computation and class separation
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, parse_ewkb_4d, parse_trajectory_time
from src.features import build_features, ALL_TEMPORAL, SIZE_MAP

ROOT = Path(__file__).resolve().parent.parent

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

TD_FEATURES = [
    "td_heading_local_var", "td_speed_consistency", "td_speed_autocorr",
    "td_speed_slope", "td_alt_smoothness", "td_heading_change_rate",
    "td_rcs_trend", "td_speed_variability",
]

WEATHER_SOLAR = [
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]

ALL_FEATURES = KEEP_FEATURES + TD_FEATURES


def add_weather_solar(feats):
    train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
    for col in train_weather.columns:
        feats[f"wx_{col}"] = train_weather[col].values
    train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
    for col in train_solar.columns:
        feats[f"sol_{col}"] = train_solar[col].values
    return feats


def part_a(train_df, y, months):
    """Per-month class distributions."""
    print("\n" + "=" * 70, flush=True)
    print("PART A: Per-Month Class Distributions".center(70), flush=True)
    print("=" * 70, flush=True)

    unique_months = sorted(np.unique(months))

    # Header
    header = f"{'Class':<18}"
    for m in unique_months:
        header += f"  M{m:02d}"
    header += "  Total  #Months"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        row = f"{cls:<18}"
        n_months_present = 0
        for m in unique_months:
            mmask = mask & (months == m)
            cnt = mmask.sum()
            row += f"  {cnt:4d}"
            if cnt > 0:
                n_months_present += 1
        row += f"  {mask.sum():5d}  {n_months_present:7d}"
        print(row, flush=True)

    # Totals
    row = f"{'TOTAL':<18}"
    for m in unique_months:
        row += f"  {(months == m).sum():4d}"
    row += f"  {len(y):5d}"
    print(row, flush=True)

    # Proportions per month
    print("\nClass proportions per month (%):", flush=True)
    header = f"{'Class':<18}"
    for m in unique_months:
        header += f"  M{m:02d}  "
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        row = f"{cls:<18}"
        for m in unique_months:
            mmask = months == m
            total_m = mmask.sum()
            cnt = (mask & mmask).sum()
            pct = 100.0 * cnt / total_m if total_m > 0 else 0
            row += f" {pct:5.1f}%"
        print(row, flush=True)

    # Flag sparse classes
    print("\n--- Classes present in only 1-2 months (WILL COLLAPSE on unseen months) ---", flush=True)
    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        present = []
        for m in unique_months:
            if (mask & (months == m)).sum() > 0:
                present.append(m)
        if len(present) <= 2:
            print(f"  {cls}: only in months {present} ({mask.sum()} samples)", flush=True)

    # Classes with very skewed distribution
    print("\n--- Classes with >80% of samples in a single month ---", flush=True)
    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        total = mask.sum()
        for m in unique_months:
            cnt = (mask & (months == m)).sum()
            if cnt / total > 0.8:
                print(f"  {cls}: {cnt}/{total} ({100*cnt/total:.0f}%) in month {m}", flush=True)


def part_b(train_df, y, months, feats):
    """Feature stability across months."""
    print("\n" + "=" * 70, flush=True)
    print("PART B: Feature Stability Across Months".center(70), flush=True)
    print("=" * 70, flush=True)

    unique_months = sorted(np.unique(months))
    results = []

    for feat_name in ALL_FEATURES:
        if feat_name not in feats.columns:
            continue

        vals = feats[feat_name].values.astype(float)

        # For each class, compute across-month variance vs within-month variance
        across_vars = []
        within_vars = []

        for ci in range(len(CLASSES)):
            cmask = (y == ci)
            if cmask.sum() < 5:
                continue

            # Per-month means for this class
            month_means = []
            month_within_vars = []
            for m in unique_months:
                mm = cmask & (months == m)
                if mm.sum() >= 2:
                    month_means.append(np.mean(vals[mm]))
                    month_within_vars.append(np.var(vals[mm]))

            if len(month_means) >= 2:
                across_vars.append(np.var(month_means))
                within_vars.append(np.mean(month_within_vars))

        if len(across_vars) == 0:
            continue

        mean_across = np.mean(across_vars)
        mean_within = np.mean(within_vars)
        # Stability ratio: across / within. High = bad (feature changes across months)
        ratio = mean_across / max(mean_within, 1e-10)

        is_wx_sol = feat_name in WEATHER_SOLAR or feat_name.startswith("wx_") or feat_name.startswith("sol_")
        results.append({
            "feature": feat_name,
            "across_month_var": mean_across,
            "within_month_var": mean_within,
            "instability_ratio": ratio,
            "is_month_proxy": is_wx_sol,
        })

    res_df = pd.DataFrame(results).sort_values("instability_ratio")

    print(f"\nAnalyzed {len(res_df)} features", flush=True)

    print("\n--- TOP 10 MOST STABLE features (low across-month variance) ---", flush=True)
    print(f"{'Rank':<5} {'Feature':<28} {'Ratio':>10} {'AcrossVar':>12} {'WithinVar':>12} {'Proxy?':>7}", flush=True)
    print("-" * 76, flush=True)
    for i, (_, row) in enumerate(res_df.head(10).iterrows()):
        tag = "YES" if row["is_month_proxy"] else ""
        print(f"{i+1:<5} {row['feature']:<28} {row['instability_ratio']:10.4f} "
              f"{row['across_month_var']:12.4f} {row['within_month_var']:12.4f} {tag:>7}", flush=True)

    print("\n--- TOP 10 LEAST STABLE features (high across-month variance) ---", flush=True)
    print(f"{'Rank':<5} {'Feature':<28} {'Ratio':>10} {'AcrossVar':>12} {'WithinVar':>12} {'Proxy?':>7}", flush=True)
    print("-" * 76, flush=True)
    for i, (_, row) in enumerate(res_df.tail(10).iloc[::-1].iterrows()):
        tag = "YES" if row["is_month_proxy"] else ""
        print(f"{i+1:<5} {row['feature']:<28} {row['instability_ratio']:10.4f} "
              f"{row['across_month_var']:12.4f} {row['within_month_var']:12.4f} {tag:>7}", flush=True)

    # Count weather/solar in top-10 least stable
    top10_unstable = res_df.tail(10)
    n_wx = top10_unstable["is_month_proxy"].sum()
    print(f"\n  Weather/solar features in top-10 least stable: {n_wx}/10", flush=True)

    # Full ranking
    print("\n--- FULL RANKING (most stable to least stable) ---", flush=True)
    print(f"{'Rank':<5} {'Feature':<28} {'Ratio':>10} {'Proxy?':>7}", flush=True)
    for i, (_, row) in enumerate(res_df.iterrows()):
        tag = "*WX*" if row["is_month_proxy"] else ""
        print(f"{i+1:<5} {row['feature']:<28} {row['instability_ratio']:10.4f} {tag:>7}", flush=True)


def part_c(train_df, y, months):
    """RCS wingbeat frequency analysis."""
    print("\n" + "=" * 70, flush=True)
    print("PART C: RCS Wingbeat Frequency Analysis".center(70), flush=True)
    print("=" * 70, flush=True)

    peak_freqs = []
    classes_arr = []
    months_arr = []

    for idx, (_, r) in enumerate(train_df.iterrows()):
        if idx % 500 == 0:
            print(f"  FFT: {idx}/{len(train_df)}", flush=True)

        pts = parse_ewkb_4d(r.trajectory)
        times = parse_trajectory_time(r.trajectory_time)
        rcs = np.array([p[3] for p in pts])
        n = len(pts)

        if n < 8:
            peak_freqs.append(np.nan)
            classes_arr.append(y[idx])
            months_arr.append(months[idx])
            continue

        try:
            # Compute actual sampling rate from trajectory_time
            dt_median = np.median(np.diff(times))
            if dt_median <= 0:
                dt_median = 1.0

            # Resample to uniform spacing
            uniform_t = np.linspace(times[0], times[-1], n)
            fs = 1.0 / max(uniform_t[1] - uniform_t[0], 0.01)
            rcs_uniform = interp1d(times, rcs, kind="linear", fill_value="extrapolate")(uniform_t)

            # Remove DC
            rcs_detrended = rcs_uniform - np.mean(rcs_uniform)

            freqs, psd = welch(rcs_detrended, fs=fs, nperseg=min(n, 32))
            # Skip DC bin (index 0)
            if len(psd) > 1:
                peak_idx = np.argmax(psd[1:]) + 1
                peak_freqs.append(freqs[peak_idx])
            else:
                peak_freqs.append(np.nan)
        except Exception:
            peak_freqs.append(np.nan)

        classes_arr.append(y[idx])
        months_arr.append(months[idx])

    print(f"  FFT: {len(train_df)}/{len(train_df)} done", flush=True)

    peak_freqs = np.array(peak_freqs)
    classes_arr = np.array(classes_arr)
    months_arr = np.array(months_arr)

    unique_months = sorted(np.unique(months))

    # Per-class peak freq distribution
    print("\n--- Peak Wingbeat Frequency (Hz) per Class ---", flush=True)
    header = f"{'Class':<18} {'Overall':>10}"
    for m in unique_months:
        header += f"  M{m:02d}          "
    print(header, flush=True)
    print("-" * len(header), flush=True)

    stability_scores = {}
    for ci, cls in enumerate(CLASSES):
        cmask = (classes_arr == ci) & ~np.isnan(peak_freqs)
        if cmask.sum() == 0:
            print(f"{cls:<18} {'N/A':>10}", flush=True)
            continue

        overall_med = np.median(peak_freqs[cmask])
        overall_iqr = np.percentile(peak_freqs[cmask], 75) - np.percentile(peak_freqs[cmask], 25)
        row = f"{cls:<18} {overall_med:5.3f}+/-{overall_iqr:5.3f}"

        month_medians = []
        for m in unique_months:
            mm = cmask & (months_arr == m)
            if mm.sum() >= 2:
                med = np.median(peak_freqs[mm])
                iqr = np.percentile(peak_freqs[mm], 75) - np.percentile(peak_freqs[mm], 25)
                row += f"  {med:5.3f}+/-{iqr:5.3f}"
                month_medians.append(med)
            else:
                row += f"  {'N/A':>14}"

        print(row, flush=True)

        # Stability: std of monthly medians / overall median
        if len(month_medians) >= 2:
            stability = np.std(month_medians) / max(abs(overall_med), 1e-6)
            stability_scores[cls] = stability

    print("\n--- Wingbeat Freq Cross-Month Stability (lower = more stable) ---", flush=True)
    for cls, score in sorted(stability_scores.items(), key=lambda x: x[1]):
        tag = "STABLE" if score < 0.3 else "UNSTABLE"
        print(f"  {cls:<18} CV={score:.3f}  [{tag}]", flush=True)


def part_d(train_df, y, months):
    """Privileged column analysis."""
    print("\n" + "=" * 70, flush=True)
    print("PART D: Privileged Column Analysis".center(70), flush=True)
    print("=" * 70, flush=True)

    unique_months = sorted(np.unique(months))

    # n_birds_observed
    print("\n--- n_birds_observed per class ---", flush=True)
    print(f"{'Class':<18} {'Mean':>8} {'Med':>6} {'Std':>8} {'Min':>6} {'Max':>6} {'N>1%':>6}", flush=True)
    print("-" * 62, flush=True)
    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        vals = train_df.loc[mask, "n_birds_observed"].values
        vals_clean = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else vals
        n_gt1 = (vals_clean > 1).sum() if len(vals_clean) > 0 else 0
        pct_gt1 = 100 * n_gt1 / max(len(vals_clean), 1)
        print(f"{cls:<18} {np.nanmean(vals):8.2f} {np.nanmedian(vals):6.1f} "
              f"{np.nanstd(vals):8.2f} {np.nanmin(vals):6.0f} {np.nanmax(vals):6.0f} "
              f"{pct_gt1:5.1f}%", flush=True)

    # radar_bird_size per class
    print("\n--- radar_bird_size distribution per class ---", flush=True)
    sizes = train_df["radar_bird_size"].unique()
    header = f"{'Class':<18}"
    for s in sorted(sizes):
        header += f"  {s:<14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for ci, cls in enumerate(CLASSES):
        mask = (y == ci)
        total = mask.sum()
        row = f"{cls:<18}"
        for s in sorted(sizes):
            cnt = (train_df.loc[mask, "radar_bird_size"] == s).sum()
            pct = 100 * cnt / total
            row += f"  {cnt:4d} ({pct:4.1f}%)  "
        print(row, flush=True)

    # radar_bird_size per month (does it shift?)
    print("\n--- radar_bird_size distribution per MONTH (shift check) ---", flush=True)
    header = f"{'Month':<10}"
    for s in sorted(sizes):
        header += f"  {s:<14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for m in unique_months:
        mmask = (months == m)
        total = mmask.sum()
        row = f"M{m:02d} (N={total:4d})"
        for s in sorted(sizes):
            cnt = (train_df.loc[mmask, "radar_bird_size"] == s).sum()
            pct = 100 * cnt / total
            row += f"  {cnt:4d} ({pct:4.1f}%)  "
        print(row, flush=True)

    # Check if radar_bird_size is in our feature set
    print("\n--- Is radar_bird_size in feature set? ---", flush=True)
    has_rbs = "radar_bird_size" in KEEP_FEATURES
    has_size_x = any("size_x" in f for f in KEEP_FEATURES)
    has_rcs_for = "rcs_for_size" in KEEP_FEATURES
    print(f"  radar_bird_size encoded via SIZE_MAP: {has_rbs}", flush=True)
    print(f"  SIZE_MAP values: {SIZE_MAP}", flush=True)
    print(f"  size_x_alt in features: {has_size_x}", flush=True)
    print(f"  rcs_for_size in features: {has_rcs_for}", flush=True)
    size_feats = [f for f in KEEP_FEATURES if "size" in f.lower()]
    print(f"  All size-related features: {size_feats}", flush=True)


def part_e(train_df, y, months):
    """Turn radius computation."""
    print("\n" + "=" * 70, flush=True)
    print("PART E: Turn Radius Analysis".center(70), flush=True)
    print("=" * 70, flush=True)

    turn_radii = []
    for idx, (_, r) in enumerate(train_df.iterrows()):
        if idx % 500 == 0:
            print(f"  TurnRadius: {idx}/{len(train_df)}", flush=True)

        pts = parse_ewkb_4d(r.trajectory)
        times = parse_trajectory_time(r.trajectory_time)
        n = len(pts)

        if n < 4:
            turn_radii.append(np.nan)
            continue

        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])

        raw_dt = np.diff(times)
        dt = np.maximum(raw_dt, 0.001)
        valid = raw_dt >= 0.5

        # Distances
        dlat = np.diff(lats) * 111000.0
        dlon = np.diff(lons) * 67000.0
        dists = np.sqrt(dlat**2 + dlon**2)
        speeds = dists / dt

        # Bearings and bearing changes
        bearings = np.arctan2(dlat, dlon)
        if len(bearings) > 1:
            bearing_changes = np.arctan2(np.sin(np.diff(bearings)), np.cos(np.diff(bearings)))
            # dt for bearing changes (between consecutive segments)
            dt_bc = 0.5 * (dt[:-1] + dt[1:])
            angular_vel = np.abs(bearing_changes) / np.maximum(dt_bc, 0.001)

            # Use valid segments for speed (midpoint speeds)
            mid_speeds = 0.5 * (speeds[:-1] + speeds[1:])

            # Filter out very small angular velocities (straight flight -> infinite radius)
            mask = angular_vel > 0.01  # ~0.6 deg/s threshold
            if valid[:-1].sum() > 0:
                # Also require valid dt for at least one segment
                mask = mask & (raw_dt[:-1] >= 0.5) | (raw_dt[1:] >= 0.5)

            if mask.sum() > 0:
                radii = mid_speeds[mask] / angular_vel[mask]
                # Clip extreme values
                radii = np.clip(radii, 0, 10000)
                turn_radii.append(np.median(radii))
            else:
                turn_radii.append(np.nan)
        else:
            turn_radii.append(np.nan)

    print(f"  TurnRadius: {len(train_df)}/{len(train_df)} done", flush=True)

    turn_radii = np.array(turn_radii)
    unique_months = sorted(np.unique(months))

    # Per-class turn radius
    print("\n--- Turn Radius (m) per Class ---", flush=True)
    print(f"{'Class':<18} {'Median':>8} {'Mean':>8} {'Std':>10} {'Q25':>8} {'Q75':>8} {'N_valid':>8}", flush=True)
    print("-" * 70, flush=True)

    class_medians = {}
    for ci, cls in enumerate(CLASSES):
        cmask = (y == ci) & ~np.isnan(turn_radii)
        if cmask.sum() < 2:
            print(f"{cls:<18} {'N/A':>8}", flush=True)
            continue
        vals = turn_radii[cmask]
        med = np.median(vals)
        class_medians[cls] = med
        print(f"{cls:<18} {med:8.1f} {np.mean(vals):8.1f} {np.std(vals):10.1f} "
              f"{np.percentile(vals, 25):8.1f} {np.percentile(vals, 75):8.1f} {cmask.sum():8d}", flush=True)

    # Hard pairs separation
    print("\n--- Hard Pair Separation (turn radius) ---", flush=True)
    pairs = [
        ("Gulls", "Waders"),
        ("Birds of Prey", "Ducks"),
        ("Pigeons", "Ducks"),
        ("Gulls", "Birds of Prey"),
        ("Songbirds", "Pigeons"),
    ]
    for cls_a, cls_b in pairs:
        if cls_a in class_medians and cls_b in class_medians:
            diff = abs(class_medians[cls_a] - class_medians[cls_b])
            avg = 0.5 * (class_medians[cls_a] + class_medians[cls_b])
            rel_diff = diff / max(avg, 1e-6) * 100
            print(f"  {cls_a} vs {cls_b}: medians {class_medians[cls_a]:.1f} vs {class_medians[cls_b]:.1f} "
                  f"(diff={diff:.1f}m, {rel_diff:.0f}% relative)", flush=True)

    # Cross-month stability of turn radius
    print("\n--- Turn Radius Cross-Month Stability per Class ---", flush=True)
    print(f"{'Class':<18}", end="", flush=True)
    for m in unique_months:
        print(f"  M{m:02d}    ", end="", flush=True)
    print("  CV", flush=True)
    print("-" * 60, flush=True)

    for ci, cls in enumerate(CLASSES):
        cmask_base = (y == ci) & ~np.isnan(turn_radii)
        if cmask_base.sum() < 5:
            continue
        row = f"{cls:<18}"
        month_meds = []
        for m in unique_months:
            mm = cmask_base & (months == m)
            if mm.sum() >= 2:
                med = np.median(turn_radii[mm])
                row += f"  {med:6.1f}"
                month_meds.append(med)
            else:
                row += f"  {'N/A':>6}"

        if len(month_meds) >= 2:
            cv = np.std(month_meds) / max(np.mean(month_meds), 1e-6)
            row += f"  {cv:.3f}"
        print(row, flush=True)


def main():
    print("=" * 70, flush=True)
    print("Month Invariance Analysis".center(70), flush=True)
    print("=" * 70, flush=True)

    # Load data
    train_df = load_train()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    unique_months = sorted(np.unique(months))
    print(f"Train months: {unique_months}", flush=True)
    print(f"N samples: {len(y)}", flush=True)

    # ========== PART A ==========
    part_a(train_df, y, months)

    # ========== Build features for PART B ==========
    print("\nBuilding features (core + rcs_fft + tabular + targeted + weakclass + temporal_dynamics)...", flush=True)
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "weakclass", "temporal_dynamics"]
    feats = build_features(train_df, feature_sets=feat_sets)
    feats = add_weather_solar(feats)

    # ========== PART B ==========
    part_b(train_df, y, months, feats)

    # ========== PART C ==========
    part_c(train_df, y, months)

    # ========== PART D ==========
    part_d(train_df, y, months)

    # ========== PART E ==========
    part_e(train_df, y, months)

    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS COMPLETE".center(70), flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
