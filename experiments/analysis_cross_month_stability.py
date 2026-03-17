"""Phase 0: Cross-Month Feature Stability Analysis.

For each of the 36 features x 9 classes, compute:
1. Per-month mean: E[feature | class, month]
2. Cross-month CV: std_across_months / mean_across_months
3. Feature -> month mutual information (can this feature predict month?)
4. Weather/solar correlation with month vs class

Output: ranked list of features by month-dependence (leaky -> invariant).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train
from src.features import ALL_TEMPORAL, build_features

ROOT = Path(__file__).resolve().parent.parent

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

WEATHER_SOLAR = [
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]


def add_weather_solar(feats):
    """Add weather + solar features."""
    train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
    for col in train_weather.columns:
        feats[f"wx_{col}"] = train_weather[col].values
    train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
    for col in train_solar.columns:
        feats[f"sol_{col}"] = train_solar[col].values
    return feats


def main():
    print("=" * 70, flush=True)
    print("PHASE 0: Cross-Month Feature Stability Analysis".center(70), flush=True)
    print("=" * 70, flush=True)

    # Load data
    train_df = load_train()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    unique_months = sorted(np.unique(months))

    print(f"\nTrain months: {unique_months}", flush=True)
    print(f"N samples: {len(y)}", flush=True)

    # Build features
    print("\nBuilding features...", flush=True)
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    train_feats = add_weather_solar(train_feats)

    available = [f for f in KEEP_FEATURES if f in train_feats.columns]
    X = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    feature_names = available

    print(f"\nAnalyzing {len(feature_names)} features across {len(unique_months)} months x {len(CLASSES)} classes", flush=True)

    # ================================================================
    # 1. Per-feature cross-month CV (averaged across classes)
    # ================================================================
    print("\n" + "-" * 70, flush=True)
    print("1. Cross-Month CV per Feature (high = month-dependent = leaky)", flush=True)
    print("-" * 70, flush=True)

    feat_cv_scores = []
    for fi, fname in enumerate(feature_names):
        vals = X[:, fi]
        class_cvs = []
        for c in range(len(CLASSES)):
            month_means = []
            for m in unique_months:
                mask = (y == c) & (months == m)
                if mask.sum() >= 3:
                    month_means.append(np.mean(vals[mask]))
            if len(month_means) >= 2:
                arr = np.array(month_means)
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                cv = std_val / max(abs(mean_val), 1e-8)
                class_cvs.append(cv)
        avg_cv = np.mean(class_cvs) if class_cvs else 0
        feat_cv_scores.append((fname, avg_cv))

    feat_cv_scores.sort(key=lambda x: -x[1])

    print(f"\n{'Feature':<30} {'Avg CV':>8}  Category", flush=True)
    print("-" * 55, flush=True)
    for fname, cv in feat_cv_scores:
        cat = "WEATHER" if fname.startswith("wx_") else ("SOLAR" if fname.startswith("sol_") else "PHYSICS")
        marker = " *** LEAKY" if cv > 1.0 else (" ** MODERATE" if cv > 0.5 else "")
        print(f"  {fname:<28} {cv:>8.3f}  {cat}{marker}", flush=True)

    # ================================================================
    # 2. Feature -> Month Mutual Information
    # ================================================================
    print("\n" + "-" * 70, flush=True)
    print("2. Feature -> Month Mutual Information (can feature predict month?)", flush=True)
    print("-" * 70, flush=True)

    mi_scores = mutual_info_classif(X, months, random_state=42, n_neighbors=5)
    mi_ranking = sorted(zip(feature_names, mi_scores), key=lambda x: -x[1])

    print(f"\n{'Feature':<30} {'MI(feat,month)':>14}  Category", flush=True)
    print("-" * 60, flush=True)
    for fname, mi in mi_ranking:
        cat = "WEATHER" if fname.startswith("wx_") else ("SOLAR" if fname.startswith("sol_") else "PHYSICS")
        marker = " *** HIGH MI" if mi > 0.5 else (" ** MODERATE" if mi > 0.2 else "")
        print(f"  {fname:<28} {mi:>14.4f}  {cat}{marker}", flush=True)

    # ================================================================
    # 3. Feature -> Class MI (useful signal)
    # ================================================================
    print("\n" + "-" * 70, flush=True)
    print("3. Feature -> Class MI (useful discriminative signal)", flush=True)
    print("-" * 70, flush=True)

    mi_class = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_class_ranking = sorted(zip(feature_names, mi_class), key=lambda x: -x[1])

    print(f"\n{'Feature':<30} {'MI(feat,class)':>14}  Category", flush=True)
    print("-" * 60, flush=True)
    for fname, mi in mi_class_ranking:
        cat = "WEATHER" if fname.startswith("wx_") else ("SOLAR" if fname.startswith("sol_") else "PHYSICS")
        print(f"  {fname:<28} {mi:>14.4f}  {cat}", flush=True)

    # ================================================================
    # 4. Month-proxy ratio: MI(feat,month) / MI(feat,class)
    # ================================================================
    print("\n" + "-" * 70, flush=True)
    print("4. Month-Proxy Ratio: MI(feat,month) / MI(feat,class)", flush=True)
    print("   High ratio = feature predicts month MORE than class = LEAKY", flush=True)
    print("-" * 70, flush=True)

    mi_month_dict = dict(zip(feature_names, mi_scores))
    mi_class_dict = dict(zip(feature_names, mi_class))
    ratios = []
    for fname in feature_names:
        m_mi = mi_month_dict[fname]
        c_mi = mi_class_dict[fname]
        ratio = m_mi / max(c_mi, 1e-8)
        ratios.append((fname, ratio, m_mi, c_mi))

    ratios.sort(key=lambda x: -x[1])

    print(f"\n{'Feature':<30} {'Ratio':>8} {'MI(month)':>10} {'MI(class)':>10}  Verdict", flush=True)
    print("-" * 75, flush=True)
    for fname, ratio, m_mi, c_mi in ratios:
        if ratio > 2.0:
            verdict = "REMOVE (month proxy)"
        elif ratio > 1.0:
            verdict = "RISKY (mixed signal)"
        elif ratio > 0.5:
            verdict = "CAUTION"
        else:
            verdict = "KEEP (class-informative)"
        print(f"  {fname:<28} {ratio:>8.2f} {m_mi:>10.4f} {c_mi:>10.4f}  {verdict}", flush=True)

    # ================================================================
    # 5. Weather/Solar specific: correlation with month
    # ================================================================
    print("\n" + "-" * 70, flush=True)
    print("5. Weather/Solar Features: Correlation with Month vs Class", flush=True)
    print("-" * 70, flush=True)

    ws_feats = [f for f in feature_names if f.startswith("wx_") or f.startswith("sol_")]
    print(f"\n{'Feature':<30} {'corr(month)':>12} {'corr(class)':>12} {'MI(month)':>10} {'MI(class)':>10}", flush=True)
    print("-" * 80, flush=True)
    for fname in ws_feats:
        fi = feature_names.index(fname)
        vals = X[:, fi]
        corr_month = np.corrcoef(vals, months.astype(float))[0, 1]
        corr_class = np.corrcoef(vals, y.astype(float))[0, 1]
        m_mi = mi_month_dict[fname]
        c_mi = mi_class_dict[fname]
        print(f"  {fname:<28} {corr_month:>12.4f} {corr_class:>12.4f} {m_mi:>10.4f} {c_mi:>10.4f}", flush=True)

    # ================================================================
    # 6. Summary: Recommended feature sets
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Feature Classification", flush=True)
    print("=" * 70, flush=True)

    keep_feats = []
    remove_feats = []
    risky_feats = []

    for fname, ratio, m_mi, c_mi in ratios:
        if ratio > 2.0 and c_mi < 0.05:
            remove_feats.append(fname)
        elif ratio > 1.5:
            risky_feats.append(fname)
        else:
            keep_feats.append(fname)

    print(f"\n  KEEP ({len(keep_feats)} features):", flush=True)
    for f in keep_feats:
        print(f"    {f}", flush=True)

    print(f"\n  RISKY ({len(risky_feats)} features) - test with/without:", flush=True)
    for f in risky_feats:
        r = next(x[1] for x in ratios if x[0] == f)
        print(f"    {f}  (ratio={r:.2f})", flush=True)

    print(f"\n  REMOVE ({len(remove_feats)} features) - month proxies:", flush=True)
    for f in remove_feats:
        r = next(x[1] for x in ratios if x[0] == f)
        print(f"    {f}  (ratio={r:.2f})", flush=True)

    # Save results
    results_path = ROOT / "data" / "cross_month_stability.csv"
    rows = []
    for fname, ratio, m_mi, c_mi in ratios:
        cv = next(x[1] for x in feat_cv_scores if x[0] == fname)
        rows.append({
            "feature": fname,
            "mi_month": m_mi,
            "mi_class": c_mi,
            "month_proxy_ratio": ratio,
            "cross_month_cv": cv,
            "category": "WEATHER" if fname.startswith("wx_") else ("SOLAR" if fname.startswith("sol_") else "PHYSICS"),
        })
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
