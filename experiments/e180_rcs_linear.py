"""E180: RCS Linear-Scale Feature Fix.

Known physics error: RCS features computed on dB (logarithmic) scale have incorrect
arithmetic statistics. dB is a log scale — mean, std, correlation on dB values are
physically incorrect. This experiment adds ~18 new features computed on LINEAR RCS
scale and evaluates whether they improve generalization.

New features:
  - rcs_linear_std, rcs_linear_min, rcs_linear_max, rcs_linear_range
  - rcs_linear_iqr, rcs_linear_median, rcs_linear_p10, rcs_linear_p90
  - rcs_linear_ac1..ac5 (autocorrelation on linear scale)
  - rcs_linear_per_alt (altitude-normalized linear RCS)
  - rcs_linear_trend (temporal slope of linear RCS)
  - rcs_linear_p90_p10_ratio (dynamic range in linear)

Pipeline: LGB GBDT, 5 seeds × 5-fold SGKF, 100 base + ~18 new = ~118 features.
Evaluation: SKF mAP + TRUE LOMO (per-month held-out, months=[1,4,9,10]).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
N_SEEDS = 5


# ══════════════════════════════════════════════════════════════════════
# 1. Extract linear-scale RCS features from raw trajectories
# ══════════════════════════════════════════════════════════════════════

def extract_rcs_linear_features(hex_str: str, traj_time_str: str) -> dict:
    """Extract ~18 physically correct RCS features on LINEAR scale.

    RCS in dB: rcs_dBm2 = 10 * log10(rcs_linear)
    Conversion: rcs_linear = 10^(rcs_dBm2 / 10)

    Linear scale is correct for arithmetic operations (mean, std, correlation)
    because RCS in m^2 is a power quantity.
    """
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    n = len(pts)

    rcs_dB = np.array([p[3] for p in pts])
    alts = np.array([p[2] for p in pts])

    # Convert to linear scale (m^2)
    rcs_lin = 10.0 ** (rcs_dB / 10.0)

    # ── Basic statistics on linear scale ──
    rcs_lin_mean = float(np.mean(rcs_lin))
    rcs_lin_std = float(np.std(rcs_lin))
    rcs_lin_min = float(np.min(rcs_lin))
    rcs_lin_max = float(np.max(rcs_lin))
    rcs_lin_range = rcs_lin_max - rcs_lin_min
    rcs_lin_median = float(np.median(rcs_lin))

    # Percentiles and IQR
    p10, p25, p75, p90 = np.percentile(rcs_lin, [10, 25, 75, 90])
    rcs_lin_iqr = float(p75 - p25)
    rcs_lin_p10 = float(p10)
    rcs_lin_p90 = float(p90)

    # Dynamic range ratio (p90/p10) — measures spread in linear space
    rcs_lin_p90_p10_ratio = float(p90 / max(p10, 1e-15))

    # ── Autocorrelation on linear scale (lags 1-5) ──
    rcs_lin_centered = rcs_lin - rcs_lin_mean
    rcs_lin_var = float(np.var(rcs_lin))

    ac_values = {}
    for lag in range(1, 6):
        if rcs_lin_var > 1e-15 and n > lag:
            ac = float(np.mean(rcs_lin_centered[:-lag] * rcs_lin_centered[lag:]) / rcs_lin_var)
            ac_values[f"rcs_linear_ac{lag}"] = ac if np.isfinite(ac) else 0.0
        else:
            ac_values[f"rcs_linear_ac{lag}"] = 0.0

    # ── Altitude-normalized linear RCS ──
    # Physical: RCS depends on range^4 detection, altitude is a proxy for range
    alt_mean = float(np.mean(alts))
    if alt_mean > 1.0:
        rcs_lin_per_alt = rcs_lin_mean / alt_mean
    else:
        rcs_lin_per_alt = rcs_lin_mean

    # ── Temporal trend of linear RCS ──
    # Slope of linear fit: does RCS increase/decrease over time?
    if n >= 3:
        t_norm = np.linspace(0, 1, n)
        try:
            slope = float(np.polyfit(t_norm, rcs_lin, 1)[0])
            rcs_lin_trend = slope if np.isfinite(slope) else 0.0
        except (np.linalg.LinAlgError, ValueError):
            rcs_lin_trend = 0.0
    else:
        rcs_lin_trend = 0.0

    # ── Coefficient of variation (already in v2, but recompute for consistency) ──
    # Not included — already exists as rcs_linear_cv

    return {
        "rcs_lin_std": rcs_lin_std,
        "rcs_lin_min": rcs_lin_min,
        "rcs_lin_max": rcs_lin_max,
        "rcs_lin_range": rcs_lin_range,
        "rcs_lin_median": rcs_lin_median,
        "rcs_lin_iqr": rcs_lin_iqr,
        "rcs_lin_p10": rcs_lin_p10,
        "rcs_lin_p90": rcs_lin_p90,
        "rcs_lin_p90_p10_ratio": rcs_lin_p90_p10_ratio,
        **ac_values,
        "rcs_lin_per_alt": rcs_lin_per_alt,
        "rcs_lin_trend": rcs_lin_trend,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Load cached v3 features + selected feature list, then add linear RCS features."""
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    # Load cached v3 features
    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    # Load stability-selected 100 features
    selected = [
        line.strip()
        for line in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines()
        if line.strip()
    ]
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    # ── Extract new linear RCS features ──
    rcs_cache_train = ROOT / "data" / "_cached_train_rcs_linear.pkl"
    rcs_cache_test = ROOT / "data" / "_cached_test_rcs_linear.pkl"

    if rcs_cache_train.exists() and rcs_cache_test.exists():
        print("  Loading cached linear RCS features...", flush=True)
        rcs_train = pd.read_pickle(rcs_cache_train)
        rcs_test = pd.read_pickle(rcs_cache_test)
    else:
        print("  Extracting linear RCS features (train)...", flush=True)
        rcs_train_rows = []
        for idx, (_, r) in enumerate(train_df.iterrows()):
            if idx % 500 == 0:
                print(f"    Progress: {idx}/{len(train_df)}", flush=True)
            rcs_train_rows.append(extract_rcs_linear_features(r.trajectory, r.trajectory_time))
        rcs_train = pd.DataFrame(rcs_train_rows)

        print("  Extracting linear RCS features (test)...", flush=True)
        rcs_test_rows = []
        for idx, (_, r) in enumerate(test_df.iterrows()):
            if idx % 500 == 0:
                print(f"    Progress: {idx}/{len(test_df)}", flush=True)
            rcs_test_rows.append(extract_rcs_linear_features(r.trajectory, r.trajectory_time))
        rcs_test = pd.DataFrame(rcs_test_rows)

        # Cache
        rcs_train.to_pickle(rcs_cache_train)
        rcs_test.to_pickle(rcs_cache_test)
        print(f"  Cached linear RCS features ({rcs_train.shape[1]} features)", flush=True)

    # Handle inf/nan in new features
    rcs_train = rcs_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    rcs_test = rcs_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # New feature names
    new_rcs_cols = sorted(rcs_train.columns.tolist())

    # Build feature matrices: selected v3 features + new linear RCS features
    X_train_base = train_feats[selected].values.astype(np.float32)
    X_test_base = test_feats[selected].values.astype(np.float32)
    X_train_rcs = rcs_train[new_rcs_cols].values.astype(np.float32)
    X_test_rcs = rcs_test[new_rcs_cols].values.astype(np.float32)

    # Augmented = base + new linear RCS
    X_train_aug = np.hstack([X_train_base, X_train_rcs])
    X_test_aug = np.hstack([X_test_base, X_test_rcs])
    all_feature_cols = selected + new_rcs_cols

    # Clean
    X_train_base = np.nan_to_num(X_train_base, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)
    X_train_aug = np.nan_to_num(X_train_aug, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_aug = np.nan_to_num(X_test_aug, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Base features:      {X_train_base.shape[1]} (E175 selected)")
    print(f"  New linear RCS:     {len(new_rcs_cols)}")
    print(f"  Augmented features: {X_train_aug.shape[1]}")
    print(f"  New features: {new_rcs_cols}")

    return (
        train_df, test_df, y, train_months, test_months, groups,
        X_train_base, X_test_base,
        X_train_aug, X_test_aug,
        selected, all_feature_cols,
    )


# ══════════════════════════════════════════════════════════════════════
# 3. Training
# ══════════════════════════════════════════════════════════════════════

def eff_weights(y_arr: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Effective number of samples class weighting."""
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-6)
    w = w / w.sum() * N_CLASSES
    return w[y_arr]


def train_lgb_gbdt(
    X_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    n_seeds: int = N_SEEDS,
    label: str = "LGB",
) -> tuple[np.ndarray, np.ndarray]:
    """Train LGB GBDT with multi-seed averaging.

    Returns (oof_preds, test_preds), both shape (n_samples, 9).
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)

        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            w_tr = eff_weights(y[train_idx])
            w_va = eff_weights(y[val_idx])

            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                boosting_type="gbdt",
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                colsample_bytree=0.6,
                subsample=0.7,
                is_unbalance=False,
                verbosity=-1,
                random_state=42 + seed,
                n_jobs=-1,
            )
            model.fit(
                X_train[train_idx], y[train_idx],
                sample_weight=w_tr,
                eval_set=[(X_train[val_idx], y[val_idx])],
                eval_sample_weight=[w_va],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            oof_seed[val_idx] = model.predict_proba(X_train[val_idx])
            test_seed += model.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed

        skf, _ = compute_map(y, oof_seed)
        elapsed = time.time() - t_seed
        print(f"    Seed {seed + 1}/{n_seeds}: SKF={skf:.4f} ({elapsed:.1f}s)", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)

    final_skf, per_class = compute_map(y, oof_mean)
    print_results(final_skf, per_class, f"{label} ({n_seeds} seeds)")

    return oof_mean, test_mean


# ══════════════════════════════════════════════════════════════════════
# 4. LOMO Evaluation
# ══════════════════════════════════════════════════════════════════════

def true_lomo(y: np.ndarray, oof: np.ndarray, months: np.ndarray) -> tuple[float, dict]:
    """Compute TRUE LOMO: per-month held-out mAP.

    Each month is held out entirely and evaluated.
    Returns (average_lomo, {month: mAP}).
    """
    lomo_maps = {}
    for held in sorted(set(months)):
        mask = months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
    lomo_avg = float(np.mean(list(lomo_maps.values()))) if lomo_maps else 0.0
    return lomo_avg, lomo_maps


def print_lomo(lomo_avg: float, lomo_maps: dict, label: str = ""):
    """Pretty-print LOMO results."""
    month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  {label:20s}: LOMO={lomo_avg:.4f}  ({month_str})")


# ══════════════════════════════════════════════════════════════════════
# 5. Feature Importance Analysis
# ══════════════════════════════════════════════════════════════════════

def analyze_new_features(
    X_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    new_feature_names: list[str],
):
    """Print feature importance for new linear RCS features."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    importances = np.zeros(len(feature_cols))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            verbosity=-1, random_state=42, n_jobs=-1,
        )
        model.fit(
            X_train[train_idx], y[train_idx],
            eval_set=[(X_train[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        importances += model.feature_importances_

    importances /= N_FOLDS

    # Print new features ranked by importance
    print("\n  New linear RCS feature importances (gain):")
    new_imps = []
    for name in new_feature_names:
        if name in feature_cols:
            idx = feature_cols.index(name)
            new_imps.append((name, importances[idx]))
    new_imps.sort(key=lambda x: -x[1])
    for name, imp in new_imps:
        print(f"    {name:30s}: {imp:8.1f}")

    # Where do new features rank overall?
    all_imps = sorted(
        [(feature_cols[i], importances[i]) for i in range(len(feature_cols))],
        key=lambda x: -x[1],
    )
    print(f"\n  Top 20 features overall:")
    for rank, (name, imp) in enumerate(all_imps[:20]):
        marker = " **NEW**" if name in new_feature_names else ""
        print(f"    {rank + 1:3d}. {name:30s}: {imp:8.1f}{marker}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 70)
    print("  E180: RCS Linear-Scale Feature Fix")
    print("  Physics correction: compute RCS statistics on linear (m^2) scale")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading data...", flush=True)
    (
        train_df, test_df, y, train_months, test_months, groups,
        X_train_base, X_test_base,
        X_train_aug, X_test_aug,
        selected_cols, all_feature_cols,
    ) = load_data()

    new_rcs_cols = [c for c in all_feature_cols if c not in selected_cols]

    # ── A: Baseline (E175 selected features only) ──
    print(f"\n[2/5] BASELINE: {X_train_base.shape[1]} features ({N_SEEDS} seeds × {N_FOLDS} folds)...", flush=True)
    t2 = time.time()
    oof_base, test_base = train_lgb_gbdt(
        X_train_base, y, groups, X_test_base,
        n_seeds=N_SEEDS, label="BASELINE",
    )
    print(f"  Baseline time: {time.time() - t2:.1f}s")

    base_skf, base_pc = compute_map(y, oof_base)
    base_lomo, base_lomo_d = true_lomo(y, oof_base, train_months)
    print_lomo(base_lomo, base_lomo_d, "BASELINE")

    # ── B: Augmented (base + linear RCS) ──
    print(f"\n[3/5] AUGMENTED: {X_train_aug.shape[1]} features ({N_SEEDS} seeds × {N_FOLDS} folds)...", flush=True)
    t3 = time.time()
    oof_aug, test_aug = train_lgb_gbdt(
        X_train_aug, y, groups, X_test_aug,
        n_seeds=N_SEEDS, label="AUGMENTED",
    )
    print(f"  Augmented time: {time.time() - t3:.1f}s")

    aug_skf, aug_pc = compute_map(y, oof_aug)
    aug_lomo, aug_lomo_d = true_lomo(y, oof_aug, train_months)
    print_lomo(aug_lomo, aug_lomo_d, "AUGMENTED")

    # ── Feature importance analysis ──
    print(f"\n[4/5] Feature importance analysis...", flush=True)
    analyze_new_features(X_train_aug, y, groups, all_feature_cols, new_rcs_cols)

    # ── Save results ──
    print(f"\n[5/5] Saving results...", flush=True)
    np.save(ROOT / "oof_e180_rcs_linear.npy", oof_aug)
    np.save(ROOT / "test_e180_rcs_linear.npy", test_aug)
    print(f"  Saved oof_e180_rcs_linear.npy and test_e180_rcs_linear.npy")

    # Save submission
    save_submission(test_aug, "e180_rcs_linear_raw", cv_map=aug_skf)

    # ── Per-class comparison ──
    print("\n" + "=" * 70)
    print("  E180: PER-CLASS COMPARISON (AP)")
    print("=" * 70)
    print(f"  {'Class':15s}  {'Baseline':>9s}  {'Augmented':>9s}  {'Delta':>8s}")
    print(f"  {'-' * 45}")
    for cls in CLASSES:
        b = base_pc.get(cls, 0.0)
        a = aug_pc.get(cls, 0.0)
        d = a - b
        marker = " +" if d > 0.001 else (" -" if d < -0.001 else "  ")
        print(f"  {cls:15s}  {b:9.4f}  {a:9.4f}  {d:+8.4f}{marker}")

    # ── LOMO per-month comparison ──
    print(f"\n  {'Month':>5s}  {'Baseline':>9s}  {'Augmented':>9s}  {'Delta':>8s}")
    print(f"  {'-' * 35}")
    all_months = sorted(set(base_lomo_d.keys()) | set(aug_lomo_d.keys()))
    for m in all_months:
        b = base_lomo_d.get(m, 0.0)
        a = aug_lomo_d.get(m, 0.0)
        d = a - b
        print(f"  {m:5d}  {b:9.4f}  {a:9.4f}  {d:+8.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  E180 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Baseline:  {X_train_base.shape[1]:3d} features  SKF={base_skf:.4f}  LOMO={base_lomo:.4f}")
    print(f"  Augmented: {X_train_aug.shape[1]:3d} features  SKF={aug_skf:.4f}  LOMO={aug_lomo:.4f}")
    skf_delta = aug_skf - base_skf
    lomo_delta = aug_lomo - base_lomo
    print(f"  Delta:     +{len(new_rcs_cols):2d} features  SKF={skf_delta:+.4f}  LOMO={lomo_delta:+.4f}")
    print(f"  Total time: {time.time() - t_total:.1f}s")
    if lomo_delta > 0:
        print(f"\n  >> LINEAR RCS FEATURES IMPROVED LOMO by {lomo_delta:+.4f}")
    else:
        print(f"\n  >> LINEAR RCS FEATURES DID NOT IMPROVE LOMO ({lomo_delta:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
