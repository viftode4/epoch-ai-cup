"""Adversarial Validation v2: Diagnose train-test distribution shift.

Uses the 36 validated features from E79 (including weather/solar).
Adds per-month and per-class adversarial scoring for actionable insights.

Train a binary classifier (train=0, test=1).
- High AUC = large distribution shift between train and test.
- Feature importances reveal WHICH features cause the shift.
- Per-month scoring shows which training months look most like test.
- Per-class scoring shows which classes are most affected by shift.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import ALL_TEMPORAL, build_features

ROOT = Path(__file__).resolve().parent.parent


def add_weather_solar(feats, split):
    """Add weather + solar columns from pre-computed CSVs."""
    for prefix, fname in [("wx_", f"{split}_weather.csv"), ("sol_", f"{split}_solar.csv")]:
        path = ROOT / "data" / fname
        if path.exists():
            df = pd.read_csv(path)
            for c in df.columns:
                if c.startswith(prefix):
                    feats[c] = df[c].values
    return feats


def run_adversarial_cv(X, y, n_folds=5, seed=42):
    """Run adversarial validation CV, return OOF probs, AUC, and feature importances."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X[va_idx], label=y[va_idx])
        mdl = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
             "num_leaves": 20, "max_depth": 4, "verbose": -1, "seed": seed + fold,
             "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20,
             "device": "gpu"},
            dtrain, num_boost_round=500, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        oof[va_idx] = mdl.predict(X[va_idx])
        importances += mdl.feature_importance(importance_type="gain")
    importances /= n_folds
    auc = roc_auc_score(y, oof)
    return oof, auc, importances


def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION v2: Train vs Test Distribution Shift")
    print("=" * 60, flush=True)

    # --- Load data ---
    train_raw = load_train()
    test_raw = load_test()
    n_train = len(train_raw)
    n_test = len(test_raw)

    # --- Build features (same 36 as E79) ---
    KEEP = [
        f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
        if f.strip()
    ]

    train_feats = build_features(train_raw).drop(columns=ALL_TEMPORAL, errors="ignore")
    test_feats = build_features(test_raw).drop(columns=ALL_TEMPORAL, errors="ignore")
    train_feats = add_weather_solar(train_feats, "train")
    test_feats = add_weather_solar(test_feats, "test")

    common = [c for c in KEEP if c in train_feats.columns and c in test_feats.columns]
    print(f"Using {len(common)} features (E79 validated set)", flush=True)

    X_train = train_feats[common].values.astype(np.float32)
    X_test = test_feats[common].values.astype(np.float32)

    # Clean inf/nan
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # --- Create adversarial dataset ---
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([np.zeros(n_train), np.ones(n_test)])

    # =========================================================================
    # TEST 1: Overall adversarial AUC with 36 features
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Adversarial validation with 36 E79 features")
    print("=" * 60, flush=True)

    oof, auc, importances = run_adversarial_cv(X_all, y_all)
    train_probs = oof[:n_train]
    test_probs = oof[n_train:]

    print(f"\n  AUC = {auc:.4f}  (0.5=no shift, 1.0=complete shift)")
    if auc < 0.55:
        print("  VERDICT: Minimal shift -- train and test look very similar")
    elif auc < 0.70:
        print("  VERDICT: Moderate shift -- some features differ")
    elif auc < 0.85:
        print("  VERDICT: Strong shift -- significant distribution gap")
    else:
        print("  VERDICT: Severe shift -- train and test very different")
    print(flush=True)

    # =========================================================================
    # Feature importance ranking
    # =========================================================================
    feat_imp = sorted(zip(common, importances), key=lambda x: -x[1])
    total_imp = sum(importances)

    print("  TOP SHIFTING FEATURES (most discriminative for train vs test):")
    for i, (name, imp) in enumerate(feat_imp[:20]):
        pct = imp / total_imp * 100
        # Categorize
        cat = ""
        if name.startswith("wx_") or name.startswith("sol_"):
            cat = " [WEATHER/SOLAR]"
        elif name in ("lon_mean", "lat_mean", "lon_std", "lat_std"):
            cat = " [SPATIAL]"
        elif name.startswith("rcs_"):
            cat = " [RCS]"
        elif name.startswith("alt_"):
            cat = " [ALTITUDE]"
        print(f"  {i+1:3d}. {name:30s}: {imp:8.0f}  ({pct:5.1f}%){cat}")
    print(flush=True)

    # =========================================================================
    # TEST 2: KS test per feature
    # =========================================================================
    print("=" * 60)
    print("TEST 2: Feature distribution shift (KS test)")
    print("=" * 60, flush=True)

    ks_results = []
    for i, name in enumerate(common):
        stat, pval = ks_2samp(X_train[:, i], X_test[:, i])
        ks_results.append((name, stat, pval))
    ks_results.sort(key=lambda x: -x[1])

    print(f"\n  {'Feature':30s}  {'KS stat':>8s}  {'p-value':>10s}  {'train_mean':>10s}  {'test_mean':>10s}  {'diff%':>7s}")
    for name, stat, pval in ks_results:
        idx = common.index(name)
        tr_m = np.mean(X_train[:, idx])
        te_m = np.mean(X_test[:, idx])
        denom = max(abs(tr_m), 1e-6)
        diff = (te_m - tr_m) / denom * 100
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "   "
        print(f"  {name:30s}  {stat:8.4f}  {pval:10.2e} {sig}  {tr_m:10.3f}  {te_m:10.3f}  {diff:+6.1f}%")
    print(flush=True)

    # =========================================================================
    # TEST 3: Per-month adversarial scoring
    # =========================================================================
    print("=" * 60)
    print("TEST 3: Per-month analysis")
    print("=" * 60, flush=True)

    train_months = pd.to_datetime(train_raw["timestamp_start_radar_utc"]).dt.month
    test_months = pd.to_datetime(test_raw["timestamp_start_radar_utc"]).dt.month

    print("\n  Train month distribution:")
    for m, cnt in train_months.value_counts().sort_index().items():
        print(f"    Month {m:2d}: {cnt:4d} ({cnt/n_train*100:5.1f}%)")

    print("  Test month distribution:")
    for m, cnt in test_months.value_counts().sort_index().items():
        print(f"    Month {m:2d}: {cnt:4d} ({cnt/n_test*100:5.1f}%)")

    print("\n  Adversarial score by TRAINING month (higher = more test-like):")
    for m in sorted(train_months.unique()):
        mask = train_months.values == m
        scores = train_probs[mask]
        print(f"    Month {m:2d} (n={mask.sum():4d}): mean={scores.mean():.4f}  "
              f"median={np.median(scores):.4f}  "
              f"pct>0.5={np.mean(scores > 0.5)*100:.1f}%")

    print("\n  Adversarial score by TEST month (higher = more different from train):")
    for m in sorted(test_months.unique()):
        mask = test_months.values == m
        scores = test_probs[mask]
        print(f"    Month {m:2d} (n={mask.sum():4d}): mean={scores.mean():.4f}  "
              f"median={np.median(scores):.4f}  "
              f"pct>0.7={np.mean(scores > 0.7)*100:.1f}%")
    print(flush=True)

    # =========================================================================
    # TEST 4: Per-class adversarial scoring
    # =========================================================================
    print("=" * 60)
    print("TEST 4: Per-class adversarial scoring (train only)")
    print("=" * 60, flush=True)

    print(f"\n  {'Class':20s}  {'N':>5s}  {'mean_score':>10s}  {'pct>0.5':>8s}  {'interpretation':>30s}")
    for cls in CLASSES:
        mask = train_raw["bird_group"].values == cls
        if mask.sum() == 0:
            continue
        scores = train_probs[mask]
        n = mask.sum()
        pct_high = np.mean(scores > 0.5) * 100
        interp = "looks like test" if scores.mean() > 0.55 else "looks like train" if scores.mean() < 0.45 else "mixed"
        print(f"  {cls:20s}  {n:5d}  {scores.mean():10.4f}  {pct_high:7.1f}%  {interp:>30s}")
    print(flush=True)

    # =========================================================================
    # TEST 5: Feature groups -- which category shifts most?
    # =========================================================================
    print("=" * 60)
    print("TEST 5: Feature group shift analysis")
    print("=" * 60, flush=True)

    groups = {
        "weather": [f for f in common if f.startswith("wx_")],
        "solar": [f for f in common if f.startswith("sol_")],
        "rcs": [f for f in common if f.startswith("rcs_")],
        "altitude": [f for f in common if f.startswith("alt_")],
        "speed": [f for f in common if any(f.startswith(p) for p in
                  ["speed_", "airspeed", "avg_ground", "accel_", "slow_flight"])],
        "spatial": [f for f in common if f.startswith(("lon_", "lat_"))],
        "interaction": [f for f in common if f in ("size_x_alt", "speed_x_alt",
                        "curvature_mean", "bearing_change_mean", "rcs_for_size", "rcs_per_alt")],
    }

    for gname, gfeats in groups.items():
        if not gfeats:
            continue
        g_idx = [common.index(f) for f in gfeats]
        X_g = X_all[:, g_idx]
        _, g_auc, _ = run_adversarial_cv(X_g, y_all)
        print(f"  {gname:15s} ({len(gfeats):2d} feats): AUC = {g_auc:.4f}", flush=True)
    print(flush=True)

    # =========================================================================
    # TEST 6: Ablation -- AUC without top shifting features
    # =========================================================================
    print("=" * 60)
    print("TEST 6: Ablation -- AUC after removing top shifting features")
    print("=" * 60, flush=True)

    for n_drop in [1, 2, 3, 5, 7, 10]:
        if n_drop > len(feat_imp):
            break
        drop_feats = [f for f, _ in feat_imp[:n_drop]]
        keep_idx = [i for i, f in enumerate(common) if f not in drop_feats]
        X_r = X_all[:, keep_idx]
        _, r_auc, _ = run_adversarial_cv(X_r, y_all)
        print(f"  Drop top {n_drop:2d} ({', '.join(drop_feats[:3])}{'...' if n_drop > 3 else ''}): "
              f"AUC = {r_auc:.4f} (was {auc:.4f}, delta {r_auc - auc:+.4f})")
    print(flush=True)

    # =========================================================================
    # TEST 7: Weather-only vs trajectory-only
    # =========================================================================
    print("=" * 60)
    print("TEST 7: Trajectory-only adversarial validation (no weather/solar)")
    print("=" * 60, flush=True)

    traj_feats = [f for f in common if not f.startswith("wx_") and not f.startswith("sol_")]
    traj_idx = [common.index(f) for f in traj_feats]
    X_traj = X_all[:, traj_idx]
    _, traj_auc, traj_imp = run_adversarial_cv(X_traj, y_all)
    print(f"  Trajectory-only ({len(traj_feats)} feats): AUC = {traj_auc:.4f}")

    wx_sol_feats = [f for f in common if f.startswith("wx_") or f.startswith("sol_")]
    if wx_sol_feats:
        wx_idx = [common.index(f) for f in wx_sol_feats]
        X_wx = X_all[:, wx_idx]
        _, wx_auc, _ = run_adversarial_cv(X_wx, y_all)
        print(f"  Weather+solar only ({len(wx_sol_feats)} feats): AUC = {wx_auc:.4f}")
    print(flush=True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print(f"  Overall adversarial AUC: {auc:.4f}")
    print(f"  Trajectory-only AUC:     {traj_auc:.4f}")
    if wx_sol_feats:
        print(f"  Weather+solar only AUC:  {wx_auc:.4f}")
    print()

    # Identify the biggest culprits
    weather_imp = sum(imp for f, imp in feat_imp if f.startswith("wx_") or f.startswith("sol_"))
    traj_imp_sum = sum(imp for f, imp in feat_imp if not f.startswith("wx_") and not f.startswith("sol_"))
    print(f"  Weather/solar features account for {weather_imp/total_imp*100:.1f}% of adversarial importance")
    print(f"  Trajectory features account for    {traj_imp_sum/total_imp*100:.1f}% of adversarial importance")
    print()

    if auc > 0.7:
        print("  ACTION: Significant shift detected. Consider:")
        print("  1. Removing top-shifting features if they don't help classification")
        print("  2. Reweighting training samples using adversarial scores")
        print("  3. Pseudo-labeling with high-confidence test predictions")
    elif auc > 0.55:
        print("  ACTION: Moderate shift. The model should be somewhat robust.")
        print("  Focus on per-class calibration rather than feature removal.")
    else:
        print("  GOOD: Minimal shift. Feature distributions are stable.")

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
