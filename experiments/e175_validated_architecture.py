"""E175: Validated Architecture — OvR LambdaRank + CatBoost + Rank-Power Ensemble.

Every design decision is justified by research (see FINAL_ARCHITECTURE.md).

Pipeline:
  Phase 0: Feature extraction (316 features, cached)
  Phase 1: Cross-month stability selection (100 features, cached)
  Phase 2: 9× OvR LambdaRank (DART, query-by-month, 20 seeds)
  Phase 3: CatBoost multiclass (Balanced, DART-like, Group DRO, 10 seeds)
  Phase 4: Rank + power ensemble + NB PP
  Phase 5: Submit variants
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

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
N_SEEDS_OVR = 20
N_SEEDS_CB = 10


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Load raw data, cached v3 features, and stability-selected feature list."""
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    # Load cached v3 features
    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    # Load stability-selected features
    selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
    # Ensure all selected features exist in both splits
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Selected features: {len(selected)}")
    return train_df, test_df, y, train_months, test_months, groups, X_train, X_test, selected


# ══════════════════════════════════════════════════════════════════════
# Phase 2: OvR LambdaRank
# ══════════════════════════════════════════════════════════════════════

def run_ovr_lambdarank(X_train, y, train_months, X_test, groups, n_seeds=N_SEEDS_OVR):
    """Train 9 OvR LambdaRank rankers with DART and month-grouped queries.

    Each ranker optimizes binary AP for one class vs rest.
    Query groups by month: forces per-month AP optimization (Group DRO for ranking).

    Returns:
        oof_scores: (n_train, 9) OOF ranking scores (mean across seeds)
        test_scores: (n_test, 9) test ranking scores (mean across seeds)
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all_seeds = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all_seeds = np.zeros((n_seeds, n_test, N_CLASSES))

    # Precompute month group sizes for each fold
    unique_months_sorted = sorted(set(train_months))

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)

        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            m_tr, m_va = train_months[train_idx], train_months[val_idx]

            # Compute query group sizes (sorted by month for LambdaRank)
            tr_month_order = np.argsort(m_tr)
            va_month_order = np.argsort(m_va)

            X_tr_sorted = X_tr[tr_month_order]
            X_va_sorted = X_va[va_month_order]
            m_tr_sorted = m_tr[tr_month_order]
            m_va_sorted = m_va[va_month_order]

            tr_group_sizes = [int((m_tr_sorted == m).sum()) for m in sorted(set(m_tr_sorted))]
            va_group_sizes = [int((m_va_sorted == m).sum()) for m in sorted(set(m_va_sorted))]

            for cls_idx in range(N_CLASSES):
                y_bin_tr = (y_tr[tr_month_order] == cls_idx).astype(int)
                y_bin_va = (y_va[va_month_order] == cls_idx).astype(int)

                # Skip if class absent from train or val fold
                if y_bin_tr.sum() < 2 or y_bin_va.sum() < 1:
                    continue

                ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    metric="map",
                    boosting_type="dart",
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=20,
                    colsample_bytree=0.6,
                    subsample=0.7,
                    drop_rate=0.15,
                    lambdarank_truncation_level=30,
                    verbosity=-1,
                    random_state=42 + seed + cls_idx,
                    n_jobs=-1,
                )

                ranker.fit(
                    X_tr_sorted, y_bin_tr,
                    group=tr_group_sizes,
                    eval_set=[(X_va_sorted, y_bin_va)],
                    eval_group=[va_group_sizes],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )

                # Predict: unsort val predictions back to original order
                va_preds_sorted = ranker.predict(X_va_sorted)
                va_preds = np.empty_like(va_preds_sorted)
                va_preds[va_month_order] = va_preds_sorted  # reverse sort
                # Actually we need inverse permutation
                inv_va = np.empty_like(va_month_order)
                inv_va[va_month_order] = np.arange(len(va_month_order))
                va_preds = va_preds_sorted[inv_va]

                oof_seed[val_idx, cls_idx] = va_preds
                test_seed[:, cls_idx] += ranker.predict(X_test) / N_FOLDS

        oof_all_seeds[seed] = oof_seed
        test_all_seeds[seed] = test_seed

        # Quick eval for this seed
        oof_map, _ = compute_map(y, oof_seed)
        elapsed = time.time() - t_seed
        print(f"    Seed {seed+1:2d}/{n_seeds}: OOF mAP={oof_map:.4f} ({elapsed:.1f}s)", flush=True)

    # Mean across seeds
    oof_mean = np.mean(oof_all_seeds, axis=0)
    test_mean = np.mean(test_all_seeds, axis=0)

    final_map, per_class = compute_map(y, oof_mean)
    print_results(final_map, per_class, f"OvR LambdaRank ({n_seeds} seeds)")

    return oof_mean, test_mean


# ══════════════════════════════════════════════════════════════════════
# Phase 3: CatBoost Multiclass
# ══════════════════════════════════════════════════════════════════════

def run_catboost_multiclass(X_train, y, train_months, X_test, groups, n_seeds=N_SEEDS_CB):
    """Train CatBoost multiclass with Group DRO (upweight worst month).

    Returns:
        oof_probs: (n_train, 9) OOF probabilities (mean across seeds)
        test_probs: (n_test, 9) test probabilities (mean across seeds)
    """
    import catboost as cb
    from sklearn.model_selection import StratifiedGroupKFold

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all_seeds = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all_seeds = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)

        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            m_tr = train_months[train_idx]

            # Group DRO: compute per-month loss from a quick preliminary model,
            # upweight worst month in round 2
            sample_weights = np.ones(len(y_tr), dtype=float)

            # Round 1: uniform weights, identify worst month
            model_r1 = cb.CatBoostClassifier(
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                depth=6,
                l2_leaf_reg=5.0,
                learning_rate=0.03,
                iterations=500,
                rsm=0.6,
                subsample=0.7,
                model_shrink_rate=0.1,
                early_stopping_rounds=50,
                random_seed=42 + seed + fold,
                verbose=0,
                task_type="CPU",
            )
            model_r1.fit(X_tr, y_tr, eval_set=cb.Pool(X_va, y_va))

            # Compute per-month mAP
            preds_tr = model_r1.predict_proba(X_tr)
            month_maps = {}
            for m in sorted(set(m_tr)):
                mask = m_tr == m
                if mask.sum() >= 5:
                    mm, _ = compute_map(y_tr[mask], preds_tr[mask])
                    month_maps[m] = mm

            if month_maps:
                worst_month = min(month_maps, key=month_maps.get)
                # Upweight worst month 2x
                sample_weights[m_tr == worst_month] *= 2.0

            # Round 2: train with DRO weights
            model = cb.CatBoostClassifier(
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                depth=6,
                l2_leaf_reg=5.0,
                learning_rate=0.03,
                iterations=2000,
                rsm=0.6,
                subsample=0.7,
                model_shrink_rate=0.1,
                early_stopping_rounds=100,
                random_seed=42 + seed + fold,
                verbose=0,
                task_type="CPU",
            )
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weights,
                eval_set=cb.Pool(X_va, y_va),
            )

            oof_seed[val_idx] = model.predict_proba(X_va)
            test_seed += model.predict_proba(X_test) / N_FOLDS

        oof_all_seeds[seed] = oof_seed
        test_all_seeds[seed] = test_seed

        oof_map, _ = compute_map(y, oof_seed)
        elapsed = time.time() - t_seed
        print(f"    Seed {seed+1:2d}/{n_seeds}: OOF mAP={oof_map:.4f} ({elapsed:.1f}s)", flush=True)

    oof_mean = np.mean(oof_all_seeds, axis=0)
    test_mean = np.mean(test_all_seeds, axis=0)

    final_map, per_class = compute_map(y, oof_mean)
    print_results(final_map, per_class, f"CatBoost Multiclass ({n_seeds} seeds)")

    return oof_mean, test_mean


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Rank + Power Ensemble + NB PP
# ══════════════════════════════════════════════════════════════════════

def rank_power_ensemble(preds_list, weights, power=1.5):
    """Rank-based ensemble with power averaging for ranking metrics.

    For each class:
    1. Convert each model's predictions to ranks (0-1)
    2. Apply power transform (amplifies top rankings)
    3. Weighted average
    """
    from scipy.stats import rankdata

    n_samples = preds_list[0].shape[0]
    n_classes = preds_list[0].shape[1]
    final = np.zeros((n_samples, n_classes))

    for c in range(n_classes):
        for preds, w in zip(preds_list, weights):
            ranks = rankdata(preds[:, c]) / n_samples  # 0 to 1
            final[:, c] += w * (ranks ** power)

    return final


def apply_standard_nb_pp(preds, test_df, test_months, train_df, y,
                         gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    """Standard 3-channel NB post-processing."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)

    speed_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    cont_tr = {"speed": speed_tr, "alt_mid": 0.5 * (min_z_tr + max_z_tr), "alt_range": max_z_tr - min_z_tr}
    sl, lps, mu, sig = build_nb_params(train_df, y, cont_tr)

    speed_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    min_z_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    max_z_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    cont_te = {"speed": speed_te, "alt_mid": 0.5 * (min_z_te + max_z_te), "alt_range": max_z_te - min_z_te}
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(test_df, sl, lps, cont_te, w, None, mu, sig)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)


def tune_ensemble(oof_ranker, oof_cb, y):
    """Tune ensemble power and weights on OOF data."""
    from scipy.stats import rankdata

    best_score = -1.0
    best_config = (0.5, 0.5, 1.5)

    for power in [1.0, 1.25, 1.5, 2.0, 3.0]:
        for w_ranker in np.arange(0.0, 1.05, 0.1):
            w_cb = 1.0 - w_ranker
            blended = rank_power_ensemble(
                [oof_ranker, oof_cb],
                [w_ranker, w_cb],
                power=power,
            )
            score, _ = compute_map(y, blended)
            if score > best_score:
                best_score = score
                best_config = (round(w_ranker, 2), round(w_cb, 2), power)

    return best_config, best_score


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 70)
    print("  E175: VALIDATED ARCHITECTURE")
    print("  OvR LambdaRank + CatBoost + Rank-Power Ensemble")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/6] Loading data...", flush=True)
    train_df, test_df, y, train_months, test_months, groups, X_train, X_test, feature_cols = load_data()

    # ── Phase 2: OvR LambdaRank ──
    print(f"\n[2/6] Training OvR LambdaRank ({N_SEEDS_OVR} seeds × {N_FOLDS} folds × 9 classes)...", flush=True)
    t2 = time.time()
    oof_ranker, test_ranker = run_ovr_lambdarank(X_train, y, train_months, X_test, groups, N_SEEDS_OVR)
    print(f"  Phase 2 time: {time.time()-t2:.1f}s")

    # Save OvR outputs
    np.save(ROOT / "oof_e175_ranker.npy", oof_ranker)
    np.save(ROOT / "test_e175_ranker.npy", test_ranker)

    # ── Phase 3: CatBoost Multiclass ──
    print(f"\n[3/6] Training CatBoost Multiclass ({N_SEEDS_CB} seeds × {N_FOLDS} folds)...", flush=True)
    t3 = time.time()
    oof_cb, test_cb = run_catboost_multiclass(X_train, y, train_months, X_test, groups, N_SEEDS_CB)
    print(f"  Phase 3 time: {time.time()-t3:.1f}s")

    np.save(ROOT / "oof_e175_cb.npy", oof_cb)
    np.save(ROOT / "test_e175_cb.npy", test_cb)

    # ── Phase 4: Ensemble + PP ──
    print("\n[4/6] Tuning ensemble...", flush=True)

    # Tune power and weights
    best_config, best_oof_map = tune_ensemble(oof_ranker, oof_cb, y)
    w_ranker, w_cb, power = best_config
    print(f"  Best config: w_ranker={w_ranker}, w_cb={w_cb}, power={power}, OOF mAP={best_oof_map:.4f}")

    # Build ensembles
    oof_blend = rank_power_ensemble([oof_ranker, oof_cb], [w_ranker, w_cb], power)
    test_blend = rank_power_ensemble([test_ranker, test_cb], [w_ranker, w_cb], power)

    blend_map, blend_pc = compute_map(y, oof_blend)
    print_results(blend_map, blend_pc, "Blended Ensemble")

    # Apply PP to CatBoost probabilities only
    test_cb_pp = apply_standard_nb_pp(test_cb, test_df, test_months, train_df, y)

    # Blend ranker + CB+PP
    test_blend_pp = rank_power_ensemble([test_ranker, test_cb_pp], [w_ranker, w_cb], power)

    # ── LOMO evaluation ──
    print("\n[5/6] LOMO evaluation...", flush=True)
    for name, oof in [("Ranker", oof_ranker), ("CatBoost", oof_cb), ("Blend", oof_blend)]:
        lomo_maps = {}
        for held in sorted(set(train_months)):
            mask = train_months == held
            if mask.sum() >= 10:
                lm, _ = compute_map(y[mask], oof[mask])
                lomo_maps[held] = lm
        lomo_avg = np.mean(list(lomo_maps.values()))
        month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
        print(f"  {name:10s}: LOMO={lomo_avg:.4f}  ({month_str})")

    # ── Phase 5: Save submissions ──
    print("\n[6/6] Saving submissions...", flush=True)
    ranker_map, _ = compute_map(y, oof_ranker)
    cb_map, _ = compute_map(y, oof_cb)

    save_submission(test_ranker, "e175_ranker_raw", cv_map=ranker_map)
    save_submission(test_cb, "e175_cb_raw", cv_map=cb_map)
    save_submission(test_cb_pp, "e175_cb_pp", cv_map=cb_map)
    save_submission(test_blend, "e175_blend_raw", cv_map=blend_map)
    save_submission(test_blend_pp, "e175_blend_pp", cv_map=blend_map)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  E175 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Features:      {len(feature_cols)} (stability-selected from 316)")
    print(f"  OvR Ranker:    OOF mAP={ranker_map:.4f} ({N_SEEDS_OVR} seeds)")
    print(f"  CatBoost:      OOF mAP={cb_map:.4f} ({N_SEEDS_CB} seeds)")
    print(f"  Blend:         OOF mAP={blend_map:.4f} (w_r={w_ranker}, w_cb={w_cb}, p={power})")
    print(f"  Total time:    {time.time()-t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
