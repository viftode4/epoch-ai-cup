"""E187: E175 EXACT architecture + relabeled data ONLY.

No other changes. Same 100 features, same hyperparameters, same seeds.
Just cleaner labels (133 agreed noisy -> consensus).

This is the minimal, lowest-risk improvement to our best LB=0.59 model.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

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


def load_data():
    train_df = load_train()
    test_df = load_test()
    y_orig = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    # SAME 100 stability-selected features as E175
    selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ONLY CHANGE: relabel 133 agreed noisy labels
    cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
    y_relabeled = y_orig.copy()
    for idx in cache['agreed_noisy'].tolist():
        y_relabeled[idx] = cache['consensus_labels'][idx]

    n_changed = np.sum(y_relabeled != y_orig)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    print(f"  Selected features: {len(selected)}", flush=True)
    print(f"  Relabeled: {n_changed} samples", flush=True)

    return train_df, test_df, y_orig, y_relabeled, train_months, test_months, groups, X_train, X_test


def run_ovr_lambdarank(X_train, y_train, y_eval, train_months, X_test, groups, n_seeds=N_SEEDS_OVR):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_eval, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            m_tr, m_va = train_months[train_idx], train_months[val_idx]

            tr_order = np.argsort(m_tr)
            va_order = np.argsort(m_va)
            X_tr_s = X_tr[tr_order]
            X_va_s = X_va[va_order]
            m_tr_s = m_tr[tr_order]
            m_va_s = m_va[va_order]
            tr_groups = [int((m_tr_s == m).sum()) for m in sorted(set(m_tr_s))]
            va_groups = [int((m_va_s == m).sum()) for m in sorted(set(m_va_s))]

            for cls_idx in range(N_CLASSES):
                y_bin_tr = (y_tr[tr_order] == cls_idx).astype(int)
                y_bin_va = (y_eval[val_idx][va_order] == cls_idx).astype(int)

                if y_bin_tr.sum() < 2 or y_bin_va.sum() < 1:
                    continue

                ranker = lgb.LGBMRanker(
                    objective="lambdarank", metric="map", boosting_type="dart",
                    n_estimators=1000, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                    drop_rate=0.15, lambdarank_truncation_level=30,
                    verbosity=-1, random_state=42 + seed + cls_idx, n_jobs=1,
                )
                ranker.fit(
                    X_tr_s, y_bin_tr, group=tr_groups,
                    eval_set=[(X_va_s, y_bin_va)], eval_group=[va_groups],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )

                va_preds_s = ranker.predict(X_va_s)
                inv_va = np.empty_like(va_order)
                inv_va[va_order] = np.arange(len(va_order))
                oof_seed[val_idx, cls_idx] = va_preds_s[inv_va]
                test_seed[:, cls_idx] += ranker.predict(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        oof_map, _ = compute_map(y_eval, oof_seed)
        print(f"    Seed {seed+1:2d}/{n_seeds}: OOF mAP={oof_map:.4f} ({time.time()-t_seed:.0f}s)", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    final_map, per_class = compute_map(y_eval, oof_mean)
    print_results(final_map, per_class, f"OvR LambdaRank ({n_seeds} seeds)")
    return oof_mean, test_mean


def run_catboost_multiclass(X_train, y_train, y_eval, train_months, X_test, groups, n_seeds=N_SEEDS_CB):
    import catboost as cb
    from sklearn.model_selection import StratifiedGroupKFold

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        t_seed = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_eval, groups)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            m_tr = train_months[train_idx]

            sample_weights = np.ones(len(y_tr), dtype=float)

            # Round 1: find worst month
            model_r1 = cb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=500,
                rsm=0.6, bootstrap_type="MVS", subsample=0.7,
                model_shrink_rate=0.1, early_stopping_rounds=50,
                random_seed=42 + seed + fold, verbose=0, task_type="CPU",
            )
            model_r1.fit(X_tr, y_tr, eval_set=cb.Pool(X_va, y_eval[val_idx]))

            preds_tr = model_r1.predict_proba(X_tr)
            month_maps = {}
            for m in sorted(set(m_tr)):
                mask = m_tr == m
                if mask.sum() >= 5:
                    mm, _ = compute_map(y_tr[mask], preds_tr[mask])
                    month_maps[m] = mm

            if month_maps:
                worst_month = min(month_maps, key=month_maps.get)
                sample_weights[m_tr == worst_month] *= 2.0

            # Round 2: DRO
            model = cb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
                rsm=0.6, bootstrap_type="MVS", subsample=0.7,
                model_shrink_rate=0.1, early_stopping_rounds=100,
                random_seed=42 + seed + fold, verbose=0, task_type="CPU",
            )
            model.fit(X_tr, y_tr, sample_weight=sample_weights,
                      eval_set=cb.Pool(X_va, y_eval[val_idx]))

            oof_seed[val_idx] = model.predict_proba(X_va)
            test_seed += model.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        oof_map, _ = compute_map(y_eval, oof_seed)
        print(f"    Seed {seed+1:2d}/{n_seeds}: OOF mAP={oof_map:.4f} ({time.time()-t_seed:.0f}s)", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    final_map, per_class = compute_map(y_eval, oof_mean)
    print_results(final_map, per_class, f"CatBoost Multiclass ({n_seeds} seeds)")
    return oof_mean, test_mean


def rank_power_ensemble(preds_list, weights, power=1.5):
    from scipy.stats import rankdata
    n = preds_list[0].shape[0]; nc = preds_list[0].shape[1]
    final = np.zeros((n, nc))
    for c in range(nc):
        for preds, w in zip(preds_list, weights):
            final[:, c] += w * (rankdata(preds[:, c]) / n) ** power
    return final


def tune_ensemble(oof_ranker, oof_cb, y_eval):
    best_score = -1.0; best_config = (0.5, 0.5, 1.5)
    for power in [1.0, 1.25, 1.5, 2.0, 3.0]:
        for w_ranker in np.arange(0.0, 1.05, 0.1):
            w_cb = 1.0 - w_ranker
            blended = rank_power_ensemble([oof_ranker, oof_cb], [w_ranker, w_cb], power)
            score, _ = compute_map(y_eval, blended)
            if score > best_score:
                best_score = score
                best_config = (round(w_ranker, 2), round(w_cb, 2), power)
    return best_config, best_score


def main():
    t_total = time.time()
    print("=" * 70, flush=True)
    print("  E187: E175 ARCHITECTURE + RELABELED DATA", flush=True)
    print("  Same features, same hyperparameters, same seeds.", flush=True)
    print("  Only change: 133 noisy labels fixed.", flush=True)
    print("=" * 70, flush=True)

    # Load
    print("\n[1/6] Loading data...", flush=True)
    train_df, test_df, y_orig, y_relabeled, train_months, test_months, groups, X_train, X_test = load_data()

    # OvR LambdaRank (train on relabeled, evaluate on original)
    print(f"\n[2/6] OvR LambdaRank ({N_SEEDS_OVR} seeds x {N_FOLDS} folds x 9 classes)...", flush=True)
    t2 = time.time()
    oof_ranker, test_ranker = run_ovr_lambdarank(
        X_train, y_relabeled, y_orig, train_months, X_test, groups, N_SEEDS_OVR)
    print(f"  Phase 2 time: {time.time()-t2:.0f}s", flush=True)

    np.save(ROOT / "oof_e187_ranker.npy", oof_ranker)
    np.save(ROOT / "test_e187_ranker.npy", test_ranker)

    # CatBoost DRO
    print(f"\n[3/6] CatBoost DRO ({N_SEEDS_CB} seeds x {N_FOLDS} folds)...", flush=True)
    t3 = time.time()
    oof_cb, test_cb = run_catboost_multiclass(
        X_train, y_relabeled, y_orig, train_months, X_test, groups, N_SEEDS_CB)
    print(f"  Phase 3 time: {time.time()-t3:.0f}s", flush=True)

    np.save(ROOT / "oof_e187_cb.npy", oof_cb)
    np.save(ROOT / "test_e187_cb.npy", test_cb)

    # Ensemble
    print("\n[4/6] Tuning ensemble...", flush=True)
    best_config, best_oof_map = tune_ensemble(oof_ranker, oof_cb, y_orig)
    w_ranker, w_cb, power = best_config
    print(f"  Best: w_ranker={w_ranker}, w_cb={w_cb}, power={power}, OOF mAP={best_oof_map:.4f}", flush=True)

    oof_blend = rank_power_ensemble([oof_ranker, oof_cb], [w_ranker, w_cb], power)
    test_blend = rank_power_ensemble([test_ranker, test_cb], [w_ranker, w_cb], power)

    blend_map, blend_pc = compute_map(y_orig, oof_blend)
    print_results(blend_map, blend_pc, "Blended Ensemble (relabeled)")

    # LOMO
    print("\n[5/6] LOMO evaluation...", flush=True)
    for name, oof in [("Ranker", oof_ranker), ("CatBoost", oof_cb), ("Blend", oof_blend)]:
        lomo_maps = {}
        for held in sorted(set(train_months)):
            mask = train_months == held
            if mask.sum() >= 10:
                lm, _ = compute_map(y_orig[mask], oof[mask])
                lomo_maps[held] = lm
        lomo_avg = np.mean(list(lomo_maps.values()))
        month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
        print(f"  {name:10s}: LOMO={lomo_avg:.4f}  ({month_str})", flush=True)

    # Save submissions
    print("\n[6/6] Saving...", flush=True)
    ranker_map, _ = compute_map(y_orig, oof_ranker)
    cb_map, _ = compute_map(y_orig, oof_cb)

    save_submission(test_ranker, "e187_ranker_raw", cv_map=ranker_map)
    save_submission(test_cb, "e187_cb_raw", cv_map=cb_map)
    save_submission(test_blend, "e187_blend_raw", cv_map=blend_map)

    # Compare to E175
    oof_e175 = np.load(ROOT / "oof_e175_best.npy")
    e175_map, e175_pc = compute_map(y_orig, oof_e175)

    print("\n" + "=" * 70, flush=True)
    print("  E187 vs E175 COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(f"  E175 (LB=0.59): SKF={e175_map:.4f}", flush=True)
    print(f"  E187 (relabel): SKF={blend_map:.4f} ({blend_map-e175_map:+.4f})", flush=True)
    print(f"\n  {'Class':15s} {'E175':>8s} {'E187':>8s} {'Delta':>8s}", flush=True)
    for cls in CLASSES:
        d = blend_pc[cls] - e175_pc[cls]
        mk = " ***" if abs(d) > 0.02 else ""
        print(f"  {cls:15s} {e175_pc[cls]:8.4f} {blend_pc[cls]:8.4f} {d:+8.4f}{mk}", flush=True)
    print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
