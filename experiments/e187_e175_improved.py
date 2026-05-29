"""E187: E175 Architecture with Improvements.

SAME architecture as E175 (proven LB=0.59):
  - OvR LambdaRank (DART, query-by-month, multi-seed)
  - CatBoost multiclass (Balanced, Group DRO, multi-seed)
  - Rank-power ensemble

BUT with improvements:
  A. ALL 327 features (vs 100 stability-selected)
  B. Relabeled data (133 agreed noisy labels fixed)
  C. Lower min_child_samples for rare classes (5 vs 20)
  D. Compare A/B/C individually to isolate what helps

Also trains ORIGINAL E175 config for fair comparison on same machine.
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
N_SEEDS_OVR = 10  # fewer seeds for faster iteration (E175 used 20)
N_SEEDS_CB = 5    # fewer seeds (E175 used 10)

# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def load_data(use_all_features=False, use_relabeled=False):
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    if use_all_features:
        selected = list(train_feats.columns)
    else:
        selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
        selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = y.copy()
    keep_mask = np.ones(len(y), dtype=bool)

    if use_relabeled:
        cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
        for idx in cache['agreed_noisy'].tolist():
            y_train[idx] = cache['consensus_labels'][idx]
        extreme = set(np.where(cache['quality'] < 0.02)[0])
        keep_mask = np.array([i not in extreme for i in range(len(y))])

    print(f"  Features: {len(selected)}, Relabeled: {use_relabeled}", flush=True)
    return train_df, test_df, y, y_train, train_months, test_months, groups, X_train, X_test, keep_mask

# ══════════════════════════════════════════════════════════════
# OvR LambdaRank (from E175, with optional improvements)
# ══════════════════════════════════════════════════════════════

def run_ovr_lambdarank(X_train, y_train, y_eval, train_months, X_test, groups,
                       n_seeds=N_SEEDS_OVR, min_child=20, keep_mask=None):
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
            # Apply keep_mask to training only
            if keep_mask is not None:
                train_idx = np.array([i for i in train_idx if keep_mask[i]])

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
                    min_child_samples=min_child, colsample_bytree=0.6, subsample=0.7,
                    drop_rate=0.15, lambdarank_truncation_level=30,
                    verbosity=-1, random_state=42 + seed + cls_idx, n_jobs=1,
                )
                ranker.fit(X_tr_s, y_bin_tr, group=tr_groups,
                           eval_set=[(X_va_s, y_bin_va)], eval_group=[va_groups],
                           callbacks=[lgb.early_stopping(100, verbose=False)])

                va_preds_s = ranker.predict(X_va_s)
                inv_va = np.empty_like(va_order)
                inv_va[va_order] = np.arange(len(va_order))
                oof_seed[val_idx, cls_idx] = va_preds_s[inv_va]
                test_seed[:, cls_idx] += ranker.predict(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        oof_map, _ = compute_map(y_eval, oof_seed)
        print(f"    Seed {seed+1}/{n_seeds}: mAP={oof_map:.4f} ({time.time()-t_seed:.0f}s)", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    final_map, per_class = compute_map(y_eval, oof_mean)
    return oof_mean, test_mean, final_map, per_class

# ══════════════════════════════════════════════════════════════
# CatBoost DRO (from E175)
# ══════════════════════════════════════════════════════════════

def run_catboost_dro(X_train, y_train, y_eval, train_months, X_test, groups,
                     n_seeds=N_SEEDS_CB, keep_mask=None):
    import catboost as catb
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
            if keep_mask is not None:
                train_idx = np.array([i for i in train_idx if keep_mask[i]])

            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]
            m_tr = train_months[train_idx]

            # Round 1: find worst month
            model_r1 = catb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=500,
                rsm=0.6, bootstrap_type="MVS", subsample=0.7,
                model_shrink_rate=0.1, early_stopping_rounds=50,
                random_seed=42 + seed + fold, verbose=0, task_type="CPU",
            )
            model_r1.fit(X_tr, y_tr, eval_set=catb.Pool(X_va, y_eval[val_idx]))

            # DRO: upweight worst month (eval on VAL, not train)
            preds_va = model_r1.predict_proba(X_va)
            month_maps = {}
            for m in sorted(set(m_tr)):
                va_m = train_months[val_idx] == m
                if va_m.sum() >= 5:
                    mm, _ = compute_map(y_eval[val_idx][va_m], preds_va[va_m])
                    month_maps[m] = mm

            sample_weights = np.ones(len(y_tr), dtype=float)
            if month_maps:
                worst = min(month_maps, key=month_maps.get)
                sample_weights[m_tr == worst] *= 2.0

            # Round 2: train with DRO weights
            model = catb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
                rsm=0.6, bootstrap_type="MVS", subsample=0.7,
                model_shrink_rate=0.1, early_stopping_rounds=100,
                random_seed=42 + seed + fold, verbose=0, task_type="CPU",
            )
            model.fit(X_tr, y_tr, sample_weight=sample_weights,
                      eval_set=catb.Pool(X_va, y_eval[val_idx]))

            oof_seed[val_idx] = model.predict_proba(X_va)
            test_seed += model.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        oof_map, _ = compute_map(y_eval, oof_seed)
        print(f"    Seed {seed+1}/{n_seeds}: mAP={oof_map:.4f} ({time.time()-t_seed:.0f}s)", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    final_map, per_class = compute_map(y_eval, oof_mean)
    return oof_mean, test_mean, final_map, per_class

# ══════════════════════════════════════════════════════════════
# Ensemble + Evaluation
# ══════════════════════════════════════════════════════════════

def rank_power_ensemble(preds_list, weights, power=1.5):
    from scipy.stats import rankdata
    n = preds_list[0].shape[0]; nc = preds_list[0].shape[1]
    final = np.zeros((n, nc))
    for c in range(nc):
        for preds, w in zip(preds_list, weights):
            final[:, c] += w * (rankdata(preds[:, c]) / n) ** power
    return final

def tune_ensemble(oof_ranker, oof_cb, y_eval):
    best_score = -1; best_config = (0.5, 0.5, 1.5)
    for power in [1.0, 1.5, 2.0, 3.0]:
        for w_r in np.arange(0.0, 1.05, 0.1):
            w_c = 1.0 - w_r
            blended = rank_power_ensemble([oof_ranker, oof_cb], [w_r, w_c], power)
            score, _ = compute_map(y_eval, blended)
            if score > best_score:
                best_score = score; best_config = (round(w_r, 2), round(w_c, 2), power)
    return best_config, best_score

def lomo_eval(oof, y_eval, months):
    maps = {}
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() >= 10:
            maps[m], _ = compute_map(y_eval[mask], oof[mask])
    avg = np.mean(list(maps.values()))
    return avg, maps

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_config(name, use_all_features, use_relabeled, min_child=20):
    print(f"\n{'='*90}", flush=True)
    print(f"  CONFIG: {name}", flush=True)
    print(f"{'='*90}", flush=True)

    t_total = time.time()
    train_df, test_df, y_orig, y_train, months, test_months, groups, X_tr, X_te, keep = \
        load_data(use_all_features=use_all_features, use_relabeled=use_relabeled)

    # OvR LambdaRank
    print(f"\n  OvR LambdaRank ({N_SEEDS_OVR} seeds)...", flush=True)
    oof_r, test_r, _, _ = run_ovr_lambdarank(
        X_tr, y_train, y_orig, months, X_te, groups,
        min_child=min_child, keep_mask=keep if use_relabeled else None)

    # CatBoost DRO
    print(f"\n  CatBoost DRO ({N_SEEDS_CB} seeds)...", flush=True)
    oof_cb, test_cb, _, _ = run_catboost_dro(
        X_tr, y_train, y_orig, months, X_te, groups,
        keep_mask=keep if use_relabeled else None)

    # Ensemble
    print(f"\n  Tuning ensemble...", flush=True)
    best_cfg, best_skf = tune_ensemble(oof_r, oof_cb, y_orig)
    w_r, w_c, pw = best_cfg
    oof_blend = rank_power_ensemble([oof_r, oof_cb], [w_r, w_c], pw)
    test_blend = rank_power_ensemble([test_r, test_cb], [w_r, w_c], pw)

    skf, pc = compute_map(y_orig, oof_blend)
    lomo, lomo_ms = lomo_eval(oof_blend, y_orig, months)

    print(f"\n  {name} RESULTS:", flush=True)
    print(f"    Weights: ranker={w_r} cb={w_c} power={pw}", flush=True)
    print(f"    SKF={skf:.4f}  LOMO={lomo:.4f}", flush=True)
    month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(lomo_ms.items()))
    print(f"    Months: {month_str}", flush=True)
    for cls in CLASSES:
        mk = " ***" if pc[cls] < 0.4 else ""
        print(f"    {cls:15s}: {pc[cls]:.4f}{mk}", flush=True)
    print(f"    Time: {time.time()-t_total:.0f}s", flush=True)

    # Save
    tag = name.lower().replace(" ", "_").replace("+", "_")
    save_submission(test_blend, f"e187_{tag}_blend", cv_map=skf)
    save_submission(test_r, f"e187_{tag}_ranker", cv_map=compute_map(y_orig, oof_r)[0])
    save_submission(test_cb, f"e187_{tag}_cb", cv_map=compute_map(y_orig, oof_cb)[0])

    np.save(ROOT / f"oof_e187_{tag}_ranker.npy", oof_r)
    np.save(ROOT / f"oof_e187_{tag}_cb.npy", oof_cb)
    np.save(ROOT / f"oof_e187_{tag}_blend.npy", oof_blend)
    np.save(ROOT / f"test_e187_{tag}_blend.npy", test_blend)

    return skf, lomo, pc

# Run configs
results = []

# A: Original E175 config (100 features, original labels)
s, l, p = run_config("Original E175", use_all_features=False, use_relabeled=False)
results.append(("Original E175 (100feat, orig labels)", s, l, p))

# B: ALL features, original labels
s, l, p = run_config("ALL features", use_all_features=True, use_relabeled=False)
results.append(("ALL features (327feat, orig labels)", s, l, p))

# C: 100 features, relabeled
s, l, p = run_config("Relabeled", use_all_features=False, use_relabeled=True)
results.append(("Relabeled (100feat, clean labels)", s, l, p))

# D: ALL features + relabeled + lower min_child
s, l, p = run_config("ALL+Relabel+MinChild5", use_all_features=True, use_relabeled=True, min_child=5)
results.append(("ALL+Relabel+mc5 (full improvements)", s, l, p))

# Summary
print(f"\n{'='*90}", flush=True)
print(f"  E187 FINAL COMPARISON", flush=True)
print(f"{'='*90}", flush=True)
print(f"\n  {'Config':45s} {'SKF':>7s} {'LOMO':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s}", flush=True)
print(f"  {'-'*80}", flush=True)
for name, skf, lomo_val, pc in results:
    print(f"  {name:45s} {skf:7.4f} {lomo_val:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} {pc['Birds of Prey']:7.4f}", flush=True)

print(f"\n  E175 reference (LB=0.59): SKF=0.7043", flush=True)
print(f"\nDone.", flush=True)
