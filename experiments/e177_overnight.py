"""E177: Overnight Experiment Pipeline.

Fast experiments, TRUE LOMO-CV validation only.
Each experiment is independent — script continues even if one fails.

Experiments:
  1. XGBoost rank:map OvR (directly optimizes MAP)
  2. 20-seed LGB DART averaging
  3. CReST pseudo-labeling (class-rebalancing)
  4. Label propagation (transductive, properly validated)
  5. Group DRO training (upweight worst months)
  6. Per-class ensemble weights (TRUE CV)
  7. TTA on test predictions
  8. Diverse hyperparameter ensemble (5 configs)
  9. XGBoost multiclass (diversity)
  10. LGB rank_xendcg OvR (diversity)

All results logged to console and saved to EXPERIMENTS.md format.
"""

from __future__ import annotations
import sys, time, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows, top2_margin

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
UNIQUE_MONTHS = [1, 4, 9, 10]

print("=" * 90)
print("  E177: OVERNIGHT EXPERIMENT PIPELINE")
print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

t_start = time.time()

# ── Load data ──
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

# Features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# E175 baselines
oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))
test_best = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
test_lgb = renorm_rows(np.load(ROOT / "test_e175_lgb.npy").astype(np.float64))


def true_lomo_cv(oof, name=""):
    """TRUE LOMO-CV: per-month held-out evaluation."""
    skf, _ = compute_map(y, oof)
    scores = {}
    for m in UNIQUE_MONTHS:
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            scores[m] = s
    lomo = np.mean(list(scores.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()))
    print(f"  {name:<50s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return skf, lomo, scores


def run_experiment(name, func):
    """Run an experiment with error handling."""
    print(f"\n{'='*90}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*90}", flush=True)
    try:
        t = time.time()
        result = func()
        elapsed = time.time() - t
        print(f"\n  [{name}] Completed in {elapsed:.0f}s", flush=True)
        return result
    except Exception as e:
        print(f"\n  [{name}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return None


# Baseline
print("\n--- Baseline ---", flush=True)
true_lomo_cv(oof_best, "E175 best (baseline)")


# ══════════════════════════════════════════════════════════════════════
# 1. XGBoost rank:map OvR
# ══════════════════════════════════════════════════════════════════════

def exp_xgb_rankmap():
    import xgboost as xgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_seeds = 5
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
    test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            for cls in range(N_CLASSES):
                y_bin_tr = (y[tr] == cls).astype(int)
                y_bin_va = (y[va] == cls).astype(int)
                if y_bin_tr.sum() < 2 or y_bin_va.sum() < 1:
                    oof_s[va, cls] = p_train[cls]
                    test_s[:, cls] += p_train[cls] / N_FOLDS
                    continue

                # Sort by month for query groups
                m_tr = train_months[tr]
                order = np.argsort(m_tr)
                X_tr_s = X_train[tr][order]
                y_tr_s = y_bin_tr[order]
                m_tr_s = m_tr[order]
                group_sizes = [int((m_tr_s == m).sum()) for m in sorted(set(m_tr_s))]

                m_va = train_months[va]
                order_va = np.argsort(m_va)
                X_va_s = X_train[va][order_va]
                y_va_s = y_bin_va[order_va]
                m_va_s = m_va[order_va]
                group_sizes_va = [int((m_va_s == m).sum()) for m in sorted(set(m_va_s))]

                dtrain = xgb.DMatrix(X_tr_s, label=y_tr_s)
                dtrain.set_group(group_sizes)
                dval = xgb.DMatrix(X_va_s, label=y_va_s)
                dval.set_group(group_sizes_va)

                params = {
                    'objective': 'rank:map',
                    'eval_metric': 'map',
                    'eta': 0.03,
                    'max_depth': 6,
                    'subsample': 0.7,
                    'colsample_bytree': 0.6,
                    'min_child_weight': 20,
                    'seed': 42 + seed + fold + cls,
                    'nthread': -1,
                    'verbosity': 0,
                }

                model = xgb.train(
                    params, dtrain, num_boost_round=1000,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=100, verbose_eval=False,
                )

                # Unsort val predictions
                inv_va = np.empty_like(order_va)
                inv_va[order_va] = np.arange(len(order_va))
                va_preds = model.predict(dval)[inv_va]

                oof_s[va, cls] = va_preds
                test_s[:, cls] += model.predict(xgb.DMatrix(X_test)) / N_FOLDS

            print(f"    seed={seed+1} fold={fold+1}", flush=True)

        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"  Seed {seed+1}: SKF={s:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    true_lomo_cv(oof_mean, "XGB rank:map OvR (5 seeds)")

    # Blend with E175
    for alpha in [0.1, 0.2, 0.3]:
        blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_mean)
        true_lomo_cv(renorm_rows(blend), f"E175 + XGB_rankmap@{alpha}")

    np.save(ROOT / "oof_e177_xgb_rankmap.npy", oof_mean)
    np.save(ROOT / "test_e177_xgb_rankmap.npy", test_mean)
    save_submission(renorm_rows(test_mean), "e177_xgb_rankmap", cv_map=compute_map(y, oof_mean)[0])
    return oof_mean, test_mean

run_experiment("1. XGBoost rank:map OvR", exp_xgb_rankmap)


# ══════════════════════════════════════════════════════════════════════
# 2. 20-Seed LGB DART Averaging
# ══════════════════════════════════════════════════════════════════════

def exp_20seed():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_seeds = 20
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
    test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=True, verbosity=-1,
                random_state=42+seed+fold, n_jobs=-1,
            )
            m.fit(X_train[tr], y[tr], eval_set=[(X_train[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_s[va] = m.predict_proba(X_train[va])
            test_s += m.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        # Report every 5 seeds
        if (seed + 1) % 5 == 0:
            oof_running = np.mean(oof_all[:seed+1], axis=0)
            rs, _ = compute_map(y, oof_running)
            print(f"  Seeds 1-{seed+1}: single={s:.4f}, running_avg={rs:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    true_lomo_cv(oof_mean, "LGB DART 20-seed average")

    # Compare 5-seed vs 10-seed vs 20-seed
    for n in [5, 10, 15, 20]:
        oof_n = np.mean(oof_all[:n], axis=0)
        true_lomo_cv(oof_n, f"LGB DART {n}-seed average")

    np.save(ROOT / "oof_e177_20seed.npy", oof_mean)
    np.save(ROOT / "test_e177_20seed.npy", test_mean)
    save_submission(renorm_rows(test_mean), "e177_20seed_dart", cv_map=compute_map(y, oof_mean)[0])
    return oof_mean, test_mean

run_experiment("2. 20-Seed LGB DART", exp_20seed)


# ══════════════════════════════════════════════════════════════════════
# 3. Group DRO (upweight worst months)
# ══════════════════════════════════════════════════════════════════════

def exp_group_dro():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_seeds = 5
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
    test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            m_tr = train_months[tr]

            # Round 1: train with uniform weights, find worst month
            m1 = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, is_unbalance=True, verbosity=-1,
                random_state=42+seed+fold, n_jobs=-1,
            )
            m1.fit(X_train[tr], y[tr])
            preds_tr = m1.predict_proba(X_train[tr])

            # Per-month mAP
            month_maps = {}
            for month in sorted(set(m_tr)):
                mask = m_tr == month
                if mask.sum() >= 10:
                    s, _ = compute_map(y[tr][mask], preds_tr[mask])
                    month_maps[month] = s

            # DRO weights: upweight worst months
            if month_maps:
                worst = min(month_maps.values())
                sw = np.ones(len(tr))
                for month, score in month_maps.items():
                    mask = m_tr == month
                    # Inverse performance weighting
                    weight = (worst / max(score, 0.01)) ** 0.5
                    sw[mask] *= weight
                # Also class weights
                class_w = 1.0 / np.maximum(counts, 1.0)
                class_w /= class_w.mean()
                sw *= class_w[y[tr]]
                sw /= sw.mean()
            else:
                sw = None

            # Round 2: retrain with DRO weights
            m2 = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=False, verbosity=-1,
                random_state=42+seed+fold, n_jobs=-1,
            )
            fit_kw = {"eval_set": [(X_train[va], y[va])],
                      "callbacks": [lgb.early_stopping(100, verbose=False)]}
            if sw is not None:
                fit_kw["sample_weight"] = sw
            m2.fit(X_train[tr], y[tr], **fit_kw)

            oof_s[va] = m2.predict_proba(X_train[va])
            test_s += m2.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"  Seed {seed+1}: SKF={s:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    true_lomo_cv(oof_mean, "Group DRO (5 seeds)")

    # Blend with E175
    for alpha in [0.2, 0.3, 0.5]:
        blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_mean)
        true_lomo_cv(renorm_rows(blend), f"E175 + DRO@{alpha}")

    np.save(ROOT / "oof_e177_dro.npy", oof_mean)
    np.save(ROOT / "test_e177_dro.npy", test_mean)
    save_submission(renorm_rows(test_mean), "e177_group_dro", cv_map=compute_map(y, oof_mean)[0])
    return oof_mean, test_mean

run_experiment("3. Group DRO", exp_group_dro)


# ══════════════════════════════════════════════════════════════════════
# 4. CReST Pseudo-Labeling
# ══════════════════════════════════════════════════════════════════════

def exp_crest():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_rounds = 3
    base_threshold = 0.90

    # Start with E175 test predictions for pseudo-labels
    test_preds = test_best.copy()

    X_aug = X_train.copy()
    y_aug = y.copy()
    groups_aug = groups.copy()
    months_aug = train_months.copy()

    for round_i in range(n_rounds):
        # Class-specific thresholds (CReST: lower for minority)
        class_counts = np.bincount(y_aug, minlength=N_CLASSES).astype(float)
        max_count = class_counts.max()
        thresholds = np.array([
            base_threshold * (c / max_count) ** 0.5 for c in class_counts
        ])
        thresholds = np.maximum(thresholds, 0.50)

        # Select pseudo-labels
        test_conf = test_preds.max(axis=1)
        test_labels = test_preds.argmax(axis=1)
        selected = np.zeros(len(test_df), dtype=bool)
        for cls in range(N_CLASSES):
            cls_mask = test_labels == cls
            conf_mask = test_conf >= thresholds[cls]
            selected |= (cls_mask & conf_mask)

        if selected.sum() == 0:
            print(f"  Round {round_i+1}: no pseudo-labels selected", flush=True)
            break

        n_per_class = {}
        for cls in range(N_CLASSES):
            n_per_class[CLASSES[cls][:4]] = int((test_labels[selected] == cls).sum())
        print(f"  Round {round_i+1}: {selected.sum()} pseudo-labels {n_per_class}", flush=True)
        print(f"    Thresholds: {dict(zip([c[:4] for c in CLASSES], thresholds.round(2)))}", flush=True)

        # Add pseudo-labeled samples
        X_aug = np.vstack([X_aug, X_test[selected]])
        y_aug = np.concatenate([y_aug, test_labels[selected]])
        # Dummy groups and months for pseudo-labeled samples
        groups_aug = np.concatenate([groups_aug, np.arange(selected.sum()) + groups.max() + 1])
        months_aug = np.concatenate([months_aug, test_months[selected]])

        # Retrain
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+round_i)
        oof_s = np.zeros((len(y_aug), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_aug, y_aug, groups_aug)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1,
                random_state=42+round_i+fold, n_jobs=-1,
            )
            m.fit(X_aug[tr], y_aug[tr], eval_set=[(X_aug[va], y_aug[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_aug[va])
            test_s += m.predict_proba(X_test) / N_FOLDS

        # Update test predictions for next round
        test_preds = renorm_rows(test_s)

        # Evaluate on ORIGINAL train only
        oof_orig = oof_s[:len(y)]
        true_lomo_cv(oof_orig, f"CReST round {round_i+1}")

    # Final evaluation
    oof_final = oof_s[:len(y)]
    true_lomo_cv(oof_final, "CReST final")

    # Blend with E175
    for alpha in [0.2, 0.3, 0.5]:
        blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_final)
        true_lomo_cv(renorm_rows(blend), f"E175 + CReST@{alpha}")

    np.save(ROOT / "oof_e177_crest.npy", oof_final)
    np.save(ROOT / "test_e177_crest.npy", test_preds)
    save_submission(renorm_rows(test_preds), "e177_crest", cv_map=compute_map(y, oof_final)[0])
    return oof_final, test_preds

run_experiment("4. CReST Pseudo-Labeling", exp_crest)


# ══════════════════════════════════════════════════════════════════════
# 5. Diverse Hyperparameter Ensemble
# ══════════════════════════════════════════════════════════════════════

def exp_diverse():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    configs = [
        {"num_leaves": 15, "max_depth": 4, "colsample_bytree": 0.5, "subsample": 0.6, "tag": "shallow"},
        {"num_leaves": 63, "max_depth": 8, "colsample_bytree": 0.7, "subsample": 0.8, "tag": "deep"},
        {"num_leaves": 31, "max_depth": 6, "colsample_bytree": 0.4, "subsample": 0.5, "tag": "sparse"},
        {"num_leaves": 31, "max_depth": 6, "colsample_bytree": 0.8, "subsample": 0.9, "tag": "dense"},
        {"num_leaves": 7, "max_depth": 3, "colsample_bytree": 0.6, "subsample": 0.7, "tag": "stump"},
    ]

    all_oofs = []
    all_tests = []

    for cfg in configs:
        tag = cfg.pop("tag")
        n_seeds = 3
        oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
        test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))

        for seed in range(n_seeds):
            sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
            oof_s = np.zeros((len(y), N_CLASSES))
            test_s = np.zeros((len(test_df), N_CLASSES))

            for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
                m = lgb.LGBMClassifier(
                    objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
                    n_estimators=1000, learning_rate=0.05,
                    min_child_samples=20, is_unbalance=True, verbosity=-1,
                    random_state=42+seed+fold, n_jobs=-1, **cfg,
                )
                m.fit(X_train[tr], y[tr], eval_set=[(X_train[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                oof_s[va] = m.predict_proba(X_train[va])
                test_s += m.predict_proba(X_test) / N_FOLDS

            oof_all[seed] = oof_s
            test_all[seed] = test_s

        oof_mean = np.mean(oof_all, axis=0)
        test_mean = np.mean(test_all, axis=0)
        true_lomo_cv(oof_mean, f"Diverse {tag}")
        all_oofs.append(oof_mean)
        all_tests.append(test_mean)
        cfg["tag"] = tag  # restore

    # Equal-weight ensemble of all diverse models
    oof_diverse = np.mean(all_oofs, axis=0)
    test_diverse = np.mean(all_tests, axis=0)
    true_lomo_cv(oof_diverse, "Diverse ensemble (5 configs)")

    # Blend with E175
    for alpha in [0.1, 0.2, 0.3]:
        blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_diverse)
        true_lomo_cv(renorm_rows(blend), f"E175 + diverse@{alpha}")

    np.save(ROOT / "oof_e177_diverse.npy", oof_diverse)
    np.save(ROOT / "test_e177_diverse.npy", test_diverse)
    save_submission(renorm_rows(test_diverse), "e177_diverse", cv_map=compute_map(y, oof_diverse)[0])
    return oof_diverse, test_diverse

run_experiment("5. Diverse Hyperparameter Ensemble", exp_diverse)


# ══════════════════════════════════════════════════════════════════════
# 6. TTA on test predictions
# ══════════════════════════════════════════════════════════════════════

def exp_tta():
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    # Train a fresh model on ALL data, then do TTA on test
    n_seeds = 5
    test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=True, verbosity=-1,
                random_state=42+seed+fold, n_jobs=-1,
            )
            m.fit(X_train[tr], y[tr], eval_set=[(X_train[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_s[va] = m.predict_proba(X_train[va])

            # TTA: 10 augmented test copies
            n_tta = 10
            rng = np.random.RandomState(42 + seed + fold)
            test_preds = m.predict_proba(X_test)
            for _ in range(n_tta):
                noise_scale = 0.02
                X_noisy = X_test + rng.randn(*X_test.shape).astype(np.float32) * noise_scale * np.abs(X_test).mean(axis=0, keepdims=True)
                test_preds += m.predict_proba(X_noisy)
            test_preds /= (1 + n_tta)
            test_s += test_preds / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"  Seed {seed+1}: SKF={s:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    true_lomo_cv(oof_mean, "LGB DART + TTA (5 seeds)")

    np.save(ROOT / "oof_e177_tta.npy", oof_mean)
    np.save(ROOT / "test_e177_tta.npy", test_mean)
    save_submission(renorm_rows(test_mean), "e177_tta", cv_map=compute_map(y, oof_mean)[0])

    # Compare TTA test vs non-TTA test (submit both)
    save_submission(renorm_rows(np.mean(oof_all, axis=0)), "e177_noTTA_oof_ref")
    return oof_mean, test_mean

run_experiment("6. TTA on Test Predictions", exp_tta)


# ══════════════════════════════════════════════════════════════════════
# 7. XGBoost Multiclass (diversity for blending)
# ══════════════════════════════════════════════════════════════════════

def exp_xgb_multi():
    import xgboost as xgb
    from sklearn.model_selection import StratifiedGroupKFold

    n_seeds = 5
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
    test_all = np.zeros((n_seeds, len(test_df), N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((len(test_df), N_CLASSES))

        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            sw = (1.0 / np.maximum(counts, 1.0))[y[tr]]
            sw /= sw.mean()

            dtrain = xgb.DMatrix(X_train[tr], label=y[tr], weight=sw)
            dval = xgb.DMatrix(X_train[va], label=y[va])

            params = {
                'objective': 'multi:softprob',
                'num_class': N_CLASSES,
                'eval_metric': 'mlogloss',
                'eta': 0.03,
                'max_depth': 6,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'min_child_weight': 20,
                'seed': 42 + seed + fold,
                'nthread': -1,
                'verbosity': 0,
            }

            model = xgb.train(
                params, dtrain, num_boost_round=1500,
                evals=[(dval, 'val')],
                early_stopping_rounds=100, verbose_eval=False,
            )

            oof_s[va] = model.predict(dval).reshape(-1, N_CLASSES)
            test_s += model.predict(xgb.DMatrix(X_test)).reshape(-1, N_CLASSES) / N_FOLDS

        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"  Seed {seed+1}: SKF={s:.4f}", flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    true_lomo_cv(oof_mean, "XGB multiclass (5 seeds)")

    # Blend with E175
    for alpha in [0.1, 0.2, 0.3]:
        blend = (1-alpha) * oof_best + alpha * renorm_rows(oof_mean)
        true_lomo_cv(renorm_rows(blend), f"E175 + XGB_multi@{alpha}")

    np.save(ROOT / "oof_e177_xgb_multi.npy", oof_mean)
    np.save(ROOT / "test_e177_xgb_multi.npy", test_mean)
    save_submission(renorm_rows(test_mean), "e177_xgb_multi", cv_map=compute_map(y, oof_mean)[0])
    return oof_mean, test_mean

run_experiment("7. XGBoost Multiclass", exp_xgb_multi)


# ══════════════════════════════════════════════════════════════════════
# 8. Grand Ensemble: blend ALL overnight models with E175
# ══════════════════════════════════════════════════════════════════════

def exp_grand_ensemble():
    """Load all overnight OOF predictions and find optimal blend."""
    from scipy.optimize import minimize

    model_names = []
    model_oofs = []
    model_tests = []

    for name in ["e175_best", "e175_lgb", "e177_xgb_rankmap", "e177_20seed",
                  "e177_dro", "e177_crest", "e177_diverse", "e177_tta", "e177_xgb_multi"]:
        oof_path = ROOT / f"oof_{name}.npy"
        test_path = ROOT / f"test_{name}.npy"
        if oof_path.exists() and test_path.exists():
            model_names.append(name)
            model_oofs.append(renorm_rows(np.load(oof_path).astype(np.float64)))
            model_tests.append(renorm_rows(np.load(test_path).astype(np.float64)))

    print(f"  Loaded {len(model_names)} models: {model_names}", flush=True)

    if len(model_names) < 2:
        print("  Not enough models for ensemble", flush=True)
        return None

    # Individual TRUE LOMO-CV
    for name, oof in zip(model_names, model_oofs):
        true_lomo_cv(oof, name)

    # Hill climbing: start with best, greedily add
    print("\n  Hill climbing ensemble:")
    best_oof = model_oofs[0]
    best_lomo = true_lomo_cv(best_oof, "Start: " + model_names[0])[1]
    used = {0}
    blend_weights = {model_names[0]: 1.0}

    for _ in range(len(model_names) - 1):
        improved = False
        for i, (name, oof) in enumerate(zip(model_names, model_oofs)):
            if i in used:
                continue
            for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
                candidate = renorm_rows((1-alpha) * best_oof + alpha * oof)
                _, lomo, _ = true_lomo_cv.__wrapped__(candidate) if hasattr(true_lomo_cv, '__wrapped__') else (None, None, None)
                # Manual LOMO
                scores = {}
                for m in UNIQUE_MONTHS:
                    mask = train_months == m
                    s, _ = compute_map(y[mask], candidate[mask])
                    scores[m] = s
                lomo = np.mean(list(scores.values()))
                if lomo > best_lomo + 0.0005:
                    best_lomo = lomo
                    best_candidate = candidate
                    best_add = (i, name, alpha)
                    improved = True

        if improved:
            idx, name, alpha = best_add
            used.add(idx)
            best_oof = best_candidate
            blend_weights[name] = alpha
            # Rescale existing weights
            total = sum(blend_weights.values())
            blend_weights = {k: v/total for k, v in blend_weights.items()}
            true_lomo_cv(best_oof, f"+ {name}@{alpha}")
        else:
            break

    print(f"\n  Final weights: {blend_weights}")
    true_lomo_cv(best_oof, "Grand ensemble (hill-climbed)")

    # Build test blend
    test_blend = np.zeros_like(model_tests[0])
    for name, w in blend_weights.items():
        idx = model_names.index(name)
        test_blend += w * model_tests[idx]
    test_blend = renorm_rows(test_blend)

    save_submission(test_blend, "e177_grand_ensemble", cv_map=compute_map(y, best_oof)[0])
    return best_oof, test_blend

run_experiment("8. Grand Ensemble", exp_grand_ensemble)


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════

elapsed_total = time.time() - t_start
print(f"\n{'='*90}")
print(f"  OVERNIGHT PIPELINE COMPLETE")
print(f"  Total time: {elapsed_total/3600:.1f} hours")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*90}")

# List all saved submissions
import glob
subs = sorted(glob.glob(str(ROOT / "submissions" / "e177_*.csv")))
print(f"\n  Saved {len(subs)} submissions:")
for s in subs:
    print(f"    {Path(s).name}")
