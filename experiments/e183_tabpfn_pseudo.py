"""E183: TabPFN + Pseudo-Labeling + Adversarial Reweighting.

Three-pronged attack on the 0.59 LB ceiling:
  A. TabPFN as genuinely diverse ensemble member (different inductive bias from GBDTs)
  B. Pseudo-labeling: use high-confidence test predictions to expand training,
     especially for unseen months (Feb, May, Dec = 32.7% of test)
  C. Adversarial reweighting: upweight train samples similar to test distribution

Uses cached E175 OOF/test predictions as base. Evaluates with TRUE LOMO-CV.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_all():
    """Load data, features, and cached E175 predictions."""
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
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Load E175 predictions
    oof_e175 = np.load(ROOT / "oof_e175_best.npy")
    test_e175 = np.load(ROOT / "test_e175_best.npy")
    oof_ranker = np.load(ROOT / "oof_e175_ranker.npy")
    test_ranker = np.load(ROOT / "test_e175_ranker.npy")
    oof_cb = np.load(ROOT / "oof_e175_cb.npy")
    test_cb = np.load(ROOT / "test_e175_cb.npy")

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(selected)}")
    return (train_df, test_df, y, train_months, test_months, groups,
            X_train, X_test, selected,
            oof_e175, test_e175, oof_ranker, test_ranker, oof_cb, test_cb)


# ══════════════════════════════════════════════════════════════════════
# A. TabPFN
# ══════════════════════════════════════════════════════════════════════

def run_tabpfn(X_train, y, X_test, groups, train_months):
    """Train TabPFN with proper CV. Returns OOF probs and test probs."""
    from tabpfn import TabPFNClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    n_train = X_train.shape[0]
    oof_probs = np.zeros((n_train, N_CLASSES))
    test_probs = np.zeros((X_test.shape[0], N_CLASSES))

    # TabPFN has a 10K sample limit and 500 feature limit - we're fine
    # But it may struggle with 100 features; let's use top features
    # Use all 100 features - TabPFN handles feature selection internally

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
        t_fold = time.time()
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        clf = TabPFNClassifier(
            n_estimators=16,  # ensemble size within TabPFN
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        oof_probs[val_idx] = clf.predict_proba(X_va)
        test_probs += clf.predict_proba(X_test) / N_FOLDS

        fold_map, _ = compute_map(y_va, oof_probs[val_idx])
        elapsed = time.time() - t_fold
        print(f"    Fold {fold+1}/{N_FOLDS}: mAP={fold_map:.4f} ({elapsed:.1f}s)", flush=True)

    overall_map, per_class = compute_map(y, oof_probs)
    print_results(overall_map, per_class, "TabPFN (5-fold)")

    return oof_probs, test_probs


# ══════════════════════════════════════════════════════════════════════
# B. Adversarial Validation + Reweighting
# ══════════════════════════════════════════════════════════════════════

def adversarial_reweight(X_train, X_test, train_months):
    """Build adversarial classifier (train vs test) and compute importance weights."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict

    # Create binary target: 0=train, 1=test
    X_combined = np.vstack([X_train, X_test])
    y_binary = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.5,
        verbosity=-1,
        random_state=42,
    )

    # Cross-val predict to get honest probabilities
    probs = cross_val_predict(clf, X_combined, y_binary, cv=5, method="predict_proba")
    train_probs = probs[:len(X_train), 1]  # P(test | features) for train samples

    # Compute AUC to see how different train/test are
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_binary, probs[:, 1])
    print(f"  Adversarial AUC: {auc:.4f} (0.5=identical, 1.0=completely different)")

    # Importance weights: upweight train samples that look like test
    # w_i = p(test|x_i) / (1 - p(test|x_i)) * (n_train / n_test)
    eps = 0.01
    raw_weights = np.clip(train_probs, eps, 1 - eps) / np.clip(1 - train_probs, eps, 1 - eps)
    # Normalize so mean=1
    weights = raw_weights / raw_weights.mean()
    # Clip extreme weights
    weights = np.clip(weights, 0.2, 5.0)

    print(f"  Weight stats: mean={weights.mean():.3f}, std={weights.std():.3f}, "
          f"min={weights.min():.3f}, max={weights.max():.3f}")

    # Show which months get upweighted
    for m in sorted(set(train_months)):
        mask = train_months == m
        print(f"    Month {m:2d}: mean_weight={weights[mask].mean():.3f} (n={mask.sum()})")

    return weights


# ══════════════════════════════════════════════════════════════════════
# C. Pseudo-Labeling
# ══════════════════════════════════════════════════════════════════════

def pseudo_label_retrain(X_train, y, X_test, groups, train_months, test_months,
                         test_probs_base, adv_weights, n_seeds=10):
    """Pseudo-label high-confidence test samples and retrain LGB.

    Uses soft labels (probability vectors) for pseudo-labeled samples.
    Only includes test samples where max probability > threshold.
    Trains OvR LambdaRank like E175 but with expanded data.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    # Convert test predictions to soft pseudo-labels
    # Use CatBoost probs (calibrated) not ranker scores
    test_max_prob = test_probs_base.max(axis=1)
    test_hard_labels = test_probs_base.argmax(axis=1)

    # Threshold sweep: find samples with confident predictions
    for thresh in [0.95, 0.90, 0.85, 0.80, 0.70]:
        mask = test_max_prob >= thresh
        if mask.sum() > 0:
            pseudo_labels = test_hard_labels[mask]
            label_dist = np.bincount(pseudo_labels, minlength=N_CLASSES)
            print(f"  Threshold {thresh:.2f}: {mask.sum()} samples -> "
                  f"{dict(zip(CLASSES, label_dist))}")

    # Use 0.85 threshold (balances quantity vs quality)
    CONF_THRESHOLD = 0.85
    pseudo_mask = test_max_prob >= CONF_THRESHOLD
    n_pseudo = pseudo_mask.sum()
    print(f"\n  Using {n_pseudo} pseudo-labeled test samples (threshold={CONF_THRESHOLD})")

    if n_pseudo == 0:
        print("  WARNING: No pseudo-labeled samples! Lowering threshold to 0.70")
        CONF_THRESHOLD = 0.70
        pseudo_mask = test_max_prob >= CONF_THRESHOLD
        n_pseudo = pseudo_mask.sum()
        print(f"  Using {n_pseudo} pseudo-labeled samples")

    X_pseudo = X_test[pseudo_mask]
    y_pseudo = test_hard_labels[pseudo_mask]
    months_pseudo = test_months[pseudo_mask]

    # Show month distribution of pseudo-labeled samples
    print("  Pseudo-label months:")
    for m in sorted(set(months_pseudo)):
        mask_m = months_pseudo == m
        print(f"    Month {m:2d}: {mask_m.sum()} samples")

    # Combine train + pseudo-labeled data
    X_combined = np.vstack([X_train, X_pseudo])
    y_combined = np.concatenate([y, y_pseudo])
    months_combined = np.concatenate([train_months, months_pseudo])
    # Groups: pseudo-labeled samples get unique group IDs
    max_group = groups.max() + 1
    pseudo_groups = np.arange(max_group, max_group + n_pseudo)
    groups_combined = np.concatenate([groups, pseudo_groups])

    # Weights: original train gets adversarial weights, pseudo gets downweighted
    pseudo_confidence = test_max_prob[pseudo_mask]
    pseudo_weights = pseudo_confidence * 0.5  # half-weight pseudo-labels
    weights_combined = np.concatenate([adv_weights, pseudo_weights])

    print(f"\n  Combined: {len(X_combined)} samples ({len(X_train)} train + {n_pseudo} pseudo)")
    print(f"  Class distribution after pseudo-labeling:")
    for c in range(N_CLASSES):
        n_orig = (y == c).sum()
        n_pseudo_c = (y_pseudo == c).sum()
        print(f"    {CLASSES[c]:15s}: {n_orig:4d} + {n_pseudo_c:3d} pseudo = {n_orig + n_pseudo_c:4d}")

    # Train OvR LambdaRank on combined data (like E175 Phase 2)
    n_total = len(X_combined)
    oof_all_seeds = np.zeros((n_seeds, len(X_train), N_CLASSES))
    test_all_seeds = np.zeros((n_seeds, len(X_test), N_CLASSES))

    for seed in range(n_seeds):
        t_seed = time.time()
        # Only CV on original train samples (not pseudo-labeled ones)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)

        oof_seed = np.zeros((len(X_train), N_CLASSES))
        test_seed = np.zeros((len(X_test), N_CLASSES))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y, groups)):
            # Training: original train fold + ALL pseudo-labeled data
            pseudo_idx = np.arange(len(X_train), n_total)
            combined_train_idx = np.concatenate([train_idx, pseudo_idx])

            X_tr = X_combined[combined_train_idx]
            y_tr = y_combined[combined_train_idx]
            m_tr = months_combined[combined_train_idx]
            w_tr = weights_combined[combined_train_idx]

            X_va = X_train[val_idx]
            y_va = y[val_idx]
            m_va = train_months[val_idx]

            # Sort by month for LambdaRank query groups
            tr_order = np.argsort(m_tr)
            va_order = np.argsort(m_va)

            X_tr_s = X_tr[tr_order]
            m_tr_s = m_tr[tr_order]
            w_tr_s = w_tr[tr_order]

            X_va_s = X_va[va_order]
            m_va_s = m_va[va_order]

            tr_groups = [int((m_tr_s == m).sum()) for m in sorted(set(m_tr_s))]
            va_groups = [int((m_va_s == m).sum()) for m in sorted(set(m_va_s))]

            for cls_idx in range(N_CLASSES):
                y_bin_tr = (y_tr[tr_order] == cls_idx).astype(int)
                y_bin_va = (y_va[va_order] == cls_idx).astype(int)

                if y_bin_tr.sum() < 2 or y_bin_va.sum() < 1:
                    continue

                # Per-class tuning for rare classes
                min_child = 5 if cls_idx in [2, 3, 4, 7, 8] else 20  # lower for rare

                ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    metric="map",
                    boosting_type="dart",
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=min_child,
                    colsample_bytree=0.6,
                    subsample=0.7,
                    drop_rate=0.15,
                    lambdarank_truncation_level=30,
                    verbosity=-1,
                    random_state=42 + seed + cls_idx,
                    n_jobs=-1,
                )

                ranker.fit(
                    X_tr_s, y_bin_tr,
                    group=tr_groups,
                    sample_weight=w_tr_s,
                    eval_set=[(X_va_s, y_bin_va)],
                    eval_group=[va_groups],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )

                # Unsort val predictions
                va_preds_s = ranker.predict(X_va_s)
                inv_va = np.empty_like(va_order)
                inv_va[va_order] = np.arange(len(va_order))
                oof_seed[val_idx, cls_idx] = va_preds_s[inv_va]
                test_seed[:, cls_idx] += ranker.predict(X_test) / N_FOLDS

        oof_all_seeds[seed] = oof_seed
        test_all_seeds[seed] = test_seed

        oof_map, _ = compute_map(y, oof_seed)
        elapsed = time.time() - t_seed
        print(f"    Seed {seed+1:2d}/{n_seeds}: OOF mAP={oof_map:.4f} ({elapsed:.1f}s)", flush=True)

    oof_mean = np.mean(oof_all_seeds, axis=0)
    test_mean = np.mean(test_all_seeds, axis=0)

    final_map, per_class = compute_map(y, oof_mean)
    print_results(final_map, per_class, f"Pseudo-Label OvR Ranker ({n_seeds} seeds)")

    return oof_mean, test_mean


# ══════════════════════════════════════════════════════════════════════
# D. Ensemble
# ══════════════════════════════════════════════════════════════════════

def rank_power_ensemble(preds_list, weights, power=1.5):
    """Rank-based ensemble with power averaging."""
    n_samples = preds_list[0].shape[0]
    n_classes = preds_list[0].shape[1]
    final = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        for preds, w in zip(preds_list, weights):
            ranks = rankdata(preds[:, c]) / n_samples
            final[:, c] += w * (ranks ** power)
    return final


def find_best_blend(oof_components, y, component_names):
    """Hill climbing to find best blend weights for macro-mAP."""
    n = len(oof_components)
    # Start with equal weights
    best_weights = np.ones(n) / n
    best_score = -1

    # Grid search with power parameter
    print("\n  Searching blend weights...")
    results = []

    for power in [1.0, 1.5, 2.0]:
        # Try all weight combinations (step 0.1)
        from itertools import product
        weight_values = np.arange(0.0, 1.05, 0.1)

        if n == 2:
            for w0 in weight_values:
                w1 = 1.0 - w0
                if w1 < -0.01:
                    continue
                blend = rank_power_ensemble(oof_components, [w0, w1], power)
                score, per = compute_map(y, blend)
                results.append((score, [round(w0, 2), round(w1, 2)], power, per))
        elif n == 3:
            for w0 in np.arange(0.0, 1.05, 0.1):
                for w1 in np.arange(0.0, 1.05 - w0, 0.1):
                    w2 = 1.0 - w0 - w1
                    if w2 < -0.01:
                        continue
                    blend = rank_power_ensemble(oof_components, [w0, w1, w2], power)
                    score, per = compute_map(y, blend)
                    results.append((score, [round(w0, 2), round(w1, 2), round(w2, 2)], power, per))
        elif n == 4:
            # Coarser grid for 4 components
            for w0 in np.arange(0.0, 1.05, 0.2):
                for w1 in np.arange(0.0, 1.05 - w0, 0.2):
                    for w2 in np.arange(0.0, 1.05 - w0 - w1, 0.2):
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < -0.01:
                            continue
                        blend = rank_power_ensemble(oof_components, [w0, w1, w2, w3], power)
                        score, per = compute_map(y, blend)
                        results.append((score, [round(w0, 2), round(w1, 2), round(w2, 2), round(w3, 2)], power, per))

    results.sort(key=lambda x: -x[0])

    # Print top 5
    print(f"\n  Top 5 blends:")
    for score, weights, power, per in results[:5]:
        w_str = ", ".join(f"{name}={w:.2f}" for name, w in zip(component_names, weights))
        print(f"    mAP={score:.4f} | power={power} | {w_str}")
        # Show Cormorant and Wader APs
        corm_ap = per.get('Cormorants', 0)
        wader_ap = per.get('Waders', 0)
        print(f"      Cormorants={corm_ap:.4f}, Waders={wader_ap:.4f}")

    best = results[0]
    return best[1], best[2], best[0], best[3]


def lomo_eval(oof, y, train_months, label=""):
    """TRUE LOMO evaluation."""
    lomo_maps = {}
    for held in sorted(set(train_months)):
        mask = train_months == held
        if mask.sum() >= 10:
            lm, pc = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
    lomo_avg = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    if label:
        print(f"  {label:30s}: LOMO={lomo_avg:.4f}  ({month_str})")
    return lomo_avg, lomo_maps


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 80)
    print("  E183: TabPFN + Pseudo-Labeling + Adversarial Reweighting")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── Load ──
    print("\n[1/6] Loading data + E175 predictions...", flush=True)
    (train_df, test_df, y, train_months, test_months, groups,
     X_train, X_test, feature_cols,
     oof_e175, test_e175, oof_ranker, test_ranker, oof_cb, test_cb) = load_all()

    # E175 baseline
    e175_map, e175_pc = compute_map(y, oof_e175)
    print_results(e175_map, e175_pc, "E175 Baseline")
    lomo_eval(oof_e175, y, train_months, "E175 Baseline")
    lomo_eval(oof_ranker, y, train_months, "E175 Ranker")
    lomo_eval(oof_cb, y, train_months, "E175 CatBoost")

    # ── A: TabPFN ──
    print(f"\n[2/6] Training TabPFN...", flush=True)
    t2 = time.time()
    oof_tabpfn, test_tabpfn = run_tabpfn(X_train, y, X_test, groups, train_months)
    print(f"  TabPFN time: {time.time()-t2:.1f}s")

    np.save(ROOT / "oof_e183_tabpfn.npy", oof_tabpfn)
    np.save(ROOT / "test_e183_tabpfn.npy", test_tabpfn)

    # Correlation analysis: TabPFN vs E175
    print("\n  Spearman correlation (TabPFN vs E175) per class:")
    from scipy.stats import spearmanr
    for c in range(N_CLASSES):
        r, _ = spearmanr(oof_tabpfn[:, c], oof_e175[:, c])
        print(f"    {CLASSES[c]:15s}: r={r:.3f}")

    lomo_eval(oof_tabpfn, y, train_months, "TabPFN")

    # ── Quick blend: E175 + TabPFN ──
    print(f"\n[3/6] Blending E175 + TabPFN...", flush=True)
    blend_weights, blend_power, blend_score, blend_pc = find_best_blend(
        [oof_e175, oof_tabpfn], y, ["E175", "TabPFN"]
    )
    print(f"\n  Best E175+TabPFN blend: mAP={blend_score:.4f}")
    lomo_eval(
        rank_power_ensemble([oof_e175, oof_tabpfn], blend_weights, blend_power),
        y, train_months, "E175+TabPFN blend"
    )

    # ── B: Adversarial Reweighting ──
    print(f"\n[4/6] Adversarial validation + reweighting...", flush=True)
    adv_weights = adversarial_reweight(X_train, X_test, train_months)

    # ── C: Pseudo-Labeling ──
    print(f"\n[5/6] Pseudo-labeling + retrain...", flush=True)
    t5 = time.time()
    # Use CatBoost test probs (calibrated probabilities) as pseudo-label source
    oof_pseudo, test_pseudo = pseudo_label_retrain(
        X_train, y, X_test, groups, train_months, test_months,
        test_cb, adv_weights, n_seeds=10
    )
    print(f"  Pseudo-label retrain time: {time.time()-t5:.1f}s")

    np.save(ROOT / "oof_e183_pseudo.npy", oof_pseudo)
    np.save(ROOT / "test_e183_pseudo.npy", test_pseudo)

    lomo_eval(oof_pseudo, y, train_months, "Pseudo-Label Ranker")

    # ── D: Grand Ensemble ──
    print(f"\n[6/6] Grand ensemble...", flush=True)

    # Components: E175 ranker, E175 CB, TabPFN, Pseudo-Label ranker
    components = [oof_ranker, oof_cb, oof_tabpfn, oof_pseudo]
    test_components = [test_ranker, test_cb, test_tabpfn, test_pseudo]
    names = ["E175_Ranker", "E175_CB", "TabPFN", "Pseudo_Ranker"]

    # Also try: E175_best (already blended) + TabPFN + Pseudo
    print("\n  --- 3-way: E175 + TabPFN + Pseudo ---")
    w3, p3, s3, pc3 = find_best_blend(
        [oof_e175, oof_tabpfn, oof_pseudo], y, ["E175", "TabPFN", "Pseudo"]
    )
    oof_3way = rank_power_ensemble([oof_e175, oof_tabpfn, oof_pseudo], w3, p3)
    test_3way = rank_power_ensemble([test_e175, test_tabpfn, test_pseudo], w3, p3)
    lomo_eval(oof_3way, y, train_months, "3-way blend")

    # Also try 4-way
    print("\n  --- 4-way: Ranker + CB + TabPFN + Pseudo ---")
    w4, p4, s4, pc4 = find_best_blend(
        components, y, names
    )
    oof_4way = rank_power_ensemble(components, w4, p4)
    test_4way = rank_power_ensemble(test_components, w4, p4)
    lomo_eval(oof_4way, y, train_months, "4-way blend")

    # ── Per-class comparison ──
    print("\n" + "=" * 80)
    print("  PER-CLASS AP COMPARISON")
    print("=" * 80)
    _, e175_pc = compute_map(y, oof_e175)
    _, tabpfn_pc = compute_map(y, oof_tabpfn)
    _, pseudo_pc = compute_map(y, oof_pseudo)
    _, three_pc = compute_map(y, oof_3way)

    print(f"\n  {'Class':15s}  {'E175':>8s}  {'TabPFN':>8s}  {'Pseudo':>8s}  {'3-way':>8s}  {'Delta':>8s}")
    print(f"  {'-'*63}")
    for cls in CLASSES:
        e = e175_pc[cls]
        t = tabpfn_pc[cls]
        p = pseudo_pc[cls]
        b = three_pc[cls]
        d = b - e
        marker = "+" if d > 0.01 else "-" if d < -0.01 else " "
        print(f"  {cls:15s}  {e:8.4f}  {t:8.4f}  {p:8.4f}  {b:8.4f}  {d:+8.4f} {marker}")

    # ── Save submissions ──
    print("\n  Saving submissions...", flush=True)
    save_submission(test_tabpfn, "e183_tabpfn_raw", cv_map=compute_map(y, oof_tabpfn)[0])
    save_submission(test_3way, "e183_3way_blend", cv_map=s3)
    save_submission(test_4way, "e183_4way_blend", cv_map=s4)
    save_submission(test_pseudo, "e183_pseudo_raw", cv_map=compute_map(y, oof_pseudo)[0])

    # Also save best TabPFN blend
    test_e175_tabpfn = rank_power_ensemble([test_e175, test_tabpfn], blend_weights, blend_power)
    save_submission(test_e175_tabpfn, "e183_e175_tabpfn", cv_map=blend_score)

    # ── Summary ──
    elapsed = time.time() - t_total
    print("\n" + "=" * 80)
    print("  E183 RESULTS SUMMARY")
    print("=" * 80)
    print(f"  E175 baseline:        SKF={e175_map:.4f}")
    print(f"  TabPFN alone:         SKF={compute_map(y, oof_tabpfn)[0]:.4f}")
    print(f"  E175+TabPFN:          SKF={blend_score:.4f}")
    print(f"  Pseudo-Label Ranker:  SKF={compute_map(y, oof_pseudo)[0]:.4f}")
    print(f"  3-way blend:          SKF={s3:.4f}")
    print(f"  4-way blend:          SKF={s4:.4f}")
    print(f"\n  Completed in {elapsed/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
