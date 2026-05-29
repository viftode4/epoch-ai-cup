"""E184: Fix Cormorants — Dedicated detectors + ensemble surgery.

Problem: 40 Cormorants, 17.5% accuracy, 40% predicted as Gulls.
Root cause: Tree models optimize global loss -> always pick Gull in overlap zone.

Strategy: Build Cormorant detectors that don't care about class frequency:
  A. KNN-based: distance-to-nearest-Cormorant vs distance-to-nearest-Gull
  B. SVM binary: max-margin separator in top feature space
  C. Tuned LGB binary: min_child_samples=3, extreme class weight, SMOTE
  D. TabPFN binary: foundation model, different inductive bias
  E. Detector ensemble -> replace E175's P(Cormorant) column
  F. Also fix Waders with same approach (TabPFN showed +0.124 gain possible)
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
CORM_IDX = CLASSES.index("Cormorants")  # 2
WADER_IDX = CLASSES.index("Waders")  # 8


def load_all():
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Also load full feature matrix for feature selection
    X_train_full = train_feats.values.astype(np.float32)
    X_test_full = test_feats.values.astype(np.float32)
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_full = np.nan_to_num(X_test_full, nan=0.0, posinf=0.0, neginf=0.0)
    all_feat_names = list(train_feats.columns)

    # E175 predictions
    oof_e175 = np.load(ROOT / "oof_e175_best.npy")
    test_e175 = np.load(ROOT / "test_e175_best.npy")

    # TabPFN if available
    tabpfn_path = ROOT / "oof_e183_tabpfn.npy"
    if tabpfn_path.exists():
        oof_tabpfn = np.load(tabpfn_path)
        test_tabpfn = np.load(ROOT / "test_e183_tabpfn.npy")
    else:
        oof_tabpfn = test_tabpfn = None

    return (train_df, test_df, y, train_months, test_months, groups,
            X_train, X_test, selected,
            X_train_full, X_test_full, all_feat_names,
            oof_e175, test_e175, oof_tabpfn, test_tabpfn)


# ══════════════════════════════════════════════════════════════════════
# Feature selection for Cormorant detection
# ══════════════════════════════════════════════════════════════════════

def select_cormorant_features(X_train, y, feature_names, target_idx=CORM_IDX):
    """Select top features for binary detection of target class."""
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import mannwhitneyu

    y_bin = (y == target_idx).astype(int)
    cls_name = CLASSES[target_idx]

    # Mann-Whitney U for each feature
    scores = []
    for i in range(X_train.shape[1]):
        pos = X_train[y_bin == 1, i]
        neg = X_train[y_bin == 0, i]
        try:
            stat, pval = mannwhitneyu(pos, neg, alternative='two-sided')
            effect = stat / (len(pos) * len(neg))
        except Exception:
            pval, effect = 1.0, 0.5
        scores.append((i, feature_names[i], abs(effect - 0.5), pval))

    scores.sort(key=lambda x: -x[2])

    print(f"\n  Top 20 features for {cls_name} detection:")
    top_indices = []
    for rank, (idx, name, eff, pval) in enumerate(scores[:20]):
        marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {rank+1:2d}. {name:30s} effect={eff:.3f} p={pval:.4f} {marker}")
        top_indices.append(idx)

    return top_indices[:15]  # top 15 features


# ══════════════════════════════════════════════════════════════════════
# Binary detectors
# ══════════════════════════════════════════════════════════════════════

def knn_detector(X_train, y, X_test, groups, target_idx, feat_indices, k=7):
    """KNN-based detector: ratio of k-nearest target vs non-target distances."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold

    X_sel_train = X_train[:, feat_indices]
    X_sel_test = X_test[:, feat_indices]
    y_bin = (y == target_idx).astype(int)

    scaler = StandardScaler()

    oof_probs = np.zeros(len(X_train))
    test_probs = np.zeros(len(X_test))

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sel_train, y_bin, groups)):
        X_tr = scaler.fit_transform(X_sel_train[tr_idx])
        X_va = scaler.transform(X_sel_train[va_idx])
        X_te = scaler.transform(X_sel_test)

        # Use higher k for stability, weight by distance
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='minkowski',
            p=2,
        )
        knn.fit(X_tr, y_bin[tr_idx])

        oof_probs[va_idx] = knn.predict_proba(X_va)[:, 1]
        test_probs += knn.predict_proba(X_te)[:, 1] / N_FOLDS

    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_bin, oof_probs)
    print(f"    KNN (k={k}): binary AP={ap:.4f}")
    return oof_probs, test_probs


def svm_detector(X_train, y, X_test, groups, target_idx, feat_indices):
    """SVM with RBF kernel — max-margin, doesn't care about class volumes."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold

    X_sel_train = X_train[:, feat_indices]
    X_sel_test = X_test[:, feat_indices]
    y_bin = (y == target_idx).astype(int)

    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos}  # balance

    scaler = StandardScaler()
    oof_probs = np.zeros(len(X_train))
    test_probs = np.zeros(len(X_test))

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sel_train, y_bin, groups)):
        X_tr = scaler.fit_transform(X_sel_train[tr_idx])
        X_va = scaler.transform(X_sel_train[va_idx])
        X_te = scaler.transform(X_sel_test)

        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight=class_weight,
            probability=True,
            random_state=42,
        )
        svm.fit(X_tr, y_bin[tr_idx])

        oof_probs[va_idx] = svm.predict_proba(X_va)[:, 1]
        test_probs += svm.predict_proba(X_te)[:, 1] / N_FOLDS

    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_bin, oof_probs)
    print(f"    SVM-RBF: binary AP={ap:.4f}")
    return oof_probs, test_probs


def lgb_tuned_detector(X_train, y, X_test, groups, target_idx, feat_indices, n_seeds=5):
    """LGB binary with extreme tuning for rare class."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold
    from imblearn.over_sampling import SMOTE

    X_sel_train = X_train[:, feat_indices]
    X_sel_test = X_test[:, feat_indices]
    y_bin = (y == target_idx).astype(int)

    oof_all = np.zeros((n_seeds, len(X_train)))
    test_all = np.zeros((n_seeds, len(X_test)))

    for seed in range(n_seeds):
        oof_seed = np.zeros(len(X_train))
        test_seed = np.zeros(len(X_test))

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sel_train, y_bin, groups)):
            X_tr, y_tr = X_sel_train[tr_idx], y_bin[tr_idx]
            X_va = X_sel_train[va_idx]

            # SMOTE: oversample minority to 30% of majority
            n_pos = y_tr.sum()
            n_neg = len(y_tr) - n_pos
            if n_pos >= 5:
                try:
                    target_count = min(int(n_neg * 0.3), n_pos * 5)
                    smote = SMOTE(
                        sampling_strategy={1: max(target_count, n_pos)},
                        k_neighbors=min(3, n_pos - 1),
                        random_state=42 + seed + fold,
                    )
                    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
                except Exception:
                    X_tr_sm, y_tr_sm = X_tr, y_tr
            else:
                X_tr_sm, y_tr_sm = X_tr, y_tr

            clf = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=3,
                is_unbalance=True,
                colsample_bytree=0.7,
                subsample=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                verbosity=-1,
                random_state=42 + seed + fold,
            )
            clf.fit(
                X_tr_sm, y_tr_sm,
                eval_set=[(X_va, y_bin[va_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            oof_seed[va_idx] = clf.predict_proba(X_va)[:, 1]
            test_seed += clf.predict_proba(X_sel_test)[:, 1] / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed

    oof_probs = oof_all.mean(axis=0)
    test_probs = test_all.mean(axis=0)

    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_bin, oof_probs)
    print(f"    LGB-tuned ({n_seeds}seeds): binary AP={ap:.4f}")
    return oof_probs, test_probs


def tabpfn_detector(X_train, y, X_test, groups, target_idx, feat_indices):
    """TabPFN binary detector."""
    from tabpfn import TabPFNClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    X_sel_train = X_train[:, feat_indices]
    X_sel_test = X_test[:, feat_indices]
    y_bin = (y == target_idx).astype(int)

    oof_probs = np.zeros(len(X_train))
    test_probs = np.zeros(len(X_test))

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sel_train, y_bin, groups)):
        clf = TabPFNClassifier(n_estimators=16, random_state=42)
        clf.fit(X_sel_train[tr_idx], y_bin[tr_idx])

        oof_probs[va_idx] = clf.predict_proba(X_sel_train[va_idx])[:, 1]
        test_probs += clf.predict_proba(X_sel_test)[:, 1] / N_FOLDS

    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_bin, oof_probs)
    print(f"    TabPFN-binary: binary AP={ap:.4f}")
    return oof_probs, test_probs


def detector_ensemble(detectors_oof, detectors_test, y, target_idx):
    """Find best blend of binary detectors."""
    from sklearn.metrics import average_precision_score
    y_bin = (y == target_idx).astype(int)

    n = len(detectors_oof)
    best_score = -1
    best_weights = None

    # Try all weight combinations
    weight_values = np.arange(0.0, 1.05, 0.1)

    if n == 4:
        for w0 in np.arange(0.0, 1.05, 0.2):
            for w1 in np.arange(0.0, 1.05 - w0, 0.2):
                for w2 in np.arange(0.0, 1.05 - w0 - w1, 0.2):
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -0.01:
                        continue
                    weights = [w0, w1, w2, w3]
                    blend = sum(w * d for w, d in zip(weights, detectors_oof))
                    ap = average_precision_score(y_bin, blend)
                    if ap > best_score:
                        best_score = ap
                        best_weights = [round(w, 2) for w in weights]

    blend_oof = sum(w * d for w, d in zip(best_weights, detectors_oof))
    blend_test = sum(w * d for w, d in zip(best_weights, detectors_test))

    print(f"    Detector ensemble: AP={best_score:.4f}, weights={best_weights}")
    return blend_oof, blend_test, best_weights


# ══════════════════════════════════════════════════════════════════════
# Ensemble surgery: replace specific class columns
# ══════════════════════════════════════════════════════════════════════

def surgery_replace_class(oof_base, test_base, y, cls_idx,
                          detector_oof, detector_test, alpha_values):
    """Replace P(class) in base predictions with detector output at various blend strengths."""
    results = []

    for alpha in alpha_values:
        oof_new = oof_base.copy()
        test_new = test_base.copy()

        # Blend: (1-alpha)*original + alpha*detector
        # Need to normalize detector to same scale as base predictions for the class
        # Use rank-based blending to handle scale differences
        base_ranks = rankdata(oof_base[:, cls_idx]) / len(oof_base)
        det_ranks = rankdata(detector_oof) / len(detector_oof)
        oof_new[:, cls_idx] = (1 - alpha) * base_ranks + alpha * det_ranks

        base_ranks_test = rankdata(test_base[:, cls_idx]) / len(test_base)
        det_ranks_test = rankdata(detector_test) / len(detector_test)
        test_new[:, cls_idx] = (1 - alpha) * base_ranks_test + alpha * det_ranks_test

        overall_map, per_class = compute_map(y, oof_new)
        target_ap = per_class[CLASSES[cls_idx]]
        results.append((alpha, overall_map, target_ap, per_class, oof_new, test_new))

    return results


def lomo_eval(oof, y, train_months, label=""):
    lomo_maps = {}
    for held in sorted(set(train_months)):
        mask = train_months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
    lomo_avg = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    if label:
        print(f"  {label:35s}: LOMO={lomo_avg:.4f}  ({month_str})")
    return lomo_avg, lomo_maps


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 80)
    print("  E184: Fix Cormorants (+ Waders)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── Load ──
    print("\n[1/5] Loading data...", flush=True)
    (train_df, test_df, y, train_months, test_months, groups,
     X_train, X_test, selected,
     X_train_full, X_test_full, all_feat_names,
     oof_e175, test_e175, oof_tabpfn, test_tabpfn) = load_all()

    # Baseline
    e175_map, e175_pc = compute_map(y, oof_e175)
    print(f"\n  E175 baseline: mAP={e175_map:.4f}")
    print(f"  Cormorants: {e175_pc['Cormorants']:.4f}")
    print(f"  Waders:     {e175_pc['Waders']:.4f}")
    lomo_eval(oof_e175, y, train_months, "E175 Baseline")

    # ── Cormorant detectors ──
    print(f"\n[2/5] Building Cormorant detectors...", flush=True)

    # Feature selection on full feature set
    corm_feat_idx = select_cormorant_features(X_train_full, y, all_feat_names, CORM_IDX)

    print(f"\n  Training 4 detector types...")
    t2 = time.time()

    # A. KNN - try multiple k values
    best_knn_ap = -1
    best_knn = None
    for k in [3, 5, 7, 11, 15]:
        oof_k, test_k = knn_detector(X_train_full, y, X_test_full, groups, CORM_IDX, corm_feat_idx, k=k)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score((y == CORM_IDX).astype(int), oof_k)
        if ap > best_knn_ap:
            best_knn_ap = ap
            best_knn = (oof_k, test_k, k)
    oof_knn, test_knn, best_k = best_knn
    print(f"    -> Best KNN: k={best_k}, AP={best_knn_ap:.4f}")

    # B. SVM
    oof_svm, test_svm = svm_detector(X_train_full, y, X_test_full, groups, CORM_IDX, corm_feat_idx)

    # C. Tuned LGB with SMOTE
    oof_lgb, test_lgb = lgb_tuned_detector(X_train_full, y, X_test_full, groups, CORM_IDX, corm_feat_idx)

    # D. TabPFN binary
    oof_tpfn, test_tpfn = tabpfn_detector(X_train_full, y, X_test_full, groups, CORM_IDX, corm_feat_idx)

    print(f"\n  Cormorant detectors built in {time.time()-t2:.1f}s")

    # E. Detector ensemble
    print(f"\n  Finding best detector ensemble...")
    oof_det, test_det, det_weights = detector_ensemble(
        [oof_knn, oof_svm, oof_lgb, oof_tpfn],
        [test_knn, test_svm, test_lgb, test_tpfn],
        y, CORM_IDX,
    )

    # ── Wader detectors ──
    print(f"\n[3/5] Building Wader detectors...", flush=True)
    wader_feat_idx = select_cormorant_features(X_train_full, y, all_feat_names, WADER_IDX)

    print(f"\n  Training 4 detector types...")
    t3 = time.time()

    best_knn_ap_w = -1
    best_knn_w = None
    for k in [3, 5, 7, 11, 15]:
        oof_k, test_k = knn_detector(X_train_full, y, X_test_full, groups, WADER_IDX, wader_feat_idx, k=k)
        ap = average_precision_score((y == WADER_IDX).astype(int), oof_k)
        if ap > best_knn_ap_w:
            best_knn_ap_w = ap
            best_knn_w = (oof_k, test_k, k)
    oof_knn_w, test_knn_w, best_k_w = best_knn_w
    print(f"    -> Best KNN: k={best_k_w}, AP={best_knn_ap_w:.4f}")

    oof_svm_w, test_svm_w = svm_detector(X_train_full, y, X_test_full, groups, WADER_IDX, wader_feat_idx)
    oof_lgb_w, test_lgb_w = lgb_tuned_detector(X_train_full, y, X_test_full, groups, WADER_IDX, wader_feat_idx)
    oof_tpfn_w, test_tpfn_w = tabpfn_detector(X_train_full, y, X_test_full, groups, WADER_IDX, wader_feat_idx)

    print(f"  Wader detectors built in {time.time()-t3:.1f}s")

    oof_det_w, test_det_w, det_weights_w = detector_ensemble(
        [oof_knn_w, oof_svm_w, oof_lgb_w, oof_tpfn_w],
        [test_knn_w, test_svm_w, test_lgb_w, test_tpfn_w],
        y, WADER_IDX,
    )

    # ── Surgery: replace class columns in E175 ──
    print(f"\n[4/5] Ensemble surgery — replacing weak class columns...", flush=True)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Surgery on Cormorants
    print("\n  --- Cormorant surgery (E175) ---")
    corm_results = surgery_replace_class(
        oof_e175, test_e175, y, CORM_IDX, oof_det, test_det, alphas
    )
    print(f"  {'alpha':>6s}  {'mAP':>7s}  {'Corm AP':>8s}  {'Wader AP':>9s}")
    best_corm_alpha = None
    best_corm_map = -1
    for alpha, mAP, corm_ap, per, _, _ in corm_results:
        wader_ap = per['Waders']
        marker = ""
        if mAP > best_corm_map:
            best_corm_map = mAP
            best_corm_alpha = alpha
            marker = " <-- best"
        print(f"  {alpha:6.2f}  {mAP:7.4f}  {corm_ap:8.4f}  {wader_ap:9.4f}{marker}")

    # Get best Cormorant-fixed predictions
    best_corm_result = [r for r in corm_results if r[0] == best_corm_alpha][0]
    oof_corm_fixed = best_corm_result[4]
    test_corm_fixed = best_corm_result[5]
    print(f"\n  Best Cormorant alpha={best_corm_alpha}: mAP={best_corm_map:.4f}")
    lomo_eval(oof_corm_fixed, y, train_months, f"E175+CormDet@{best_corm_alpha}")

    # Surgery on Waders (on top of Cormorant fix)
    print("\n  --- Wader surgery (on Cormorant-fixed base) ---")
    wader_results = surgery_replace_class(
        oof_corm_fixed, test_corm_fixed, y, WADER_IDX, oof_det_w, test_det_w, alphas
    )
    print(f"  {'alpha':>6s}  {'mAP':>7s}  {'Corm AP':>8s}  {'Wader AP':>9s}")
    best_wader_alpha = None
    best_both_map = -1
    for alpha, mAP, _, per, _, _ in wader_results:
        corm_ap = per['Cormorants']
        wader_ap = per['Waders']
        marker = ""
        if mAP > best_both_map:
            best_both_map = mAP
            best_wader_alpha = alpha
            marker = " <-- best"
        print(f"  {alpha:6.2f}  {mAP:7.4f}  {corm_ap:8.4f}  {wader_ap:9.4f}{marker}")

    best_both_result = [r for r in wader_results if r[0] == best_wader_alpha][0]
    oof_both_fixed = best_both_result[4]
    test_both_fixed = best_both_result[5]
    print(f"\n  Best Wader alpha={best_wader_alpha}: mAP={best_both_map:.4f}")
    lomo_eval(oof_both_fixed, y, train_months, f"E175+Corm+Wader fixed")

    # ── Also try surgery on TabPFN blend if available ──
    if oof_tabpfn is not None:
        print(f"\n  --- Surgery on TabPFN+E175 blend ---")
        # First make TabPFN+E175 blend (70/30 from E183 results)
        from scipy.stats import rankdata as rd
        def rpe(preds_list, weights, power=2.0):
            n = preds_list[0].shape[0]
            nc = preds_list[0].shape[1]
            out = np.zeros((n, nc))
            for c in range(nc):
                for p, w in zip(preds_list, weights):
                    out[:, c] += w * (rd(p[:, c]) / n) ** power
            return out

        oof_blend = rpe([oof_e175, oof_tabpfn], [0.3, 0.7])
        test_blend = rpe([test_e175, test_tabpfn], [0.3, 0.7])

        blend_map, blend_pc = compute_map(y, oof_blend)
        print(f"  TabPFN+E175 blend: mAP={blend_map:.4f}, Corm={blend_pc['Cormorants']:.4f}, Wader={blend_pc['Waders']:.4f}")
        lomo_eval(oof_blend, y, train_months, "TabPFN+E175 blend")

        # Cormorant surgery on blend
        corm_results_b = surgery_replace_class(
            oof_blend, test_blend, y, CORM_IDX, oof_det, test_det, alphas
        )
        best_alpha_b = max(corm_results_b, key=lambda x: x[1])
        print(f"\n  Cormorant on blend: best alpha={best_alpha_b[0]}, mAP={best_alpha_b[1]:.4f}, Corm={best_alpha_b[2]:.4f}")
        oof_blend_cf = best_alpha_b[4]
        test_blend_cf = best_alpha_b[5]

        # Wader surgery on blend+corm
        wader_results_b = surgery_replace_class(
            oof_blend_cf, test_blend_cf, y, WADER_IDX, oof_det_w, test_det_w, alphas
        )
        best_both_b = max(wader_results_b, key=lambda x: x[1])
        print(f"  +Wader on blend: best alpha={best_both_b[0]}, mAP={best_both_b[1]:.4f}")
        print(f"    Corm={best_both_b[3]['Cormorants']:.4f}, Wader={best_both_b[3]['Waders']:.4f}")
        lomo_eval(best_both_b[4], y, train_months, "TabPFN+E175+surgery")

        # Save
        save_submission(test_blend, "e184_tabpfn_blend", cv_map=blend_map)
        save_submission(best_both_b[5], "e184_tabpfn_surgery", cv_map=best_both_b[1])

    # ── Full comparison ──
    print("\n" + "=" * 80)
    print("  PER-CLASS AP COMPARISON")
    print("=" * 80)

    configs = [("E175 baseline", oof_e175)]
    if oof_tabpfn is not None:
        configs.append(("TabPFN+E175", oof_blend))
    configs.extend([
        ("E175+CormDet", oof_corm_fixed),
        ("E175+Corm+Wader", oof_both_fixed),
    ])
    if oof_tabpfn is not None:
        configs.append(("TabPFN+E175+surgery", best_both_b[4]))

    print(f"\n  {'Class':15s}", end="")
    for name, _ in configs:
        print(f"  {name:>18s}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in configs:
        print(f"  {'-'*18}", end="")
    print()

    for cls in CLASSES:
        print(f"  {cls:15s}", end="")
        for name, oof in configs:
            _, pc = compute_map(y, oof)
            print(f"  {pc[cls]:18.4f}", end="")
        print()

    # Overall + LOMO
    print(f"\n  {'METRIC':15s}", end="")
    for name, _ in configs:
        print(f"  {name:>18s}", end="")
    print()

    print(f"  {'SKF mAP':15s}", end="")
    for name, oof in configs:
        m, _ = compute_map(y, oof)
        print(f"  {m:18.4f}", end="")
    print()

    print(f"  {'LOMO':15s}", end="")
    for name, oof in configs:
        l, _ = lomo_eval(oof, y, train_months)
        print(f"  {l:18.4f}", end="")
    print()

    # ── Save best submissions ──
    print(f"\n[5/5] Saving submissions...", flush=True)
    save_submission(test_corm_fixed, "e184_corm_fixed", cv_map=best_corm_map)
    save_submission(test_both_fixed, "e184_both_fixed", cv_map=best_both_map)

    np.save(ROOT / "oof_e184_corm_det.npy", oof_det)
    np.save(ROOT / "test_e184_corm_det.npy", test_det)
    np.save(ROOT / "oof_e184_wader_det.npy", oof_det_w)
    np.save(ROOT / "test_e184_wader_det.npy", test_det_w)

    elapsed = time.time() - t_total
    print(f"\n  Completed in {elapsed/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
