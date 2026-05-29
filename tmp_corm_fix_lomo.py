import sys, time, os
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import numpy as np
import pandas as pd
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map
from scipy.special import softmax
import lightgbm as lgb

def main():
    train = load_train()
    test = load_test()
    y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
    groups = train['primary_observation_id'].values
    months = pd.to_datetime(train['timestamp_start_radar_utc']).dt.month.values
    test_months = pd.to_datetime(test['timestamp_start_radar_utc']).dt.month.values
    N_CLASSES = len(CLASSES)
    CORM = 2

    # Load features - use the 100 stability-selected (proven on LB=0.59)
    train_feats = pd.read_pickle('G:/Projects/epoch-ai-cup/data/_cached_train_features_v3.pkl')
    test_feats = pd.read_pickle('G:/Projects/epoch-ai-cup/data/_cached_test_features_v3.pkl')
    selected = [l.strip() for l in open('G:/Projects/epoch-ai-cup/data/best_features_e175.txt').readlines() if l.strip()]
    selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

    X_train = train_feats[selected].values.astype(np.float32)
    X_test = test_feats[selected].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Identify noisy labels using consensus of existing models
    oof_tabpfn = np.load('G:/Projects/epoch-ai-cup/oof_e183_tabpfn.npy')
    oof_e175 = np.load('G:/Projects/epoch-ai-cup/oof_e175_best.npy')
    oof_cb = np.load('G:/Projects/epoch-ai-cup/oof_e175_cb.npy')
    oof_ranker = np.load('G:/Projects/epoch-ai-cup/oof_e175_ranker.npy')
    oof_e175_prob = softmax(oof_e175, axis=1)

    # Consensus prediction for each sample
    from collections import Counter
    def consensus_label(idx):
        preds = [oof_tabpfn[idx].argmax(), oof_e175_prob[idx].argmax(),
                 oof_cb[idx].argmax(), oof_ranker[idx].argmax()]
        vote = Counter(preds).most_common(1)[0]
        return vote[0], vote[1]  # label, agreement count

    # Find agreed noisy labels (cleanlab intersection) - force n_jobs=1 for Windows
    from cleanlab.filter import find_label_issues
    issues_t = set(find_label_issues(labels=y, pred_probs=oof_tabpfn, return_indices_ranked_by='self_confidence', n_jobs=1))
    issues_e = set(find_label_issues(labels=y, pred_probs=oof_e175_prob, return_indices_ranked_by='self_confidence', n_jobs=1))
    agreed_noisy = sorted(issues_t & issues_e)
    print(f"Agreed noisy labels: {len(agreed_noisy)}")

    # Categorize the 40 Cormorants
    corm_idx = np.where(y == CORM)[0]
    corm_real = []      # 4/4 consensus = Cormorant
    corm_relabel = []   # 3/4+ consensus != Cormorant AND P(new) > 0.5
    corm_ambiguous = [] # 2/4 or low confidence

    for idx in corm_idx:
        cons_label, agreement = consensus_label(idx)
        p_new = oof_tabpfn[idx, cons_label]
        if cons_label == CORM:
            corm_real.append(idx)
        elif agreement >= 3 and p_new > 0.3:
            corm_relabel.append((idx, cons_label))
        else:
            corm_ambiguous.append(idx)

    print(f"Cormorants: {len(corm_real)} real, {len(corm_relabel)} relabel, {len(corm_ambiguous)} ambiguous")

    # ======================================================================
    # TRUE LOMO EVALUATION (train on 3 months, predict held-out 4th)
    # ======================================================================

    def true_lomo(X, y_labels, months_arr, label="", params=None):
        """TRUE LOMO: train on 3 months, predict held-out 4th."""
        if params is None:
            params = dict(
                n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=7,
                subsample=0.7, colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1,
                class_weight="balanced", random_state=42, verbose=-1, n_jobs=1,
            )

        unique_months = sorted(np.unique(months_arr))
        oof = np.zeros((len(y_labels), N_CLASSES))

        for month in unique_months:
            va = months_arr == month
            tr = ~va

            # Need at least 2 classes in train
            if len(np.unique(y_labels[tr])) < 2:
                continue

            model = lgb.LGBMClassifier(**params)
            model.fit(X[tr], y_labels[tr])
            oof[va] = model.predict_proba(X[va])

        overall, per_class = compute_map(y_labels, oof)
        month_scores = {}
        for m in unique_months:
            mask = months_arr == m
            if mask.sum() >= 5:
                ms, _ = compute_map(y_labels[mask], oof[mask])
                month_scores[m] = ms
        lomo = np.mean(list(month_scores.values()))

        month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(month_scores.items()))
        print(f"\n  {label}:")
        print(f"    TRUE LOMO={lomo:.4f}  SKF-like={overall:.4f}")
        print(f"    Months: {month_str}")
        for cls in CLASSES:
            ap = per_class[cls]
            marker = " ***" if cls == "Cormorants" else ""
            print(f"    {cls:15s}: {ap:.4f}{marker}")

        return oof, lomo, per_class

    # ======================================================================
    # TEST CONFIGURATIONS
    # ======================================================================

    print("=" * 80)
    print("CORMORANT FIX: TRUE LOMO VALIDATION")
    print("=" * 80)

    # Config 1: BASELINE (original labels, no cleaning)
    print("\n[1] BASELINE (original labels)")
    oof_base, lomo_base, pc_base = true_lomo(X_train, y, months, "Baseline (original labels)")

    # Config 2: RELABEL high-confidence Cormorant mislabels
    y_relabel_corm = y.copy()
    for idx, new_label in corm_relabel:
        y_relabel_corm[idx] = new_label
    n_changed = np.sum(y_relabel_corm != y)
    print(f"\n[2] RELABEL {n_changed} high-confidence Cormorant mislabels")
    for c in range(N_CLASSES):
        orig = (y == c).sum()
        new = (y_relabel_corm == c).sum()
        if orig != new:
            print(f"    {CLASSES[c]}: {orig} -> {new} ({new-orig:+d})")
    oof_rc, lomo_rc, pc_rc = true_lomo(X_train, y_relabel_corm, months, "Relabel Cormorant mislabels")

    # Config 3: RELABEL ALL agreed noisy labels (all classes)
    y_relabel_all = y.copy()
    for idx in agreed_noisy:
        cons, _ = consensus_label(idx)
        y_relabel_all[idx] = cons
    n_changed_all = np.sum(y_relabel_all != y)
    print(f"\n[3] RELABEL ALL {n_changed_all} agreed noisy labels across all classes")
    for c in range(N_CLASSES):
        orig = (y == c).sum()
        new = (y_relabel_all == c).sum()
        if orig != new:
            print(f"    {CLASSES[c]}: {orig} -> {new} ({new-orig:+d})")
    oof_ra, lomo_ra, pc_ra = true_lomo(X_train, y_relabel_all, months, "Relabel ALL noise")

    # Config 4: REMOVE Cormorant mislabels (train on real + ambiguous only)
    remove_idx = set(idx for idx, _ in corm_relabel)
    keep_mask = np.array([i not in remove_idx for i in range(len(y))])
    print(f"\n[4] REMOVE {len(remove_idx)} Cormorant mislabels (keep {keep_mask.sum()} samples)")
    oof_rm = np.zeros((len(y), N_CLASSES))  # full-size for eval
    unique_months = sorted(np.unique(months))
    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=7,
                  subsample=0.7, colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1,
                  class_weight="balanced", random_state=42, verbose=-1, n_jobs=1)
    for month in unique_months:
        va = months == month
        tr = (~va) & keep_mask  # train on non-removed, non-held-out
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train[tr], y[tr])
        oof_rm[va] = model.predict_proba(X_train[va])
    overall_rm, pc_rm = compute_map(y, oof_rm)
    month_scores_rm = {}
    for m in unique_months:
        mask = months == m
        ms, _ = compute_map(y[mask], oof_rm[mask])
        month_scores_rm[m] = ms
    lomo_rm = np.mean(list(month_scores_rm.values()))
    print(f"\n  Remove Cormorant mislabels:")
    print(f"    TRUE LOMO={lomo_rm:.4f}")
    print(f"    Months: {' '.join(f'm{m}={v:.3f}' for m, v in sorted(month_scores_rm.items()))}")
    for cls in CLASSES:
        marker = " ***" if cls == "Cormorants" else ""
        print(f"    {cls:15s}: {pc_rm[cls]:.4f}{marker}")

    # Config 5: REMOVE ALL agreed noisy labels
    remove_all_idx = set(agreed_noisy)
    keep_all_mask = np.array([i not in remove_all_idx for i in range(len(y))])
    print(f"\n[5] REMOVE ALL {len(remove_all_idx)} agreed noisy labels")
    oof_rma = np.zeros((len(y), N_CLASSES))
    for month in unique_months:
        va = months == month
        tr = (~va) & keep_all_mask
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train[tr], y[tr])
        oof_rma[va] = model.predict_proba(X_train[va])
    overall_rma, pc_rma = compute_map(y, oof_rma)
    month_scores_rma = {}
    for m in unique_months:
        mask = months == m
        ms, _ = compute_map(y[mask], oof_rma[mask])
        month_scores_rma[m] = ms
    lomo_rma = np.mean(list(month_scores_rma.values()))
    print(f"\n  Remove ALL agreed noise:")
    print(f"    TRUE LOMO={lomo_rma:.4f}")
    print(f"    Months: {' '.join(f'm{m}={v:.3f}' for m, v in sorted(month_scores_rma.items()))}")
    for cls in CLASSES:
        marker = " ***" if cls == "Cormorants" else ""
        print(f"    {cls:15s}: {pc_rma[cls]:.4f}{marker}")

    # Config 6: SOFT LABELS for Cormorants (use TabPFN probs as training target)
    y_soft_approx = y.copy()
    for idx in corm_ambiguous:
        y_soft_approx[idx] = oof_tabpfn[idx].argmax()
    # Also relabel the high-confidence ones
    for idx, new_label in corm_relabel:
        y_soft_approx[idx] = new_label
    n_changed_soft = np.sum(y_soft_approx != y)
    print(f"\n[6] SOFT-APPROX: Relabel ALL non-real Cormorants ({n_changed_soft} changed)")
    for c in range(N_CLASSES):
        orig = (y == c).sum()
        new = (y_soft_approx == c).sum()
        if orig != new:
            print(f"    {CLASSES[c]}: {orig} -> {new} ({new-orig:+d})")
    oof_soft, lomo_soft, pc_soft = true_lomo(X_train, y_soft_approx, months, "Soft-approx (all suspect Corm relabeled)")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("SUMMARY (TRUE LOMO)")
    print("=" * 80)
    print(f"\n{'Config':45s} {'LOMO':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Gulls':>7s} {'Delta':>7s}")
    print("-" * 100)

    configs = [
        ("1. Baseline (original labels)", lomo_base, pc_base),
        ("2. Relabel Corm mislabels only", lomo_rc, pc_rc),
        ("3. Relabel ALL noise", lomo_ra, pc_ra),
        ("4. Remove Corm mislabels", lomo_rm, pc_rm),
        ("5. Remove ALL noise", lomo_rma, pc_rma),
        ("6. Soft-approx (all suspect Corm)", lomo_soft, pc_soft),
    ]

    for name, lomo_val, pc in configs:
        delta = lomo_val - lomo_base
        print(f"  {name:45s} {lomo_val:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} "
              f"{pc['Birds of Prey']:7.4f} {pc['Gulls']:7.4f} {delta:+7.4f}")

    print("\nDone.")

if __name__ == '__main__':
    main()
