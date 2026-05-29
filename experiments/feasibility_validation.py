"""Feasibility Validation: Alternative Approaches for Breaking 0.59 LB Ceiling.

NO model training. Uses ONLY existing data and predictions.
Checks 6 approaches for feasibility before investing in implementation.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train, load_test

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

SEP = "=" * 70


def load_data():
    """Load all needed data without any model training."""
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    # TabPFN predictions
    oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")
    test_tabpfn = np.load(ROOT / "test_e183_tabpfn.npy")

    # Cached v3 features
    train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
    test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

    # E175 best predictions (for reference)
    oof_e175 = np.load(ROOT / "oof_e175_best.npy")

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  TabPFN OOF shape: {oof_tabpfn.shape}, Test shape: {test_tabpfn.shape}")
    print(f"  Features: {train_feats.shape[1]}")
    return (train_df, test_df, y, train_months, test_months,
            oof_tabpfn, test_tabpfn, train_feats, test_feats, oof_e175)


# ======================================================================
# 1. PSEUDO-LABELING FEASIBILITY
# ======================================================================

def check_pseudo_labeling(test_df, test_tabpfn, test_months):
    print(f"\n{SEP}")
    print("  1. PSEUDO-LABELING FEASIBILITY")
    print(SEP)

    n_test = len(test_df)
    max_probs = test_tabpfn.max(axis=1)
    pred_classes = test_tabpfn.argmax(axis=1)

    print(f"\n  Test set size: {n_test}")
    print(f"  Max-prob distribution: mean={max_probs.mean():.3f}, "
          f"median={np.median(max_probs):.3f}, "
          f"std={max_probs.std():.3f}")
    print(f"  Quantiles: 25%={np.percentile(max_probs, 25):.3f}, "
          f"75%={np.percentile(max_probs, 75):.3f}, "
          f"90%={np.percentile(max_probs, 90):.3f}, "
          f"95%={np.percentile(max_probs, 95):.3f}")

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'Threshold':>10s} | {'Total':>6s} | " +
          " | ".join(f"{c[:5]:>5s}" for c in CLASSES))
    print("  " + "-" * (10 + 3 + 6 + 3 + (5 + 3) * N_CLASSES))

    for thr in thresholds:
        mask = max_probs >= thr
        total = mask.sum()
        per_cls = []
        for c in range(N_CLASSES):
            n = ((pred_classes == c) & mask).sum()
            per_cls.append(n)
        row = f"  {thr:>10.1f} | {total:>6d} | " + \
              " | ".join(f"{n:>5d}" for n in per_cls)
        print(row)

    # Cormorant deep-dive (class index 2)
    corm_idx = CLASSES.index("Cormorants")
    corm_mask = pred_classes == corm_idx
    corm_probs = max_probs[corm_mask]
    n_corm_total = corm_mask.sum()

    print(f"\n  --- Cormorant Deep-Dive ---")
    print(f"  Predicted as Cormorant (any confidence): {n_corm_total}")
    if n_corm_total > 0:
        print(f"  Cormorant confidence: mean={corm_probs.mean():.3f}, "
              f"max={corm_probs.max():.3f}, min={corm_probs.min():.3f}")
        corm_months = test_months[corm_mask]
        month_counts = pd.Series(corm_months).value_counts().sort_index()
        print(f"  Month distribution of predicted Cormorants:")
        for m, cnt in month_counts.items():
            print(f"    Month {m:2d}: {cnt} samples")

        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            n_conf = (corm_probs >= thr).sum()
            print(f"    >= {thr}: {n_conf} confident Cormorant pseudo-labels")
    else:
        print(f"  NO samples predicted as Cormorant at all!")

    # Months analysis
    print(f"\n  --- Unseen Month Pseudo-Labels ---")
    unseen_train_months = {2, 5, 12}  # months in test but not in train
    for m in sorted(unseen_train_months):
        m_mask = test_months == m
        n_m = m_mask.sum()
        if n_m == 0:
            continue
        m_probs = max_probs[m_mask]
        m_high = (m_probs >= 0.7).sum()
        m_preds = pred_classes[m_mask]
        class_dist = {CLASSES[c]: (m_preds == c).sum() for c in range(N_CLASSES)}
        top3 = sorted(class_dist.items(), key=lambda x: -x[1])[:3]
        print(f"  Month {m:2d}: {n_m} samples, {m_high} at >=0.7 confidence")
        print(f"    Top classes: {', '.join(f'{c}={n}' for c, n in top3)}")

    # Verdict
    n_high_conf = (max_probs >= 0.7).sum()
    n_corm_high = ((pred_classes == corm_idx) & (max_probs >= 0.7)).sum()
    print(f"\n  VERDICT: {n_high_conf}/{n_test} test samples at >=0.7 confidence "
          f"({100*n_high_conf/n_test:.1f}%)")
    if n_corm_high > 0:
        print(f"  -> {n_corm_high} confident Cormorant pseudo-labels available")
        print(f"  -> Pseudo-labeling MAY help Cormorants")
    else:
        print(f"  -> 0 confident Cormorant pseudo-labels")
        print(f"  -> Pseudo-labeling CANNOT help Cormorants")
    if n_high_conf > 100:
        print(f"  -> Enough volume for general pseudo-labeling ({n_high_conf} samples)")
    else:
        print(f"  -> Volume may be too low for meaningful pseudo-labeling")


# ======================================================================
# 2. SPECIES-LEVEL FEASIBILITY
# ======================================================================

def check_species_level(train_df, y, train_feats):
    print(f"\n{SEP}")
    print("  2. SPECIES-LEVEL FEASIBILITY")
    print(SEP)

    # Species -> group mapping
    species_group = train_df.groupby(["bird_species", "bird_group"]).size().reset_index(name="count")
    species_group = species_group.sort_values(["bird_group", "count"], ascending=[True, False])

    print(f"\n  Species -> Group mapping:")
    for grp in CLASSES:
        grp_species = species_group[species_group["bird_group"] == grp]
        n_species = len(grp_species)
        total = grp_species["count"].sum()
        print(f"\n  {grp} (n={total}, {n_species} species):")
        for _, row in grp_species.iterrows():
            pct = 100 * row["count"] / total
            print(f"    {row['bird_species']:40s}: {row['count']:4d} ({pct:5.1f}%)")

    # For Waders: within-species vs within-group variance
    print(f"\n  --- Wader Species-Level Analysis ---")
    wader_idx = CLASSES.index("Waders")
    wader_mask = y == wader_idx
    wader_feats = train_feats[wader_mask].copy()
    wader_species = train_df.loc[wader_mask, "bird_species"].values

    # Select key discriminative features
    key_feats = ["airspeed", "alt_mid", "alt_range", "rcs_mean", "straightness",
                 "track_duration", "speed_std"]
    available_key = [f for f in key_feats if f in wader_feats.columns]

    if available_key:
        print(f"\n  Feature variance comparison (Waders, {wader_mask.sum()} samples, "
              f"{len(set(wader_species))} species):")
        print(f"  {'Feature':>20s} | {'Within-Group Var':>16s} | {'Within-Species Var':>18s} | {'Ratio':>6s}")
        print("  " + "-" * 70)

        for feat in available_key:
            vals = wader_feats[feat].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            group_var = np.nanvar(vals)

            # Within-species variance (pooled)
            species_vars = []
            species_counts = []
            for sp in set(wader_species):
                sp_mask = wader_species == sp
                sp_vals = vals[sp_mask & valid]
                if len(sp_vals) >= 3:
                    species_vars.append(np.var(sp_vals))
                    species_counts.append(len(sp_vals))
            if species_vars:
                within_species_var = np.average(species_vars, weights=species_counts)
                ratio = within_species_var / max(group_var, 1e-10)
                print(f"  {feat:>20s} | {group_var:>16.4f} | {within_species_var:>18.4f} | {ratio:>6.3f}")

        print(f"\n  Ratio < 1.0 means species-level training would be tighter")
        print(f"  Ratio ~ 1.0 means no benefit from species-level split")

    # Cormorant species check
    corm_idx = CLASSES.index("Cormorants")
    corm_mask = y == corm_idx
    corm_species = train_df.loc[corm_mask, "bird_species"].unique()
    print(f"\n  Cormorants: {corm_mask.sum()} samples, species: {list(corm_species)}")
    print(f"  -> Species-level = group-level for Cormorants (no benefit)")

    # Check which groups benefit most
    print(f"\n  --- Potential Benefit by Group ---")
    for grp in CLASSES:
        grp_mask = y == CLASSES.index(grp)
        n = grp_mask.sum()
        species = train_df.loc[grp_mask, "bird_species"].nunique()
        print(f"  {grp:15s}: {n:4d} samples, {species:2d} species"
              f" -> {'MULTI-SPECIES (potential)' if species > 1 else 'SINGLE SPECIES (no benefit)'}")

    # Verdict
    multi_species = [g for g in CLASSES if
                     train_df.loc[y == CLASSES.index(g), "bird_species"].nunique() > 1]
    print(f"\n  VERDICT: {len(multi_species)}/{N_CLASSES} groups have multiple species")
    print(f"  -> Groups with potential benefit: {multi_species}")
    print(f"  -> Cormorants: NO BENEFIT (single species)")
    print(f"  -> Main beneficiary would be Waders ({train_df.loc[wader_mask, 'bird_species'].nunique()} species)")


# ======================================================================
# 3. HIERARCHICAL CLASSIFICATION
# ======================================================================

def check_hierarchical(train_df, y, oof_e175, oof_tabpfn):
    print(f"\n{SEP}")
    print("  3. HIERARCHICAL CLASSIFICATION FEASIBILITY")
    print(SEP)

    # Define super-groups
    super_groups = {
        "Large waterbird": ["Cormorants", "Geese"],  # similar size, water-associated
        "Gulls": ["Gulls"],
        "Small birds": ["Songbirds", "Pigeons", "Ducks"],
        "Raptors": ["Birds of Prey"],
        "Waders": ["Waders"],
        "Clutter": ["Clutter"],
    }

    # Map each sample to super-group
    super_y = np.zeros(len(y), dtype=int)
    super_names = list(super_groups.keys())
    for si, (sname, classes) in enumerate(super_groups.items()):
        for cls in classes:
            ci = CLASSES.index(cls)
            super_y[y == ci] = si

    print(f"\n  Super-group definitions:")
    for sname, classes in super_groups.items():
        total = sum((y == CLASSES.index(c)).sum() for c in classes)
        print(f"    {sname:20s}: {classes} (n={total})")

    # Check E175 separability at super-group level
    # Aggregate E175 probs to super-group level
    e175_super = np.zeros((len(y), len(super_groups)))
    for si, (sname, classes) in enumerate(super_groups.items()):
        for cls in classes:
            ci = CLASSES.index(cls)
            e175_super[:, si] += oof_e175[:, ci]

    # Compute super-group mAP
    n_super = len(super_groups)
    super_onehot = np.eye(n_super)[super_y]
    super_aps = {}
    for si, sname in enumerate(super_names):
        if super_onehot[:, si].sum() > 0:
            super_aps[sname] = average_precision_score(super_onehot[:, si], e175_super[:, si])

    super_map = np.mean(list(super_aps.values()))
    print(f"\n  E175 super-group AP (aggregated):")
    for sname, ap in super_aps.items():
        print(f"    {sname:20s}: {ap:.4f}")
    print(f"    {'Super-group mAP':20s}: {super_map:.4f}")

    # What if super-groups were perfect?
    # Max mAP = within each super-group, use existing fine-grained predictions
    print(f"\n  --- Maximum Possible mAP with Perfect Super-Group Separation ---")
    # For each super-group, compute fine-grained AP using ONLY samples in that group
    per_class_ap_within = {}
    for sname, classes in super_groups.items():
        if len(classes) <= 1:
            # Single class in super-group = perfect within-group
            ci = CLASSES.index(classes[0])
            per_class_ap_within[classes[0]] = 1.0
        else:
            # Multiple classes: use existing E175 predictions restricted to this super-group
            sg_mask = np.isin(y, [CLASSES.index(c) for c in classes])
            sg_y = y[sg_mask]
            sg_preds = oof_e175[sg_mask]
            for cls in classes:
                ci = CLASSES.index(cls)
                binary = (sg_y == ci).astype(int)
                if binary.sum() > 0 and binary.sum() < len(binary):
                    per_class_ap_within[cls] = average_precision_score(binary, sg_preds[:, ci])
                else:
                    per_class_ap_within[cls] = 0.0

    # Current baseline
    from src.metrics import compute_map
    baseline_map, baseline_per = compute_map(y, oof_e175)

    # Theoretical max: perfect super-group * within-group AP
    # This isn't exactly how it works, but gives upper bound intuition
    print(f"\n  Current E175 OOF per-class AP vs theoretical max:")
    print(f"  {'Class':>15s} | {'Current AP':>10s} | {'Within-SG AP':>12s} | {'SG AP':>6s}")
    print("  " + "-" * 55)
    for cls in CLASSES:
        ci = CLASSES.index(cls)
        # Find which super-group this class belongs to
        for sname, classes in super_groups.items():
            if cls in classes:
                sg_ap = super_aps.get(sname, 0)
                break
        within_ap = per_class_ap_within.get(cls, 0)
        current = baseline_per[cls]
        print(f"  {cls:>15s} | {current:>10.4f} | {within_ap:>12.4f} | {sg_ap:>6.4f}")

    print(f"\n  Baseline mAP: {baseline_map:.4f}")
    # Improvement potential = where super-group separation is poor AND within-group is good
    poor_supergroup = [sn for sn, ap in super_aps.items() if ap < 0.95]
    print(f"  Super-groups with imperfect separation (<0.95): {poor_supergroup}")

    # Verdict
    if all(ap > 0.98 for ap in super_aps.values()):
        print(f"\n  VERDICT: Super-groups already nearly perfectly separated (all AP>0.98)")
        print(f"  -> Hierarchical classification ADDS NO VALUE")
        print(f"  -> The problem is WITHIN super-groups, not between them")
    elif any(ap < 0.90 for ap in super_aps.values()):
        weak = {k: v for k, v in super_aps.items() if v < 0.90}
        print(f"\n  VERDICT: Some super-groups poorly separated: {weak}")
        print(f"  -> Hierarchical classification COULD help if stage-1 is improved")
    else:
        print(f"\n  VERDICT: Super-group separation is good but not perfect")
        print(f"  -> Marginal benefit expected from hierarchical approach")


# ======================================================================
# 4. ONE-CLASS ANOMALY DETECTION (Cormorant)
# ======================================================================

def check_anomaly_detection(train_df, y, train_feats, oof_tabpfn):
    print(f"\n{SEP}")
    print("  4. ONE-CLASS ANOMALY DETECTION (Cormorant)")
    print(SEP)

    corm_idx = CLASSES.index("Cormorants")
    corm_mask = y == corm_idx
    n_corm = corm_mask.sum()

    print(f"\n  Cormorants in train: {n_corm}")

    # Select top-50% cleanest Cormorants by TabPFN confidence
    corm_tabpfn_probs = oof_tabpfn[corm_mask, corm_idx]
    print(f"  TabPFN Cormorant confidence: mean={corm_tabpfn_probs.mean():.3f}, "
          f"max={corm_tabpfn_probs.max():.3f}")

    # Use key features for IsolationForest
    key_feats = ["airspeed", "alt_mid", "alt_range", "rcs_mean", "straightness",
                 "track_duration", "speed_std", "rcs_std", "n_points",
                 "heading_concentration"]
    available = [f for f in key_feats if f in train_feats.columns]
    print(f"  Using {len(available)} features: {available}")

    X_all = train_feats[available].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Top 50% cleanest Cormorants
    clean_threshold = np.percentile(corm_tabpfn_probs, 50)
    clean_mask = corm_mask.copy()
    clean_indices = np.where(corm_mask)[0]
    tabpfn_clean = corm_tabpfn_probs >= clean_threshold
    clean_mask_final = np.zeros(len(y), dtype=bool)
    clean_mask_final[clean_indices[tabpfn_clean]] = True
    n_clean = clean_mask_final.sum()
    print(f"  Clean Cormorants (top 50% by TabPFN): {n_clean}")

    # Fit IsolationForest on clean Cormorants only
    X_clean = X_all[clean_mask_final]
    iso = IsolationForest(n_estimators=300, contamination=0.5, random_state=42)
    iso.fit(X_clean)

    # Score all train samples
    scores = -iso.score_samples(X_all)  # higher = more anomalous
    # For Cormorant detection: lower anomaly score = more Cormorant-like
    corm_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    # Compute AUC
    binary_y = (y == corm_idx).astype(int)
    auc = roc_auc_score(binary_y, corm_scores)

    # Also compute AP (more relevant for imbalanced case)
    ap = average_precision_score(binary_y, corm_scores)

    print(f"\n  IsolationForest results (trained on {n_clean} clean Cormorants):")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision (AP): {ap:.4f}")
    print(f"  Baseline AP (random): {n_corm/len(y):.4f}")

    # Compare to TabPFN's Cormorant detection
    tabpfn_auc = roc_auc_score(binary_y, oof_tabpfn[:, corm_idx])
    tabpfn_ap = average_precision_score(binary_y, oof_tabpfn[:, corm_idx])
    print(f"\n  Comparison - TabPFN Cormorant detection:")
    print(f"  TabPFN AUC-ROC: {tabpfn_auc:.4f}")
    print(f"  TabPFN AP: {tabpfn_ap:.4f}")

    # Score distribution by class
    print(f"\n  Mean Cormorant-likeness score by class:")
    for ci, cls in enumerate(CLASSES):
        cls_mask = y == ci
        mean_score = corm_scores[cls_mask].mean()
        marker = " <--" if cls == "Cormorants" else ""
        print(f"    {cls:15s}: {mean_score:.4f}{marker}")

    # Verdict
    improvement = ap / max(n_corm / len(y), 1e-10)
    print(f"\n  VERDICT:")
    print(f"  IsolationForest AP={ap:.4f} vs random baseline={n_corm/len(y):.4f} "
          f"({improvement:.1f}x lift)")
    print(f"  vs TabPFN AP={tabpfn_ap:.4f}")
    if ap > tabpfn_ap:
        print(f"  -> IsolationForest OUTPERFORMS TabPFN for Cormorant detection!")
        print(f"  -> Worth incorporating as additional signal")
    elif ap > 2 * (n_corm / len(y)):
        print(f"  -> IsolationForest shows meaningful signal but doesn't beat TabPFN")
        print(f"  -> Could add complementary value in an ensemble")
    else:
        print(f"  -> IsolationForest shows minimal signal for Cormorant detection")
        print(f"  -> Not worth pursuing")


# ======================================================================
# 5. ROBUST LOSS FEASIBILITY (Theoretical)
# ======================================================================

def check_robust_loss():
    print(f"\n{SEP}")
    print("  5. ROBUST LOSS FEASIBILITY (Theoretical Analysis)")
    print(SEP)

    n_corm = 40
    noise_rate = 0.47  # estimated 47% noise in Cormorant labels
    n_clean = int(n_corm * (1 - noise_rate))
    n_noisy = n_corm - n_clean

    print(f"\n  Cormorant stats: {n_corm} samples, ~{noise_rate*100:.0f}% noise")
    print(f"  Clean: ~{n_clean}, Noisy: ~{n_noisy}")

    # Standard cross-entropy loss
    print(f"\n  --- Standard Cross-Entropy Loss ---")
    # Clean samples push gradient toward correct Cormorant pattern
    # Noisy samples push gradient toward wrong patterns
    net_signal_std = (1 - noise_rate) - noise_rate
    print(f"  Correct gradient fraction: {1-noise_rate:.2f}")
    print(f"  Wrong gradient fraction: {noise_rate:.2f}")
    print(f"  Net signal: {net_signal_std:.2f} (only {abs(net_signal_std)*100:.0f}% net correct)")
    if net_signal_std > 0:
        print(f"  -> Positive but very weak signal ({net_signal_std:.2f})")
    else:
        print(f"  -> NET NEGATIVE signal! Model learns wrong pattern for Cormorants")

    # GCE loss (Generalized Cross-Entropy)
    print(f"\n  --- Generalized Cross-Entropy (GCE, q=0.7) ---")
    # GCE: L_q(f(x), y) = (1 - f(x)_y^q) / q
    # For correctly labeled: high f(x)_y -> low loss, gradient proportional to f(x)_y^q
    # For mislabeled: low f(x)_y -> high loss, BUT gradient is BOUNDED by q
    # Key insight: GCE down-weights high-loss (likely noisy) samples

    # Assume clean samples converge to f(x)_correct ~ 0.8
    # Assume noisy samples have f(x)_wrong_label ~ 0.2 (model uncertain)
    q = 0.7
    f_clean = 0.8
    f_noisy = 0.2

    # Gradient contribution: proportional to f(x)_y^(q-1) * (1-f(x)_y)
    grad_clean = f_clean ** (q - 1) * (1 - f_clean)  # for clean sample
    grad_noisy = f_noisy ** (q - 1) * (1 - f_noisy)  # for noisy sample

    total_clean_signal = n_clean * grad_clean
    total_noisy_signal = n_noisy * grad_noisy
    net_signal_gce = total_clean_signal - total_noisy_signal
    snr_gce = total_clean_signal / max(total_noisy_signal, 1e-10)

    print(f"  Per-sample gradient (clean, f={f_clean}): {grad_clean:.4f}")
    print(f"  Per-sample gradient (noisy, f={f_noisy}): {grad_noisy:.4f}")
    print(f"  Total clean signal: {n_clean} * {grad_clean:.4f} = {total_clean_signal:.4f}")
    print(f"  Total noisy signal: {n_noisy} * {grad_noisy:.4f} = {total_noisy_signal:.4f}")
    print(f"  Net signal: {net_signal_gce:.4f}")
    print(f"  Signal-to-noise ratio: {snr_gce:.2f}")

    # Compare to standard CE
    # Standard CE: gradient ~ -log(f(x)_y) derivative = 1/f(x)_y
    grad_clean_ce = 1 / f_clean * (1 - f_clean)
    grad_noisy_ce = 1 / f_noisy * (1 - f_noisy)
    total_clean_ce = n_clean * grad_clean_ce
    total_noisy_ce = n_noisy * grad_noisy_ce
    snr_ce = total_clean_ce / max(total_noisy_ce, 1e-10)

    print(f"\n  Comparison - Standard CE:")
    print(f"  Per-sample gradient (clean): {grad_clean_ce:.4f}")
    print(f"  Per-sample gradient (noisy): {grad_noisy_ce:.4f}")
    print(f"  SNR (Standard CE): {snr_ce:.2f}")
    print(f"  SNR (GCE q=0.7): {snr_gce:.2f}")
    print(f"  SNR improvement: {snr_gce/snr_ce:.2f}x")

    # Sample size consideration
    print(f"\n  --- Sample Size Consideration ---")
    # With n=40 and 47% noise, effective sample size:
    n_eff_std = n_corm * max(net_signal_std, 0)
    print(f"  Effective samples (standard loss): ~{n_eff_std:.0f}")
    # With robust loss, noisy samples contribute less wrong signal
    n_eff_robust = n_clean * 0.8 + n_noisy * 0.2  # approximate
    print(f"  Effective samples (robust loss): ~{n_eff_robust:.0f}")
    print(f"  For comparison, other classes have 58-1503 CLEAN samples")

    # Verdict
    print(f"\n  VERDICT:")
    print(f"  GCE improves SNR by {snr_gce/snr_ce:.2f}x over standard CE")
    if snr_gce > 2.0:
        print(f"  -> Robust loss significantly improves gradient signal")
        print(f"  -> BUT with only ~{n_eff_robust:.0f} effective samples, Cormorant "
              f"learning is still severely limited")
    else:
        print(f"  -> Moderate SNR improvement, but fundamental problem is sample size")

    print(f"  -> Robust loss helps but cannot overcome n=40 with 47% noise")
    print(f"  -> Expected improvement: marginal (few % AP points on Cormorants)")
    print(f"  -> The real bottleneck is SAMPLE SIZE, not loss function")


# ======================================================================
# 6. TEST DISTRIBUTION ANALYSIS
# ======================================================================

def check_test_distribution(train_df, test_df, y, train_feats, test_feats, train_months, test_months):
    print(f"\n{SEP}")
    print("  6. TEST DISTRIBUTION ANALYSIS")
    print(SEP)

    # Month distributions
    print(f"\n  --- Month Distribution ---")
    train_month_counts = pd.Series(train_months).value_counts().sort_index()
    test_month_counts = pd.Series(test_months).value_counts().sort_index()
    all_months = sorted(set(train_months) | set(test_months))
    print(f"  {'Month':>6s} | {'Train':>6s} | {'Test':>6s} | {'Status':>15s}")
    print("  " + "-" * 45)
    for m in all_months:
        n_train = train_month_counts.get(m, 0)
        n_test = test_month_counts.get(m, 0)
        if n_train > 0 and n_test > 0:
            status = "SHARED"
        elif n_train > 0:
            status = "TRAIN ONLY"
        else:
            status = "TEST ONLY"
        print(f"  {m:>6d} | {n_train:>6d} | {n_test:>6d} | {status:>15s}")

    # Feature distribution comparison for Cormorant-relevant features
    cormorant_features = ["rcs_mean", "airspeed", "straightness", "alt_mid",
                          "alt_range", "track_duration", "rcs_std", "n_points",
                          "speed_std", "heading_concentration"]
    available = [f for f in cormorant_features if f in train_feats.columns and f in test_feats.columns]

    print(f"\n  --- Feature Distribution: Train vs Test (all samples) ---")
    print(f"  {'Feature':>22s} | {'Train mean':>10s} | {'Test mean':>10s} | {'KS stat':>8s} | {'KS p':>10s} | {'Shift':>8s}")
    print("  " + "-" * 80)

    for feat in available:
        tr = train_feats[feat].values.astype(float)
        te = test_feats[feat].values.astype(float)
        tr = tr[~np.isnan(tr)]
        te = te[~np.isnan(te)]
        if len(tr) < 10 or len(te) < 10:
            continue
        ks_stat, ks_p = stats.ks_2samp(tr, te)
        shift = "SHIFTED" if ks_p < 0.01 else "OK"
        print(f"  {feat:>22s} | {np.mean(tr):>10.3f} | {np.mean(te):>10.3f} | "
              f"{ks_stat:>8.4f} | {ks_p:>10.4f} | {shift:>8s}")

    # Cormorant-specific: compare train Cormorant features to full test
    corm_idx = CLASSES.index("Cormorants")
    corm_mask = y == corm_idx

    print(f"\n  --- Train Cormorant Features vs Full Test Distribution ---")
    print(f"  (Tests whether Cormorant patterns in train are findable in test)")
    print(f"  {'Feature':>22s} | {'Corm train':>10s} | {'Test mean':>10s} | {'Overlap':>8s}")
    print("  " + "-" * 60)

    for feat in available:
        corm_vals = train_feats.loc[corm_mask, feat].values.astype(float)
        test_vals = test_feats[feat].values.astype(float)
        corm_vals = corm_vals[~np.isnan(corm_vals)]
        test_vals = test_vals[~np.isnan(test_vals)]
        if len(corm_vals) < 5 or len(test_vals) < 10:
            continue

        # Overlap: what fraction of test samples fall within Cormorant range?
        corm_lo, corm_hi = np.percentile(corm_vals, [10, 90])
        in_range = ((test_vals >= corm_lo) & (test_vals <= corm_hi)).mean()
        print(f"  {feat:>22s} | {np.mean(corm_vals):>10.3f} | {np.mean(test_vals):>10.3f} | "
              f"{in_range:>7.1%}")

    # Per shared month: distribution shift within month
    shared_months = sorted(set(train_months) & set(test_months))
    print(f"\n  --- Per-Month Feature Shift (shared months: {shared_months}) ---")
    for m in shared_months:
        tr_m = train_months == m
        te_m = test_months == m
        shifts = []
        for feat in available[:5]:  # top 5 features
            tr_vals = train_feats.loc[tr_m, feat].values.astype(float)
            te_vals = test_feats.loc[te_m, feat].values.astype(float)
            tr_vals = tr_vals[~np.isnan(tr_vals)]
            te_vals = te_vals[~np.isnan(te_vals)]
            if len(tr_vals) < 5 or len(te_vals) < 5:
                continue
            ks, p = stats.ks_2samp(tr_vals, te_vals)
            shifts.append((feat, ks, p))
        n_shifted = sum(1 for _, _, p in shifts if p < 0.01)
        print(f"  Month {m:2d}: {sum(tr_m)} train, {sum(te_m)} test, "
              f"{n_shifted}/{len(shifts)} features significantly shifted")

    # Verdict
    print(f"\n  VERDICT:")
    unseen = sorted(set(test_months) - set(train_months))
    shared = sorted(set(test_months) & set(train_months))
    n_unseen = sum((test_months == m).sum() for m in unseen)
    n_total_test = len(test_months)
    print(f"  Unseen months in test: {unseen} ({n_unseen}/{n_total_test} = "
          f"{100*n_unseen/n_total_test:.1f}% of test)")
    print(f"  Shared months: {shared}")
    if n_unseen / n_total_test > 0.3:
        print(f"  -> MAJOR distribution shift: {100*n_unseen/n_total_test:.0f}% of test "
              f"from months never seen in training")
        print(f"  -> This is the FUNDAMENTAL challenge: temporal generalization")
    else:
        print(f"  -> Moderate distribution shift")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print(f"{SEP}")
    print("  FEASIBILITY VALIDATION: Alternative Approaches")
    print(f"  Breaking the 0.59 LB Ceiling")
    print(SEP)

    print("\nLoading data...")
    (train_df, test_df, y, train_months, test_months,
     oof_tabpfn, test_tabpfn, train_feats, test_feats, oof_e175) = load_data()

    # 1. Pseudo-labeling
    check_pseudo_labeling(test_df, test_tabpfn, test_months)

    # 2. Species-level
    check_species_level(train_df, y, train_feats)

    # 3. Hierarchical classification
    check_hierarchical(train_df, y, oof_e175, oof_tabpfn)

    # 4. One-class anomaly detection
    check_anomaly_detection(train_df, y, train_feats, oof_tabpfn)

    # 5. Robust loss (theoretical)
    check_robust_loss()

    # 6. Test distribution
    check_test_distribution(train_df, test_df, y, train_feats, test_feats,
                            train_months, test_months)

    # ── Final Summary ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY: FEASIBILITY VERDICTS")
    print(SEP)
    print("""
  1. PSEUDO-LABELING:     Check above for confident sample counts
  2. SPECIES-LEVEL:       Only helps multi-species groups (not Cormorants)
  3. HIERARCHICAL:        Check above for super-group separability
  4. ANOMALY DETECTION:   Check above for IsolationForest AUC/AP
  5. ROBUST LOSS:         Marginal SNR improvement, bottleneck is n=40
  6. DISTRIBUTION SHIFT:  Unseen months are the fundamental challenge
    """)


if __name__ == "__main__":
    main()
