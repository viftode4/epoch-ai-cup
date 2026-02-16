"""Cormorant detection: semi-supervised using train+test data.

Uses unlabeled test data to improve Cormorant detection via:
1. Manifold analysis (do Cormorants cluster in feature space?)
2. Label Propagation on KNN graph over all 4473 samples
3. Label Spreading (softer version)
4. Self-training with CatBoost

Evaluated on LOMO for train OOF, and checks test predictions.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")  # = 2

# ======================================================================
# Load data and build features
# ======================================================================
print("=" * 60, flush=True)
print("CORMORANT SEMI-SUPERVISED ANALYSIS", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

# Months for LOMO
ts_train = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months_train = ts_train.dt.month.values
unique_months = sorted(np.unique(months_train))

ts_test = pd.to_datetime(test_df["timestamp_start_radar_utc"])
months_test = ts_test.dt.month.values

print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
print(f"  Cormorants in train: {y_bin.sum()}", flush=True)
print(f"  Train months: {unique_months}", flush=True)
print(f"  Test months: {sorted(np.unique(months_test))}", flush=True)

# Build features for both
print("\nBuilding features...", flush=True)
X_train = build_features(train_df)
X_test = build_features(test_df)

# Remove temporal features
drop_cols = [c for c in ALL_TEMPORAL if c in X_train.columns]
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# Align columns
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Clean
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  Features: {len(common_cols)}", flush=True)

# ======================================================================
# Feature selection: top-K by t-stat (Cormorant vs rest)
# ======================================================================
print("\nSelecting top features by t-stat...", flush=True)
corm_mask = y_bin == 1
t_stats = []
for col in common_cols:
    vals_corm = X_train.loc[corm_mask, col].values
    vals_rest = X_train.loc[~corm_mask, col].values
    t, p = ttest_ind(vals_corm, vals_rest, equal_var=False)
    t_stats.append((col, abs(t), p))

t_stats.sort(key=lambda x: -x[1])
TOP_K = 50
top_feats = [x[0] for x in t_stats[:TOP_K]]
print(f"  Top {TOP_K} features selected", flush=True)
print(f"  Top 5: {[f'{x[0]}(t={x[1]:.1f})' for x in t_stats[:5]]}", flush=True)

X_train_sel = X_train[top_feats].values
X_test_sel = X_test[top_feats].values

# Combined matrix
X_all = np.vstack([X_train_sel, X_test_sel])
n_train = len(X_train_sel)
n_test = len(X_test_sel)
print(f"  Combined: {X_all.shape}", flush=True)

# Scale
scaler = StandardScaler()
X_all_sc = scaler.fit_transform(X_all)
X_train_sc = X_all_sc[:n_train]
X_test_sc = X_all_sc[n_train:]

# ======================================================================
# 1. Manifold analysis: nearest neighbor structure
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("1. NEAREST NEIGHBOR ANALYSIS", flush=True)
print("=" * 60, flush=True)

# For each Cormorant: how many of its K nearest neighbors are also Cormorants?
for k in [5, 10, 20, 50]:
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    # Train-only first
    nn.fit(X_train_sc)
    dists, indices = nn.kneighbors(X_train_sc[corm_mask])
    # Exclude self (first neighbor)
    neighbor_labels = y_bin[indices[:, 1:]]
    corm_neighbors = neighbor_labels.sum(axis=1)
    mean_corm = corm_neighbors.mean()
    expected = k * 40 / 2601
    print(f"  k={k:2d} (train only): Cormorants have {mean_corm:.1f}/{k} "
          f"Cormorant neighbors (expected: {expected:.1f})", flush=True)

    # Now with train+test
    nn_all = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn_all.fit(X_all_sc)
    dists_all, indices_all = nn_all.kneighbors(X_all_sc[:n_train][corm_mask])
    # Count neighbors that are labeled Cormorants (only train labels known)
    neighbor_is_train = indices_all[:, 1:] < n_train
    neighbor_is_corm = np.zeros_like(neighbor_is_train)
    for i in range(len(indices_all)):
        for j in range(k):
            idx = indices_all[i, j+1]
            if idx < n_train:
                neighbor_is_corm[i, j] = y_bin[idx]
    train_neighbors = neighbor_is_train.sum(axis=1)
    corm_in_neighbors = neighbor_is_corm.sum(axis=1)
    test_neighbors = k - train_neighbors
    print(f"  k={k:2d} (train+test):  Cormorant neighbors: {corm_in_neighbors.mean():.1f}, "
          f"test neighbors: {test_neighbors.mean():.1f}/{k}", flush=True)

# How many test samples are nearest to Cormorants?
print("\n  Test samples closest to Cormorants:", flush=True)
nn_train = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn_train.fit(X_train_sc)
dists_test, indices_test = nn_train.kneighbors(X_test_sc)
nearest_class = y[indices_test.ravel()]
nearest_is_corm = (nearest_class == CORM_IDX)
print(f"  Test samples with nearest train = Cormorant: {nearest_is_corm.sum()}/{n_test} "
      f"({nearest_is_corm.mean()*100:.1f}%)", flush=True)

# By test month
for m in sorted(np.unique(months_test)):
    mask = months_test == m
    n_corm_nearest = nearest_is_corm[mask].sum()
    print(f"    Month {m:2d}: {n_corm_nearest}/{mask.sum()} "
          f"({n_corm_nearest/mask.sum()*100:.1f}%)", flush=True)

# ======================================================================
# 2. Label Propagation on combined data (binary: Corm vs rest)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("2. LABEL PROPAGATION (Cormorant binary)", flush=True)
print("=" * 60, flush=True)

# Labels: train = 0 (not corm) or 1 (corm), test = -1 (unlabeled)
labels_all = np.concatenate([y_bin, np.full(n_test, -1)])

for kernel in ["knn"]:
    for n_neighbors in [7, 15, 30, 50]:
        lp = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=1000)
        lp.fit(X_all_sc, labels_all)

        # Get propagated probabilities
        proba_all = lp.predict_proba(X_all_sc)  # (4473, 2)
        proba_train = proba_all[:n_train, 1]  # P(Cormorant) for train
        proba_test = proba_all[n_train:, 1]   # P(Cormorant) for test

        # Evaluate on train (not truly OOF but indicative)
        ap_train = average_precision_score(y_bin, proba_train)

        # Check test distribution
        test_corm_pred = (proba_test > 0.5).sum()

        print(f"  KNN(k={n_neighbors:2d}): train AP={ap_train:.4f}, "
              f"test P(corm)>0.5: {test_corm_pred}", flush=True)

# ======================================================================
# 3. Label Spreading (softer, with alpha parameter)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("3. LABEL SPREADING (Cormorant binary)", flush=True)
print("=" * 60, flush=True)

for alpha in [0.2, 0.5, 0.8]:
    for n_neighbors in [15, 30]:
        ls = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=1000)
        ls.fit(X_all_sc, labels_all)

        proba_all = ls.predict_proba(X_all_sc)
        proba_train = proba_all[:n_train, 1]
        proba_test = proba_all[n_train:, 1]

        ap_train = average_precision_score(y_bin, proba_train)
        test_corm_pred = (proba_test > 0.5).sum()

        print(f"  alpha={alpha:.1f}, k={n_neighbors:2d}: train AP={ap_train:.4f}, "
              f"test P(corm)>0.5: {test_corm_pred}", flush=True)

# ======================================================================
# 4. LOMO evaluation of Label Propagation
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("4. LOMO EVALUATION: Label Propagation", flush=True)
print("=" * 60, flush=True)

# True LOMO: for each held-out month, run label propagation with:
#   - train (minus held-out month) labels known
#   - held-out month + test = unlabeled
# Then check held-out month predictions

best_ap = 0
best_config = ""

for n_neighbors in [7, 15, 30, 50]:
    oof_scores = np.full(n_train, np.nan)

    for held_month in unique_months:
        val_mask = months_train == held_month
        train_mask = ~val_mask

        n_corm_val = y_bin[val_mask].sum()
        if n_corm_val == 0:
            continue

        # Build labels: train (not held) = known, held + test = -1
        labels_lomo = np.concatenate([
            np.where(train_mask, y_bin, -1),  # train: known if not held
            np.full(n_test, -1)               # test: unlabeled
        ])

        lp = LabelPropagation(kernel="knn", n_neighbors=n_neighbors, max_iter=1000)
        lp.fit(X_all_sc, labels_lomo)

        proba_all = lp.predict_proba(X_all_sc)
        proba_val = proba_all[:n_train][val_mask, 1]
        oof_scores[val_mask] = proba_val

        ap = average_precision_score(y_bin[val_mask], proba_val)
        print(f"    k={n_neighbors:2d}, M{held_month:2d}: val={val_mask.sum()} "
              f"(corm={n_corm_val}), AP={ap:.4f}", flush=True)

    valid = ~np.isnan(oof_scores)
    if valid.sum() > 0 and y_bin[valid].sum() > 0:
        overall_ap = average_precision_score(y_bin[valid], oof_scores[valid])
    else:
        overall_ap = 0.0
    print(f"  LP(k={n_neighbors}) LOMO AP: {overall_ap:.4f}", flush=True)

    if overall_ap > best_ap:
        best_ap = overall_ap
        best_config = f"LP(k={n_neighbors})"

print(f"\n  Best: {best_config} = {best_ap:.4f}", flush=True)

# ======================================================================
# 5. LOMO: Label Spreading
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("5. LOMO EVALUATION: Label Spreading", flush=True)
print("=" * 60, flush=True)

for alpha in [0.2, 0.5, 0.8]:
    for n_neighbors in [15, 30]:
        oof_scores = np.full(n_train, np.nan)

        for held_month in unique_months:
            val_mask = months_train == held_month
            train_mask = ~val_mask

            n_corm_val = y_bin[val_mask].sum()
            if n_corm_val == 0:
                continue

            labels_lomo = np.concatenate([
                np.where(train_mask, y_bin, -1),
                np.full(n_test, -1)
            ])

            ls = LabelSpreading(kernel="knn", n_neighbors=n_neighbors,
                                alpha=alpha, max_iter=1000)
            ls.fit(X_all_sc, labels_lomo)

            proba_all = ls.predict_proba(X_all_sc)
            proba_val = proba_all[:n_train][val_mask, 1]
            oof_scores[val_mask] = proba_val

        valid = ~np.isnan(oof_scores)
        if valid.sum() > 0 and y_bin[valid].sum() > 0:
            overall_ap = average_precision_score(y_bin[valid], oof_scores[valid])
        else:
            overall_ap = 0.0
        print(f"  LS(alpha={alpha:.1f}, k={n_neighbors:2d}) LOMO AP: {overall_ap:.4f}", flush=True)

        if overall_ap > best_ap:
            best_ap = overall_ap
            best_config = f"LS(alpha={alpha}, k={n_neighbors})"

# ======================================================================
# 6. Self-training: iterative pseudo-labeling with CatBoost
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("6. SELF-TRAINING with CatBoost", flush=True)
print("=" * 60, flush=True)

from catboost import CatBoostClassifier

def self_train_lomo(threshold=0.8, max_rounds=5):
    """Self-training on LOMO: predict test, add confident Cormorants, retrain."""
    oof_scores = np.full(n_train, np.nan)

    for held_month in unique_months:
        val_mask_train = months_train == held_month
        train_mask_train = ~val_mask_train

        n_corm_val = y_bin[val_mask_train].sum()
        if n_corm_val == 0:
            continue

        # Start with labeled train (minus held month)
        X_labeled = X_train_sel[train_mask_train]
        y_labeled = y_bin[train_mask_train]

        for round_i in range(max_rounds):
            # Compute class weights
            n_pos = y_labeled.sum()
            n_neg = len(y_labeled) - n_pos
            if n_pos == 0:
                break
            scale = n_neg / max(n_pos, 1)

            cb = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale,
                random_seed=42,
                verbose=0,
                task_type="GPU",
            )
            cb.fit(X_labeled, y_labeled)

            # Predict on test
            proba_test = cb.predict_proba(X_test_sel)[:, 1]
            confident_corm = proba_test > threshold
            n_new = confident_corm.sum()

            if n_new == 0:
                break

            # Add confident test predictions to training
            X_new = X_test_sel[confident_corm]
            y_new = np.ones(n_new, dtype=int)
            X_labeled = np.vstack([X_labeled, X_new])
            y_labeled = np.concatenate([y_labeled, y_new])

        # Final model: predict on held-out validation
        proba_val = cb.predict_proba(X_train_sel[val_mask_train])[:, 1]
        oof_scores[val_mask_train] = proba_val

        ap = average_precision_score(y_bin[val_mask_train], proba_val)
        print(f"    M{held_month:2d}: added {len(X_labeled)-train_mask_train.sum()} "
              f"pseudo-labels, AP={ap:.4f}", flush=True)

    valid = ~np.isnan(oof_scores)
    if valid.sum() > 0 and y_bin[valid].sum() > 0:
        overall_ap = average_precision_score(y_bin[valid], oof_scores[valid])
    else:
        overall_ap = 0.0
    return overall_ap

for thresh in [0.5, 0.7, 0.9]:
    print(f"\n  threshold={thresh}:", flush=True)
    ap = self_train_lomo(threshold=thresh)
    print(f"  Self-train(thresh={thresh}) LOMO AP: {ap:.4f}", flush=True)

    if ap > best_ap:
        best_ap = ap
        best_config = f"SelfTrain(thresh={thresh})"

# ======================================================================
# 7. Baseline: CatBoost binary on top-50 (no test data)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("7. BASELINE: CatBoost binary top-50 (no test data)", flush=True)
print("=" * 60, flush=True)

oof_baseline = np.full(n_train, np.nan)
for held_month in unique_months:
    val_mask = months_train == held_month
    train_mask = ~val_mask
    n_corm_val = y_bin[val_mask].sum()
    if n_corm_val == 0:
        continue

    n_pos = y_bin[train_mask].sum()
    n_neg = train_mask.sum() - n_pos
    scale = n_neg / max(n_pos, 1)

    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_train_sel[train_mask], y_bin[train_mask])
    proba = cb.predict_proba(X_train_sel[val_mask])[:, 1]
    oof_baseline[val_mask] = proba
    ap = average_precision_score(y_bin[val_mask], proba)
    print(f"  M{held_month:2d}: AP={ap:.4f}", flush=True)

valid = ~np.isnan(oof_baseline)
baseline_ap = average_precision_score(y_bin[valid], oof_baseline[valid])
print(f"  Baseline LOMO AP: {baseline_ap:.4f}", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Tabular top-50 ensemble (prev session): 0.1568", flush=True)
print(f"  CatBoost binary top-50 baseline:        {baseline_ap:.4f}", flush=True)
print(f"  Best semi-supervised:                   {best_ap:.4f} ({best_config})", flush=True)
print(f"  DTW/MiniRocket (prev session):          0.01-0.04", flush=True)
print("\nDone!", flush=True)
