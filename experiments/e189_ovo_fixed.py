"""E189: OvO with proper Hastie-Tibshirani pairwise coupling.

Fix: save individual pairwise predictions, aggregate with iterative
pairwise coupling instead of naive vote accumulation.
"""
import sys
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from tabpfn import TabPFNClassifier

from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)

# ── Load data ──
print("Loading data...", flush=True)
train = load_train()
test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values

feats_train = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
feats_test = pd.read_pickle(ROOT / "data/_cached_test_features_v3.pkl")
X_train = np.nan_to_num(feats_train.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(feats_test.values.astype(np.float32), nan=0, posinf=0, neginf=0)

# Relabeled data (consensus from E185)
cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
agreed_noisy = cache['agreed_noisy'].tolist()
consensus_labels = cache['consensus_labels']
y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]

n_train = len(y)
n_test = len(X_test)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Train: {n_train}, Test: {n_test}, Features: {X_train.shape[1]}", flush=True)


# ── Hastie-Tibshirani pairwise coupling ──
def pairwise_coupling(r_matrix, pair_indices, n_classes=9, max_iter=100, tol=1e-6):
    """Solve for class probabilities from pairwise classifier outputs.

    Given pairwise predictions r_ij = P(class_i | class_i vs class_j),
    find p such that p_i / (p_i + p_j) ≈ r_ij.

    Uses the iterative algorithm from:
    Hastie & Tibshirani (1998) "Classification by Pairwise Coupling"

    Args:
        r_matrix: dict of (i,j) -> float, pairwise prediction for this sample
        pair_indices: list of (i,j) pairs that were trained
        n_classes: number of classes

    Returns:
        p: (n_classes,) probability vector
    """
    p = np.ones(n_classes) / n_classes

    for _ in range(max_iter):
        p_old = p.copy()
        for k in range(n_classes):
            numerator = 0.0
            denominator = 0.0
            for (i, j) in pair_indices:
                if i == k:
                    r_ij = r_matrix[(i, j)]
                    numerator += r_ij
                    denominator += p[j] / (p[i] + p[j]) if (p[i] + p[j]) > 1e-12 else 0.5
                elif j == k:
                    r_ji = 1.0 - r_matrix[(i, j)]
                    numerator += r_ji
                    denominator += p[i] / (p[i] + p[j]) if (p[i] + p[j]) > 1e-12 else 0.5

            if denominator > 1e-12:
                p[k] = numerator / denominator

        # Normalize
        p = np.maximum(p, 1e-12)
        p = p / p.sum()

        if np.abs(p - p_old).max() < tol:
            break

    return p


def pairwise_coupling_batch(r_all, pair_indices, n_samples, n_classes=9):
    """Vectorized pairwise coupling for all samples.

    Args:
        r_all: dict of (i,j) -> (n_samples,) array of pairwise predictions
        pair_indices: list of (i,j) tuples
        n_samples: number of samples
        n_classes: number of classes

    Returns:
        (n_samples, n_classes) probability matrix
    """
    result = np.zeros((n_samples, n_classes))
    for s in range(n_samples):
        if s % 500 == 0:
            print(f"  Coupling: {s}/{n_samples}", flush=True)
        r_sample = {(i, j): r_all[(i, j)][s] for (i, j) in pair_indices}
        result[s] = pairwise_coupling(r_sample, pair_indices, n_classes)
    return result


# ── Train pairwise classifiers ──
print(f"\n{'='*70}")
print("TRAINING 36 PAIRWISE TabPFN CLASSIFIERS")
print(f"{'='*70}", flush=True)

pair_indices = []
oof_pairwise = {}  # (i,j) -> (n_train,) predictions for samples in pair
test_pairwise = {}  # (i,j) -> (n_test,) predictions for ALL test samples

# For OOF: we need per-sample predictions, but only samples in the pair have meaningful OOF
# For coupling: we need ALL pairwise predictions for ALL samples
# So we also predict OOF for samples NOT in the pair (using all 5 folds averaged)

oof_all_pairwise = {}  # (i,j) -> (n_train,) predictions for ALL train samples

t0 = time.time()
pair_count = 0

for i in range(N):
    for j in range(i + 1, N):
        pair_mask = (y_relabeled == i) | (y_relabeled == j)
        if pair_mask.sum() < 10:
            continue

        pair_count += 1
        pair_indices.append((i, j))

        y_pair = (y_relabeled[pair_mask] == i).astype(int)
        X_pair = X_train[pair_mask]
        g_pair = groups[pair_mask]

        oof_pair = np.zeros(pair_mask.sum())
        oof_full = np.zeros(n_train)  # predictions for ALL train samples
        test_pair = np.zeros(n_test)

        for fold, (tr, va) in enumerate(
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
                X_pair, y_pair, g_pair
            )
        ):
            clf = TabPFNClassifier(n_estimators=4, random_state=42)
            clf.fit(X_pair[tr], y_pair[tr])

            # OOF for in-pair validation samples
            probs_va = clf.predict_proba(X_pair[va])
            p1 = probs_va[:, 1] if probs_va.shape[1] > 1 else probs_va[:, 0]
            oof_pair[va] = p1

            # Predict ALL train samples (for coupling on non-pair samples)
            probs_all = clf.predict_proba(X_train)
            p1_all = probs_all[:, 1] if probs_all.shape[1] > 1 else probs_all[:, 0]
            oof_full += p1_all / 5

            # Test
            probs_test = clf.predict_proba(X_test)
            p1_test = probs_test[:, 1] if probs_test.shape[1] > 1 else probs_test[:, 0]
            test_pair += p1_test / 5

        # Store
        # For OOF: use true OOF for in-pair samples, averaged prediction for out-of-pair
        oof_combined = oof_full.copy()
        train_indices = np.where(pair_mask)[0]
        for k, idx in enumerate(train_indices):
            oof_combined[idx] = oof_pair[k]

        oof_all_pairwise[(i, j)] = oof_combined
        test_pairwise[(i, j)] = test_pair

        elapsed = time.time() - t0
        eta = elapsed / pair_count * (36 - pair_count)
        print(f"  Pair {pair_count}/36: {CLASSES[i][:4]} vs {CLASSES[j][:4]} "
              f"(n={pair_mask.sum()}) done. "
              f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s", flush=True)

print(f"\n{pair_count} pairs trained in {time.time()-t0:.0f}s", flush=True)

# ── Pairwise coupling aggregation ──
print(f"\n{'='*70}")
print("PAIRWISE COUPLING AGGREGATION")
print(f"{'='*70}", flush=True)

print("\nCoupling OOF predictions...", flush=True)
oof_coupled = pairwise_coupling_batch(oof_all_pairwise, pair_indices, n_train, N)

print("\nCoupling test predictions...", flush=True)
test_coupled = pairwise_coupling_batch(test_pairwise, pair_indices, n_test, N)

# ── Evaluate ──
print(f"\n{'='*70}")
print("EVALUATION")
print(f"{'='*70}", flush=True)

skf, pc = compute_map(y, oof_coupled)
print(f"\nOvO-coupled SKF: {skf:.4f}")
print(f"Per-class: {' '.join(f'{c[:4]}={pc[c]:.3f}' for c in CLASSES)}")

# Compare with old OvO
oof_old = np.load(ROOT / "oof_e186_ovo.npy")
skf_old, _ = compute_map(y, oof_old)
print(f"\nOld OvO (naive): SKF={skf_old:.4f}")
print(f"New OvO (coupled): SKF={skf:.4f}")

# Check test calibration
months_test = pd.to_datetime(test['timestamp_start_radar_utc']).dt.month.values
print(f"\nTest calibration check (mean probs on unseen months):")
for m in [2, 5, 12]:
    mask = months_test == m
    mean_p = test_coupled[mask].mean(axis=0)
    entropy = -np.sum(test_coupled[mask] * np.log(np.clip(test_coupled[mask], 1e-10, 1)), axis=1).mean()
    top1 = test_coupled[mask].max(axis=1).mean()
    print(f"  M{m:02d}: entropy={entropy:.3f} top1={top1:.3f} "
          f"Gulls={mean_p[CLASSES.index('Gulls')]:.3f} "
          f"probs=[{' '.join(f'{p:.3f}' for p in mean_p)}]")

# Compare with E79
e79_test = np.load(ROOT / "test_e79.npy")
print(f"\nE79 test (reference):")
for m in [2, 5, 12]:
    mask = months_test == m
    mean_p = e79_test[mask].mean(axis=0)
    entropy = -np.sum(e79_test[mask] * np.log(np.clip(e79_test[mask], 1e-10, 1)), axis=1).mean()
    top1 = e79_test[mask].max(axis=1).mean()
    print(f"  M{m:02d}: entropy={entropy:.3f} top1={top1:.3f} "
          f"Gulls={mean_p[CLASSES.index('Gulls')]:.3f}")

# ── Save ──
np.save(ROOT / "oof_e189_ovo_coupled.npy", oof_coupled)
np.save(ROOT / "test_e189_ovo_coupled.npy", test_coupled)
print(f"\nSaved oof_e189_ovo_coupled.npy and test_e189_ovo_coupled.npy")

# ── Generate submissions ──
SUB_CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls',
               'Birds of Prey', 'Waders', 'Songbirds']
track_ids = test['track_id'].values


def save_sub(preds, tag):
    df = pd.DataFrame({'track_id': track_ids})
    for cls in SUB_CLASSES:
        ci = CLASSES.index(cls)
        df[cls] = preds[:, ci]
    fname = ROOT / "submissions" / f"e189_{tag}.csv"
    df.to_csv(fname, index=False)
    df.to_csv(ROOT / "submission.csv", index=False)
    print(f"  Saved {fname}")


# Raw OvO coupled
save_sub(test_coupled, "ovo_coupled_raw")

# Blends with E79
tpfn_test = np.load(ROOT / "test_e185_tabpfn_relabel.npy")
for w79, w_ovo, w_tpfn in [
    (0.70, 0.15, 0.15),
    (0.60, 0.20, 0.20),
    (0.75, 0.10, 0.15),
    (0.80, 0.10, 0.10),
]:
    blend = w79 * e79_test + w_ovo * test_coupled + w_tpfn * tpfn_test
    tag = f"e79_{int(w79*100)}_ovo_{int(w_ovo*100)}_tpfn_{int(w_tpfn*100)}"
    save_sub(blend, tag)

print("\nDone!", flush=True)
