"""E11: Meta-Learner Stacking Ensemble

Combines OOF predictions from diverse base models into a level-1 meta-learner.
Each base model sees a different view of the data:
  - E10: LGB+XGB+CB on core+tabular features (trees)
  - E08: MiniRocket on raw trajectory (random convolutional kernels)
  - E06: 1D-CNN on raw trajectory (learned patterns)
  - E09: SVM on CWT+core features (kernel distance on spectral profile)

Meta-learner: Logistic Regression on stacked OOF probabilities.
Also tries simple weighted average with optimization.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

N_CLASSES = len(CLASSES)
N_FOLDS = 5
ROOT = Path(__file__).resolve().parent.parent

# ── Load base model OOF predictions ──────────────────────────────
print("Loading base model predictions...", flush=True)

models = {}
for name, oof_file, test_file in [
    ("E10_tree", "oof_e10.npy", "test_e10.npy"),
    ("E08_rocket", "oof_e08.npy", "test_e08.npy"),
    ("E06_cnn", "oof_e06.npy", "test_e06.npy"),
    ("E09_svm", "oof_e09.npy", "test_e09.npy"),
]:
    oof_path = ROOT / oof_file
    test_path = ROOT / test_file
    if oof_path.exists() and test_path.exists():
        models[name] = {
            "oof": np.load(oof_path),
            "test": np.load(test_path),
        }
        print(f"  Loaded {name}: oof={models[name]['oof'].shape}, test={models[name]['test'].shape}",
              flush=True)
    else:
        print(f"  MISSING {name}: {oof_file}", flush=True)

if len(models) < 2:
    print("Need at least 2 base models. Exiting.", flush=True)
    sys.exit(1)

# Load labels
from src.data import load_train
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# ── Method 1: Optimized weighted average ─────────────────────────
print(f"\n{'='*60}", flush=True)
print("Method 1: Optimized Weighted Average", flush=True)
print(f"{'='*60}", flush=True)

model_names = list(models.keys())
n_models = len(model_names)

# First show individual model performance
for name in model_names:
    m, per = compute_map(y, models[name]["oof"])
    print(f"  {name}: mAP={m:.4f}", flush=True)

# Grid search over weights (for 2-4 models)
if n_models == 2:
    best_map = 0
    best_w = None
    for w1 in np.arange(0.0, 1.01, 0.02):
        w2 = 1 - w1
        oof = w1 * models[model_names[0]]["oof"] + w2 * models[model_names[1]]["oof"]
        m, _ = compute_map(y, oof)
        if m > best_map:
            best_map = m
            best_w = {model_names[0]: w1, model_names[1]: w2}
elif n_models == 3:
    best_map = 0
    best_w = None
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            oof = (w1 * models[model_names[0]]["oof"] +
                   w2 * models[model_names[1]]["oof"] +
                   w3 * models[model_names[2]]["oof"])
            m, _ = compute_map(y, oof)
            if m > best_map:
                best_map = m
                best_w = {model_names[0]: w1, model_names[1]: w2, model_names[2]: w3}
elif n_models == 4:
    best_map = 0
    best_w = None
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            for w3 in np.arange(0.0, 1.01 - w1 - w2, 0.05):
                w4 = 1 - w1 - w2 - w3
                if w4 < 0:
                    continue
                oof = (w1 * models[model_names[0]]["oof"] +
                       w2 * models[model_names[1]]["oof"] +
                       w3 * models[model_names[2]]["oof"] +
                       w4 * models[model_names[3]]["oof"])
                m, _ = compute_map(y, oof)
                if m > best_map:
                    best_map = m
                    best_w = {model_names[0]: w1, model_names[1]: w2,
                              model_names[2]: w3, model_names[3]: w4}

print(f"\nBest weighted average: mAP={best_map:.4f}", flush=True)
print("Weights:", flush=True)
for name, w in best_w.items():
    print(f"  {name}: {w:.2f}", flush=True)

# Compute best weighted average predictions
oof_wavg = sum(best_w[name] * models[name]["oof"] for name in model_names)
test_wavg = sum(best_w[name] * models[name]["test"] for name in model_names)
wavg_map, wavg_per = compute_map(y, oof_wavg)
print_results(wavg_map, wavg_per, "E11 Weighted Average")

# ── Method 2: Logistic Regression meta-learner ───────────────────
print(f"\n{'='*60}", flush=True)
print("Method 2: Logistic Regression Meta-Learner (nested CV)", flush=True)
print(f"{'='*60}", flush=True)

# Stack all OOF predictions as features
# Shape: (N, n_models * N_CLASSES)
X_meta = np.hstack([models[name]["oof"] for name in model_names])
X_meta_test = np.hstack([models[name]["test"] for name in model_names])
print(f"Meta-features: {X_meta.shape[1]} ({n_models} models x {N_CLASSES} classes)",
      flush=True)

# Nested CV to avoid data leakage (OOF predictions were made fold-by-fold,
# so we can directly use them as meta-features with a new CV split)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_lr = np.zeros((len(y), N_CLASSES))
test_lr = np.zeros((X_meta_test.shape[0], N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y)):
    X_tr, X_va = X_meta[tr_idx], X_meta[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs",
        multi_class="multinomial", class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    oof_lr[va_idx] = clf.predict_proba(X_va)
    test_lr += clf.predict_proba(X_meta_test) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_lr[va_idx])
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

lr_map, lr_per = compute_map(y, oof_lr)
print_results(lr_map, lr_per, "E11 LR Meta-Learner")

# ── Pick best method ─────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Weighted Average: {wavg_map:.4f}", flush=True)
print(f"  LR Meta-Learner: {lr_map:.4f}", flush=True)

if lr_map > wavg_map:
    print(f"\nLR Meta-Learner wins (+{lr_map - wavg_map:.4f})", flush=True)
    final_oof = oof_lr
    final_test = test_lr
    final_map = lr_map
    final_per = lr_per
    method = "lr_meta"
else:
    print(f"\nWeighted Average wins (+{wavg_map - lr_map:.4f})", flush=True)
    final_oof = oof_wavg
    final_test = test_wavg
    final_map = wavg_map
    final_per = wavg_per
    method = "weighted_avg"

print_results(final_map, final_per, f"E11 Best ({method})")

np.save("oof_e11.npy", final_oof)
np.save("test_e11.npy", final_test)
print(f"Saved oof_e11.npy and test_e11.npy", flush=True)

save_submission(final_test, f"e11_stacking_{method}", cv_map=final_map)
