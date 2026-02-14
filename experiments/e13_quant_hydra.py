"""E13: QUANT + Hydra+MultiRocket (T01 + T02)

Two new aeon TSC models on 8ch x 128 trajectory:
  A) QUANT: quantile-interval features over dyadic intervals (Dempster 2024)
  B) Hydra: competing convolutional kernel groups (Dempster 2023)
  C) MultiRocket: multiple pooling operators + transforms (Tan 2022)

Each saves OOF predictions for stacking. Compare standalone to E08 MiniRocket (0.4799).
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 128
N_CLASSES = len(CLASSES)
N_FOLDS = 5

# ── Load data ────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

print("Preparing train sequences (8ch x 128)...", flush=True)
X_all = prepare_sequences(train_df, seq_len=SEQ_LEN)
print(f"  Shape: {X_all.shape}", flush=True)

print("Preparing test sequences...", flush=True)
X_test_all = prepare_sequences(test_df, seq_len=SEQ_LEN)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


def run_transform_model(transform_cls, transform_kwargs, clf_cls, clf_kwargs,
                        name, X_all, X_test_all, y, skf):
    """Run a transform + classifier pipeline with 5-fold CV, save OOF + test preds."""
    oof_preds = np.zeros((len(y), N_CLASSES))
    test_preds = np.zeros((len(X_test_all), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        print(f"\n--- {name} Fold {fold} ---", flush=True)
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Transform
        print(f"  Fitting {name} transform...", flush=True)
        transformer = transform_cls(**transform_kwargs)
        transformer.fit(X_tr)

        X_tr_feat = transformer.transform(X_tr)
        X_va_feat = transformer.transform(X_va)
        print(f"  Features: {X_tr_feat.shape[1]}", flush=True)

        # Scale
        scaler = StandardScaler()
        X_tr_feat = scaler.fit_transform(X_tr_feat)
        X_va_feat = scaler.transform(X_va_feat)

        # Classifier
        print(f"  Training classifier...", flush=True)
        clf = clf_cls(**clf_kwargs)
        clf.fit(X_tr_feat, y_tr)

        # Predictions
        if hasattr(clf, "predict_proba"):
            va_proba = clf.predict_proba(X_va_feat)
        else:
            dec = clf.decision_function(X_va_feat)
            exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
            va_proba = exp_dec / exp_dec.sum(axis=1, keepdims=True)

        oof_preds[va_idx] = va_proba

        fold_map, _ = compute_map(y_va, va_proba)
        print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

        # Test
        X_test_feat = transformer.transform(X_test_all)
        X_test_feat = scaler.transform(X_test_feat)
        if hasattr(clf, "predict_proba"):
            test_preds += clf.predict_proba(X_test_feat) / N_FOLDS
        else:
            dec = clf.decision_function(X_test_feat)
            exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
            test_preds += (exp_dec / exp_dec.sum(axis=1, keepdims=True)) / N_FOLDS

    final_map, final_per = compute_map(y, oof_preds)
    return oof_preds, test_preds, final_map, final_per


def run_classifier_model(clf_cls, clf_kwargs, name, X_all, X_test_all, y, skf):
    """Run an aeon end-to-end classifier with 5-fold CV."""
    oof_preds = np.zeros((len(y), N_CLASSES))
    test_preds = np.zeros((len(X_test_all), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        print(f"\n--- {name} Fold {fold} ---", flush=True)
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        print(f"  Fitting {name}...", flush=True)
        clf = clf_cls(**clf_kwargs)
        clf.fit(X_tr, y_tr)

        va_proba = clf.predict_proba(X_va)
        oof_preds[va_idx] = va_proba

        fold_map, _ = compute_map(y_va, va_proba)
        print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

        test_preds += clf.predict_proba(X_test_all) / N_FOLDS

    final_map, final_per = compute_map(y, oof_preds)
    return oof_preds, test_preds, final_map, final_per


# ── A) QUANT (T01) — load from previous run ──────────────────────
print(f"\n{'='*60}", flush=True)
print("A) QUANT Classifier (T01)", flush=True)
print(f"{'='*60}", flush=True)

quant_oof_path = ROOT / "oof_e13a_quant.npy"
quant_test_path = ROOT / "test_e13a_quant.npy"
if quant_oof_path.exists():
    print("  Loading QUANT from previous run...", flush=True)
    oof_quant = np.load(quant_oof_path)
    test_quant = np.load(quant_test_path)
    map_quant, per_quant = compute_map(y, oof_quant)
    print_results(map_quant, per_quant, "E13a QUANT (cached)")
else:
    from aeon.classification.interval_based import QUANTClassifier
    oof_quant, test_quant, map_quant, per_quant = run_classifier_model(
        QUANTClassifier,
        {"random_state": 42},
        "QUANT",
        X_all, X_test_all, y, skf,
    )
    print_results(map_quant, per_quant, "E13a QUANT")
    np.save(ROOT / "oof_e13a_quant.npy", oof_quant)
    np.save(ROOT / "test_e13a_quant.npy", test_quant)


# ── B) Hydra (T02) ──────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("B) Hydra Classifier (T02)", flush=True)
print(f"{'='*60}", flush=True)

from aeon.transformations.collection.convolution_based import HydraTransformer
from aeon.transformations.collection.convolution_based import MultiRocket

# Hydra transform + LogisticRegression
oof_hydra, test_hydra, map_hydra, per_hydra = run_transform_model(
    HydraTransformer,
    {"random_state": 42},
    LogisticRegression,
    {"C": 1.0, "max_iter": 2000, "solver": "lbfgs",
     "multi_class": "multinomial", "class_weight": "balanced",
     "random_state": 42, "n_jobs": -1},
    "Hydra",
    X_all, X_test_all, y, skf,
)
print_results(map_hydra, per_hydra, "E13b Hydra + LogReg")
np.save(ROOT / "oof_e13b_hydra.npy", oof_hydra)
np.save(ROOT / "test_e13b_hydra.npy", test_hydra)


# ── C) MultiRocket (T02 alternative) ────────────────────────────
print(f"\n{'='*60}", flush=True)
print("C) MultiRocket (T02 alternative)", flush=True)
print(f"{'='*60}", flush=True)

oof_mrocket, test_mrocket, map_mrocket, per_mrocket = run_transform_model(
    MultiRocket,
    {"random_state": 42},
    LogisticRegression,
    {"C": 1.0, "max_iter": 2000, "solver": "lbfgs",
     "multi_class": "multinomial", "class_weight": "balanced",
     "random_state": 42, "n_jobs": -1},
    "MultiRocket",
    X_all, X_test_all, y, skf,
)
print_results(map_mrocket, per_mrocket, "E13c MultiRocket + LogReg")
np.save(ROOT / "oof_e13c_mrocket.npy", oof_mrocket)
np.save(ROOT / "test_e13c_mrocket.npy", test_mrocket)


# ── Comparison ───────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("COMPARISON (vs E08 MiniRocket = 0.4799)", flush=True)
print(f"{'='*60}", flush=True)
print(f"  QUANT:       {map_quant:.4f} ({map_quant - 0.4799:+.4f} vs MiniRocket)", flush=True)
print(f"  Hydra:       {map_hydra:.4f} ({map_hydra - 0.4799:+.4f} vs MiniRocket)", flush=True)
print(f"  MultiRocket: {map_mrocket:.4f} ({map_mrocket - 0.4799:+.4f} vs MiniRocket)", flush=True)

# Find best
results = [
    ("QUANT", map_quant, oof_quant, test_quant, per_quant),
    ("Hydra", map_hydra, oof_hydra, test_hydra, per_hydra),
    ("MultiRocket", map_mrocket, oof_mrocket, test_mrocket, per_mrocket),
]
results.sort(key=lambda x: -x[1])
best_name, best_map, best_oof, best_test, best_per = results[0]
print(f"\nBest: {best_name} ({best_map:.4f})", flush=True)

# ── Quick stacking test: replace E08 with best new model ─────────
print(f"\n{'='*60}", flush=True)
print("Stacking test: replace E08 MiniRocket in E11 stack", flush=True)
print(f"{'='*60}", flush=True)

oof_e10 = np.load(ROOT / "oof_e10.npy")
oof_e06 = np.load(ROOT / "oof_e06.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")

# Original E11 weights: E10=70%, E08=10%, E06=10%, E09=10%
# Replace E08 with each new model
for name, oof_new in [("QUANT", oof_quant), ("Hydra", oof_hydra), ("MultiRocket", oof_mrocket)]:
    # Simple replace at same weight
    oof_stack = 0.70 * oof_e10 + 0.10 * oof_new + 0.10 * oof_e06 + 0.10 * oof_e09
    m, _ = compute_map(y, oof_stack)
    print(f"  Replace E08 with {name:12s}: mAP={m:.4f} (E11 was 0.7396, delta={m - 0.7396:+.4f})",
          flush=True)

# Also test adding as 5th model
print(f"\nStacking with 5 models (add best new as 5th):", flush=True)
for w5 in [0.05, 0.10, 0.15]:
    remaining = 1.0 - w5
    oof_5 = (remaining * (0.70/0.90 * oof_e10 + 0.10/0.90 * oof_e06 + 0.10/0.90 * oof_e09)
             + w5 * best_oof)
    # Keep E08 too
    for w8 in [0.05, 0.10]:
        r2 = 1.0 - w5 - w8
        oof_5m = (r2 * (0.70/0.90 * oof_e10 + 0.10/0.90 * oof_e06 + 0.10/0.90 * oof_e09)
                  + w5 * best_oof + w8 * np.load(ROOT / "oof_e08.npy"))
        m5, _ = compute_map(y, oof_5m)
        print(f"  w_new={w5:.2f}, w_e08={w8:.2f}: mAP={m5:.4f}", flush=True)

print("\nDone!", flush=True)
