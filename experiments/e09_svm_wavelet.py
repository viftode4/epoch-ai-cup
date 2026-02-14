"""E09: SVM on Zaugg-style CWT + Core Features

Zaugg 2008 used CWT on RCS -> SVM. But CWT alone (0.16 mAP) isn't enough
at our radar's sampling rate. Combining CWT spectral profile with core
trajectory features gives SVM both shape info (via kernel distance) AND
the basic trajectory stats. SVM handles the correlated CWT bands better
than trees (proven in ablation: wavelet hurt LGB by -0.009).

Two sub-models tested:
  A: SVM (RBF, C=10) on CWT + core features
  B: SVM (RBF, C=10) on CWT + core + tabular features
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

N_CLASSES = len(CLASSES)
N_FOLDS = 5

# ── Main ──────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Extract CWT + core + tabular features
# SVM should handle the CWT spectral profile better than trees
print("Extracting features (core + zaugg_cwt + tabular)...", flush=True)
train_feats = build_features(train_df, feature_sets=["core", "zaugg_cwt", "tabular"])
print(f"  Feature count: {train_feats.shape[1]}", flush=True)

print("Extracting features (test)...", flush=True)
test_feats = build_features(test_df, feature_sets=["core", "zaugg_cwt", "tabular"])

X = train_feats.values.astype(np.float64)
X_test = test_feats.values.astype(np.float64)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Try multiple SVM configs
configs = [
    ("RBF_C10", {"kernel": "rbf", "C": 10, "gamma": "scale"}),
    ("RBF_C100", {"kernel": "rbf", "C": 100, "gamma": "scale"}),
]

for config_name, svm_params in configs:
    print(f"\n{'='*50}", flush=True)
    print(f"Config: {config_name} | params={svm_params}", flush=True)
    print(f"{'='*50}", flush=True)

    oof_preds = np.zeros((len(y), N_CLASSES))
    test_preds = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold} ---", flush=True)

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        print(f"  Training SVM ({config_name})...", flush=True)
        base_svm = SVC(
            **svm_params,
            class_weight="balanced", random_state=42,
            decision_function_shape="ovr",
        )
        clf = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
        clf.fit(X_tr, y_tr)

        va_proba = clf.predict_proba(X_va)
        oof_preds[va_idx] = va_proba

        fold_map, _ = compute_map(y_va, va_proba)
        print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

        X_test_scaled = scaler.transform(X_test)
        test_preds += clf.predict_proba(X_test_scaled) / N_FOLDS

    final_map, final_per = compute_map(y, oof_preds)
    print_results(final_map, final_per, f"E09 SVM {config_name}")

    # Save best config's predictions
    if config_name == "RBF_C10":
        best_oof = oof_preds.copy()
        best_test = test_preds.copy()
        best_map = final_map
        best_name = config_name

    if final_map > best_map:
        best_oof = oof_preds.copy()
        best_test = test_preds.copy()
        best_map = final_map
        best_name = config_name

print(f"\n\nBest config: {best_name} with mAP={best_map:.4f}", flush=True)
np.save("oof_e09.npy", best_oof)
np.save("test_e09.npy", best_test)
print("Saved oof_e09.npy and test_e09.npy for stacking", flush=True)

save_submission(best_test, "e09_svm_wavelet", cv_map=best_map)
