"""E08: MiniRocket on Raw Trajectory Time Series

Completely different model paradigm: random convolutional kernel transform
(Dempster et al. 2021) on 8-channel trajectory -> 9,996 features ->
LogisticRegressionCV for calibrated probabilities.

No learned kernels -> minimal overfitting. Extremely fast (~30s for 5-fold CV).
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from aeon.transformations.collection.convolution_based import MiniRocket
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results
from src.submission import save_submission

SEQ_LEN = 128  # longer than E06's 64 -- MiniRocket benefits from more timesteps
N_CLASSES = len(CLASSES)
N_FOLDS = 5

# ── Main ──────────────────────────────────────────────────────────
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

# aeon expects (n_samples, n_channels, n_timepoints) -- already our format
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), N_CLASSES))
test_preds = np.zeros((len(X_test_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
    print(f"\n--- Fold {fold} ---", flush=True)

    X_tr = X_all[tr_idx]
    X_va = X_all[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    # MiniRocket transform (fit on train, transform train+val+test)
    print("  Fitting MiniRocket...", flush=True)
    rocket = MiniRocket(random_state=42)
    rocket.fit(X_tr)

    print("  Transforming train...", flush=True)
    X_tr_feat = rocket.transform(X_tr)
    print(f"  MiniRocket features: {X_tr_feat.shape[1]}", flush=True)

    print("  Transforming val...", flush=True)
    X_va_feat = rocket.transform(X_va)

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_va_feat = scaler.transform(X_va_feat)

    # Logistic Regression for calibrated probabilities
    # Use class_weight='balanced' for imbalanced data
    print("  Training LogisticRegression...", flush=True)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs",
        multi_class="multinomial", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr_feat, y_tr)

    # Validation predictions
    va_proba = clf.predict_proba(X_va_feat)
    oof_preds[va_idx] = va_proba

    fold_map, _ = compute_map(y_va, va_proba)
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

    # Test predictions (transform test with this fold's rocket)
    print("  Transforming test...", flush=True)
    X_test_feat = rocket.transform(X_test_all)
    X_test_feat = scaler.transform(X_test_feat)
    test_preds += clf.predict_proba(X_test_feat) / N_FOLDS

# ── Results ───────────────────────────────────────────────────────
final_map, final_per = compute_map(y, oof_preds)
print_results(final_map, final_per, "E08 MiniRocket + LogReg")

np.save("oof_e08.npy", oof_preds)
np.save("test_e08.npy", test_preds)
print("Saved oof_e08.npy and test_e08.npy for stacking", flush=True)

save_submission(test_preds, "e08_minirocket", cv_map=final_map)
