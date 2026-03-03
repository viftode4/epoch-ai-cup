"""E110: Pseudo-Label Distillation (Self-Training on Unseen Months).

Mathematical Motivation:
1. We are stuck at 0.59. We have exhausted post-processing (PoE) and base model tweaks.
2. The fundamental issue is Domain Shift: P(x|y) changes between train (Sep/Oct) and test (Feb/May/Dec).
3. We have a highly accurate 0.59 teacher model (E96/E101) that has already corrected many
   of these shift errors using physics priors.
4. We can use this 0.59 teacher to generate soft pseudo-labels for the test set.
5. By training a student model on BOTH train (hard labels) and test (soft pseudo-labels),
   the student learns a feature representation that naturally bridges the domain gap.
   This is standard Semi-Supervised Domain Adaptation.
6. Crucially, we only pseudo-label the UNSEEN months (Feb/May/Dec) where the shift occurs,
   and we use temperature scaling to soften the labels, preventing confirmation bias.

Pipeline:
1. Load E101 (our best ensemble) as the Teacher.
2. Extract soft labels for test set (temperature scaled).
3. Train a new LGBM Student on Train + Test.
4. Predict on Test using the Student.
5. Apply E96 post-processing to the Student's predictions (to ensure we don't lose the physics).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from experiments.e96_nbalt_heading_ac1 import (
    BASE_ALPHA, TAU_PRIOR, TAU_NB, GAMMA,
    build_gbif_priors, apply_gated_ratio_priors, extract_heading_ac1,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe, renorm_rows, top2_margin
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42
UNSEEN_MONTHS = (2, 5, 12)

# E79 Pruned Features
PRUNED_FEATURES = [
    'airspeed', 'radar_bird_size', 'min_z', 'max_z', 'duration', 'track_length',
    'z_mean', 'z_std', 'rcs_mean', 'rcs_std', 'rcs_max', 'rcs_min',
    'speed_mean', 'speed_std', 'speed_max', 'speed_min', 'accel_mean', 'accel_std',
    'accel_max', 'accel_min', 'turn_mean', 'turn_std', 'turn_max', 'turn_min',
    'sinuosity', 'rcs_range', 'rcs_q25', 'rcs_q75', 'rcs_iqr', 'rcs_skew', 'rcs_kurtosis',
    'rcs_trend', 'rcs_p2p', 'rcs_smoothness', 'rcs_ac1', 'rcs_ac2'
]

def main() -> None:
    print("=" * 70, flush=True)
    print("E110 PSEUDO-LABEL DISTILLATION (STUDENT-TEACHER)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y_train = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    # 1. Build Features
    print("\nBuilding features...", flush=True)
    feature_sets = ["core", "tabular", "rcs_fft"]
    Xtr_base = build_features(train_df, feature_sets=feature_sets)
    Xte_base = build_features(test_df, feature_sets=feature_sets)
    
    keep_cols = [c for c in PRUNED_FEATURES if c in Xtr_base.columns]
    Xtr_df = Xtr_base[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte_df = Xte_base[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xtr = Xtr_df.to_numpy(np.float32)
    Xte = Xte_df.to_numpy(np.float32)

    # 2. Load Teacher Predictions (E101 Geo Ensemble)
    # We use E101 because it's our most robust 0.59 submission
    try:
        teacher_df = pd.read_csv(ROOT / "submissions" / "e101_ens_geo4_20260228_2132.csv")
        teacher_preds = teacher_df[CLASSES].to_numpy(dtype=float)
        print("Loaded Teacher: E101 Geo Ensemble", flush=True)
    except FileNotFoundError:
        print("E101 not found, falling back to E96 heading+ac1", flush=True)
        teacher_df = pd.read_csv(ROOT / "submissions" / "e96_nbalt_heading_ac1_tau0.25_g0.10_waltR0.50_priortau0.15_20260227_2008.csv")
        teacher_preds = teacher_df[CLASSES].to_numpy(dtype=float)

    # 3. Prepare Pseudo-Labels
    print("\nPreparing Pseudo-Labels...", flush=True)
    # We only pseudo-label the unseen months where the domain shift is
    unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
    Xte_unseen = Xte[unseen_mask]
    P_teacher_unseen = teacher_preds[unseen_mask]
    
    # Temperature scaling to soften the labels (prevents overconfidence/confirmation bias)
    TEMP = 1.5
    logits = np.log(np.clip(P_teacher_unseen, 1e-12, 1.0))
    P_soft = np.exp(logits / TEMP)
    P_soft = renorm_rows(P_soft)
    
    # Convert train hard labels to one-hot
    Y_train_oh = np.zeros((len(y_train), N_CLASSES), dtype=float)
    Y_train_oh[np.arange(len(y_train)), y_train] = 1.0
    
    # Combine Train and Pseudo-Labeled Test
    # We weight the pseudo-labels lower than real train data
    PSEUDO_WEIGHT = 0.3
    
    X_combined = np.vstack([Xtr, Xte_unseen])
    Y_combined = np.vstack([Y_train_oh, P_soft])
    
    # Sample weights: 1.0 for train, PSEUDO_WEIGHT for test
    # Also apply class balancing based on the train set
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    w_class = len(y_train) / (N_CLASSES * np.maximum(counts, 1.0))
    
    w_train = w_class[y_train]
    
    # For test, we use the expected class weight based on the soft labels
    w_test = np.sum(P_soft * w_class[None, :], axis=1) * PSEUDO_WEIGHT
    
    W_combined = np.concatenate([w_train, w_test])

    # 4. Train Student Model
    print(f"\nTraining Student Model on {len(X_combined)} samples ({len(Xtr)} train + {len(Xte_unseen)} pseudo)...", flush=True)
    
    # We must use a model that accepts soft targets. LGBM does not natively support soft targets
    # in the scikit-learn API, so we use the native API.
    import lightgbm as lgb
    
    # Create dataset (labels must be 1D for init, we pass soft labels in custom obj)
    # LightGBM requires label length to match data length, so we pass dummy labels
    dummy_labels = np.zeros(len(X_combined))
    train_data = lgb.Dataset(X_combined, label=dummy_labels, weight=W_combined)
    
    # We pass the real soft labels via a closure
    def soft_logloss_closure(preds, train_data):
        labels = Y_combined
        preds = preds.reshape(-1, N_CLASSES)
        # softmax
        preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        preds = preds / np.sum(preds, axis=1, keepdims=True)
        # grad and hess
        grad = preds - labels
        hess = preds * (1.0 - preds)
        return grad.flatten(), hess.flatten()

    params = {
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'num_class': N_CLASSES,
        'verbose': -1,
        'objective': soft_logloss_closure
    }
    
    student = lgb.train(
        params,
        train_data,
        num_boost_round=400
    )
    
    # Predict on full test set
    logits_test = student.predict(Xte)
    # Softmax
    logits_test = logits_test - np.max(logits_test, axis=1, keepdims=True)
    P_student = np.exp(logits_test)
    P_student = renorm_rows(P_student)
    
    save_submission(P_student, "e110_student_raw", cv_map=None)

    # 5. Apply E96 Post-Processing
    print("\nApplying E96 Unseen-Month Post-Processing to Student...", flush=True)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_heading, train_ac1, train_ok = extract_heading_ac1(train_df)
    test_heading, test_ac1, test_ok = extract_heading_ac1(test_df)

    test_p0, changed = apply_gated_ratio_priors(
        P_student, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    
    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
    
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y_train, train_heading, train_ac1, train_ok, use_heading=True, use_ac1=True
    )
    loglike_test = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, mu, sig, test_heading, test_ac1, test_ok, use_heading=True, use_ac1=True
    )
    
    P_final = apply_nb_poe(test_p0, loglike_test, gamma=GAMMA, gate=gate)
    
    name = f"e110_student_pp_heading_ac1_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}"
    save_submission(P_final, name, cv_map=None)
    
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
