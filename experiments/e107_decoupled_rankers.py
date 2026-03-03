"""E107: Decoupled Pairwise Rankers (LambdaMART) + Calibrated PP.

Mathematical Motivation:
Multi-class Cross-Entropy (softmax) couples class probabilities. If a sample is 99% Gull,
it is forced to be 1% Wader. This destroys the intra-class ranking for minority classes,
which is what macro-mAP actually evaluates.

Solution:
1. Train 9 independent XGBRankers (objective='rank:pairwise') to explicitly maximize
   the Area Under the PR Curve (ranking) for each class independently.
2. Calibrate the 9 unbounded ranking scores into a joint probability distribution using
   Logistic Regression on the OOF predictions.
3. Apply our proven E96 (heading+ac1) unseen-month post-processing on the calibrated base.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from experiments.e96_nbalt_heading_ac1 import (
    BASE_ALPHA, TAU_PRIOR, TAU_NB, GAMMA, W_ALTRANGE,
    build_gbif_priors, apply_gated_ratio_priors, extract_heading_ac1,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe, renorm_rows, top2_margin
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42
UNSEEN_MONTHS = (2, 5, 12)

def main() -> None:
    print("=" * 70, flush=True)
    print("E107 DECOUPLED PAIRWISE RANKERS + CALIBRATED PP".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    # 1. Build Features (Strictly no temporal leaks)
    feature_sets = ["core", "tabular", "rcs_fft", "weakclass", "flight_physics"]
    Xtr_df = build_features(train_df, feature_sets=feature_sets)
    Xte_df = build_features(test_df, feature_sets=feature_sets)

    drop_cols = [c for c in ALL_TEMPORAL if c in Xtr_df.columns]
    if drop_cols:
        Xtr_df = Xtr_df.drop(columns=drop_cols, errors="ignore")
        Xte_df = Xte_df.drop(columns=drop_cols, errors="ignore")

    Xtr_df = Xtr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte_df = Xte_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xtr = Xtr_df.to_numpy(np.float32)
    Xte = Xte_df.to_numpy(np.float32)

    # 2. Train 9 Independent Rankers (LOMO CV for OOF scores)
    print("\nTraining 9 Decoupled XGBRankers (LOMO)...", flush=True)
    unique_months = sorted(set(train_months))
    
    S_oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    S_test = np.zeros((len(test_df), N_CLASSES), dtype=np.float32)

    for c in range(N_CLASSES):
        print(f"  -> Ranking Class {c}: {CLASSES[c]}", flush=True)
        y_bin = (y == c).astype(int)
        
        # LOMO for OOF
        for m in unique_months:
            tr_idx = np.where(train_months != m)[0]
            va_idx = np.where(train_months == m)[0]
            
            ranker = XGBRanker(
                objective='rank:pairwise',
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1
            )
            # Single query group for the whole fold
            ranker.fit(Xtr[tr_idx], y_bin[tr_idx], qid=np.zeros(len(tr_idx)))
            S_oof[va_idx, c] = ranker.predict(Xtr[va_idx])
            
        # Full train for Test
        ranker_full = XGBRanker(
            objective='rank:pairwise',
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1
        )
        ranker_full.fit(Xtr, y_bin, qid=np.zeros(len(Xtr)))
        S_test[:, c] = ranker_full.predict(Xte)

    # 3. Calibrate Ranking Scores into Probabilities
    print("\nCalibrating Ranking Scores into Joint Probabilities...", flush=True)
    # Use a cross-validated Logistic Regression to prevent overfitting the calibration
    calibrator = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced', random_state=SEED)
    
    P_oof = np.zeros_like(S_oof)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(S_oof, y):
        calibrator.fit(S_oof[tr_idx], y[tr_idx])
        P_oof[va_idx] = calibrator.predict_proba(S_oof[va_idx])
        
    calibrator.fit(S_oof, y)
    P_test = calibrator.predict_proba(S_test)
    
    P_oof = renorm_rows(P_oof)
    P_test = renorm_rows(P_test)
    
    m, per = compute_map(y, P_oof)
    print_results(m, per, "E107 Decoupled Rankers (Calibrated LOMO)")
    
    np.save("oof_e107.npy", P_oof)
    np.save("test_e107.npy", P_test)
    save_submission(P_test, "e107_ranker_calibrated_raw", cv_map=m)

    # 4. Apply E96 Post-Processing (Heading + AC1)
    print("\nApplying E96 Unseen-Month Post-Processing...", flush=True)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_heading, train_ac1, train_ok = extract_heading_ac1(train_df)
    test_heading, test_ac1, test_ok = extract_heading_ac1(test_df)

    test_p0, changed = apply_gated_ratio_priors(
        P_test, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    
    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
    
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, train_heading, train_ac1, train_ok, use_heading=True, use_ac1=True
    )
    loglike_test = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, mu, sig, test_heading, test_ac1, test_ok, use_heading=True, use_ac1=True
    )
    
    P_final = apply_nb_poe(test_p0, loglike_test, gamma=GAMMA, gate=gate)
    
    name = f"e107_ranker_pp_heading_ac1_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}"
    save_submission(P_final, name, cv_map=None)
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
