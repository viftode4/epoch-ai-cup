"""E109: Shared-Month Targeted Specialists (Gulls vs Waders vs Pigeons).

Mathematical Motivation:
1. The 0.59 plateau is strictly bottlenecked by the shared months (Sep/Oct).
2. Unseen-month post-processing (E75, E78, E79, E96, E98) has maxed out its potential.
3. Attempts to fix the base model globally (E107 Rankers, E108 Wind Features) failed
   because they disrupt the delicate balance of the 0.59 base model.
4. The remaining errors in Sep/Oct are highly localized: the model confuses Gulls,
   Waders, and Pigeons.
5. Instead of changing the global base model, we train highly targeted binary specialists
   (Gulls vs Waders, Gulls vs Pigeons) using ONLY the shared months (Sep/Oct) data.
6. We apply these specialists ONLY to samples in the shared months where the base model
   is uncertain between these specific pairs. This surgically fixes the bottleneck
   without touching the rest of the distribution.

Pipeline:
1. Load E79 base predictions (our 0.59 anchor).
2. Train binary LGBM specialists on Sep/Oct data only.
3. Apply specialists to Sep/Oct test samples where top-2 are the target pair.
4. Apply E96 (heading+ac1) post-processing on unseen months.
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
SHARED_MONTHS = (9, 10)

# E79 Pruned Features
PRUNED_FEATURES = [
    'airspeed', 'radar_bird_size', 'min_z', 'max_z', 'duration', 'track_length',
    'z_mean', 'z_std', 'rcs_mean', 'rcs_std', 'rcs_max', 'rcs_min',
    'speed_mean', 'speed_std', 'speed_max', 'speed_min', 'accel_mean', 'accel_std',
    'accel_max', 'accel_min', 'turn_mean', 'turn_std', 'turn_max', 'turn_min',
    'sinuosity', 'rcs_range', 'rcs_q25', 'rcs_q75', 'rcs_iqr', 'rcs_skew', 'rcs_kurtosis',
    'rcs_trend', 'rcs_p2p', 'rcs_smoothness', 'rcs_ac1', 'rcs_ac2'
]

def train_and_apply_specialist(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
    c1: int, c2: int,
    base_oof: np.ndarray, base_test: np.ndarray,
    train_months: np.ndarray, test_months: np.ndarray,
    alpha: float = 0.5, tau: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Train binary specialist on shared months, apply to shared months."""
    print(f"\n--- Specialist: {CLASSES[c1]} vs {CLASSES[c2]} ---", flush=True)
    
    # 1. Filter training data: only c1/c2 AND only shared months
    mask_tr = np.isin(ytr, [c1, c2]) & np.isin(train_months, SHARED_MONTHS)
    X_sub = Xtr[mask_tr]
    y_sub = (ytr[mask_tr] == c1).astype(int)  # 1 if c1, 0 if c2
    
    print(f"Training on {len(y_sub)} samples ({CLASSES[c1]}:{y_sub.sum()}, {CLASSES[c2]}:{len(y_sub)-y_sub.sum()})", flush=True)
    
    clf = LGBMClassifier(
        n_estimators=200, learning_rate=0.02, max_depth=5, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    
    # 2. OOF Predictions (for validation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_spec = np.zeros(len(y_sub), dtype=float)
    for tr_idx, va_idx in skf.split(X_sub, y_sub):
        clf.fit(X_sub[tr_idx], y_sub[tr_idx])
        oof_spec[va_idx] = clf.predict_proba(X_sub[va_idx])[:, 1]
        
    # 3. Full Train & Test Predict
    clf.fit(X_sub, y_sub)
    test_spec = clf.predict_proba(Xte)[:, 1]
    
    # 4. Apply to Base Predictions (Gated)
    new_oof = base_oof.copy()
    new_test = base_test.copy()
    
    def apply_injection(base_p: np.ndarray, spec_p: np.ndarray, months: np.ndarray, is_train: bool) -> tuple[np.ndarray, int]:
        out = base_p.copy()
        
        # Find rows where top 2 classes are exactly c1 and c2
        order = np.argsort(-out, axis=1)
        top1 = order[:, 0]
        top2 = order[:, 1]
        
        is_pair = ((top1 == c1) & (top2 == c2)) | ((top1 == c2) & (top2 == c1))
        margin = out[np.arange(len(out)), top1] - out[np.arange(len(out)), top2]
        
        # Gate: must be shared month, must be the target pair, must be uncertain
        gate = np.isin(months, SHARED_MONTHS) & is_pair & (margin < tau)
        
        if is_train:
            # For OOF, we only have specialist predictions for the subset
            # We need to map the subset indices back to the full array
            sub_idx = np.where(mask_tr)[0]
            idx_map = {full_i: sub_i for sub_i, full_i in enumerate(sub_idx)}
            
            valid_gate = []
            for i in np.where(gate)[0]:
                if i in idx_map:
                    valid_gate.append(i)
            gate_idx = np.array(valid_gate, dtype=int)
            
            if len(gate_idx) > 0:
                sub_indices = [idx_map[i] for i in gate_idx]
                p_c1 = oof_spec[sub_indices]
        else:
            gate_idx = np.where(gate)[0]
            if len(gate_idx) > 0:
                p_c1 = spec_p[gate_idx]
                
        if len(gate_idx) > 0:
            p_c2 = 1.0 - p_c1
            
            # Extract current mass assigned to the pair
            mass = out[gate_idx, c1] + out[gate_idx, c2]
            
            # Blend base ratio with specialist ratio
            base_ratio_c1 = out[gate_idx, c1] / np.clip(mass, 1e-12, None)
            base_ratio_c2 = out[gate_idx, c2] / np.clip(mass, 1e-12, None)
            
            new_ratio_c1 = (1 - alpha) * base_ratio_c1 + alpha * p_c1
            new_ratio_c2 = (1 - alpha) * base_ratio_c2 + alpha * p_c2
            
            # Reassign mass
            out[gate_idx, c1] = mass * new_ratio_c1
            out[gate_idx, c2] = mass * new_ratio_c2
            
        return renorm_rows(out), len(gate_idx)

    new_oof, oof_changed = apply_injection(new_oof, oof_spec, train_months, is_train=True)
    new_test, test_changed = apply_injection(new_test, test_spec, test_months, is_train=False)
    
    print(f"Applied to {oof_changed} OOF rows, {test_changed} Test rows.", flush=True)
    return new_oof, new_test

def main() -> None:
    print("=" * 70, flush=True)
    print("E109 SHARED-MONTH TARGETED SPECIALISTS".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
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

    # 2. Load Base Predictions (E79 -> actually E50 since E79 wasn't saved)
    oof_base = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
    
    base_map, _ = compute_map(y, oof_base)
    print(f"\nBase E50 mAP: {base_map:.4f}", flush=True)

    # 3. Apply Targeted Specialists
    c_gull = CLASSES.index("Gulls")
    c_wader = CLASSES.index("Waders")
    c_pigeon = CLASSES.index("Pigeons")
    
    ALPHA = 0.40
    TAU_SPEC = 0.30
    
    oof_spec, test_spec = train_and_apply_specialist(
        Xtr, y, Xte, c_gull, c_wader, oof_base, test_base, train_months, test_months, alpha=ALPHA, tau=TAU_SPEC
    )
    
    oof_spec, test_spec = train_and_apply_specialist(
        Xtr, y, Xte, c_gull, c_pigeon, oof_spec, test_spec, train_months, test_months, alpha=ALPHA, tau=TAU_SPEC
    )
    
    spec_map, _ = compute_map(y, oof_spec)
    print(f"\nSpecialist OOF mAP: {spec_map:.4f} (Delta: {spec_map - base_map:+.4f})", flush=True)

    # 4. Apply E96 Unseen-Month Post-Processing
    print("\nApplying E96 Unseen-Month Post-Processing...", flush=True)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_heading, train_ac1, train_ok = extract_heading_ac1(train_df)
    test_heading, test_ac1, test_ok = extract_heading_ac1(test_df)

    test_p0, changed = apply_gated_ratio_priors(
        test_spec, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
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
    
    name = f"e109_shared_specialists_a{ALPHA:.2f}_t{TAU_SPEC:.2f}_pp_heading_ac1"
    save_submission(P_final, name, cv_map=None)
    
    # Save a conservative variant
    oof_spec_c, test_spec_c = train_and_apply_specialist(
        Xtr, y, Xte, c_gull, c_wader, oof_base, test_base, train_months, test_months, alpha=0.20, tau=0.20
    )
    oof_spec_c, test_spec_c = train_and_apply_specialist(
        Xtr, y, Xte, c_gull, c_pigeon, oof_spec_c, test_spec_c, train_months, test_months, alpha=0.20, tau=0.20
    )
    test_p0_c, _ = apply_gated_ratio_priors(test_spec_c, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
    margin0_c = top2_margin(test_p0_c)
    gate_c = np.isin(test_months, UNSEEN_MONTHS) & (margin0_c < TAU_NB)
    P_final_c = apply_nb_poe(test_p0_c, loglike_test, gamma=GAMMA, gate=gate_c)
    
    name_c = f"e109_shared_specialists_a0.20_t0.20_pp_heading_ac1"
    save_submission(P_final_c, name_c, cv_map=None)
    
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
