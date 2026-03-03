"""E112: Geographic / Spatial Priors.

Mathematical Motivation:
1. We have exhausted kinematics (speed, altitude, turning, flocking).
2. We have exhausted temporal priors (month, time of day).
3. We have NOT yet exploited spatial priors (latitude/longitude).
4. The dataset spans different radar locations in the Netherlands.
5. Bird migration routes (flyways) and local habitats are highly geographically dependent.
   For example, Waders are much more common near the coast (Wadden Sea / Zeeland) than inland.
   Cormorants are tied to large bodies of water.
6. If the train and test sets have different spatial distributions (or if the base model
   under-utilizes spatial coordinates because it relies too heavily on kinematics),
   a spatial prior P(class | lat, lon) can provide the missing orthogonal signal.

Pipeline:
1. Load train data, extract mean lat/lon per track.
2. Train a spatial Kernel Density Estimator (KDE) or simple spatial KNN for P(class | lat, lon).
3. Load E101 (our best 0.59 ensemble) as the base.
4. Apply a Bayesian update using the spatial prior, gated by uncertainty.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.submission import save_submission
from experiments.e96_nbalt_heading_ac1 import renorm_rows, top2_margin

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

def extract_spatial(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    coords = np.zeros((n, 2), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) > 0:
                lons = [p[0] for p in pts]
                lats = [p[1] for p in pts]
                coords[i, 0] = np.mean(lons)
                coords[i, 1] = np.mean(lats)
        except Exception:
            pass
    return coords

def main() -> None:
    print("=" * 70, flush=True)
    print("E112 GEOGRAPHIC / SPATIAL PRIORS".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()
    y_train = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)

    # 1. Extract Spatial Coordinates
    print("\nExtracting spatial coordinates...", flush=True)
    X_train_sp = extract_spatial(train_df)
    X_test_sp = extract_spatial(test_df)
    
    # Handle missing coords by imputing with global mean
    train_mean = np.nanmean(np.where(X_train_sp == 0, np.nan, X_train_sp), axis=0)
    X_train_sp[X_train_sp[:, 0] == 0] = train_mean
    X_test_sp[X_test_sp[:, 0] == 0] = train_mean

    # 2. Train Spatial Prior Model P(y | lat, lon)
    # We use KNN to get local geographical probabilities
    print("\nTraining Spatial KNN Prior...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)
    knn.fit(X_train_sp, y_train)
    
    # Get spatial probabilities for test set
    P_spatial = knn.predict_proba(X_test_sp)
    P_spatial = np.clip(P_spatial, 1e-4, 1.0)
    
    # 3. Load Base Predictions (E101 Geo Ensemble)
    try:
        base_df = pd.read_csv(ROOT / "submissions" / "e101_ens_geo4_20260228_2132.csv")
        P_base = base_df[CLASSES].to_numpy(dtype=float)
        print("Loaded Base: E101 Geo Ensemble", flush=True)
    except FileNotFoundError:
        print("E101 not found, falling back to E96", flush=True)
        base_df = pd.read_csv(ROOT / "submissions" / "e96_nbalt_heading_ac1_tau0.25_g0.10_waltR0.50_priortau0.15_20260227_2008.csv")
        P_base = base_df[CLASSES].to_numpy(dtype=float)

    # 4. Apply Bayesian Spatial Update
    print("\nApplying Spatial Prior Update...", flush=True)
    
    # Calculate global train prior to convert P(y|x_sp) to a likelihood ratio
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    P_train_prior = counts / counts.sum()
    
    # Spatial Likelihood Ratio: L = P(y|lat,lon) / P(y)
    L_spatial = P_spatial / np.clip(P_train_prior, 1e-12, None)
    
    # We apply this update only where the base model is uncertain
    TAU_SPATIAL = 0.30
    GAMMA_SPATIAL = 0.20  # Temperature/trust parameter
    
    margin = top2_margin(P_base)
    gate = margin < TAU_SPATIAL
    
    P_final = P_base.copy()
    P_final[gate] = P_final[gate] * (L_spatial[gate] ** GAMMA_SPATIAL)
    P_final = renorm_rows(P_final)
    
    print(f"Applied spatial update to {gate.sum()} uncertain rows.", flush=True)
    
    # 5. Save Submissions
    name = f"e112_spatial_prior_tau{TAU_SPATIAL:.2f}_g{GAMMA_SPATIAL:.2f}"
    save_submission(P_final, name, cv_map=None)
    
    # Conservative variant
    gate_c = margin < 0.20
    P_final_c = P_base.copy()
    P_final_c[gate_c] = P_final_c[gate_c] * (L_spatial[gate_c] ** 0.10)
    P_final_c = renorm_rows(P_final_c)
    name_c = f"e112_spatial_prior_tau0.20_g0.10"
    save_submission(P_final_c, name_c, cv_map=None)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
