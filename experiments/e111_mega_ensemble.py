import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.data import CLASSES
from src.submission import save_submission

files = [
    "submissions/e101_ens_geo4_20260228_2132.csv",
    "submissions/e109_shared_specialists_a0.40_t0.30_pp_heading_ac1_20260302_1304.csv",
    "submissions/e98_nbalt_flock_tau0.25_g0.10_waltR0.50_priortau0.15_20260228_2040.csv",
    "submissions/e100_windcomp_A_tau0.25_gw0.10_priortau0.15_20260228_2121.csv",
    "submissions/e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv"
]

preds = []
for f in files:
    df = pd.read_csv(f)
    # Ensure order matches
    preds.append(df[CLASSES].values)

# Geometric Mean
stack = np.stack([np.clip(p, 1e-15, 1.0) for p in preds], axis=0)
m = np.mean(np.log(stack), axis=0)
geo_ens = np.exp(m)
geo_ens = geo_ens / geo_ens.sum(axis=1, keepdims=True)

save_submission(geo_ens, "e111_mega_ensemble_geo5", cv_map=None)

# Arithmetic Mean
avg_ens = np.mean(stack, axis=0)
avg_ens = avg_ens / avg_ens.sum(axis=1, keepdims=True)

save_submission(avg_ens, "e111_mega_ensemble_avg5", cv_map=None)

print("Saved E111 Mega Ensembles.")
