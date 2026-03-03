import pandas as pd
import numpy as np
from src.data import CLASSES

files = [
    "submissions/e101_ens_geo4_20260228_2132.csv",
    "submissions/e109_shared_specialists_a0.40_t0.30_pp_heading_ac1_20260302_1304.csv",
    "submissions/e98_nbalt_flock_tau0.25_g0.10_waltR0.50_priortau0.15_20260228_2040.csv",
    "submissions/e99_waderspair_flock_tau0.25_gb0.10_gf0.10_waltR0.50_priortau0.15_20260228_2052.csv",
    "submissions/e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv"
]

preds = []
for f in files:
    df = pd.read_csv(f)
    preds.append(df[CLASSES].values)

print("Top-1 Agreement Matrix:")
n = len(files)
for i in range(n):
    row = []
    for j in range(n):
        t1 = np.argmax(preds[i], axis=1)
        t2 = np.argmax(preds[j], axis=1)
        agree = np.mean(t1 == t2)
        row.append(f"{agree:.3f}")
    print(" ".join(row))

