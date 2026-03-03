import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")
subs = {
    "E101": "submissions/e101_ens_geo4_20260228_2132.csv",
    "E109": "submissions/e109_shared_specialists_a0.40_t0.30_pp_heading_ac1_20260302_1304.csv",
    "E99": "submissions/e99_waderspair_flock_tau0.25_gb0.10_gf0.10_waltR0.50_priortau0.15_20260228_2052.csv",
    "E96": "submissions/e96_nbalt_heading_ac1_tau0.25_g0.10_waltR0.50_priortau0.15_20260227_2008.csv"
}

dfs = {}
for name, path in subs.items():
    try:
        dfs[name] = pd.read_csv(ROOT / path).set_index("track_id")
    except Exception as e:
        print(f"Missing {name}: {e}")

if len(dfs) > 1:
    keys = list(dfs.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            diff = np.abs(dfs[k1].values - dfs[k2].values).mean()
            print(f"MAE {k1} vs {k2}: {diff:.5f}")
            
            # Top-1 disagreement
            top1_1 = np.argmax(dfs[k1].values, axis=1)
            top1_2 = np.argmax(dfs[k2].values, axis=1)
            disagree = np.sum(top1_1 != top1_2)
            print(f"Top-1 Disagreement {k1} vs {k2}: {disagree} / {len(top1_1)}")
