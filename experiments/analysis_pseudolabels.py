import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_test, CLASSES

test_df = load_test()
# Since we don't have test_e54_winter_tilt.npy, we can parse the CSV
sub_path = ROOT / "submissions" / "e54_e50_winter_tilt_m2_0.22_m5_0.12_m12_0.24_20260218_2229.csv"
sub_df = pd.read_csv(sub_path)

print("Pseudo-labeling candidates from best LB model (0.56):")
for cls in CLASSES:
    probs = sub_df[cls].values
    count_90 = np.sum(probs > 0.50)
    count_80 = np.sum(probs > 0.40)
    count_70 = np.sum(probs > 0.30)
    print(f"{cls:15s} | >0.5: {count_90:4d} | >0.4: {count_80:4d} | >0.3: {count_70:4d}")
