import numpy as np
import pandas as pd
from src.data import CLASSES

df = pd.read_csv("submissions/e96_nbalt_heading_ac1_tau0.25_g0.10_waltR0.50_priortau0.15_20260227_2008.csv")
preds = df[CLASSES].to_numpy()
top1 = np.argmax(preds, axis=1)
counts = np.bincount(top1, minlength=len(CLASSES))
for i, c in enumerate(CLASSES):
    print(f"{c:15s}: {counts[i]}")
