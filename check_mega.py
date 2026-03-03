import pandas as pd
import numpy as np
from src.data import CLASSES

df1 = pd.read_csv("submissions/e101_ens_geo4_20260228_2132.csv")
df2 = pd.read_csv("submissions/e111_mega_ensemble_geo5_20260302_1333.csv")

p1 = df1[CLASSES].values
p2 = df2[CLASSES].values

t1 = np.argmax(p1, axis=1)
t2 = np.argmax(p2, axis=1)

print(f"Agreement: {np.mean(t1 == t2):.4f}")
print("Changes:")
for i in range(len(t1)):
    if t1[i] != t2[i]:
        print(f"Row {i}: {CLASSES[t1[i]]} -> {CLASSES[t2[i]]}")
