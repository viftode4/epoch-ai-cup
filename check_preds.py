import numpy as np
import pandas as pd
from src.data import CLASSES

preds = np.load("test_e50.npy")
top1 = np.argmax(preds, axis=1)
counts = np.bincount(top1, minlength=len(CLASSES))
for i, c in enumerate(CLASSES):
    print(f"{c:15s}: {counts[i]}")
