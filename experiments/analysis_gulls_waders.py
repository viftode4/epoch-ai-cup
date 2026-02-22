import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_train, CLASSES

train_df = load_train()
le = LabelEncoder()
y = le.fit_transform(train_df["bird_group"])
train_df["label"] = y

# Print distribution of radar_bird_size per class
print("Class | Size Distribution")
size_map = {"Small bird": 0, "Medium bird": 1, "Large bird": 2, "Flock": 3}
train_df["size_val"] = train_df["radar_bird_size"].map(size_map)

for i, cls in enumerate(CLASSES):
    subset = train_df[train_df["label"] == i]
    sizes = subset["radar_bird_size"].value_counts(normalize=True) * 100
    size_str = ", ".join([f"{k}: {v:.1f}%" for k, v in sizes.items()])
    print(f"{cls:15s} | {size_str}")

print("\nAirspeed stats per class:")
for i, cls in enumerate(CLASSES):
    subset = train_df[train_df["label"] == i]
    mean_speed = subset["airspeed"].mean()
    std_speed = subset["airspeed"].std()
    print(f"{cls:15s} | Mean: {mean_speed:.2f} m/s | Std: {std_speed:.2f} m/s")
