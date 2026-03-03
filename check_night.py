import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def get_night_frac(df):
    dt = pd.to_datetime(df["timestamp_start_radar_utc"])
    hour = dt.dt.hour
    # rough night definition: 19:00 to 05:00
    is_night = (hour >= 19) | (hour <= 5)
    return is_night.mean()

print(f"Train night frac: {get_night_frac(train):.3f}")
print(f"Test night frac:  {get_night_frac(test):.3f}")
