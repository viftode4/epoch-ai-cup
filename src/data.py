"""Data loading and EWKB parsing utilities."""
import pandas as pd
import numpy as np
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CLASSES = [
    "Birds of Prey", "Clutter", "Cormorants", "Ducks", "Geese",
    "Gulls", "Pigeons", "Songbirds", "Waders",
]


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def parse_ewkb_4d(hex_str: str) -> list[tuple[float, float, float, float]]:
    """Parse EWKB hex into list of (lon, lat, altitude_m, rcs_dBm2) tuples."""
    raw = bytes.fromhex(hex_str)
    offset = 0
    bo = "<" if raw[offset] == 1 else ">"
    offset += 1
    geom_type = struct.unpack_from(f"{bo}I", raw, offset)[0]
    offset += 4
    if geom_type & 0x20000000:  # SRID flag
        offset += 4
    n_points = struct.unpack_from(f"{bo}I", raw, offset)[0]
    offset += 4
    points = []
    for _ in range(n_points):
        lon, lat, alt, rcs = struct.unpack_from(f"{bo}4d", raw, offset)
        points.append((lon, lat, alt, rcs))
        offset += 32
    return points


def parse_trajectory_time(traj_time_str: str) -> np.ndarray:
    """Parse trajectory_time string into numpy array."""
    import json
    return np.array(json.loads(traj_time_str))
