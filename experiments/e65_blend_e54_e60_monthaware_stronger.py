"""E65: Stronger month-aware blend of E54 (best LB) with E60 (tuned LGB).

Same idea as E63, but with a stronger injection of E60 on Sep/Oct (9/10).
Use this only as an exploratory submission (higher risk / higher variance).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

E54_PATH = ROOT / "submissions" / "e54_e50_winter_tilt_m2_0.22_m5_0.12_m12_0.24_20260218_2229.csv"
E60_PATH = ROOT / "submissions" / "e60_lgbm_tuned_0.7524_20260221_2328.csv"

W_E60_M9 = 0.20
W_E60_M10 = 0.20


def _load_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"track_id", *CLASSES}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df[["track_id", *CLASSES]]


def _renorm_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


print("=" * 70, flush=True)
print("E65 MONTH-AWARE BLEND (STRONG) E54 + E60".center(70), flush=True)
print("=" * 70, flush=True)

test = load_test()
test_months = pd.to_datetime(test["timestamp_start_radar_utc"]).dt.month.values

e54 = _load_sub(E54_PATH)
e60 = _load_sub(E60_PATH)

df = (
    test[["track_id"]]
    .merge(e54, on="track_id", how="left", validate="one_to_one")
    .merge(e60, on="track_id", how="left", suffixes=("_e54", "_e60"), validate="one_to_one")
)

P54 = df[[f"{c}_e54" for c in CLASSES]].to_numpy(dtype=float)
P60 = df[[f"{c}_e60" for c in CLASSES]].to_numpy(dtype=float)

out = P54.copy()
for m, w in [(9, W_E60_M9), (10, W_E60_M10)]:
    mask = test_months == m
    out[mask] = (1.0 - w) * P54[mask] + w * P60[mask]

out = _renorm_rows(out)

save_submission(out, f"e65_blend_e54_e60_m9_{W_E60_M9:.2f}_m10_{W_E60_M10:.2f}", cv_map=None)
print("\nDone.", flush=True)

