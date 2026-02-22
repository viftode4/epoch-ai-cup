"""E64: Unseen-month blend of E54 (best LB) with E62 (soft pseudo-label model).

E62 by itself collapses (very Wader-heavy) on test, so we only inject a tiny
amount of it on the unseen months (Feb/May/Dec = 2/5/12) where E54 priors
already dominate. This is intended to add minority-class ranking diversity
without breaking the baseline.
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
E62_PATH = ROOT / "submissions" / "e62_soft_pseudolabels_0.7711_20260221_2150.csv"

# Tiny injection weight (E62 is very aggressive on Waders/Gulls).
W_E62_UNSEEN = 0.05


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
print("E64 UNSEEN-MONTH BLEND E54 + E62".center(70), flush=True)
print("=" * 70, flush=True)

test = load_test()
test_months = pd.to_datetime(test["timestamp_start_radar_utc"]).dt.month.values

e54 = _load_sub(E54_PATH)
e62 = _load_sub(E62_PATH)

df = (
    test[["track_id"]]
    .merge(e54, on="track_id", how="left", validate="one_to_one")
    .merge(e62, on="track_id", how="left", suffixes=("_e54", "_e62"), validate="one_to_one")
)

P54 = df[[f"{c}_e54" for c in CLASSES]].to_numpy(dtype=float)
P62 = df[[f"{c}_e62" for c in CLASSES]].to_numpy(dtype=float)

out = P54.copy()
for m in [2, 5, 12]:
    mask = test_months == m
    out[mask] = (1.0 - W_E62_UNSEEN) * P54[mask] + W_E62_UNSEEN * P62[mask]

out = _renorm_rows(out)

save_submission(out, f"e64_blend_e54_e62_unseen_w{W_E62_UNSEEN:.2f}", cv_map=None)
print("\nDone.", flush=True)

