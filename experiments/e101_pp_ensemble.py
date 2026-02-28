"""E101: Ensemble diverse 0.59 submissions (post-processing level).

Why
---
Public LB uses ~24% of test, so many materially different solutions tie at 0.59.
Ensembling diverse post-processing pipelines can reduce variance and improve the
private ranking signal (macro-AP), even when each member ties publicly.

We intentionally ensemble *diverse* 0.59 submissions (measured by mean L1/JS):
  - E78A: NB-alt weighted (tau=0.30)
  - E79A: NB-alt weighted (tau=0.25)
  - E96:  NB-alt + heading_R + rcs_ac1
  - E100A: baseline + wind-comp evidence stage

Outputs
-------
Saves two candidates:
  1) geometric mean in log-space (product of experts over submissions)
  2) arithmetic mean (mixture)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_sample_submission  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


FILES = [
    "submissions/e78_nbalt_weighted_ws1.00_wv1.00_wm1.00_wr0.50_tau0.30_g0.10_20260224_1602.csv",
    "submissions/e79_nbalt_wr0.50_tau0.25_g0.10_priortau0.15_20260225_1116.csv",
    "submissions/e96_nbalt_heading_ac1_tau0.25_g0.10_waltR0.50_priortau0.15_20260227_2008.csv",
    "submissions/e100_windcomp_A_tau0.25_gw0.10_priortau0.15_20260228_2121.csv",
]


def _read_sub(path: Path, cols: list[str], ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if ids is None:
        ids = df["track_id"].to_numpy()
        arr = df[cols].to_numpy(float)
        return ids, arr
    df2 = df.set_index("track_id").loc[ids].reset_index()
    arr = df2[cols].to_numpy(float)
    return ids, arr


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, None)
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    sample = load_sample_submission()
    sub_cols = [c for c in sample.columns if c != "track_id"]

    ids = None
    preds = []
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            raise FileNotFoundError(str(p))
        ids, arr = _read_sub(p, sub_cols, ids)
        preds.append(_renorm(arr.astype(float)))

    stack = np.stack(preds, axis=0)  # (m, n, k)
    m, n, k = stack.shape
    print(f"Loaded {m} submissions, rows={n}, classes={k}", flush=True)

    # Candidate 1: geometric mean (log-space average)
    logp = np.log(np.clip(stack, 1e-15, 1.0))
    logp_mean = np.mean(logp, axis=0)  # (n, k)
    geo = np.exp(logp_mean)
    geo = _renorm(geo)

    # Candidate 2: arithmetic mean
    ar = _renorm(np.mean(stack, axis=0))

    # Convert from submission column order -> CLASSES order for save_submission()
    col_to_idx = {c: i for i, c in enumerate(sub_cols)}
    cls_idx = [col_to_idx[c] for c in CLASSES]
    geo_cls = geo[:, cls_idx]
    ar_cls = ar[:, cls_idx]

    save_submission(geo_cls, "e101_ens_geo4", cv_map=None)
    save_submission(ar_cls, "e101_ens_avg4", cv_map=None)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

