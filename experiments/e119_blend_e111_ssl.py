"""E119: Blend strong baseline (E111) with SSL trajectory model (E118).

Purpose
-------
If E118 learned any *orthogonal* within-track signal (trajectory rhythm/shape),
it may complement the saturated E111 tree+PP ensemble. A conservative geometric
blend is the safest way to test that hypothesis on LB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def load_submission_probs(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(CLASSES):
        p[:, j] = df[cls].to_numpy(dtype=np.float32)
    return renorm_rows(p)


def geo_blend(p_a: np.ndarray, p_b: np.ndarray, w_b: float) -> np.ndarray:
    w_b = float(w_b)
    w_a = 1.0 - w_b
    logp = w_a * np.log(np.clip(p_a, 1e-12, 1.0)) + w_b * np.log(np.clip(p_b, 1e-12, 1.0))
    p = np.exp(logp).astype(np.float32)
    return renorm_rows(p)


def main() -> None:
    print("=" * 70, flush=True)
    print("E119 BLEND E111 + E118(SSL)".center(70), flush=True)
    print("=" * 70, flush=True)

    p_e111 = load_submission_probs(ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv")
    p_e118 = load_submission_probs(ROOT / "submissions" / "e118_ssl_simclr_lgbm_0.4141_20260303_1231.csv")

    for w in (0.10, 0.20, 0.30):
        p = geo_blend(p_e111, p_e118, w_b=w)
        save_submission(p, f"e119_geo_blend_e111_e118_w{w:.2f}", cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

