"""E120: Gate SSL trajectory expert to unseen months + low-margin only.

Motivation
----------
E119 showed that mixing SSL (E118) globally hurts LB (0.57). That does *not* imply
the SSL model has zero value; it likely hurts shared months (9/10) where E111 is
already strong, while potentially helping the unseen months (2/5/12) where our
pipeline historically needed extra evidence.

This script applies E118 as a *conditional expert*:
  - start from E111 (strong baseline)
  - only for unseen months AND low-confidence rows (margin < tau)
    apply a geometric blend with a small SSL weight.

If SSL carries any useful within-track signal, this is the safest way to test it
with one remaining Kaggle submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_test  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def top2_margin(p: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, axis=1)
    p1 = p[np.arange(len(p)), order[:, 0]]
    p2 = p[np.arange(len(p)), order[:, 1]]
    return p1 - p2


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
    print("=" * 72, flush=True)
    print("E120 GATED SSL EXPERT (UNSEEN MONTHS ONLY)".center(72), flush=True)
    print("=" * 72, flush=True)

    base_path = ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv"
    ssl_path = ROOT / "submissions" / "e118_ssl_simclr_lgbm_0.4141_20260303_1231.csv"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not ssl_path.exists():
        raise FileNotFoundError(ssl_path)

    p_base = load_submission_probs(base_path)
    p_ssl = load_submission_probs(ssl_path)

    test_df = load_test()
    months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.to_numpy()
    unseen = np.isin(months, [2, 5, 12])
    margin = top2_margin(p_base)

    # Grid a couple of safe settings (only one should be submitted)
    configs = [
        ("A", 0.25, 0.12),
        ("B", 0.30, 0.12),
        ("C", 0.25, 0.18),
    ]
    for tag, tau, w in configs:
        gate = unseen & (margin < float(tau))
        p_out = p_base.copy()
        if gate.sum() > 0:
            p_out[gate] = geo_blend(p_base[gate], p_ssl[gate], w_b=float(w))
        name = f"e120_gated_ssl_{tag}_tau{tau:.2f}_w{w:.2f}"
        print(f"  {name}: gated {int(gate.sum())}/{len(gate)} rows", flush=True)
        save_submission(p_out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

