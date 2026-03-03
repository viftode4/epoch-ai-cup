"""E115: Month-wise label-shift correction via MLLS (Saerens EM) on posteriors.

Motivation
----------
Public LB appears saturated at ~0.59 for models that only use per-track evidence.
The one remaining "cheap" lever that can move *shared months* (9/10) without adding
new raw signals is to correct *label shift* (changing class priors) more accurately
than fixed ecological priors.

This script applies month-wise MLLS / EM to adjust posteriors:
  q_i,c ∝ p_i,c * (π_m(c) / π_train(c))^alpha
with π_m estimated from unlabeled month predictions by EM (tempered by alpha).

It also reports an OOF sanity-check using `oof_e50.npy` on the train set:
apply the same month-wise EM per month (treating each month as "target") and
measure macro-mAP delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.metrics import compute_map, print_results  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _safe_prior(pi: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    pi = np.clip(pi.astype(np.float64), eps, None)
    return (pi / pi.sum()).astype(np.float64)


def em_mlls_month(
    p: np.ndarray,
    pi_train: np.ndarray,
    pi_init: np.ndarray,
    *,
    alpha: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return adjusted posteriors q and estimated target prior pi."""
    p = renorm_rows(p.astype(np.float64), eps=eps)
    pi_train = _safe_prior(pi_train)
    pi = _safe_prior(pi_init)

    ratio_pow = (pi / pi_train) ** float(alpha)
    for it in range(max_iter):
        numer = p * ratio_pow[None, :]
        q = renorm_rows(numer, eps=eps)
        pi_new = _safe_prior(q.mean(axis=0))

        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            return q.astype(np.float32), pi.astype(np.float32), it + 1

        pi = pi_new
        ratio_pow = (pi / pi_train) ** float(alpha)

    return q.astype(np.float32), pi.astype(np.float32), max_iter


def load_submission_probs(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    track_id = df["track_id"].to_numpy()
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(CLASSES):
        p[:, j] = df[cls].to_numpy(dtype=np.float32)
    return renorm_rows(p), track_id


def load_gbif_priors() -> pd.DataFrame:
    pri = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
    pri = pri.set_index("month")
    # Ensure CLASSES order
    pri = pri[CLASSES]
    return pri


def main() -> None:
    print("=" * 78, flush=True)
    print("E115: MONTH-WISE MLLS / EM LABEL-SHIFT CORRECTION".center(78), flush=True)
    print("=" * 78, flush=True)

    train_df = load_train()
    test_df = load_test()
    gbif = load_gbif_priors()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    pi_train = counts / counts.sum()

    # ---- OOF sanity check (does EM help within known months?) ----
    oof_path = ROOT / "oof_e50.npy"
    if oof_path.exists():
        p_oof = renorm_rows(np.load(oof_path).astype(np.float32))
        m0, per0 = compute_map(y, p_oof)
        print_results(m0, per0, label="Baseline OOF (oof_e50.npy)")

        months_tr = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.to_numpy()
        p_adj = p_oof.copy()

        for m in np.unique(months_tr):
            idx = np.where(months_tr == m)[0]
            if len(idx) < 25:
                continue
            pi0 = gbif.loc[int(m)].to_numpy(dtype=np.float64) if int(m) in gbif.index else pi_train
            q_m, pi_m, n_it = em_mlls_month(p_oof[idx], pi_train, pi0, alpha=0.85)
            p_adj[idx] = q_m
            print(f"  month={int(m):02d} | n={len(idx):4d} | it={n_it:3d} | pi_m={pi_m.round(3)}", flush=True)

        m1, per1 = compute_map(y, p_adj)
        print_results(m1, per1, label="OOF after month-wise EM (alpha=0.85)")
        print(f"\n  OOF delta: {m1 - m0:+.4f}\n", flush=True)
    else:
        print("Missing oof_e50.npy; skipping OOF sanity check.", flush=True)

    # ---- Build submissions by adjusting an existing strong submission ----
    base_csv = ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv"
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing base submission: {base_csv}")

    p_base, _track = load_submission_probs(base_csv)
    months_te = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.to_numpy()

    for alpha in (0.75, 0.95):
        p_out = p_base.copy()
        for m in np.unique(months_te):
            idx = np.where(months_te == m)[0]
            pi0 = gbif.loc[int(m)].to_numpy(dtype=np.float64) if int(m) in gbif.index else pi_train
            # Mild shrinkage toward train prior for stability
            pi0 = 0.70 * pi0 + 0.30 * pi_train
            q_m, pi_m, n_it = em_mlls_month(p_base[idx], pi_train, pi0, alpha=float(alpha))
            p_out[idx] = q_m
            print(f"[test] month={int(m):02d} | n={len(idx):4d} | it={n_it:3d} | alpha={alpha:.2f}", flush=True)

        tag = f"e115_mlls_em_geo5_alpha{alpha:.2f}"
        save_submission(p_out, tag, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

