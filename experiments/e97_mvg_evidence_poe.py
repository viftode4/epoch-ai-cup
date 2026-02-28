"""E97: Correlated evidence PoE (multivariate Gaussian) on unseen months.

Goal
----
We are currently at a Kaggle LB plateau around 0.59 using:
  base preds (trees) -> gated GBIF month-prior ratio tilt -> gated NB evidence update.

The NB step assumes conditional independence across evidence channels u, but our
channels are correlated (e.g. altitude mid vs range; heading consistency vs speed;
RCS texture vs size). Under correlation, NB effectively double-counts evidence and
forces us to use very small gamma to avoid over-sharpening.

This experiment replaces the factorized Gaussian NB likelihood with a *regularized
multivariate Gaussian* likelihood for the continuous evidence vector, while keeping
the same prior-tilt stage and the same uncertainty gating strategy.

Pipeline
--------
1) Start from base predictions (oof_e50/test_e50).
2) Apply E67-style gated GBIF ratio priors on unseen months only.
3) Apply a gated PoE update using evidence:
     - categorical: radar_bird_size
     - continuous: (airspeed, alt_mid, alt_range, heading_R, rcs_ac1)
   but model the continuous part with a shared-covariance multivariate Gaussian.

Outputs
-------
Saves one (or more) submission(s) in submissions/ and writes submission.csv at root.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# Domain structure
UNSEEN_MONTHS = (2, 5, 12)

# Priors stage (best known LB-performing settings)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Evidence stage (gate on uncertainty; apply only to unseen months)
TAU_EVID = 0.25

# Evidence weights: keep E78 insight (alt_range partially redundant/noisy)
W_SPEED = 1.00
W_ALTMID = 1.00
W_ALTRANGE = 0.50
W_HEADING = 1.00
W_AC1 = 1.00

LAPLACE = 1.0


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si: dict[int, np.ndarray] = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(len(CLASSES))
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals

    priors: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_gated_ratio_priors(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
    tau: float,
) -> tuple[np.ndarray, int]:
    out = preds.copy()
    margin = top2_margin(out)
    changed = 0
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


def extract_heading_ac1(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (heading_R, rcs_ac1, ok_mask) from raw trajectories.

    ok_mask is True when we computed both values from >= 6 points.
    """
    n = len(df)
    heading_r = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            npts = len(pts)
            if npts < 6:
                continue

            rcs = np.array([p[3] for p in pts], dtype=float)
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            # RCS autocorrelation lag-1 (centered)
            rcs_c = rcs - float(np.mean(rcs))
            var_rcs = float(np.var(rcs_c))
            if var_rcs > 1e-12:
                ac1_val = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs)
                if np.isfinite(ac1_val):
                    ac1[i] = ac1_val

            # Heading consistency (circular resultant length R)
            times = parse_trajectory_time(row["trajectory_time"])
            _ = np.diff(times)  # ensure it parses; dt not used further
            # Scale degrees -> meters approximately; only direction matters for R.
            dx = np.diff(lons) * 67000.0
            dy = np.diff(lats) * 111000.0
            headings = np.arctan2(dy, dx)
            if len(headings) > 1:
                R = float(np.sqrt(np.mean(np.sin(headings)) ** 2 + np.mean(np.cos(headings)) ** 2))
                if np.isfinite(R):
                    heading_r[i] = R
        except Exception:
            continue

    ok = np.isfinite(heading_r) & np.isfinite(ac1)
    heading_r = np.where(np.isfinite(heading_r), heading_r, 0.0)
    ac1 = np.where(np.isfinite(ac1), ac1, 0.0)
    print(f"  Evidence valid: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)
    return heading_r, ac1, ok


def build_size_logprob(train_df: pd.DataFrame, y: np.ndarray) -> tuple[list[str], np.ndarray]:
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        train_df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )

    K, S = N_CLASSES, len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y == c
        counts_c[c] = float(mask.sum())
        if counts_c[c] > 0:
            counts_cs[c] = np.bincount(size_idx[mask], minlength=S).astype(float)

    p_size = (counts_cs + LAPLACE) / np.clip(counts_c[:, None] + LAPLACE * S, 1e-12, None)
    log_p_size = np.log(np.clip(p_size, 1e-12, None))
    return size_levels, log_p_size


def _standardize_fit(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(U, axis=0)
    sig = np.nanstd(U, axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 1e-6), sig, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, sig


def _standardize_apply(U: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (U - mu[None, :]) / sig[None, :]


def fit_shared_cov_mvg(
    train_df: pd.DataFrame,
    y: np.ndarray,
    heading_r: np.ndarray,
    ac1: np.ndarray,
    ok: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit a shared-covariance multivariate Gaussian evidence model.

    We fit on rows with valid trajectory evidence (ok==True), and we standardize
    channels globally to stabilize estimation. We incorporate feature weights by
    scaling the corresponding columns by sqrt(weight).
    """
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    # Apply sqrt-weights via scaling
    U = np.column_stack(
        [
            np.sqrt(W_SPEED) * speed,
            np.sqrt(W_ALTMID) * alt_mid,
            np.sqrt(W_ALTRANGE) * alt_range,
            np.sqrt(W_HEADING) * heading_r,
            np.sqrt(W_AC1) * ac1,
        ]
    ).astype(float)

    valid = ok & np.all(np.isfinite(U), axis=1)
    Uv = U[valid]
    yv = y[valid]
    if Uv.shape[0] < 200:
        raise RuntimeError(f"Too few valid evidence rows to fit MVG: {Uv.shape[0]}")

    mu0, sig0 = _standardize_fit(Uv)
    Z = _standardize_apply(Uv, mu0, sig0)

    # Shrinkage covariance (Ledoit-Wolf) for stability under correlation
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(Z)
    cov = lw.covariance_.astype(float)
    cov = cov + 1e-6 * np.eye(cov.shape[0])  # numerical safety
    inv_cov = np.linalg.inv(cov)

    # Class means in standardized space
    mu_c = np.zeros((N_CLASSES, Z.shape[1]), dtype=float)
    global_mean = Z.mean(axis=0)
    for c in range(N_CLASSES):
        zc = Z[yv == c]
        if zc.shape[0] >= 10:
            mu_c[c] = zc.mean(axis=0)
        else:
            mu_c[c] = global_mean

    return {
        "mu0": mu0,
        "sig0": sig0,
        "mu_c": mu_c,
        "inv_cov": inv_cov,
    }


def loglik_mvg(
    model: dict[str, np.ndarray],
    df: pd.DataFrame,
    heading_r: np.ndarray,
    ac1: np.ndarray,
    ok: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row per-class MVG log-likelihood (up to a constant).

    Returns:
      loglik: (n, K)
      valid:  (n,) boolean rows with finite evidence
    """
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    U = np.column_stack(
        [
            np.sqrt(W_SPEED) * speed,
            np.sqrt(W_ALTMID) * alt_mid,
            np.sqrt(W_ALTRANGE) * alt_range,
            np.sqrt(W_HEADING) * heading_r,
            np.sqrt(W_AC1) * ac1,
        ]
    ).astype(float)

    valid = ok & np.all(np.isfinite(U), axis=1)
    Z = _standardize_apply(U, model["mu0"], model["sig0"])

    mu_c = model["mu_c"]  # (K, d)
    inv_cov = model["inv_cov"]  # (d, d)
    n, d = Z.shape
    K = mu_c.shape[0]

    # Quadratic form: (z - mu_c)^T inv_cov (z - mu_c)
    loglik = np.zeros((n, K), dtype=float)
    for c in range(K):
        diff = Z - mu_c[c][None, :]
        # row-wise quadratic: sum((diff @ inv_cov) * diff, axis=1)
        q = np.sum((diff @ inv_cov) * diff, axis=1)
        loglik[:, c] = -0.5 * q
    return loglik, valid


def apply_poe_update(
    base: np.ndarray,
    log_p_size: np.ndarray,
    size_levels: list[str],
    loglik_cont: np.ndarray,
    gamma: float,
    gate: np.ndarray,
    df: pd.DataFrame,
) -> np.ndarray:
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)

    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"]
        .fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )
    loglike = log_p_size[:, size_idx].T + loglik_cont  # (n, K)

    # Stabilize: subtract row max before exponentiating (constant cancels in renorm)
    ll = loglike[gate]
    ll = ll - ll.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def main() -> None:
    print("=" * 70, flush=True)
    print("E97 MV-GAUSS EVIDENCE PoE (UNSEEN ONLY)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    # Base predictions
    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
    print(f"\nBase preds: test_e50.npy shape={test_base.shape}", flush=True)

    # Evidence extraction
    print("\nExtracting trajectory evidence on train...", flush=True)
    tr_heading, tr_ac1, tr_ok = extract_heading_ac1(train_df)
    print("\nExtracting trajectory evidence on test...", flush=True)
    te_heading, te_ac1, te_ok = extract_heading_ac1(test_df)

    # Priors stage (unseen only)
    test_p0, changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

    # Fit MVG evidence model on train
    print("\nFitting shared-cov MVG evidence model...", flush=True)
    size_levels, log_p_size = build_size_logprob(train_df, y)
    model = fit_shared_cov_mvg(train_df, y, tr_heading, tr_ac1, tr_ok)
    loglik_test, valid_test = loglik_mvg(model, test_df, te_heading, te_ac1, te_ok)

    # Gate: unseen months + uncertain after priors + valid evidence
    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_EVID) & valid_test
    print(
        f"Evidence gate: unseen-only, tau_evid={TAU_EVID:.2f}, "
        f"valid={int(valid_test.sum())}, rows={int(gate.sum())}",
        flush=True,
    )

    # Candidate gammas: slightly larger than NB is plausible due to correlation modeling.
    for gamma in (0.15, 0.25):
        out = apply_poe_update(
            test_p0,
            log_p_size=log_p_size,
            size_levels=size_levels,
            loglik_cont=loglik_test,
            gamma=gamma,
            gate=gate,
            df=test_df,
        )
        name = (
            f"e97_mvgpue_tau{TAU_EVID:.2f}_g{gamma:.2f}_"
            f"waltR{W_ALTRANGE:.2f}_priortau{TAU_PRIOR:.2f}"
        )
        save_submission(out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

