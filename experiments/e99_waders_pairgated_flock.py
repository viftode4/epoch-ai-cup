"""E99: Pair-gated flock evidence for Waders confusions (Option A).

Motivation
----------
E98 added flock/RCS distribution evidence globally (under the usual unseen+margin gate) and
matched best LB (0.59) but did not improve. The most likely explanation is that flock evidence
is only relevant for *specific confusion pairs* (notably Waders vs large waterbirds), so global
application creates small regressions that cancel gains in macro-mAP.

E99 makes the correction conditional:
  - Apply the proven pipeline first (priors -> baseline NB evidence).
  - Then apply *additional* flock evidence only when:
      month in {2,5,12} (unseen),
      margin < tau (still uncertain),
      top-2 predicted classes include Waders and one of {Geese, Gulls, Ducks}.

This concentrates the correction on cases where flockness is plausibly discriminative.
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

UNSEEN_MONTHS = (2, 5, 12)

# Priors stage (best known)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Baseline evidence stage (as in E96/E98)
TAU_NB = 0.25
GAMMA_BASE = 0.10

# Evidence weights (keep E78 insight: alt_range partial)
W_SPEED = 1.00
W_ALTMID = 1.00
W_ALTRANGE = 0.50
W_HEADING = 1.00
W_AC1 = 1.00

# Flock evidence weights (conservative)
W_SI = 0.50
W_FADE = 0.50
W_KURT = 0.25

LAPLACE = 1.0

DEFAULT_MIN_SIGMA = 0.50
MIN_SIGMA = {
    "speed": 0.50,
    "alt_mid": 0.50,
    "alt_range": 0.50,
    "heading_R": 0.10,
    "rcs_ac1": 0.10,
    "rcs_si": 0.10,
    "rcs_fade_frac": 0.05,
    "rcs_kurt": 0.50,
}


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_info(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (top2_idx, top2_margin) for each row."""
    order = np.argsort(-pred, axis=1)
    top2 = order[:, :2]
    p1 = pred[np.arange(pred.shape[0]), top2[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), top2[:, 1]]
    return top2, (p1 - p2)


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
    _, margin = top2_info(out)
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


def extract_track_evidence(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    heading_r = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)
    si = np.full(n, np.nan)
    fade = np.full(n, np.nan)
    kurt = np.full(n, np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Evidence extraction: {i}/{n}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 6:
                continue

            rcs = np.array([p[3] for p in pts], dtype=float)
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            # heading_R
            times = parse_trajectory_time(row["trajectory_time"])
            _ = np.diff(times)
            dx = np.diff(lons) * 67000.0
            dy = np.diff(lats) * 111000.0
            headings = np.arctan2(dy, dx)
            if len(headings) > 1:
                R = float(np.sqrt(np.mean(np.sin(headings)) ** 2 + np.mean(np.cos(headings)) ** 2))
                if np.isfinite(R):
                    heading_r[i] = R

            # rcs_ac1
            rcs_c = rcs - float(np.mean(rcs))
            var_rcs = float(np.var(rcs_c))
            if var_rcs > 1e-12:
                ac1_val = float(np.mean(rcs_c[:-1] * rcs_c[1:]) / var_rcs)
                if np.isfinite(ac1_val):
                    ac1[i] = ac1_val

            # flockness stats on linear scale
            rcs_lin = 10.0 ** (rcs / 10.0)
            m = float(np.mean(rcs_lin))
            v = float(np.var(rcs_lin))
            if np.isfinite(m) and m > 1e-12 and np.isfinite(v):
                si_val = v / (m * m)
                si[i] = float(np.clip(si_val, 0.0, 10.0))
                fade[i] = float(np.mean(rcs_lin < (0.1 * m)))

            # excess kurtosis in dB space (clipped)
            if len(rcs) >= 8 and np.std(rcs) > 1e-6:
                z = (rcs - float(np.mean(rcs))) / float(np.std(rcs))
                k = float(np.mean(z**4) - 3.0)
                kurt[i] = float(np.clip(k, -5.0, 25.0))
        except Exception:
            continue

    ok = (
        np.isfinite(heading_r)
        & np.isfinite(ac1)
        & np.isfinite(si)
        & np.isfinite(fade)
        & np.isfinite(kurt)
    )
    print(f"  Evidence valid: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%)", flush=True)

    def _fill(x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, 0.0)

    return {
        "heading_R": _fill(heading_r),
        "rcs_ac1": _fill(ac1),
        "rcs_si": _fill(si),
        "rcs_fade_frac": _fill(fade),
        "rcs_kurt": _fill(kurt),
        "ok": ok,
    }


def log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_nb_params(
    df: pd.DataFrame,
    y: np.ndarray,
    ev: dict[str, np.ndarray],
    include_flock: bool,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )

    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z

    feats: dict[str, np.ndarray] = {
        "speed": speed,
        "alt_mid": alt_mid,
        "alt_range": alt_range,
        "heading_R": ev["heading_R"],
        "rcs_ac1": ev["rcs_ac1"],
    }
    if include_flock:
        feats["rcs_si"] = ev["rcs_si"]
        feats["rcs_fade_frac"] = ev["rcs_fade_frac"]
        feats["rcs_kurt"] = ev["rcs_kurt"]

    # P(size|class)
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

    mu: dict[str, np.ndarray] = {}
    sig: dict[str, np.ndarray] = {}
    ok = ev["ok"]
    for feat_name, x in feats.items():
        min_s = MIN_SIGMA.get(feat_name, DEFAULT_MIN_SIGMA)
        if feat_name in ("heading_R", "rcs_ac1", "rcs_si", "rcs_fade_frac", "rcs_kurt"):
            x_use = np.where(ok, x, np.nan)
        else:
            x_use = x

        gm = float(np.nanmean(x_use))
        gs = float(np.nanstd(x_use))
        if not np.isfinite(gs) or gs < min_s:
            gs = min_s

        mu_f = np.full(K, gm, dtype=float)
        sig_f = np.full(K, gs, dtype=float)
        for c in range(K):
            xc = x_use[y == c]
            ok_c = np.isfinite(xc)
            if ok_c.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > min_s else min_s

        mu[feat_name] = mu_f
        sig[feat_name] = sig_f

    return size_levels, log_p_size, mu, sig


def loglike_nb(
    df: pd.DataFrame,
    ev: dict[str, np.ndarray],
    size_levels: list[str],
    log_p_size: np.ndarray,
    mu: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
    include_flock: bool,
) -> np.ndarray:
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"]))
        .values.astype(int)
    )
    loglike = log_p_size[:, size_idx].T  # (n, K)

    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    ok = ev["ok"]

    channels: list[tuple[str, np.ndarray, float, np.ndarray]] = [
        ("speed", speed, W_SPEED, np.isfinite(speed)),
        ("alt_mid", alt_mid, W_ALTMID, np.isfinite(alt_mid)),
        ("alt_range", alt_range, W_ALTRANGE, np.isfinite(alt_range)),
        ("heading_R", ev["heading_R"], W_HEADING, ok),
        ("rcs_ac1", ev["rcs_ac1"], W_AC1, ok),
    ]
    if include_flock:
        channels.extend(
            [
                ("rcs_si", ev["rcs_si"], W_SI, ok),
                ("rcs_fade_frac", ev["rcs_fade_frac"], W_FADE, ok),
                ("rcs_kurt", ev["rcs_kurt"], W_KURT, ok),
            ]
        )

    for feat_name, x, w, valid0 in channels:
        if w == 0:
            continue
        valid = valid0 & np.isfinite(x)
        if valid.sum() == 0:
            continue
        lg = log_gaussian(np.where(np.isfinite(x), x, 0.0), mu[feat_name], sig[feat_name])
        loglike[valid] += w * lg[valid]

    return loglike


def apply_poe(base: np.ndarray, loglike: np.ndarray, gate: np.ndarray, gamma: float) -> np.ndarray:
    out = base.copy()
    if gate.sum() == 0:
        return renorm_rows(out)
    ll = loglike[gate]
    ll = ll - ll.max(axis=1, keepdims=True)
    fac = np.exp(np.clip(gamma * ll, -50.0, 50.0))
    out[gate] = out[gate] * fac
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def main() -> None:
    print("=" * 70, flush=True)
    print("E99 WADERS PAIR-GATED FLOCK EVIDENCE".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)

    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
    print(f"\nBase preds: test_e50.npy shape={test_base.shape}", flush=True)

    print("\nExtracting evidence on train...", flush=True)
    tr_ev = extract_track_evidence(train_df)
    print("\nExtracting evidence on test...", flush=True)
    te_ev = extract_track_evidence(test_df)

    # Stage 1: priors
    test_p0, changed = apply_gated_ratio_priors(
        test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

    # Stage 2: baseline evidence (no flock stats)
    top2_0, margin0 = top2_info(test_p0)
    gate0 = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
    print(f"Baseline evidence gate: unseen only, tau={TAU_NB:.2f} rows={int(gate0.sum())}", flush=True)

    size_levels0, log_p_size0, mu0, sig0 = build_nb_params(train_df, y, tr_ev, include_flock=False)
    ll0 = loglike_nb(test_df, te_ev, size_levels0, log_p_size0, mu0, sig0, include_flock=False)
    test_p1 = apply_poe(test_p0, ll0, gate0, gamma=GAMMA_BASE)

    # Stage 3: additional flock evidence (pair-gated, Option A)
    waders_idx = CLASSES.index("Waders")
    other_set = {CLASSES.index("Geese"), CLASSES.index("Gulls"), CLASSES.index("Ducks")}

    top2_1, margin1 = top2_info(test_p1)
    has_waders = (top2_1[:, 0] == waders_idx) | (top2_1[:, 1] == waders_idx)
    other = np.where(top2_1[:, 0] == waders_idx, top2_1[:, 1], top2_1[:, 0])
    is_pair = np.array([o in other_set for o in other], dtype=bool)

    gate_flock = np.isin(test_months, UNSEEN_MONTHS) & (margin1 < TAU_NB) & has_waders & is_pair
    print(
        f"Flock evidence gate: unseen only, tau={TAU_NB:.2f}, "
        f"waders-pair rows={int(gate_flock.sum())}",
        flush=True,
    )

    size_levelsF, log_p_sizeF, muF, sigF = build_nb_params(train_df, y, tr_ev, include_flock=True)
    llF = loglike_nb(test_df, te_ev, size_levelsF, log_p_sizeF, muF, sigF, include_flock=True)

    # We only want the incremental flock terms. Compute baseline ll again but with same size_levels mapping.
    ll_base_same = loglike_nb(
        test_df, te_ev, size_levelsF, log_p_sizeF, muF, sigF, include_flock=False
    )
    ll_delta = llF - ll_base_same

    for gamma_flock in (0.10, 0.15):
        out = apply_poe(test_p1, ll_delta, gate_flock, gamma=gamma_flock)
        name = (
            f"e99_waderspair_flock_tau{TAU_NB:.2f}_gb{GAMMA_BASE:.2f}_gf{gamma_flock:.2f}_"
            f"waltR{W_ALTRANGE:.2f}_priortau{TAU_PRIOR:.2f}"
        )
        save_submission(out, name, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

