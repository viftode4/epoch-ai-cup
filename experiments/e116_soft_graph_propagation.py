"""E116: Soft graph label propagation using learned same-flock edge weights.

Goal
----
E113 used hard connected components + uniform averaging inside each component.
That can *over-smooth* and mix species when a component contains a few wrong edges.

This experiment keeps the same overall idea (infer flock/session structure from
relative features) but switches to:
  - edge-probability weights (soft)
  - uncertainty gating (only update low-margin nodes)
  - neighbor-confidence gating (only listen to high-confidence neighbors)

We apply propagation on top of the strongest existing submission (E111 geo5).
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def top2_margin(p: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, axis=1)
    p1 = p[np.arange(len(p)), order[:, 0]]
    p2 = p[np.arange(len(p)), order[:, 1]]
    return p1 - p2


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    # Small-distance safe haversine in meters
    r = 6371000.0
    x1 = np.radians(lon1)
    y1 = np.radians(lat1)
    x2 = np.radians(lon2)
    y2 = np.radians(lat2)
    dx = x2 - x1
    dy = y2 - y1
    a = np.sin(dy / 2.0) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _heading_R(lons: np.ndarray, lats: np.ndarray) -> float:
    if len(lons) < 6:
        return 0.0
    # meters-ish; absolute scale doesn't matter for atan2
    dx = np.diff(lons) * 67000.0
    dy = np.diff(lats) * 111000.0
    h = np.arctan2(dy, dx)
    if len(h) < 2:
        return 0.0
    s = float(np.mean(np.sin(h)))
    c = float(np.mean(np.cos(h)))
    R = float(np.sqrt(s * s + c * c))
    return float(R if np.isfinite(R) else 0.0)


def _rcs_ac1(rcs_db: np.ndarray) -> float:
    if len(rcs_db) < 6:
        return 0.0
    z = rcs_db - float(np.mean(rcs_db))
    v = float(np.var(z))
    if v <= 1e-12:
        return 0.0
    ac1 = float(np.mean(z[:-1] * z[1:]) / v)
    return float(ac1 if np.isfinite(ac1) else 0.0)


@dataclass
class TrackSummary:
    time_s: float
    lon: float
    lat: float
    alt_mid: float
    alt_range: float
    rcs_mean: float
    rcs_std: float
    heading_R: float
    rcs_ac1: float
    size_code: float


SIZE_MAP = {"Small bird": 0.0, "Medium": 1.0, "Large": 2.0, "Flock": 3.0}


def summarize_tracks(df: pd.DataFrame) -> list[TrackSummary]:
    out: list[TrackSummary] = []
    ts = pd.to_datetime(df["timestamp_start_radar_utc"]).astype("int64") / 1e9  # seconds
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Summaries: {i}/{len(df)}", flush=True)
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            if len(pts) < 6:
                # fall back to tabular-only
                out.append(
                    TrackSummary(
                        time_s=float(ts.iloc[i]),
                        lon=0.0,
                        lat=0.0,
                        alt_mid=float(0.5 * (row["min_z"] + row["max_z"])),
                        alt_range=float(row["max_z"] - row["min_z"]),
                        rcs_mean=0.0,
                        rcs_std=0.0,
                        heading_R=0.0,
                        rcs_ac1=0.0,
                        size_code=float(SIZE_MAP.get(str(row.get("radar_bird_size", "")), 1.0)),
                    )
                )
                continue

            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)
            alts = np.array([p[2] for p in pts], dtype=float)
            rcs = np.array([p[3] for p in pts], dtype=float)

            out.append(
                TrackSummary(
                    time_s=float(ts.iloc[i]),
                    lon=float(np.mean(lons)),
                    lat=float(np.mean(lats)),
                    alt_mid=float(0.5 * (np.min(alts) + np.max(alts))),
                    alt_range=float(np.max(alts) - np.min(alts)),
                    rcs_mean=float(np.mean(rcs)),
                    rcs_std=float(np.std(rcs)),
                    heading_R=_heading_R(lons, lats),
                    rcs_ac1=_rcs_ac1(rcs),
                    size_code=float(SIZE_MAP.get(str(row.get("radar_bird_size", "")), 1.0)),
                )
            )
        except Exception:
            out.append(
                TrackSummary(
                    time_s=float(ts.iloc[i]),
                    lon=0.0,
                    lat=0.0,
                    alt_mid=float(0.5 * (row["min_z"] + row["max_z"])),
                    alt_range=float(row["max_z"] - row["min_z"]),
                    rcs_mean=0.0,
                    rcs_std=0.0,
                    heading_R=0.0,
                    rcs_ac1=0.0,
                    size_code=float(SIZE_MAP.get(str(row.get("radar_bird_size", "")), 1.0)),
                )
            )
    print(f"  Summaries: {len(df)}/{len(df)} done", flush=True)
    return out


def build_pairs(
    sums: list[TrackSummary],
    ids: np.ndarray,
    *,
    dt_max_s: float = 120.0,
    max_neighbors: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pair_features, i_idx, j_idx) for close-by candidate pairs.

    We use a time-window + limited nearest neighbors by distance to keep O(N) small.
    """
    n = len(sums)
    order = np.argsort([s.time_s for s in sums])
    i_idx = []
    j_idx = []
    feats = []

    # Precompute to arrays for speed
    t = np.array([s.time_s for s in sums], dtype=np.float64)[order]
    lon = np.array([s.lon for s in sums], dtype=np.float64)[order]
    lat = np.array([s.lat for s in sums], dtype=np.float64)[order]
    alt_mid = np.array([s.alt_mid for s in sums], dtype=np.float64)[order]
    alt_rng = np.array([s.alt_range for s in sums], dtype=np.float64)[order]
    rcs_mu = np.array([s.rcs_mean for s in sums], dtype=np.float64)[order]
    rcs_sd = np.array([s.rcs_std for s in sums], dtype=np.float64)[order]
    hR = np.array([s.heading_R for s in sums], dtype=np.float64)[order]
    ac1 = np.array([s.rcs_ac1 for s in sums], dtype=np.float64)[order]
    sz = np.array([s.size_code for s in sums], dtype=np.float64)[order]

    for a in range(n):
        # look forward in time window
        b = a + 1
        cand = []
        while b < n and (t[b] - t[a]) <= dt_max_s:
            # compute rough geo dist (meters)
            d = haversine_m(lon[a], lat[a], lon[b], lat[b])
            cand.append((d, b))
            b += 1

        if not cand:
            continue
        cand.sort(key=lambda x: x[0])
        cand = cand[:max_neighbors]
        for d, b_idx in cand:
            ai = int(order[a])
            bj = int(order[b_idx])
            i_idx.append(ai)
            j_idx.append(bj)
            feats.append(
                [
                    float(t[b_idx] - t[a]),
                    float(d),
                    float(abs(alt_mid[b_idx] - alt_mid[a])),
                    float(abs(alt_rng[b_idx] - alt_rng[a])),
                    float(abs(rcs_mu[b_idx] - rcs_mu[a])),
                    float(abs(rcs_sd[b_idx] - rcs_sd[a])),
                    float(abs(hR[b_idx] - hR[a])),
                    float(abs(ac1[b_idx] - ac1[a])),
                    float(abs(sz[b_idx] - sz[a])),
                ]
            )

    return np.asarray(feats, dtype=np.float32), np.asarray(i_idx, dtype=np.int32), np.asarray(j_idx, dtype=np.int32)


def load_submission_probs(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(CLASSES):
        p[:, j] = df[cls].to_numpy(dtype=np.float32)
    return renorm_rows(p)


def propagate(
    p0: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    w: np.ndarray,
    *,
    beta: float = 0.55,
    iters: int = 5,
    tau_update: float = 0.08,
    tau_neighbor: float = 0.18,
) -> np.ndarray:
    p = p0.copy()
    n = len(p)

    # adjacency lists
    adj_i: list[list[int]] = [[] for _ in range(n)]
    adj_w: list[list[float]] = [[] for _ in range(n)]
    for a, b, ww in zip(i_idx, j_idx, w):
        if ww <= 0:
            continue
        adj_i[a].append(int(b))
        adj_w[a].append(float(ww))
        adj_i[b].append(int(a))
        adj_w[b].append(float(ww))

    for _ in range(iters):
        margin = top2_margin(p)
        upd = margin < tau_update
        p_new = p.copy()
        for i in np.where(upd)[0]:
            nb = adj_i[i]
            if not nb:
                continue
            ww = np.asarray(adj_w[i], dtype=np.float32)
            nb = np.asarray(nb, dtype=np.int32)

            # Only listen to confident neighbors
            nb_margin = margin[nb]
            keep = nb_margin > tau_neighbor
            if keep.sum() == 0:
                continue
            nb = nb[keep]
            ww = ww[keep]
            ww = ww / max(float(ww.sum()), 1e-12)
            p_nb = (p[nb] * ww[:, None]).sum(axis=0)
            p_new[i] = renorm_rows(((1.0 - beta) * p[i] + beta * p_nb)[None, :])[0]
        p = p_new
    return p


def main() -> None:
    print("=" * 70, flush=True)
    print("E116 SOFT GRAPH PROPAGATION".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    # ---- Train edge model on train (primary_observation_id) ----
    print("\nBuilding train summaries...", flush=True)
    sums_tr = summarize_tracks(train_df)
    pid = train_df["primary_observation_id"].to_numpy()
    track_ids_tr = train_df["track_id"].to_numpy()

    print("\nBuilding train candidate pairs...", flush=True)
    Xp, ii, jj = build_pairs(sums_tr, track_ids_tr, dt_max_s=120.0, max_neighbors=14)
    y_pair = (pid[ii] == pid[jj]).astype(int)
    print(f"  Pairs: {len(y_pair)} | positive: {int(y_pair.sum())} ({100*y_pair.mean():.1f}%)", flush=True)

    # Cross-fit AUC to sanity-check (fast)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_pair), dtype=np.float32)
    for k, (tr, va) in enumerate(skf.split(Xp, y_pair)):
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.07,
            max_iter=250,
            l2_regularization=0.1,
            min_samples_leaf=30,
            random_state=42 + k,
        )
        clf.fit(Xp[tr], y_pair[tr])
        oof[va] = clf.predict_proba(Xp[va])[:, 1]
    auc = roc_auc_score(y_pair, oof)
    print(f"  Edge model OOF AUC: {auc:.4f}", flush=True)

    # Fit final edge model on all pairs
    edge = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.07,
        max_iter=350,
        l2_regularization=0.1,
        min_samples_leaf=30,
        random_state=42,
    )
    edge.fit(Xp, y_pair)

    # ---- Build test graph + propagate on top of E111 ----
    base_csv = ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv"
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing base submission: {base_csv}")
    p0 = load_submission_probs(base_csv)

    print("\nBuilding test summaries...", flush=True)
    sums_te = summarize_tracks(test_df)
    track_ids_te = test_df["track_id"].to_numpy()

    print("\nBuilding test candidate pairs...", flush=True)
    Xp_te, ii_te, jj_te = build_pairs(sums_te, track_ids_te, dt_max_s=120.0, max_neighbors=16)
    w = edge.predict_proba(Xp_te)[:, 1].astype(np.float32)
    keep = w >= 0.62
    print(f"  Test pairs: {len(w)} | kept edges: {int(keep.sum())}", flush=True)

    p_prop = propagate(
        p0,
        ii_te[keep],
        jj_te[keep],
        w[keep],
        beta=0.55,
        iters=6,
        tau_update=0.08,
        tau_neighbor=0.18,
    )

    save_submission(p_prop, "e116_soft_graph_prop_geo5", cv_map=None)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

