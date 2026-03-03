"""E117: External flight-style teacher (Col de la Croix 1988) as a PoE expert.

Why this might break 0.59
------------------------
We keep failing to extract robust "bounding flight" / flight-style signals with
deep models trained on our small dataset (overfit + shift).

Instead, we use an *external labeled radar dataset* (Col de la Croix 1988) that
explicitly encodes wingbeat-pattern classes including passerine bounding flight,
wader continuous flapping, raptor, flock.

We train a simple multiclass teacher to predict flight-style from coarse physics
features available in BOTH datasets:
  - mean altitude
  - mean ground speed
  - mean airspeed
  - mean climb rate

Then we transfer to the competition by treating the teacher outputs as an
additional "expert" and applying a conservative gated Product-of-Experts update
to an already-strong baseline submission (E111 geo5).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time  # noqa: E402
from src.metrics import compute_map, print_results  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def top2_margin(p: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, axis=1)
    p1 = p[np.arange(len(p)), order[:, 0]]
    p2 = p[np.arange(len(p)), order[:, 1]]
    return p1 - p2


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    x1 = np.radians(lon1)
    y1 = np.radians(lat1)
    x2 = np.radians(lon2)
    y2 = np.radians(lat2)
    dx = x2 - x1
    dy = y2 - y1
    a = np.sin(dy / 2.0) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def mean_ground_speed_and_climb(hex_str: str, traj_time_str: str) -> tuple[float, float]:
    """Compute mean ground speed (m/s) and mean climb rate (m/s) from trajectory."""
    pts = parse_ewkb_4d(hex_str)
    t = parse_trajectory_time(traj_time_str).astype(float)
    if len(pts) < 4:
        return 0.0, 0.0
    lons = np.array([p[0] for p in pts], dtype=float)
    lats = np.array([p[1] for p in pts], dtype=float)
    alts = np.array([p[2] for p in pts], dtype=float)

    dt = np.diff(t)
    dt = np.maximum(dt, 1e-3)
    d = np.array([_haversine_m(lons[i], lats[i], lons[i + 1], lats[i + 1]) for i in range(len(pts) - 1)])
    vg = float(np.mean(d / dt))  # m/s
    vz = float(np.mean(np.diff(alts) / dt))  # m/s
    if not np.isfinite(vg):
        vg = 0.0
    if not np.isfinite(vz):
        vz = 0.0
    return vg, vz


def load_col_de_la_croix(path: Path) -> pd.DataFrame:
    """Parse the Col de la Croix CSV (semicolon, header after metadata)."""
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(raw):
        if line.strip().startswith("Code;Date;FNr;Z;Rg;Ra;Vg;Va;Vz;FieldClass"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find Col de la Croix table header.")
    from io import StringIO

    txt = "\n".join(raw[header_idx:])
    df = pd.read_csv(StringIO(txt), sep=";")
    return df


STYLE_NAMES = ["wader_cont", "passerine_bound", "raptor", "flock", "other"]


def map_fieldclass_to_style(fc: int) -> int:
    # 1-2: wader-type (continuous)
    if fc in (1, 2):
        return 0
    # 3-4: passerine-type (bounding)
    if fc in (3, 4):
        return 1
    # 6: raptor
    if fc == 6:
        return 2
    # 8: flock
    if fc == 8:
        return 3
    # 5 swift-type, 7 single large, 9 unknown -> other
    return 4


def train_style_teacher(col_df: pd.DataFrame) -> LGBMClassifier:
    df = col_df.copy()
    df = df[pd.to_numeric(df["FieldClass"], errors="coerce").notna()]
    df["FieldClass"] = df["FieldClass"].astype(int)
    y = df["FieldClass"].apply(map_fieldclass_to_style).to_numpy(dtype=int)

    # Convert speeds from cm/s -> m/s
    z = pd.to_numeric(df["Z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vg = pd.to_numeric(df["Vg"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
    va = pd.to_numeric(df["Va"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
    vz = pd.to_numeric(df["Vz"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0

    X = np.vstack([z, vg, va, vz]).T.astype(np.float32)

    # Class weights via is_unbalance-like behavior: give minorities a chance
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=len(STYLE_NAMES),
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.2,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y)
    return clf


def style_probs_for_df(df: pd.DataFrame, teacher: LGBMClassifier) -> np.ndarray:
    n = len(df)
    X = np.zeros((n, 4), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"  Style features: {i}/{n}", flush=True)
        vg, vz = mean_ground_speed_and_climb(row["trajectory"], row["trajectory_time"])
        alt_mid = 0.5 * float(row["min_z"] + row["max_z"])
        va = float(row["airspeed"]) if np.isfinite(row["airspeed"]) else 0.0
        X[i] = [alt_mid, vg, va, vz]
    print(f"  Style features: {n}/{n} done", flush=True)
    p = teacher.predict_proba(X)
    return np.asarray(p, dtype=np.float32)


def load_submission_probs(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(CLASSES):
        p[:, j] = df[cls].to_numpy(dtype=np.float32)
    return renorm_rows(p)


def build_class_style_prototypes(train_style: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mu[c,k] = mean P(style=k | x) over class c."""
    n_classes = len(CLASSES)
    k = train_style.shape[1]
    mu = np.zeros((n_classes, k), dtype=np.float32)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            mu[c] = 1.0 / k
        else:
            mu[c] = train_style[idx].mean(axis=0)
    mu = mu / np.clip(mu.sum(axis=1, keepdims=True), 1e-12, None)
    return mu


def apply_style_poe(
    p: np.ndarray,
    style_p: np.ndarray,
    mu: np.ndarray,
    *,
    lam: float,
    tau: float,
) -> np.ndarray:
    """q_c ∝ p_c * (eps + <style_p, mu_c>)^lam, gated by margin<tau."""
    out = p.copy().astype(np.float32)
    gate = top2_margin(out) < float(tau)
    if gate.sum() == 0:
        return out

    # evidence score for each class
    # (N, K) dot (C, K)^T -> (N, C)
    e = (style_p @ mu.T).astype(np.float32)
    e = np.clip(e, 1e-4, 10.0)
    out[gate] = out[gate] * (e[gate] ** float(lam))
    out[gate] = renorm_rows(out[gate])
    return out


def main() -> None:
    print("=" * 78, flush=True)
    print("E117: EXTERNAL FLIGHT-STYLE TEACHER (COL DE LA CROIX) POE".center(78), flush=True)
    print("=" * 78, flush=True)

    # --- Train external teacher ---
    col_path = ROOT / "data" / "other_datasets" / "Col de la Croix 1988.csv"
    col_df = load_col_de_la_croix(col_path)
    print(f"\nLoaded Col de la Croix: {len(col_df)} rows", flush=True)
    teacher = train_style_teacher(col_df)

    # --- Compute style probs for competition train/test ---
    train_df = load_train()
    test_df = load_test()
    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)

    print("\nComputing train style probabilities...", flush=True)
    style_tr = style_probs_for_df(train_df, teacher)
    mu = build_class_style_prototypes(style_tr, y)
    print("Class style prototypes (rows=classes, cols=styles):", flush=True)
    for c, name in enumerate(CLASSES):
        v = np.round(mu[c], 3)
        print(f"  {name:15s}: {dict(zip(STYLE_NAMES, v))}", flush=True)

    print("\nComputing test style probabilities...", flush=True)
    style_te = style_probs_for_df(test_df, teacher)

    # --- OOF sanity-check (quick): does this expert help at all? ---
    oof_path = ROOT / "oof_e50.npy"
    if oof_path.exists():
        p_oof = renorm_rows(np.load(oof_path).astype(np.float32))
        m0, per0 = compute_map(y, p_oof)
        print_results(m0, per0, label="Baseline OOF (oof_e50.npy)")

        best = (m0, None)
        for tau in (0.05, 0.08, 0.12, 0.18):
            for lam in (0.10, 0.20, 0.35):
                p_adj = apply_style_poe(p_oof, style_tr, mu, lam=lam, tau=tau)
                m1, _ = compute_map(y, p_adj)
                if m1 > best[0]:
                    best = (m1, (tau, lam))
        if best[1] is not None:
            tau_best, lam_best = best[1]
            print(f"\nBest OOF (grid): {best[0]:.4f} at tau={tau_best:.2f}, lam={lam_best:.2f}", flush=True)
        else:
            tau_best, lam_best = 0.08, 0.20
    else:
        print("Missing oof_e50.npy; using default hyperparams.", flush=True)
        tau_best, lam_best = 0.08, 0.20

    # --- Apply to strong baseline submission (E111 geo5) ---
    base_csv = ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv"
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing base submission: {base_csv}")
    p_base = load_submission_probs(base_csv)

    p_out = apply_style_poe(p_base, style_te, mu, lam=float(lam_best), tau=float(tau_best))
    tag = f"e117_colflightstyle_geo5_tau{tau_best:.2f}_lam{lam_best:.2f}"
    save_submission(p_out, tag, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

