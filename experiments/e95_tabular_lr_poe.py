"""E95: Likelihood-ratio PoE from discriminative tabular model.

E94 failed because it used r(y|u) directly in a product-of-experts, which
double-counts class priors under shift. The correct Bayesian object is a
likelihood ratio:

  r(y|u) ∝ π_train(y) P(u|y)  =>  P(u|y) ∝ r(y|u) / π_train(y).

We therefore define an evidence factor:
  L_{i,c} = clip( r_{i,c} / π_train(c) )
and apply:
  q ∝ p^(m) ⊙ L^λ

As usual:
  - p^(m) is E67 gated month-prior ratio tilt (unseen months only)
  - evidence update applies on unseen months only and is uncertainty gated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.metrics import compute_map  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

UNSEEN_MONTHS = (2, 5, 12)

# Priors stage (fixed)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Evidence stage candidates (small set for Kaggle probing)
TAU_NB = 0.30
LAMBDAS = [0.20, 0.30]

# Likelihood ratio clipping (avoid extreme multipliers)
LR_MIN = 0.25
LR_MAX = 4.00


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
    si = {}
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

    priors = {}
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


def poe_update(base: np.ndarray, L: np.ndarray, lam: float, gate: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[gate] = out[gate] * (L[gate] ** lam)
    out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    return renorm_rows(out)


def build_u(df: pd.DataFrame) -> pd.DataFrame:
    u = pd.DataFrame(index=df.index)
    u["radar_bird_size"] = df["radar_bird_size"].astype(str)
    u["airspeed"] = pd.to_numeric(df["airspeed"], errors="coerce")
    min_z = pd.to_numeric(df["min_z"], errors="coerce")
    max_z = pd.to_numeric(df["max_z"], errors="coerce")
    u["alt_mid"] = 0.5 * (min_z + max_z)
    u["alt_range"] = max_z - min_z
    return u.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_r_model() -> Pipeline:
    cat_cols = ["radar_bird_size"]
    num_cols = ["airspeed", "alt_mid", "alt_range"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


print("=" * 70, flush=True)
print("E95 TABULAR LR-PoE (DISCRIMINATIVE -> LIKELIHOOD RATIO)".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

oof_base = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))

u_train = build_u(train_df)
u_test = build_u(test_df)

base_map, _ = compute_map(y, oof_base)
print(f"\nBase oof_e50 mAP: {base_map:.4f}", flush=True)

# Train r(y|u) on all train (keep it simple; tuning is done on Kaggle).
model = make_r_model()
model.fit(u_train, y)
r_test = np.clip(model.predict_proba(u_test), 1e-12, None)

# Convert to likelihood ratio L ∝ r / pi_train and clip.
L = r_test / np.maximum(p_train[None, :], 1e-12)
L = np.clip(L, LR_MIN, LR_MAX)

# Apply priors first.
test_p0, changed = apply_gated_ratio_priors(
    test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
)
print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

margin0 = top2_margin(test_p0)
gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
print(f"Evidence gate (unseen only): tau_nb={TAU_NB:.2f} rows={int(gate.sum())}", flush=True)

for lam in LAMBDAS:
    out = poe_update(test_p0, L, lam=lam, gate=gate)
    save_submission(
        out,
        f"e95_lrpoe_tau{TAU_NB:.2f}_lam{lam:.2f}_lrclip{LR_MIN:.2f}-{LR_MAX:.2f}_priortau{TAU_PRIOR:.2f}",
        cv_map=None,
    )

print("\nDone.", flush=True)

