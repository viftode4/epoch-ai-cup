"""E94: Discriminative tabular evidence via product-of-experts (PoE).

Goal
----
Replace the Naive Bayes evidence factor P(u|y) used in E73–E79 with a
discriminative model r(y|u) trained on stable tabular cues u. This avoids the
conditional independence approximation while keeping the same PoE mechanism.

Setup
-----
u := {radar_bird_size (one-hot), airspeed, alt_mid, alt_range}

We build r(y|u) with multinomial logistic regression and apply (on test):
  1) month prior ratio-tilt p -> p^(m) (E67, unseen months only, gated)
  2) evidence update on unseen months only (2/5/12), gated by uncertainty:
       q ∝ p^(m) ⊙ r(y|u)^λ

We pick (tau_nb, lambda) from a small grid using month-wise cross-fitting on
train months (LOMO-like): each train month is held out, r is trained on other
months, and the PoE update is applied only to the held-out month.
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

# Priors stage (fixed, best-known)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

# Evidence stage: tune these
TAUS_NB = [0.20, 0.25, 0.30]
LAMBDAS = [0.15, 0.25, 0.35, 0.50]


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


def poe_update(
    base: np.ndarray,
    r: np.ndarray,
    lam: float,
    gate: np.ndarray,
) -> np.ndarray:
    out = base.copy()
    r = np.clip(r, 1e-12, None)
    out[gate] = out[gate] * (r[gate] ** lam)
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
        n_jobs=None,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


print("=" * 70, flush=True)
print("E94 TABULAR EVIDENCE PoE (DISCRIMINATIVE)".center(70), flush=True)
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

unique_months = sorted(np.unique(train_months))

# Crossfit r(y|u) by month and evaluate PoE update on held-out month.
oof_r = np.zeros_like(oof_base)
test_r_acc = np.zeros_like(test_base)

print("\nTraining r(y|u) with month-wise crossfit...", flush=True)
for m in unique_months:
    tr_idx = train_months != m
    va_idx = train_months == m

    model = make_r_model()
    model.fit(u_train.loc[tr_idx], y[tr_idx])
    oof_r[va_idx] = model.predict_proba(u_train.loc[va_idx])
    test_r_acc += model.predict_proba(u_test) / len(unique_months)

base_map, _ = compute_map(y, oof_base)
print(f"\nBase oof_e50 mAP: {base_map:.4f}", flush=True)

best = {"mAP": -1.0, "tau_nb": None, "lam": None}
print("\nGrid search (LOMO-like by month):", flush=True)
for tau_nb in TAUS_NB:
    mgn = top2_margin(oof_base)
    gate = mgn < tau_nb
    for lam in LAMBDAS:
        oof_adj = poe_update(oof_base, oof_r, lam=lam, gate=gate)
        m, _ = compute_map(y, oof_adj)
        if m > best["mAP"]:
            best = {"mAP": m, "tau_nb": tau_nb, "lam": lam}
        print(f"  tau_nb={tau_nb:.2f} lam={lam:.2f} -> mAP={m:.4f}", flush=True)

print("\nBest (by OOF proxy):", best, flush=True)

# Train final r(y|u) on all train (for test inference).
final_r = make_r_model()
final_r.fit(u_train, y)
test_r = final_r.predict_proba(u_test)

# Apply priors first (E67), then evidence PoE on unseen months only.
test_p0, changed = apply_gated_ratio_priors(
    test_base, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
)
print(f"\nApplied priors: tau_prior={TAU_PRIOR:.2f} changed_rows={changed}", flush=True)

margin0 = top2_margin(test_p0)
gate_unseen = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < float(best['tau_nb']))
print(f"Evidence gate (unseen only): tau_nb={best['tau_nb']:.2f} rows={int(gate_unseen.sum())}", flush=True)

test_final = poe_update(test_p0, test_r, lam=float(best["lam"]), gate=gate_unseen)

save_submission(
    test_final,
    f"e94_tabular_poe_tau{best['tau_nb']:.2f}_lam{best['lam']:.2f}_priortau{TAU_PRIOR:.2f}",
    cv_map=float(best["mAP"]),
)

# Also save one more candidate for exploration (slightly more conservative lambda).
lam2 = max(0.10, float(best["lam"]) - 0.10)
test_final2 = poe_update(test_p0, test_r, lam=lam2, gate=gate_unseen)
save_submission(
    test_final2,
    f"e94_tabular_poe_tau{best['tau_nb']:.2f}_lam{lam2:.2f}_priortau{TAU_PRIOR:.2f}_cons",
    cv_map=float(best["mAP"]),
)

print("\nDone.", flush=True)

