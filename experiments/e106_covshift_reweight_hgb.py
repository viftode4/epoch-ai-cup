"""E106: Covariate-shift reweighting (train->test) + sklearn HGB classifier (new approach).

Core idea (mathematical)
-----------------------
We observe strong distribution shift (esp. time/season). Many 0.59 post-processing variants
are effectively *small perturbations* on a fixed base posterior, so they don't reliably move
public LB (24% slice) and may not improve private.

Instead of more post-processing, we train a new base model under covariate shift:

  Target risk:   E_{x~p_test} [ L(f(x), y) ]
  Approximate via importance weighting on train:
                 E_{x~p_train} [ w(x) L(f(x), y) ],
  where          w(x) = p_test(x) / p_train(x).

We estimate w(x) using a domain classifier d(x)=P(test|x):
  w(x) ∝ d(x)/(1-d(x))  (up to constant).

We use ONLY non-temporal / physics-ish features and explicitly drop `ALL_TEMPORAL`.
This is a genuinely different lever: it changes the fitted decision boundary to match test.

Outputs
-------
- `oof_e106.npy`, `test_e106.npy`
- submissions:
    - `e106_covshift_hgb_raw_*` (new base)
    - `e106_covshift_geo_e50_*` (new base + strong baseline)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.features import ALL_TEMPORAL, build_features  # noqa: E402
from src.metrics import compute_map, print_results  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

SEED = 42
UNSEEN_MONTHS = (2, 5, 12)

# Clip importance weights to avoid blow-ups
W_MIN, W_MAX = 0.05, 20.0


def renorm_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, None)
    return p / p.sum(axis=1, keepdims=True)


def geo_mean(ps: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.clip(p, 1e-15, 1.0) for p in ps], axis=0)
    m = np.mean(np.log(stack), axis=0)
    return renorm_rows(np.exp(m))


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    return w[y]


def main() -> None:
    print("=" * 70, flush=True)
    print("E106 COVARIATE-SHIFT REWEIGHTING + HGB".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

    # --- Feature matrix (no temporal) ---
    feature_sets = ["core", "tabular", "rcs_fft", "weakclass", "flight_physics"]
    print("\nBuilding features (train/test)...", flush=True)
    Xtr_df = build_features(train_df, feature_sets=feature_sets)
    Xte_df = build_features(test_df, feature_sets=feature_sets)

    # Drop temporal leakage features explicitly
    drop_cols = [c for c in ALL_TEMPORAL if c in Xtr_df.columns]
    if drop_cols:
        Xtr_df = Xtr_df.drop(columns=drop_cols, errors="ignore")
        Xte_df = Xte_df.drop(columns=drop_cols, errors="ignore")
        print(f"Dropped temporal cols: {len(drop_cols)}", flush=True)

    # Handle inf/nan
    Xtr_df = Xtr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte_df = Xte_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xtr = Xtr_df.to_numpy(np.float32)
    Xte = Xte_df.to_numpy(np.float32)
    print(f"Feature dim: {Xtr.shape[1]}", flush=True)

    # --- Domain classifier for importance weights ---
    print("\nFitting domain classifier for importance weights...", flush=True)
    X_dom = np.vstack([Xtr, Xte])
    y_dom = np.concatenate([np.zeros(len(Xtr), dtype=int), np.ones(len(Xte), dtype=int)])

    dom = LogisticRegression(
        C=0.5,
        max_iter=4000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=SEED,
    )
    dom.fit(X_dom, y_dom)
    p_test_on_train = np.clip(dom.predict_proba(Xtr)[:, 1], 1e-4, 1.0 - 1e-4)

    w_imp = p_test_on_train / (1.0 - p_test_on_train)
    w_imp = np.clip(w_imp, W_MIN, W_MAX)

    # Combine with class-balance weights (macro-mAP cares about minority classes)
    w_cls = class_balance_weights(y)
    w = w_imp * w_cls
    w = w / np.mean(w)

    print(
        f"Importance weights: mean={w_imp.mean():.3f} p50={np.quantile(w_imp,0.50):.3f} p95={np.quantile(w_imp,0.95):.3f} max={w_imp.max():.3f}",
        flush=True,
    )

    # --- LOMO evaluation ---
    unique_months = sorted(set(train_months))
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    for m in unique_months:
        tr_idx = np.where(train_months != m)[0]
        va_idx = np.where(train_months == m)[0]
        print(f"\nLOMO month {m}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        clf = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=600,
            l2_regularization=0.2,
            random_state=SEED,
        )
        clf.fit(Xtr[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        p_va = clf.predict_proba(Xtr[va_idx]).astype(np.float32)
        p_va = renorm_rows(p_va)
        oof[va_idx] = p_va

        fold_map, _ = compute_map(y[va_idx], p_va)
        print(f"  mAP={fold_map:.4f}", flush=True)

    m, per = compute_map(y, oof)
    print_results(m, per, "E106 cov-shift HGB (LOMO)")
    np.save("oof_e106.npy", oof)
    print("Saved oof_e106.npy", flush=True)

    # --- Train full + predict test ---
    print("\nTraining full and predicting test...", flush=True)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=800,
        l2_regularization=0.2,
        random_state=SEED,
    )
    clf.fit(Xtr, y, sample_weight=w)
    p_test = renorm_rows(clf.predict_proba(Xte).astype(np.float32))
    np.save("test_e106.npy", p_test)
    print("Saved test_e106.npy", flush=True)

    save_submission(p_test, "e106_covshift_hgb_raw", cv_map=m)

    # New-signal ensemble with strong baseline posterior
    base_e50 = renorm_rows(np.load(ROOT / "test_e50.npy").astype(np.float32))
    ens = geo_mean([p_test, base_e50])
    save_submission(ens, "e106_covshift_geo_e50", cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

