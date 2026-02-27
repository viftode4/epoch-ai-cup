"""E82: Full pipeline Optuna optimization.

Joint optimization of all post-processing hyperparameters using Optuna.
Currently alpha, tau_prior, tau_nb, gamma, shared-month params are all
hand-tuned independently. Joint optimization may find better interactions.

Objective: shared-month LOMO mAP (proxy for LB, since we can't validate
unseen months offline).

Pipeline:
  oof/test_e79.npy -> gated priors -> unseen NB -> shared NB -> submission
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)
LAPLACE = 1.0
MIN_SIGMA = 0.50


def renorm_rows(pred):
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred):
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_gbif_priors(p_train):
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(N_CLASSES)
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


def build_nb_params(train_df):
    """NB params: size + speed + alt_mid + alt_range."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])
    size_idx = (
        train_df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}
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
    mu, sig = {}, {}
    for feat, x in feats.items():
        mu_f, sig_f = np.zeros(K), np.zeros(K)
        gm, gs = float(np.nanmean(x)), float(np.nanstd(x))
        if not np.isfinite(gs) or gs < MIN_SIGMA:
            gs = MIN_SIGMA
        for c in range(K):
            xc = x[y == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat], sig[feat] = mu_f, sig_f
    return size_levels, log_p_size, mu, sig


def compute_nb_factors(df, size_levels, log_p_size, mu, sig):
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    size_idx = (
        df["radar_bird_size"].fillna("__UNK__")
        .map(lambda v: size_to_idx.get(v, size_to_idx["__UNK__"])).values
    )
    speed = pd.to_numeric(df["airspeed"], errors="coerce").values.astype(float)
    min_z = pd.to_numeric(df["min_z"], errors="coerce").values.astype(float)
    max_z = pd.to_numeric(df["max_z"], errors="coerce").values.astype(float)
    alt_mid = 0.5 * (min_z + max_z)
    alt_range = max_z - min_z
    ok = np.isfinite(speed) & np.isfinite(alt_mid) & np.isfinite(alt_range)
    loglik = log_p_size[:, size_idx].T
    if ok.any():
        loglik[ok] += log_gaussian(speed[ok], mu["speed"], sig["speed"])
        loglik[ok] += log_gaussian(alt_mid[ok], mu["alt_mid"], sig["alt_mid"])
        loglik[ok] += log_gaussian(alt_range[ok], mu["alt_range"], sig["alt_range"])
    loglik = loglik - loglik.max(axis=1, keepdims=True)
    return np.exp(loglik), ok


def apply_full_pipeline(
    preds, months, p_train, priors, nb_factors, ok_nb,
    alpha_m2, alpha_m5, alpha_m12, tau_prior,
    tau_nb_unseen, gamma_unseen,
    tau_nb_shared, gamma_shared,
):
    """Apply the full pipeline: gated priors -> unseen NB -> shared NB."""
    out = preds.copy()

    # Stage 1: Gated ratio priors (unseen months only)
    alpha_map = {2: alpha_m2, 5: alpha_m5, 12: alpha_m12}
    margin = top2_margin(out)
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau_prior)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    out = renorm_rows(out)

    # Stage 2: NB correction on unseen months
    unseen_mask = np.isin(months, UNSEEN_MONTHS)
    margin2 = top2_margin(out)
    gate_unseen = unseen_mask & ok_nb & (margin2 < tau_nb_unseen)
    if gate_unseen.any():
        out[gate_unseen] = out[gate_unseen] * (nb_factors[gate_unseen] ** gamma_unseen)
        out = renorm_rows(out)

    # Stage 3: NB correction on shared months
    shared_mask = np.isin(months, SHARED_MONTHS)
    margin3 = top2_margin(out)
    gate_shared = shared_mask & ok_nb & (margin3 < tau_nb_shared)
    if gate_shared.any():
        out[gate_shared] = out[gate_shared] * (nb_factors[gate_shared] ** gamma_shared)
        out = renorm_rows(out)

    return out


# ====================================================================
print("=" * 70, flush=True)
print("E82 FULL PIPELINE OPTUNA OPTIMIZATION".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Build NB params
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors_train, ok_train = compute_nb_factors(train_df, size_levels, log_p_size, mu, sig)
factors_test, ok_test = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

# Load base model
base_name = None
for name, path in [("E79", "oof_e79.npy"), ("E50", "oof_e50.npy")]:
    p = ROOT / path
    if p.exists():
        try:
            b = np.load(p, allow_pickle=True)
            b = np.array(b, dtype=float)
            if b.size > 100:
                oof_base = renorm_rows(b)
                base_name = name
                break
        except Exception:
            continue

if base_name is None:
    print("ERROR: No OOF base model found.", flush=True)
    sys.exit(1)

test_base_path = ROOT / f"test_{base_name.lower()}.npy"
test_base = renorm_rows(np.array(np.load(test_base_path, allow_pickle=True), dtype=float))
print(f"\nUsing base: {base_name}", flush=True)

base_map, base_per = compute_map(y, oof_base)
print(f"Baseline LOMO mAP: {base_map:.4f}", flush=True)

# -- Optuna optimization --
print("\n--- Optuna optimization (100 trials) ---", flush=True)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def pipeline_objective(trial):
        alpha_m2 = trial.suggest_float("alpha_m2", 0.05, 0.40)
        alpha_m5 = trial.suggest_float("alpha_m5", 0.02, 0.25)
        alpha_m12 = trial.suggest_float("alpha_m12", 0.05, 0.40)
        tau_prior = trial.suggest_float("tau_prior", 0.08, 0.25)
        tau_nb_unseen = trial.suggest_float("tau_nb_unseen", 0.15, 0.40)
        gamma_unseen = trial.suggest_float("gamma_unseen", 0.04, 0.15)
        tau_nb_shared = trial.suggest_float("tau_nb_shared", 0.05, 0.20)
        gamma_shared = trial.suggest_float("gamma_shared", 0.01, 0.08)

        out = apply_full_pipeline(
            oof_base, train_months, p_train, priors, factors_train, ok_train,
            alpha_m2, alpha_m5, alpha_m12, tau_prior,
            tau_nb_unseen, gamma_unseen,
            tau_nb_shared, gamma_shared,
        )
        m, _ = compute_map(y, out)
        return m

    study = optuna.create_study(direction="maximize")
    study.optimize(pipeline_objective, n_trials=100, show_progress_bar=False)

    print(f"\n  Best OOF mAP: {study.best_value:.4f} (delta={study.best_value - base_map:+.4f})", flush=True)
    print(f"  Best params:", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k}: {v:.4f}", flush=True)

    # Get top-3 trials
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    top3 = trials_sorted[:3]

    print(f"\n--- Top 3 configurations ---", flush=True)
    for rank, trial in enumerate(top3, 1):
        print(f"\n  Rank {rank}: OOF mAP = {trial.value:.4f}", flush=True)
        params = trial.params

        # Apply to test
        test_out = apply_full_pipeline(
            test_base, test_months, p_train, priors, factors_test, ok_test,
            params["alpha_m2"], params["alpha_m5"], params["alpha_m12"],
            params["tau_prior"],
            params["tau_nb_unseen"], params["gamma_unseen"],
            params["tau_nb_shared"], params["gamma_shared"],
        )

        # Diagnostic: count changes
        top_before = test_base.argmax(1)
        top_after = test_out.argmax(1)
        total_flips = int((top_before != top_after).sum())
        unseen_flips = int(((top_before != top_after) & np.isin(test_months, UNSEEN_MONTHS)).sum())
        shared_flips = int(((top_before != top_after) & np.isin(test_months, SHARED_MONTHS)).sum())
        print(f"  Flips: total={total_flips} unseen={unseen_flips} shared={shared_flips}", flush=True)

        # Print key params
        for k, v in params.items():
            print(f"    {k}: {v:.4f}", flush=True)

        save_submission(
            test_out,
            f"e82_optuna_rank{rank}_{base_name.lower()}",
            cv_map=trial.value,
        )

    # Also apply to OOF and print per-class breakdown for best
    best_params = study.best_params
    best_oof = apply_full_pipeline(
        oof_base, train_months, p_train, priors, factors_train, ok_train,
        best_params["alpha_m2"], best_params["alpha_m5"], best_params["alpha_m12"],
        best_params["tau_prior"],
        best_params["tau_nb_unseen"], best_params["gamma_unseen"],
        best_params["tau_nb_shared"], best_params["gamma_shared"],
    )
    best_map, best_per = compute_map(y, best_oof)
    print_results(best_map, best_per, label="E82 best (OOF)")

except ImportError:
    print("  Optuna not installed! Falling back to manual grid.", flush=True)

    # Manual fallback: just use E75 params + gentle shared
    configs = [
        {"alpha_m2": 0.22, "alpha_m5": 0.12, "alpha_m12": 0.24, "tau_prior": 0.15,
         "tau_nb_unseen": 0.30, "gamma_unseen": 0.10, "tau_nb_shared": 0.12, "gamma_shared": 0.03},
        {"alpha_m2": 0.22, "alpha_m5": 0.12, "alpha_m12": 0.24, "tau_prior": 0.15,
         "tau_nb_unseen": 0.30, "gamma_unseen": 0.10, "tau_nb_shared": 0.10, "gamma_shared": 0.04},
        {"alpha_m2": 0.24, "alpha_m5": 0.10, "alpha_m12": 0.26, "tau_prior": 0.15,
         "tau_nb_unseen": 0.30, "gamma_unseen": 0.10, "tau_nb_shared": 0.15, "gamma_shared": 0.03},
    ]
    for i, params in enumerate(configs, 1):
        oof_out = apply_full_pipeline(
            oof_base, train_months, p_train, priors, factors_train, ok_train, **params
        )
        m, per = compute_map(y, oof_out)
        print(f"\n  Config {i}: OOF mAP = {m:.4f} (delta={m - base_map:+.4f})", flush=True)

        test_out = apply_full_pipeline(
            test_base, test_months, p_train, priors, factors_test, ok_test, **params
        )
        save_submission(test_out, f"e82_manual_config{i}_{base_name.lower()}", cv_map=m)

print("\nDone.", flush=True)
