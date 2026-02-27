"""E81: Shared month corrections (Sep/Oct).

For the first time, applies post-processing to the 67% of test that falls in
shared months (Sep=24.4%, Oct=42.9%). All previous post-processing only
touched the 33% in unseen months {2,5,12}.

Two strategies:
  A) Gentle NB on shared months -- same framework as E75/E80, but much more
     conservative (higher tau, lower gamma). Validated on SKF OOF since we have
     labeled data for Sep/Oct.
  B) Binary specialist injection on shared months -- for uncertain samples,
     blend specialist predictions for confused classes.

Pipeline:
  best_base.npy -> unseen-month pipeline (priors + NB) -> shared-month correction

Note: E79 now uses SKF, so OOF quality is better for validating shared-month params.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)

# Unseen month pipeline params (fixed from E75)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15

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


def build_gbif_priors(p_train):
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


def apply_gated_ratio_priors(preds, months, p_train, priors, alpha_map, tau):
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
        out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


def log_gaussian(x, mu, sigma):
    x = x[:, None]
    z = (x - mu[None, :]) / sigma[None, :]
    return -0.5 * z * z - np.log(sigma[None, :])


def build_nb_params(train_df):
    """NB params: size + speed + alt_mid + alt_range (E75 recipe)."""
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
    K, S = len(CLASSES), len(size_levels)
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
    return size_levels, log_p_size, mu, sig, y


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


# ====================================================================
print("=" * 70, flush=True)
print("E81 SHARED MONTH CORRECTIONS".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

unseen_mask_test = np.isin(test_months, UNSEEN_MONTHS)
shared_mask_test = np.isin(test_months, SHARED_MONTHS)

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Build NB params from full train
size_levels, log_p_size, mu, sig, _ = build_nb_params(train_df)

# Compute NB factors for train (for OOF validation) and test
factors_train, ok_train = compute_nb_factors(train_df, size_levels, log_p_size, mu, sig)
factors_test, ok_test = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

print(f"\nTrain NB ok: {ok_train.sum()}/{len(ok_train)}", flush=True)
print(f"Test NB ok:  {ok_test.sum()}/{len(ok_test)}", flush=True)
print(f"Test unseen: {unseen_mask_test.sum()} ({100*unseen_mask_test.mean():.1f}%)", flush=True)
print(f"Test shared: {shared_mask_test.sum()} ({100*shared_mask_test.mean():.1f}%)", flush=True)

# -- Load base models -----------------------------------------------
bases = {}
for name, path in [("E50", "test_e50.npy"), ("E79", "test_e79.npy")]:
    p = ROOT / path
    if p.exists():
        try:
            b = np.load(p, allow_pickle=True)
            b = np.array(b, dtype=float)
            if b.size > 100:
                bases[name] = renorm_rows(b)
        except Exception as e:
            print(f"  WARNING: Could not load {name} test: {e}", flush=True)

# Load OOF for validation
oof_bases = {}
for name, path in [("E50", "oof_e50.npy"), ("E79", "oof_e79.npy")]:
    p = ROOT / path
    if p.exists():
        try:
            b = np.load(p, allow_pickle=True)
            b = np.array(b, dtype=float)
            if b.size > 100:
                oof_bases[name] = renorm_rows(b)
        except Exception as e:
            print(f"  WARNING: Could not load {name} OOF: {e}", flush=True)

if not bases:
    print("\nERROR: No base models found.", flush=True)
    sys.exit(1)

print(f"\nBase models: {list(bases.keys())}", flush=True)
print(f"OOF models:  {list(oof_bases.keys())}", flush=True)

# ====================================================================
# Strategy A: Gentle NB on shared months
# Validate on OOF first to find optimal gamma_shared and tau_shared
# ====================================================================
print("\n" + "=" * 60, flush=True)
print("  STRATEGY A: Gentle NB on shared months", flush=True)
print("=" * 60, flush=True)

# Validate on OOF (shared months = months 9, 10 in train)
shared_mask_train = np.isin(train_months, SHARED_MONTHS)
print(f"\n  OOF shared-month samples: {shared_mask_train.sum()}", flush=True)

TAU_SHARED_GRID = [0.08, 0.10, 0.12, 0.15]
GAMMA_SHARED_GRID = [0.02, 0.03, 0.04, 0.05]

# Best unseen-month NB params (from E75)
UNSEEN_TAU_NB = 0.30
UNSEEN_GAMMA = 0.10

for base_name, oof_base in oof_bases.items():
    print(f"\n  --- OOF validation: {base_name} ---", flush=True)

    # Baseline: no shared-month correction
    base_map, base_per = compute_map(y, oof_base)
    # Shared-month-only mAP
    shared_idx = np.where(shared_mask_train)[0]
    if len(shared_idx) > 0:
        sm_map, sm_per = compute_map(y[shared_idx], oof_base[shared_idx])
        print(f"  Baseline shared-month mAP: {sm_map:.4f}", flush=True)
    print(f"  Baseline full mAP: {base_map:.4f}", flush=True)

    best_sm = {"map": -1, "tau": 0, "gamma": 0}
    for tau_s in TAU_SHARED_GRID:
        for gamma_s in GAMMA_SHARED_GRID:
            out = oof_base.copy()
            margin = top2_margin(out)
            gate = shared_mask_train & ok_train & (margin < tau_s)
            if gate.sum() == 0:
                continue
            out[gate] = out[gate] * (factors_train[gate] ** gamma_s)
            out = renorm_rows(out)
            m, _ = compute_map(y, out)
            sm_m, _ = compute_map(y[shared_idx], out[shared_idx])
            flips = int(((oof_base.argmax(1) != out.argmax(1)) & shared_mask_train).sum())
            if m > best_sm["map"]:
                best_sm = {"map": m, "sm_map": sm_m, "tau": tau_s, "gamma": gamma_s, "flips": flips, "gated": int(gate.sum())}

    print(f"\n  Best shared-month NB on OOF:", flush=True)
    print(f"    tau={best_sm['tau']:.2f} gamma={best_sm['gamma']:.2f} "
          f"gated={best_sm.get('gated',0)} flips={best_sm.get('flips',0)}", flush=True)
    print(f"    Full mAP: {best_sm['map']:.4f} (delta={best_sm['map'] - base_map:+.4f})", flush=True)
    if "sm_map" in best_sm:
        print(f"    Shared-month mAP: {best_sm['sm_map']:.4f}", flush=True)

# -- Apply to test -------------------------------------------------
for base_name, base_test in bases.items():
    print(f"\n  --- Generating test submissions: {base_name} ---", flush=True)

    # Step 1: Unseen-month pipeline (priors + NB)
    pred_priors, ch_prior = apply_gated_ratio_priors(
        base_test, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    margin_priors = top2_margin(pred_priors)

    # Apply unseen-month NB
    gate_unseen = unseen_mask_test & ok_test & (margin_priors < UNSEEN_TAU_NB)
    pred_unseen = pred_priors.copy()
    pred_unseen[gate_unseen] = pred_unseen[gate_unseen] * (factors_test[gate_unseen] ** UNSEEN_GAMMA)
    pred_unseen = renorm_rows(pred_unseen)
    unseen_flips = int(((pred_priors.argmax(1) != pred_unseen.argmax(1)) & unseen_mask_test).sum())
    print(f"  Unseen NB: gated={gate_unseen.sum()} flips={unseen_flips}", flush=True)

    # Step 2: Shared-month NB (gentle)
    margin_after_unseen = top2_margin(pred_unseen)

    # Generate submissions for a range of shared-month params
    for tau_s in [0.10, 0.12, 0.15]:
        for gamma_s in [0.03, 0.04, 0.05]:
            gate_shared = shared_mask_test & ok_test & (margin_after_unseen < tau_s)
            out = pred_unseen.copy()
            out[gate_shared] = out[gate_shared] * (factors_test[gate_shared] ** gamma_s)
            out = renorm_rows(out)
            shared_flips = int(((pred_unseen.argmax(1) != out.argmax(1)) & shared_mask_test).sum())
            total_flips = int((base_test.argmax(1) != out.argmax(1)).sum())
            print(
                f"    tau_s={tau_s:.2f} gamma_s={gamma_s:.2f} "
                f"shared_gated={gate_shared.sum()} shared_flips={shared_flips} total_flips={total_flips}",
                flush=True,
            )
            save_submission(
                out,
                f"e81_{base_name.lower()}_shared_tau{tau_s:.2f}_g{gamma_s:.2f}",
                cv_map=None,
            )

    # Also save the unseen-only version as reference
    save_submission(
        pred_unseen,
        f"e81_{base_name.lower()}_unseen_only",
        cv_map=None,
    )

# ====================================================================
# Strategy B: Binary specialist injection on shared months
# ====================================================================
print("\n" + "=" * 60, flush=True)
print("  STRATEGY B: Specialist injection on shared months", flush=True)
print("=" * 60, flush=True)

# Build features for specialist training
print("\nBuilding features for specialists...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal
keep_cols = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep_cols]
test_feats = test_feats[keep_cols]

# Add weather + solar
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

# Prune to 36 features
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
train_feats = train_feats[available]
test_feats = test_feats[available]

Xf = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
Xf_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Train specialists for confused classes (Waders, Pigeons, BoP)
SPECIALIST_CLASSES_B = ["Waders", "Pigeons", "Birds of Prey"]
unique_months = sorted(np.unique(train_months))

specialist_test_preds = {}
specialist_oof_preds = {}

for cls in SPECIALIST_CLASSES_B:
    idx = CLASSES.index(cls)
    y_bin = (y == idx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(Xf_test), dtype=np.float32)

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        pos_tr = int(y_bin[tr_idx].sum())

        if pos_tr < 4:
            rate = float(pos_tr / len(tr_idx))
            oof_bin[va_idx] = rate
            test_bin += rate / len(unique_months)
            continue

        cb = CatBoostClassifier(
            iterations=1200, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=42, verbose=0,
            early_stopping_rounds=80, task_type="GPU",
        )
        cb.fit(Xf[tr_idx], y_bin[tr_idx], eval_set=(Xf[va_idx], y_bin[va_idx]), verbose=0)
        oof_bin[va_idx] = cb.predict_proba(Xf[va_idx])[:, 1]
        test_bin += cb.predict_proba(Xf_test)[:, 1] / len(unique_months)

    specialist_oof_preds[cls] = oof_bin
    specialist_test_preds[cls] = test_bin
    ap = average_precision_score(y_bin, oof_bin)
    print(f"  {cls:<15s}: specialist AP = {ap:.4f}", flush=True)

# -- Validate on OOF shared months ---------------------------------
print("\n  --- OOF validation of specialist injection ---", flush=True)
TAU_INJECT_GRID = [0.10, 0.15, 0.20]
ALPHA_INJECT_GRID = [0.1, 0.2, 0.3]

for base_name, oof_base in oof_bases.items():
    print(f"\n  Base: {base_name}", flush=True)
    base_map, _ = compute_map(y, oof_base)

    best_inject = {"map": base_map, "params": None}

    for tau_inj in TAU_INJECT_GRID:
        for alpha_inj in ALPHA_INJECT_GRID:
            out = oof_base.copy()
            margin = top2_margin(out)
            gate = shared_mask_train & (margin < tau_inj)

            for cls in SPECIALIST_CLASSES_B:
                cidx = CLASSES.index(cls)
                # Blend specialist into the class column for gated samples
                out[gate, cidx] = (1.0 - alpha_inj) * out[gate, cidx] + alpha_inj * specialist_oof_preds[cls][gate]
            out = renorm_rows(out)
            m, _ = compute_map(y, out)
            if m > best_inject["map"]:
                best_inject = {"map": m, "params": (tau_inj, alpha_inj), "gated": int(gate.sum())}

    if best_inject["params"]:
        tau_inj, alpha_inj = best_inject["params"]
        print(f"  Best inject: tau={tau_inj:.2f} alpha={alpha_inj:.2f} "
              f"gated={best_inject.get('gated',0)} "
              f"mAP={best_inject['map']:.4f} (delta={best_inject['map'] - base_map:+.4f})", flush=True)
    else:
        print(f"  No improvement from specialist injection on OOF.", flush=True)

# -- Apply to test -------------------------------------------------
for base_name, base_test in bases.items():
    print(f"\n  --- Test submissions (specialist injection): {base_name} ---", flush=True)

    # Step 1: full unseen pipeline
    pred_priors, _ = apply_gated_ratio_priors(
        base_test, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    gate_unseen = unseen_mask_test & ok_test & (top2_margin(pred_priors) < UNSEEN_TAU_NB)
    pred_unseen = pred_priors.copy()
    pred_unseen[gate_unseen] = pred_unseen[gate_unseen] * (factors_test[gate_unseen] ** UNSEEN_GAMMA)
    pred_unseen = renorm_rows(pred_unseen)

    # Step 2: specialist injection on shared months
    for tau_inj in [0.10, 0.15, 0.20]:
        for alpha_inj in [0.1, 0.2, 0.3]:
            out = pred_unseen.copy()
            margin = top2_margin(out)
            gate = shared_mask_test & (margin < tau_inj)
            for cls in SPECIALIST_CLASSES_B:
                cidx = CLASSES.index(cls)
                out[gate, cidx] = (1.0 - alpha_inj) * out[gate, cidx] + alpha_inj * specialist_test_preds[cls][gate]
            out = renorm_rows(out)
            flips = int(((pred_unseen.argmax(1) != out.argmax(1)) & shared_mask_test).sum())
            print(
                f"    tau={tau_inj:.2f} alpha={alpha_inj:.2f} "
                f"gated={gate.sum()} flips={flips}",
                flush=True,
            )
            save_submission(
                out,
                f"e81_{base_name.lower()}_inject_tau{tau_inj:.2f}_a{alpha_inj:.1f}",
                cv_map=None,
            )

print("\nDone.", flush=True)
