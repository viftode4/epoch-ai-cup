"""E83: TabPFN v2.5 + tree ensemble.

TabPFN is a transformer foundation model for tabular data (Nature 2024).
Fundamentally different inductive bias from tree models (attention-based).
36 features, 2601 samples = well within TabPFN's sweet spot (<10K samples).

Pipeline:
  1. Same 36-feature set as E79
  2. Train LGB + XGB + CB + TabPFN with 5-fold SKF
  3. Optimize 4-model weights on SKF OOF
  4. Apply best E82 post-processing pipeline
  5. Save oof_e83.npy, test_e83.npy + submissions
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

# 36 validated features from backward elimination
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# Best post-processing params from E82 Optuna
PP_PARAMS = {
    "alpha_m2": 0.24, "alpha_m5": 0.11, "alpha_m12": 0.19,
    "tau_prior": 0.23,
    "tau_nb_unseen": 0.20, "gamma_unseen": 0.13,
    "tau_nb_shared": 0.11, "gamma_shared": 0.07,
}

UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)
LAPLACE = 1.0
MIN_SIGMA = 0.50


def add_weather_solar(train_feats, test_feats):
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
    return train_feats, test_feats


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
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
    size_to_idx = {s: i for i, s in enumerate(size_levels)}
    le = LabelEncoder()
    le.fit(CLASSES)
    y_nb = le.transform(train_df["bird_group"])
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
        mask = y_nb == c
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
            xc = x[y_nb == c]
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
    out = preds.copy()
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
    unseen_mask = np.isin(months, UNSEEN_MONTHS)
    margin2 = top2_margin(out)
    gate_unseen = unseen_mask & ok_nb & (margin2 < tau_nb_unseen)
    if gate_unseen.any():
        out[gate_unseen] = out[gate_unseen] * (nb_factors[gate_unseen] ** gamma_unseen)
        out = renorm_rows(out)
    shared_mask = np.isin(months, SHARED_MONTHS)
    margin3 = top2_margin(out)
    gate_shared = shared_mask & ok_nb & (margin3 < tau_nb_shared)
    if gate_shared.any():
        out[gate_shared] = out[gate_shared] * (nb_factors[gate_shared] ** gamma_shared)
        out = renorm_rows(out)
    return out


# ====================================================================
print("=" * 70, flush=True)
print("E83 TABPFN + TREE ENSEMBLE".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data and build features -----------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values

print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar
train_feats, test_feats = add_weather_solar(train_feats, test_feats)

# Prune to 36 validated features
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
missing = [f for f in KEEP_FEATURES if f not in train_feats.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing: {missing}", flush=True)
print(f"  Using {len(available)}/{len(KEEP_FEATURES)} pruned features", flush=True)

train_feats = train_feats[available]
test_feats = test_feats[available]

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X.shape[1]}", flush=True)

# -- Effective number class weights (beta=0.999) ---
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]
p_train = counts / counts.sum()

# -- SKF CV with LGB + XGB + CB + TabPFN ---
print("\n--- SKF ensemble training (5-fold) ---", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_tpfn = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_tpfn = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

# CatBoost params from E79 Optuna (hardcoded to save time)
cb_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "border_count": 128,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "auto_class_weights": "Balanced",
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "task_type": "GPU",
}

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS
    m_lgb, _ = compute_map(y[va_idx], oof_lgb[va_idx])
    print(f"    LGB fold mAP: {m_lgb:.4f}", flush=True)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X[va_idx])
    test_xgb += xgb.predict_proba(X_test) / N_FOLDS
    m_xgb, _ = compute_map(y[va_idx], oof_xgb[va_idx])
    print(f"    XGB fold mAP: {m_xgb:.4f}", flush=True)

    # CatBoost
    cb = CatBoostClassifier(**cb_params)
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS
    m_cb, _ = compute_map(y[va_idx], oof_cb[va_idx])
    print(f"    CB  fold mAP: {m_cb:.4f}", flush=True)

    # TabPFN
    tpfn = TabPFNClassifier(
        n_estimators=16,
        device="cuda",
        random_state=SEED + fold_i,
    )
    tpfn.fit(X[tr_idx], y[tr_idx])
    proba_va = tpfn.predict_proba(X[va_idx])
    # TabPFN may not output all classes if some are missing in fold
    # Map to full N_CLASSES array
    tpfn_classes = list(tpfn.classes_)
    full_va = np.zeros((len(va_idx), N_CLASSES), dtype=np.float64)
    full_test = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for ci, c in enumerate(tpfn_classes):
        full_va[:, c] = proba_va[:, ci]
        full_test[:, c] = tpfn.predict_proba(X_test)[:, ci] if fold_i == 0 else full_test[:, c]
    # Actually re-predict test for each fold
    proba_test = tpfn.predict_proba(X_test)
    full_test_fold = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for ci, c in enumerate(tpfn_classes):
        full_va[:, c] = proba_va[:, ci]
        full_test_fold[:, c] = proba_test[:, ci]

    oof_tpfn[va_idx] = full_va
    test_tpfn += full_test_fold / N_FOLDS
    m_tpfn, _ = compute_map(y[va_idx], full_va)
    print(f"    TPFN fold mAP: {m_tpfn:.4f}", flush=True)

# Individual model scores
print("\n--- Individual model SKF mAP ---", flush=True)
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb), ("TabPFN", oof_tpfn)]:
    m, per = compute_map(y, oof)
    print(f"  {name:8s}: {m:.4f}", flush=True)

# -- Weight optimization on SKF OOF ---
print("\n--- 4-model ensemble weight optimization ---", flush=True)
best_w = None
best_ens_map = -1.0

# Grid over 4 weights (step 0.05)
for w_lgb in np.arange(0.0, 0.65, 0.05):
    for w_xgb in np.arange(0.0, 0.65, 0.05):
        for w_cb in np.arange(0.0, 0.65, 0.05):
            w_tpfn = 1.0 - w_lgb - w_xgb - w_cb
            if w_tpfn < -0.01 or w_tpfn > 1.01:
                continue
            w_tpfn = max(0.0, w_tpfn)
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb + w_tpfn * oof_tpfn
            m, _ = compute_map(y, oof_ens)
            if m > best_ens_map:
                best_ens_map = m
                best_w = (w_lgb, w_xgb, w_cb, w_tpfn)

w_lgb, w_xgb, w_cb, w_tpfn = best_w
print(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f} TabPFN={w_tpfn:.2f}", flush=True)
print(f"  Ensemble SKF mAP: {best_ens_map:.4f}", flush=True)

oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb + w_tpfn * oof_tpfn
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb + w_tpfn * test_tpfn

base_map, base_per = compute_map(y, oof_ens)
print_results(base_map, base_per, label="E83 ensemble (SKF OOF)")

# Also try "best 3" and "best 2" combos from the 4 models
print("\n--- Ablation: pairwise / triple combos ---", flush=True)
models = {"LGB": oof_lgb, "XGB": oof_xgb, "CB": oof_cb, "TPFN": oof_tpfn}
test_models = {"LGB": test_lgb, "XGB": test_xgb, "CB": test_cb, "TPFN": test_tpfn}
for skip in models:
    subset = {k: v for k, v in models.items() if k != skip}
    # Equal weights among remaining 3
    oof_sub = sum(subset.values()) / 3.0
    m, _ = compute_map(y, oof_sub)
    print(f"  Without {skip:5s} (equal): {m:.4f}", flush=True)

# -- Save raw ensemble predictions ---
np.save(ROOT / "oof_e83.npy", oof_ens)
np.save(ROOT / "test_e83.npy", test_ens)
save_submission(test_ens, "e83_tabpfn_ensemble_raw", cv_map=base_map)

# -- Apply best E82 post-processing pipeline ---
print("\n--- Applying post-processing (E82 best params) ---", flush=True)

priors = build_gbif_priors(p_train)
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors_train, ok_train = compute_nb_factors(train_df, size_levels, log_p_size, mu, sig)
factors_test, ok_test = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

oof_pp = apply_full_pipeline(
    oof_ens, train_months, p_train, priors, factors_train, ok_train, **PP_PARAMS
)
test_pp = apply_full_pipeline(
    test_ens, test_months, p_train, priors, factors_test, ok_test, **PP_PARAMS
)

pp_map, pp_per = compute_map(y, oof_pp)
print(f"  Post-processed SKF mAP: {pp_map:.4f} (delta={pp_map - base_map:+.4f})", flush=True)
print_results(pp_map, pp_per, label="E83 post-processed")

# Compare to E79
try:
    oof_e79 = np.array(np.load(ROOT / "oof_e79.npy", allow_pickle=True), dtype=float)
    e79_map, _ = compute_map(y, oof_e79)
    print(f"\n  E79 baseline: {e79_map:.4f}", flush=True)
    print(f"  E83 vs E79:   {base_map - e79_map:+.4f} (raw) / {pp_map - e79_map:+.4f} (post-proc)", flush=True)
except Exception:
    pass

# Diagnostic: test pred top-1 changes vs E79
try:
    test_e79 = np.array(np.load(ROOT / "test_e79.npy", allow_pickle=True), dtype=float)
    top_e79 = test_e79.argmax(1)
    top_e83 = test_pp.argmax(1)
    total_flips = int((top_e79 != top_e83).sum())
    unseen_flips = int(((top_e79 != top_e83) & np.isin(test_months, UNSEEN_MONTHS)).sum())
    shared_flips = int(((top_e79 != top_e83) & np.isin(test_months, SHARED_MONTHS)).sum())
    print(f"  Flips vs E79: total={total_flips} unseen={unseen_flips} shared={shared_flips}", flush=True)
except Exception:
    pass

save_submission(test_pp, "e83_tabpfn_ensemble_pp", cv_map=pp_map)

# -- Multi-seed variant (seeds 42, 123, 456) ---
print("\n--- Multi-seed TabPFN (3 seeds) ---", flush=True)
oof_tpfn_ms = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_tpfn_ms = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for seed_i, seed_val in enumerate([42, 123, 456]):
    skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_val)
    oof_s = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_s = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for fold_i, (tr_idx, va_idx) in enumerate(skf_s.split(X, y)):
        tpfn = TabPFNClassifier(
            n_estimators=16,
            device="cuda",
            random_state=seed_val + fold_i,
        )
        tpfn.fit(X[tr_idx], y[tr_idx])
        proba_va = tpfn.predict_proba(X[va_idx])
        proba_test = tpfn.predict_proba(X_test)
        tpfn_classes = list(tpfn.classes_)
        full_va = np.zeros((len(va_idx), N_CLASSES), dtype=np.float64)
        full_test_fold = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        for ci, c in enumerate(tpfn_classes):
            full_va[:, c] = proba_va[:, ci]
            full_test_fold[:, c] = proba_test[:, ci]
        oof_s[va_idx] = full_va
        test_s += full_test_fold / N_FOLDS
    m_s, _ = compute_map(y, oof_s)
    print(f"  Seed {seed_val}: TabPFN mAP = {m_s:.4f}", flush=True)
    oof_tpfn_ms += oof_s / 3.0
    test_tpfn_ms += test_s / 3.0

m_ms, _ = compute_map(y, oof_tpfn_ms)
print(f"  Multi-seed TabPFN average: {m_ms:.4f}", flush=True)

# Re-optimize ensemble weights with multi-seed TabPFN
print("\n--- Re-optimize with multi-seed TabPFN ---", flush=True)
best_w2 = None
best_ens_map2 = -1.0
for w_lgb in np.arange(0.0, 0.65, 0.05):
    for w_xgb in np.arange(0.0, 0.65, 0.05):
        for w_cb in np.arange(0.0, 0.65, 0.05):
            w_tpfn = 1.0 - w_lgb - w_xgb - w_cb
            if w_tpfn < -0.01 or w_tpfn > 1.01:
                continue
            w_tpfn = max(0.0, w_tpfn)
            oof_ens2 = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb + w_tpfn * oof_tpfn_ms
            m, _ = compute_map(y, oof_ens2)
            if m > best_ens_map2:
                best_ens_map2 = m
                best_w2 = (w_lgb, w_xgb, w_cb, w_tpfn)

w2_lgb, w2_xgb, w2_cb, w2_tpfn = best_w2
print(f"  Best weights: LGB={w2_lgb:.2f} XGB={w2_xgb:.2f} CB={w2_cb:.2f} TabPFN={w2_tpfn:.2f}", flush=True)
print(f"  Multi-seed ensemble SKF mAP: {best_ens_map2:.4f}", flush=True)

if best_ens_map2 > best_ens_map:
    print(f"  Multi-seed IMPROVES by {best_ens_map2 - best_ens_map:+.4f}!", flush=True)
    oof_final = w2_lgb * oof_lgb + w2_xgb * oof_xgb + w2_cb * oof_cb + w2_tpfn * oof_tpfn_ms
    test_final = w2_lgb * test_lgb + w2_xgb * test_xgb + w2_cb * test_cb + w2_tpfn * test_tpfn_ms
else:
    print(f"  Single-seed is better, keeping original.", flush=True)
    oof_final = oof_ens
    test_final = test_ens

# Apply post-processing to best ensemble
oof_final_pp = apply_full_pipeline(
    oof_final, train_months, p_train, priors, factors_train, ok_train, **PP_PARAMS
)
test_final_pp = apply_full_pipeline(
    test_final, test_months, p_train, priors, factors_test, ok_test, **PP_PARAMS
)

final_map, final_per = compute_map(y, oof_final_pp)
print(f"\n  Final post-processed mAP: {final_map:.4f}", flush=True)
print_results(final_map, final_per, label="E83 final")

np.save(ROOT / "oof_e83.npy", oof_final)
np.save(ROOT / "test_e83.npy", test_final)
save_submission(test_final_pp, "e83_tabpfn_final", cv_map=final_map)

print("\nDone.", flush=True)
