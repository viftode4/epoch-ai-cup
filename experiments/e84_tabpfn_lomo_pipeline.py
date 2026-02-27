"""E84: TabPFN + tree ensemble with LOMO-optimized weights and E75 post-processing.

E83 showed TabPFN has the best LOMO score (0.3778 vs LGB 0.3557), but the
ensemble weights were optimized on SKF (giving 80% TabPFN), which overfits
to shared months. LB = 0.52.

Fix: optimize weights on LOMO OOF (honest temporal eval), then apply E75's
battle-tested post-processing pipeline (gated GBIF priors + NB physics on
unseen months). Also generate hybrid variants.

Pipeline:
  1. LOMO OOF for TabPFN + LGB + XGB + CB (4 month folds)
  2. Optimize 4-model weights on LOMO OOF
  3. Average LOMO fold test predictions
  4. Apply E75 post-processing (gated priors + NB physics, unseen months only)
  5. Also generate hybrid: TabPFN-blend for shared months, E75 for unseen months
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
SEED = 42

# 36 validated features
KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# E75 post-processing params (LB = 0.59, battle-tested)
BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
TAU_PRIOR = 0.15
TAU_NB = 0.30
GAMMA = 0.10

UNSEEN_MONTHS = (2, 5, 12)
SHARED_MONTHS = (9, 10)
LAPLACE = 1.0
MIN_SIGMA = 0.50


def add_weather_solar(train_feats, test_feats):
    for prefix, fname in [("wx", "weather"), ("sol", "solar")]:
        for split, feats in [("train", train_feats), ("test", test_feats)]:
            df = pd.read_csv(ROOT / "data" / f"{split}_{fname}.csv")
            for col in df.columns:
                feats[f"{prefix}_{col}"] = df[col].values
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
    yy = le.transform(train_df["bird_group"])
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
        mask = yy == c
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
            xc = x[yy == c]
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


def apply_e75_postprocessing(preds, months, p_train, priors, nb_factors, ok_nb):
    """Exact E75 post-processing: gated GBIF priors + NB physics on unseen months."""
    out = preds.copy()

    # Stage 1: Gated GBIF ratio priors (unseen months only)
    margin = top2_margin(out)
    for month, alpha in BASE_ALPHA.items():
        mask_m = months == month
        if mask_m.sum() == 0:
            continue
        gate = mask_m & (margin < TAU_PRIOR)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] /= np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
    out = renorm_rows(out)

    # Stage 2: NB physics correction (unseen months only)
    unseen_mask = np.isin(months, UNSEEN_MONTHS)
    margin2 = top2_margin(out)
    gate = unseen_mask & ok_nb & (margin2 < TAU_NB)
    if gate.any():
        out[gate] = out[gate] * (nb_factors[gate] ** GAMMA)
        out = renorm_rows(out)

    return out


# ====================================================================
print("=" * 70, flush=True)
print("E84 TABPFN LOMO PIPELINE".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data and build features -----------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(np.unique(train_months))

print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

train_feats, test_feats = add_weather_solar(train_feats, test_feats)

available = [f for f in KEEP_FEATURES if f in train_feats.columns]
print(f"  Using {len(available)}/{len(KEEP_FEATURES)} features", flush=True)
train_feats = train_feats[available]
test_feats = test_feats[available]

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]
p_train = counts / counts.sum()

# -- LOMO training: 4 month folds ------------------------------------
print("\n--- LOMO training (4 month folds) ---", flush=True)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_tpfn = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_tpfn = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

n_folds = len(unique_months)

for fold_i, month in enumerate(unique_months):
    va_idx = np.where(train_months == month)[0]
    tr_idx = np.where(train_months != month)[0]
    print(f"\n  Fold {fold_i+1}/{n_folds} (hold-out month={month}, "
          f"train={len(tr_idx)} val={len(va_idx)})", flush=True)

    # LGB
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
    )
    lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X[va_idx])
    test_lgb += lgb.predict_proba(X_test) / n_folds

    # XGB
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
    test_xgb += xgb.predict_proba(X_test) / n_folds

    # CB
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        bagging_temperature=0.5, random_strength=1.0, border_count=128,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(X_test) / n_folds

    # TabPFN (multi-seed: 3 seeds per fold for stability)
    tpfn_va_acc = np.zeros((len(va_idx), N_CLASSES), dtype=np.float64)
    tpfn_test_acc = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    tpfn_seeds = [SEED + fold_i, SEED + fold_i + 100, SEED + fold_i + 200]
    for s in tpfn_seeds:
        tpfn = TabPFNClassifier(n_estimators=16, device="cuda", random_state=s)
        tpfn.fit(X[tr_idx], y[tr_idx])
        proba_va = tpfn.predict_proba(X[va_idx])
        proba_test = tpfn.predict_proba(X_test)
        tpfn_classes = list(tpfn.classes_)
        full_va = np.zeros((len(va_idx), N_CLASSES), dtype=np.float64)
        full_test = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
        for ci, c in enumerate(tpfn_classes):
            full_va[:, c] = proba_va[:, ci]
            full_test[:, c] = proba_test[:, ci]
        tpfn_va_acc += full_va / len(tpfn_seeds)
        tpfn_test_acc += full_test / len(tpfn_seeds)

    oof_tpfn[va_idx] = tpfn_va_acc
    test_tpfn += tpfn_test_acc / n_folds

    # Per-fold scores
    for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb), ("TPFN", oof_tpfn)]:
        m, _ = compute_map(y[va_idx], oof[va_idx])
        print(f"    {name:5s}: {m:.4f}", flush=True)

# -- Individual model LOMO scores ---
print("\n--- Individual LOMO mAP ---", flush=True)
for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb), ("TabPFN", oof_tpfn)]:
    m, per = compute_map(y, oof)
    print(f"  {name:8s}: {m:.4f}", flush=True)

# -- LOMO weight optimization (4 models) ---
print("\n--- LOMO ensemble weight optimization ---", flush=True)
best_w = None
best_map = -1.0

for w_lgb in np.arange(0.0, 0.65, 0.05):
    for w_xgb in np.arange(0.0, 0.65, 0.05):
        for w_cb in np.arange(0.0, 0.65, 0.05):
            w_tpfn = 1.0 - w_lgb - w_xgb - w_cb
            if w_tpfn < -0.01 or w_tpfn > 1.01:
                continue
            w_tpfn = max(0.0, w_tpfn)
            oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb + w_tpfn * oof_tpfn
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w_lgb, w_xgb, w_cb, w_tpfn)

w_lgb, w_xgb, w_cb, w_tpfn = best_w
print(f"  Best: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f} TabPFN={w_tpfn:.2f}", flush=True)
print(f"  LOMO mAP: {best_map:.4f}", flush=True)

oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb + w_tpfn * oof_tpfn
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb + w_tpfn * test_tpfn

ens_map, ens_per = compute_map(y, oof_ens)
print_results(ens_map, ens_per, label="E84 ensemble (LOMO)")

# Also try trees-only for comparison
print("\n--- Trees-only LOMO for comparison ---", flush=True)
best_w_trees = None
best_map_trees = -1.0
for w_lgb_t in np.arange(0.0, 1.05, 0.05):
    for w_xgb_t in np.arange(0.0, 1.05 - w_lgb_t, 0.05):
        w_cb_t = 1.0 - w_lgb_t - w_xgb_t
        if w_cb_t < -0.01:
            continue
        oof_t = w_lgb_t * oof_lgb + w_xgb_t * oof_xgb + w_cb_t * oof_cb
        m_t, _ = compute_map(y, oof_t)
        if m_t > best_map_trees:
            best_map_trees = m_t
            best_w_trees = (w_lgb_t, w_xgb_t, w_cb_t)

print(f"  Trees-only best: LGB={best_w_trees[0]:.2f} XGB={best_w_trees[1]:.2f} "
      f"CB={best_w_trees[2]:.2f} -> LOMO={best_map_trees:.4f}", flush=True)
print(f"  TabPFN ensemble improvement: {best_map - best_map_trees:+.4f}", flush=True)

# -- Build NB params + GBIF priors ---
priors = build_gbif_priors(p_train)
size_levels, log_p_size, mu, sig = build_nb_params(train_df)
factors_train, ok_train = compute_nb_factors(train_df, size_levels, log_p_size, mu, sig)
factors_test, ok_test = compute_nb_factors(test_df, size_levels, log_p_size, mu, sig)

# -- Apply E75 post-processing to ensemble ---
print("\n--- E75 post-processing (unseen months: gated priors + NB physics) ---", flush=True)

test_pp = apply_e75_postprocessing(
    test_ens, test_months, p_train, priors, factors_test, ok_test
)

# Diagnostic flips
top_raw = test_ens.argmax(1)
top_pp = test_pp.argmax(1)
flips = int((top_raw != top_pp).sum())
unseen_flips = int(((top_raw != top_pp) & np.isin(test_months, UNSEEN_MONTHS)).sum())
print(f"  Post-processing flips: total={flips} unseen={unseen_flips}", flush=True)

# Save: raw ensemble
save_submission(test_ens, "e84_tabpfn_lomo_raw", cv_map=ens_map)

# Save: post-processed
save_submission(test_pp, "e84_tabpfn_lomo_pp", cv_map=ens_map)

# -- Hybrid variant: E75 for unseen, TabPFN-blend for shared ---
print("\n--- Hybrid variants ---", flush=True)

# Load E75's best submission (LB = 0.59)
e75_file = None
for candidate in [
    "submissions/e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv",
]:
    if (ROOT / candidate).exists():
        e75_file = candidate
        break

if e75_file is None:
    # Try to find any E75 submission
    import glob
    e75_candidates = sorted(glob.glob(str(ROOT / "submissions" / "e75*")))
    if e75_candidates:
        e75_file = e75_candidates[0]

if e75_file:
    print(f"  Using E75 file: {e75_file}", flush=True)
    e75_df = pd.read_csv(ROOT / e75_file)
    e75_probs = e75_df[CLASSES].values

    shared = np.isin(test_months, SHARED_MONTHS)
    unseen = np.isin(test_months, UNSEEN_MONTHS)

    # Hybrid 1: TabPFN-ensemble shared + E75 unseen
    hybrid1 = test_ens.copy()
    hybrid1[unseen] = e75_probs[unseen]
    save_submission(hybrid1, "e84_hybrid_tpfnens_shared_e75_unseen", cv_map=0.0)

    # Hybrid 2: TabPFN-only shared + E75 unseen
    hybrid2 = test_tpfn.copy()
    hybrid2[unseen] = e75_probs[unseen]
    save_submission(hybrid2, "e84_hybrid_tpfnonly_shared_e75_unseen", cv_map=0.0)

    # Hybrid 3: Trees-only (LOMO) shared + E75 unseen
    test_trees = best_w_trees[0] * test_lgb + best_w_trees[1] * test_xgb + best_w_trees[2] * test_cb
    hybrid3 = test_trees.copy()
    hybrid3[unseen] = e75_probs[unseen]
    save_submission(hybrid3, "e84_hybrid_trees_shared_e75_unseen", cv_map=0.0)

    # Hybrid 4: Post-processed ensemble shared + E75 unseen
    hybrid4 = test_pp.copy()
    hybrid4[unseen] = e75_probs[unseen]
    save_submission(hybrid4, "e84_hybrid_tpfnpp_shared_e75_unseen", cv_map=0.0)

    # Compare top-1 predictions for shared months across variants
    print("\n  Shared-month top-1 comparison vs E75:", flush=True)
    top_e75 = e75_probs.argmax(1)
    for name, preds in [("TabPFN-ens", test_ens), ("TabPFN-only", test_tpfn),
                         ("Trees-only", test_trees), ("TabPFN-pp", test_pp)]:
        top = preds.argmax(1)
        diff = int((top[shared] != top_e75[shared]).sum())
        print(f"    {name:12s}: {diff}/{shared.sum()} differ from E75 on shared months", flush=True)
else:
    print("  E75 submission not found -- skipping hybrids", flush=True)

# -- Also apply post-processing to trees-only for comparison ---
print("\n--- Trees-only + E75 post-processing (for LB comparison) ---", flush=True)
test_trees_pp = apply_e75_postprocessing(
    test_trees, test_months, p_train, priors, factors_test, ok_test
)
save_submission(test_trees_pp, "e84_trees_lomo_pp", cv_map=best_map_trees)

top_t_raw = test_trees.argmax(1)
top_t_pp = test_trees_pp.argmax(1)
flips_t = int((top_t_raw != top_t_pp).sum())
print(f"  Trees post-processing flips: {flips_t}", flush=True)

# -- Summary ---
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  TabPFN LOMO:      {compute_map(y, oof_tpfn)[0]:.4f}", flush=True)
print(f"  Trees LOMO:       {best_map_trees:.4f}", flush=True)
print(f"  Ensemble LOMO:    {ens_map:.4f}", flush=True)
print(f"  Weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f} TPFN={w_tpfn:.2f}", flush=True)
print(f"\nSubmissions saved:", flush=True)
print(f"  e84_tabpfn_lomo_raw       -- raw ensemble, no post-processing", flush=True)
print(f"  e84_tabpfn_lomo_pp        -- ensemble + E75 post-processing", flush=True)
print(f"  e84_trees_lomo_pp         -- trees-only + E75 post-processing", flush=True)
if e75_file:
    print(f"  e84_hybrid_tpfnens_shared_e75_unseen  -- TabPFN-ensemble shared + E75 unseen", flush=True)
    print(f"  e84_hybrid_tpfnonly_shared_e75_unseen  -- TabPFN-only shared + E75 unseen", flush=True)
    print(f"  e84_hybrid_trees_shared_e75_unseen     -- Trees shared + E75 unseen", flush=True)
    print(f"  e84_hybrid_tpfnpp_shared_e75_unseen    -- TabPFN-pp shared + E75 unseen", flush=True)

np.save(ROOT / "oof_e84.npy", oof_ens)
np.save(ROOT / "test_e84.npy", test_ens)

print("\nDone.", flush=True)
