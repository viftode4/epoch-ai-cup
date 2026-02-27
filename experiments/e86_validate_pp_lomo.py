"""E86: LOMO validation of post-processing strategies.

The ONLY honest way to validate unseen-month PP is LOMO:
  - Train on 3 months, predict on 4th (held-out = "unseen")
  - Apply PP to held-out month predictions
  - Measure per-class AP before/after PP
  - Average across 4 LOMO folds

This tells us WHICH PP strategy actually helps on months the model
has never seen, before wasting Kaggle uploads.

Strategies tested (from E85):
  A. E75-style NB (size+speed+alt) at various gamma
  B. Ungated NB (no margin threshold)
  C. Pure NB replacement (varying blend weights)
  D. GBIF-only (various strengths)
  E. Per-class gamma
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

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


def build_nb_params_from_data(df, y_arr):
    """Build NB params: size + speed + alt_mid + alt_range."""
    size_levels = ["Small bird", "Medium bird", "Large bird", "Flock", "__UNK__"]
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
    feats = {"speed": speed, "alt_mid": alt_mid, "alt_range": alt_range}

    K, S = N_CLASSES, len(size_levels)
    counts_cs = np.zeros((K, S), dtype=float)
    counts_c = np.zeros(K, dtype=float)
    for c in range(K):
        mask = y_arr == c
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
            xc = x[y_arr == c]
            ok = np.isfinite(xc)
            if ok.sum() >= 5:
                mu_f[c] = float(np.nanmean(xc))
                sc = float(np.nanstd(xc))
                sig_f[c] = sc if sc > MIN_SIGMA else MIN_SIGMA
            else:
                mu_f[c], sig_f[c] = gm, gs
        mu[feat], sig[feat] = mu_f, sig_f

    return size_levels, log_p_size, mu, sig


def compute_nb_factors_for_df(df, size_levels, log_p_size, mu, sig):
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


def compute_pure_nb(df, size_levels, log_p_size, mu, sig, p_train, gbif_priors, month):
    """Full NB posterior for a single month."""
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

    # GBIF prior for this month
    if month in gbif_priors:
        log_prior = np.log(np.clip(gbif_priors[month], 1e-12, None))
    else:
        log_prior = np.log(np.clip(p_train, 1e-12, None))

    log_posterior = loglik + log_prior[None, :]
    log_posterior = log_posterior - log_posterior.max(axis=1, keepdims=True)
    return renorm_rows(np.exp(log_posterior)), ok


def add_weather_solar(df_feats, split="train"):
    """Add weather + solar features."""
    weather = pd.read_csv(ROOT / "data" / f"{split}_weather.csv")
    solar = pd.read_csv(ROOT / "data" / f"{split}_solar.csv")
    for col in weather.columns:
        df_feats[f"wx_{col}"] = weather[col].values
    for col in solar.columns:
        df_feats[f"sol_{col}"] = solar[col].values
    return df_feats


def compute_fold_map(y_true, preds, label=""):
    """Compute mAP and per-class AP, return dict."""
    K = N_CLASSES
    per_class = {}
    for c in range(K):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0:
            per_class[CLASSES[c]] = float("nan")
        else:
            per_class[CLASSES[c]] = average_precision_score(y_bin, preds[:, c])
    valid = [v for v in per_class.values() if np.isfinite(v)]
    macro = float(np.mean(valid)) if valid else 0.0
    return macro, per_class


# ====================================================================
print("=" * 70, flush=True)
print("E86: LOMO VALIDATION OF PP STRATEGIES".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
unique_months = sorted(np.unique(train_months))
print(f"  Train months: {unique_months}", flush=True)
print(f"  Samples per month: {dict(zip(unique_months, [int((train_months==m).sum()) for m in unique_months]))}", flush=True)

# -- Build features ---------------------------------------------------
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
train_feats = add_weather_solar(train_feats, "train")
available = [f for f in KEEP_FEATURES if f in train_feats.columns]
train_feats = train_feats[available]
X_all = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X_all.shape[1]}", flush=True)

# -- Class weights (same as E79) --------------------------------------
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# -- GBIF priors (computed from FULL train) ---------------------------
p_train_full = counts / counts.sum()
gbif_priors = build_gbif_priors(p_train_full)

# -- LOMO loop --------------------------------------------------------
print("\n" + "=" * 70, flush=True)
print("  LOMO TRAINING + PP VALIDATION".center(70), flush=True)
print("=" * 70, flush=True)

# Store results per strategy
strategy_results = {}  # strategy_name -> list of (fold_month, mAP_before, mAP_after, per_class_before, per_class_after)

oof_preds = np.zeros((len(y), N_CLASSES), dtype=np.float64)

for fold_i, held_out_month in enumerate(unique_months):
    va_idx = np.where(train_months == held_out_month)[0]
    tr_idx = np.where(train_months != held_out_month)[0]

    print(f"\n--- LOMO fold {fold_i+1}/4: held-out month={held_out_month} "
          f"(train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)

    # Class distribution in validation set
    va_classes = np.bincount(y[va_idx], minlength=N_CLASSES)
    present_classes = [CLASSES[i] for i in range(N_CLASSES) if va_classes[i] > 0]
    absent_classes = [CLASSES[i] for i in range(N_CLASSES) if va_classes[i] == 0]
    print(f"  Present: {', '.join(present_classes)}", flush=True)
    if absent_classes:
        print(f"  ABSENT:  {', '.join(absent_classes)}", flush=True)

    # -- Train LGB + XGB + CB ensemble --------------------------------
    X_tr, y_tr = X_all[tr_idx], y[tr_idx]
    X_va, y_va = X_all[va_idx], y[va_idx]

    # LGB
    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    oof_lgb = lgb.predict_proba(X_va)

    # XGB
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb = xgb.predict_proba(X_va)

    # CB
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb = cb.predict_proba(X_va)

    # Ensemble (E79 weights: LGB=0.50, XGB=0.40, CB=0.10)
    raw_preds = 0.50 * oof_lgb + 0.40 * oof_xgb + 0.10 * oof_cb
    raw_preds = renorm_rows(raw_preds)
    oof_preds[va_idx] = raw_preds

    # -- Baseline (no PP) ---------------------------------------------
    base_mAP, base_per = compute_fold_map(y_va, raw_preds)
    print(f"  Raw ensemble mAP: {base_mAP:.4f}", flush=True)

    # -- Build NB params from TRAIN fold only (no leakage!) ------------
    train_fold_df = train_df.iloc[tr_idx].reset_index(drop=True)
    size_levels, log_p_size, mu, sig = build_nb_params_from_data(
        train_fold_df, y[tr_idx]
    )

    # Compute NB factors for validation fold
    val_fold_df = train_df.iloc[va_idx].reset_index(drop=True)
    factors, ok_feats = compute_nb_factors_for_df(
        val_fold_df, size_levels, log_p_size, mu, sig
    )

    # Train fold class frequencies
    tr_counts = np.bincount(y[tr_idx], minlength=N_CLASSES).astype(float)
    p_train_fold = tr_counts / tr_counts.sum()

    # GBIF priors for this fold
    fold_priors = build_gbif_priors(p_train_fold)

    # Pure NB posteriors for held-out month
    nb_probs, nb_ok = compute_pure_nb(
        val_fold_df, size_levels, log_p_size, mu, sig,
        p_train_fold, fold_priors, held_out_month
    )

    margin = top2_margin(raw_preds)

    # ================================================================
    #  Apply each PP strategy and measure improvement
    # ================================================================
    strategies = {}

    # -- 0. Baseline (no PP) --
    strategies["00_raw"] = raw_preds.copy()

    # -- A. GBIF prior only (various strengths) --
    for alpha_scale in [1.0, 2.0, 3.0]:
        alpha = {held_out_month: 0.22 * alpha_scale}
        for tau in [0.15, 0.30, 1.0]:
            out = raw_preds.copy()
            gate = margin < tau
            if gate.any():
                ratio = (fold_priors[held_out_month] / np.maximum(p_train_fold, 1e-12)) ** (0.22 * alpha_scale)
                out[gate] = out[gate] * ratio
                out = renorm_rows(out)
            strategies[f"A_gbif_s{alpha_scale:.0f}_t{tau:.2f}"] = out

    # -- B. GBIF + NB (gated, various gamma) --
    for gamma in [0.10, 0.15, 0.20, 0.30, 0.50]:
        for tau_nb in [0.30, 0.50]:
            # First apply GBIF
            out = raw_preds.copy()
            gbif_gate = margin < 0.15
            if gbif_gate.any():
                ratio = (fold_priors[held_out_month] / np.maximum(p_train_fold, 1e-12)) ** 0.22
                out[gbif_gate] = out[gbif_gate] * ratio
                out = renorm_rows(out)
            # Then NB
            margin2 = top2_margin(out)
            nb_gate = ok_feats & (margin2 < tau_nb)
            if nb_gate.any():
                out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** gamma)
                out = renorm_rows(out)
            strategies[f"B_nb_g{gamma:.2f}_t{tau_nb:.2f}"] = out

    # -- C. Ungated NB (no margin threshold) --
    for gamma in [0.10, 0.20, 0.30]:
        # GBIF first
        out = raw_preds.copy()
        gbif_gate = margin < 0.15
        if gbif_gate.any():
            ratio = (fold_priors[held_out_month] / np.maximum(p_train_fold, 1e-12)) ** 0.22
            out[gbif_gate] = out[gbif_gate] * ratio
            out = renorm_rows(out)
        # Ungated NB
        if ok_feats.any():
            out[ok_feats] = out[ok_feats] * (factors[ok_feats] ** gamma)
            out = renorm_rows(out)
        strategies[f"C_ungated_g{gamma:.2f}"] = out

    # -- D. Pure NB replacement (various blend weights) --
    for nb_w in [0.3, 0.5, 0.7, 1.0]:
        out = raw_preds.copy()
        mask = nb_ok
        if mask.any():
            out[mask] = (1.0 - nb_w) * raw_preds[mask] + nb_w * nb_probs[mask]
            out = renorm_rows(out)
        strategies[f"D_pureNB_w{nb_w:.1f}"] = out

    # -- E. Per-class gamma --
    gamma_profiles = {
        "E_aggressive": {0: 0.30, 1: 0.05, 2: 0.50, 3: 0.20, 4: 0.15,
                         5: 0.02, 6: 0.10, 7: 0.10, 8: 0.20},
        "E_moderate":   {0: 0.20, 1: 0.05, 2: 0.30, 3: 0.15, 4: 0.10,
                         5: 0.02, 6: 0.10, 7: 0.05, 8: 0.15},
    }
    for pname, gamma_per_class in gamma_profiles.items():
        for tau_nb in [0.30, 0.50]:
            # GBIF first
            out = raw_preds.copy()
            gbif_gate = margin < 0.15
            if gbif_gate.any():
                ratio = (fold_priors[held_out_month] / np.maximum(p_train_fold, 1e-12)) ** 0.22
                out[gbif_gate] = out[gbif_gate] * ratio
                out = renorm_rows(out)
            margin2 = top2_margin(out)
            nb_gate = ok_feats & (margin2 < tau_nb)
            if nb_gate.any():
                gamma_vec = np.array([gamma_per_class[c] for c in range(N_CLASSES)])
                out[nb_gate] = out[nb_gate] * (factors[nb_gate] ** gamma_vec[None, :])
                out = renorm_rows(out)
            strategies[f"{pname}_t{tau_nb:.2f}"] = out

    # -- Evaluate all strategies on this fold --------------------------
    for sname, preds in strategies.items():
        s_mAP, s_per = compute_fold_map(y_va, preds)
        if sname not in strategy_results:
            strategy_results[sname] = []
        strategy_results[sname].append({
            "month": held_out_month,
            "mAP": s_mAP,
            "per_class": s_per,
            "n_val": len(va_idx),
        })

    # Print top strategies for this fold
    fold_scores = [(sname, strategy_results[sname][-1]["mAP"]) for sname in strategies]
    fold_scores.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 strategies for month {held_out_month}:", flush=True)
    for i, (sname, score) in enumerate(fold_scores[:10]):
        delta = score - base_mAP
        print(f"    {i+1:2d}. {sname:35s} mAP={score:.4f} ({delta:+.4f})", flush=True)
    print(f"  ...", flush=True)
    print(f"  Worst: {fold_scores[-1][0]:35s} mAP={fold_scores[-1][1]:.4f} ({fold_scores[-1][1]-base_mAP:+.4f})", flush=True)


# ====================================================================
#  AGGREGATE RESULTS ACROSS ALL FOLDS
# ====================================================================
print("\n" + "=" * 70, flush=True)
print("  AGGREGATE LOMO RESULTS (average across 4 folds)".center(70), flush=True)
print("=" * 70, flush=True)

# Compute weighted-average mAP (weighted by fold size)
agg = {}
for sname, fold_list in strategy_results.items():
    total_n = sum(f["n_val"] for f in fold_list)
    weighted_mAP = sum(f["mAP"] * f["n_val"] for f in fold_list) / total_n
    simple_mAP = np.mean([f["mAP"] for f in fold_list])
    agg[sname] = {
        "weighted_mAP": weighted_mAP,
        "simple_mAP": simple_mAP,
        "per_month": {f["month"]: f["mAP"] for f in fold_list},
    }

# Sort by weighted mAP
ranking = sorted(agg.items(), key=lambda x: -x[1]["weighted_mAP"])
base_wmAP = agg["00_raw"]["weighted_mAP"]

print(f"\n{'Rank':>4s}  {'Strategy':35s}  {'wt mAP':>8s}  {'delta':>8s}  {'M1':>6s}  {'M4':>6s}  {'M9':>6s}  {'M10':>6s}", flush=True)
print("-" * 105, flush=True)

for rank, (sname, data) in enumerate(ranking):
    delta = data["weighted_mAP"] - base_wmAP
    pm = data["per_month"]
    m_strs = []
    for m in unique_months:
        if m in pm:
            m_strs.append(f"{pm[m]:.4f}")
        else:
            m_strs.append("  --  ")
    marker = " ***" if rank < 3 and sname != "00_raw" else ""
    print(f"{rank+1:4d}  {sname:35s}  {data['weighted_mAP']:.4f}  {delta:+.4f}  {'  '.join(m_strs)}{marker}", flush=True)

# -- Per-class improvement for top strategy --
print("\n" + "=" * 70, flush=True)
print("  PER-CLASS AP: RAW vs BEST STRATEGY".center(70), flush=True)
print("=" * 70, flush=True)

best_name = ranking[0][0] if ranking[0][0] != "00_raw" else ranking[1][0]
print(f"\nBest strategy: {best_name}", flush=True)
print(f"\n{'Class':15s}  {'Raw AP':>8s}  {'Best AP':>8s}  {'Delta':>8s}", flush=True)
print("-" * 50, flush=True)

for cls in CLASSES:
    raw_aps = [f["per_class"].get(cls, float("nan")) for f in strategy_results["00_raw"]]
    best_aps = [f["per_class"].get(cls, float("nan")) for f in strategy_results[best_name]]
    raw_avg = np.nanmean(raw_aps)
    best_avg = np.nanmean(best_aps)
    delta = best_avg - raw_avg
    print(f"{cls:15s}  {raw_avg:8.4f}  {best_avg:8.4f}  {delta:+8.4f}", flush=True)

# -- Overall LOMO mAP (full OOF) --
print(f"\n\nFull LOMO mAP (raw ensemble): ", flush=True, end="")
lomo_map, lomo_per = compute_map(y, oof_preds)
print(f"{lomo_map:.4f}", flush=True)
for cls_name, ap in zip(CLASSES, lomo_per):
    print(f"  {cls_name:15s}: {ap:.4f}", flush=True)

print("\nDone.", flush=True)
