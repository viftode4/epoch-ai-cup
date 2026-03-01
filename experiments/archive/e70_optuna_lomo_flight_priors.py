"""E70: Optuna-tuned LGB on E48-C (flight priors), LOMO-optimised.

Why this should beat E54 (LB 0.56):
  - E54 used E50 which used E48-C default hyperparameters.
  - E60 showed Optuna tuning on E38 gave SKF +0.002. On E48-C the gain
    should be at least as large.
  - Crucially, we optimise LOMO (not SKF) so tuned params generalise to
    the unseen months that matter on the LB.
  - Same E54 winter_tilt priors applied at the end.

Key design choices vs E60:
  - Feature set: E48-C (E38 + AVONET/BirdWingData flight priors)
    rather than plain E38. These features are what made E50 better than
    E42 on the LB.
  - Optimisation target: LOMO mAP (4-fold month-out) instead of SKF mAP.
  - LGB tuned; XGB + CB keep E48 defaults (too slow to tune all three
    on CPU within a single run).
  - Final test predictions averaged over LOMO folds (not SKF) for
    consistency with the evaluation regime.
  - E54 winter_tilt applied for unseen months {2, 5, 12}.
"""

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.external_priors import build_external_class_priors
from src.features import ALL_TEMPORAL, add_external_prior_features, build_features
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

# ── E48 default XGB / CB params (unchanged from E48) ───────────────
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cpu", "tree_method": "hist",
}
CB_PARAMS = {
    "iterations": 1400, "learning_rate": 0.05,
    "depth": 6, "l2_leaf_reg": 3,
    "loss_function": "MultiClass", "eval_metric": "MultiClass",
    "random_seed": 42, "verbose": 0,
    "early_stopping_rounds": 80, "task_type": "CPU",
}

print("=" * 65, flush=True)
print("E70: OPTUNA-TUNED LGB | E48-C FEATURES | LOMO OBJECTIVE", flush=True)
print("=" * 65, flush=True)

# ── Data + labels ──────────────────────────────────────────────────
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w /= class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# ── Build E48-C features ───────────────────────────────────────────
print("\nBuilding E38 base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep].copy()
test_feats = test_feats[keep].copy()

print("Adding weather + solar + GBIF...", flush=True)
for split, feats in [("train", train_feats), ("test", test_feats)]:
    months = train_months if split == "train" else test_months
    weather = pd.read_csv(ROOT / "data" / f"{split}_weather.csv")
    solar = pd.read_csv(ROOT / "data" / f"{split}_solar.csv")
    for col in weather.columns:
        feats[f"wx_{col}"] = weather[col].values
    for col in solar.columns:
        feats[f"sol_{col}"] = solar[col].values

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls != "Clutter":
            cmean = gbif[cls].values.mean()
            si[i] = row[cls] / cmean if cmean > 0 else 1.0
    gbif_si[m] = si
for i, cls in enumerate(CLASSES):
    col = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col] = [gbif_si[m][i] for m in train_months]
    test_feats[col] = [gbif_si[m][i] for m in test_months]
gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

print("Adding E48-C flight priors (AVONET + BirdWingData)...", flush=True)
priors = build_external_class_priors(ROOT)
train_feats = add_external_prior_features(
    train_feats, priors, include_morph=False, include_flight=True
)
test_feats = add_external_prior_features(
    test_feats, priors, include_morph=False, include_flight=True
)

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"  Total features: {len(fn)}", flush=True)

# ── Optuna: tune LGB against LOMO ─────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("OPTUNA TUNING (LGB, LOMO objective, 30 trials)", flush=True)
print("=" * 65, flush=True)

def lomo_lgb_objective(trial):
    params = {
        "objective": "multiclass", "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 90),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "verbose": -1, "seed": 42, "n_jobs": -1, "device": "cpu",
    }
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx],
                             weight=sample_weights[tr_idx], feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y[va_idx],
                           feature_name=fn, reference=dtrain)
        mdl = lgb.train(params, dtrain, 800,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])
        oof[va_idx] = mdl.predict(X[va_idx]).reshape(-1, N_CLASSES)
    mAP, _ = compute_map(y, oof)
    return mAP

study = optuna.create_study(direction="maximize")
study.optimize(lomo_lgb_objective, n_trials=30, show_progress_bar=True)

print(f"\nBest LOMO: {study.best_value:.4f}", flush=True)
print("Best LGB params:", flush=True)
for k, v in study.best_params.items():
    print(f"  {k}: {v}", flush=True)

best_lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "cpu",
    **study.best_params,
}
# Rename 'lr' key back to 'learning_rate'
if "lr" in best_lgb_params:
    best_lgb_params["learning_rate"] = best_lgb_params.pop("lr")

# ── Final training: LGB (tuned) + XGB + CB (defaults) via LOMO ────
print("\n" + "=" * 65, flush=True)
print("FINAL LOMO TRAINING — LGB (tuned) + XGB + CB", flush=True)
print("=" * 65, flush=True)

W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34

oof_lgb = np.zeros((len(y), N_CLASSES))
oof_xgb = np.zeros((len(y), N_CLASSES))
oof_cb = np.zeros((len(y), N_CLASSES))
test_lgb_preds, test_xgb_preds, test_cb_preds = [], [], []

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    print(f"\n  Month {m} (train={len(tr_idx)}, val={len(va_idx)})", flush=True)

    # LGB (tuned)
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx],
                         weight=sample_weights[tr_idx], feature_name=fn)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx],
                       feature_name=fn, reference=dtrain)
    lgb_mdl = lgb.train(best_lgb_params, dtrain, 1500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80),
                                   lgb.log_evaluation(0)])
    oof_lgb[va_idx] = lgb_mdl.predict(X[va_idx]).reshape(-1, N_CLASSES)
    test_lgb_preds.append(lgb_mdl.predict(X_test).reshape(-1, N_CLASSES))

    # XGB (E48 defaults)
    dtr_xgb = xgb.DMatrix(X[tr_idx], label=y[tr_idx],
                           weight=sample_weights[tr_idx], feature_names=fn)
    dva_xgb = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=fn)
    xgb_mdl = xgb.train(XGB_PARAMS, dtr_xgb, 1400,
                         evals=[(dva_xgb, "val")],
                         early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = xgb_mdl.predict(dva_xgb)
    test_xgb_preds.append(xgb_mdl.predict(
        xgb.DMatrix(X_test, feature_names=fn)))

    # CB (E48 defaults)
    cb_mdl = CatBoostClassifier(**CB_PARAMS)
    cb_mdl.fit(X[tr_idx], y[tr_idx],
               eval_set=(X[va_idx], y[va_idx]),
               sample_weight=sample_weights[tr_idx], verbose=0)
    oof_cb[va_idx] = cb_mdl.predict_proba(X[va_idx])
    test_cb_preds.append(cb_mdl.predict_proba(X_test))

    fold_ens = (W_LGB * oof_lgb[va_idx]
                + W_XGB * oof_xgb[va_idx]
                + W_CB * oof_cb[va_idx])
    m_map, _ = compute_map(y[va_idx], fold_ens)
    print(f"  Ensemble fold mAP: {m_map:.4f}", flush=True)

test_lgb = np.mean(test_lgb_preds, axis=0)
test_xgb = np.mean(test_xgb_preds, axis=0)
test_cb = np.mean(test_cb_preds, axis=0)

oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb

lomo_map, lomo_per = compute_map(y, oof_ens)
lgb_lomo, _ = compute_map(y, oof_lgb)
xgb_lomo, _ = compute_map(y, oof_xgb)
cb_lomo, _ = compute_map(y, oof_cb)

print(f"\n{'='*65}", flush=True)
print("LOMO RESULTS", flush=True)
print(f"{'='*65}", flush=True)
print(f"  LGB (tuned): {lgb_lomo:.4f}", flush=True)
print(f"  XGB (default): {xgb_lomo:.4f}", flush=True)
print(f"  CB (default):  {cb_lomo:.4f}", flush=True)
print(f"  Ensemble:    {lomo_map:.4f}  (E48 was 0.3526, E50 was 0.3625)", flush=True)
print(f"\n  Per-class LOMO:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {lomo_per.get(cls, 0):.4f}", flush=True)

np.save(ROOT / "oof_e70.npy", oof_ens)
np.save(ROOT / "test_e70.npy", test_ens)
save_submission(test_ens, "e70_optuna_flight_priors_base", cv_map=lomo_map)

# ── Apply E54 winter_tilt priors for unseen months ─────────────────
print(f"\n{'='*65}", flush=True)
print("APPLYING E54 WINTER-TILT PRIORS", flush=True)
print(f"{'='*65}", flush=True)

gbif_prior_map = {}
for _, row in gbif_priors_df.iterrows():
    m_val = int(row["month"])
    prior = np.array([row[cls] for cls in CLASSES], dtype=float)
    gbif_prior_map[m_val] = prior / prior.sum()

def apply_priors(preds, months, alphas):
    out = preds.copy()
    n_adj = 0
    for i in range(len(out)):
        m = months[i]
        if m in alphas:
            prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
            adj = (1 - alphas[m]) * out[i] + alphas[m] * prior
            out[i] = adj / adj.sum()
            n_adj += 1
    print(f"  Adjusted {n_adj} rows for months {sorted(alphas.keys())}", flush=True)
    return out

# Variant A — same as E54 (direct comparison)
alphas_winter = {2: 0.22, 5: 0.12, 12: 0.24}
test_A = apply_priors(test_ens, test_months, alphas_winter)
save_submission(test_A, "e70_winter_tilt_m2_0.22_m5_0.12_m12_0.24",
                cv_map=lomo_map)

# Variant B — slightly stronger winter (based on E55 pattern)
alphas_stronger = {2: 0.26, 5: 0.12, 12: 0.28}
test_B = apply_priors(test_ens, test_months, alphas_stronger)
save_submission(test_B, "e70_stronger_winter_m2_0.26_m5_0.12_m12_0.28",
                cv_map=lomo_map)

# ── Weight sweep: find best LGB/XGB/CB blend on LOMO OOF ──────────
print(f"\n{'='*65}", flush=True)
print("ENSEMBLE WEIGHT SWEEP (LOMO OOF)", flush=True)
print(f"{'='*65}", flush=True)

best_w_map, best_w = -1.0, (W_LGB, W_XGB, W_CB)
for w_lgb in [0.2, 0.3, 0.4, 0.5]:
    for w_cb in [0.2, 0.3, 0.4, 0.5]:
        w_xgb = round(1.0 - w_lgb - w_cb, 2)
        if w_xgb < 0.05:
            continue
        oof_sw = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        sw_map, _ = compute_map(y, oof_sw)
        if sw_map > best_w_map:
            best_w_map = sw_map
            best_w = (w_lgb, w_xgb, w_cb)

print(f"  Best weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, "
      f"CB={best_w[2]:.2f}  →  LOMO {best_w_map:.4f}", flush=True)

if abs(best_w_map - lomo_map) > 0.002:
    # Worth using tuned weights
    test_ens_tuned = (best_w[0] * test_lgb
                      + best_w[1] * test_xgb
                      + best_w[2] * test_cb)
    test_C = apply_priors(test_ens_tuned, test_months, alphas_winter)
    save_submission(test_C,
                    f"e70_tunedw_lgb{best_w[0]}_xgb{best_w[1]}_cb{best_w[2]}",
                    cv_map=best_w_map)
    print(f"  Saved tuned-weight variant (Variant C)", flush=True)
else:
    print(f"  Weight difference < 0.002 — default weights sufficient", flush=True)

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*65}", flush=True)
print(f"  LOMO: {lomo_map:.4f}  vs E48=0.3526 / E50=0.3625 / E42=0.3799", flush=True)
print(f"  Baseline (E54): LB 0.56", flush=True)
print(f"  Submit A first (same alphas as E54, better model).", flush=True)
print(f"  If A > 0.56: submit B (stronger winter) next.", flush=True)
print(f"  If A = 0.56: model improvement not reflected in LB.", flush=True)
print("\nDone.", flush=True)
