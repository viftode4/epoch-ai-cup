"""E48: External morphology + flight priors (LOMO-first).

Adds class-level priors from AVONET + BirdWingData + Col de la Croix and
evaluates four configurations:
  A) E38 base
  B) E38 + morphology priors
  C) E38 + flight priors
  D) E38 + morphology + flight priors

Primary metric: LOMO mAP.
Secondary metric: SKF mAP for the best LOMO config (used for submission).
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.external_priors import build_external_class_priors, priors_to_frame
from src.features import ALL_TEMPORAL, add_external_prior_features, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999
W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 47,
    "max_depth": 7,
    "min_child_samples": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
    "device": "cpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "seed": 42,
    "nthread": -1,
    "verbosity": 0,
    "device": "cpu",
    "tree_method": "hist",
}
CB_PARAMS = {
    "iterations": 1400,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 80,
    "task_type": "CPU",
}


def train_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label):
    """Train LGB + XGB + CB on one fold."""
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=1400,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    oof_lgb = m_lgb.predict(X_va)
    test_lgb = m_lgb.predict(X_test) if X_test is not None else None

    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn)
    dva_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=fn)
    m_xgb = xgb.train(
        XGB_PARAMS,
        dtr_xgb,
        num_boost_round=1400,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=80,
        verbose_eval=0,
    )
    oof_xgb = m_xgb.predict(dva_xgb)
    test_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    cb = CatBoostClassifier(**CB_PARAMS)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), sample_weight=w_tr, verbose=0)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb if X_test is not None else None

    m, _ = compute_map(y_va, oof)
    print(f"  {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_ens


def add_weather_solar_gbif(train_feats, test_feats, train_months, test_months):
    """Add E38 weather + solar + GBIF features."""
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

    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    gbif_si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        si = np.ones(N_CLASSES)
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                si[i] = 1.0
            else:
                class_counts = gbif[cls].values
                class_mean = class_counts.mean()
                si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        gbif_si[month] = si

    for i, cls in enumerate(CLASSES):
        col = f"gbif_si_{cls.lower().replace(' ', '_')}"
        train_feats[col] = [gbif_si[m][i] for m in train_months]
        test_feats[col] = [gbif_si[m][i] for m in test_months]

    gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
    month_entropy = {}
    for _, row in gbif_priors_df.iterrows():
        month = int(row["month"])
        probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
        month_entropy[month] = -np.sum(probs * np.log(probs))

    train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
    test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]
    return train_feats, test_feats


print("=" * 70, flush=True)
print("E48 EXTERNAL PRIORS (MORPH + FLIGHT)".center(70), flush=True)
print("=" * 70, flush=True)

print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

print("\nBuilding E38 base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_base = build_features(train_df, feature_sets=feat_sets)
test_base = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_base.columns if c not in ALL_TEMPORAL]
train_base = train_base[keep]
test_base = test_base[keep]
train_base, test_base = add_weather_solar_gbif(train_base, test_base, train_months, test_months)
train_base = train_base.replace([np.inf, -np.inf], np.nan).fillna(0)
test_base = test_base.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  Base features: {len(train_base.columns)}", flush=True)

print("\nLoading external class priors...", flush=True)
priors = build_external_class_priors(ROOT)
priors_df = priors_to_frame(priors)
display_cols = [
    "class",
    "mass_g",
    "wing_mm",
    "hwi",
    "speed_ms",
    "wingbeat_hz",
    "expected_rcs_db",
    "n_avonet",
    "n_birdwing",
]
print(priors_df[display_cols].round(3).to_string(index=False), flush=True)

train_morph = add_external_prior_features(
    train_base.copy(), priors, include_morph=True, include_flight=False
)
test_morph = add_external_prior_features(
    test_base.copy(), priors, include_morph=True, include_flight=False
)
train_flight = add_external_prior_features(
    train_base.copy(), priors, include_morph=False, include_flight=True
)
test_flight = add_external_prior_features(
    test_base.copy(), priors, include_morph=False, include_flight=True
)
train_both = add_external_prior_features(
    train_base.copy(), priors, include_morph=True, include_flight=True
)
test_both = add_external_prior_features(
    test_base.copy(), priors, include_morph=True, include_flight=True
)

configs = {
    "A: E38 base": (train_base, test_base),
    "B: +morph priors": (train_morph, test_morph),
    "C: +flight priors": (train_flight, test_flight),
    "D: +morph+flight priors": (train_both, test_both),
}

print("\nFeature counts by config:", flush=True)
for cfg, (tr_df, _) in configs.items():
    print(f"  {cfg:<25s}: {len(tr_df.columns)}", flush=True)

print("\n" + "=" * 70, flush=True)
print("LOMO EVALUATION (PRIMARY)".center(70), flush=True)
print("=" * 70, flush=True)

lomo_results = {}
for cfg_name, (tr_df, _) in configs.items():
    print(f"\n--- {cfg_name} ---", flush=True)
    cols = list(tr_df.columns)
    X = tr_df.values.astype(np.float32)
    oof_lomo = np.zeros((len(y), N_CLASSES))

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        oof_fold, _ = train_fold(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            sample_weights[tr_idx],
            None,
            cols,
            f"LOMO month {month}",
        )
        oof_lomo[va_idx] = oof_fold

    lomo_map, lomo_per = compute_map(y, oof_lomo)
    lomo_results[cfg_name] = {
        "map": lomo_map,
        "per": lomo_per,
        "oof_lomo": oof_lomo,
    }
    print_results(lomo_map, lomo_per, label=f"{cfg_name} -- LOMO")

print("\n" + "=" * 70, flush=True)
print("LOMO SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
base_lomo = lomo_results["A: E38 base"]["map"]
print(f"\n  {'Config':<28s} {'LOMO':>8s} {'Delta':>8s}", flush=True)
print(f"  {'-' * 48}", flush=True)
for cfg_name, res in lomo_results.items():
    delta = res["map"] - base_lomo
    delta_str = f"{delta:+.4f}" if cfg_name != "A: E38 base" else "---"
    print(f"  {cfg_name:<28s} {res['map']:>8.4f} {delta_str:>8s}", flush=True)

best_cfg = max(lomo_results, key=lambda k: lomo_results[k]["map"])
best_lomo = lomo_results[best_cfg]["map"]
print(f"\nBest by LOMO: {best_cfg} ({best_lomo:.4f})", flush=True)

print("\n" + "=" * 70, flush=True)
print("SKF EVALUATION + TEST PREDICTIONS (BEST LOMO CONFIG)".center(70), flush=True)
print("=" * 70, flush=True)

train_best, test_best = configs[best_cfg]
best_cols = list(train_best.columns)
X = train_best.values.astype(np.float32)
X_test = test_best.values.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.zeros((len(y), N_CLASSES))
test_pred = np.zeros((len(X_test), N_CLASSES))
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    oof_fold, test_fold = train_fold(
        X[tr_idx],
        y[tr_idx],
        X[va_idx],
        y[va_idx],
        sample_weights[tr_idx],
        X_test,
        best_cols,
        f"SKF fold {fold_idx}",
    )
    oof_skf[va_idx] = oof_fold
    test_pred += test_fold / 5.0

skf_map, skf_per = compute_map(y, oof_skf)
print_results(skf_map, skf_per, label=f"{best_cfg} -- SKF")
print(f"\n  LOMO (best config): {best_lomo:.4f}", flush=True)
print(f"  SKF (best config):  {skf_map:.4f}", flush=True)
print(f"  Gap:                {skf_map - best_lomo:+.4f}", flush=True)

print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e48.npy", oof_skf)
np.save(ROOT / "oof_e48_lomo.npy", lomo_results[best_cfg]["oof_lomo"])
np.save(ROOT / "test_e48.npy", test_pred)
save_submission(test_pred, "e48_external_priors", cv_map=skf_map)
print("Saved: oof_e48.npy, oof_e48_lomo.npy, test_e48.npy", flush=True)

print("\nDone.", flush=True)
