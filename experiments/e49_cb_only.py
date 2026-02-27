"""E49: CatBoost Only -- Tuned params from E48

E48 showed CB solo LOMO = 0.4023, beating the 3-model ensemble (0.3884).
Weight grid confirmed: 80% CB optimal. LGB/XGB hurt on LOMO.

This experiment:
  A: CB-only with E48 best params, LOMO eval
  B: Multi-seed CB (seeds 42,123,777,2024,9999) for variance reduction
  C: CB with deeper Optuna tuning (30 more trials around E48 best)
  D: Best config -> SKF + test predictions
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999
P = lambda *a, **kw: print(*a, **kw, flush=True)

# E48 best CB params
CB_TUNED = {
    "iterations": 2000,
    "learning_rate": 0.06539588141518164,
    "depth": 7,
    "l2_leaf_reg": 1.2175669230733548,
    "bagging_temperature": 1.2459595026927008,
    "random_strength": 1.1621399282689326,
    "min_data_in_leaf": 14,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "verbose": 0,
    "early_stopping_rounds": 80,
    "task_type": "GPU",
}

# ======================================================================
# Load data
# ======================================================================
P("=" * 60)
P("E49: CATBOOST ONLY -- TUNED + MULTI-SEED + DEEPER TUNING")
P("=" * 60)

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
unique_months = sorted(np.unique(train_months))

# Build features
P("\nBuilding features...")
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Load weather + solar
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

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

# 36 best features from backward elimination
best_feats = [f.strip() for f in open(ROOT / "data" / "best_features.txt") if f.strip()]
available = [c for c in best_feats if c in train_feats.columns]
P(f"Using {len(available)} features")

X_all = train_feats[available].values.astype(np.float32)
X_test_all = test_feats[available].values.astype(np.float32)
fn = list(available)


def lomo_cb(params, seed=42):
    """Run CB LOMO and return score + OOF predictions."""
    p = {**params, "random_seed": seed}
    oof = np.zeros((len(y), N_CLASSES))
    for m in unique_months:
        va = np.where(train_months == m)[0]
        tr = np.where(train_months != m)[0]
        cb = CatBoostClassifier(**p)
        cb.fit(X_all[tr], y[tr], eval_set=(X_all[va], y[va]),
               verbose=0, sample_weight=sample_weights[tr])
        oof[va] = cb.predict_proba(X_all[va])
    score, per = compute_map(y, oof)
    return score, per, oof


# ======================================================================
# Part A: CB-only with E48 tuned params
# ======================================================================
P("\n" + "=" * 60)
P("PART A: CB-ONLY WITH E48 TUNED PARAMS")
P("=" * 60)

t0 = time.time()
score_a, per_a, oof_a = lomo_cb(CB_TUNED, seed=42)
P(f"\n  CB-only LOMO: {score_a:.4f} ({time.time()-t0:.0f}s)")
P(f"  Per-class:")
for cls in CLASSES:
    P(f"    {cls:<15s}: {per_a.get(cls, 0):.4f}")

# ======================================================================
# Part B: Multi-seed averaging
# ======================================================================
P("\n" + "=" * 60)
P("PART B: MULTI-SEED CB (5 seeds)")
P("=" * 60)

seeds = [42, 123, 777, 2024, 9999]
oof_seeds = []
t0 = time.time()

for s in seeds:
    sc, _, oof_s = lomo_cb(CB_TUNED, seed=s)
    oof_seeds.append(oof_s)
    P(f"  Seed {s}: LOMO = {sc:.4f}")

oof_multi = np.mean(oof_seeds, axis=0)
multi_score, multi_per = compute_map(y, oof_multi)
P(f"\n  Multi-seed avg LOMO: {multi_score:.4f} ({time.time()-t0:.0f}s)")
P(f"  Delta vs single seed: {multi_score - score_a:+.4f}")
P(f"  Per-class:")
for cls in CLASSES:
    P(f"    {cls:<15s}: {multi_per.get(cls, 0):.4f}")

# ======================================================================
# Part C: Deeper Optuna around E48 best (narrower ranges)
# ======================================================================
P("\n" + "=" * 60)
P("PART C: DEEPER OPTUNA TUNING (30 trials, narrow ranges)")
P("=" * 60)

t0 = time.time()


def cb_objective_deep(trial):
    params = {
        "iterations": 3000,  # more iterations
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "depth": trial.suggest_int("depth", 5, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.3, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 3.0),
        "random_strength": trial.suggest_float("random_strength", 0.3, 3.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 25),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "verbose": 0,
        "early_stopping_rounds": 100,
        "task_type": "GPU",
        "random_seed": 42,
    }
    score, _, _ = lomo_cb(params, seed=42)
    return score


study = optuna.create_study(direction="maximize", study_name="cb_deep")
study.optimize(cb_objective_deep, n_trials=30, show_progress_bar=False)

P(f"\n  Deep tuning best LOMO: {study.best_value:.4f} ({time.time()-t0:.0f}s)")
P(f"  Best params: {study.best_params}")

# Decide best config
best_configs = {
    "E48 tuned (seed=42)": (score_a, CB_TUNED),
    "Multi-seed avg": (multi_score, None),
    "Deep tuning": (study.best_value, None),
}
P(f"\n  Config comparison:")
for name, (sc, _) in best_configs.items():
    P(f"    {name:<25s}: LOMO = {sc:.4f}")

# Build final params from deep tuning
deep_params = {
    "iterations": 3000,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "verbose": 0,
    "early_stopping_rounds": 100,
    "task_type": "GPU",
    **study.best_params,
}

# Use whichever is best between E48 params and deep params
if study.best_value > score_a:
    final_params = deep_params
    final_name = "deep_tuned"
    P(f"\n  Using DEEP TUNED params (LOMO={study.best_value:.4f})")
else:
    final_params = CB_TUNED
    final_name = "e48_tuned"
    P(f"\n  Using E48 TUNED params (LOMO={score_a:.4f})")

# ======================================================================
# Part D: Final predictions with multi-seed
# ======================================================================
P("\n" + "=" * 60)
P("PART D: FINAL -- MULTI-SEED CB + SKF TEST PREDICTIONS")
P("=" * 60)

# Multi-seed LOMO for final score
P("\n  Multi-seed LOMO with best params...")
oof_final_seeds = []
t0 = time.time()
for s in seeds:
    sc, _, oof_s = lomo_cb(final_params, seed=s)
    oof_final_seeds.append(oof_s)
    P(f"    Seed {s}: LOMO = {sc:.4f}")

oof_final = np.mean(oof_final_seeds, axis=0)
final_lomo, final_per = compute_map(y, oof_final)
P(f"\n  Final multi-seed LOMO: {final_lomo:.4f}")
P(f"  Per-class:")
for cls in CLASSES:
    P(f"    {cls:<15s}: {final_per.get(cls, 0):.4f}")

# SKF + test predictions (multi-seed)
P(f"\n  SKF + test predictions (multi-seed)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.zeros((len(y), N_CLASSES))
test_pred = np.zeros((len(X_test_all), N_CLASSES))

for s in seeds:
    p = {**final_params, "random_seed": s}
    oof_skf_s = np.zeros((len(y), N_CLASSES))
    test_s = np.zeros((len(X_test_all), N_CLASSES))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        cb = CatBoostClassifier(**p)
        cb.fit(X_all[tr_idx], y[tr_idx],
               eval_set=(X_all[va_idx], y[va_idx]),
               verbose=0, sample_weight=sample_weights[tr_idx])
        oof_skf_s[va_idx] = cb.predict_proba(X_all[va_idx])
        test_s += cb.predict_proba(X_test_all) / 5

    sc, _ = compute_map(y, oof_skf_s)
    P(f"    Seed {s} SKF: {sc:.4f}")
    oof_skf += oof_skf_s / len(seeds)
    test_pred += test_s / len(seeds)

skf_map, skf_per = compute_map(y, oof_skf)
P(f"\n  Multi-seed SKF CV mAP: {skf_map:.4f}")
P(f"  Multi-seed LOMO mAP:   {final_lomo:.4f}")

P(f"\n  Per-class SKF:")
P(f"  {'Class':<15s} {'SKF':>7s} {'LOMO':>7s}")
for cls in CLASSES:
    s = skf_per.get(cls, 0)
    l = final_per.get(cls, 0)
    P(f"  {cls:<15s} {s:>7.4f} {l:>7.4f}")

# Test distribution
P(f"\n  Test class distribution (argmax):")
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    P(f"    {cls:<15s}: {dist[i]}")

# Save
np.save(ROOT / "oof_e49.npy", oof_skf)
np.save(ROOT / "test_e49.npy", test_pred)
save_submission(test_pred, "e49_cb_only", cv_map=skf_map)

# ======================================================================
# Summary
# ======================================================================
P("\n" + "=" * 60)
P("E49 SUMMARY")
P("=" * 60)
P(f"  Features: {len(available)} (pruned)")
P(f"  Model: CatBoost only, multi-seed ({seeds})")
P(f"  Params: {final_name}")
if final_name == "deep_tuned":
    P(f"  Deep params: {study.best_params}")
else:
    P(f"  E48 params: lr=0.065, depth=7, l2=1.22, bagging_temp=1.25")
P(f"  Single-seed LOMO: {score_a:.4f}")
P(f"  Multi-seed LOMO:  {final_lomo:.4f}")
P(f"  SKF CV mAP:       {skf_map:.4f}")
P(f"  Deep Optuna best: {study.best_value:.4f}")
P("\nDone!")
