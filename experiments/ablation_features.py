"""Ablation study: isolate the contribution of each feature group.

Uses single LGB model (fastest) to test each feature combination.
This tells us EXACTLY what each piece adds before we combine anything.

Tests:
  A. core only                           -- floor
  B. core + tabular                      -- how much does metadata help?
  C. core + rcs_fft + tabular            -- E02 equivalent (modular)
  D. core + rcs_fft + tabular + targeted -- full E02 equivalent
  E. core + tabular + wavelet            -- CWT replaces FFT
  F. core + tabular + rcs_fft + wavelet  -- CWT alongside FFT
  G. core + tabular + flight_mode        -- flight mode alone (no FFT/CWT)
  H. core + tabular + targeted + wavelet -- targeted + CWT, no FFT
  I. core + rcs_fft + tabular + targeted + wavelet           -- all signal features
  J. core + rcs_fft + tabular + targeted + flight_mode       -- all except wavelet
  K. core + rcs_fft + tabular + targeted + wavelet + flight_mode -- kitchen sink

Then: model ablation on the best feature set
  - LGB alone
  - XGB alone
  - CB alone
  - LGB+XGB+CB ensemble
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.features import build_features
from src.metrics import compute_map

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

lgb_params = {
    "objective": "multiclass", "num_class": len(CLASSES),
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "is_unbalance": True,
}


def run_lgb_cv(X, y, label=""):
    """Run LGB 5-fold CV, return (mAP, per_class_dict)."""
    n_classes = len(CLASSES)
    oof = np.zeros((len(X), n_classes))
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_names)
        dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=feature_names,
                           reference=dtrain)
        model = lgb.train(lgb_params, dtrain, num_boost_round=2000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof[va_idx] = model.predict(X[va_idx])

    return compute_map(y, oof)


def run_xgb_cv(X, y):
    """Run XGB 5-fold CV."""
    n_classes = len(CLASSES)
    oof = np.zeros((len(X), n_classes))
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    class_counts = np.bincount(y, minlength=n_classes)
    class_weights = len(y) / (n_classes * class_counts)
    sample_weights = np.array([class_weights[yi] for yi in y])

    xgb_params = {
        "objective": "multi:softprob", "num_class": n_classes,
        "eval_metric": "mlogloss", "learning_rate": 0.05,
        "max_depth": 6, "min_child_weight": 3,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
        "seed": 42, "nthread": -1, "verbosity": 0,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx],
                             feature_names=feature_names)
        dval = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=feature_names)
        model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")], early_stopping_rounds=80,
                          verbose_eval=0)
        oof[va_idx] = model.predict(dval)

    return compute_map(y, oof)


def run_cb_cv(X, y):
    """Run CatBoost 5-fold CV."""
    n_classes = len(CLASSES)
    oof = np.zeros((len(X), n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            auto_class_weights="Balanced",
        )
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0)
        oof[va_idx] = model.predict_proba(X[va_idx])

    return compute_map(y, oof)


# ── Load data ─────────────────────────────────────────────────────
print("Loading data...", flush=True)
train = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train["bird_group"])

# ── Pre-extract ALL feature sets (extract once, subset later) ─────
print("\nExtracting ALL features (one pass)...", flush=True)
all_feats = build_features(train, ["core", "rcs_fft", "wavelet", "flight_mode", "tabular", "targeted"])
print(f"Total features extracted: {all_feats.shape[1]}", flush=True)

# Identify which columns belong to each feature group
# We'll do this by extracting each group separately and recording the column names
print("Mapping feature groups...", flush=True)
core_cols = list(pd.DataFrame([{}]).columns)  # empty, we'll figure this out differently

# Extract minimal sets to identify columns
_core = build_features(train.iloc[:1], ["core"])
_fft = build_features(train.iloc[:1], ["core", "rcs_fft"])
_wav = build_features(train.iloc[:1], ["core", "wavelet"])
_fm = build_features(train.iloc[:1], ["core", "flight_mode"])
_tab = build_features(train.iloc[:1], ["core", "tabular"])
_tgt = build_features(train.iloc[:1], ["core", "tabular", "targeted"])

core_cols = set(_core.columns)
fft_cols = set(_fft.columns) - core_cols
wavelet_cols = set(_wav.columns) - core_cols
flight_cols = set(_fm.columns) - core_cols
tabular_cols = set(_tab.columns) - core_cols
targeted_cols = set(_tgt.columns) - core_cols - tabular_cols

print(f"  core:     {len(core_cols)} cols")
print(f"  rcs_fft:  {len(fft_cols)} cols")
print(f"  wavelet:  {len(wavelet_cols)} cols")
print(f"  flight:   {len(flight_cols)} cols")
print(f"  tabular:  {len(tabular_cols)} cols")
print(f"  targeted: {len(targeted_cols)} cols")

# ── Define ablation configurations ────────────────────────────────
configs = {
    "A: core":
        list(core_cols),
    "B: core+tab":
        list(core_cols | tabular_cols),
    "C: core+fft+tab":
        list(core_cols | fft_cols | tabular_cols),
    "D: core+fft+tab+tgt":
        list(core_cols | fft_cols | tabular_cols | targeted_cols),
    "E: core+tab+wav":
        list(core_cols | tabular_cols | wavelet_cols),
    "F: core+fft+tab+wav":
        list(core_cols | fft_cols | tabular_cols | wavelet_cols),
    "G: core+tab+flight":
        list(core_cols | tabular_cols | flight_cols),
    "H: core+tab+tgt+wav":
        list(core_cols | tabular_cols | targeted_cols | wavelet_cols),
    "I: core+fft+tab+tgt+wav":
        list(core_cols | fft_cols | tabular_cols | targeted_cols | wavelet_cols),
    "J: core+fft+tab+tgt+flight":
        list(core_cols | fft_cols | tabular_cols | targeted_cols | flight_cols),
    "K: kitchen_sink":
        list(all_feats.columns),
}

# ── Run feature ablations (LGB only) ─────────────────────────────
print("\n" + "=" * 80)
print("PHASE 1: FEATURE ABLATION (LGB only)")
print("=" * 80)

results = []
for name, cols in configs.items():
    t0 = time.time()
    X = all_feats[cols].values.astype(np.float32)
    mAP, per_class = run_lgb_cv(X, y)
    elapsed = time.time() - t0

    weak = [c for c, ap in per_class.items() if ap < 0.6]
    results.append({
        "config": name, "n_feats": len(cols), "mAP": mAP,
        **{c: per_class[c] for c in CLASSES}, "time": elapsed,
    })

    print(f"\n  {name} ({len(cols)} feats) -> mAP = {mAP:.4f}  [{elapsed:.0f}s]")
    for cls in CLASSES:
        marker = " <--" if per_class[cls] < 0.6 else ""
        print(f"    {cls:15s}: {per_class[cls]:.4f}{marker}")

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FEATURE ABLATION SUMMARY (sorted by mAP)")
print("=" * 80)
results_sorted = sorted(results, key=lambda x: x["mAP"], reverse=True)
print(f"{'Config':<30s} {'#F':>3s} {'mAP':>7s} {'Clut':>6s} {'Corm':>6s} "
      f"{'Pig':>6s} {'Duck':>6s} {'Gees':>6s} {'Gull':>6s} {'BoP':>6s} "
      f"{'Wad':>6s} {'Song':>6s}")
print("-" * 110)
for r in results_sorted:
    print(f"{r['config']:<30s} {r['n_feats']:>3d} {r['mAP']:>7.4f} "
          f"{r['Clutter']:>6.3f} {r['Cormorants']:>6.3f} "
          f"{r['Pigeons']:>6.3f} {r['Ducks']:>6.3f} {r['Geese']:>6.3f} "
          f"{r['Gulls']:>6.3f} {r['Birds of Prey']:>6.3f} "
          f"{r['Waders']:>6.3f} {r['Songbirds']:>6.3f}")

# ── Find the best feature set ─────────────────────────────────────
best_config = results_sorted[0]
best_name = best_config["config"]
best_cols = configs[best_name]
print(f"\nBest feature set: {best_name} ({len(best_cols)} feats, mAP={best_config['mAP']:.4f})")

# ── PHASE 2: MODEL ABLATION on best feature set ──────────────────
print("\n" + "=" * 80)
print(f"PHASE 2: MODEL ABLATION on '{best_name}'")
print("=" * 80)

X_best = all_feats[best_cols].values.astype(np.float32)

print("\n  Running LGB...", flush=True)
t0 = time.time()
lgb_map, lgb_per = run_lgb_cv(X_best, y)
lgb_time = time.time() - t0
print(f"  LGB:  mAP = {lgb_map:.4f} [{lgb_time:.0f}s]")

print("  Running XGB...", flush=True)
t0 = time.time()
xgb_map, xgb_per = run_xgb_cv(X_best, y)
xgb_time = time.time() - t0
print(f"  XGB:  mAP = {xgb_map:.4f} [{xgb_time:.0f}s]")

print("  Running CatBoost...", flush=True)
t0 = time.time()
cb_map, cb_per = run_cb_cv(X_best, y)
cb_time = time.time() - t0
print(f"  CB:   mAP = {cb_map:.4f} [{cb_time:.0f}s]")

# Ensemble: get OOF from each, optimize weights
print("  Running ensemble weight search...", flush=True)
n_classes = len(CLASSES)
oof_lgb = np.zeros((len(X_best), n_classes))
oof_xgb = np.zeros((len(X_best), n_classes))
oof_cb = np.zeros((len(X_best), n_classes))
feature_names = [f"f{i}" for i in range(X_best.shape[1])]

class_counts = np.bincount(y, minlength=n_classes)
class_weights = len(y) / (n_classes * class_counts)
sample_weights = np.array([class_weights[yi] for yi in y])

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_best, y)):
    # LGB
    dtrain = lgb.Dataset(X_best[tr_idx], label=y[tr_idx], feature_name=feature_names)
    dval = lgb.Dataset(X_best[va_idx], label=y[va_idx], feature_name=feature_names, reference=dtrain)
    m = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = m.predict(X_best[va_idx])

    # XGB
    xgb_params_local = {
        "objective": "multi:softprob", "num_class": n_classes,
        "eval_metric": "mlogloss", "learning_rate": 0.05,
        "max_depth": 6, "min_child_weight": 3,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
        "seed": 42, "nthread": -1, "verbosity": 0,
    }
    dt = xgb.DMatrix(X_best[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx], feature_names=feature_names)
    dv = xgb.DMatrix(X_best[va_idx], label=y[va_idx], feature_names=feature_names)
    m = xgb.train(xgb_params_local, dt, num_boost_round=2000,
                  evals=[(dv, "val")], early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = m.predict(dv)

    # CB
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        auto_class_weights="Balanced",
    )
    cb.fit(X_best[tr_idx], y[tr_idx], eval_set=(X_best[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X_best[va_idx])

# Weight search
best_ens_map = 0
best_w = (1/3, 1/3, 1/3)
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        ens_map, _ = compute_map(y, oof_ens)
        if ens_map > best_ens_map:
            best_ens_map = ens_map
            best_w = (w1, w2, w3)

_, ens_per = compute_map(y, best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb)

print(f"\n{'Model':<25s} {'mAP':>7s} {'Clut':>6s} {'Corm':>6s} {'Pig':>6s} "
      f"{'Duck':>6s} {'Gees':>6s} {'Gull':>6s} {'BoP':>6s} {'Wad':>6s} {'Song':>6s}")
print("-" * 100)
for label, mAP, per in [("LGB", lgb_map, lgb_per), ("XGB", xgb_map, xgb_per),
                          ("CatBoost", cb_map, cb_per),
                          (f"Ensemble({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f})",
                           best_ens_map, ens_per)]:
    print(f"{label:<25s} {mAP:>7.4f} {per['Clutter']:>6.3f} {per['Cormorants']:>6.3f} "
          f"{per['Pigeons']:>6.3f} {per['Ducks']:>6.3f} {per['Geese']:>6.3f} "
          f"{per['Gulls']:>6.3f} {per['Birds of Prey']:>6.3f} "
          f"{per['Waders']:>6.3f} {per['Songbirds']:>6.3f}")

# Also test: does each model carry its weight?
print("\n--- Pairwise ensemble (is 3 models better than 2?) ---")
for pair_name, oof_a, oof_b, na, nb in [
    ("LGB+XGB", oof_lgb, oof_xgb, "LGB", "XGB"),
    ("LGB+CB", oof_lgb, oof_cb, "LGB", "CB"),
    ("XGB+CB", oof_xgb, oof_cb, "XGB", "CB"),
]:
    best_pair_map = 0
    for w in np.arange(0.0, 1.01, 0.05):
        m, _ = compute_map(y, w * oof_a + (1 - w) * oof_b)
        if m > best_pair_map:
            best_pair_map = m
    print(f"  {pair_name:<12s}: {best_pair_map:.4f}")

print(f"  LGB+XGB+CB  : {best_ens_map:.4f}")
print(f"\n  Weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f}")
