"""E160: Optuna feature group + HP selection on Kaggle GPU.

Builds ALL available features (164+), uses Optuna to find the best
feature group combination + LGB hyperparameters, then retrains a full
LGB+XGB+CB ensemble with the winning config.

Objective: test-weighted proxy fold mAP (optimizes for ALL test months).
"""
import sys, os, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Kaggle path discovery ─────────────────────────────────────────
INPUT = Path("/kaggle/input")

def _find_file(marker):
    for p in INPUT.rglob(marker):
        return p
    raise FileNotFoundError(f"Cannot find {marker} under {INPUT}")

_data_py = _find_file(os.path.join("src", "data.py"))
SRC_PARENT = _data_py.parent.parent
_best_feat = _find_file("best_features.txt")
EXT_DATA_DIR = _best_feat.parent

COMP_DIR = None
for p in INPUT.rglob("train.csv"):
    if "sample_submission.csv" in [f.name for f in p.parent.iterdir()]:
        COMP_DIR = p.parent
        break
if COMP_DIR is None:
    raise FileNotFoundError("Cannot find competition data")

sys.path.insert(0, str(SRC_PARENT))

import src.data as _dm
_dm.ROOT = SRC_PARENT
_dm.DATA_DIR = COMP_DIR

print(f"src/ at {SRC_PARENT}", flush=True)
print(f"ext data at {EXT_DATA_DIR}", flush=True)
print(f"comp data at {COMP_DIR}", flush=True)

# ── Imports ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data import CLASSES, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results

N_CLASSES = len(CLASSES)
SEED = 42

print("=" * 70, flush=True)
print("E160: OPTUNA FEATURE+HP SELECTION (KAGGLE)".center(70), flush=True)
print("=" * 70, flush=True)

# ── Load data ─────────────────────────────────────────────────────
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

# ── Build ALL features ────────────────────────────────────────────
print("\nBuilding ALL feature sets...", flush=True)
ALL_FEAT_SETS = [
    "core", "rcs_fft", "tabular", "targeted",
    "flight_mode", "weakclass", "rcs_slope", "trajectory_separators",
    "radar_physics", "absolute_wingbeat", "linearity", "enhanced_bio_shape",
]
t0 = time.time()
train_feats = build_features(train_df, feature_sets=ALL_FEAT_SETS)
test_feats = build_features(test_df, feature_sets=ALL_FEAT_SETS)
print(f"  Feature extraction: {time.time()-t0:.1f}s", flush=True)

# Remove temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar from external data
for name, prefix in [("weather", "wx_"), ("solar", "sol_")]:
    train_ext = pd.read_csv(EXT_DATA_DIR / f"train_{name}.csv")
    test_ext = pd.read_csv(EXT_DATA_DIR / f"test_{name}.csv")
    for col in train_ext.columns:
        train_feats[f"{prefix}{col}"] = train_ext[col].values
        test_feats[f"{prefix}{col}"] = test_ext[col].values

all_feature_names = list(train_feats.columns)
X_all = train_feats.values.astype(np.float32)
X_test_all = test_feats.values.astype(np.float32)
print(f"  Total features: {len(all_feature_names)}", flush=True)

# ── Feature groups ────────────────────────────────────────────────
old_36 = open(EXT_DATA_DIR / "best_features.txt").read().strip().split("\n")
old_36 = [f.strip() for f in old_36 if f.strip()]

traj_sep_names = ["heading_R", "rcs_spectral_entropy", "speed_autocorr",
                  "alt_ascending_frac", "alt_descending_frac", "alt_flat_frac",
                  "soaring_frac", "rcs_burst_frac", "rcs_smooth_frac"]

bio_shape_names = ["turn_dir_consistency", "max_sustained_turn_frac", "turn_reversal_rate",
                   "rcs_dominant_ac_lag", "rcs_flap_regularity", "rcs_glide_flap_var_ratio",
                   "rcs_burst_fraction", "path_loop_fraction"]

FEATURE_GROUPS = {}
FEATURE_GROUPS["old_36"] = [f for f in old_36 if f in all_feature_names]
FEATURE_GROUPS["traj_sep"] = [f for f in traj_sep_names if f in all_feature_names]
FEATURE_GROUPS["rcs_fft"] = [f for f in all_feature_names
                              if f.startswith("rcs_") and f not in old_36 and f not in traj_sep_names]
FEATURE_GROUPS["speed_extras"] = [f for f in all_feature_names
                                   if any(f.startswith(p) for p in ["speed_", "accel_"])
                                   and f not in old_36 and f not in traj_sep_names]
FEATURE_GROUPS["alt_extras"] = [f for f in all_feature_names
                                 if f.startswith("alt_") and f not in old_36 and f not in traj_sep_names]
flight_feats = ["climb_frac", "descent_frac", "level_frac", "net_turning",
                "turning_rate_std", "circling_index"]
FEATURE_GROUPS["flight_mode"] = [f for f in flight_feats if f in all_feature_names and f not in old_36]
FEATURE_GROUPS["geo_extras"] = [f for f in all_feature_names
                                 if any(f.startswith(p) for p in ["lon_", "lat_", "dist_"])
                                 and f not in old_36]
FEATURE_GROUPS["interactions"] = [f for f in all_feature_names
                                   if any(x in f for x in ["_x_", "_vs_", "_per_", "_for_", "ratio"])
                                   and f not in old_36]
FEATURE_GROUPS["rcs_slope"] = [f for f in ["rcs_slope"] if f in all_feature_names and f not in old_36]
FEATURE_GROUPS["radar_physics"] = [f for f in all_feature_names if f.startswith("rp_")]
FEATURE_GROUPS["wingbeat"] = [f for f in all_feature_names if f.startswith("wb_")]
FEATURE_GROUPS["linearity"] = [f for f in all_feature_names if f.startswith("lin_")]
FEATURE_GROUPS["bio_shape"] = [f for f in bio_shape_names if f in all_feature_names]

# Everything else
assigned = set(old_36 + traj_sep_names)
for gname in FEATURE_GROUPS:
    if gname != "old_36":
        assigned.update(FEATURE_GROUPS[gname])
FEATURE_GROUPS["other"] = [f for f in all_feature_names
                            if f not in assigned
                            and not f.startswith("wx_") and not f.startswith("sol_")]

print("\nFeature groups:", flush=True)
for gname, gfeats in FEATURE_GROUPS.items():
    if gfeats:
        print(f"  {gname:20s}: {len(gfeats):>3d} features", flush=True)

feat_to_idx = {f: i for i, f in enumerate(all_feature_names)}

# ── Month masks ───────────────────────────────────────────────────
jan_mask = months == 1
apr_mask = months == 4
sep_mask = months == 9
oct_mask = months == 10

# Class weights
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]


def proxy_fold_map(oof, mask):
    """Compute macro mAP on a month subset."""
    va_y = y[mask]
    va_oof = oof[mask]
    if len(va_y) == 0:
        return 0.0
    y_bin = np.zeros((len(va_y), N_CLASSES), dtype=int)
    y_bin[np.arange(len(va_y)), va_y] = 1
    aps = []
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() > 0:
            aps.append(average_precision_score(y_bin[:, c], va_oof[:, c]))
    return np.mean(aps) if aps else 0.0


def eval_features_skf(feature_indices, lgb_params, n_folds=5):
    """Train LGB with SKF, return per-fold proxy mAPs."""
    X = X_all[:, feature_indices]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X, y):
        lgb = LGBMClassifier(**lgb_params, random_state=SEED, verbose=-1,
                             device="gpu", n_jobs=-1)
        lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])])
        oof[va_idx] = lgb.predict_proba(X[va_idx])

    jan_map = proxy_fold_map(oof, jan_mask)
    apr_map = proxy_fold_map(oof, apr_mask)
    sep_map = proxy_fold_map(oof, sep_mask)
    oct_map = proxy_fold_map(oof, oct_mask)

    # Test-weighted objective
    objective = 0.165 * jan_map + 0.162 * apr_map + 0.244 * sep_map + 0.429 * oct_map

    return oof, {"jan": jan_map, "apr": apr_map, "sep": sep_map,
                 "oct": oct_map, "objective": objective}


# ── Optuna optimization ──────────────────────────────────────────
N_TRIALS = 200
print(f"\n--- Optuna optimization ({N_TRIALS} trials) ---", flush=True)
print("  Objective: test-weighted proxy fold mAP", flush=True)
print("  Weights: Oct=42.9%, Sep=24.4%, May=16.2%, Feb+Dec=16.5%\n", flush=True)

best_results = {"objective": -1.0}
trial_log = []


def objective(trial):
    global best_results

    # Feature group selection (old_36 always included)
    selected_features = list(FEATURE_GROUPS["old_36"])

    for gname in FEATURE_GROUPS:
        if gname == "old_36":
            continue
        if len(FEATURE_GROUPS[gname]) == 0:
            continue
        if trial.suggest_categorical(f"use_{gname}", [True, False]):
            selected_features.extend(FEATURE_GROUPS[gname])

    # Allow dropping weather/solar subgroups
    drop_weather = trial.suggest_categorical("drop_weather", [True, False])
    drop_solar = trial.suggest_categorical("drop_solar", [True, False])
    if drop_weather:
        selected_features = [f for f in selected_features if not f.startswith("wx_")]
    if drop_solar:
        selected_features = [f for f in selected_features if not f.startswith("sol_")]

    # LGB hyperparameters
    lgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127, step=16),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "class_weight": "balanced",
    }

    feat_indices = [feat_to_idx[f] for f in selected_features if f in feat_to_idx]
    if len(feat_indices) < 10:
        return 0.0

    oof, results = eval_features_skf(feat_indices, lgb_params, n_folds=5)

    obj = results["objective"]

    if obj > best_results["objective"]:
        best_results = results.copy()
        best_results["features"] = selected_features
        best_results["lgb_params"] = lgb_params
        best_results["n_features"] = len(feat_indices)
        print(f"  NEW BEST (trial {trial.number}): obj={obj:.4f} "
              f"Jan={results['jan']:.3f} Apr={results['apr']:.3f} "
              f"Sep={results['sep']:.3f} Oct={results['oct']:.3f} "
              f"nfeat={len(feat_indices)}", flush=True)

    trial_log.append({
        "trial": trial.number, "objective": obj,
        **{k: v for k, v in results.items() if k != "objective"},
        "n_features": len(feat_indices),
    })

    return obj


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS)
elapsed = time.time() - t0
print(f"\n  Optuna done in {elapsed/60:.1f} min", flush=True)

# ── Best configuration ───────────────────────────────────────────
print(f"\n{'=' * 70}", flush=True)
print("  BEST CONFIGURATION", flush=True)
print(f"{'=' * 70}", flush=True)

best = study.best_trial
print(f"  Trial: {best.number}", flush=True)
print(f"  Objective: {best.value:.4f}", flush=True)
print(f"  Features: {best_results['n_features']}", flush=True)
print(f"  Jan (->Feb+Dec): {best_results['jan']:.4f}", flush=True)
print(f"  Apr (->May):     {best_results['apr']:.4f}", flush=True)
print(f"  Sep (shared):    {best_results['sep']:.4f}", flush=True)
print(f"  Oct (shared):    {best_results['oct']:.4f}", flush=True)

print(f"\n  Feature groups:", flush=True)
for gname in FEATURE_GROUPS:
    if gname == "old_36":
        continue
    key = f"use_{gname}"
    if key in best.params:
        status = "ON" if best.params[key] else "OFF"
        n = len(FEATURE_GROUPS[gname])
        print(f"    {gname:20s}: {status:3s} ({n} feats)", flush=True)
print(f"    {'drop_weather':20s}: {best.params.get('drop_weather', False)}", flush=True)
print(f"    {'drop_solar':20s}: {best.params.get('drop_solar', False)}", flush=True)

print(f"\n  LGB params:", flush=True)
for k in ["n_estimators", "learning_rate", "num_leaves", "max_depth",
           "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]:
    print(f"    {k:20s}: {best.params[k]}", flush=True)

# ── Retrain full ensemble with best features ─────────────────────
print(f"\n--- Full ensemble (LGB+XGB+CB, 5-fold SKF) ---", flush=True)
best_feats = best_results["features"]
best_feat_indices = [feat_to_idx[f] for f in best_feats if f in feat_to_idx]
X_train = X_all[:, best_feat_indices]
X_test = X_test_all[:, best_feat_indices]
best_lgb_params = best_results["lgb_params"]

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    # LGB with optimized params
    lgb = LGBMClassifier(**best_lgb_params, random_state=SEED, verbose=-1,
                         device="gpu", n_jobs=-1)
    lgb.fit(X_train[tr_idx], y[tr_idx], eval_set=[(X_train[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X_train[va_idx])
    test_lgb += lgb.predict_proba(X_test) / N_FOLDS

    # XGB
    xgb = XGBClassifier(
        n_estimators=best_lgb_params["n_estimators"],
        learning_rate=best_lgb_params["learning_rate"],
        max_depth=min(best_lgb_params["max_depth"], 8),
        subsample=best_lgb_params["subsample"],
        colsample_bytree=best_lgb_params["colsample_bytree"],
        reg_alpha=best_lgb_params["reg_alpha"],
        reg_lambda=best_lgb_params["reg_lambda"],
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_train[va_idx])
    test_xgb += xgb.predict_proba(X_test) / N_FOLDS

    # CatBoost
    cb = CatBoostClassifier(
        iterations=best_lgb_params["n_estimators"],
        learning_rate=best_lgb_params["learning_rate"],
        depth=min(best_lgb_params["max_depth"], 8),
        l2_leaf_reg=5.0, bagging_temperature=1.0,
        class_weights={i: class_weights_arr[i] for i in range(N_CLASSES)},
        random_seed=SEED, verbose=0, task_type="GPU",
    )
    cb.fit(X_train[tr_idx], y[tr_idx],
           eval_set=(X_train[va_idx], y[va_idx]))
    oof_cb[va_idx] = cb.predict_proba(X_train[va_idx])
    test_cb += cb.predict_proba(X_test) / N_FOLDS

# Model scores
m_lgb, _ = compute_map(y, oof_lgb)
m_xgb, _ = compute_map(y, oof_xgb)
m_cb, _ = compute_map(y, oof_cb)
print(f"  LGB mAP: {m_lgb:.4f}", flush=True)
print(f"  XGB mAP: {m_xgb:.4f}", flush=True)
print(f"  CB  mAP: {m_cb:.4f}", flush=True)

# ── Ensemble weight optimization (proxy-fold objective) ──────────
print("\n--- Ensemble weight optimization ---", flush=True)
best_w, best_obj = None, -1.0
for w_lgb in np.arange(0.0, 1.05, 0.05):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.05):
        w_cb = round(1.0 - w_lgb - w_xgb, 2)
        if w_cb < -0.01:
            continue
        oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        jan_m = proxy_fold_map(oof_ens, jan_mask)
        apr_m = proxy_fold_map(oof_ens, apr_mask)
        sep_m = proxy_fold_map(oof_ens, sep_mask)
        oct_m = proxy_fold_map(oof_ens, oct_mask)
        obj = 0.165 * jan_m + 0.162 * apr_m + 0.244 * sep_m + 0.429 * oct_m
        if obj > best_obj:
            best_obj = obj
            best_w = (w_lgb, w_xgb, w_cb)

w_lgb, w_xgb, w_cb = best_w
oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cb * test_cb

print(f"  Best weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CB={w_cb:.2f}", flush=True)
print(f"  Proxy objective: {best_obj:.4f}", flush=True)

# ── Full evaluation ──────────────────────────────────────────────
m_ens, per_ens = compute_map(y, oof_ens)
print_results(m_ens, per_ens, label="E160 optimized ensemble (SKF OOF)")

print(f"\n  Proxy fold breakdown:", flush=True)
for name, mask in [("Jan (->Feb+Dec)", jan_mask), ("Apr (->May)", apr_mask),
                   ("Sep (shared)", sep_mask), ("Oct (shared)", oct_mask)]:
    m = proxy_fold_map(oof_ens, mask)
    print(f"    {name:20s}: {m:.4f}", flush=True)

# ── Feature importance ────────────────────────────────────────────
print(f"\n--- Top 30 feature importances ---", flush=True)
imp = lgb.feature_importances_
feat_names_used = best_feats[:len(best_feat_indices)]
idx_sorted = np.argsort(imp)[::-1][:30]
new_feats = set(traj_sep_names + bio_shape_names + [f for f in all_feature_names if f.startswith(("rp_", "wb_", "lin_"))])
for rank, i in enumerate(idx_sorted):
    marker = " [NEW]" if feat_names_used[i] in new_feats else ""
    print(f"  {rank+1:>2d}. {feat_names_used[i]:>30s}: {imp[i]:>5d}{marker}", flush=True)

# ── Save submission ──────────────────────────────────────────────
sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
sub_columns = [c for c in sample_sub.columns if c != "track_id"]
sub = pd.DataFrame({"track_id": test_df["track_id"]})
for col in sub_columns:
    cls_idx = CLASSES.index(col)
    sub[col] = test_ens[:, cls_idx]
sub.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv ({len(sub)} rows)", flush=True)

# Save OOF + test arrays
np.save("oof_e160.npy", oof_ens)
np.save("test_e160.npy", test_ens)

# Save best feature list
with open("best_features_e160.txt", "w") as f:
    for feat in sorted(best_feats):
        f.write(feat + "\n")

print(f"\nSaved oof_e160.npy, test_e160.npy, best_features_e160.txt", flush=True)
print("\nDone.", flush=True)
