"""E25: Remove Temporal Overfitting Features

Train months: [1,4,9,10], Test months: [2,5,9,10,12].
33% of test is from months NEVER in training (Feb, May, Dec).

Strip features that overfit to specific months/hours.
Keep only trajectory-based features that generalize.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
BETA = 0.999

# ── Data ─────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Effective Number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

# ── Feature configs to test ──────────────────────────────────────
# Overfit temporal features to DROP:
TEMPORAL_OVERFIT = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "timestamp_duration",
    # from targeted features:
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring",
    "hour_bin_3h",
]

# Model params
lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

xgb_params = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), y))


def run_config(X, X_test, feature_names, label):
    """Train LGB+XGB+CB ensemble."""
    sample_weights = np.array([class_w[yi] for yi in y])

    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
        mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = mdl.predict(X_va)
        test_lgb += mdl.predict(X_test) / N_FOLDS

        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
        mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
        oof_xgb[va_idx] = mdl.predict(dval_xgb)
        test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

        cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

        print(f"  Fold {fold} done", flush=True)

    # Optimize weights
    best_map = 0
    best_w = None
    for w1 in np.arange(0.05, 0.50, 0.05):
        for w2 in np.arange(0.05, 0.50, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w1, w2, w3)

    oof = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    m, per = compute_map(y, oof)
    print(f"  {label}: mAP={m:.4f} (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f})", flush=True)
    return oof, test, m, per


# ── Config A: Full features (E15 baseline) ───────────────────────
print("\n" + "=" * 60, flush=True)
print("A) Full features (E15 baseline, 105 feats)", flush=True)
print("=" * 60, flush=True)

FEAT_SETS_FULL = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
train_full = build_features(train_df, feature_sets=FEAT_SETS_FULL)
test_full = build_features(test_df, feature_sets=FEAT_SETS_FULL)
print(f"  Full features: {train_full.shape[1]}", flush=True)

X_full = train_full.values.astype(np.float32)
Xt_full = test_full.values.astype(np.float32)
fn_full = list(train_full.columns)

oof_a, test_a, map_a, per_a = run_config(X_full, Xt_full, fn_full, "Full")
print_results(map_a, per_a, "A) Full (105 feats)")

# ── Config B: Drop overfit temporal features ─────────────────────
print("\n" + "=" * 60, flush=True)
print("B) Drop temporal overfit features", flush=True)
print("=" * 60, flush=True)

keep_cols = [c for c in train_full.columns if c not in TEMPORAL_OVERFIT]
train_notmp = train_full[keep_cols]
test_notmp = test_full[keep_cols]
print(f"  Dropped: {sorted(set(train_full.columns) - set(keep_cols))}", flush=True)
print(f"  Remaining features: {train_notmp.shape[1]}", flush=True)

X_notmp = train_notmp.values.astype(np.float32)
Xt_notmp = test_notmp.values.astype(np.float32)
fn_notmp = list(train_notmp.columns)

oof_b, test_b, map_b, per_b = run_config(X_notmp, Xt_notmp, fn_notmp, "No temporal")
print_results(map_b, per_b, "B) No temporal overfit")

# ── Config C: Core + flight only (no tabular/targeted at all) ────
print("\n" + "=" * 60, flush=True)
print("C) Core + flight only (pure trajectory features)", flush=True)
print("=" * 60, flush=True)

FEAT_SETS_TRAJ = ["core", "rcs_fft", "flight_mode"]
train_traj = build_features(train_df, feature_sets=FEAT_SETS_TRAJ)
test_traj = build_features(test_df, feature_sets=FEAT_SETS_TRAJ)

# Add ONLY generalizable tabular features (airspeed, altitude, size)
for df_feat, df_orig in [(train_traj, train_df), (test_traj, test_df)]:
    df_feat["airspeed"] = df_orig["airspeed"].values
    df_feat["min_z"] = df_orig["min_z"].values
    df_feat["max_z"] = df_orig["max_z"].values
    df_feat["z_range"] = df_orig["max_z"].values - df_orig["min_z"].values
    df_feat["z_mean"] = (df_orig["max_z"].values + df_orig["min_z"].values) / 2
    size_map = {"Small bird": 0, "Medium bird": 1, "Large bird": 2, "Flock": 3}
    df_feat["radar_bird_size"] = df_orig["radar_bird_size"].map(size_map).values
    df_feat["airspeed_vs_ground"] = df_feat["airspeed"] / df_feat["avg_ground_speed"].clip(lower=0.01)
    # Size interactions (generalizable)
    df_feat["size_x_airspeed"] = df_feat["radar_bird_size"] * df_feat["airspeed"]
    df_feat["size_x_rcs"] = df_feat["radar_bird_size"] * df_feat["rcs_mean"]
    df_feat["size_x_alt"] = df_feat["radar_bird_size"] * df_feat["alt_mean"]
    df_feat["airspeed_high"] = (df_feat["airspeed"] > 17).astype(int)
    df_feat["airspeed_low"] = (df_feat["airspeed"] < 12).astype(int)
    df_feat["duration_short"] = (df_feat["duration"] < 25).astype(int)
    df_feat["duration_long"] = (df_feat["duration"] > 60).astype(int)
    # Size one-hot
    for name, val in [("small_bird", 0), ("medium", 1), ("large", 2), ("flock", 3)]:
        df_feat[f"is_{name}"] = (df_feat["radar_bird_size"] == val).astype(int)

train_traj = train_traj.replace([np.inf, -np.inf], np.nan).fillna(0)
test_traj = test_traj.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"  Trajectory features: {train_traj.shape[1]}", flush=True)

X_traj = train_traj.values.astype(np.float32)
Xt_traj = test_traj.values.astype(np.float32)
fn_traj = list(train_traj.columns)

oof_c, test_c, map_c, per_c = run_config(X_traj, Xt_traj, fn_traj, "Trajectory only")
print_results(map_c, per_c, "C) Trajectory + generalizable tabular")

# ── Config D: B + weakclass features ─────────────────────────────
print("\n" + "=" * 60, flush=True)
print("D) No temporal + weakclass features", flush=True)
print("=" * 60, flush=True)

FEAT_SETS_WC = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_wc = build_features(train_df, feature_sets=FEAT_SETS_WC)
test_wc = build_features(test_df, feature_sets=FEAT_SETS_WC)

# Drop temporal overfit
keep_wc = [c for c in train_wc.columns if c not in TEMPORAL_OVERFIT]
train_wc = train_wc[keep_wc]
test_wc = test_wc[keep_wc]
print(f"  Features: {train_wc.shape[1]}", flush=True)

X_wc = train_wc.values.astype(np.float32)
Xt_wc = test_wc.values.astype(np.float32)
fn_wc = list(train_wc.columns)

oof_d, test_d, map_d, per_d = run_config(X_wc, Xt_wc, fn_wc, "No temporal + weakclass")
print_results(map_d, per_d, "D) No temporal + weakclass")

# ── Logit adjustment on each ─────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Logit adjustment on all configs", flush=True)
print("=" * 60, flush=True)

priors = counts / counts.sum()

def apply_logit_adj(oof, test_preds, label):
    tau = np.zeros(N_CLASSES)
    best_map, _ = compute_map(y, oof)
    for iteration in range(3):
        improved = False
        for c in range(N_CLASSES):
            best_t = tau[c]
            best_m = best_map
            for t in np.arange(-0.5, 1.51, 0.02):
                tau[c] = t
                adj = priors ** (-tau)
                a = oof * adj[None, :]
                a = a / a.sum(axis=1, keepdims=True)
                m, _ = compute_map(y, a)
                if m > best_m:
                    best_m = m
                    best_t = t
            tau[c] = best_t
            if best_m > best_map:
                best_map = best_m
                improved = True
        if not improved:
            break
    adj = priors ** (-tau)
    oof_adj = oof * adj[None, :]
    oof_adj = oof_adj / oof_adj.sum(axis=1, keepdims=True)
    test_adj = test_preds * adj[None, :]
    test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)
    m, per = compute_map(y, oof_adj)
    print(f"  {label}: {m:.4f}", flush=True)
    return oof_adj, test_adj, m, per

oof_a2, test_a2, map_a2, _ = apply_logit_adj(oof_a, test_a, "A) Full + logit")
oof_b2, test_b2, map_b2, _ = apply_logit_adj(oof_b, test_b, "B) No temporal + logit")
oof_c2, test_c2, map_c2, _ = apply_logit_adj(oof_c, test_c, "C) Trajectory + logit")
oof_d2, test_d2, map_d2, _ = apply_logit_adj(oof_d, test_d, "D) No temp + wc + logit")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  A) Full (105 feats):                {map_a:.4f} -> +logit: {map_a2:.4f}", flush=True)
print(f"  B) No temporal (87 feats):          {map_b:.4f} -> +logit: {map_b2:.4f}", flush=True)
print(f"  C) Trajectory only (86 feats):      {map_c:.4f} -> +logit: {map_c2:.4f}", flush=True)
print(f"  D) No temporal + weakclass:         {map_d:.4f} -> +logit: {map_d2:.4f}", flush=True)

# Check test predictions
print(f"\nTest prediction distribution (argmax):", flush=True)
for label, t in [("A-Full", test_a2), ("B-NoTmp", test_b2), ("C-Traj", test_c2), ("D-NoTmp+WC", test_d2)]:
    preds = t.argmax(axis=1)
    dist = np.bincount(preds, minlength=N_CLASSES)
    print(f"  {label}: {dict(zip(CLASSES, dist))}", flush=True)

# Save best
configs_adj = [("A", map_a2, test_a2), ("B", map_b2, test_b2), ("C", map_c2, test_c2), ("D", map_d2, test_d2)]
configs_adj.sort(key=lambda x: -x[1])
# Save the config with LEAST temporal overfitting (C or B) rather than highest CV
# Because CV will favor temporal features but LB won't
print(f"\nSaving B (no temporal) and C (trajectory only) for LB comparison:", flush=True)
save_submission(test_b2, "e25_no_temporal", cv_map=map_b2)
save_submission(test_c2, "e25_trajectory_only", cv_map=map_c2)

# Also save D
np.save(ROOT / "oof_e25b.npy", oof_b2)
np.save(ROOT / "test_e25b.npy", test_b2)
np.save(ROOT / "oof_e25c.npy", oof_c2)
np.save(ROOT / "test_e25c.npy", test_c2)

print("\nDone!", flush=True)
