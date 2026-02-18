"""
E07: Biological & physics-grounded features over honest GroupKFold CV.

New features over E06:
1. Biological time (sun elevation, solar noon offset) — generalises across months
2. Trajectory shape (circling index, path straightness, direction autocorrelation)
3. Flight mode (flap/glide fractions, bout counts, RCS variance texture)

All evaluated with StratifiedGroupKFold on primary_observation_id.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from src.data import load_train, load_test, CLASSES
from src.features import (
    extract_core_features, extract_rcs_fft_features,
    extract_wingbeat_features, extract_shape_features,
    extract_flight_mode_features, add_tabular_features,
    add_biological_time_features,
)
from src.metrics import compute_map, print_results
from src.submission import save_submission

# ============================================================
# 1. LOAD DATA
# ============================================================
train = load_train()
test  = load_test()
print(f"Train: {len(train)}, Test: {len(test)}")

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def extract_all(df):
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        feats = {}
        feats.update(extract_core_features(r.trajectory, r.trajectory_time))
        feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
        feats.update(extract_wingbeat_features(r.trajectory, r.trajectory_time))
        feats.update(extract_shape_features(r.trajectory, r.trajectory_time))
        feats.update(extract_flight_mode_features(r.trajectory, r.trajectory_time))
        rows.append(feats)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df)}")
    feat_df = pd.DataFrame(rows)
    feat_df = add_tabular_features(feat_df, df)
    feat_df = add_biological_time_features(feat_df, df)
    return feat_df

print("Extracting train features...")
train_feats = extract_all(train)
print("Extracting test features...")
test_feats  = extract_all(test)

# ============================================================
# 3. REMOVE HARMFUL FEATURES
# ============================================================
# Keep bio_time (sun_elevation etc.) but drop raw temporal indicators
DROP = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_pigeon_window",
    # location noise (single radar site)
    "lon_mean", "lat_mean", "lon_std", "lat_std", "spatial_spread",
]
train_feats = train_feats.drop(columns=[c for c in DROP if c in train_feats.columns])
test_feats  = test_feats.drop(columns=[c for c in DROP if c in test_feats.columns])

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats  = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nFeature count: {train_feats.shape[1]}")
new_cols = [c for c in train_feats.columns
            if c.startswith(("sun_", "solar_", "is_day", "is_civil",
                             "turn_rate", "curvature", "dir_auto", "circling",
                             "path_straight", "bbox_eff", "max_sus",
                             "flap_", "glide_", "n_flap", "mean_flap",
                             "rcs_var_", "flap_glide"))]
print(f"New bio/shape/flight features ({len(new_cols)}): {new_cols}")

# ============================================================
# 4. TARGET & GROUPS
# ============================================================
le = LabelEncoder()
le.fit(CLASSES)
y      = le.transform(train["bird_group"])
groups = train["primary_observation_id"].values
n_cls  = len(CLASSES)

X       = train_feats.values.astype(np.float32)
X_test  = test_feats.values.astype(np.float32)
feat_names = list(train_feats.columns)

class_counts  = np.bincount(y, minlength=n_cls)
class_weights = len(y) / (n_cls * class_counts)
sample_weights = np.array([class_weights[yi] for yi in y])

# ============================================================
# 5. TRAIN — StratifiedGroupKFold
# ============================================================
N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), n_cls))
oof_xgb = np.zeros((len(X), n_cls))
oof_cb  = np.zeros((len(X), n_cls))
tst_lgb = np.zeros((len(X_test), n_cls))
tst_xgb = np.zeros((len(X_test), n_cls))
tst_cb  = np.zeros((len(X_test), n_cls))

lgb_params = {
    "objective": "multiclass", "num_class": n_cls, "metric": "multi_logloss",
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 15, "subsample": 0.7, "colsample_bytree": 0.6,
    "reg_alpha": 1.0, "reg_lambda": 3.0,
    "verbose": -1, "seed": 42, "n_jobs": -1, "is_unbalance": True,
}
xgb_params = {
    "objective": "multi:softprob", "num_class": n_cls, "eval_metric": "mlogloss",
    "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
    "subsample": 0.7, "colsample_bytree": 0.6,
    "reg_alpha": 1.0, "reg_lambda": 3.0,
    "seed": 42, "nthread": -1, "verbosity": 0,
}

print("\n" + "="*60)
print("TRAINING — StratifiedGroupKFold (5 folds)")
print("="*60)

last_lgb_model = None
for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    print(f"\nFold {fold}: train={len(tr_idx)}, val={len(va_idx)}")

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_names)
    dval   = lgb.Dataset(X_va, label=y_va, feature_name=feat_names, reference=dtrain)
    lgb_m  = lgb.train(lgb_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = lgb_m.predict(X_va)
    tst_lgb += lgb_m.predict(X_test) / N_FOLDS
    last_lgb_model = lgb_m
    m, _ = compute_map(y_va, oof_lgb[va_idx])
    print(f"  LGB  mAP={m:.4f}  (trees={lgb_m.best_iteration})")

    # XGBoost
    dtrain_x = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feat_names)
    dval_x   = xgb.DMatrix(X_va, label=y_va, feature_names=feat_names)
    xgb_m    = xgb.train(xgb_params, dtrain_x, num_boost_round=3000,
                          evals=[(dval_x, "val")], early_stopping_rounds=100,
                          verbose_eval=0)
    oof_xgb[va_idx] = xgb_m.predict(dval_x)
    tst_xgb += xgb_m.predict(xgb.DMatrix(X_test, feature_names=feat_names)) / N_FOLDS
    m, _ = compute_map(y_va, oof_xgb[va_idx])
    print(f"  XGB  mAP={m:.4f}  (trees={xgb_m.best_iteration})")

    # CatBoost
    cb_m = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=5, l2_leaf_reg=5,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=100,
        auto_class_weights="Balanced",
    )
    cb_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb[va_idx] = cb_m.predict_proba(X_va)
    tst_cb += cb_m.predict_proba(X_test) / N_FOLDS
    m, _ = compute_map(y_va, oof_cb[va_idx])
    print(f"  CB   mAP={m:.4f}")

# ============================================================
# 6. RESULTS
# ============================================================
lgb_map, lgb_per = compute_map(y, oof_lgb)
xgb_map, xgb_per = compute_map(y, oof_xgb)
cb_map,  cb_per  = compute_map(y, oof_cb)
print_results(lgb_map, lgb_per, "LightGBM")
print_results(xgb_map, xgb_per, "XGBoost")
print_results(cb_map,  cb_per,  "CatBoost")

# Ensemble optimisation
def neg_map(w):
    w = np.abs(w); w /= w.sum()
    return -compute_map(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cb)[0]

res   = minimize(neg_map, [0.33, 0.33, 0.34], method="Nelder-Mead",
                 options={"maxiter": 1000})
bw    = np.abs(res.x); bw /= bw.sum()
print(f"\nEnsemble weights: LGB={bw[0]:.3f} XGB={bw[1]:.3f} CB={bw[2]:.3f}")

oof_final = bw[0]*oof_lgb + bw[1]*oof_xgb + bw[2]*oof_cb
tst_final = bw[0]*tst_lgb + bw[1]*tst_xgb + bw[2]*tst_cb
final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E07 Ensemble")

# Compare with E06
e06_per = {
    "Clutter": 0.578, "Cormorants": 0.875, "Pigeons": 0.169,
    "Ducks": 0.445, "Geese": 0.508, "Gulls": 0.893,
    "Birds of Prey": 0.672, "Waders": 0.642, "Songbirds": 0.125,
}
print(f"\n{'Class':15s} {'E06':>8s} {'E07':>8s} {'Δ':>8s}")
for cls in CLASSES:
    d = final_per[cls] - e06_per[cls]
    print(f"{cls:15s} {e06_per[cls]:8.4f} {final_per[cls]:8.4f} {d:+8.4f}")
print(f"{'OVERALL':15s} {'0.5452':>8s} {final_map:8.4f} {final_map-0.5452:+8.4f}")

# Feature importance
imp = pd.DataFrame({
    "feature": feat_names,
    "importance": last_lgb_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("\nTop 25 features (LGB gain):")
for _, row in imp.head(25).iterrows():
    tag = " *NEW*" if row["feature"] in new_cols else ""
    print(f"  {row['feature']:32s}: {row['importance']:.1f}{tag}")

# ============================================================
# 7. SAVE SUBMISSION
# ============================================================
save_submission(tst_final, "e07_bio_features", cv_map=final_map)
print("\nDone.")
