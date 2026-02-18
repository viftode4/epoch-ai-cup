"""
E06: Fix CV leakage + remove temporal overfitting.

Key changes vs E02:
1. StratifiedGroupKFold with primary_observation_id (fixes 43% validation leakage)
2. Remove all temporal indicator features that overfit to train's time distribution
3. Remove location features (single radar site = noise)
4. Stronger regularization (fewer leaves, more L1/L2)
5. Adversarial validation to identify train/test distinguishing features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from src.data import load_train, load_test, CLASSES, ROOT
from src.features import (
    extract_core_features, extract_rcs_fft_features, add_tabular_features,
    extract_wingbeat_features,
)
from src.metrics import compute_map, print_results
from src.submission import save_submission

# ============================================================
# 1. LOAD DATA
# ============================================================
train = load_train()
test = load_test()
print(f"Train: {len(train)}, Test: {len(test)}")
print(f"Unique primary_observation_ids: {train['primary_observation_id'].nunique()}")

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
print("\nExtracting train features...")
train_rows = []
for _, r in train.iterrows():
    feats = {}
    feats.update(extract_core_features(r.trajectory, r.trajectory_time))
    feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
    feats.update(extract_wingbeat_features(r.trajectory, r.trajectory_time))
    train_rows.append(feats)
train_feats = pd.DataFrame(train_rows)

print("Extracting test features...")
test_rows = []
for _, r in test.iterrows():
    feats = {}
    feats.update(extract_core_features(r.trajectory, r.trajectory_time))
    feats.update(extract_rcs_fft_features(r.trajectory, r.trajectory_time))
    feats.update(extract_wingbeat_features(r.trajectory, r.trajectory_time))
    test_rows.append(feats)
test_feats = pd.DataFrame(test_rows)

# Add tabular features (timestamps, airspeed, altitude, radar size)
train_feats = add_tabular_features(train_feats, train)
test_feats = add_tabular_features(test_feats, test)

# ============================================================
# 3. REMOVE HARMFUL FEATURES
# ============================================================
# Features that exploit temporal patterns in train that don't transfer to test:
# - Train has Oct=55%, hours 13-15=22%; test has different temporal distribution
# - 32.7% of test comes from months not in train (Feb, May, Dec)
# Location features are noise (single radar site)
DROP_FEATURES = [
    # Temporal indicators that overfit to train's time distribution
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_pigeon_window",
    # Location features (single radar site = noise)
    "lon_mean", "lat_mean", "lon_std", "lat_std", "spatial_spread",
]

existing_drops = [c for c in DROP_FEATURES if c in train_feats.columns]
print(f"\nDropping {len(existing_drops)} harmful features: {existing_drops}")
train_feats = train_feats.drop(columns=existing_drops, errors='ignore')
test_feats = test_feats.drop(columns=existing_drops, errors='ignore')

# Handle inf/nan
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Final feature count: {train_feats.shape[1]}")
print(f"Features: {list(train_feats.columns)}")

# ============================================================
# 4. ADVERSARIAL VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("ADVERSARIAL VALIDATION (train vs test distinguishability)")
print("=" * 60)

adv_X = pd.concat([train_feats, test_feats], axis=0).values.astype(np.float32)
adv_y = np.array([0] * len(train_feats) + [1] * len(test_feats))
adv_feature_names = list(train_feats.columns)

from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score

adv_oof = np.zeros(len(adv_X))
adv_skf = SKF(n_splits=3, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(adv_skf.split(adv_X, adv_y)):
    adv_dtrain = lgb.Dataset(adv_X[tr_idx], label=adv_y[tr_idx])
    adv_dval = lgb.Dataset(adv_X[va_idx], label=adv_y[va_idx])
    adv_model = lgb.train(
        {"objective": "binary", "metric": "auc", "verbose": -1,
         "num_leaves": 31, "learning_rate": 0.05, "seed": 42},
        adv_dtrain, num_boost_round=200, valid_sets=[adv_dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    adv_oof[va_idx] = adv_model.predict(adv_X[va_idx])

adv_auc = roc_auc_score(adv_y, adv_oof)
print(f"\nAdversarial AUC: {adv_auc:.4f}")
if adv_auc > 0.7:
    print("  WARNING: Significant train/test distribution shift detected!")
elif adv_auc > 0.6:
    print("  Moderate train/test distribution shift.")
else:
    print("  Train/test distributions are similar (good).")

# Feature importance for adversarial model (which features distinguish train/test?)
adv_imp = pd.DataFrame({
    "feature": adv_feature_names,
    "importance": adv_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("\nTop features distinguishing train vs test:")
for _, row in adv_imp.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.1f}")

# ============================================================
# 5. PREPARE TARGET & GROUPS
# ============================================================
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train["bird_group"])
groups = train["primary_observation_id"].values
n_classes = len(CLASSES)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

# Class weights (inverse frequency)
class_counts = np.bincount(y, minlength=n_classes)
class_weights = len(y) / (n_classes * class_counts)
sample_weights = np.array([class_weights[yi] for yi in y])
print(f"\nClass weights: {dict(zip(CLASSES, class_weights.round(2)))}")

# ============================================================
# 6. TRAIN WITH StratifiedGroupKFold
# ============================================================
N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(X), n_classes))
oof_xgb = np.zeros((len(X), n_classes))
oof_cb = np.zeros((len(X), n_classes))
test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_cb = np.zeros((len(X_test), n_classes))

# Stronger regularization than E02
lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,         # E02 used 47
    "max_depth": 6,           # E02 used 7
    "min_child_samples": 15,  # E02 used 8
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,         # E02 used 0.3
    "reg_lambda": 3.0,        # E02 used 1.5
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
    "is_unbalance": True,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 5,           # E02 used 6
    "min_child_weight": 5,    # E02 used 3
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "seed": 42,
    "nthread": -1,
    "verbosity": 0,
}

print("\n" + "=" * 60)
print("TRAINING WITH StratifiedGroupKFold (honest CV)")
print("=" * 60)

for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr = sample_weights[tr_idx]

    # Check no group leakage
    train_groups = set(groups[tr_idx])
    val_groups = set(groups[va_idx])
    assert len(train_groups & val_groups) == 0, "Group leakage detected!"

    print(f"\nFold {fold}: train={len(tr_idx)}, val={len(va_idx)}, "
          f"train_groups={len(train_groups)}, val_groups={len(val_groups)}")

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params, dtrain, num_boost_round=3000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_lgb[va_idx])
    print(f"  LGB fold {fold}: mAP = {fold_map:.4f}")

    # XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    xgb_model = xgb.train(
        xgb_params, dtrain_xgb, num_boost_round=3000,
        evals=[(dval_xgb, "val")], early_stopping_rounds=100, verbose_eval=0,
    )
    oof_xgb[va_idx] = xgb_model.predict(dval_xgb)
    test_xgb += xgb_model.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_xgb[va_idx])
    print(f"  XGB fold {fold}: mAP = {fold_map:.4f}")

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5,          # E02 used 3
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=100,
        auto_class_weights="Balanced",
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    oof_cb[va_idx] = cb_model.predict_proba(X_va)
    test_cb += cb_model.predict_proba(X_test) / N_FOLDS

    fold_map, _ = compute_map(y_va, oof_cb[va_idx])
    print(f"  CB  fold {fold}: mAP = {fold_map:.4f}")

# ============================================================
# 7. INDIVIDUAL MODEL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("INDIVIDUAL MODEL RESULTS (honest CV)")
print("=" * 60)

lgb_map, lgb_per = compute_map(y, oof_lgb)
xgb_map, xgb_per = compute_map(y, oof_xgb)
cb_map, cb_per = compute_map(y, oof_cb)

print_results(lgb_map, lgb_per, "LightGBM")
print_results(xgb_map, xgb_per, "XGBoost")
print_results(cb_map, cb_per, "CatBoost")

# ============================================================
# 8. ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE OPTIMIZATION")
print("=" * 60)

from scipy.optimize import minimize

def neg_map(weights):
    w = np.abs(weights)
    w = w / w.sum()
    oof_ens = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb
    m, _ = compute_map(y, oof_ens)
    return -m

result = minimize(neg_map, x0=[0.33, 0.33, 0.34], method="Nelder-Mead",
                  options={"maxiter": 1000})
best_w = np.abs(result.x)
best_w = best_w / best_w.sum()
print(f"Optimal weights: LGB={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CB={best_w[2]:.3f}")

oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E06 Ensemble (honest CV)")

# Compare with E02 (leaky CV)
print("\n" + "=" * 60)
print("COMPARISON: E02 (leaky) vs E06 (honest)")
print("=" * 60)
v2_aps = {
    "Clutter": 0.610, "Cormorants": 0.939, "Pigeons": 0.254,
    "Ducks": 0.666, "Geese": 0.728, "Gulls": 0.956,
    "Birds of Prey": 0.885, "Waders": 0.816, "Songbirds": 0.640,
}
print(f"  {'Class':15s} {'E02(leaky)':>12s} {'E06(honest)':>12s}")
for cls in CLASSES:
    v2 = v2_aps[cls]
    v6 = final_per[cls]
    print(f"  {cls:15s} {v2:12.4f} {v6:12.4f}")
print(f"  {'OVERALL':15s} {'0.7214':>12s} {final_map:12.4f}")
print(f"\n  Note: E06 scores are LOWER but HONEST — they predict real test performance.")

# ============================================================
# 9. FEATURE IMPORTANCE (from last LGB model)
# ============================================================
print("\n" + "=" * 60)
print("TOP 20 FEATURES (LGB gain)")
print("=" * 60)
imp = pd.DataFrame({
    "feature": feature_names,
    "importance": lgb_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
for i, row in imp.head(20).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.1f}")

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
save_submission(test_final, "e06_fix_cv", cv_map=final_map)
print("\nDone.")
