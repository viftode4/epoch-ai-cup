"""E26: Simplicity vs Complexity Diagnostic

Test whether the 0.18 CV-LB gap (0.70 vs 0.52) comes from model overfitting
or from fundamental data issues. Compare:
  A) Logistic regression on all features
  B) Logistic regression on top 20 features
  C) Single shallow tree (max_depth=3)
  D) Heavily regularized GBM (few leaves, high reg)
  E) Our current best (E25D config)

If simpler models have smaller CV-LB gap, overfitting is the issue.
If all models get ~0.52 LB, the data itself is the bottleneck.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
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

TEMPORAL_OVERFIT = [
    "hour", "month", "dayofweek", "time_of_day",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "timestamp_duration",
    "is_afternoon", "is_october", "oct_afternoon", "month_x_hour",
    "is_april", "is_early_morning", "is_migration", "is_spring",
    "hour_bin_3h",
]

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Effective number weights
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

# Build features (no temporal)
print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in TEMPORAL_OVERFIT]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
print(f"  Features: {train_feats.shape[1]}", flush=True)

X_all = train_feats.values.astype(np.float32)
Xt_all = test_feats.values.astype(np.float32)
fn_all = list(train_feats.columns)
sample_weights = np.array([class_w[yi] for yi in y])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), y))

results = {}


# ── A) Logistic Regression on all features ────────────────────────
print("\n" + "=" * 60, flush=True)
print("A) Logistic Regression (all features)", flush=True)
print("=" * 60, flush=True)

oof_a = np.zeros((len(X_all), N_CLASSES))
test_a = np.zeros((len(Xt_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[tr_idx])
    X_va = scaler.transform(X_all[va_idx])
    X_te = scaler.transform(Xt_all)

    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                            multi_class="multinomial", class_weight="balanced")
    lr.fit(X_tr, y[tr_idx])
    oof_a[va_idx] = lr.predict_proba(X_va)
    test_a += lr.predict_proba(X_te) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_a, per_a = compute_map(y, oof_a)
print_results(map_a, per_a, "A) LogReg all features")
results["A_logreg_all"] = map_a


# ── B) Logistic Regression on top 20 features ─────────────────────
print("\n" + "=" * 60, flush=True)
print("B) Logistic Regression (top 20 features)", flush=True)
print("=" * 60, flush=True)

# Get feature importance from a quick LGB
dtrain = lgb.Dataset(X_all, label=y, weight=sample_weights, feature_name=fn_all)
quick_lgb = lgb.train(
    {"objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
     "learning_rate": 0.1, "num_leaves": 31, "verbose": -1, "seed": 42,
     "n_jobs": -1, "device": "gpu"},
    dtrain, num_boost_round=100,
)
importances = quick_lgb.feature_importance(importance_type="gain")
top_idx = np.argsort(importances)[-20:]
top_names = [fn_all[i] for i in top_idx]
print(f"  Top 20: {top_names}", flush=True)

X_top = X_all[:, top_idx]
Xt_top = Xt_all[:, top_idx]

oof_b = np.zeros((len(X_top), N_CLASSES))
test_b = np.zeros((len(Xt_top), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_top[tr_idx])
    X_va = scaler.transform(X_top[va_idx])
    X_te = scaler.transform(Xt_top)

    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                            multi_class="multinomial", class_weight="balanced")
    lr.fit(X_tr, y[tr_idx])
    oof_b[va_idx] = lr.predict_proba(X_va)
    test_b += lr.predict_proba(X_te) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_b, per_b = compute_map(y, oof_b)
print_results(map_b, per_b, "B) LogReg top 20")
results["B_logreg_top20"] = map_b


# ── C) Single shallow tree ────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("C) Single shallow CatBoost (depth=3, 100 trees)", flush=True)
print("=" * 60, flush=True)

oof_c = np.zeros((len(X_all), N_CLASSES))
test_c = np.zeros((len(Xt_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    cb = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=3, l2_leaf_reg=10,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_all[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx], verbose=0)
    oof_c[va_idx] = cb.predict_proba(X_all[va_idx])
    test_c += cb.predict_proba(Xt_all) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_c, per_c = compute_map(y, oof_c)
print_results(map_c, per_c, "C) Shallow CatBoost")
results["C_shallow_cb"] = map_c


# ── D) Heavily regularized GBM ────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("D) Heavily regularized CatBoost", flush=True)
print("=" * 60, flush=True)

oof_d = np.zeros((len(X_all), N_CLASSES))
test_d = np.zeros((len(Xt_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=4, l2_leaf_reg=30,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=100,
        task_type="GPU",
        bootstrap_type="Bernoulli", subsample=0.6,
        min_data_in_leaf=20,
    )
    cb.fit(X_all[tr_idx], y[tr_idx], eval_set=(X_all[va_idx], y[va_idx]),
           sample_weight=sample_weights[tr_idx], verbose=0)
    oof_d[va_idx] = cb.predict_proba(X_all[va_idx])
    test_d += cb.predict_proba(Xt_all) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_d, per_d = compute_map(y, oof_d)
print_results(map_d, per_d, "D) Heavy regularized CB")
results["D_heavy_reg"] = map_d


# ── E) Our current best for comparison ────────────────────────────
print("\n" + "=" * 60, flush=True)
print("E) Current best (LGB+XGB+CB ensemble)", flush=True)
print("=" * 60, flush=True)

lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

oof_e = np.zeros((len(X_all), N_CLASSES))
test_e = np.zeros((len(Xt_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    dtrain = lgb.Dataset(X_all[tr_idx], label=y[tr_idx],
                         weight=sample_weights[tr_idx], feature_name=fn_all)
    dval = lgb.Dataset(X_all[va_idx], label=y[va_idx],
                       feature_name=fn_all, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_e[va_idx] = mdl.predict(X_all[va_idx])
    test_e += mdl.predict(Xt_all) / N_FOLDS

    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X_all[tr_idx], y[tr_idx], eval_set=(X_all[va_idx], y[va_idx]),
           sample_weight=sample_weights[tr_idx], verbose=0)
    oof_e[va_idx] = 0.15 * oof_e[va_idx] + 0.85 * cb.predict_proba(X_all[va_idx])
    test_e = 0.15 * test_e + 0.85 * cb.predict_proba(Xt_all) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_e, per_e = compute_map(y, oof_e)
print_results(map_e, per_e, "E) LGB+CB ensemble")
results["E_ensemble"] = map_e


# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("SUMMARY -- Submit all to Kaggle to find smallest CV-LB gap", flush=True)
print("=" * 60, flush=True)
for name, m in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:25s}: CV = {m:.4f}", flush=True)

# Save submissions for ALL configs
for label, test_pred, cv in [
    ("e26a_logreg_all", test_a, map_a),
    ("e26b_logreg_top20", test_b, map_b),
    ("e26c_shallow_cb", test_c, map_c),
    ("e26d_heavy_reg", test_d, map_d),
    ("e26e_ensemble", test_e, map_e),
]:
    save_submission(test_pred, label, cv_map=cv)

print("\nDone! Submit each to find which generalizes best.", flush=True)
