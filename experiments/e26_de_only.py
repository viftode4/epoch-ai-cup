"""E26 D+E only (A/B/C already done)."""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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

effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()

print("Building features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in TEMPORAL_OVERFIT]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

X = train_feats.values.astype(np.float32)
Xt = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
sw = np.array([class_w[yi] for yi in y])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), y))

# ── D) Heavily regularized CatBoost ──────────────────────────────
print("\n" + "=" * 60, flush=True)
print("D) Heavily regularized CatBoost", flush=True)
print("=" * 60, flush=True)

oof_d = np.zeros((len(X), N_CLASSES))
test_d = np.zeros((len(Xt), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=4, l2_leaf_reg=30,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=100,
        task_type="GPU",
        bootstrap_type="Bernoulli", subsample=0.6,
        min_data_in_leaf=20,
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
           sample_weight=sw[tr_idx], verbose=0)
    oof_d[va_idx] = cb.predict_proba(X[va_idx])
    test_d += cb.predict_proba(Xt) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

map_d, per_d = compute_map(y, oof_d)
print_results(map_d, per_d, "D) Heavy reg CB")

# ── E) Current best ensemble ─────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("E) LGB+CB ensemble (current best)", flush=True)
print("=" * 60, flush=True)

lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}

oof_lgb = np.zeros((len(X), N_CLASSES))
oof_cb = np.zeros((len(X), N_CLASSES))
test_lgb = np.zeros((len(Xt), N_CLASSES))
test_cb = np.zeros((len(Xt), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(folds):
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sw[tr_idx], feature_name=fn)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=fn, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(X[va_idx])
    test_lgb += mdl.predict(Xt) / N_FOLDS

    cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80,
        task_type="GPU",
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
           sample_weight=sw[tr_idx], verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X[va_idx])
    test_cb += cb.predict_proba(Xt) / N_FOLDS
    print(f"  Fold {fold} done", flush=True)

oof_e = 0.15 * oof_lgb + 0.85 * oof_cb
test_e = 0.15 * test_lgb + 0.85 * test_cb
map_e, per_e = compute_map(y, oof_e)
print_results(map_e, per_e, "E) LGB+CB ensemble")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("FULL SUMMARY (A/B/C from prior run)", flush=True)
print("=" * 60, flush=True)
print(f"  A) LogReg all features:   CV = 0.5030", flush=True)
print(f"  B) LogReg top 20:         CV = 0.5390", flush=True)
print(f"  C) Shallow CB (d=3,100t): CV = 0.5678", flush=True)
print(f"  D) Heavy reg CB:          CV = {map_d:.4f}", flush=True)
print(f"  E) LGB+CB ensemble:       CV = {map_e:.4f}", flush=True)

# Save all submissions
save_submission(test_d, "e26d_heavy_reg", cv_map=map_d)
save_submission(test_e, "e26e_ensemble", cv_map=map_e)

# Also save A/B/C submissions (re-using from the failed run is not possible,
# but the user can re-run e26 if needed for those)
print("\nDone! D+E saved. For A/B/C submissions, re-run e26_simplicity_test.py", flush=True)
