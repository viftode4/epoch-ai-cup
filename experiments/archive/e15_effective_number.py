"""E15: Effective Number of Samples Class Weights (T09)

Replace is_unbalance / auto_class_weights with Cui et al. 2019 weights:
  weight_c = (1 - beta) / (1 - beta^n_c)

Sweep beta to find optimal reweighting. Retrain LGB+XGB+CB on GPU.

Reference: Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples"
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
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

# GPU check
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── Data ─────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print(f"Extracting features: {FEATURE_SETS}", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)
print(f"  Feature count: {train_feats.shape[1]}", flush=True)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)

counts = np.bincount(y, minlength=N_CLASSES)


def effective_number_weights(counts, beta):
    """Cui et al. 2019: weight = (1-beta) / (1 - beta^n)."""
    effective_n = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective_n + 1e-10)
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


# ── Show weight profiles for different betas ──────────────────────
print(f"\n{'='*60}", flush=True)
print("Effective Number weight profiles", flush=True)
print(f"{'='*60}", flush=True)

for beta in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
    w = effective_number_weights(counts, beta)
    print(f"\nbeta={beta}:", flush=True)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:15s} (n={counts[i]:4d}): weight={w[i]:.3f}", flush=True)


# ── Sweep beta with full 5-fold CV ──────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Beta sweep with 5-fold CV (LGB+XGB+CB, GPU)", flush=True)
print(f"{'='*60}", flush=True)

betas_to_try = [0.9, 0.99, 0.999, 0.9999, 0.99999]

# Also try original is_unbalance (inverse frequency) as baseline
inv_freq_weights = len(y) / (N_CLASSES * counts)
inv_freq_weights = inv_freq_weights / inv_freq_weights.mean()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Model params (GPU)
lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1,
    "device": "gpu",
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


def train_ensemble(sample_weights, label, use_cb_balanced=False):
    """Train LGB+XGB+CB ensemble with given sample weights. Returns OOF mAP."""
    oof_lgb = np.zeros((len(X), N_CLASSES))
    oof_xgb = np.zeros((len(X), N_CLASSES))
    oof_cb = np.zeros((len(X), N_CLASSES))
    test_lgb = np.zeros((len(X_test), N_CLASSES))
    test_xgb = np.zeros((len(X_test), N_CLASSES))
    test_cb = np.zeros((len(X_test), N_CLASSES))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = sample_weights[tr_idx]

        # LightGBM (no is_unbalance, use manual weights)
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
        mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = mdl.predict(X_va)
        test_lgb += mdl.predict(X_test) / N_FOLDS

        # XGBoost
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
        dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
        mdl = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, "val")], early_stopping_rounds=80, verbose_eval=0)
        oof_xgb[va_idx] = mdl.predict(dval_xgb)
        test_xgb += mdl.predict(xgb.DMatrix(X_test, feature_names=feature_names)) / N_FOLDS

        # CatBoost
        cb_kwargs = dict(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=42, verbose=0, early_stopping_rounds=80,
            task_type="GPU",
        )
        if use_cb_balanced:
            cb_kwargs["auto_class_weights"] = "Balanced"
        cb = CatBoostClassifier(**cb_kwargs)
        # CatBoost uses sample_weight in fit
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0,
               sample_weight=w_tr if not use_cb_balanced else None)
        oof_cb[va_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / N_FOLDS

    # Optimize ensemble weights
    best_map = 0
    best_w = None
    for w1 in np.arange(0.10, 0.50, 0.05):
        for w2 in np.arange(0.10, 0.50, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.10:
                continue
            oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
            m, _ = compute_map(y, oof_ens)
            if m > best_map:
                best_map = m
                best_w = (w1, w2, w3)

    oof_final = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
    test_final = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb
    final_map, final_per = compute_map(y, oof_final)

    print(f"  {label}: mAP={final_map:.4f} (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CB={best_w[2]:.2f})",
          flush=True)

    return oof_final, test_final, final_map, final_per


# ── Run E10 baseline (is_unbalance) ──────────────────────────────
print("\nBaseline (inv-freq weights, E10 repro):", flush=True)
inv_sample_w = np.array([inv_freq_weights[yi] for yi in y])
oof_base, test_base, map_base, per_base = train_ensemble(inv_sample_w, "inv_freq", use_cb_balanced=True)

# ── Run effective number betas ───────────────────────────────────
results = []
for beta in betas_to_try:
    class_w = effective_number_weights(counts, beta)
    sample_w = np.array([class_w[yi] for yi in y])
    oof, test, m, per = train_ensemble(sample_w, f"beta={beta}")
    results.append((beta, m, per, oof, test))

# ── Comparison ───────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("COMPARISON (E10 baseline = 0.7322)", flush=True)
print(f"{'='*60}", flush=True)
print(f"  inv_freq (E10 repro): {map_base:.4f} ({map_base - 0.7322:+.4f} vs E10)", flush=True)
for beta, m, per, _, _ in results:
    print(f"  beta={beta:8s}: {m:.4f} ({m - 0.7322:+.4f} vs E10)", flush=True)

# Pick best
all_results = [("inv_freq", map_base, per_base, oof_base, test_base)] + \
              [(f"beta={b}", m, p, o, t) for b, m, p, o, t in results]
all_results.sort(key=lambda x: -x[1])
best_name, best_map, best_per, best_oof, best_test = all_results[0]

print(f"\nBest: {best_name} ({best_map:.4f})", flush=True)
print_results(best_map, best_per, f"E15 Best ({best_name})")

# Save
np.save(ROOT / "oof_e15.npy", best_oof)
np.save(ROOT / "test_e15.npy", best_test)
print(f"\nSaved oof_e15.npy and test_e15.npy", flush=True)

# ── Quick stacking test ──────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Stacking: replace E10 with E15 in 4-model stack", flush=True)
print(f"{'='*60}", flush=True)

oof_e08 = np.load(ROOT / "oof_e08.npy")
oof_e06 = np.load(ROOT / "oof_e06.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")

for w0 in np.arange(0.60, 0.85, 0.05):
    for w1 in np.arange(0.05, 0.20, 0.05):
        for w2 in np.arange(0.05, 0.20, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.05:
                continue
            oof_stack = w0 * best_oof + w1 * oof_e08 + w2 * oof_e06 + w3 * oof_e09
            m, _ = compute_map(y, oof_stack)
            if m > 0.7396:
                print(f"  w_tree={w0:.2f} w_rocket={w1:.2f} w_cnn={w2:.2f} w_svm={w3:.2f}: mAP={m:.4f} ({m - 0.7396:+.4f} vs E11)",
                      flush=True)

save_submission(best_test, f"e15_effective_number_{best_name}", cv_map=best_map)
print("\nDone!", flush=True)
