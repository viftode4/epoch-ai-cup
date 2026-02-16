"""E40: Balanced Augmentation + Heavier Regularization

Problem: macro mAP weights all 9 classes equally, but Cormorants (40 samples)
drags down the average. Effective number weights reweight loss, but can't
create diversity in feature space. Need more diverse minority examples.

Approach:
1. Mixup augmentation for minority classes (target 200+ per class)
2. Undersample Gulls (1503 -> 500)
3. E38 external features (weather+solar+GBIF, best LB = 0.53)
4. Heavier regularization (LOMO-tuned)
5. Multiple regularization configs tested on LOMO

Mixup: x_new = lambda*x1 + (1-lambda)*x2, y stays same (same class).
Creates diverse within-class samples without leaving the data manifold.
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
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999


def mixup_augment(X, y, weights, months, target_per_class=200, seed=42):
    """Mixup augmentation for minority classes + undersampling Gulls.

    For each class with < target_per_class samples:
      - Create new samples by mixing random pairs within the class
      - x_new = lambda * x_i + (1-lambda) * x_j, lambda ~ Beta(0.4, 0.4)

    For Gulls (class with most samples): undersample to 500.
    """
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=N_CLASSES)
    gull_idx = list(CLASSES).index("Gulls")

    X_aug, y_aug, w_aug, m_aug = [], [], [], []

    for cls in range(N_CLASSES):
        cls_mask = y == cls
        X_cls = X[cls_mask]
        w_cls = weights[cls_mask]
        m_cls = months[cls_mask]
        n = len(X_cls)

        if cls == gull_idx and n > 500:
            # Undersample Gulls
            idx = rng.choice(n, 500, replace=False)
            X_aug.append(X_cls[idx])
            y_aug.append(np.full(500, cls))
            w_aug.append(w_cls[idx])
            m_aug.append(m_cls[idx])
            print(f"    {CLASSES[cls]}: {n} -> 500 (undersampled)", flush=True)
        elif n < target_per_class:
            # Keep all originals
            X_aug.append(X_cls)
            y_aug.append(np.full(n, cls))
            w_aug.append(w_cls)
            m_aug.append(m_cls)

            # Mixup to reach target
            n_new = target_per_class - n
            idx1 = rng.randint(0, n, n_new)
            idx2 = rng.randint(0, n, n_new)
            # Ensure different samples mixed (resample if same)
            same = idx1 == idx2
            idx2[same] = (idx2[same] + 1) % n

            lam = rng.beta(0.4, 0.4, n_new).reshape(-1, 1)
            X_new = lam * X_cls[idx1] + (1 - lam) * X_cls[idx2]
            w_new = lam.flatten() * w_cls[idx1] + (1 - lam.flatten()) * w_cls[idx2]
            # Assign month from first parent
            m_new = m_cls[idx1]

            X_aug.append(X_new)
            y_aug.append(np.full(n_new, cls))
            w_aug.append(w_new)
            m_aug.append(m_new)
            print(f"    {CLASSES[cls]}: {n} -> {n + n_new} (+{n_new} mixup)", flush=True)
        else:
            # Keep as-is (Songbirds etc.)
            X_aug.append(X_cls)
            y_aug.append(np.full(n, cls))
            w_aug.append(w_cls)
            m_aug.append(m_cls)
            print(f"    {CLASSES[cls]}: {n} (kept)", flush=True)

    return (np.vstack(X_aug).astype(np.float32),
            np.concatenate(y_aug),
            np.concatenate(w_aug),
            np.concatenate(m_aug))


def train_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label, params_key="base"):
    """Train LGB+XGB+CB with configurable regularization."""

    if params_key == "base":
        lgb_p = {
            "objective": "multiclass", "num_class": N_CLASSES,
            "metric": "multi_logloss", "learning_rate": 0.05,
            "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
            "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
        }
        xgb_p = {
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "eval_metric": "mlogloss", "learning_rate": 0.05,
            "max_depth": 6, "min_child_weight": 3,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
            "seed": 42, "nthread": -1, "verbosity": 0,
            "device": "cuda", "tree_method": "hist",
        }
        cb_depth = 6
        cb_l2 = 3
    elif params_key == "heavy_reg":
        lgb_p = {
            "objective": "multiclass", "num_class": N_CLASSES,
            "metric": "multi_logloss", "learning_rate": 0.03,
            "num_leaves": 23, "max_depth": 5, "min_child_samples": 15,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "min_gain_to_split": 0.1,
            "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
        }
        xgb_p = {
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "eval_metric": "mlogloss", "learning_rate": 0.03,
            "max_depth": 4, "min_child_weight": 8,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "gamma": 1.0,
            "seed": 42, "nthread": -1, "verbosity": 0,
            "device": "cuda", "tree_method": "hist",
        }
        cb_depth = 4
        cb_l2 = 10
    elif params_key == "mid_reg":
        lgb_p = {
            "objective": "multiclass", "num_class": N_CLASSES,
            "metric": "multi_logloss", "learning_rate": 0.04,
            "num_leaves": 31, "max_depth": 6, "min_child_samples": 12,
            "subsample": 0.75, "colsample_bytree": 0.6,
            "reg_alpha": 0.5, "reg_lambda": 3.0,
            "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
        }
        xgb_p = {
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "eval_metric": "mlogloss", "learning_rate": 0.04,
            "max_depth": 5, "min_child_weight": 5,
            "subsample": 0.75, "colsample_bytree": 0.6,
            "reg_alpha": 0.5, "reg_lambda": 3.0,
            "gamma": 0.3,
            "seed": 42, "nthread": -1, "verbosity": 0,
            "device": "cuda", "tree_method": "hist",
        }
        cb_depth = 5
        cb_l2 = 5

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(lgb_p, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb = m_lgb.predict(X_va)
    test_lgb = m_lgb.predict(X_test) if X_test is not None else None

    m_xgb = xgb.train(xgb_p, xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn),
                       2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                       early_stopping_rounds=80, verbose_eval=0)
    oof_xgb = m_xgb.predict(xgb.DMatrix(X_va, feature_names=fn))
    test_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    cb = CatBoostClassifier(iterations=2000, learning_rate=lgb_p["learning_rate"],
                            depth=cb_depth, l2_leaf_reg=cb_l2,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof = 0.33 * oof_lgb + 0.33 * oof_xgb + 0.34 * oof_cb
    test_ens = (0.33 * test_lgb + 0.33 * test_xgb + 0.34 * test_cb) if X_test is not None else None
    m, _ = compute_map(y_va, oof)
    print(f"    {label}: mAP={m:.4f}", flush=True)
    return oof, test_ens


# ======================================================================
# Load data + build features (E38 config: base + weather + solar + GBIF)
# ======================================================================
print("=" * 60, flush=True)
print("E40 BALANCED AUGMENTATION + HEAVIER REGULARIZATION", flush=True)
print("=" * 60, flush=True)

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
sample_weights = np.array([class_w[yi] for yi in y])

# Timestamps
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# Base features (temporal-free)
print("\nBuilding base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
base_cols = list(train_feats.columns)
print(f"  Base features: {len(base_cols)}", flush=True)

# Weather features
print("Loading weather features...", flush=True)
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

# Solar features
print("Loading solar features...", flush=True)
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

# GBIF features
print("Loading GBIF features...", flush=True)
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_counts = gbif[cls].values
            class_mean = class_counts.mean()
            si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    gbif_si[m] = si

for i, cls in enumerate(CLASSES):
    col = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col] = [gbif_si[m][i] for m in train_months]
    test_feats[col] = [gbif_si[m][i] for m in test_months]

gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

# Clean
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

all_cols = list(train_feats.columns)
X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(all_cols)
print(f"  Total features: {len(fn)}", flush=True)

# Class distribution
print(f"\n  Class distribution:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {counts[i]:>5d}", flush=True)

# ======================================================================
# LOMO evaluation: compare configs
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("LOMO EVALUATION (Leave-One-Month-Out)", flush=True)
print("=" * 60, flush=True)

configs = {
    "A: E38 baseline (no aug, base reg)": {"augment": False, "params": "base"},
    "B: Mixup aug + base reg": {"augment": True, "params": "base"},
    "C: No aug + mid reg": {"augment": False, "params": "mid_reg"},
    "D: Mixup aug + mid reg": {"augment": True, "params": "mid_reg"},
    "E: No aug + heavy reg": {"augment": False, "params": "heavy_reg"},
    "F: Mixup aug + heavy reg": {"augment": True, "params": "heavy_reg"},
}

lomo_results = {}
for cfg_name, cfg in configs.items():
    print(f"\n--- {cfg_name} ---", flush=True)
    oof_lomo = np.zeros((len(y), N_CLASSES))

    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        w_tr = sample_weights[tr_idx]
        m_tr = train_months[tr_idx]

        if cfg["augment"]:
            X_tr, y_tr, w_tr, m_tr = mixup_augment(X_tr, y_tr, w_tr, m_tr,
                                                      target_per_class=200, seed=42 + m)

        oof_fold, _ = train_fold(
            X_tr, y_tr, X[va_idx], y[va_idx], w_tr,
            None, fn, f"M{m}", params_key=cfg["params"])
        oof_lomo[va_idx] = oof_fold

    lomo_map, lomo_per = compute_map(y, oof_lomo)
    lomo_results[cfg_name] = {"map": lomo_map, "per": lomo_per, "oof": oof_lomo}
    print(f"  => LOMO mAP: {lomo_map:.4f}", flush=True)

# LOMO summary
print("\n" + "=" * 60, flush=True)
print("LOMO SUMMARY", flush=True)
print("=" * 60, flush=True)

ref_map = lomo_results["A: E38 baseline (no aug, base reg)"]["map"]
print(f"\n  {'Config':<40s} {'LOMO':>7s} {'Delta':>7s}", flush=True)
print(f"  {'-'*54}", flush=True)
for name, res in lomo_results.items():
    delta = res["map"] - ref_map
    d_str = f"{delta:+.4f}" if "A:" not in name else "---"
    print(f"  {name:<40s} {res['map']:>7.4f} {d_str:>7s}", flush=True)

# Per-class for top 2
print(f"\n  Per-class LOMO comparison:", flush=True)
sorted_configs = sorted(lomo_results.items(), key=lambda x: x[1]["map"], reverse=True)
top_names = [sorted_configs[0][0], "A: E38 baseline (no aug, base reg)"]
header = f"  {'Class':<15s}"
for name in top_names:
    short = name.split(":")[0]
    header += f" {short:>7s}"
header += "   Delta"
print(header, flush=True)
for cls in CLASSES:
    line = f"  {cls:<15s}"
    aps = []
    for name in top_names:
        ap = lomo_results[name]["per"].get(cls, 0)
        line += f" {ap:>7.4f}"
        aps.append(ap)
    line += f" {aps[0]-aps[1]:>+7.4f}"
    print(line, flush=True)

# ======================================================================
# SKF + test predictions for best LOMO config
# ======================================================================
best_cfg_name = sorted_configs[0][0]
best_cfg = configs[best_cfg_name]
print(f"\n\nBest LOMO config: {best_cfg_name}", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"SKF + TEST PREDICTIONS for {best_cfg_name}", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.zeros((len(y), N_CLASSES))
test_pred = np.zeros((len(X_test), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    w_tr = sample_weights[tr_idx]
    m_tr = train_months[tr_idx]

    if best_cfg["augment"]:
        X_tr, y_tr, w_tr, m_tr = mixup_augment(X_tr, y_tr, w_tr, m_tr,
                                                  target_per_class=200, seed=42 + fold_idx)

    oof_fold, test_fold = train_fold(
        X_tr, y_tr, X[va_idx], y[va_idx], w_tr,
        X_test, fn, f"SKF Fold {fold_idx}", params_key=best_cfg["params"])
    oof_skf[va_idx] = oof_fold
    test_pred += test_fold / 5

skf_map, skf_per = compute_map(y, oof_skf)
best_lomo_map = lomo_results[best_cfg_name]["map"]

print(f"\n  SKF CV mAP:  {skf_map:.4f}", flush=True)
print(f"  LOMO mAP:    {best_lomo_map:.4f}", flush=True)
print(f"  E38 LB:      0.53", flush=True)

# Test distribution
print(f"\n  Test class distribution (argmax):", flush=True)
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {dist[i]}", flush=True)

# Also generate test predictions with E38 baseline config for comparison
print("\n  Also generating E38-baseline test predictions...", flush=True)
oof_e38_skf = np.zeros((len(y), N_CLASSES))
test_e38 = np.zeros((len(X_test), N_CLASSES))
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    oof_fold, test_fold = train_fold(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test, fn, f"E38ref F{fold_idx}", params_key="base")
    oof_e38_skf[va_idx] = oof_fold
    test_e38 += test_fold / 5

e38_skf_map, _ = compute_map(y, oof_e38_skf)
print(f"  E38-baseline SKF: {e38_skf_map:.4f}", flush=True)

# ======================================================================
# Save
# ======================================================================
np.save(ROOT / "oof_e40.npy", oof_skf)
np.save(ROOT / "test_e40.npy", test_pred)
save_submission(test_pred, "e40_balanced_aug", cv_map=skf_map)

# Also save E38-baseline retrained (in case current submission.csv is stale)
np.save(ROOT / "test_e38_v2.npy", test_e38)
save_submission(test_e38, "e38_v2_baseline", cv_map=e38_skf_map)

print("\nDone!", flush=True)
