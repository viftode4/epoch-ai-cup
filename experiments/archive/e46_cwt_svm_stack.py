"""E46: Zaugg CWT + SVM Stacking

Literature (Zaugg 2008) shows SVM achieves AUC 0.965+ on CWT spectral features.
Our CWT features HURT tree models (-0.009 in ablation) but this is because trees
make axis-aligned splits -- inefficient for correlated spectral bands.

This experiment:
1. Trains SVM (RBF kernel) on 64 CWT features -> OOF predictions
2. Trains tree ensemble on base tabular features -> OOF predictions
3. Blends SVM + tree OOF predictions at various weights
4. Evaluates with LOMO CV

PRIMARY EVALUATION: LOMO
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
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
W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}


def train_tree_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label):
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb = m_lgb.predict(X_va)
    test_lgb = m_lgb.predict(X_test) if X_test is not None else None

    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn),
                       2000, evals=[(xgb.DMatrix(X_va, label=y_va, feature_names=fn), "val")],
                       early_stopping_rounds=80, verbose_eval=0)
    oof_xgb = m_xgb.predict(xgb.DMatrix(X_va, feature_names=fn))
    test_xgb = m_xgb.predict(xgb.DMatrix(X_test, feature_names=fn)) if X_test is not None else None

    cb = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0, sample_weight=w_tr)
    oof_cb = cb.predict_proba(X_va)
    test_cb = cb.predict_proba(X_test) if X_test is not None else None

    oof = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
    test_ens = (W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb) if X_test is not None else None

    m, _ = compute_map(y_va, oof)
    print(f"  Tree {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_ens


def train_svm_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, label):
    """Train SVM with RBF kernel on CWT features."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True,
              class_weight='balanced', random_state=42, cache_size=1000)
    svm.fit(X_tr_s, y_tr, sample_weight=w_tr)

    oof = svm.predict_proba(X_va_s)
    test_pred = None
    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        test_pred = svm.predict_proba(X_test_s)

    m, _ = compute_map(y_va, oof)
    print(f"  SVM {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_pred


# ======================================================================
# Load data
# ======================================================================
print("=" * 60, flush=True)
print("E46 ZAUGG CWT + SVM STACKING", flush=True)
print("=" * 60, flush=True)

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
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# ======================================================================
# Build features
# ======================================================================
print("\nBuilding features (base + CWT)...", flush=True)

# Tree features (E38 pipeline)
feat_sets_tree = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets_tree)
test_feats = build_features(test_df, feature_sets=feat_sets_tree)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
tree_cols = list(train_feats.columns)

# CWT features (for SVM)
print("Building CWT features for SVM...", flush=True)
feat_sets_cwt = ["zaugg_cwt"]
train_cwt = build_features(train_df, feature_sets=feat_sets_cwt)
test_cwt = build_features(test_df, feature_sets=feat_sets_cwt)
# Also add core features to give SVM more context
feat_sets_cwt_core = ["core", "zaugg_cwt"]
train_cwt_core = build_features(train_df, feature_sets=feat_sets_cwt_core)
test_cwt_core = build_features(test_df, feature_sets=feat_sets_cwt_core)

cwt_cols = list(train_cwt.columns)
cwt_core_cols = list(train_cwt_core.columns)
print(f"  CWT-only features: {len(cwt_cols)}", flush=True)
print(f"  CWT+core features: {len(cwt_core_cols)}", flush=True)

# Add weather + solar + GBIF to tree features
print("Loading weather + solar + GBIF...", flush=True)
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

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
train_cwt_core = train_cwt_core.replace([np.inf, -np.inf], np.nan).fillna(0)
test_cwt_core = test_cwt_core.replace([np.inf, -np.inf], np.nan).fillna(0)

e38_cols = list(train_feats.columns)
print(f"\n  Tree (E38) features: {len(e38_cols)}", flush=True)

# ======================================================================
# LOMO evaluation
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("LOMO EVALUATION", flush=True)
print("=" * 60, flush=True)

# Get tree and SVM OOF predictions via LOMO
X_tree = train_feats[e38_cols].values.astype(np.float32)
X_svm = train_cwt_core[cwt_core_cols].values.astype(np.float32)
fn_tree = list(e38_cols)

oof_tree = np.zeros((len(y), N_CLASSES))
oof_svm = np.zeros((len(y), N_CLASSES))

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    # Tree
    tree_fold, _ = train_tree_fold(
        X_tree[tr_idx], y[tr_idx], X_tree[va_idx], y[va_idx],
        sample_weights[tr_idx], None, fn_tree, f"LOMO Month {m}",
    )
    oof_tree[va_idx] = tree_fold

    # SVM
    svm_fold, _ = train_svm_fold(
        X_svm[tr_idx], y[tr_idx], X_svm[va_idx], y[va_idx],
        sample_weights[tr_idx], None, f"LOMO Month {m}",
    )
    oof_svm[va_idx] = svm_fold

tree_map, tree_per = compute_map(y, oof_tree)
svm_map, svm_per = compute_map(y, oof_svm)

print(f"\n  Tree LOMO: {tree_map:.4f}", flush=True)
print(f"  SVM LOMO:  {svm_map:.4f}", flush=True)

# Blend at various weights
print("\n  Blend sweep:", flush=True)
best_blend = None
best_blend_map = 0
for svm_w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    blended = (1 - svm_w) * oof_tree + svm_w * oof_svm
    blend_map, _ = compute_map(y, blended)
    delta = blend_map - tree_map
    print(f"    SVM={svm_w:.0%}, Tree={1-svm_w:.0%}: LOMO={blend_map:.4f} (delta={delta:+.4f})", flush=True)
    if blend_map > best_blend_map:
        best_blend_map = blend_map
        best_blend = svm_w

print(f"\n  Best blend: SVM={best_blend:.0%}, LOMO={best_blend_map:.4f}", flush=True)

# Per-class comparison
print(f"\n  Per-class LOMO:", flush=True)
print(f"  {'Class':<15s} {'Tree':>7s} {'SVM':>7s} {'Best':>7s}", flush=True)
blended = (1 - best_blend) * oof_tree + best_blend * oof_svm
_, blend_per = compute_map(y, blended)
for cls in CLASSES:
    t = tree_per.get(cls, 0)
    s = svm_per.get(cls, 0)
    b = blend_per.get(cls, 0)
    print(f"  {cls:<15s} {t:>7.4f} {s:>7.4f} {b:>7.4f}", flush=True)

# ======================================================================
# SKF + test predictions with best blend
# ======================================================================
print("\n" + "=" * 60, flush=True)
print(f"SKF EVALUATION + TEST PREDICTIONS (SVM={best_blend:.0%})", flush=True)
print("=" * 60, flush=True)

X_test_tree = test_feats[e38_cols].values.astype(np.float32)
X_test_svm = test_cwt_core[cwt_core_cols].values.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_tree_skf = np.zeros((len(y), N_CLASSES))
oof_svm_skf = np.zeros((len(y), N_CLASSES))
test_tree = np.zeros((len(X_test_tree), N_CLASSES))
test_svm_pred = np.zeros((len(X_test_svm), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tree, y)):
    tree_fold, tree_test = train_tree_fold(
        X_tree[tr_idx], y[tr_idx], X_tree[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test_tree, fn_tree, f"SKF Fold {fold_idx}",
    )
    oof_tree_skf[va_idx] = tree_fold
    test_tree += tree_test / 5

    svm_fold, svm_test = train_svm_fold(
        X_svm[tr_idx], y[tr_idx], X_svm[va_idx], y[va_idx],
        sample_weights[tr_idx], X_test_svm, f"SKF Fold {fold_idx}",
    )
    oof_svm_skf[va_idx] = svm_fold
    test_svm_pred += svm_test / 5

# Blend
oof_blend = (1 - best_blend) * oof_tree_skf + best_blend * oof_svm_skf
test_pred = (1 - best_blend) * test_tree + best_blend * test_svm_pred

skf_map, skf_per = compute_map(y, oof_blend)
print(f"\n  SKF CV mAP: {skf_map:.4f}", flush=True)
print(f"  LOMO mAP:   {best_blend_map:.4f}", flush=True)

# Test distribution
print(f"\n  Test class distribution (argmax):", flush=True)
dist = np.bincount(test_pred.argmax(axis=1), minlength=N_CLASSES)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: {dist[i]}", flush=True)

# Save
np.save(ROOT / "oof_e46.npy", oof_blend)
np.save(ROOT / "test_e46.npy", test_pred)
save_submission(test_pred, "e46_cwt_svm_stack", cv_map=skf_map)

print("\nDone!", flush=True)
