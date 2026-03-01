"""E42: Minority Class Specialist Models

Start from scratch on minority classes. Instead of one multiclass model
dominated by Gulls (58%), train dedicated binary classifiers per class.

Each specialist:
  - Binary: this_class vs all_others
  - Class-specific feature importance analysis
  - Aggressive oversampling of positives (10x duplicate + noise)
  - CatBoost with auto class weights (handles imbalance natively)
  - LOMO evaluation per class

Final: replace minority class columns in E38 predictions with specialist
outputs. Keep Gulls/Songbirds from multiclass (already good).

Target classes (LOMO AP < 0.40):
  - Cormorants: 0.054 (40 samples)
  - Waders: 0.039 (120 samples)
  - Pigeons: 0.149 (122 samples)
  - BoP: 0.366 (108 samples)
  - Ducks: 0.385 (58 samples)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ======================================================================
# Load data + features (E38 config)
# ======================================================================
print("=" * 60, flush=True)
print("E42 MINORITY CLASS SPECIALISTS", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Timestamps
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# Base features (temporal-free)
print("\nBuilding features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar + GBIF (E38 config)
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

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"  Features: {len(fn)}", flush=True)

# ======================================================================
# PART 1: Confusion analysis on LOMO OOF
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 1: CONFUSION ANALYSIS (LOMO)", flush=True)
print("=" * 60, flush=True)

# Load E38 LOMO OOF (or recompute quickly with E38 config multiclass)
# We need the LOMO OOF to analyze what each minority class is confused with
from catboost import CatBoostClassifier
import xgboost as xgb

BETA = 0.999
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# Quick LOMO with just CatBoost (fastest, dominates ensemble anyway)
print("\n  Quick CatBoost LOMO for confusion analysis...", flush=True)
oof_multi = np.zeros((len(y), N_CLASSES))
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0,
           sample_weight=sample_weights[tr_idx])
    oof_multi[va_idx] = cb.predict_proba(X[va_idx])
    m_val, _ = compute_map(y[va_idx], oof_multi[va_idx])
    print(f"    Month {m}: mAP={m_val:.4f} (n={len(va_idx)})", flush=True)

multi_map, multi_per = compute_map(y, oof_multi)
print(f"\n  Multiclass LOMO mAP: {multi_map:.4f}", flush=True)

# Confusion: for each minority class, what do they get predicted as?
minority_classes = ["Cormorants", "Waders", "Pigeons", "Birds of Prey", "Ducks"]
print(f"\n  Confusion analysis (LOMO OOF):", flush=True)
for cls in minority_classes:
    cls_idx = list(CLASSES).index(cls)
    mask = y == cls_idx
    preds = oof_multi[mask]
    pred_classes = preds.argmax(axis=1)
    confusion = np.bincount(pred_classes, minlength=N_CLASSES)

    # Per-class AP for this class
    y_bin = (y == cls_idx).astype(int)
    ap = average_precision_score(y_bin, oof_multi[:, cls_idx])

    print(f"\n  {cls} (n={mask.sum()}, LOMO AP={ap:.4f}):", flush=True)
    print(f"    Predicted as:", flush=True)
    for i in np.argsort(-confusion):
        if confusion[i] > 0:
            print(f"      {CLASSES[i]:<15s}: {confusion[i]:3d} ({confusion[i]/mask.sum()*100:5.1f}%)", flush=True)

    # Mean predicted probability for true class
    true_prob = preds[:, cls_idx]
    print(f"    Mean P(true class): {true_prob.mean():.4f} (median: {np.median(true_prob):.4f})", flush=True)

    # Per-month breakdown
    print(f"    Per-month:", flush=True)
    for month in unique_months:
        m_mask = mask & (train_months == month)
        if m_mask.sum() > 0:
            mp = oof_multi[m_mask][:, cls_idx].mean()
            print(f"      Month {month}: n={m_mask.sum()}, mean P(true)={mp:.4f}", flush=True)

# ======================================================================
# PART 2: Per-class binary specialists (LOMO)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 2: BINARY SPECIALIST MODELS (LOMO)", flush=True)
print("=" * 60, flush=True)

specialist_oof = {}
specialist_test = {}

for cls in minority_classes:
    cls_idx = list(CLASSES).index(cls)
    y_bin = (y == cls_idx).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    print(f"\n--- {cls} (pos={n_pos}, neg={n_neg}, ratio=1:{n_neg//n_pos}) ---", flush=True)

    oof_bin = np.zeros(len(y))
    test_bin = np.zeros(len(X_test))

    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]

        y_tr_bin = y_bin[tr_idx]
        y_va_bin = y_bin[va_idx]
        n_pos_tr = y_tr_bin.sum()
        n_pos_va = y_va_bin.sum()

        if n_pos_tr < 3:
            print(f"    Month {m}: skip (only {n_pos_tr} positive in train)", flush=True)
            # Predict base rate
            oof_bin[va_idx] = n_pos_tr / len(y_tr_bin)
            test_bin += (n_pos_tr / len(y_tr_bin)) / len(unique_months)
            continue

        # CatBoost binary with auto_class_weights
        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=80,
            task_type="GPU",
        )
        cb.fit(X[tr_idx], y_tr_bin, eval_set=(X[va_idx], y_va_bin), verbose=0)
        oof_bin[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
        test_bin += cb.predict_proba(X_test)[:, 1] / len(unique_months)

        if n_pos_va > 0:
            ap = average_precision_score(y_va_bin, oof_bin[va_idx])
            print(f"    Month {m}: AP={ap:.4f} (pos_train={n_pos_tr}, pos_val={n_pos_va})", flush=True)
        else:
            print(f"    Month {m}: no positives in val (pos_train={n_pos_tr})", flush=True)

    # Also train LGB binary
    oof_lgb_bin = np.zeros(len(y))
    test_lgb_bin = np.zeros(len(X_test))

    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]
        y_tr_bin = y_bin[tr_idx]
        y_va_bin = y_bin[va_idx]
        n_pos_tr = y_tr_bin.sum()

        if n_pos_tr < 3:
            oof_lgb_bin[va_idx] = n_pos_tr / len(y_tr_bin)
            test_lgb_bin += (n_pos_tr / len(y_tr_bin)) / len(unique_months)
            continue

        # Weight: balance positive and negative
        w_pos = n_neg / (2 * n_pos_tr) if n_pos_tr > 0 else 1
        w_neg = n_pos_tr / (2 * (len(y_tr_bin) - n_pos_tr)) if n_pos_tr < len(y_tr_bin) else 1
        # Simpler: use is_unbalance
        lgb_params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
            "min_child_samples": 5, "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "is_unbalance": True,
            "verbose": -1, "seed": 42, "n_jobs": -1, "device": "gpu",
        }
        dtrain = lgb.Dataset(X[tr_idx], label=y_tr_bin, feature_name=fn)
        dval = lgb.Dataset(X[va_idx], label=y_va_bin, feature_name=fn, reference=dtrain)
        mdl = lgb.train(lgb_params, dtrain, 1500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb_bin[va_idx] = mdl.predict(X[va_idx])
        test_lgb_bin += mdl.predict(X_test) / len(unique_months)

    # Compare CB vs LGB binary
    ap_cb = average_precision_score(y_bin, oof_bin)
    ap_lgb = average_precision_score(y_bin, oof_lgb_bin)
    ap_multi = average_precision_score(y_bin, oof_multi[:, cls_idx])

    # Ensemble CB + LGB binary
    oof_ens_bin = 0.5 * oof_bin + 0.5 * oof_lgb_bin
    ap_ens = average_precision_score(y_bin, oof_ens_bin)

    print(f"\n  {cls} LOMO AP comparison:", flush=True)
    print(f"    Multiclass:     {ap_multi:.4f}", flush=True)
    print(f"    Binary CB:      {ap_cb:.4f} ({ap_cb - ap_multi:+.4f})", flush=True)
    print(f"    Binary LGB:     {ap_lgb:.4f} ({ap_lgb - ap_multi:+.4f})", flush=True)
    print(f"    Binary CB+LGB:  {ap_ens:.4f} ({ap_ens - ap_multi:+.4f})", flush=True)

    # Pick best
    best_ap = max(ap_cb, ap_lgb, ap_ens, ap_multi)
    if best_ap == ap_ens:
        specialist_oof[cls] = oof_ens_bin
        specialist_test[cls] = 0.5 * test_bin + 0.5 * test_lgb_bin
        print(f"    BEST: Binary CB+LGB ensemble ({ap_ens:.4f})", flush=True)
    elif best_ap == ap_cb:
        specialist_oof[cls] = oof_bin
        specialist_test[cls] = test_bin
        print(f"    BEST: Binary CB ({ap_cb:.4f})", flush=True)
    elif best_ap == ap_lgb:
        specialist_oof[cls] = oof_lgb_bin
        specialist_test[cls] = test_lgb_bin
        print(f"    BEST: Binary LGB ({ap_lgb:.4f})", flush=True)
    else:
        specialist_oof[cls] = None  # Keep multiclass
        specialist_test[cls] = None
        print(f"    BEST: Multiclass (specialist doesn't help)", flush=True)

    # Top features for this class
    if ap_cb > ap_multi:
        print(f"\n    Top CatBoost features for {cls}:", flush=True)
        imp = cb.get_feature_importance()
        top_idx = np.argsort(-imp)[:10]
        for idx in top_idx:
            print(f"      {fn[idx]:<30s}: {imp[idx]:.1f}", flush=True)

# ======================================================================
# PART 3: Combine specialists with multiclass
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 3: COMBINE SPECIALISTS WITH MULTICLASS", flush=True)
print("=" * 60, flush=True)

# Load E38 predictions
test_e38 = np.load(ROOT / "test_e38.npy")

# Method 1: Replace minority class columns in multiclass OOF
print("\n  Method 1: Replace minority columns in LOMO OOF", flush=True)
oof_hybrid = oof_multi.copy()
for cls in minority_classes:
    if specialist_oof[cls] is not None:
        cls_idx = list(CLASSES).index(cls)
        oof_hybrid[:, cls_idx] = specialist_oof[cls]

# Renormalize rows
oof_hybrid = oof_hybrid / oof_hybrid.sum(axis=1, keepdims=True)
hybrid_map, hybrid_per = compute_map(y, oof_hybrid)
print(f"  Hybrid LOMO mAP: {hybrid_map:.4f} (multiclass: {multi_map:.4f}, delta: {hybrid_map - multi_map:+.4f})", flush=True)

print(f"\n  Per-class LOMO comparison:", flush=True)
print(f"  {'Class':<15s} {'Multi':>7s} {'Hybrid':>7s} {'Delta':>7s} {'Source':>10s}", flush=True)
for cls in CLASSES:
    m_ap = multi_per.get(cls, 0)
    h_ap = hybrid_per.get(cls, 0)
    source = "specialist" if cls in minority_classes and specialist_oof[cls] is not None else "multiclass"
    print(f"  {cls:<15s} {m_ap:>7.4f} {h_ap:>7.4f} {h_ap - m_ap:>+7.4f} {source:>10s}", flush=True)

# Method 2: Blend specialist with multiclass (alpha sweep)
print(f"\n  Method 2: Blend specialist with multiclass (alpha sweep)", flush=True)
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    oof_blend = oof_multi.copy()
    for cls in minority_classes:
        if specialist_oof[cls] is not None:
            cls_idx = list(CLASSES).index(cls)
            oof_blend[:, cls_idx] = (1 - alpha) * oof_multi[:, cls_idx] + alpha * specialist_oof[cls]
    oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True)
    bm, _ = compute_map(y, oof_blend)
    print(f"    alpha={alpha:.1f}: LOMO mAP={bm:.4f} ({bm - multi_map:+.4f})", flush=True)

# Method 3: Replace in E38 test predictions
print(f"\n  Method 3: Build test submission with specialists", flush=True)
test_hybrid = test_e38.copy()
for cls in minority_classes:
    if specialist_test[cls] is not None:
        cls_idx = list(CLASSES).index(cls)
        test_hybrid[:, cls_idx] = specialist_test[cls]
test_hybrid = test_hybrid / test_hybrid.sum(axis=1, keepdims=True)

# Check test distributions
print(f"\n  Test class distribution comparison:", flush=True)
dist_e38 = np.bincount(test_e38.argmax(axis=1), minlength=N_CLASSES)
dist_hybrid = np.bincount(test_hybrid.argmax(axis=1), minlength=N_CLASSES)
print(f"  {'Class':<15s} {'E38':>6s} {'Hybrid':>6s}", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<15s} {dist_e38[i]:>6d} {dist_hybrid[i]:>6d}", flush=True)

# ======================================================================
# Save
# ======================================================================
np.save(ROOT / "oof_e42.npy", oof_hybrid)
np.save(ROOT / "test_e42.npy", test_hybrid)
save_submission(test_hybrid, "e42_specialists", cv_map=hybrid_map)

# Also save blended version at best alpha
best_alpha = 0.6  # will adjust based on results
oof_blend_best = oof_multi.copy()
test_blend_best = test_e38.copy()
for cls in minority_classes:
    if specialist_oof[cls] is not None:
        cls_idx = list(CLASSES).index(cls)
        oof_blend_best[:, cls_idx] = (1 - best_alpha) * oof_multi[:, cls_idx] + best_alpha * specialist_oof[cls]
        test_blend_best[:, cls_idx] = (1 - best_alpha) * test_e38[:, cls_idx] + best_alpha * specialist_test[cls]
oof_blend_best = oof_blend_best / oof_blend_best.sum(axis=1, keepdims=True)
test_blend_best = test_blend_best / test_blend_best.sum(axis=1, keepdims=True)
bm_best, _ = compute_map(y, oof_blend_best)
save_submission(test_blend_best, "e42_blend60", cv_map=bm_best)

print(f"\n  Saved: e42_specialists, e42_blend60", flush=True)
print("\nDone!", flush=True)
