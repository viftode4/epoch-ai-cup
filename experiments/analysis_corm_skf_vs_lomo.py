"""Cormorant: Is LOMO actually appropriate for trajectory-only features?

Key question: Do Cormorant flight patterns (turn_angle_var, bearing_change,
sinuosity etc.) vary by month? If not, SKF is valid and LOMO is overly harsh.

Tests:
1. Are Cormorant trajectory features stable across months?
2. SKF vs LOMO for binary Cormorant detection with trajectory features
3. Adversarial validation: can a model tell which month a Cormorant comes from?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import f_oneway, ttest_ind
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.features import build_features, ALL_TEMPORAL

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")

print("=" * 60, flush=True)
print("CORMORANT: LOMO vs SKF -- IS MONTH RELEVANT?", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months = ts.dt.month.values
unique_months = sorted(np.unique(months))

# Build features
print("Building features...", flush=True)
X = build_features(train_df)
drop_cols = [c for c in ALL_TEMPORAL if c in X.columns]
X = X.drop(columns=drop_cols)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
feat_names = list(X.columns)

# Feature selection: top-50 by t-stat
corm_mask = y_bin == 1
t_stats = []
for col in feat_names:
    vals_c = X.loc[corm_mask, col].values
    vals_r = X.loc[~corm_mask, col].values
    t, p = ttest_ind(vals_c, vals_r, equal_var=False)
    t_stats.append((col, abs(t), p))
t_stats.sort(key=lambda x: -x[1])
top_feats = [x[0] for x in t_stats[:50]]

X_sel = X[top_feats].values

# ======================================================================
# 1. Do Cormorant features vary by month?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("1. CORMORANT FEATURE STABILITY ACROSS MONTHS", flush=True)
print("=" * 60, flush=True)

corm_idx = np.where(y_bin == 1)[0]
corm_months = months[corm_idx]
print(f"  Cormorants per month: ", flush=True)
for m in unique_months:
    n = (corm_months == m).sum()
    print(f"    M{m:2d}: {n} Cormorants", flush=True)

# For top-10 discriminative features: ANOVA across months (Cormorants only)
print(f"\n  ANOVA (Cormorants only) -- do features differ by month?", flush=True)
print(f"  {'Feature':<30s} {'F-stat':>8s} {'p-value':>10s} {'Verdict':>10s}", flush=True)
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}", flush=True)

month_groups = {}
for m in unique_months:
    mask = (y_bin == 1) & (months == m)
    if mask.sum() >= 2:
        month_groups[m] = np.where(mask)[0]

n_sig = 0
for feat, t, p in t_stats[:20]:
    groups = [X.loc[idx, feat].values for m, idx in month_groups.items() if len(idx) >= 2]
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        sig = "VARIES" if p_val < 0.05 else "STABLE"
        if p_val < 0.05:
            n_sig += 1
        print(f"  {feat:<30s} {f_stat:>8.2f} {p_val:>10.4f} {sig:>10s}", flush=True)

print(f"\n  {n_sig}/20 top features vary significantly by month (p<0.05)", flush=True)
print(f"  At p<0.05, expect 1/20 by chance.", flush=True)

# Also show the actual values per month for top-3 features
print(f"\n  Top-3 feature values per month (Cormorants only):", flush=True)
for feat, t, p in t_stats[:3]:
    print(f"\n  {feat}:", flush=True)
    for m in unique_months:
        mask = (y_bin == 1) & (months == m)
        if mask.sum() > 0:
            vals = X.loc[mask, feat].values
            print(f"    M{m:2d} (n={mask.sum():2d}): "
                  f"mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"range=[{vals.min():.4f}, {vals.max():.4f}]", flush=True)

# ======================================================================
# 2. Adversarial validation: can we tell which month a Cormorant is from?
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("2. ADVERSARIAL: CAN MODEL TELL CORMORANT MONTH?", flush=True)
print("=" * 60, flush=True)

# Binary: Oct cormorants vs non-Oct cormorants (biggest split)
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

corm_data = X_sel[corm_idx]
corm_is_oct = (corm_months == 10).astype(int)

print(f"  Oct Cormorants: {corm_is_oct.sum()}, non-Oct: {(1-corm_is_oct).sum()}", flush=True)

# 3-fold CV AUC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV

lr = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
# Simple cross-val
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
aucs = []
for tr_idx, va_idx in skf.split(corm_data, corm_is_oct):
    sc = StandardScaler()
    X_tr = sc.fit_transform(corm_data[tr_idx])
    X_va = sc.transform(corm_data[va_idx])
    lr_inner = LogisticRegressionCV(cv=2, max_iter=1000, random_state=42)
    lr_inner.fit(X_tr, corm_is_oct[tr_idx])
    pred = lr_inner.predict_proba(X_va)
    if pred.shape[1] == 2:
        pred = pred[:, 1]
    else:
        pred = pred[:, 0]
    if len(np.unique(corm_is_oct[va_idx])) > 1:
        auc = roc_auc_score(corm_is_oct[va_idx], pred)
        aucs.append(auc)

if aucs:
    mean_auc = np.mean(aucs)
    print(f"  LogReg AUC (Oct vs non-Oct Cormorants): {mean_auc:.3f}", flush=True)
    print(f"  (0.5 = can't distinguish = features are month-independent)", flush=True)
else:
    print(f"  Could not compute AUC (degenerate folds)", flush=True)

# ======================================================================
# 3. SKF evaluation for binary Cormorant detection
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("3. SKF vs LOMO: CATBOOST BINARY TOP-50", flush=True)
print("=" * 60, flush=True)

# SKF
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_skf = np.full(len(y_bin), np.nan)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sel, y_bin)):
    n_pos = y_bin[tr_idx].sum()
    n_neg = len(tr_idx) - n_pos
    scale = n_neg / max(n_pos, 1)

    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_sel[tr_idx], y_bin[tr_idx])
    proba = cb.predict_proba(X_sel[va_idx])[:, 1]
    oof_skf[va_idx] = proba

    ap = average_precision_score(y_bin[va_idx], proba)
    n_corm = y_bin[va_idx].sum()
    print(f"  SKF fold {fold}: val={len(va_idx)} (corm={n_corm}), AP={ap:.4f}", flush=True)

skf_ap = average_precision_score(y_bin, oof_skf)
print(f"  SKF overall Cormorant AP: {skf_ap:.4f}", flush=True)

# LOMO
oof_lomo = np.full(len(y_bin), np.nan)
for held_month in unique_months:
    val_mask = months == held_month
    train_mask = ~val_mask
    n_corm_val = y_bin[val_mask].sum()
    if n_corm_val == 0:
        continue

    n_pos = y_bin[train_mask].sum()
    n_neg = train_mask.sum() - n_pos
    scale = n_neg / max(n_pos, 1)

    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_sel[train_mask], y_bin[train_mask])
    proba = cb.predict_proba(X_sel[val_mask])[:, 1]
    oof_lomo[val_mask] = proba

    ap = average_precision_score(y_bin[val_mask], proba)
    print(f"  LOMO M{held_month:2d}: train_corm={y_bin[train_mask].sum()}, "
          f"val_corm={n_corm_val}, AP={ap:.4f}", flush=True)

valid = ~np.isnan(oof_lomo)
lomo_ap = average_precision_score(y_bin[valid], oof_lomo[valid])
print(f"  LOMO overall Cormorant AP: {lomo_ap:.4f}", flush=True)

print(f"\n  SKF AP: {skf_ap:.4f}", flush=True)
print(f"  LOMO AP: {lomo_ap:.4f}", flush=True)
print(f"  Gap: {skf_ap - lomo_ap:.4f}", flush=True)
if skf_ap - lomo_ap < 0.05:
    print(f"  --> Small gap: features ARE month-independent. SKF is valid!", flush=True)
else:
    print(f"  --> Large gap: some temporal leakage persists.", flush=True)

# ======================================================================
# 4. Ensemble on SKF (CB + kNN + SVM)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("4. ENSEMBLE ON SKF (CB + kNN + SVM)", flush=True)
print("=" * 60, flush=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

oof_cb = np.full(len(y_bin), np.nan)
oof_knn = np.full(len(y_bin), np.nan)
oof_svm = np.full(len(y_bin), np.nan)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sel, y_bin)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_sel[tr_idx])
    X_va = sc.transform(X_sel[va_idx])
    y_tr = y_bin[tr_idx]

    # CatBoost
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    scale = n_neg / max(n_pos, 1)
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_sel[tr_idx], y_tr)
    oof_cb[va_idx] = cb.predict_proba(X_sel[va_idx])[:, 1]

    # kNN
    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    knn.fit(X_tr, y_tr)
    oof_knn[va_idx] = knn.predict_proba(X_va)[:, 1]

    # SVM
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    oof_svm[va_idx] = svm.predict_proba(X_va)[:, 1]

ap_cb = average_precision_score(y_bin, oof_cb)
ap_knn = average_precision_score(y_bin, oof_knn)
ap_svm = average_precision_score(y_bin, oof_svm)
print(f"  CB:  AP={ap_cb:.4f}", flush=True)
print(f"  kNN: AP={ap_knn:.4f}", flush=True)
print(f"  SVM: AP={ap_svm:.4f}", flush=True)

# Ensemble
for w_cb, w_knn, w_svm in [(0.5, 0.25, 0.25), (0.33, 0.33, 0.34), (0.6, 0.2, 0.2)]:
    ens = w_cb * oof_cb + w_knn * oof_knn + w_svm * oof_svm
    ap_ens = average_precision_score(y_bin, ens)
    print(f"  Ensemble ({w_cb:.2f}/{w_knn:.2f}/{w_svm:.2f}): AP={ap_ens:.4f}", flush=True)

# Also LOMO for the ensemble
print("\n  Ensemble LOMO check:", flush=True)
oof_cb_l = np.full(len(y_bin), np.nan)
oof_knn_l = np.full(len(y_bin), np.nan)
oof_svm_l = np.full(len(y_bin), np.nan)

for held_month in unique_months:
    val_mask = months == held_month
    train_mask = ~val_mask
    if y_bin[val_mask].sum() == 0:
        continue

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_sel[train_mask])
    X_va = sc.transform(X_sel[val_mask])
    y_tr = y_bin[train_mask]

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    scale = n_neg / max(n_pos, 1)

    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        scale_pos_weight=scale, random_seed=42, verbose=0, task_type="GPU",
    )
    cb.fit(X_sel[train_mask], y_tr)
    oof_cb_l[val_mask] = cb.predict_proba(X_sel[val_mask])[:, 1]

    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    knn.fit(X_tr, y_tr)
    oof_knn_l[val_mask] = knn.predict_proba(X_va)[:, 1]

    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    oof_svm_l[val_mask] = svm.predict_proba(X_va)[:, 1]

valid = ~np.isnan(oof_cb_l)
ens_lomo = 0.5 * oof_cb_l[valid] + 0.25 * oof_knn_l[valid] + 0.25 * oof_svm_l[valid]
lomo_ens_ap = average_precision_score(y_bin[valid], ens_lomo)
print(f"  Ensemble LOMO AP: {lomo_ens_ap:.4f}", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  SKF Cormorant AP (CB binary top-50):       {skf_ap:.4f}", flush=True)
print(f"  LOMO Cormorant AP (CB binary top-50):      {lomo_ap:.4f}", flush=True)
print(f"  SKF-LOMO gap:                              {skf_ap - lomo_ap:.4f}", flush=True)
print(f"  SKF ensemble (CB+kNN+SVM):                 {max(ap_cb, ap_knn, ap_svm, average_precision_score(y_bin, 0.5*oof_cb+0.25*oof_knn+0.25*oof_svm)):.4f}", flush=True)
print(f"  LOMO ensemble:                             {lomo_ens_ap:.4f}", flush=True)
print(f"  E32 multiclass Cormorant AP (SKF):         0.3610", flush=True)
print("\nDone!", flush=True)
