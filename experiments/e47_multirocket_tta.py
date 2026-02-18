"""E47: MultiRocket + Test-Time Augmentation

A. MultiRocket + Ridge standalone (upgrade from MiniRocket which gave 0.245 LOMO)
B. MultiRocket + tree stacking (blend MultiRocket Ridge with E38 tree ensemble)
C. Tree TTA (Gaussian noise injection averaging on tabular features at test time)

PRIMARY EVALUATION: LOMO
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score
from aeon.transformations.collection.convolution_based import MultiRocket
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.sequence import prepare_sequences
from src.submission import save_submission

N_CLASSES = len(CLASSES)

# -- E38 tree params (same as E45/E46) --
LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
    "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 10,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.3, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
    "device": "gpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 5, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cuda", "tree_method": "hist",
}
W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34


def effective_number_weights(y, beta=0.999):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = (1.0 / np.maximum(eff, 1e-6))
    w = w / w.sum() * N_CLASSES
    return w[y]


def train_tree_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label):
    """Train LGB+XGB+CB ensemble on one fold."""
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
    print(f"  {label}: mAP={m:.4f} (n={len(y_va)})", flush=True)
    return oof, test_ens


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60, flush=True)
print("E47 MULTIROCKET + TTA", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
y = le.fit_transform(train_df["bird_group"].values)
sample_weights = effective_number_weights(y)

# Months for LOMO
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
unique_months = sorted(set(train_months))
print(f"Train months: {unique_months}", flush=True)

# -- Prepare sequences for MultiRocket --
print("\nPreparing sequences...", flush=True)
SEQ_LEN = 64
train_seq = prepare_sequences(train_df, seq_len=SEQ_LEN)
test_seq = prepare_sequences(test_df, seq_len=SEQ_LEN)
print(f"Train sequences: {train_seq.shape}", flush=True)
print(f"Test sequences: {test_seq.shape}", flush=True)

# -- Prepare tabular features for tree ensemble --
print("\nBuilding tabular features (E38 config)...", flush=True)
feature_sets = ["core", "rcs_fft", "tabular", "weakclass"]
train_feats = build_features(train_df, feature_sets)
test_feats = build_features(test_df, feature_sets)

# Remove temporal features
drop_cols = [c for c in ALL_TEMPORAL if c in train_feats.columns]
train_feats.drop(columns=drop_cols, inplace=True, errors="ignore")
test_feats.drop(columns=drop_cols, inplace=True, errors="ignore")

# Add weather + solar + GBIF
for fname, df_ref in [("weather", train_df), ("solar", train_df), ("weather", test_df), ("solar", test_df)]:
    pass  # load below

for split, df in [("train", train_df), ("test", test_df)]:
    feats_ref = train_feats if split == "train" else test_feats
    for ext_name in ["weather", "solar"]:
        ext_path = Path(__file__).resolve().parent.parent / "data" / f"{split}_{ext_name}.csv"
        if ext_path.exists():
            ext = pd.read_csv(ext_path)
            ext_cols = [c for c in ext.columns if c != "track_id"]
            for c in ext_cols:
                feats_ref[c] = ext[c].values

# GBIF
gbif_path = Path(__file__).resolve().parent.parent / "data" / "gbif_monthly_counts.csv"
if gbif_path.exists():
    gbif = pd.read_csv(gbif_path)
    # Compute seasonal index per class per month
    for cls in CLASSES:
        if cls == "Clutter":
            continue
        if cls in gbif.columns:
            class_mean = gbif[cls].mean()
            gbif[f"si_{cls}"] = gbif[cls] / class_mean if class_mean > 0 else 1.0

    for split, df, feats_ref in [("train", train_df, train_feats), ("test", test_df, test_feats)]:
        months = pd.to_datetime(df["timestamp_start_radar_utc"]).dt.month.values
        month_to_idx = {int(gbif.iloc[i]["month"]): i for i in range(len(gbif))}
        for cls in CLASSES:
            col_name = f"gbif_si_{cls.lower().replace(' ', '_')}"
            if cls == "Clutter":
                feats_ref[col_name] = 1.0
            elif f"si_{cls}" in gbif.columns:
                feats_ref[col_name] = [float(gbif.iloc[month_to_idx.get(m, 0)][f"si_{cls}"]) if m in month_to_idx else 1.0 for m in months]
            else:
                feats_ref[col_name] = 1.0
        # Shannon diversity
        from scipy.stats import entropy as sp_entropy
        diversity = []
        for m in months:
            if m in month_to_idx:
                vals = [float(gbif.iloc[month_to_idx[m]].get(cls, 0)) for cls in CLASSES if cls != "Clutter" and cls in gbif.columns]
                if vals and sum(vals) > 0:
                    p = np.array(vals) / sum(vals)
                    diversity.append(sp_entropy(p))
                else:
                    diversity.append(0)
            else:
                diversity.append(0)
        feats_ref["gbif_month_diversity"] = diversity

# Clean
train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

tab_cols = list(train_feats.columns)
X_tab = train_feats.values.astype(np.float32)
X_tab_test = test_feats.values.astype(np.float32)
print(f"Tabular features: {len(tab_cols)}", flush=True)

# ============================================================
# A: MultiRocket + Ridge LOMO
# ============================================================
print("\n" + "=" * 60, flush=True)
print("A: MultiRocket + Ridge -- LOMO", flush=True)
print("=" * 60, flush=True)

oof_mr = np.zeros((len(y), N_CLASSES))
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    print(f"\n  LOMO Month {m}: train={len(tr_idx)}, val={len(va_idx)}", flush=True)

    mr = MultiRocket(n_jobs=-1, random_state=42)
    X_tr_mr = mr.fit_transform(train_seq[tr_idx])
    X_va_mr = mr.transform(train_seq[va_idx])
    print(f"    MultiRocket features: {X_tr_mr.shape[1]}", flush=True)

    # Scale
    scaler = StandardScaler()
    X_tr_mr = scaler.fit_transform(X_tr_mr)
    X_va_mr = scaler.transform(X_va_mr)

    # Ridge classifier (fast, handles high-dim well)
    ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
    ridge.fit(X_tr_mr, y[tr_idx])

    # Convert decision function to probabilities (softmax)
    decisions = ridge.decision_function(X_va_mr)
    exp_d = np.exp(decisions - decisions.max(axis=1, keepdims=True))
    probs = exp_d / exp_d.sum(axis=1, keepdims=True)
    oof_mr[va_idx] = probs

    fold_map, _ = compute_map(y[va_idx], probs)
    print(f"    Month {m}: mAP={fold_map:.4f}", flush=True)

mr_lomo, mr_per = compute_map(y, oof_mr)
print(f"\n  A: MultiRocket+Ridge LOMO: {mr_lomo:.4f}", flush=True)
print(f"  Per-class:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<15s}: {mr_per.get(cls, 0):.4f}", flush=True)

# ============================================================
# B: Tree LOMO baseline (for stacking comparison)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("B: Tree ensemble LOMO baseline", flush=True)
print("=" * 60, flush=True)

oof_tree = np.zeros((len(y), N_CLASSES))
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    oof_fold, _ = train_tree_fold(
        X_tab[tr_idx], y[tr_idx], X_tab[va_idx], y[va_idx],
        sample_weights[tr_idx], None, tab_cols, f"LOMO Month {m}",
    )
    oof_tree[va_idx] = oof_fold

tree_lomo, tree_per = compute_map(y, oof_tree)
print(f"\n  B: Tree ensemble LOMO: {tree_lomo:.4f}", flush=True)

# ============================================================
# C: Stacking -- blend tree + MultiRocket
# ============================================================
print("\n" + "=" * 60, flush=True)
print("C: Stacking -- tree + MultiRocket blend", flush=True)
print("=" * 60, flush=True)

best_blend_map = 0
best_w_mr = 0
for w_mr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blended = (1 - w_mr) * oof_tree + w_mr * oof_mr
    blend_map, _ = compute_map(y, blended)
    marker = ""
    if blend_map > best_blend_map:
        best_blend_map = blend_map
        best_w_mr = w_mr
        marker = " <-- best"
    print(f"  w_mr={w_mr:.2f}: LOMO={blend_map:.4f}{marker}", flush=True)

print(f"\n  Best blend: w_mr={best_w_mr:.2f}, LOMO={best_blend_map:.4f}", flush=True)
print(f"  Delta vs tree: {best_blend_map - tree_lomo:+.4f}", flush=True)
print(f"  Delta vs E38:  {best_blend_map - 0.3615:+.4f}", flush=True)

# ============================================================
# D: Tree TTA (noise injection averaging)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("D: Tree TTA -- noise injection at test time", flush=True)
print("=" * 60, flush=True)

N_TTA = 10
NOISE_SCALE = 0.02  # 2% Gaussian noise

# Re-run tree LOMO but with TTA on validation predictions
oof_tta = np.zeros((len(y), N_CLASSES))
for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    X_tr, y_tr = X_tab[tr_idx], y[tr_idx]
    X_va = X_tab[va_idx]
    w_tr = sample_weights[tr_idx]
    fn = tab_cols

    # Train models once
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=fn)
    dval = lgb.Dataset(X_va, label=y[va_idx], feature_name=fn, reference=dtrain)
    m_lgb = lgb.train(LGB_PARAMS, dtrain, 2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    m_xgb = xgb.train(XGB_PARAMS, xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=fn),
                       2000, evals=[(xgb.DMatrix(X_va, label=y[va_idx], feature_names=fn), "val")],
                       early_stopping_rounds=80, verbose_eval=0)

    cb = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                            loss_function="MultiClass", eval_metric="MultiClass",
                            random_seed=42, verbose=0, early_stopping_rounds=80, task_type="GPU")
    cb.fit(X_tr, y_tr, eval_set=(X_va, y[va_idx]), verbose=0, sample_weight=w_tr)

    # TTA: average predictions with noise-augmented inputs
    preds_accum = np.zeros((len(va_idx), N_CLASSES))
    for tta_i in range(N_TTA):
        rng = np.random.RandomState(42 + tta_i)
        noise = 1.0 + NOISE_SCALE * rng.randn(*X_va.shape).astype(np.float32)
        X_va_aug = X_va * noise

        p_lgb = m_lgb.predict(X_va_aug)
        p_xgb = m_xgb.predict(xgb.DMatrix(X_va_aug, feature_names=fn))
        p_cb = cb.predict_proba(X_va_aug)

        preds_accum += W_LGB * p_lgb + W_XGB * p_xgb + W_CB * p_cb

    preds_accum /= N_TTA
    oof_tta[va_idx] = preds_accum

    fold_map, _ = compute_map(y[va_idx], preds_accum)
    print(f"  LOMO Month {m}: mAP={fold_map:.4f} (n={len(va_idx)})", flush=True)

tta_lomo, tta_per = compute_map(y, oof_tta)
print(f"\n  D: Tree TTA LOMO: {tta_lomo:.4f}", flush=True)
print(f"  Delta vs tree: {tta_lomo - tree_lomo:+.4f}", flush=True)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)

results = {
    "A: MultiRocket+Ridge": (mr_lomo, mr_per),
    "B: Tree ensemble": (tree_lomo, tree_per),
    f"C: Blend (w_mr={best_w_mr:.2f})": (best_blend_map, None),
    "D: Tree TTA": (tta_lomo, tta_per),
}

print(f"\n  {'Config':<30s} {'LOMO':>7s} {'vs E38':>7s}", flush=True)
print(f"  {'-'*44}", flush=True)
for name, (lomo, _) in results.items():
    delta = lomo - 0.3615
    print(f"  {name:<30s} {lomo:>7.4f} {delta:>+7.4f}", flush=True)

# Per-class comparison
print(f"\n  Per-class LOMO:", flush=True)
print(f"  {'Class':<15s} {'MRocket':>7s} {'Tree':>7s} {'TTA':>7s}", flush=True)
for cls in CLASSES:
    mr_v = mr_per.get(cls, 0)
    tr_v = tree_per.get(cls, 0)
    tt_v = tta_per.get(cls, 0)
    print(f"  {cls:<15s} {mr_v:>7.4f} {tr_v:>7.4f} {tt_v:>7.4f}", flush=True)

# If any config beats E38, generate submission
best_name = max(results, key=lambda k: results[k][0])
best_lomo = results[best_name][0]
if best_lomo > 0.3615:
    print(f"\n  Best config {best_name} beats E38! Generating submission...", flush=True)
    # Would need to retrain on full data -- skip for now, just report
    print(f"  (Submission generation skipped -- would need full retrain)", flush=True)
else:
    print(f"\n  No config beats E38 baseline (0.3615). No submission generated.", flush=True)

print("\nDone!", flush=True)
