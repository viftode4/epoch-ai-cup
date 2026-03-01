"""E55: Absolute-Hz Wingbeat + Month-Specific Bayesian Prior.

Combines:
- E16 biology: absolute-Hz wingbeat bands + trajectory linearity.
- E54 structure: month-specific GBIF Bayesian prior adjustment as post-processing.

Feature set: core + flight_mode + flight_physics + weakclass + absolute_wingbeat
             + linearity + tabular (ALL_TEMPORAL dropped).
Training:    LGB(40%) + XGB(30%) + CB(30%) with minority oversampling.
Post-proc:   Month-specific GBIF prior adjustment (spring_tilt, winter_tilt).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

OVERSAMPLE = {1: 4, 6: 3, 7: 3, 3: 2}  # Cormorants, BoP, Waders, Ducks

# ── Month-specific Bayesian adjustment (from E54) ──────────────────────────

def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def build_gbif_priors(p_train):
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(N_CLASSES)
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals

    priors = {}
    for month in range(1, 13):
        raw = p_train * si[month]
        raw = np.maximum(raw, 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_month_adjustment(preds, months, priors, p_train, alpha_map):
    out = preds.copy()
    for month, alpha in alpha_map.items():
        mask = months == month
        if mask.sum() == 0 or alpha == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[mask] = out[mask] * ratio
        out[mask] = out[mask] / np.clip(out[mask].sum(axis=1, keepdims=True), 1e-12, None)
    return out


# ── Data & features ────────────────────────────────────────────────────────

print("=" * 70, flush=True)
print("E55 ABS-HZ WINGBEAT + MONTH-SPECIFIC PRIOR".center(70), flush=True)
print("=" * 70, flush=True)

print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

print("\nBuilding features...", flush=True)
feat_sets = [
    "core", "flight_mode", "flight_physics",
    "weakclass", "absolute_wingbeat", "linearity",
    "tabular",
]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
feature_names = list(train_feats.columns)
print(f"  Features: {X.shape[1]}", flush=True)

# ── Training ───────────────────────────────────────────────────────────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), N_CLASSES), dtype=np.float32)
test_preds = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)

print("\nTraining 5-fold LGB+XGB+CB ensemble...", flush=True)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # Minority oversampling
    X_parts, y_parts = [X_tr], [y_tr]
    for cls_idx, mult in OVERSAMPLE.items():
        mask = y_tr == cls_idx
        if mask.sum() > 0:
            X_parts.append(np.tile(X_tr[mask], (mult - 1, 1)))
            y_parts.append(np.tile(y_tr[mask], (mult - 1,)))
    X_res = np.vstack(X_parts)
    y_res = np.concatenate(y_parts)
    perm = np.random.RandomState(fold).permutation(len(X_res))
    X_res, y_res = X_res[perm], y_res[perm]

    counts = np.bincount(y_res, minlength=N_CLASSES).astype(float)
    weights_per_class = len(y_res) / (N_CLASSES * counts)
    w_res = np.array([weights_per_class[yi] for yi in y_res])

    # LightGBM (40%)
    dtrain = lgb.Dataset(X_res, label=y_res, weight=w_res, feature_name=feature_names)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)
    lgb_model = lgb.train(
        {
            "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
            "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
            "subsample": 0.7, "colsample_bytree": 0.6, "reg_lambda": 3,
            "verbose": -1, "seed": 42,
        },
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_preds[va_idx] += lgb_model.predict(X_va) * 0.4
    test_preds += lgb_model.predict(X_test) * 0.4 / 5

    # XGBoost (30%)
    dtrain_xgb = xgb.DMatrix(X_res, label=y_res, weight=w_res, feature_names=feature_names)
    dval_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    xgb_model = xgb.train(
        {
            "objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": "mlogloss",
            "eta": 0.03, "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.6,
            "seed": 42, "verbosity": 0,
        },
        dtrain_xgb,
        num_boost_round=2000,
        evals=[(dval_xgb, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    oof_preds[va_idx] += xgb_model.predict(dval_xgb) * 0.3
    test_preds += xgb_model.predict(xgb.DMatrix(X_test, feature_names=feature_names)) * 0.3 / 5

    # CatBoost (30%)
    cb_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=5,
        loss_function="MultiClass", random_seed=42,
        verbose=0, allow_writing_files=False,
    )
    cb_model.fit(X_res, y_res, eval_set=(X_va, y_va), verbose=0)
    oof_preds[va_idx] += cb_model.predict_proba(X_va) * 0.3
    test_preds += cb_model.predict_proba(X_test) * 0.3 / 5

    fold_map, _ = compute_map(y_va, oof_preds[va_idx])
    print(f"  Fold {fold}: val mAP = {fold_map:.4f}", flush=True)

cv_map, per_class = compute_map(y, oof_preds)
print_results(cv_map, per_class, label="E55 base (CV OOF)")

np.save(ROOT / "oof_e55.npy", oof_preds)
np.save(ROOT / "test_e55.npy", test_preds)
save_submission(test_preds, "e55_base", cv_map=cv_map)

# ── Month-specific Bayesian adjustment (E54 style) ─────────────────────────

counts_train = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts_train / counts_train.sum()
priors = build_gbif_priors(p_train)

variants = {
    "spring_tilt":  {2: 0.15, 5: 0.28, 12: 0.15},
    "winter_tilt":  {2: 0.22, 5: 0.12, 12: 0.24},
}

print("\nApplying month-specific prior adjustments...", flush=True)
for name, alpha_map in variants.items():
    pred = apply_month_adjustment(test_preds, test_months, priors, p_train, alpha_map)
    pred = renorm_rows(pred)

    unseen_mask = np.isin(test_months, [2, 5, 12])
    unseen_dist = np.bincount(pred[unseen_mask].argmax(axis=1), minlength=N_CLASSES)
    full_dist = np.bincount(pred.argmax(axis=1), minlength=N_CLASSES)

    print(f"\n  Variant {name}: alpha_map={alpha_map}", flush=True)
    print(
        f"    unseen -> Gulls:{int(unseen_dist[CLASSES.index('Gulls')])} "
        f"Waders:{int(unseen_dist[CLASSES.index('Waders')])} "
        f"Pigeons:{int(unseen_dist[CLASSES.index('Pigeons')])} "
        f"Songbirds:{int(unseen_dist[CLASSES.index('Songbirds')])}",
        flush=True,
    )
    print(
        f"    full   -> Gulls:{int(full_dist[CLASSES.index('Gulls')])} "
        f"Waders:{int(full_dist[CLASSES.index('Waders')])} "
        f"Pigeons:{int(full_dist[CLASSES.index('Pigeons')])}",
        flush=True,
    )

    tag = f"e55_{name}_m2_{alpha_map[2]:.2f}_m5_{alpha_map[5]:.2f}_m12_{alpha_map[12]:.2f}"
    save_submission(pred, tag, cv_map=None)

print("\nDone.", flush=True)
