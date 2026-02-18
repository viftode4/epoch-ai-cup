"""E50: Per-class specialist blending on top of E48 flight-prior base.

Compared to E49's single global alpha, this script optimizes one alpha per
improving specialist class using LOMO OOF.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.external_priors import build_external_class_priors
from src.features import ALL_TEMPORAL, add_external_prior_features, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

SPECIALIST_CLASSES = ["Cormorants", "Waders", "Pigeons", "Ducks", "Birds of Prey"]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def add_weather_solar_gbif(train_feats, test_feats, train_months, test_months):
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
        month = int(row["month"])
        si = np.ones(N_CLASSES)
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                si[i] = 1.0
            else:
                class_counts = gbif[cls].values
                class_mean = class_counts.mean()
                si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        gbif_si[month] = si

    for i, cls in enumerate(CLASSES):
        col = f"gbif_si_{cls.lower().replace(' ', '_')}"
        train_feats[col] = [gbif_si[m][i] for m in train_months]
        test_feats[col] = [gbif_si[m][i] for m in test_months]

    gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
    month_entropy = {}
    for _, row in gbif_priors_df.iterrows():
        month = int(row["month"])
        probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
        month_entropy[month] = -np.sum(probs * np.log(probs))

    train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
    test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]
    return train_feats, test_feats


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def apply_blend(base_pred, specialist_pred, alpha_map):
    out = base_pred.copy()
    for cls, alpha in alpha_map.items():
        idx = CLASSES.index(cls)
        out[:, idx] = (1.0 - alpha) * base_pred[:, idx] + alpha * specialist_pred[cls]
    return renorm_rows(out)


print("=" * 70, flush=True)
print("E50 PER-CLASS SPECIALIST BLEND".center(70), flush=True)
print("=" * 70, flush=True)

print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

print("\nBuilding E48-C feature space...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
train_feats, test_feats = add_weather_solar_gbif(train_feats, test_feats, train_months, test_months)

priors = build_external_class_priors(ROOT)
train_feats = add_external_prior_features(train_feats, priors, include_morph=False, include_flight=True)
test_feats = add_external_prior_features(test_feats, priors, include_morph=False, include_flight=True)

X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
print(f"  Features: {X.shape[1]}", flush=True)

base_oof = np.load(ROOT / "oof_e48_lomo.npy")
base_test = np.load(ROOT / "test_e48.npy")
base_map, base_per = compute_map(y, base_oof)
print_results(base_map, base_per, label="E48 base (LOMO OOF)")

print("\nTraining specialists...", flush=True)
specialist_oof = {}
specialist_test = {}
ap_delta = {}

for cls in SPECIALIST_CLASSES:
    idx = CLASSES.index(cls)
    y_bin = (y == idx).astype(int)
    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test), dtype=np.float32)

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        y_tr_bin = y_bin[tr_idx]
        y_va_bin = y_bin[va_idx]
        pos_tr = int(y_tr_bin.sum())

        if pos_tr < 4:
            base_rate = float(pos_tr / len(y_tr_bin))
            oof_bin[va_idx] = base_rate
            test_bin += base_rate / len(unique_months)
            continue

        cb = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=80,
            task_type="CPU",
        )
        cb.fit(X[tr_idx], y_tr_bin, eval_set=(X[va_idx], y_va_bin), verbose=0)
        oof_bin[va_idx] = cb.predict_proba(X[va_idx])[:, 1]
        test_bin += cb.predict_proba(X_test)[:, 1] / len(unique_months)

    ap_base = average_precision_score(y_bin, base_oof[:, idx])
    ap_spec = average_precision_score(y_bin, oof_bin)
    ap_delta[cls] = ap_spec - ap_base
    specialist_oof[cls] = oof_bin
    specialist_test[cls] = test_bin
    print(
        f"  {cls:<15s}: AP base={ap_base:.4f} | spec={ap_spec:.4f} | delta={ap_delta[cls]:+0.4f}",
        flush=True,
    )

improving = [cls for cls in SPECIALIST_CLASSES if ap_delta[cls] > 0.002]
print(f"\nImproving specialists (>0.002 AP): {improving}", flush=True)

if not improving:
    best_alpha_map = {}
    best_oof = base_oof.copy()
    best_test = base_test.copy()
    best_map = base_map
else:
    search_classes = improving[:3]  # keep combinatorial search bounded.
    best_map = -1.0
    best_alpha_map = None
    best_oof = None

    print("\nBrute-force alpha search:", flush=True)
    for combo in itertools.product(ALPHA_GRID, repeat=len(search_classes)):
        alpha_map = {cls: alpha for cls, alpha in zip(search_classes, combo)}
        oof_blend = apply_blend(base_oof, specialist_oof, alpha_map)
        m, _ = compute_map(y, oof_blend)
        if m > best_map:
            best_map = m
            best_alpha_map = alpha_map
            best_oof = oof_blend

    print(f"  Best alpha map: {best_alpha_map}", flush=True)
    print(f"  Best LOMO mAP:  {best_map:.4f} ({best_map - base_map:+.4f})", flush=True)
    best_test = apply_blend(base_test, specialist_test, best_alpha_map)

best_per = compute_map(y, best_oof)[1]
print_results(best_map, best_per, label="E50 blended (LOMO OOF)")

print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e50.npy", best_oof)
np.save(ROOT / "test_e50.npy", best_test)
save_submission(best_test, "e50_perclass_specialist_blend", cv_map=best_map)

print("\nDone.", flush=True)
