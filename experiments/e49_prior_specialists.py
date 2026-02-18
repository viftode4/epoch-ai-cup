"""E49: Minority specialists on top of E48 flight-prior model.

Pipeline:
1) Rebuild E48-C feature space (E38 base + external flight priors).
2) Train class-specific binary specialists with LOMO.
3) Blend specialist columns into E48 LOMO OOF and E48 test predictions.
4) Sweep blend alpha on LOMO, then save best submission.
"""

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
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8]


def add_weather_solar_gbif(train_feats, test_feats, train_months, test_months):
    """Add E38 weather + solar + GBIF features."""
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


print("=" * 70, flush=True)
print("E49 PRIOR + SPECIALISTS".center(70), flush=True)
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

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
print(f"  Features: {X.shape[1]}", flush=True)

print("\nLoading E48 predictions...", flush=True)
base_oof_lomo_path = ROOT / "oof_e48_lomo.npy"
base_test_path = ROOT / "test_e48.npy"
if not base_oof_lomo_path.exists() or not base_test_path.exists():
    raise FileNotFoundError(
        "Missing E48 artifacts. Run experiments/e48_external_priors.py first."
    )
base_oof = np.load(base_oof_lomo_path)
base_test = np.load(base_test_path)
base_lomo_map, base_lomo_per = compute_map(y, base_oof)
print_results(base_lomo_map, base_lomo_per, label="E48 base (LOMO OOF)")

print("\n" + "=" * 70, flush=True)
print("TRAINING BINARY SPECIALISTS (LOMO)".center(70), flush=True)
print("=" * 70, flush=True)

specialist_oof = {}
specialist_test = {}
improved_classes = []

for cls in SPECIALIST_CLASSES:
    cls_idx = CLASSES.index(cls)
    y_bin = (y == cls_idx).astype(int)
    n_pos = int(y_bin.sum())
    print(f"\n--- {cls} (positives={n_pos}) ---", flush=True)

    oof_bin = np.zeros(len(y), dtype=np.float32)
    test_bin = np.zeros(len(X_test), dtype=np.float32)

    for month in unique_months:
        va_idx = np.where(train_months == month)[0]
        tr_idx = np.where(train_months != month)[0]
        y_tr_bin = y_bin[tr_idx]
        y_va_bin = y_bin[va_idx]

        pos_tr = int(y_tr_bin.sum())
        if pos_tr < 4:
            # Fallback to base rate if too few positives in this fold.
            base_rate = float(pos_tr / len(y_tr_bin))
            oof_bin[va_idx] = base_rate
            test_bin += base_rate / len(unique_months)
            print(f"  Month {month}: fallback base-rate ({base_rate:.4f})", flush=True)
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

        if int(y_va_bin.sum()) > 0:
            ap_month = average_precision_score(y_va_bin, oof_bin[va_idx])
            print(f"  Month {month}: AP={ap_month:.4f}", flush=True)
        else:
            print(f"  Month {month}: no positives in validation", flush=True)

    ap_base = average_precision_score(y_bin, base_oof[:, cls_idx])
    ap_spec = average_precision_score(y_bin, oof_bin)
    delta = ap_spec - ap_base
    print(
        f"  AP base={ap_base:.4f} | AP specialist={ap_spec:.4f} | delta={delta:+.4f}",
        flush=True,
    )

    specialist_oof[cls] = oof_bin
    specialist_test[cls] = test_bin
    if delta > 0.002:
        improved_classes.append(cls)

print(f"\nClasses with specialist gain > 0.002: {improved_classes}", flush=True)

print("\n" + "=" * 70, flush=True)
print("BLEND SWEEP".center(70), flush=True)
print("=" * 70, flush=True)

if not improved_classes:
    print("No improving specialists found. Keeping E48 base predictions.", flush=True)
    best_alpha = 0.0
    best_oof = base_oof.copy()
    best_test = base_test.copy()
    best_map = base_lomo_map
else:
    best_alpha = None
    best_map = -1.0
    best_oof = None
    for alpha in ALPHAS:
        oof_blend = base_oof.copy()
        for cls in improved_classes:
            idx = CLASSES.index(cls)
            oof_blend[:, idx] = (1.0 - alpha) * base_oof[:, idx] + alpha * specialist_oof[cls]
        oof_blend = renorm_rows(oof_blend)
        m, _ = compute_map(y, oof_blend)
        print(f"  alpha={alpha:.1f} -> LOMO mAP={m:.4f} ({m - base_lomo_map:+.4f})", flush=True)
        if m > best_map:
            best_map = m
            best_alpha = alpha
            best_oof = oof_blend

    best_test = base_test.copy()
    for cls in improved_classes:
        idx = CLASSES.index(cls)
        best_test[:, idx] = (1.0 - best_alpha) * base_test[:, idx] + best_alpha * specialist_test[cls]
    best_test = renorm_rows(best_test)

print(f"\nBest alpha: {best_alpha:.2f}", flush=True)
best_per = compute_map(y, best_oof)[1]
print_results(best_map, best_per, label="E49 blended (LOMO OOF)")

print("\nSaving artifacts...", flush=True)
np.save(ROOT / "oof_e49.npy", best_oof)
np.save(ROOT / "test_e49.npy", best_test)
save_submission(best_test, "e49_prior_specialists", cv_map=best_map)

dist = np.bincount(best_test.argmax(axis=1), minlength=N_CLASSES)
print("\nTest argmax distribution:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<15s}: {dist[i]}", flush=True)

print("\nDone.", flush=True)
