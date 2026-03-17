"""E157: MultiRocket + LGB -- Kaggle GPU notebook.

Setup: Add datasets:
  - ai-cup-2026 (competition)
  - epoch-src (private: src/, data/best_features.txt, data/*weather*, data/*solar*)
Enable GPU accelerator.

pip install aeon (if not available)
"""

import subprocess, sys
try:
    from aeon.transformations.collection.convolution_based import MultiRocket
    print("aeon already installed")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "aeon"])
    from aeon.transformations.collection.convolution_based import MultiRocket
    print("aeon installed")

import os, importlib
from pathlib import Path

INPUT = Path("/kaggle/input")

def _find_file(marker):
    for p in INPUT.rglob(marker):
        return p
    raise FileNotFoundError(f"Cannot find {marker} under {INPUT}")

_data_py = _find_file(os.path.join("src", "data.py"))
SRC_PARENT = _data_py.parent.parent
_best_feat = _find_file("best_features.txt")
EXT_DATA_DIR = _best_feat.parent
COMP_DIR = None
for p in INPUT.rglob("train.csv"):
    if "sample_submission.csv" in [f.name for f in p.parent.iterdir()]:
        COMP_DIR = p.parent
        break
if COMP_DIR is None:
    raise FileNotFoundError("Cannot find competition data dir")

print(f"src/ at {SRC_PARENT / 'src'}", flush=True)
print(f"ext data at {EXT_DATA_DIR}", flush=True)
print(f"comp data at {COMP_DIR}", flush=True)
sys.path.insert(0, str(SRC_PARENT))

import src.data as _dm
_dm.ROOT = SRC_PARENT
_dm.DATA_DIR = COMP_DIR

# ?? Imports ???????????????????????????????????????????????????????????
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data import CLASSES, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.sequence import prepare_sequences_v2

N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42

KEEP_FEATURES = [
    f.strip() for f in (EXT_DATA_DIR / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# ??????????????????????????????????????????????????????????????????????
print("=" * 70, flush=True)
print("E157 MULTIROCKET + LGB".center(70), flush=True)
print("=" * 70, flush=True)

# ?? Load data ?????????????????????????????????????????????????????????
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# ?? Prepare sequences (v2, resample for ROCKET) ??????????????????????
print("\nPreparing sequences (v2, resample, max_len=200)...", flush=True)
train_seq, train_mask, train_lengths = prepare_sequences_v2(
    train_df, mode="resample", max_len=200, resample_rate=1.0,
)
test_seq, test_mask, test_lengths = prepare_sequences_v2(
    test_df, mode="resample", max_len=200, resample_rate=1.0,
)
print(f"  Train sequences: {train_seq.shape}", flush=True)
print(f"  Test sequences:  {test_seq.shape}", flush=True)

# ?? Extract MultiRocket features ?????????????????????????????????????
print("\nExtracting MultiRocket features...", flush=True)
N_KERNELS = 2000
mr = MultiRocket(n_kernels=N_KERNELS, random_state=SEED, n_jobs=-1)
print(f"  Using {N_KERNELS} kernels", flush=True)

mr.fit(train_seq, y)
X_mr_train = np.asarray(mr.transform(train_seq), dtype=np.float32)
X_mr_test = np.asarray(mr.transform(test_seq), dtype=np.float32)
print(f"  MultiRocket features: {X_mr_train.shape[1]}", flush=True)

X_mr_train = np.where(np.isfinite(X_mr_train), X_mr_train, 0.0)
X_mr_test = np.where(np.isfinite(X_mr_test), X_mr_test, 0.0)

# ?? Build tabular features (36 pruned) ???????????????????????????????
print("\nBuilding tabular features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Weather + solar
train_weather = pd.read_csv(EXT_DATA_DIR / "train_weather.csv")
test_weather = pd.read_csv(EXT_DATA_DIR / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

train_solar = pd.read_csv(EXT_DATA_DIR / "train_solar.csv")
test_solar = pd.read_csv(EXT_DATA_DIR / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

available = [f for f in KEEP_FEATURES if f in train_feats.columns]
print(f"  Using {len(available)}/{len(KEEP_FEATURES)} tabular features", flush=True)

X_tab_train = train_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_tab_test = test_feats[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

X_combined_train = np.hstack([X_mr_train, X_tab_train])
X_combined_test = np.hstack([X_mr_test, X_tab_test])
print(f"  Combined features: {X_combined_train.shape[1]}", flush=True)

# ?? Class weights ????????????????????????????????????????????????????
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]

# ?? CV helper ????????????????????????????????????????????????????????
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = {}


def train_lgb_cv(X_train, X_test_full, tag):
    print(f"\n--- {tag} ---", flush=True)
    print(f"  Features: {X_train.shape[1]}", flush=True)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_preds = np.zeros((len(X_test_full), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}", flush=True)
        lgb = LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu", n_jobs=-1,
        )
        lgb.fit(X_train[tr_idx], y[tr_idx], eval_set=[(X_train[va_idx], y[va_idx])])
        oof[va_idx] = lgb.predict_proba(X_train[va_idx])
        test_preds += lgb.predict_proba(X_test_full) / N_FOLDS

    m, per = compute_map(y, oof)
    print_results(m, per, label=f"E157 {tag}")
    results[tag] = {"map": m, "per_class": per, "oof": oof, "test": test_preds}
    return oof, test_preds, m


# ?? Variant A: MultiRocket only ??????????????????????????????????????
oof_a, test_a, map_a = train_lgb_cv(X_mr_train, X_mr_test, "A: MultiRocket only")

# ?? Variant B: MultiRocket + Tabular ?????????????????????????????????
oof_b, test_b, map_b = train_lgb_cv(X_combined_train, X_combined_test, "B: MultiRocket + Tabular")

# ?? Variant C: LGB + XGB on combined ?????????????????????????????????
if map_b > map_a:
    print("\n--- Variant C: LGB + XGB ensemble ---", flush=True)
    oof_lgb_c = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    oof_xgb_c = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    test_lgb_c = np.zeros((len(X_combined_test), N_CLASSES), dtype=np.float64)
    test_xgb_c = np.zeros((len(X_combined_test), N_CLASSES), dtype=np.float64)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_combined_train, y)):
        print(f"  Fold {fold_i+1}/{N_FOLDS}", flush=True)
        lgb = LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu", n_jobs=-1,
        )
        lgb.fit(X_combined_train[tr_idx], y[tr_idx], eval_set=[(X_combined_train[va_idx], y[va_idx])])
        oof_lgb_c[va_idx] = lgb.predict_proba(X_combined_train[va_idx])
        test_lgb_c += lgb.predict_proba(X_combined_test) / N_FOLDS

        xgb = XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
            objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", random_state=SEED, verbosity=0,
            device="cuda", tree_method="hist",
        )
        xgb.fit(X_combined_train[tr_idx], y[tr_idx],
                eval_set=[(X_combined_train[va_idx], y[va_idx])],
                sample_weight=sample_weights[tr_idx], verbose=False)
        oof_xgb_c[va_idx] = xgb.predict_proba(X_combined_train[va_idx])
        test_xgb_c += xgb.predict_proba(X_combined_test) / N_FOLDS

    best_w_c, best_map_c = None, -1.0
    for w in np.arange(0.0, 1.05, 0.05):
        oof_ens = w * oof_lgb_c + (1 - w) * oof_xgb_c
        m, _ = compute_map(y, oof_ens)
        if m > best_map_c:
            best_map_c = m
            best_w_c = w
    oof_c = best_w_c * oof_lgb_c + (1 - best_w_c) * oof_xgb_c
    test_c = best_w_c * test_lgb_c + (1 - best_w_c) * test_xgb_c
    _, per_c = compute_map(y, oof_c)
    print(f"  Best weights: LGB={best_w_c:.2f} XGB={1-best_w_c:.2f}", flush=True)
    print_results(best_map_c, per_c, label="E157 C: LGB+XGB ensemble")
    results["C: LGB+XGB"] = {"map": best_map_c, "per_class": per_c, "oof": oof_c, "test": test_c}

# ?? Summary ??????????????????????????????????????????????????????????
print("\n" + "=" * 70, flush=True)
print("SUMMARY".center(70), flush=True)
print("=" * 70, flush=True)
print(f"  E79 reference:  SKF 0.7736", flush=True)
for tag, r in results.items():
    print(f"  E157 {tag}: SKF {r['map']:.4f}", flush=True)

best_tag = max(results, key=lambda k: results[k]["map"])
best_r = results[best_tag]
print(f"\nBest: {best_tag} (SKF {best_r['map']:.4f})", flush=True)

# ?? Save ?????????????????????????????????????????????????????????????
sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
sub_columns = [c for c in sample_sub.columns if c != "track_id"]
sub = pd.DataFrame({"track_id": test_df["track_id"]})
for col in sub_columns:
    cls_idx = CLASSES.index(col)
    sub[col] = best_r["test"][:, cls_idx]
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"\nSaved submission.csv ({len(sub)} rows)", flush=True)

np.save("/kaggle/working/oof_e157.npy", best_r["oof"])
np.save("/kaggle/working/test_e157.npy", best_r["test"])
print("Done.", flush=True)
