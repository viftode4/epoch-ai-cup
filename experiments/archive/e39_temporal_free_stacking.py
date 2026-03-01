"""E39: Temporal-Free Stacking Ensemble

Combines diverse base models that are ALL temporal-free:
  - E32: LGB+XGB+CB on 114 tabular features (23 temporal removed)
  - E08: MiniRocket on raw 8-channel trajectory (no features at all)
  - E06: 1D-CNN on raw trajectory (no features at all)

The sequence models (MiniRocket, CNN) learn from physical flight characteristics
(altitude, RCS, speed, bearing) which should generalize across months.

Two evaluations:
  1. SKF: uses existing OOF predictions (quick)
  2. LOMO: retrains all base models from scratch (honest)

Goal: diversity from sequence models helps unseen-month generalization.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aeon.transformations.collection.convolution_based import MiniRocket
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# E32 tree params
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

# CNN params
SEQ_LEN_CNN = 64
N_CHANNELS_CNN = 8
CNN_EPOCHS = 120
CNN_BATCH = 64
CNN_LR = 1e-3
CNN_PATIENCE = 15

# MiniRocket params
SEQ_LEN_ROCKET = 128


# ======================================================================
# CNN model (from E06, updated to 8 channels)
# ======================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class BirdCNN(nn.Module):
    def __init__(self, in_channels=8, n_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7),
            nn.Dropout(0.1),
            ConvBlock(64, 128, kernel_size=5),
            nn.Dropout(0.1),
            ConvBlock(128, 256, kernel_size=3),
            nn.Dropout(0.1),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# ======================================================================
# Helper functions
# ======================================================================
def train_tree_fold(X_tr, y_tr, X_va, y_va, w_tr, X_test, fn, label):
    """Train LGB+XGB+CB on one fold."""
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

    oof = 0.33 * oof_lgb + 0.33 * oof_xgb + 0.34 * oof_cb
    test_ens = (0.33 * test_lgb + 0.33 * test_xgb + 0.34 * test_cb) if X_test is not None else None
    m, _ = compute_map(y_va, oof)
    print(f"    Tree {label}: mAP={m:.4f}", flush=True)
    return oof, test_ens


def train_cnn_fold(X_tr, y_tr, X_va, y_va, X_test, class_weights, label):
    """Train CNN on one fold."""
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_va_n = (X_va - mean) / std

    train_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_va_n, dtype=torch.float32),
                           torch.tensor(y_va, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=CNN_BATCH * 2, shuffle=False, num_workers=0)

    model = BirdCNN(N_CHANNELS_CNN, N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)

    best_val_loss = float("inf")
    best_preds = None
    best_state = None
    patience_counter = 0

    for epoch in range(CNN_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        val_loss /= len(val_ds)
        val_preds = np.concatenate(val_preds)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CNN_PATIENCE:
            break

    # Test predictions
    test_pred = None
    if X_test is not None:
        model.load_state_dict(best_state)
        model.eval()
        X_test_n = (X_test - mean) / std
        test_ds = TensorDataset(torch.tensor(X_test_n, dtype=torch.float32))
        test_dl = DataLoader(test_ds, batch_size=CNN_BATCH * 2, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for (xb,) in test_dl:
                xb = xb.to(DEVICE)
                preds.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        test_pred = np.concatenate(preds)

    m, _ = compute_map(y_va, best_preds)
    print(f"    CNN {label}: mAP={m:.4f}", flush=True)
    return best_preds, test_pred


def train_rocket_fold(X_tr, y_tr, X_va, y_va, X_test, label):
    """Train MiniRocket on one fold."""
    rocket = MiniRocket(random_state=42)
    rocket.fit(X_tr)
    X_tr_f = rocket.transform(X_tr)
    X_va_f = rocket.transform(X_va)

    scaler = StandardScaler()
    X_tr_f = scaler.fit_transform(X_tr_f)
    X_va_f = scaler.transform(X_va_f)

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                             multi_class="multinomial", class_weight="balanced",
                             random_state=42, n_jobs=-1)
    clf.fit(X_tr_f, y_tr)

    oof = clf.predict_proba(X_va_f)
    test_pred = None
    if X_test is not None:
        X_test_f = scaler.transform(rocket.transform(X_test))
        test_pred = clf.predict_proba(X_test_f)

    m, _ = compute_map(y_va, oof)
    print(f"    Rocket {label}: mAP={m:.4f}", flush=True)
    return oof, test_pred


# ======================================================================
# Load data
# ======================================================================
print("=" * 60, flush=True)
print("E39 TEMPORAL-FREE STACKING", flush=True)
print("=" * 60, flush=True)

train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

# Effective number weights for trees
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# CNN class weights
cnn_class_weights = (len(y) / (N_CLASSES * counts)).astype(np.float32)

# Build tabular features (temporal-free)
print("\nBuilding tabular features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]
X_tab = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_tab_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
fn_tab = list(train_feats.columns)
print(f"  Tabular features: {len(fn_tab)}", flush=True)

# Build sequences
print("Building CNN sequences (8ch x 64)...", flush=True)
X_seq_cnn = prepare_sequences(train_df, seq_len=SEQ_LEN_CNN)
X_seq_cnn_test = prepare_sequences(test_df, seq_len=SEQ_LEN_CNN)

print("Building MiniRocket sequences (8ch x 128)...", flush=True)
X_seq_rocket = prepare_sequences(train_df, seq_len=SEQ_LEN_ROCKET)
X_seq_rocket_test = prepare_sequences(test_df, seq_len=SEQ_LEN_ROCKET)

# Months for LOMO
ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
train_months = ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# ======================================================================
# PART 1: Quick SKF stacking with existing OOF files
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 1: SKF STACKING (existing OOF predictions)", flush=True)
print("=" * 60, flush=True)

existing_oofs = {}
for name, oof_file in [("tree", "oof_e32.npy"), ("rocket", "oof_e08.npy"), ("cnn", "oof_e06.npy")]:
    path = ROOT / oof_file
    if path.exists():
        arr = np.load(path)
        existing_oofs[name] = arr
        m, _ = compute_map(y, arr)
        print(f"  {name}: shape={arr.shape}, mAP={m:.4f}", flush=True)
    else:
        print(f"  {name}: MISSING ({oof_file})", flush=True)

if len(existing_oofs) >= 2:
    # Fixed weight combinations
    print("\n  Fixed-weight ensembles:", flush=True)
    combos = [
        ("70/15/15", {"tree": 0.70, "rocket": 0.15, "cnn": 0.15}),
        ("60/20/20", {"tree": 0.60, "rocket": 0.20, "cnn": 0.20}),
        ("50/25/25", {"tree": 0.50, "rocket": 0.25, "cnn": 0.25}),
        ("33/33/34", {"tree": 0.33, "rocket": 0.33, "cnn": 0.34}),
    ]
    for combo_name, weights in combos:
        oof_blend = sum(weights.get(n, 0) * existing_oofs[n] for n in existing_oofs)
        m, _ = compute_map(y, oof_blend)
        print(f"    {combo_name}: mAP={m:.4f}", flush=True)

    # Grid search (tree vs rest)
    best_skf_map = 0
    best_skf_w = None
    names = list(existing_oofs.keys())
    if len(names) == 3:
        for w1 in np.arange(0.0, 1.01, 0.05):
            for w2 in np.arange(0.0, 1.01 - w1, 0.05):
                w3 = 1 - w1 - w2
                if w3 < -0.01:
                    continue
                w3 = max(w3, 0)
                oof = w1 * existing_oofs[names[0]] + w2 * existing_oofs[names[1]] + w3 * existing_oofs[names[2]]
                m, _ = compute_map(y, oof)
                if m > best_skf_map:
                    best_skf_map = m
                    best_skf_w = {names[0]: w1, names[1]: w2, names[2]: w3}
        print(f"\n  Best SKF grid search: mAP={best_skf_map:.4f}", flush=True)
        for n, w in best_skf_w.items():
            print(f"    {n}: {w:.2f}", flush=True)

    # WARNING: SKF weights are optimized on OOF and WILL be biased
    print("\n  WARNING: SKF stacking weights are biased (optimized on OOF).", flush=True)
    print("  Using FIXED weights for submission. LOMO results below are honest.", flush=True)

# ======================================================================
# PART 2: LOMO stacking (retrain everything from scratch)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 2: LOMO STACKING (retrain all base models)", flush=True)
print("=" * 60, flush=True)
print(f"  Train months: {unique_months}", flush=True)

oof_tree_lomo = np.zeros((len(y), N_CLASSES))
oof_cnn_lomo = np.zeros((len(y), N_CLASSES))
oof_rocket_lomo = np.zeros((len(y), N_CLASSES))

test_tree_lomo = np.zeros((len(X_tab_test), N_CLASSES))
test_cnn_lomo = np.zeros((len(X_seq_cnn_test), N_CLASSES))
test_rocket_lomo = np.zeros((len(X_seq_rocket_test), N_CLASSES))

n_months = len(unique_months)

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    print(f"\n--- LOMO Month {m} (train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)

    # Tree
    oof_t, test_t = train_tree_fold(
        X_tab[tr_idx], y[tr_idx], X_tab[va_idx], y[va_idx],
        sample_weights[tr_idx], X_tab_test, fn_tab, f"M{m}")
    oof_tree_lomo[va_idx] = oof_t
    test_tree_lomo += test_t / n_months

    # MiniRocket
    oof_r, test_r = train_rocket_fold(
        X_seq_rocket[tr_idx], y[tr_idx], X_seq_rocket[va_idx], y[va_idx],
        X_seq_rocket_test, f"M{m}")
    oof_rocket_lomo[va_idx] = oof_r
    test_rocket_lomo += test_r / n_months

    # CNN
    oof_c, test_c = train_cnn_fold(
        X_seq_cnn[tr_idx], y[tr_idx], X_seq_cnn[va_idx], y[va_idx],
        X_seq_cnn_test, cnn_class_weights, f"M{m}")
    oof_cnn_lomo[va_idx] = oof_c
    test_cnn_lomo += test_c / n_months

# Individual LOMO scores
print("\n  Individual LOMO scores:", flush=True)
for name, oof in [("Tree", oof_tree_lomo), ("Rocket", oof_rocket_lomo), ("CNN", oof_cnn_lomo)]:
    m_val, per = compute_map(y, oof)
    print(f"    {name}: {m_val:.4f}", flush=True)

# LOMO ensemble combinations
print("\n  LOMO ensemble combinations:", flush=True)
lomo_combos = [
    ("Tree only", {"tree": 1.0, "rocket": 0.0, "cnn": 0.0}),
    ("70/15/15", {"tree": 0.70, "rocket": 0.15, "cnn": 0.15}),
    ("60/20/20", {"tree": 0.60, "rocket": 0.20, "cnn": 0.20}),
    ("50/25/25", {"tree": 0.50, "rocket": 0.25, "cnn": 0.25}),
    ("33/33/34", {"tree": 0.33, "rocket": 0.33, "cnn": 0.34}),
]

best_lomo_map = 0
best_lomo_name = ""
best_lomo_weights = {}

for combo_name, weights in lomo_combos:
    oof_blend = (weights["tree"] * oof_tree_lomo +
                 weights["rocket"] * oof_rocket_lomo +
                 weights["cnn"] * oof_cnn_lomo)
    m_val, per = compute_map(y, oof_blend)
    print(f"    {combo_name}: LOMO={m_val:.4f}", flush=True)
    if m_val > best_lomo_map:
        best_lomo_map = m_val
        best_lomo_name = combo_name
        best_lomo_weights = weights

# Fine grid around best
print(f"\n  Best fixed: {best_lomo_name} (LOMO={best_lomo_map:.4f})", flush=True)
print("  Fine grid search around best...", flush=True)

best_fine_map = best_lomo_map
best_fine_w = best_lomo_weights.copy()
for w_tree in np.arange(0.3, 0.85, 0.05):
    for w_rocket in np.arange(0.0, 1.01 - w_tree, 0.05):
        w_cnn = 1.0 - w_tree - w_rocket
        if w_cnn < -0.01:
            continue
        w_cnn = max(w_cnn, 0)
        oof_blend = w_tree * oof_tree_lomo + w_rocket * oof_rocket_lomo + w_cnn * oof_cnn_lomo
        m_val, _ = compute_map(y, oof_blend)
        if m_val > best_fine_map:
            best_fine_map = m_val
            best_fine_w = {"tree": w_tree, "rocket": w_rocket, "cnn": w_cnn}

print(f"  Best LOMO weights: tree={best_fine_w['tree']:.2f}, "
      f"rocket={best_fine_w['rocket']:.2f}, cnn={best_fine_w['cnn']:.2f}", flush=True)
print(f"  Best LOMO mAP: {best_fine_map:.4f} (E32 LOMO: 0.3321)", flush=True)

# Per-class LOMO for best blend
oof_best_lomo = (best_fine_w["tree"] * oof_tree_lomo +
                 best_fine_w["rocket"] * oof_rocket_lomo +
                 best_fine_w["cnn"] * oof_cnn_lomo)
_, per_best_lomo = compute_map(y, oof_best_lomo)
print(f"\n  Per-class LOMO (best blend):", flush=True)
print(f"  {'Class':<15s} {'Blend':>7s} {'E32':>7s}", flush=True)
e32_lomo_per = {}
_, e32_lomo_per = compute_map(y, oof_tree_lomo)
for cls in CLASSES:
    b = per_best_lomo.get(cls, 0)
    e = e32_lomo_per.get(cls, 0)
    print(f"  {cls:<15s} {b:>7.4f} {e:>7.4f}", flush=True)

# ======================================================================
# PART 3: SKF stacking (retrain for test predictions)
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PART 3: SKF + TEST PREDICTIONS", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_tree_skf = np.zeros((len(y), N_CLASSES))
oof_cnn_skf = np.zeros((len(y), N_CLASSES))
oof_rocket_skf = np.zeros((len(y), N_CLASSES))

test_tree_skf = np.zeros((len(X_tab_test), N_CLASSES))
test_cnn_skf = np.zeros((len(X_seq_cnn_test), N_CLASSES))
test_rocket_skf = np.zeros((len(X_seq_rocket_test), N_CLASSES))

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n--- SKF Fold {fold_idx} ---", flush=True)

    # Tree
    oof_t, test_t = train_tree_fold(
        X_tab[tr_idx], y[tr_idx], X_tab[va_idx], y[va_idx],
        sample_weights[tr_idx], X_tab_test, fn_tab, f"F{fold_idx}")
    oof_tree_skf[va_idx] = oof_t
    test_tree_skf += test_t / 5

    # MiniRocket
    oof_r, test_r = train_rocket_fold(
        X_seq_rocket[tr_idx], y[tr_idx], X_seq_rocket[va_idx], y[va_idx],
        X_seq_rocket_test, f"F{fold_idx}")
    oof_rocket_skf[va_idx] = oof_r
    test_rocket_skf += test_r / 5

    # CNN
    oof_c, test_c = train_cnn_fold(
        X_seq_cnn[tr_idx], y[tr_idx], X_seq_cnn[va_idx], y[va_idx],
        X_seq_cnn_test, cnn_class_weights, f"F{fold_idx}")
    oof_cnn_skf[va_idx] = oof_c
    test_cnn_skf += test_c / 5

# SKF scores
print("\n  Individual SKF scores:", flush=True)
for name, oof in [("Tree", oof_tree_skf), ("Rocket", oof_rocket_skf), ("CNN", oof_cnn_skf)]:
    m_val, _ = compute_map(y, oof)
    print(f"    {name}: {m_val:.4f}", flush=True)

# Use LOMO-derived fixed weights for honest ensemble
print(f"\n  Using LOMO-best weights: tree={best_fine_w['tree']:.2f}, "
      f"rocket={best_fine_w['rocket']:.2f}, cnn={best_fine_w['cnn']:.2f}", flush=True)

oof_skf_blend = (best_fine_w["tree"] * oof_tree_skf +
                 best_fine_w["rocket"] * oof_rocket_skf +
                 best_fine_w["cnn"] * oof_cnn_skf)
test_skf_blend = (best_fine_w["tree"] * test_tree_skf +
                  best_fine_w["rocket"] * test_rocket_skf +
                  best_fine_w["cnn"] * test_cnn_skf)

skf_map, skf_per = compute_map(y, oof_skf_blend)
print(f"  SKF blend mAP: {skf_map:.4f}", flush=True)

# Also make a LOMO-test prediction (average of LOMO fold test preds)
test_lomo_blend = (best_fine_w["tree"] * test_tree_lomo +
                   best_fine_w["rocket"] * test_rocket_lomo +
                   best_fine_w["cnn"] * test_cnn_lomo)

# Test class distributions
print(f"\n  Test class distribution (SKF blend):", flush=True)
dist_skf = np.bincount(test_skf_blend.argmax(axis=1), minlength=N_CLASSES)
dist_lomo = np.bincount(test_lomo_blend.argmax(axis=1), minlength=N_CLASSES)
print(f"  {'Class':<15s} {'SKF':>6s} {'LOMO':>6s}", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<15s} {dist_skf[i]:>6d} {dist_lomo[i]:>6d}", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("E39 SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  LOMO blend mAP: {best_fine_map:.4f} (E32 LOMO: 0.3321, delta: {best_fine_map - 0.3321:+.4f})", flush=True)
print(f"  SKF blend mAP:  {skf_map:.4f} (E32 SKF: 0.6808, delta: {skf_map - 0.6808:+.4f})", flush=True)
print(f"  Weights: tree={best_fine_w['tree']:.2f}, rocket={best_fine_w['rocket']:.2f}, cnn={best_fine_w['cnn']:.2f}", flush=True)

# ======================================================================
# Save
# ======================================================================
# Save both SKF-based and LOMO-based test predictions
np.save(ROOT / "oof_e39_skf.npy", oof_skf_blend)
np.save(ROOT / "test_e39_skf.npy", test_skf_blend)
np.save(ROOT / "oof_e39_lomo.npy", oof_best_lomo)
np.save(ROOT / "test_e39_lomo.npy", test_lomo_blend)

# Submit SKF-based (trained on more data per fold) with LOMO weights
save_submission(test_skf_blend, "e39_stack_skf", cv_map=skf_map)
# Also save LOMO-based submission
save_submission(test_lomo_blend, "e39_stack_lomo", cv_map=best_fine_map)

print("\nDone!", flush=True)
