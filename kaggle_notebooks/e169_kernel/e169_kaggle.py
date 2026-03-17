"""E169: DeepSets Attention-Pooling + Tree Ensemble Blend -- Kaggle GPU notebook.

Setup: Add datasets:
  - ai-cup-2026-performance (competition)
  - epoch-src (private: src/, data/best_features.txt, data/*weather*, data/*solar*)
Enable GPU accelerator. PyTorch is pre-installed on Kaggle GPU.

Architecture:
  DeepSets (~10K params): per-point MLP(6->32->32) + 3-head pooling (attn+mean+max)
  + 25 physics tabular features -> Linear(121,64) -> Linear(64,9)
  Blended with LGB+XGB+CB tree ensemble on 36+8 features.
"""

import sys
import os
from pathlib import Path

# -- Kaggle paths (dynamic discovery like E158) ---------------------------
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

SRC_DIR = SRC_PARENT
print(f"src/ at {SRC_PARENT / 'src'}", flush=True)
print(f"ext data at {EXT_DATA_DIR}", flush=True)
print(f"comp data at {COMP_DIR}", flush=True)
sys.path.insert(0, str(SRC_PARENT))

import importlib
import src.data as _data_mod
_data_mod.ROOT = SRC_PARENT
_data_mod.DATA_DIR = COMP_DIR

from src.data import CLASSES, load_train, load_test
print(f"Classes: {CLASSES}")
train_check = load_train()
print(f"Train shape: {train_check.shape}")

# -- Imports -----------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.sequence import prepare_sequences_v2

N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 200
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
PATIENCE = 15
N_SEEDS = 3

# 36 validated features
KEEP_FEATURES = [
    f.strip() for f in (EXT_DATA_DIR / "best_features.txt").read_text().splitlines()
    if f.strip()
]

WX_SOL_FEATURES = [
    "wx_wind_speed", "wx_wind_gust", "wx_wind_u", "wx_wind_v",
    "wx_temp_c", "wx_dewpoint_c", "wx_humidity",
    "sol_solar_elevation", "sol_daylight_hours",
    "sol_hours_since_sunrise", "sol_daylight_fraction",
]

PHYSICS_FEATURES = [f for f in KEEP_FEATURES if f not in WX_SOL_FEATURES]

TD_FEATURES = [
    "td_heading_local_var", "td_speed_consistency", "td_speed_autocorr",
    "td_speed_slope", "td_alt_smoothness", "td_heading_change_rate",
    "td_rcs_trend", "td_speed_variability",
]


# -- DeepSets Dataset --------------------------------------------------
class DeepSetsDataset(Dataset):
    def __init__(self, sequences, masks, lengths, labels=None, tabular=None, augment=False):
        self.sequences = torch.from_numpy(sequences).float()
        self.masks = torch.from_numpy(masks).bool()
        self.lengths = torch.from_numpy(lengths).long()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.tabular = torch.from_numpy(tabular).float() if tabular is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx].clone()
        m = self.masks[idx].clone()
        L = self.lengths[idx].item()

        if self.augment and L > 5:
            if np.random.random() < 0.15:
                ch = np.random.randint(0, 6)
                x[:L, ch] = 0.0

            noise_scale = 0.05
            for ch in range(6):
                valid = x[:L, ch]
                std = valid.std()
                if std > 1e-8:
                    x[:L, ch] += torch.randn(L) * std * noise_scale

            max_crop = max(1, L // 10)
            crop_start = np.random.randint(0, max_crop + 1)
            crop_end = np.random.randint(0, max_crop + 1)
            if crop_start + crop_end > 0 and L - crop_start - crop_end > 5:
                new_L = L - crop_start - crop_end
                new_x = torch.zeros_like(x)
                new_x[:new_L] = x[crop_start:crop_start + new_L]
                x = new_x
                m = torch.zeros_like(m)
                m[:new_L] = True

            if np.random.random() < 0.5:
                x_valid = x[:L].clone()
                x_valid = x_valid.flip(0)
                x_valid[:, 3] = -x_valid[:, 3]
                x[:L] = x_valid

        items = [x, m]
        if self.tabular is not None:
            items.append(self.tabular[idx])
        if self.labels is not None:
            items.append(self.labels[idx])
        return tuple(items)


# -- DeepSets Model ----------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask):
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(~mask, 0.0)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class DeepSetsClassifier(nn.Module):
    def __init__(self, in_channels=6, hidden=32, n_tabular=0, n_classes=9,
                 dropout_enc=0.2, dropout_head=0.3):
        super().__init__()
        self.n_tabular = n_tabular
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout_enc),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout_enc),
        )
        self.attn_pool = AttentionPool(hidden)
        pool_dim = hidden * 3
        head_in = pool_dim + n_tabular
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout_head),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, mask, tab=None):
        h = self.encoder(x)
        attn_out = self.attn_pool(h, mask)
        mask_exp = mask.unsqueeze(-1).float()
        h_masked = h * mask_exp
        n_valid = mask_exp.sum(dim=1).clamp(min=1)
        mean_out = h_masked.sum(dim=1) / n_valid
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_out = h_for_max.max(dim=1).values.clamp(min=0.0)
        pooled = torch.cat([attn_out, mean_out, max_out], dim=1)
        if tab is not None and self.n_tabular > 0:
            pooled = torch.cat([pooled, tab], dim=1)
        return self.head(pooled)


# -- Helpers -----------------------------------------------------------
def normalize_sequences_transposed(seq, mask, fit_seq=None, fit_mask=None):
    ref_seq = fit_seq if fit_seq is not None else seq
    ref_mask = fit_mask if fit_mask is not None else mask
    N, T, C = ref_seq.shape
    means = np.zeros(C, dtype=np.float64)
    stds = np.zeros(C, dtype=np.float64)
    for c in range(C):
        vals = ref_seq[:, :, c][ref_mask]
        if len(vals) > 0:
            means[c] = vals.mean()
            stds[c] = vals.std()
    stds = np.where(stds < 1e-8, 1.0, stds)
    out = seq.copy()
    for c in range(C):
        out[:, :, c] = (out[:, :, c] - means[c]) / stds[c]
    out *= mask[:, :, np.newaxis].astype(np.float32)
    return out.astype(np.float32)


def _unpack_batch(batch, has_tabular, device):
    idx = 0
    x_seq = batch[idx].to(device); idx += 1
    x_mask = batch[idx].to(device); idx += 1
    x_tab = None
    if has_tabular:
        x_tab = batch[idx].to(device); idx += 1
    labels = batch[idx] if idx < len(batch) else None
    if labels is not None:
        labels = labels.to(device)
    return x_seq, x_mask, x_tab, labels


def train_deepsets_fold(model, train_loader, val_loader, class_weights,
                        n_epochs, lr, wd, device, has_tabular=False, patience=15):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_map, best_preds, best_state = -1.0, None, None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            x_seq, x_mask, x_tab, labels = _unpack_batch(batch, has_tabular, device)
            logits = model(x_seq, x_mask, tab=x_tab)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_seq, x_mask, x_tab, labels = _unpack_batch(batch, has_tabular, device)
                logits = model(x_seq, x_mask, tab=x_tab)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_map, _ = compute_map(val_labels, val_preds)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{n_epochs}: loss={train_loss/n_batches:.4f}, "
                  f"val_mAP={val_map:.4f}", flush=True)

        if val_map > best_map:
            best_map = val_map
            best_preds = val_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop ep {epoch+1} (best={best_map:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_preds, best_map


def predict_deepsets_test(model, test_loader, device, has_tabular=False):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x_seq, x_mask, x_tab, _ = _unpack_batch(batch, has_tabular, device)
            logits = model(x_seq, x_mask, tab=x_tab)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
    return np.concatenate(all_preds)


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


# ======================================================================
# MAIN
# ======================================================================
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70, flush=True)
print("E169: DeepSets + Tree Blend (Kaggle)".center(70), flush=True)
print("=" * 70, flush=True)

# -- Load data ---------------------------------------------------------
print("\nLoading data...", flush=True)
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Device: {DEVICE}", flush=True)

# -- Prepare sequences -------------------------------------------------
print("\nPreparing sequences (pad mode)...", flush=True)
train_seq, train_mask, train_lens = prepare_sequences_v2(train_df, mode="pad", max_len=MAX_LEN)
test_seq, test_mask, test_lens = prepare_sequences_v2(test_df, mode="pad", max_len=MAX_LEN)
# v2 returns (N, 6, T) -> transpose to (N, T, 6) for DeepSets
train_seq = np.transpose(train_seq, (0, 2, 1))
test_seq = np.transpose(test_seq, (0, 2, 1))
train_seq = np.nan_to_num(train_seq, nan=0.0, posinf=0.0, neginf=0.0)
test_seq = np.nan_to_num(test_seq, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Train: {train_seq.shape}, Test: {test_seq.shape}", flush=True)

# -- Build tabular features --------------------------------------------
print("\nBuilding tabular features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode",
             "weakclass", "temporal_dynamics"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# Add weather + solar from epoch-src
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

# Full 36+8 for trees
tree_features = KEEP_FEATURES + TD_FEATURES
tree_available = [f for f in tree_features if f in train_feats.columns]
print(f"  Tree features: {len(tree_available)}", flush=True)
X_tree = train_feats[tree_available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
X_tree_test = test_feats[tree_available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Physics-only for neural (25 features)
physics_available = [f for f in PHYSICS_FEATURES if f in train_feats.columns]
print(f"  Neural physics features: {len(physics_available)}", flush=True)
tab_neural_train = train_feats[physics_available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
tab_neural_test = test_feats[physics_available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

# Class weights
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
beta = 0.999
eff_n = (1.0 - beta ** counts) / (1.0 - beta)
class_weights_arr = 1.0 / np.maximum(eff_n, 1e-6)
class_weights_arr /= class_weights_arr.sum() / N_CLASSES
sample_weights = class_weights_arr[y]
class_weights_torch = torch.from_numpy((len(y) / (N_CLASSES * counts)).astype(np.float32))

# ======================================================================
# PART 1: DeepSets (5-fold x 3 seeds)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("PART 1: DeepSets Training".center(70), flush=True)
print("=" * 70, flush=True)

n_tab = len(physics_available)
has_tab = n_tab > 0
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_ds = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_ds = np.zeros((len(test_seq), N_CLASSES), dtype=np.float64)

dummy_model = DeepSetsClassifier(in_channels=6, hidden=32, n_tabular=n_tab, n_classes=N_CLASSES)
n_params = sum(p.numel() for p in dummy_model.parameters())
print(f"  Model params: {n_params:,}", flush=True)
del dummy_model

for fold_i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n--- Fold {fold_i+1}/{N_FOLDS} (train={len(train_idx)}, val={len(val_idx)}) ---", flush=True)

    raw_train_fold = train_seq[train_idx]
    raw_mask_fold = train_mask[train_idx]
    seq_tr = normalize_sequences_transposed(train_seq[train_idx], train_mask[train_idx])
    seq_va = normalize_sequences_transposed(train_seq[val_idx], train_mask[val_idx],
                                            fit_seq=raw_train_fold, fit_mask=raw_mask_fold)
    seq_te = normalize_sequences_transposed(test_seq, test_mask,
                                            fit_seq=raw_train_fold, fit_mask=raw_mask_fold)

    tab_mean = tab_neural_train[train_idx].mean(axis=0, keepdims=True)
    tab_std = tab_neural_train[train_idx].std(axis=0, keepdims=True)
    tab_std = np.where(tab_std < 1e-8, 1.0, tab_std)
    tab_tr = ((tab_neural_train[train_idx] - tab_mean) / tab_std).astype(np.float32)
    tab_va = ((tab_neural_train[val_idx] - tab_mean) / tab_std).astype(np.float32)
    tab_te = ((tab_neural_test - tab_mean) / tab_std).astype(np.float32)

    y_tr, y_va = y[train_idx], y[val_idx]

    fold_oof_sum = np.zeros((len(val_idx), N_CLASSES), dtype=np.float64)
    fold_test_sum = np.zeros((len(test_seq), N_CLASSES), dtype=np.float64)
    fold_maps = []

    for seed_i in range(N_SEEDS):
        seed_val = SEED + fold_i * 100 + seed_i
        torch.manual_seed(seed_val)
        print(f"  Seed {seed_i+1}/{N_SEEDS} (seed={seed_val})", flush=True)

        ds_train = DeepSetsDataset(seq_tr, train_mask[train_idx], train_lens[train_idx],
                                   y_tr, tab_tr, augment=True)
        ds_val = DeepSetsDataset(seq_va, train_mask[val_idx], train_lens[val_idx],
                                 y_va, tab_va, augment=False)
        ds_test = DeepSetsDataset(seq_te, test_mask, test_lens,
                                  tabular=tab_te, augment=False)

        loader_tr = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
        loader_va = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)
        loader_te = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)

        model = DeepSetsClassifier(in_channels=6, hidden=32, n_tabular=n_tab,
                                   n_classes=N_CLASSES, dropout_enc=0.2, dropout_head=0.3)

        val_preds, val_map = train_deepsets_fold(
            model, loader_tr, loader_va, class_weights_torch,
            N_EPOCHS, LR, WD, DEVICE, has_tabular=has_tab, patience=PATIENCE
        )
        fold_oof_sum += val_preds
        fold_maps.append(val_map)
        print(f"    Seed {seed_i+1} mAP: {val_map:.4f}", flush=True)

        test_preds = predict_deepsets_test(model, loader_te, DEVICE, has_tabular=has_tab)
        fold_test_sum += test_preds

    oof_ds[val_idx] = fold_oof_sum / N_SEEDS
    test_ds += fold_test_sum / (N_SEEDS * N_FOLDS)

    avg_map = np.mean(fold_maps)
    print(f"  Fold {fold_i+1} avg mAP: {avg_map:.4f}", flush=True)

ds_map, ds_per = compute_map(y, oof_ds)
print_results(ds_map, ds_per, "DeepSets Standalone (SKF)")

# ======================================================================
# PART 2: Tree Ensemble (LGB + XGB + CB, no Optuna)
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("PART 2: Tree Ensemble Training".center(70), flush=True)
print("=" * 70, flush=True)

oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float64)
test_lgb = np.zeros((len(X_tree_test), N_CLASSES), dtype=np.float64)
test_xgb = np.zeros((len(X_tree_test), N_CLASSES), dtype=np.float64)
test_cb = np.zeros((len(X_tree_test), N_CLASSES), dtype=np.float64)

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_tree, y)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    lgb = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        n_jobs=-1,
    )
    lgb.fit(X_tree[tr_idx], y[tr_idx], eval_set=[(X_tree[va_idx], y[va_idx])])
    oof_lgb[va_idx] = lgb.predict_proba(X_tree[va_idx])
    test_lgb += lgb.predict_proba(X_tree_test) / N_FOLDS

    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", random_state=SEED, verbosity=0,
        device="cuda", tree_method="hist",
    )
    xgb.fit(X_tree[tr_idx], y[tr_idx], eval_set=[(X_tree[va_idx], y[va_idx])],
            sample_weight=sample_weights[tr_idx], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_tree[va_idx])
    test_xgb += xgb.predict_proba(X_tree_test) / N_FOLDS

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, bagging_temperature=0.5,
        random_strength=1.0, border_count=128,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type="GPU",
    )
    cb.fit(X_tree[tr_idx], y[tr_idx], eval_set=(X_tree[va_idx], y[va_idx]), verbose=0)
    oof_cb[va_idx] = cb.predict_proba(X_tree[va_idx])
    test_cb += cb.predict_proba(X_tree_test) / N_FOLDS

for name, oof in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CB", oof_cb)]:
    m, _ = compute_map(y, oof)
    print(f"  {name} SKF mAP: {m:.4f}", flush=True)

# Ensemble weight optimization
print("\n--- Tree ensemble weight optimization ---", flush=True)
best_w, best_tree_map = None, 0
for w_lgb in np.arange(0.0, 1.05, 0.1):
    for w_xgb in np.arange(0.0, 1.05 - w_lgb, 0.1):
        w_cb = 1.0 - w_lgb - w_xgb
        if w_cb < -0.01:
            continue
        oof_blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb
        m, _ = compute_map(y, oof_blend)
        if m > best_tree_map:
            best_tree_map = m
            best_w = (round(w_lgb, 2), round(w_xgb, 2), round(w_cb, 2))
print(f"  Best weights: LGB={best_w[0]}, XGB={best_w[1]}, CB={best_w[2]}", flush=True)
print(f"  Best tree mAP: {best_tree_map:.4f}", flush=True)

oof_trees = best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cb
test_trees = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cb

tree_map, tree_per = compute_map(y, oof_trees)
print_results(tree_map, tree_per, "Tree Ensemble (SKF)")

# ======================================================================
# PART 3: Blend Optimization
# ======================================================================
print("\n" + "=" * 70, flush=True)
print("PART 3: DeepSets + Tree Blend".center(70), flush=True)
print("=" * 70, flush=True)

alpha_grid = np.arange(0.0, 0.35, 0.05)
best_alpha, best_blend_map = 0.0, 0.0
print(f"\n  Alpha sweep (neural weight):", flush=True)
for alpha in alpha_grid:
    oof_blend = (1.0 - alpha) * oof_trees + alpha * oof_ds
    m, per = compute_map(y, oof_blend)
    marker = " <-- best" if m > best_blend_map else ""
    print(f"    alpha={alpha:.2f}: mAP={m:.4f}{marker}", flush=True)
    if m > best_blend_map:
        best_blend_map = m
        best_alpha = alpha

print(f"\n  Best alpha: {best_alpha:.2f}", flush=True)
print(f"  Best blend mAP: {best_blend_map:.4f}", flush=True)
print(f"  Delta vs trees: {best_blend_map - tree_map:+.4f}", flush=True)

oof_final = (1.0 - best_alpha) * oof_trees + best_alpha * oof_ds
test_final = (1.0 - best_alpha) * test_trees + best_alpha * test_ds

final_map, final_per = compute_map(y, oof_final)
print_results(final_map, final_per, "E169 FINAL BLEND (SKF)")

# Per-class impact
print("\n  Per-class blend impact (trees -> blend):", flush=True)
for cls in CLASSES:
    t_val = tree_per[cls]
    b_val = final_per[cls]
    d_val = ds_per.get(cls, 0)
    diff = b_val - t_val
    arrow = "+" if diff >= 0 else ""
    print(f"    {cls:15s}: tree={t_val:.4f}  ds={d_val:.4f}  blend={b_val:.4f}  ({arrow}{diff:.4f})", flush=True)

# ======================================================================
# PART 4: Save submission
# ======================================================================
sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
sub = sample_sub.copy()
prob_cols = [c for c in sub.columns if c != "track_id"]

for i, cls in enumerate(CLASSES):
    if cls in prob_cols:
        sub[cls] = test_final[:, i]

sub.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv ({len(sub)} rows)", flush=True)

# Also save trees-only submission for comparison
sub_trees = sample_sub.copy()
for i, cls in enumerate(CLASSES):
    if cls in prob_cols:
        sub_trees[cls] = test_trees[:, i]
sub_trees.to_csv("submission_trees_only.csv", index=False)

print(f"\n{'='*70}", flush=True)
print("E169 SUMMARY".center(70), flush=True)
print(f"{'='*70}", flush=True)
print(f"  DeepSets standalone: {ds_map:.4f}", flush=True)
print(f"  Tree ensemble:      {tree_map:.4f}", flush=True)
print(f"  Best blend (a={best_alpha:.2f}):  {best_blend_map:.4f}", flush=True)
print(f"  Delta:              {best_blend_map - tree_map:+.4f}", flush=True)
print(f"\nDone.", flush=True)
