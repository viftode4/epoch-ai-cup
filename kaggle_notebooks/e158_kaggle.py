"""E158: 1D-CNN on raw padded sequences — Kaggle GPU notebook.

Setup: Add datasets:
  - ai-cup-2026 (competition)
  - epoch-src (private: src/, data/best_features.txt, data/*weather*, data/*solar*)
Enable GPU accelerator (T4/P100).

Uses prepare_sequences_v2(mode="pad") — raw variable-length sequences
with masked global pooling. NO resampling, NO interpolation.
"""

import sys, os, importlib
from pathlib import Path

# ── Kaggle paths ──────────────────────────────────────────────────────
COMP_DIR = Path("/kaggle/input/ai-cup-2026")
SRC_DIR = Path("/kaggle/input/epoch-src")

sys.path.insert(0, str(SRC_DIR))
import src.data as _data_mod
_data_mod.ROOT = SRC_DIR
_data_mod.DATA_DIR = COMP_DIR
importlib.reload(_data_mod)

# ── Imports ───────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from src.data import CLASSES, load_train, load_test
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
DROPOUT = 0.3

KEEP_FEATURES = [
    f.strip() for f in (SRC_DIR / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]


# ── Dataset ──────────────────────────────────────────────────────────
class RadarDataset(Dataset):
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
                x[ch, :L] = 0.0

            noise_scale = 0.05
            for ch in range(6):
                valid = x[ch, :L]
                std = valid.std()
                if std > 1e-8:
                    x[ch, :L] += torch.randn(L) * std * noise_scale

            max_crop = max(1, L // 10)
            crop_start = np.random.randint(0, max_crop + 1)
            crop_end = np.random.randint(0, max_crop + 1)
            if crop_start + crop_end > 0 and L - crop_start - crop_end > 5:
                new_L = L - crop_start - crop_end
                new_x = torch.zeros_like(x)
                new_x[:, :new_L] = x[:, crop_start:crop_start + new_L]
                x = new_x
                m = torch.zeros_like(m)
                m[:new_L] = True

        items = [x, m]
        if self.tabular is not None:
            items.append(self.tabular[idx])
        if self.labels is not None:
            items.append(self.labels[idx])
        return tuple(items)


# ── Model ────────────────────────────────────────────────────────────
class RadarCNN(nn.Module):
    def __init__(self, n_classes=9, n_tabular=0, dropout=0.3):
        super().__init__()
        self.n_tabular = n_tabular

        self.conv1 = nn.Conv1d(6, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        pool_dim = 256 * 2
        fc_in = pool_dim + n_tabular
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x, mask=None, tab=None):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        if mask is not None:
            m = mask.float()
            m = F.max_pool1d(m.unsqueeze(1), 2).squeeze(1)
            m = F.max_pool1d(m.unsqueeze(1), 2).squeeze(1)
            m = (m > 0.5)

            m_expanded = m.unsqueeze(1).expand_as(x)
            x_masked = x * m_expanded.float()
            n_valid = m.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(1)
            avg_pool = x_masked.sum(dim=2) / n_valid.squeeze(2)

            x_for_max = x.masked_fill(~m_expanded, float('-inf'))
            max_pool = x_for_max.max(dim=2).values
            max_pool = max_pool.clamp(min=0.0)
        else:
            avg_pool = x.mean(dim=2)
            max_pool = x.max(dim=2).values

        x = torch.cat([avg_pool, max_pool], dim=1)

        if tab is not None and self.n_tabular > 0:
            x = torch.cat([x, tab], dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────────
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


def normalize_sequences_masked(seq, mask, fit_seq=None, fit_mask=None):
    ref_seq = fit_seq if fit_seq is not None else seq
    ref_mask = fit_mask if fit_mask is not None else mask
    N, C, T = ref_seq.shape
    means = np.zeros(C, dtype=np.float64)
    stds = np.zeros(C, dtype=np.float64)
    for c in range(C):
        vals = ref_seq[:, c, :][ref_mask]
        if len(vals) > 0:
            means[c] = vals.mean()
            stds[c] = vals.std()
    stds = np.where(stds < 1e-8, 1.0, stds)
    out = seq.copy()
    for c in range(C):
        out[:, c, :] = (out[:, c, :] - means[c]) / stds[c]
    out *= mask[:, np.newaxis, :].astype(np.float32)
    return out.astype(np.float32)


def train_fold(model, train_loader, val_loader, class_weights, n_epochs,
               lr, wd, device, has_tabular=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_map = -1.0
    best_preds = None
    best_state = None
    patience_counter = 0
    PATIENCE = 20

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x_seq, x_mask, x_tab, labels = _unpack_batch(batch, has_tabular, device)
            logits = model(x_seq, mask=x_mask, tab=x_tab)
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
                logits = model(x_seq, mask=x_mask, tab=x_tab)
                all_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_map, _ = compute_map(val_labels, val_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={train_loss/n_batches:.4f}, "
                  f"val_mAP={val_map:.4f}, lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

        if val_map > best_map:
            best_map = val_map
            best_preds = val_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1} (best={best_map:.4f})", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_preds, best_map


def predict_test(model, test_loader, device, has_tabular=False):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x_seq, x_mask, x_tab, _ = _unpack_batch(batch, has_tabular, device)
            logits = model(x_seq, mask=x_mask, tab=x_tab)
            all_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def run_variant(name, seq_train, mask_train, lens_train,
                seq_test, mask_test, lens_test,
                y, class_weights, tabular_train=None, tabular_test=None):
    n_tabular = tabular_train.shape[1] if tabular_train is not None else 0
    has_tabular = n_tabular > 0

    print(f"\n{'='*60}\n  {name}\n  n_tabular={n_tabular}, device={DEVICE}\n{'='*60}", flush=True)

    N = len(y)
    oof_preds = np.zeros((N, N_CLASSES), dtype=np.float32)
    test_preds_sum = np.zeros((len(seq_test), N_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), y)):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---", flush=True)

        s_tr, s_va = seq_train[train_idx], seq_train[val_idx]
        m_tr, m_va = mask_train[train_idx], mask_train[val_idx]
        l_tr, l_va = lens_train[train_idx], lens_train[val_idx]

        s_tr_n = normalize_sequences_masked(s_tr, m_tr)
        s_va_n = normalize_sequences_masked(s_va, m_va, fit_seq=seq_train[train_idx], fit_mask=mask_train[train_idx])
        s_te_n = normalize_sequences_masked(seq_test, mask_test, fit_seq=seq_train[train_idx], fit_mask=mask_train[train_idx])

        tab_tr = tabular_train[train_idx] if has_tabular else None
        tab_va = tabular_train[val_idx] if has_tabular else None

        ds_train = RadarDataset(s_tr_n, m_tr, l_tr, y[train_idx], tab_tr, augment=True)
        ds_val = RadarDataset(s_va_n, m_va, l_va, y[val_idx], tab_va, augment=False)
        ds_test = RadarDataset(s_te_n, mask_test, lens_test, tabular=tabular_test, augment=False)

        ldr_tr = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        ldr_va = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        ldr_te = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        torch.manual_seed(SEED + fold_i)
        model = RadarCNN(n_classes=N_CLASSES, n_tabular=n_tabular, dropout=DROPOUT)

        fold_preds, fold_map = train_fold(model, ldr_tr, ldr_va, class_weights,
                                          N_EPOCHS, LR, WD, DEVICE, has_tabular)
        oof_preds[val_idx] = fold_preds
        print(f"  Fold {fold_i+1} mAP: {fold_map:.4f}", flush=True)

        test_preds_sum += predict_test(model, ldr_te, DEVICE, has_tabular)

    test_preds = test_preds_sum / N_FOLDS
    overall_map, per_class = compute_map(y, oof_preds)
    print_results(overall_map, per_class, label=name)
    return oof_preds, test_preds, overall_map, per_class


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70, flush=True)
    print("E158: 1D-CNN on v2 Raw Padded Sequences".center(70), flush=True)
    print("=" * 70, flush=True)

    print(f"\nDevice: {DEVICE}", flush=True)
    train_df = load_train()
    test_df = load_test()
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])
    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    class_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    class_weights = torch.from_numpy(len(y) / (N_CLASSES * class_counts))

    # ── Prepare sequences (RAW PAD) ──────────────────────────────────
    print("\nPreparing v2 sequences (pad mode = raw data)...", flush=True)
    train_seq, train_mask, train_lens = prepare_sequences_v2(train_df, mode="pad", max_len=MAX_LEN)
    test_seq, test_mask, test_lens = prepare_sequences_v2(test_df, mode="pad", max_len=MAX_LEN)
    print(f"  Train: {train_seq.shape}, lengths: {train_lens.min()}-{train_lens.max()} (med={int(np.median(train_lens))})", flush=True)
    print(f"  Test:  {test_seq.shape}, lengths: {test_lens.min()}-{test_lens.max()} (med={int(np.median(test_lens))})", flush=True)

    train_seq = np.nan_to_num(train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    test_seq = np.nan_to_num(test_seq, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Variant A: Pure CNN ──────────────────────────────────────────
    oof_a, test_a, map_a, pc_a = run_variant(
        "Variant A: Pure CNN (6ch, raw padded)",
        train_seq, train_mask, train_lens,
        test_seq, test_mask, test_lens,
        y, class_weights
    )

    # ── Build tabular features for Variant B ─────────────────────────
    print("\n\nBuilding tabular features...", flush=True)
    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]

    train_weather = pd.read_csv(SRC_DIR / "data" / "train_weather.csv")
    test_weather = pd.read_csv(SRC_DIR / "data" / "test_weather.csv")
    for col in train_weather.columns:
        train_feats[f"wx_{col}"] = train_weather[col].values
        test_feats[f"wx_{col}"] = test_weather[col].values

    train_solar = pd.read_csv(SRC_DIR / "data" / "train_solar.csv")
    test_solar = pd.read_csv(SRC_DIR / "data" / "test_solar.csv")
    for col in train_solar.columns:
        train_feats[f"sol_{col}"] = train_solar[col].values
        test_feats[f"sol_{col}"] = test_solar[col].values

    available = [f for f in KEEP_FEATURES if f in train_feats.columns]
    print(f"  Using {len(available)}/{len(KEEP_FEATURES)} tabular features", flush=True)

    train_tab = train_feats[available].values.astype(np.float32)
    test_tab = test_feats[available].values.astype(np.float32)
    train_tab = np.nan_to_num(train_tab, nan=0.0, posinf=0.0, neginf=0.0)
    test_tab = np.nan_to_num(test_tab, nan=0.0, posinf=0.0, neginf=0.0)

    tab_mean = train_tab.mean(axis=0, keepdims=True)
    tab_std = train_tab.std(axis=0, keepdims=True)
    tab_std = np.where(tab_std < 1e-8, 1.0, tab_std)
    train_tab = ((train_tab - tab_mean) / tab_std).astype(np.float32)
    test_tab = ((test_tab - tab_mean) / tab_std).astype(np.float32)

    # ── Variant B: CNN + Tabular ─────────────────────────────────────
    oof_b, test_b, map_b, pc_b = run_variant(
        f"Variant B: CNN + {len(available)} Tabular",
        train_seq, train_mask, train_lens,
        test_seq, test_mask, test_lens,
        y, class_weights,
        tabular_train=train_tab, tabular_test=test_tab
    )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*60}\n  E158 SUMMARY\n{'='*60}", flush=True)
    print(f"  Variant A (Pure CNN):      mAP = {map_a:.4f}", flush=True)
    print(f"  Variant B (CNN + Tab):     mAP = {map_b:.4f}", flush=True)
    for cls in CLASSES:
        a_val = pc_a.get(cls, 0)
        b_val = pc_b.get(cls, 0)
        diff = b_val - a_val
        print(f"    {cls:15s}: {a_val:.4f} / {b_val:.4f}  ({'+' if diff >= 0 else ''}{diff:.4f})", flush=True)

    best_name = "A" if map_a >= map_b else "B"
    best_map = max(map_a, map_b)
    best_oof = oof_a if map_a >= map_b else oof_b
    best_test = test_a if map_a >= map_b else test_b

    # ── Save submission ──────────────────────────────────────────────
    sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
    sub_columns = [c for c in sample_sub.columns if c != "track_id"]
    sub = pd.DataFrame({"track_id": test_df["track_id"]})
    for col in sub_columns:
        cls_idx = CLASSES.index(col)
        sub[col] = best_test[:, cls_idx]
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"\nSaved submission.csv (variant {best_name}, {len(sub)} rows)", flush=True)

    np.save("/kaggle/working/oof_e158.npy", best_oof)
    np.save("/kaggle/working/test_e158.npy", best_test)
    np.save("/kaggle/working/oof_e158_a.npy", oof_a)
    np.save("/kaggle/working/test_e158_a.npy", test_a)
    np.save("/kaggle/working/oof_e158_b.npy", oof_b)
    np.save("/kaggle/working/test_e158_b.npy", test_b)
    print("Done!", flush=True)
