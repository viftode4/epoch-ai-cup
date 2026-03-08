"""E158: 1D-CNN on v2 preprocessed radar sequences (RAW PAD MODE).

Two variants:
  A) Pure CNN on 6-channel sequences (N, 6, max_len) with mask
  B) CNN + 36 tabular features concatenated after masked global pooling

Uses prepare_sequences_v2(mode="pad") — raw variable-length sequences,
zero-padded with a boolean mask. NO resampling, NO interpolation.
The model uses masked pooling to ignore padding.

Architecture:
  Conv1d(6,64,k=7) -> BN -> ReLU -> MaxPool(2)
  Conv1d(64,128,k=5) -> BN -> ReLU -> MaxPool(2)
  Conv1d(128,256,k=3) -> BN -> ReLU
  MaskedGlobalAvgPool + MaskedGlobalMaxPool -> 512
  [+ optional 36 tabular features -> 548]
  Dropout(0.3) -> Linear -> 9
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_train, load_test
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.sequence import prepare_sequences_v2

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams
MAX_LEN = 200
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
DROPOUT = 0.3


# ── Dataset ──────────────────────────────────────────────────────────
class RadarDataset(Dataset):
    def __init__(self, sequences, masks, lengths, labels=None, tabular=None, augment=False):
        self.sequences = torch.from_numpy(sequences).float()  # (N, 6, max_len)
        self.masks = torch.from_numpy(masks).bool()            # (N, max_len)
        self.lengths = torch.from_numpy(lengths).long()        # (N,)
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.tabular = torch.from_numpy(tabular).float() if tabular is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx].clone()    # (6, max_len)
        m = self.masks[idx].clone()         # (max_len,)
        L = self.lengths[idx].item()

        if self.augment and L > 5:
            # Channel dropout: zero out 1 channel with p=0.15 (only valid region)
            if np.random.random() < 0.15:
                ch = np.random.randint(0, 6)
                x[ch, :L] = 0.0

            # Gaussian noise on valid region only
            noise_scale = 0.05
            for ch in range(6):
                valid = x[ch, :L]
                std = valid.std()
                if std > 1e-8:
                    x[ch, :L] += torch.randn(L) * std * noise_scale

            # Random crop: drop up to 10% of timesteps from start or end
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

        pool_dim = 256 * 2  # avg + max pool concatenated
        fc_in = pool_dim + n_tabular

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x, mask=None, tab=None):
        # x: (B, 6, T), mask: (B, T) — True for valid positions
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B, 64, T//2)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))   # (B, 128, T//4)
        x = F.relu(self.bn3(self.conv3(x)))                # (B, 256, T//4)

        # Downsample mask to match conv output (2 MaxPool(2) = /4)
        if mask is not None:
            # Pool mask: a position is valid if ANY of the 2 inputs were valid
            m = mask.float()
            m = F.max_pool1d(m.unsqueeze(1), 2).squeeze(1)  # T -> T//2
            m = F.max_pool1d(m.unsqueeze(1), 2).squeeze(1)  # T//2 -> T//4
            m = (m > 0.5)  # (B, T//4) bool

            # Masked global average pool
            m_expanded = m.unsqueeze(1).expand_as(x)  # (B, 256, T//4)
            x_masked = x * m_expanded.float()
            n_valid = m.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(1)  # (B, 1, 1)
            avg_pool = x_masked.sum(dim=2) / n_valid.squeeze(2)  # (B, 256)

            # Masked global max pool (set padding to -inf before max)
            x_for_max = x.masked_fill(~m_expanded, float('-inf'))
            max_pool = x_for_max.max(dim=2).values  # (B, 256)
            # Safety: if all masked, replace -inf with 0
            max_pool = max_pool.clamp(min=0.0)
        else:
            avg_pool = x.mean(dim=2)
            max_pool = x.max(dim=2).values

        x = torch.cat([avg_pool, max_pool], dim=1)  # (B, 512)

        if tab is not None and self.n_tabular > 0:
            x = torch.cat([x, tab], dim=1)  # (B, 512 + n_tabular)

        x = self.dropout(x)
        x = self.fc(x)
        return x


# ── Training loop ────────────────────────────────────────────────────
def _unpack_batch(batch, has_tabular, device):
    """Unpack batch: (seq, mask, [tab,] [label])."""
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


def train_fold(model, train_loader, val_loader, class_weights, n_epochs,
               lr, wd, device, has_tabular=False):
    """Train one fold, return best val predictions and best val mAP."""
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
        # Train
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

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x_seq, x_mask, x_tab, labels = _unpack_batch(batch, has_tabular, device)
                logits = model(x_seq, mask=x_mask, tab=x_tab)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_map, _ = compute_map(val_labels, val_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"train_loss={train_loss/n_batches:.4f}, "
                  f"val_mAP={val_map:.4f}, lr={scheduler.get_last_lr()[0]:.6f}",
                  flush=True)

        if val_map > best_map:
            best_map = val_map
            best_preds = val_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best mAP={best_map:.4f})",
                      flush=True)
                break

    # Restore best model for test prediction
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_preds, best_map


def predict_test(model, test_loader, device, has_tabular=False):
    """Generate test predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x_seq, x_mask, x_tab, _ = _unpack_batch(batch, has_tabular, device)
            logits = model(x_seq, mask=x_mask, tab=x_tab)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
    return np.concatenate(all_preds)


# ── Per-channel normalization (mask-aware) ───────────────────────────
def normalize_sequences_masked(seq, mask, fit_seq=None, fit_mask=None):
    """Per-channel normalization using only valid (unpadded) positions.

    If fit_seq/fit_mask provided, compute stats from those (for test normalization).
    Returns normalized copy; padding positions remain 0.
    """
    ref_seq = fit_seq if fit_seq is not None else seq
    ref_mask = fit_mask if fit_mask is not None else mask
    # ref_seq: (N, C, T), ref_mask: (N, T)
    N, C, T = ref_seq.shape
    means = np.zeros(C, dtype=np.float64)
    stds = np.zeros(C, dtype=np.float64)
    for c in range(C):
        vals = ref_seq[:, c, :][ref_mask]  # all valid values for channel c
        if len(vals) > 0:
            means[c] = vals.mean()
            stds[c] = vals.std()
    stds = np.where(stds < 1e-8, 1.0, stds)

    out = seq.copy()
    for c in range(C):
        out[:, c, :] = (out[:, c, :] - means[c]) / stds[c]
    # Zero out padding positions
    out *= mask[:, np.newaxis, :].astype(np.float32)
    return out.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────
def run_variant(name, sequences_train, masks_train, lens_train,
                sequences_test, masks_test, lens_test,
                y, class_weights,
                tabular_train=None, tabular_test=None):
    """Run 5-fold CV for one variant. Returns oof_preds, test_preds, mAP."""
    n_tabular = tabular_train.shape[1] if tabular_train is not None else 0
    has_tabular = n_tabular > 0

    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  n_tabular={n_tabular}, device={DEVICE}", flush=True)
    print(f"{'='*60}", flush=True)

    N = len(y)
    oof_preds = np.zeros((N, N_CLASSES), dtype=np.float32)
    test_preds_sum = np.zeros((len(sequences_test), N_CLASSES), dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), y)):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---", flush=True)

        # Split sequences + masks
        seq_tr = sequences_train[train_idx]
        seq_va = sequences_train[val_idx]
        mask_tr = masks_train[train_idx]
        mask_va = masks_train[val_idx]
        lens_tr = lens_train[train_idx]
        lens_va = lens_train[val_idx]

        # Mask-aware per-channel normalization (fit on fold train)
        seq_tr = normalize_sequences_masked(seq_tr, mask_tr)
        seq_va = normalize_sequences_masked(seq_va, mask_va,
                                            fit_seq=sequences_train[train_idx],
                                            fit_mask=masks_train[train_idx])
        seq_te = normalize_sequences_masked(sequences_test, masks_test,
                                            fit_seq=sequences_train[train_idx],
                                            fit_mask=masks_train[train_idx])

        y_tr, y_va = y[train_idx], y[val_idx]

        # Optional tabular
        tab_tr = tabular_train[train_idx] if has_tabular else None
        tab_va = tabular_train[val_idx] if has_tabular else None
        tab_te = tabular_test if has_tabular else None

        # Datasets (pass masks + lengths)
        ds_train = RadarDataset(seq_tr, mask_tr, lens_tr, y_tr, tab_tr, augment=True)
        ds_val = RadarDataset(seq_va, mask_va, lens_va, y_va, tab_va, augment=False)
        ds_test = RadarDataset(seq_te, masks_test, lens_test, tabular=tab_te, augment=False)

        loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
        loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)
        loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Model
        torch.manual_seed(SEED + fold_i)
        model = RadarCNN(n_classes=N_CLASSES, n_tabular=n_tabular, dropout=DROPOUT)

        # Train
        fold_preds, fold_map = train_fold(
            model, loader_train, loader_val, class_weights,
            N_EPOCHS, LR, WD, DEVICE, has_tabular=has_tabular
        )
        oof_preds[val_idx] = fold_preds
        print(f"  Fold {fold_i+1} mAP: {fold_map:.4f}", flush=True)

        # Test predictions (using best model state from training)
        test_fold = predict_test(model, loader_test, DEVICE, has_tabular=has_tabular)
        test_preds_sum += test_fold

    test_preds = test_preds_sum / N_FOLDS

    # Overall OOF metrics
    overall_map, per_class = compute_map(y, oof_preds)
    print_results(overall_map, per_class, label=name)

    return oof_preds, test_preds, overall_map, per_class


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70, flush=True)
    print("E158: 1D-CNN on v2 Sequences".center(70), flush=True)
    print("=" * 70, flush=True)

    # -- Load data -------------------------------------------------------
    print("\nLoading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Device: {DEVICE}", flush=True)

    # -- Class weights (inverse frequency) --------------------------------
    class_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    class_weights = torch.from_numpy(len(y) / (N_CLASSES * class_counts))
    print(f"  Class weights: {class_weights.numpy().round(2)}", flush=True)

    # -- Prepare sequences (RAW PAD MODE) --------------------------------
    print("\nPreparing v2 sequences (pad mode = raw data)...", flush=True)
    train_seq, train_mask, train_lens = prepare_sequences_v2(
        train_df, mode="pad", max_len=MAX_LEN
    )
    test_seq, test_mask, test_lens = prepare_sequences_v2(
        test_df, mode="pad", max_len=MAX_LEN
    )
    print(f"  Train sequences: {train_seq.shape}", flush=True)
    print(f"  Test sequences:  {test_seq.shape}", flush=True)
    print(f"  Train lengths: min={train_lens.min()}, max={train_lens.max()}, "
          f"median={int(np.median(train_lens))}", flush=True)
    print(f"  Test lengths:  min={test_lens.min()}, max={test_lens.max()}, "
          f"median={int(np.median(test_lens))}", flush=True)

    # Handle inf/nan in sequences (preserve mask)
    train_seq = np.nan_to_num(train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    test_seq = np.nan_to_num(test_seq, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Variant A: Pure CNN ──────────────────────────────────────────
    oof_a, test_a, map_a, pc_a = run_variant(
        "Variant A: Pure CNN (6ch, raw padded)",
        train_seq, train_mask, train_lens,
        test_seq, test_mask, test_lens,
        y, class_weights
    )

    # -- Build tabular features for variant B ----------------------------
    print("\n\nBuilding tabular features for Variant B...", flush=True)
    best_features = (ROOT / "data" / "best_features.txt").read_text().strip().split("\n")
    print(f"  Using {len(best_features)} tabular features", flush=True)

    feat_sets = [
        "core", "rcs_fft", "tabular", "targeted",
        "flight_mode", "weakclass", "flight_physics",
        "enhanced_bio_shape", "radar_physics",
    ]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    # Remove temporal
    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]

    # Add weather + solar (with wx_/sol_ prefix, same as E79)
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

    # Select only the 36 best features
    available = [f for f in best_features if f in train_feats.columns]
    missing = [f for f in best_features if f not in train_feats.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}", flush=True)
    print(f"  Available: {len(available)}/{len(best_features)} features", flush=True)

    train_tab = train_feats[available].values.astype(np.float32)
    test_tab = test_feats[available].values.astype(np.float32)

    # Handle inf/nan
    train_tab = np.nan_to_num(train_tab, nan=0.0, posinf=0.0, neginf=0.0)
    test_tab = np.nan_to_num(test_tab, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize tabular features (z-score on train)
    tab_mean = train_tab.mean(axis=0, keepdims=True)
    tab_std = train_tab.std(axis=0, keepdims=True)
    tab_std = np.where(tab_std < 1e-8, 1.0, tab_std)
    train_tab = ((train_tab - tab_mean) / tab_std).astype(np.float32)
    test_tab = ((test_tab - tab_mean) / tab_std).astype(np.float32)

    # ── Variant B: CNN + Tabular ─────────────────────────────────────
    oof_b, test_b, map_b, pc_b = run_variant(
        f"Variant B: CNN + {len(available)} Tabular Features",
        train_seq, train_mask, train_lens,
        test_seq, test_mask, test_lens,
        y, class_weights,
        tabular_train=train_tab, tabular_test=test_tab
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60, flush=True)
    print("  E158 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"\n  Variant A (Pure CNN):        mAP = {map_a:.4f}", flush=True)
    print(f"  Variant B (CNN + Tabular):   mAP = {map_b:.4f}", flush=True)
    print(f"\n  Per-class comparison (A / B):", flush=True)
    for cls in CLASSES:
        a_val = pc_a.get(cls, 0)
        b_val = pc_b.get(cls, 0)
        diff = b_val - a_val
        sign = "+" if diff >= 0 else ""
        print(f"    {cls:15s}: {a_val:.4f} / {b_val:.4f}  ({sign}{diff:.4f})", flush=True)

    # ── Save artifacts ───────────────────────────────────────────────
    best_name = "A" if map_a >= map_b else "B"
    best_map = max(map_a, map_b)
    best_oof = oof_a if map_a >= map_b else oof_b
    best_test = test_a if map_a >= map_b else test_b

    np.save(ROOT / "oof_e158.npy", best_oof)
    np.save(ROOT / "test_e158.npy", best_test)
    print(f"\n  Saved oof_e158.npy (variant {best_name})", flush=True)
    print(f"  Saved test_e158.npy (variant {best_name})", flush=True)

    # Also save both variants
    np.save(ROOT / "oof_e158_a.npy", oof_a)
    np.save(ROOT / "test_e158_a.npy", test_a)
    np.save(ROOT / "oof_e158_b.npy", oof_b)
    np.save(ROOT / "test_e158_b.npy", test_b)

    # Save submission for best variant
    save_submission(best_test, f"e158_cnn_v2_{best_name}", cv_map=best_map)

    print("\nDone!", flush=True)
