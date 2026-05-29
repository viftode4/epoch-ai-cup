"""E180: 1D-CNN on raw trajectory sequences for diversity blending.

V2: Global normalization (not per-track z-score!) to preserve absolute scale.

Architecture:
  - Parse EWKB trajectories -> 10 channels with GLOBAL normalization
  - Channels: displacement_x, displacement_y, altitude, rcs, speed, turn_rate,
              climb_rate, rcs_change, distance_from_start, cumulative_turn
  - Pad/truncate to 128 steps -> (batch, 10, 128) tensor
  - 3-layer Conv1d + BN + ReLU + AdaptiveAvgPool + AdaptiveMaxPool -> 512-dim
  - Dropout(0.3) + Linear(512, 9)
  - Focal loss (handles class imbalance), AdamW + cosine LR, 80 epochs
  - Augmentation: random time crop (80%) + Gaussian noise + time reversal

Key fix: V1 used per-track z-score which destroyed absolute altitude/RCS/speed —
the most discriminative features. V2 uses dataset-wide percentile scaling.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, ROOT, load_test, load_train, parse_ewkb_4d
from src.metrics import compute_map, print_results

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEQ_LEN = 128
N_CHANNELS = 10
EPOCHS = 80
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4
SEED = 42


# ======================================================================
# Trajectory Preprocessing (GLOBAL normalization)
# ======================================================================

def extract_raw_channels(ewkb_hex: str, traj_time_str: str) -> np.ndarray:
    """Extract raw (unnormalized) channels from trajectory.

    Returns (10, seq_len_variable) array with:
      0: dx  - displacement in x (meters) relative to track start
      1: dy  - displacement in y (meters) relative to track start
      2: alt - raw altitude (meters)
      3: rcs - raw RCS (dBm2)
      4: speed - instantaneous speed (m/s)
      5: turn_rate - heading change / dt (rad/s)
      6: climb_rate - altitude change / dt (m/s)
      7: rcs_change - RCS change between steps (dB/s)
      8: dist_from_start - cumulative distance from track start (m)
      9: cumulative_turn - cumulative absolute turn angle (rad)
    """
    points = parse_ewkb_4d(ewkb_hex)
    n = len(points)
    if n < 2:
        return np.zeros((N_CHANNELS, 1), dtype=np.float32)

    lons = np.array([p[0] for p in points], dtype=np.float64)
    lats = np.array([p[1] for p in points], dtype=np.float64)
    alts = np.array([p[2] for p in points], dtype=np.float64)
    rcs = np.array([p[3] for p in points], dtype=np.float64)

    # Parse trajectory times
    try:
        times = np.array(json.loads(traj_time_str), dtype=np.float64)
    except (json.JSONDecodeError, TypeError):
        times = np.arange(n, dtype=np.float64)

    # Convert to meters from track start
    cos_lat = np.cos(np.radians(lats[0]))
    dx = (lons - lons[0]) * 111320.0 * cos_lat
    dy = (lats - lats[0]) * 110540.0

    # Step differences
    step_dx = np.diff(dx)
    step_dy = np.diff(dy)
    dt = np.diff(times)
    dt = np.maximum(dt, 0.5)

    # Speed
    step_dist = np.sqrt(step_dx**2 + step_dy**2)
    speed = np.concatenate([[0.0], step_dist / dt])

    # Heading and turn rate
    heading = np.arctan2(step_dy, step_dx)
    heading_change = np.diff(heading)
    heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi
    # Turn rate: heading_change / dt (skip first dt since heading_change is one shorter)
    turn_rate = np.concatenate([[0.0, 0.0], heading_change / dt[1:]])

    # Climb rate
    alt_diff = np.diff(alts)
    climb_rate = np.concatenate([[0.0], alt_diff / dt])

    # RCS change rate
    rcs_diff = np.diff(rcs)
    rcs_change = np.concatenate([[0.0], rcs_diff / dt])

    # Cumulative distance from start
    dist_from_start = np.concatenate([[0.0], np.cumsum(step_dist)])

    # Cumulative absolute turn
    cumulative_turn = np.concatenate([[0.0, 0.0], np.cumsum(np.abs(heading_change))])

    out = np.stack([
        dx, dy, alts, rcs, speed,
        turn_rate, climb_rate, rcs_change,
        dist_from_start, cumulative_turn,
    ], axis=0).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (with zeros) or truncate a (channels, seq_len) array."""
    _, cur_len = seq.shape
    if cur_len >= target_len:
        return seq[:, :target_len]
    pad = np.zeros((seq.shape[0], target_len - cur_len), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)


def extract_all_raw(df: pd.DataFrame) -> np.ndarray:
    """Extract raw channels for all tracks. Returns (n, C, L)."""
    results = []
    for i in range(len(df)):
        seq = extract_raw_channels(df.iloc[i]["trajectory"], df.iloc[i]["trajectory_time"])
        seq = pad_or_truncate(seq, SEQ_LEN)
        results.append(seq)
    return np.stack(results, axis=0)


def compute_global_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust global normalization stats (median, IQR) per channel.

    Only considers non-zero values (padded regions are zero).
    Returns (medians, scales) each of shape (C,).
    """
    C = X.shape[1]
    medians = np.zeros(C, dtype=np.float32)
    scales = np.ones(C, dtype=np.float32)

    for c in range(C):
        vals = X[:, c, :].ravel()
        # Only use non-zero values for stats (exclude padding)
        nonzero = vals[vals != 0]
        if len(nonzero) < 10:
            continue
        medians[c] = np.median(nonzero)
        q25, q75 = np.percentile(nonzero, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-6:
            scales[c] = max(np.std(nonzero), 1e-6)
        else:
            scales[c] = iqr
    return medians, scales


def normalize_global(X: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Apply global robust normalization: (x - median) / IQR, clipped to [-5, 5]."""
    X_norm = X.copy()
    for c in range(X.shape[1]):
        X_norm[:, c, :] = (X[:, c, :] - medians[c]) / scales[c]
    X_norm = np.clip(X_norm, -5.0, 5.0)
    # Re-zero the padding
    mask = np.abs(X).sum(axis=1) == 0  # (n, L) - True where all channels are 0
    X_norm[np.broadcast_to(mask[:, np.newaxis, :], X_norm.shape)] = 0.0
    return X_norm.astype(np.float32)


# ======================================================================
# PyTorch Dataset with augmentation
# ======================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (C, L)

        if self.augment:
            # Find real sequence length (non-padded)
            nonzero = x.abs().sum(dim=0) > 0  # (L,)
            nz_idx = nonzero.nonzero(as_tuple=True)[0]
            real_len = nz_idx[-1].item() + 1 if len(nz_idx) > 0 else 1

            # Random time crop: take 80% of sequence
            if real_len > 5:
                crop_len = max(4, int(real_len * 0.8))
                max_start = real_len - crop_len
                if max_start > 0:
                    start = torch.randint(0, max_start, (1,)).item()
                else:
                    start = 0
                cropped = x[:, start:start + crop_len]
                x = torch.zeros_like(self.X[0])
                x[:, :crop_len] = cropped

            # Time reversal (50% chance)
            if torch.rand(1).item() < 0.5:
                nz2 = x.abs().sum(dim=0) > 0
                nz_idx2 = nz2.nonzero(as_tuple=True)[0]
                if len(nz_idx2) > 1:
                    rl2 = nz_idx2[-1].item() + 1
                    x[:, :rl2] = x[:, :rl2].flip(dims=[1])

            # Gaussian noise (sigma=0.02 on normalized data)
            noise = torch.randn_like(x) * 0.02
            mask = (x.abs().sum(dim=0, keepdim=True) > 0).float()
            x = x + noise * mask

        if self.y is not None:
            return x, self.y[idx]
        return x


# ======================================================================
# Model: 1D-CNN with residual connections
# ======================================================================

class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TrajCNN(nn.Module):
    """1D-CNN with residual blocks and dual global pooling."""

    def __init__(self, in_channels=N_CHANNELS, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        # Stem: project to hidden dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResBlock1d(128, kernel_size=5),
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResBlock1d(256, kernel_size=3),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        avg = self.avg_pool(x).squeeze(-1)
        mx = self.max_pool(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)  # (batch, 512)
        return self.head(x)


# ======================================================================
# Focal Loss for class imbalance
# ======================================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        return focal


# ======================================================================
# Training
# ======================================================================

def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    class_weights: np.ndarray,
    fold_idx: int, seed: int,
) -> tuple[np.ndarray, nn.Module]:
    """Train one fold, return (val_predictions, model)."""
    torch.manual_seed(seed + fold_idx * 1000)
    np.random.seed(seed + fold_idx * 1000)

    device = torch.device("cpu")

    train_ds = TrajectoryDataset(X_tr, y_tr, augment=True)
    val_ds = TrajectoryDataset(X_va, y_va, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    model = TrajCNN().to(device)
    weight_tensor = torch.from_numpy(class_weights).float().to(device)
    criterion = FocalLoss(weight=weight_tensor, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=LR * 0.01
    )

    best_val_map = -1.0
    best_val_preds = None
    best_model_state = None
    best_epoch = 0
    patience = 25
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validate
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                all_logits.append(logits.cpu())
                all_y.append(yb)

        logits_cat = torch.cat(all_logits, dim=0)
        y_cat = torch.cat(all_y, dim=0).numpy()
        val_probs = F.softmax(logits_cat, dim=1).numpy()
        val_map, _ = compute_map(y_cat, val_probs)

        if val_map > best_val_map:
            best_val_map = val_map
            best_val_preds = val_probs.copy()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = train_loss / max(n_batches, 1)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"      Ep {epoch+1:3d}/{EPOCHS}: loss={avg_loss:.4f} val_mAP={val_map:.4f} "
                  f"(best={best_val_map:.4f}@{best_epoch}) lr={lr_now:.1e}", flush=True)

        if no_improve >= patience:
            print(f"      Early stop at epoch {epoch+1}, best={best_val_map:.4f}@{best_epoch}", flush=True)
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return best_val_preds, model


def predict_test(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    """Get test predictions."""
    device = torch.device("cpu")
    model.eval()
    ds = TrajectoryDataset(X_test, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(xb.to(device))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ======================================================================
# LOMO
# ======================================================================

def eval_lomo(oof: np.ndarray, y: np.ndarray, months: np.ndarray) -> tuple[float, dict]:
    """Leave-one-month-out mAP."""
    per_month = {}
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() >= 5:
            mm, _ = compute_map(y[mask], oof[mask])
            per_month[m] = mm
    lomo = float(np.mean(list(per_month.values())))
    return lomo, per_month


# ======================================================================
# MAIN
# ======================================================================

def main():
    from sklearn.model_selection import StratifiedGroupKFold

    t_total = time.time()
    print("=" * 70, flush=True)
    print("  E180: TRAJECTORY 1D-CNN (v2 - global normalization)", flush=True)
    print("  Raw sequences -> Conv1d+ResBlocks -> Global Pool -> Focal Loss", flush=True)
    print("=" * 70, flush=True)

    # -- Load data --
    print("\n[1/5] Loading data...", flush=True)
    train_df = load_train()
    test_df = load_test()
    y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    groups = train_df["primary_observation_id"].values

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    counts = np.bincount(y, minlength=N_CLASSES)
    for i, c in enumerate(CLASSES):
        print(f"    {c:15s}: {counts[i]:5d}", flush=True)

    # -- Extract raw channels --
    print("\n[2/5] Extracting trajectory channels...", flush=True)
    t_prep = time.time()

    cache_v2_train = ROOT / "data" / "_cached_traj_cnn_v2_train.npy"
    cache_v2_test = ROOT / "data" / "_cached_traj_cnn_v2_test.npy"

    if cache_v2_train.exists() and cache_v2_test.exists():
        print("  Loading cached v2 trajectories...", flush=True)
        X_train_raw = np.load(cache_v2_train)
        X_test_raw = np.load(cache_v2_test)
    else:
        print("  Processing train trajectories...", flush=True)
        X_train_raw = extract_all_raw(train_df)
        print("  Processing test trajectories...", flush=True)
        X_test_raw = extract_all_raw(test_df)
        np.save(cache_v2_train, X_train_raw)
        np.save(cache_v2_test, X_test_raw)

    print(f"  Raw shapes: train={X_train_raw.shape}, test={X_test_raw.shape}", flush=True)

    # Compute global stats from TRAIN only, apply to both
    print("  Computing global normalization stats from train...", flush=True)
    medians, scales = compute_global_stats(X_train_raw)
    channel_names = ['dx', 'dy', 'alt', 'rcs', 'speed', 'turn_rate',
                     'climb_rate', 'rcs_change', 'dist_start', 'cum_turn']
    for i, name in enumerate(channel_names):
        print(f"    {name:12s}: median={medians[i]:8.2f}, IQR={scales[i]:8.2f}", flush=True)

    X_train = normalize_global(X_train_raw, medians, scales)
    X_test = normalize_global(X_test_raw, medians, scales)

    # Track length stats
    real_lens = []
    for i in range(len(X_train)):
        nz = np.abs(X_train[i]).sum(axis=0) > 0
        real_lens.append(nz.sum())
    real_lens = np.array(real_lens)
    print(f"  Track lengths: min={real_lens.min()}, median={np.median(real_lens):.0f}, "
          f"max={real_lens.max()}, truncated(>{SEQ_LEN}): {(real_lens >= SEQ_LEN).sum()}", flush=True)
    print(f"  Preprocessing time: {time.time()-t_prep:.1f}s", flush=True)

    # -- Class weights --
    class_weights = len(y) / (N_CLASSES * np.maximum(counts, 1).astype(float))
    class_weights = np.clip(class_weights, 0.3, 8.0)
    # Soften: sqrt to avoid over-boosting rare classes
    class_weights = np.sqrt(class_weights)
    print(f"\n  Class weights (sqrt): {np.round(class_weights, 2)}", flush=True)

    # -- Train CNN --
    print(f"\n[3/5] Training CNN ({N_FOLDS} folds, {EPOCHS} max epochs)...", flush=True)
    t_train = time.time()

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    test_preds_sum = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
        print(f"\n    Fold {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(va_idx)})", flush=True)
        t_fold = time.time()

        va_preds, model = train_one_fold(
            X_train[tr_idx], y[tr_idx],
            X_train[va_idx], y[va_idx],
            class_weights, fold, SEED,
        )
        oof_preds[va_idx] = va_preds
        test_preds_sum += predict_test(model, X_test)

        fold_map, fold_pc = compute_map(y[va_idx], va_preds)
        fold_scores.append(fold_map)
        elapsed = time.time() - t_fold
        print(f"    Fold {fold+1} mAP: {fold_map:.4f} ({elapsed:.1f}s)", flush=True)
        # Per-class for this fold
        for cls in CLASSES:
            if fold_pc[cls] < 0.3:
                print(f"      WEAK: {cls} AP={fold_pc[cls]:.4f}", flush=True)

        del model

    test_preds = test_preds_sum / N_FOLDS
    print(f"\n  Training time: {time.time()-t_train:.1f}s", flush=True)

    # -- Evaluation --
    print("\n[4/5] Evaluation...", flush=True)
    overall_map, per_class = compute_map(y, oof_preds)
    print_results(overall_map, per_class, "E180 CNN (global norm, focal loss)")

    print(f"\n  Per-fold mAPs: {[f'{s:.4f}' for s in fold_scores]}", flush=True)
    print(f"  Mean fold mAP: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}", flush=True)

    lomo, per_month = eval_lomo(oof_preds, y, train_months)
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(per_month.items()))
    print(f"\n  LOMO: {lomo:.4f}  ({month_str})", flush=True)

    # Save
    np.save(ROOT / "oof_e180_cnn.npy", oof_preds)
    np.save(ROOT / "test_e180_cnn.npy", test_preds)
    print(f"  Saved: oof_e180_cnn.npy, test_e180_cnn.npy", flush=True)

    # -- Blend with E175 --
    print("\n[5/5] Blending with E175...", flush=True)
    e175_oof_path = ROOT / "oof_e175_best.npy"
    e175_test_path = ROOT / "test_e175_best.npy"

    if e175_oof_path.exists() and e175_test_path.exists():
        oof_e175 = np.load(e175_oof_path)
        test_e175 = np.load(e175_test_path)

        oof_e175 = np.clip(oof_e175, 1e-8, None)
        oof_e175 = oof_e175 / oof_e175.sum(axis=1, keepdims=True)

        e175_map, _ = compute_map(y, oof_e175)
        e175_lomo, _ = eval_lomo(oof_e175, y, train_months)

        print(f"\n  {'Method':>8} {'SKF':>10} {'LOMO':>10}", flush=True)
        print(f"  {'-'*32}", flush=True)
        print(f"  {'E175':>8} {e175_map:>10.4f} {e175_lomo:>10.4f}", flush=True)
        print(f"  {'CNN':>8} {overall_map:>10.4f} {lomo:>10.4f}", flush=True)

        best_lomo = e175_lomo
        best_alpha = 0.0

        for alpha in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            blended = alpha * oof_preds + (1 - alpha) * oof_e175
            blended = np.clip(blended, 1e-8, None)
            blended = blended / blended.sum(axis=1, keepdims=True)

            b_map, _ = compute_map(y, blended)
            b_lomo, _ = eval_lomo(blended, y, train_months)

            marker = ""
            if b_lomo > best_lomo:
                best_lomo = b_lomo
                best_alpha = alpha
                marker = " ***"

            print(f"  {alpha:>8.2f} {b_map:>10.4f} {b_lomo:>10.4f}{marker}", flush=True)

        if best_alpha > 0:
            print(f"\n  Best blend: alpha={best_alpha:.2f}, LOMO={best_lomo:.4f}", flush=True)
            best_oof = best_alpha * oof_preds + (1 - best_alpha) * oof_e175
            best_test = best_alpha * test_preds + (1 - best_alpha) * test_e175
            np.save(ROOT / "oof_e180_blend.npy", best_oof)
            np.save(ROOT / "test_e180_blend.npy", best_test)

            # Per-class comparison
            _, e175_pc = compute_map(y, oof_e175)
            best_oof_norm = np.clip(best_oof, 1e-8, None)
            best_oof_norm = best_oof_norm / best_oof_norm.sum(axis=1, keepdims=True)
            _, blend_pc = compute_map(y, best_oof_norm)
            print(f"\n  Per-class (E175 vs Blend alpha={best_alpha:.2f}):", flush=True)
            print(f"  {'Class':>15s} {'E175':>8} {'Blend':>8} {'Delta':>8}", flush=True)
            for cls in CLASSES:
                d = blend_pc[cls] - e175_pc[cls]
                flag = " <+" if d > 0.005 else (" <-" if d < -0.005 else "")
                print(f"  {cls:>15s} {e175_pc[cls]:>8.4f} {blend_pc[cls]:>8.4f} {d:>+8.4f}{flag}", flush=True)
        else:
            print(f"\n  No blend beat E175 standalone (LOMO={e175_lomo:.4f})", flush=True)
    else:
        print("  E175 files not found, skipping blend.", flush=True)
        e175_lomo = 0.0
        best_alpha = 0.0

    # -- Correlation analysis --
    print("\n  Prediction correlation analysis:", flush=True)
    if e175_oof_path.exists():
        # Spearman rank correlation between CNN and E175 per class
        from scipy.stats import spearmanr
        print(f"  {'Class':>15s} {'Spearman r':>12}", flush=True)
        for i, cls in enumerate(CLASSES):
            r, _ = spearmanr(oof_preds[:, i], oof_e175[:, i])
            print(f"  {cls:>15s} {r:>12.4f}", flush=True)

    # -- Summary --
    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"  E180 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  CNN SKF mAP:     {overall_map:.4f}", flush=True)
    print(f"  CNN LOMO:        {lomo:.4f}", flush=True)
    print(f"  E175 LOMO:       {e175_lomo:.4f}", flush=True)
    if best_alpha > 0:
        print(f"  Best blend:      alpha={best_alpha:.2f}, LOMO={best_lomo:.4f}", flush=True)
    print(f"  Mean fold mAP:   {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}", flush=True)
    print(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
