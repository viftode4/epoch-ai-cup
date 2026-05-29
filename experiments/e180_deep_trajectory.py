"""E180: Deep Trajectory Model — Properly Architected.

Architecture designed for radar bird track data:

1. INPUT ENGINEERING (not just raw channels):
   - Raw: lon, lat, alt, RCS (4 channels)
   - Derived: speed, heading, climb_rate, rcs_change (4 channels)
   - Physics: curvature, heading_change_rate (2 channels)
   - Invariant: path is made translation/rotation invariant
   Total: 10 channels × T timesteps

2. MULTI-SCALE TEMPORAL ENCODER:
   - InceptionTime-style: parallel convolutions at 3, 7, 15, 31 kernel sizes
   - Captures wingbeat patterns (short), flight segments (medium), track shape (long)
   - Each scale has its own BatchNorm + ReLU
   - Concatenate across scales → rich multi-scale representation

3. CHANNEL INTERACTION MODULE:
   - Cross-channel attention: learns which channel combinations matter
   - E.g., "RCS dropping while altitude rising" = specific flight mode
   - Squeeze-and-excitation block adapted for time series

4. TEMPORAL ATTENTION POOLING:
   - Not just global average pool — learns WHICH timesteps are most informative
   - E.g., the turning point in a soaring BoP track is more informative than straight segments

5. CLASSIFICATION HEAD:
   - Embedding (128-dim) → dropout → 9-class output
   - Class-balanced focal loss
   - The embedding can be extracted for GBDT blending

6. TRAINING:
   - 5-fold SGKF with group-aware splits
   - Augmentation: time crop, Gaussian noise, speed jitter, random channel dropout
   - Cosine LR with warmup
   - 80 epochs, early stopping on val mAP
"""

from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
MONTHS = [1, 4, 9, 10]
SEQ_LEN = 128  # pad/truncate to this length
N_CHANNELS = 10
DEVICE = torch.device("cpu")

print("=" * 90)
print("  E180: Deep Trajectory Model — Properly Architected")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)


def extract_sequence(row):
    """Extract 10-channel sequence from a trajectory row.

    Makes the path translation and rotation invariant:
    - Position relative to centroid (not absolute lon/lat)
    - Heading relative to initial heading
    """
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    n = len(pts)

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])

    # Convert to meters relative to centroid (translation invariant)
    x = (lons - lons.mean()) * 67000  # approx meters
    z = (lats - lats.mean()) * 111000

    # Normalize altitude relative to track
    alt_norm = alts - alts.mean()

    # RCS: keep on dB scale but center
    rcs_norm = rcs - rcs.mean()

    # ── Derived channels ──
    dt = np.maximum(np.diff(times), 0.001)

    # Speed (m/s)
    dx = np.diff(x)
    dz = np.diff(z)
    speed = np.sqrt(dx**2 + dz**2) / dt
    speed = np.concatenate([[speed[0] if len(speed) > 0 else 0], speed])

    # Heading (radians, rotation invariant by subtracting initial)
    heading = np.arctan2(dz, dx)
    if len(heading) > 0:
        heading = heading - heading[0]  # relative to initial heading
    heading = np.concatenate([[0], heading])

    # Climb rate (m/s)
    climb = np.diff(alts) / dt
    climb = np.concatenate([[0], climb])

    # RCS change rate
    rcs_change = np.diff(rcs) / dt
    rcs_change = np.concatenate([[0], rcs_change])

    # Curvature (heading change rate, rad/s)
    if len(heading) > 1:
        dheading = np.diff(heading)
        # Wrap to [-pi, pi]
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
        curvature = np.concatenate([[0], dheading])
    else:
        curvature = np.zeros(n)

    # Heading change magnitude (unsigned curvature)
    heading_change_mag = np.abs(curvature)

    # Stack: 10 channels
    seq = np.stack([
        x, z, alt_norm, rcs_norm,       # raw (translation invariant)
        speed, heading, climb, rcs_change,  # derived
        curvature, heading_change_mag,      # physics
    ], axis=0)  # (10, T)

    # Replace NaN/Inf
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    return seq.astype(np.float32)


def pad_or_truncate(seq, target_len=SEQ_LEN):
    """Pad or truncate sequence to fixed length. Returns (seq, mask)."""
    C, T = seq.shape
    if T >= target_len:
        # Take center crop (preserves both start and end patterns)
        start = (T - target_len) // 2
        return seq[:, start:start+target_len], np.ones(target_len, dtype=np.float32)
    else:
        padded = np.zeros((C, target_len), dtype=np.float32)
        mask = np.zeros(target_len, dtype=np.float32)
        padded[:, :T] = seq
        mask[:T] = 1.0
        return padded, mask


print("  Extracting sequences...", flush=True)
train_seqs = []
test_seqs = []
train_masks = []
test_masks = []

for i, (_, row) in enumerate(train_df.iterrows()):
    seq = extract_sequence(row)
    seq_p, mask = pad_or_truncate(seq)
    train_seqs.append(seq_p)
    train_masks.append(mask)
    if (i + 1) % 500 == 0:
        print(f"    Train: {i+1}/{len(train_df)}", flush=True)

for i, (_, row) in enumerate(test_df.iterrows()):
    seq = extract_sequence(row)
    seq_p, mask = pad_or_truncate(seq)
    test_seqs.append(seq_p)
    test_masks.append(mask)
    if (i + 1) % 500 == 0:
        print(f"    Test: {i+1}/{len(test_df)}", flush=True)

X_seq_train = np.stack(train_seqs)  # (N, 10, 128)
X_seq_test = np.stack(test_seqs)
M_train = np.stack(train_masks)  # (N, 128)
M_test = np.stack(test_masks)

# Per-channel normalization (fit on train)
for c in range(N_CHANNELS):
    mu = X_seq_train[:, c, :].mean()
    std = X_seq_train[:, c, :].std() + 1e-8
    X_seq_train[:, c, :] = (X_seq_train[:, c, :] - mu) / std
    X_seq_test[:, c, :] = (X_seq_test[:, c, :] - mu) / std

print(f"  Sequences: train={X_seq_train.shape}, test={X_seq_test.shape}")
print(f"  Extraction time: {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class InceptionBlock(nn.Module):
    """Multi-scale convolution block (InceptionTime-style)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=31, padding=15),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)


class SqueezeExcitation(nn.Module):
    """Channel attention: learns which channels matter for each sample."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=2)  # (B, C)
        w = self.fc(w).unsqueeze(2)  # (B, C, 1)
        return x * w


class TemporalAttentionPool(nn.Module):
    """Learn which timesteps are most informative."""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x, mask=None):
        # x: (B, C, T), mask: (B, T)
        attn = self.attention(x)  # (B, 1, T)
        if mask is not None:
            attn = attn * mask.unsqueeze(1)
            attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
        return (x * attn).sum(dim=2)  # (B, C)


class BirdTrackNet(nn.Module):
    """Full architecture for radar bird track classification."""
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES, embed_dim=128):
        super().__init__()

        # Multi-scale inception blocks (3 stacked)
        self.inc1 = InceptionBlock(n_channels, 64)
        self.se1 = SqueezeExcitation(64)

        self.inc2 = InceptionBlock(64, 128)
        self.se2 = SqueezeExcitation(128)

        self.inc3 = InceptionBlock(128, 256)
        self.se3 = SqueezeExcitation(256)

        # Residual connection from input (downsampled)
        self.residual = nn.Conv1d(n_channels, 256, kernel_size=1)

        # Temporal attention pooling
        self.attn_pool = TemporalAttentionPool(256)

        # Also global avg + max pool for robustness
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head: 256*3 (attn + avg + max) -> embed -> classes
        self.embed = nn.Sequential(
            nn.Linear(256 * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, mask=None):
        # x: (B, C, T), mask: (B, T)
        residual = self.residual(x)

        h = self.inc1(x)
        h = self.se1(h)
        h = self.inc2(h)
        h = self.se2(h)
        h = self.inc3(h)
        h = self.se3(h)

        # Add residual
        h = h + residual

        # Three pooling strategies
        h_attn = self.attn_pool(h, mask)        # (B, 256)
        h_avg = self.avg_pool(h).squeeze(2)      # (B, 256)
        h_max = self.max_pool(h).squeeze(2)      # (B, 256)

        h_cat = torch.cat([h_attn, h_avg, h_max], dim=1)  # (B, 768)

        embedding = self.embed(h_cat)  # (B, 128)
        logits = self.classifier(embedding)  # (B, 9)
        return logits, embedding

    def get_embedding(self, x, mask=None):
        with torch.no_grad():
            _, emb = self.forward(x, mask)
        return emb


# ══════════════════════════════════════════════════════════════════════
# 3. DATASET & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

class TrackDataset(Dataset):
    def __init__(self, sequences, masks, labels=None, augment=False):
        self.sequences = sequences
        self.masks = masks
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment:
            # Time crop: randomly remove 20% from start or end
            T = int(mask.sum())
            if T > 20:
                crop = np.random.randint(0, max(1, T // 5))
                if np.random.random() > 0.5:
                    seq[:, :crop] = 0
                    mask[:crop] = 0
                else:
                    seq[:, T-crop:T] = 0
                    mask[T-crop:T] = 0

            # Gaussian noise (scale-aware)
            noise_scale = 0.05
            noise = np.random.randn(*seq.shape).astype(np.float32) * noise_scale
            seq = seq + noise * mask[np.newaxis, :]

            # Random channel dropout (drop 1-2 channels)
            if np.random.random() > 0.7:
                drop_ch = np.random.choice(N_CHANNELS, size=np.random.randint(1, 3), replace=False)
                seq[drop_ch] = 0

            # Speed jitter (multiply speed-related channels by random factor)
            if np.random.random() > 0.5:
                jitter = 1.0 + np.random.randn() * 0.1
                seq[4] *= jitter  # speed channel

        out = {"seq": torch.tensor(seq), "mask": torch.tensor(mask)}
        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out


# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_model(X_tr, M_tr, y_tr, X_va, M_va, y_va, X_te, M_te,
                n_epochs=80, lr=1e-3, batch_size=64, patience=15):
    """Train one fold, return val predictions, test predictions, embeddings."""

    # Class weights for focal loss
    class_weights = torch.tensor(
        1.0 / np.maximum(np.bincount(y_tr, minlength=N_CLASSES), 1).astype(np.float32),
        dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * N_CLASSES

    train_ds = TrackDataset(X_tr, M_tr, y_tr, augment=True)
    val_ds = TrackDataset(X_va, M_va, y_va, augment=False)
    test_ds = TrackDataset(X_te, M_te, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BirdTrackNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_map = -1
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            logits, _ = model(batch["seq"].to(DEVICE), batch["mask"].to(DEVICE))
            loss = F.cross_entropy(logits, batch["label"].to(DEVICE), weight=class_weights.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    logits, _ = model(batch["seq"].to(DEVICE), batch["mask"].to(DEVICE))
                    val_preds.append(F.softmax(logits, dim=1).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_map, _ = compute_map(y_va, val_preds)

            if val_map > best_val_map:
                best_val_map = val_map
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 5

            if (epoch + 1) % 20 == 0:
                print(f"      Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_mAP={val_map:.4f}, best={best_val_map:.4f}", flush=True)

            if patience_counter >= patience:
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Get final predictions
    val_preds = []
    test_preds = []
    with torch.no_grad():
        for batch in val_loader:
            logits, _ = model(batch["seq"].to(DEVICE), batch["mask"].to(DEVICE))
            val_preds.append(F.softmax(logits, dim=1).cpu().numpy())
        for batch in test_loader:
            logits, _ = model(batch["seq"].to(DEVICE), batch["mask"].to(DEVICE))
            test_preds.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(val_preds), np.concatenate(test_preds), best_val_map


# ══════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Training BirdTrackNet (5-fold SGKF)...", flush=True)

N_FOLDS = 5
oof_cnn = np.zeros((len(y), N_CLASSES))
test_cnn = np.zeros((len(test_df), N_CLASSES))

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_seq_train, y, groups)):
    t_fold = time.time()
    print(f"\n    Fold {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(va_idx)})", flush=True)

    va_preds, te_preds, best_map = train_model(
        X_seq_train[tr_idx], M_train[tr_idx], y[tr_idx],
        X_seq_train[va_idx], M_train[va_idx], y[va_idx],
        X_seq_test, M_test,
        n_epochs=80, lr=1e-3, batch_size=64, patience=20,
    )

    oof_cnn[va_idx] = va_preds
    test_cnn += te_preds / N_FOLDS

    fold_map, _ = compute_map(y[va_idx], va_preds)
    print(f"    Fold {fold+1}: val_mAP={fold_map:.4f}, best_epoch_mAP={best_map:.4f} ({time.time()-t_fold:.0f}s)", flush=True)


# ══════════════════════════════════════════════════════════════════════
# 6. EVALUATION & BLENDING
# ══════════════════════════════════════════════════════════════════════

def true_lomo(oof, name=""):
    skf, _ = compute_map(y, oof)
    scores = {}
    for m in MONTHS:
        mask = train_months == m
        s, _ = compute_map(y[mask], oof[mask])
        scores[m] = s
    lomo = np.mean(list(scores.values()))
    ms = " ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()))
    print(f"  {name:<50s} SKF={skf:.4f} LOMO={lomo:.4f} [{ms}]", flush=True)
    return skf, lomo

print(f"\n{'='*90}")
print("  RESULTS")
print(f"{'='*90}")

oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))

true_lomo(oof_e175, "E175 best (baseline)")
true_lomo(oof_cnn, "BirdTrackNet CNN")

# Blend at various alphas
print("\n  Blends:")
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    blend = renorm_rows((1-alpha) * oof_e175 + alpha * oof_cnn)
    true_lomo(blend, f"E175 + CNN@{alpha}")

# Save
np.save(ROOT / "oof_e180_cnn.npy", oof_cnn)
np.save(ROOT / "test_e180_cnn.npy", test_cnn)
save_submission(renorm_rows(test_cnn), "e180_cnn_raw", cv_map=compute_map(y, oof_cnn)[0])

# Best blend submission
best_blend_lomo = -1
best_alpha = 0.1
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
    blend = renorm_rows((1-alpha) * oof_e175 + alpha * oof_cnn)
    _, lomo = true_lomo(blend, "")
    if lomo > best_blend_lomo:
        best_blend_lomo = lomo
        best_alpha = alpha

blend_test = renorm_rows((1-best_alpha) * test_e175 + best_alpha * test_cnn)
skf_blend, _ = compute_map(y, renorm_rows((1-best_alpha) * oof_e175 + best_alpha * oof_cnn))
save_submission(blend_test, f"e180_cnn_blend_{int(best_alpha*100)}", cv_map=skf_blend)

elapsed = time.time() - t0
print(f"\n{'='*90}")
print(f"  E180 complete in {elapsed/60:.1f} min")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*90}")
