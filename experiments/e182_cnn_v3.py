"""E182: CNN v3 — Properly Architected Trajectory Model.

Fixes from v2 audit:
1. InceptionTime multi-scale convolutions (3, 7, 15, 31, 63 kernels)
2. Temporal attention pooling (learn WHICH timesteps matter)
3. Squeeze-Excitation channel interaction (learn channel combinations)
4. Center crop for long tracks (not just first 128)
5. Mask-aware pooling (short tracks don't get diluted by padding)
6. RCS linear channel added (11 channels total)
7. 3-seed averaging per fold for variance reduction
8. Proper length encoding as additional channel

Architecture:
  Input: (batch, 12, 128) — 12 channels × 128 timesteps
  → InceptionBlock (multi-scale: 3,7,15,31 kernels) → 128 channels
  → SqueezeExcitation (channel interaction)
  → InceptionBlock → 256 channels
  → SqueezeExcitation
  → TemporalAttentionPool (masked) + MaskedAvgPool + MaskedMaxPool → 768-dim
  → Dropout → Linear(768, 128) → ReLU → Dropout → Linear(128, 9)
  → Focal loss, AdamW, CosineAnnealingWarmRestarts
"""

from __future__ import annotations
import json, sys, time, warnings
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

from src.data import CLASSES, ROOT, load_test, load_train, parse_ewkb_4d
from src.metrics import compute_map

N_CLASSES = len(CLASSES)
N_FOLDS = 5
SEQ_LEN = 128
N_CHANNELS = 12  # 10 from v2 + rcs_linear + length_encoding
EPOCHS = 100
BATCH_SIZE = 64
LR = 5e-4
DROPOUT = 0.35
N_SEEDS = 3  # multi-seed averaging

print("=" * 80)
print("  E182: CNN v3 — InceptionTime + Attention + Multi-Seed")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
t0 = time.time()


# ══════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def extract_channels(ewkb_hex, traj_time_str):
    """Extract 12 channels from trajectory."""
    points = parse_ewkb_4d(ewkb_hex)
    n = len(points)
    if n < 2:
        return np.zeros((N_CHANNELS, 1), dtype=np.float32), 1

    lons = np.array([p[0] for p in points], dtype=np.float64)
    lats = np.array([p[1] for p in points], dtype=np.float64)
    alts = np.array([p[2] for p in points], dtype=np.float64)
    rcs_dB = np.array([p[3] for p in points], dtype=np.float64)
    rcs_lin = 10 ** (rcs_dB / 10.0)  # NEW: linear RCS

    try:
        times = np.array(json.loads(traj_time_str), dtype=np.float64)
    except:
        times = np.arange(n, dtype=np.float64)

    cos_lat = np.cos(np.radians(lats[0]))
    dx = (lons - lons[0]) * 111320.0 * cos_lat
    dy = (lats - lats[0]) * 110540.0

    dt = np.maximum(np.diff(times), 0.5)
    step_dx = np.diff(dx); step_dy = np.diff(dy)
    step_dist = np.sqrt(step_dx**2 + step_dy**2)

    speed = np.concatenate([[0.0], step_dist / dt])
    heading = np.arctan2(step_dy, step_dx)
    heading_change = np.diff(heading)
    heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi
    turn_rate = np.concatenate([[0.0, 0.0], heading_change / dt[1:]])
    climb_rate = np.concatenate([[0.0], np.diff(alts) / dt])
    rcs_change = np.concatenate([[0.0], np.diff(rcs_dB) / dt])
    dist_from_start = np.concatenate([[0.0], np.cumsum(step_dist)])
    cum_turn = np.concatenate([[0.0, 0.0], np.cumsum(np.abs(heading_change))])

    # Length encoding: position within track (0 to 1)
    length_enc = np.linspace(0, 1, n)

    out = np.stack([
        dx, dy, alts, rcs_dB, speed, turn_rate,
        climb_rate, rcs_change, dist_from_start, cum_turn,
        rcs_lin, length_enc,
    ], axis=0).astype(np.float32)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), n


def pad_or_truncate_center(seq, real_len, target_len=SEQ_LEN):
    """Center crop for long tracks, right-pad for short."""
    C, T = seq.shape
    if T >= target_len:
        start = (T - target_len) // 2
        return seq[:, start:start+target_len], target_len
    pad = np.zeros((C, target_len - T), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1), T


def compute_global_stats(X, real_lens):
    """Robust global normalization (median/IQR), mask-aware."""
    C = X.shape[1]
    medians = np.zeros(C, dtype=np.float32)
    scales = np.ones(C, dtype=np.float32)
    for c in range(C):
        vals = []
        for i in range(len(X)):
            vals.extend(X[i, c, :real_lens[i]].tolist())
        vals = np.array(vals)
        if len(vals) < 10:
            continue
        medians[c] = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25
        scales[c] = max(iqr, np.std(vals) * 0.5, 1e-6)
    return medians, scales


def normalize_global(X, real_lens, medians, scales):
    """Normalize, preserving zeros in padding."""
    X_norm = X.copy()
    for c in range(X.shape[1]):
        X_norm[:, c, :] = (X[:, c, :] - medians[c]) / scales[c]
    X_norm = np.clip(X_norm, -5.0, 5.0)
    # Re-zero padding
    for i in range(len(X_norm)):
        X_norm[i, :, real_lens[i]:] = 0.0
    return X_norm.astype(np.float32)


# Extract
print("  Extracting trajectories...", flush=True)
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES)

train_seqs, train_lens = [], []
for i in range(len(train_df)):
    seq, rl = extract_channels(train_df.iloc[i]["trajectory"], train_df.iloc[i]["trajectory_time"])
    seq, rl = pad_or_truncate_center(seq, rl)
    train_seqs.append(seq); train_lens.append(rl)
    if (i+1) % 1000 == 0:
        print(f"    Train: {i+1}/{len(train_df)}", flush=True)

test_seqs, test_lens = [], []
for i in range(len(test_df)):
    seq, rl = extract_channels(test_df.iloc[i]["trajectory"], test_df.iloc[i]["trajectory_time"])
    seq, rl = pad_or_truncate_center(seq, rl)
    test_seqs.append(seq); test_lens.append(rl)

X_train = np.stack(train_seqs); lens_train = np.array(train_lens)
X_test = np.stack(test_seqs); lens_test = np.array(test_lens)

print(f"  Shapes: train={X_train.shape}, test={X_test.shape}")
print(f"  Track lengths: min={lens_train.min()}, median={np.median(lens_train):.0f}, max={lens_train.max()}")

medians, scales = compute_global_stats(X_train, lens_train)
X_train = normalize_global(X_train, lens_train, medians, scales)
X_test = normalize_global(X_test, lens_test, medians, scales)

# Masks: (N, SEQ_LEN) — 1 for real data, 0 for padding
masks_train = np.zeros((len(X_train), SEQ_LEN), dtype=np.float32)
masks_test = np.zeros((len(X_test), SEQ_LEN), dtype=np.float32)
for i in range(len(X_train)):
    masks_train[i, :lens_train[i]] = 1.0
for i in range(len(X_test)):
    masks_test[i, :lens_test[i]] = 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. MODEL
# ══════════════════════════════════════════════════════════════════════

class InceptionBlock(nn.Module):
    """Multi-scale parallel convolutions (memory-efficient)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        b = out_ch // 4
        # Use depthwise-separable for large kernels to save memory
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, b, 3, padding=1), nn.BatchNorm1d(b), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=3), nn.BatchNorm1d(b), nn.ReLU())
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 15, padding=7, groups=in_ch),  # depthwise
            nn.Conv1d(in_ch, b, 1),  # pointwise
            nn.BatchNorm1d(b), nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 31, padding=15, groups=in_ch),  # depthwise
            nn.Conv1d(in_ch, b, 1),  # pointwise
            nn.BatchNorm1d(b), nn.ReLU()
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return F.relu(self.bn(out + self.residual(x)))


class SqueezeExcitation(nn.Module):
    """Channel attention."""
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())

    def forward(self, x, mask=None):
        if mask is not None:
            # Masked global average
            mask_3d = mask.unsqueeze(1)  # (B, 1, T)
            w = (x * mask_3d).sum(dim=2) / mask_3d.sum(dim=2).clamp(min=1)
        else:
            w = x.mean(dim=2)
        return x * self.fc(w).unsqueeze(2)


class TemporalAttention(nn.Module):
    """Learn which timesteps matter, mask-aware."""
    def __init__(self, ch):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv1d(ch, ch//4, 1), nn.ReLU(), nn.Conv1d(ch//4, 1, 1))

    def forward(self, x, mask=None):
        scores = self.attn(x)  # (B, 1, T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        weights = F.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)  # (B, C)


class MaskedPool(nn.Module):
    """Mask-aware avg and max pooling."""
    def forward(self, x, mask):
        mask_3d = mask.unsqueeze(1)  # (B, 1, T)
        # Avg pool
        x_masked = x * mask_3d
        avg = x_masked.sum(dim=2) / mask_3d.sum(dim=2).clamp(min=1)
        # Max pool
        x_for_max = x.masked_fill(mask_3d == 0, -1e9)
        mx = x_for_max.max(dim=2).values
        return avg, mx


class BirdTrackNetV3(nn.Module):
    def __init__(self, in_ch=N_CHANNELS, n_classes=N_CLASSES):
        super().__init__()
        self.inc1 = InceptionBlock(in_ch, 64)
        self.se1 = SqueezeExcitation(64)
        self.inc2 = InceptionBlock(64, 128)
        self.se2 = SqueezeExcitation(128)

        self.attn_pool = TemporalAttention(128)
        self.masked_pool = MaskedPool()

        # 128 (attn) + 128 (avg) + 128 (max) = 384
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, mask):
        h = self.inc1(x)
        h = self.se1(h, mask)
        h = self.inc2(h)
        h = self.se2(h, mask)

        h_attn = self.attn_pool(h, mask)   # (B, 256)
        h_avg, h_max = self.masked_pool(h, mask)  # (B, 256) each
        h_cat = torch.cat([h_attn, h_avg, h_max], dim=1)  # (B, 768)
        return self.head(h_cat)


# ══════════════════════════════════════════════════════════════════════
# 3. DATASET + AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

class TrackDataset(Dataset):
    def __init__(self, X, masks, y=None, augment=False):
        self.X = torch.from_numpy(X).float()
        self.masks = torch.from_numpy(masks).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        m = self.masks[idx].clone()

        if self.augment:
            real_len = int(m.sum().item())
            if real_len > 5:
                # Random time crop (75-100%)
                crop_frac = 0.75 + 0.25 * torch.rand(1).item()
                crop_len = max(4, int(real_len * crop_frac))
                max_start = real_len - crop_len
                if max_start > 0:
                    start = torch.randint(0, max_start, (1,)).item()
                    cropped = x[:, start:start+crop_len].clone()
                    x.zero_()
                    m.zero_()
                    x[:, :crop_len] = cropped
                    m[:crop_len] = 1.0

                # Time reversal (30%)
                if torch.rand(1).item() < 0.3:
                    rl = int(m.sum().item())
                    x[:, :rl] = x[:, :rl].flip(dims=[1])

                # Gaussian noise (scale=0.03)
                noise = torch.randn_like(x) * 0.03
                x = x + noise * m.unsqueeze(0)

                # Random channel dropout (20% chance, drop 1 channel)
                if torch.rand(1).item() < 0.2:
                    ch = torch.randint(0, N_CHANNELS, (1,)).item()
                    x[ch] = 0.0

                # Speed jitter (10% chance)
                if torch.rand(1).item() < 0.1:
                    jitter = 1.0 + 0.15 * (torch.rand(1).item() - 0.5)
                    x[4] *= jitter  # speed
                    x[8] *= jitter  # dist_from_start

        out = (x, m)
        if self.y is not None:
            return (*out, self.y[idx])
        return out


# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train_fold(X_tr, M_tr, y_tr, X_va, M_va, y_va, X_te, M_te, class_w, seed, fold):
    """Train one fold with one seed."""
    torch.manual_seed(seed * 1000 + fold)
    np.random.seed(seed * 1000 + fold)

    train_ds = TrackDataset(X_tr, M_tr, y_tr, augment=True)
    val_ds = TrackDataset(X_va, M_va, y_va)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)

    model = BirdTrackNetV3()
    cw = torch.tensor(class_w, dtype=torch.float32)
    criterion = FocalLoss(weight=cw, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=LR*0.01)

    best_map, best_preds, best_state, best_ep = -1, None, None, 0
    patience, no_improve = 30, 0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        for xb, mb, yb in train_ld:
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()

        # Validate every 5 epochs
        if (epoch+1) % 5 == 0 or epoch == EPOCHS-1:
            model.eval()
            vp = []
            with torch.no_grad():
                for xb, mb, yb in val_ld:
                    vp.append(F.softmax(model(xb, mb), dim=1).numpy())
            vp = np.concatenate(vp)
            vm, _ = compute_map(y_va, vp)
            if vm > best_map:
                best_map = vm; best_preds = vp.copy(); best_ep = epoch+1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if (epoch+1) % 20 == 0:
                print(f"        ep={epoch+1}: loss={loss_sum/len(train_ld):.4f} val={vm:.4f} best={best_map:.4f}@{best_ep}", flush=True)
            if no_improve >= patience:
                break

    # Test predictions
    model.load_state_dict(best_state)
    model.eval()
    test_ds = TrackDataset(X_te, M_te)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)
    tp = []
    with torch.no_grad():
        for batch in test_ld:
            xb, mb = batch[0], batch[1]
            tp.append(F.softmax(model(xb, mb), dim=1).numpy())
    return best_preds, np.concatenate(tp), best_map


# ══════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION WITH MULTI-SEED
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Training BirdTrackNetV3 ({N_FOLDS} folds x {N_SEEDS} seeds)...", flush=True)

class_w = np.sqrt(len(y) / (N_CLASSES * np.maximum(counts, 1).astype(float)))
class_w = np.clip(class_w, 0.3, 8.0)
print(f"  Class weights: {np.round(class_w, 2)}")

oof_all_seeds = np.zeros((N_SEEDS, len(y), N_CLASSES))
test_all_seeds = np.zeros((N_SEEDS, len(test_df), N_CLASSES))

for seed in range(N_SEEDS):
    print(f"\n  === Seed {seed+1}/{N_SEEDS} ===", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
    oof_s = np.zeros((len(y), N_CLASSES))
    test_s = np.zeros((len(test_df), N_CLASSES))

    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        t_fold = time.time()
        print(f"    Fold {fold+1}/{N_FOLDS} (train={len(tr)}, val={len(va)})", flush=True)

        va_preds, te_preds, best = train_fold(
            X_train[tr], masks_train[tr], y[tr],
            X_train[va], masks_train[va], y[va],
            X_test, masks_test, class_w, seed, fold,
        )
        oof_s[va] = va_preds
        test_s += te_preds / N_FOLDS
        print(f"    Fold {fold+1}: mAP={best:.4f} ({time.time()-t_fold:.0f}s)", flush=True)

    oof_all_seeds[seed] = oof_s
    test_all_seeds[seed] = test_s
    s, _ = compute_map(y, oof_s)
    print(f"  Seed {seed+1} overall: SKF={s:.4f}", flush=True)

# Average across seeds
oof_cnn = np.mean(oof_all_seeds, axis=0)
test_cnn = np.mean(test_all_seeds, axis=0)


# ══════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("  RESULTS")
print(f"{'='*80}")

MONTHS = [1, 4, 9, 10]

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

from src.postprocessing import renorm_rows
oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))

true_lomo(oof_e175, "E175 best")
true_lomo(oof_cnn, "CNN v3 (multi-seed)")

# Per-seed progression
for n in range(1, N_SEEDS+1):
    oof_n = np.mean(oof_all_seeds[:n], axis=0)
    true_lomo(oof_n, f"CNN {n}-seed average")

# Blends
print("\n  Blends:")
best_lomo, best_alpha = -1, 0
for alpha in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = renorm_rows((1-alpha) * oof_e175 + alpha * oof_cnn)
    _, lomo = true_lomo(blend, f"E175 + CNN_v3@{alpha}")
    if lomo > best_lomo:
        best_lomo = lomo; best_alpha = alpha

# Per-class analysis at best alpha
blend_best = renorm_rows((1-best_alpha) * oof_e175 + best_alpha * oof_cnn)
_, e175_pc = compute_map(y, oof_e175)
_, blend_pc = compute_map(y, blend_best)
print(f"\n  Per-class (E175 vs E175+CNN@{best_alpha}):")
print(f"  {'Class':<20s} {'E175':>6s} {'Blend':>6s} {'Delta':>7s}")
for cls in CLASSES:
    d = blend_pc[cls] - e175_pc[cls]
    flag = " +" if d > 0.005 else (" -" if d < -0.005 else "  ")
    print(f"  {cls:<20s} {e175_pc[cls]:>6.3f} {blend_pc[cls]:>6.3f} {d:>+7.3f}{flag}")

# Spearman correlation
from scipy.stats import spearmanr
print(f"\n  Spearman correlation CNN vs E175:")
for i, cls in enumerate(CLASSES):
    r, _ = spearmanr(oof_cnn[:, i], oof_e175[:, i])
    print(f"  {cls:<20s} r={r:.3f}")

# Save
np.save(ROOT / "oof_e182_cnn_v3.npy", oof_cnn)
np.save(ROOT / "test_e182_cnn_v3.npy", test_cnn)
blend_test = renorm_rows((1-best_alpha) * test_e175 + best_alpha * test_cnn)
from src.submission import save_submission
save_submission(blend_test, f"e182_cnn_v3_blend_{int(best_alpha*100)}", cv_map=compute_map(y, blend_best)[0])

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed/60:.1f} min")
print(f"{'='*80}")
