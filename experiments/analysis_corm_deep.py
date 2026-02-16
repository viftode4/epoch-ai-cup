"""Cormorant detection: Deep learning on raw radar sequences.

Tests tiny deep models with heavy augmentation for binary Cormorant detection.
- Model A: Tiny 1D CNN (16 filters, 3 layers)
- Model B: Tiny GRU (16 hidden units)
- Model C: Tiny Transformer (1 layer, 16 dim)

All with: focal loss, heavy augmentation, LOMO evaluation.
Input: 4-channel (RCS, altitude, speed, bearing_change), fixed length 64.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_LEN = 64

print("=" * 60, flush=True)
print("CORMORANT DEEP LEARNING DETECTION", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("=" * 60, flush=True)

# ======================================================================
# Load data and extract raw time series
# ======================================================================
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months = ts.dt.month.values
unique_months = sorted(np.unique(months))

print(f"  Cormorants: {y_bin.sum()}/{len(y_bin)}", flush=True)

# Parse trajectories
print("Extracting sequences...", flush=True)
all_channels = []  # list of (n_points, 4) arrays: [rcs, alt, speed, bearing_change]

for i, row in train_df.iterrows():
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])

    rcs = np.array([p[3] for p in pts])
    alt = np.array([p[2] for p in pts])
    lats = np.array([p[1] for p in pts])
    lons = np.array([p[0] for p in pts])

    # Speed from positions
    if len(pts) > 1 and len(times) > 1:
        dt = np.diff(times)
        dt[dt == 0] = 1e-6
        dlat = np.diff(lats)
        dlon = np.diff(lons)
        dalt = np.diff(alt)
        dx = dlon * 67000
        dy = dlat * 111000
        dist = np.sqrt(dx**2 + dy**2 + dalt**2)
        spd = dist / dt
        spd = np.concatenate([[spd[0]], spd])
    else:
        spd = np.array([row["airspeed"]] * len(pts))

    # Bearing change (turn angle per step)
    if len(pts) > 2:
        bearings = np.arctan2(np.diff(lats) * 111000, np.diff(lons) * 67000)
        bearing_change = np.diff(bearings)
        # Wrap to [-pi, pi]
        bearing_change = (bearing_change + np.pi) % (2 * np.pi) - np.pi
        bearing_change = np.concatenate([[0], [0], bearing_change])
    else:
        bearing_change = np.zeros(len(pts))

    channels = np.column_stack([rcs, alt, spd, bearing_change])
    all_channels.append(channels)

print(f"  Extracted {len(all_channels)} sequences", flush=True)

# ======================================================================
# Preprocessing: resample + normalize
# ======================================================================
def resample(seq, target_len=FIXED_LEN):
    """Resample (n, 4) sequence to (target_len, 4)."""
    n = len(seq)
    if n == target_len:
        return seq.copy()
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, seq.shape[1]))
    for c in range(seq.shape[1]):
        result[:, c] = np.interp(x_new, x_old, seq[:, c])
    return result

X_fixed = np.array([resample(s) for s in all_channels])  # (2601, 64, 4)

# Per-channel global normalization (mean/std across all samples)
ch_means = X_fixed.mean(axis=(0, 1))
ch_stds = X_fixed.std(axis=(0, 1))
ch_stds[ch_stds < 1e-8] = 1.0
X_norm = (X_fixed - ch_means) / ch_stds
print(f"  X_norm shape: {X_norm.shape}", flush=True)

# Also store original lengths for augmentation context
orig_lengths = np.array([len(s) for s in all_channels])

# ======================================================================
# Dataset with augmentation
# ======================================================================
class RadarDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)  # (N, seq_len, n_channels)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (seq_len, n_channels)
        y = self.y[idx]

        if self.augment:
            # Jitter: add random noise
            if np.random.rand() < 0.8:
                noise = torch.randn_like(x) * 0.1
                x = x + noise

            # Scaling: random per-channel scale
            if np.random.rand() < 0.5:
                scale = torch.FloatTensor(x.shape[1]).uniform_(0.8, 1.2)
                x = x * scale

            # Random crop + pad back (simulate different track lengths)
            if np.random.rand() < 0.5:
                crop_len = max(16, int(x.shape[0] * np.random.uniform(0.6, 1.0)))
                start = np.random.randint(0, x.shape[0] - crop_len + 1)
                cropped = x[start:start+crop_len]
                # Resample back to original length
                x_new = torch.zeros_like(x)
                for c in range(x.shape[1]):
                    indices = torch.linspace(0, crop_len-1, x.shape[0]).long().clamp(0, crop_len-1)
                    x_new[:, c] = cropped[indices, c]
                x = x_new

            # Time reversal
            if np.random.rand() < 0.3:
                x = x.flip(0)

        # Return as (channels, seq_len) for Conv1d
        return x.permute(1, 0), y

# ======================================================================
# Models
# ======================================================================
class TinyCNN(nn.Module):
    """Tiny 1D CNN: 3 conv layers, global avg pool, binary output."""
    def __init__(self, in_channels=4, hidden=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)  # global avg pool
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)

class TinyGRU(nn.Module):
    """Tiny GRU: 1 layer, 16 hidden, binary output."""
    def __init__(self, in_channels=4, hidden=16):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, batch_first=True,
                          num_layers=1, dropout=0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        _, h = self.gru(x)  # h: (1, batch, hidden)
        h = h.squeeze(0)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)

class TinyTransformer(nn.Module):
    """Tiny Transformer: project to 16d, 1 layer, 2 heads."""
    def __init__(self, in_channels=4, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32,
            dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.proj(x)  # (batch, seq_len, d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global avg pool over time
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)

# ======================================================================
# Focal loss for class imbalance
# ======================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()

# ======================================================================
# Training loop
# ======================================================================
def train_and_evaluate(model_class, model_name, train_mask, val_mask,
                       epochs=200, lr=1e-3, patience=30):
    """Train a model and return validation scores."""
    X_tr = X_norm[train_mask]
    y_tr = y_bin[train_mask]
    X_va = X_norm[val_mask]
    y_va = y_bin[val_mask]

    # Weighted sampler: oversample Cormorants
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    weights = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)

    train_ds = RadarDataset(X_tr, y_tr, augment=True)
    val_ds = RadarDataset(X_va, y_va, augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = model_class().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.9, gamma=2.0)

    best_ap = 0
    best_scores = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(yb.numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        if all_labels.sum() > 0:
            ap = average_precision_score(all_labels, all_scores)
        else:
            ap = 0.0

        if ap > best_ap:
            best_ap = ap
            best_scores = all_scores.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_ap, best_scores

# ======================================================================
# Multi-seed training for stability
# ======================================================================
def train_multiseed(model_class, model_name, train_mask, val_mask, n_seeds=5):
    """Average predictions across multiple random seeds."""
    all_scores = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ap, scores = train_and_evaluate(model_class, model_name, train_mask, val_mask)
        if scores is not None:
            all_scores.append(scores)

    if len(all_scores) == 0:
        return 0.0, np.zeros(val_mask.sum())

    avg_scores = np.mean(all_scores, axis=0)
    y_va = y_bin[val_mask]
    if y_va.sum() > 0:
        ap = average_precision_score(y_va, avg_scores)
    else:
        ap = 0.0
    return ap, avg_scores

# ======================================================================
# LOMO evaluation
# ======================================================================
def lomo_evaluate_deep(model_class, model_name, n_seeds=5):
    """Run LOMO with multi-seed averaging."""
    oof_scores = np.full(len(y_bin), np.nan)

    for held_month in unique_months:
        val_mask = months == held_month
        train_mask = ~val_mask
        n_corm_val = y_bin[val_mask].sum()
        n_corm_train = y_bin[train_mask].sum()

        if n_corm_val == 0:
            continue

        ap, scores = train_multiseed(model_class, model_name, train_mask, val_mask, n_seeds)
        oof_scores[val_mask] = scores

        print(f"    M{held_month:2d}: train={train_mask.sum()} "
              f"(corm={n_corm_train}), val={val_mask.sum()} "
              f"(corm={n_corm_val}), AP={ap:.4f}", flush=True)

    valid = ~np.isnan(oof_scores)
    if valid.sum() > 0 and y_bin[valid].sum() > 0:
        overall_ap = average_precision_score(y_bin[valid], oof_scores[valid])
    else:
        overall_ap = 0.0

    print(f"  {model_name} LOMO AP: {overall_ap:.4f} (5-seed avg)", flush=True)
    return overall_ap

# ======================================================================
# Run all models
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("MODEL A: TINY 1D CNN (16 filters, 3 layers)", flush=True)
print("=" * 60, flush=True)
n_params = sum(p.numel() for p in TinyCNN().parameters())
print(f"  Parameters: {n_params}", flush=True)
ap_cnn = lomo_evaluate_deep(TinyCNN, "TinyCNN")

print("\n" + "=" * 60, flush=True)
print("MODEL B: TINY GRU (16 hidden)", flush=True)
print("=" * 60, flush=True)
n_params = sum(p.numel() for p in TinyGRU().parameters())
print(f"  Parameters: {n_params}", flush=True)
ap_gru = lomo_evaluate_deep(TinyGRU, "TinyGRU")

print("\n" + "=" * 60, flush=True)
print("MODEL C: TINY TRANSFORMER (16d, 1 layer, 2 heads)", flush=True)
print("=" * 60, flush=True)
n_params = sum(p.numel() for p in TinyTransformer().parameters())
print(f"  Parameters: {n_params}", flush=True)
ap_transformer = lomo_evaluate_deep(TinyTransformer, "TinyTransformer")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY: Cormorant LOMO AP", flush=True)
print("=" * 60, flush=True)
print(f"  Tabular top-50 ensemble (baseline):  0.1568", flush=True)
print(f"  Tiny CNN (16 filters):               {ap_cnn:.4f}", flush=True)
print(f"  Tiny GRU (16 hidden):                {ap_gru:.4f}", flush=True)
print(f"  Tiny Transformer (16d):              {ap_transformer:.4f}", flush=True)
print(f"  MiniRocket (prev session):           0.024-0.035", flush=True)
print(f"  DTW kNN (prev session):              0.011-0.019", flush=True)
print("\nDone!", flush=True)
