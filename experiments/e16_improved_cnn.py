"""E16: Improved 1D-CNN with Training Tricks (T15 + T16 + T17 + augmentation)

Improves E06 CNN (0.5238) with:
  T15: Label smoothing (alpha=0.1)
  T16: SWA (stochastic weight averaging)
  T17: Snapshot ensembles (cosine annealing, save at cycle mins)
  + Data augmentation: jitter, scaling, window warping

Current E06 = 0.5238. Target: 0.60+ to meaningfully boost stacking.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

SEQ_LEN = 128  # longer than E06's 64
N_CHANNELS = 8  # all 8 channels (E06 used 6)
N_CLASSES = len(CLASSES)
N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
LABEL_SMOOTH = 0.1
N_SNAPSHOTS = 5  # save 5 snapshots per cosine cycle
SWA_START = 150  # start SWA in last 50 epochs


# ── Augmentation ─────────────────────────────────────────────────

def augment_batch(x, p=0.5):
    """Apply time series augmentations to a batch. x: (B, C, T)."""
    B, C, T = x.shape

    # Jittering (add small noise)
    if np.random.random() < p:
        noise = torch.randn_like(x) * 0.02
        x = x + noise

    # Scaling (multiply by random factor per channel)
    if np.random.random() < p:
        scale = 1.0 + torch.randn(B, C, 1, device=x.device) * 0.1
        x = x * scale

    # Window warping (stretch/compress random segment)
    if np.random.random() < 0.3:
        warp_size = int(T * 0.1)
        if warp_size > 2:
            start = np.random.randint(0, T - warp_size)
            warp_factor = np.random.choice([0.5, 2.0])
            segment = x[:, :, start:start + warp_size]
            new_len = max(2, int(warp_size * warp_factor))
            warped = F.interpolate(segment, size=new_len, mode='linear', align_corners=False)
            # Reconstruct by replacing segment
            new_x = torch.cat([
                x[:, :, :start],
                warped,
                x[:, :, start + warp_size:]
            ], dim=2)
            # Resize back to original length
            x = F.interpolate(new_x, size=T, mode='linear', align_corners=False)

    return x


# ── Model ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class ImprovedCNN(nn.Module):
    """Deeper CNN with multi-scale kernels and residual connections."""
    def __init__(self, in_channels=8, n_classes=9):
        super().__init__()
        # Multi-scale first layer (like InceptionTime)
        self.conv_small = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.conv_large = nn.Conv1d(in_channels, 32, kernel_size=15, padding=7)
        self.bn_first = nn.BatchNorm1d(96)
        self.pool_first = nn.MaxPool1d(2)

        # Deeper blocks
        self.block2 = ConvBlock(96, 128, kernel_size=7)
        self.block3 = ConvBlock(128, 128, kernel_size=5)
        self.block4 = ConvBlock(128, 256, kernel_size=3)

        self.dropout = nn.Dropout(0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # Multi-scale first layer
        h1 = self.conv_small(x)
        h2 = self.conv_med(x)
        h3 = self.conv_large(x)
        x = torch.cat([h1, h2, h3], dim=1)
        x = self.pool_first(torch.relu(self.bn_first(x)))

        x = self.dropout(self.block2(x))
        x = self.dropout(self.block3(x))
        x = self.dropout(self.block4(x))

        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# ── Label-smoothed cross entropy ─────────────────────────────────

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=1)

        # Weighted one-hot
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_prob).sum(dim=1)

        # Apply class weights
        if self.weight is not None:
            sample_w = self.weight[target]
            loss = loss * sample_w

        return loss.mean()


# ── Training with snapshots + SWA ────────────────────────────────

def train_one_fold(X_train, y_train, X_val, y_val, class_weights, fold_idx):
    """Train with augmentation, label smoothing, snapshot ensembles, and SWA."""
    # Normalize
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(DEVICE)
    val_labels = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # Class-balanced sampling
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
    train_dl = DataLoader(
        TensorDataset(train_tensor, train_labels),
        batch_size=BATCH_SIZE, sampler=sampler,
        pin_memory=True, num_workers=0,
    )

    model = ImprovedCNN(N_CHANNELS, N_CLASSES).to(DEVICE)
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = LabelSmoothCrossEntropy(smoothing=LABEL_SMOOTH, weight=cw_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # Cosine annealing with restarts for snapshot collection
    cycle_len = EPOCHS // N_SNAPSHOTS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cycle_len, T_mult=1
    )

    # SWA model
    swa_model = AveragedModel(model)

    snapshots = []
    best_val_map = 0
    best_preds = None

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb, p=0.5)  # augmentation
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ── SWA update ──
        if epoch >= SWA_START:
            swa_model.update_parameters(model)

        # ── Snapshot at cycle minimums ──
        if (epoch + 1) % cycle_len == 0:
            model.eval()
            with torch.no_grad():
                val_logits = []
                for i in range(0, len(val_tensor), BATCH_SIZE * 2):
                    batch = val_tensor[i:i + BATCH_SIZE * 2]
                    val_logits.append(model(batch))
                val_logits = torch.cat(val_logits)
                val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

            snap_map, _ = compute_map(y_val, val_probs)
            snapshots.append({
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "map": snap_map,
            })
            print(f"    Snapshot {len(snapshots)}/{N_SNAPSHOTS} at epoch {epoch+1}: mAP={snap_map:.4f}",
                  flush=True)

        # ── Periodic eval ──
        if (epoch + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = []
                for i in range(0, len(val_tensor), BATCH_SIZE * 2):
                    batch = val_tensor[i:i + BATCH_SIZE * 2]
                    val_logits.append(model(batch))
                val_logits = torch.cat(val_logits)
                val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_map, _ = compute_map(y_val, val_probs)
            print(f"    Epoch {epoch+1}: mAP={val_map:.4f}", flush=True)
            if val_map > best_val_map:
                best_val_map = val_map
                best_preds = val_probs.copy()

    # ── Collect all prediction methods ──
    results = {}

    # 1) Best single model (early-stopping equivalent)
    if best_preds is not None:
        results["best_single"] = best_preds

    # 2) Snapshot ensemble (average predictions from all snapshots)
    if len(snapshots) >= 2:
        snap_preds = np.zeros((len(y_val), N_CLASSES))
        for snap in snapshots:
            model.load_state_dict(snap["state"])
            model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                val_logits = []
                for i in range(0, len(val_tensor), BATCH_SIZE * 2):
                    batch = val_tensor[i:i + BATCH_SIZE * 2]
                    val_logits.append(model(batch))
                val_logits = torch.cat(val_logits)
                snap_preds += torch.softmax(val_logits, dim=1).cpu().numpy() / len(snapshots)
        results["snapshot_ensemble"] = snap_preds

    # 3) SWA model
    # Update BN stats
    torch.optim.swa_utils.update_bn(train_dl, swa_model, device=DEVICE)
    swa_model.eval()
    with torch.no_grad():
        val_logits = []
        for i in range(0, len(val_tensor), BATCH_SIZE * 2):
            batch = val_tensor[i:i + BATCH_SIZE * 2]
            val_logits.append(swa_model(batch))
        val_logits = torch.cat(val_logits)
        swa_preds = torch.softmax(val_logits, dim=1).cpu().numpy()
    results["swa"] = swa_preds

    # Print comparison
    for name, preds in results.items():
        m, _ = compute_map(y_val, preds)
        print(f"    {name}: mAP={m:.4f}", flush=True)

    # Pick best method
    best_method = max(results.items(), key=lambda x: compute_map(y_val, x[1])[0])
    best_method_name = best_method[0]
    best_method_preds = best_method[1]

    return best_method_preds, best_method_name, mean, std, snapshots, swa_model, model


# ── Main ─────────────────────────────────────────────────────────

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

print("Preparing sequences (8ch x 128)...", flush=True)
X_all = prepare_sequences(train_df, seq_len=SEQ_LEN)
X_test_all = prepare_sequences(test_df, seq_len=SEQ_LEN)
print(f"  Shape: {X_all.shape}", flush=True)

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

counts = np.bincount(y, minlength=N_CLASSES)
class_weights = len(y) / (N_CLASSES * counts)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), N_CLASSES))
test_preds = np.zeros((len(X_test_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
    print(f"\n{'='*40} Fold {fold} {'='*40}", flush=True)

    fold_preds, method, mean, std, snapshots, swa_model, model = train_one_fold(
        X_all[tr_idx], y[tr_idx], X_all[va_idx], y[va_idx],
        class_weights, fold,
    )
    oof_preds[va_idx] = fold_preds
    fold_map, _ = compute_map(y[va_idx], fold_preds)
    print(f"  Fold {fold} best ({method}): mAP={fold_map:.4f}", flush=True)

    # Predict test with snapshot ensemble (most robust)
    X_test_norm = (X_test_all - mean) / (std + 1e-8)
    test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(DEVICE)

    if snapshots:
        fold_test = np.zeros((len(X_test_all), N_CLASSES))
        for snap in snapshots:
            model.load_state_dict(snap["state"])
            model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                test_logits = []
                for i in range(0, len(test_tensor), BATCH_SIZE * 2):
                    batch = test_tensor[i:i + BATCH_SIZE * 2]
                    test_logits.append(model(batch))
                test_logits = torch.cat(test_logits)
                fold_test += torch.softmax(test_logits, dim=1).cpu().numpy() / len(snapshots)
        test_preds += fold_test / N_FOLDS
    else:
        # Fallback to SWA model
        swa_model.eval()
        with torch.no_grad():
            test_logits = []
            for i in range(0, len(test_tensor), BATCH_SIZE * 2):
                batch = test_tensor[i:i + BATCH_SIZE * 2]
                test_logits.append(swa_model(batch))
            test_logits = torch.cat(test_logits)
            test_preds += torch.softmax(test_logits, dim=1).cpu().numpy() / N_FOLDS


# ── Results ──────────────────────────────────────────────────────
final_map, final_per = compute_map(y, oof_preds)
print_results(final_map, final_per, "E16 Improved CNN")

print(f"\nE06 CNN baseline: 0.5238", flush=True)
print(f"E16 Improved CNN: {final_map:.4f} ({final_map - 0.5238:+.4f})", flush=True)

np.save(ROOT / "oof_e16.npy", oof_preds)
np.save(ROOT / "test_e16.npy", test_preds)
print("Saved oof_e16.npy and test_e16.npy", flush=True)

# Quick stacking check
print(f"\n{'='*60}", flush=True)
print("Stacking: replace E06 CNN with E16 in 4-model stack", flush=True)
print(f"{'='*60}", flush=True)

oof_e10 = np.load(ROOT / "oof_e10.npy")
oof_e08 = np.load(ROOT / "oof_e08.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")

# E11 original: 70% E10 + 10% E08 + 10% E06 + 10% E09
oof_replace = 0.70 * oof_e10 + 0.10 * oof_e08 + 0.10 * oof_preds + 0.10 * oof_e09
m_replace, _ = compute_map(y, oof_replace)
print(f"  Replace E06 at 10%: mAP={m_replace:.4f} (E11 was 0.7396, delta={m_replace - 0.7396:+.4f})",
      flush=True)

# Try higher weight for improved CNN
for w_cnn in [0.15, 0.20, 0.25]:
    remaining = 1.0 - w_cnn
    oof_s = remaining * (0.70/0.90 * oof_e10 + 0.10/0.90 * oof_e08 + 0.10/0.90 * oof_e09) + w_cnn * oof_preds
    m, _ = compute_map(y, oof_s)
    print(f"  CNN at {w_cnn:.0%}: mAP={m:.4f} ({m - 0.7396:+.4f} vs E11)", flush=True)

save_submission(test_preds, "e16_improved_cnn", cv_map=final_map)
print("\nDone!", flush=True)
