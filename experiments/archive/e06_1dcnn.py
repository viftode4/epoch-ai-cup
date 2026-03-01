"""E06: 1D-CNN on Raw Trajectory Time Series

Different paradigm: feed raw trajectory as a 6-channel time series
into a 1D-CNN. Can learn patterns we can't hand-engineer.
Architecture: 3x [Conv1D → BN → ReLU → MaxPool] → GAP → FC → 9-class.
Runs on GPU (CUDA).
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results
from src.submission import save_submission

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

SEQ_LEN = 64
N_CHANNELS = 6
N_CLASSES = len(CLASSES)
N_FOLDS = 5
EPOCHS = 120
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 15


# ── Model ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class BirdCNN(nn.Module):
    def __init__(self, in_channels=6, n_classes=9):
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


# ── Training ──────────────────────────────────────────────────────

def train_one_fold(X_train, y_train, X_val, y_val, class_weights, fold_idx):
    """Train one fold, return val predictions."""
    # Normalize per channel (fit on train)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                        pin_memory=True, num_workers=0)

    model = BirdCNN(N_CHANNELS, N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_preds = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
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

        if (epoch + 1) % 20 == 0:
            val_map, _ = compute_map(y_val, val_preds)
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_mAP={val_map:.4f}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    # Return best validation predictions and normalization params + model state
    return best_preds, mean, std, best_state


# ── Main ──────────────────────────────────────────────────────────

print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

print("Preparing train sequences...", flush=True)
X_all = prepare_sequences(train_df, seq_len=SEQ_LEN)
print("Preparing test sequences...", flush=True)
X_test_all = prepare_sequences(test_df, seq_len=SEQ_LEN)

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

class_counts = np.bincount(y, minlength=N_CLASSES)
class_weights = len(y) / (N_CLASSES * class_counts)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), N_CLASSES))
test_preds = np.zeros((len(X_test_all), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
    print(f"\n--- Fold {fold} ---", flush=True)
    fold_preds, mean, std, best_state = train_one_fold(
        X_all[tr_idx], y[tr_idx], X_all[va_idx], y[va_idx],
        class_weights, fold,
    )
    oof_preds[va_idx] = fold_preds
    fold_map, _ = compute_map(y[va_idx], fold_preds)
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

    # Predict test with this fold's model + normalization
    model = BirdCNN(N_CHANNELS, N_CLASSES).to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    X_test_norm = (X_test_all - mean) / (std + 1e-8)
    test_ds = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32))
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                         pin_memory=True, num_workers=0)
    fold_test_preds = []
    with torch.no_grad():
        for (xb,) in test_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            fold_test_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    test_preds += np.concatenate(fold_test_preds) / N_FOLDS

# ── Results ───────────────────────────────────────────────────────
final_map, final_per = compute_map(y, oof_preds)
print_results(final_map, final_per, "E06 1D-CNN")

np.save("oof_e06.npy", oof_preds)
np.save("test_e06.npy", test_preds)
print("Saved oof_e06.npy and test_e06.npy for blending", flush=True)

save_submission(test_preds, "e06_1dcnn", cv_map=final_map)
