"""E43: Pretrained trajectory embeddings for Cormorant detection.

Approach:
1. Train 9-class models (CNN + Transformer) on raw radar trajectories
2. Extract 32-dim embeddings from the penultimate layer (OOF-style)
3. Use embeddings (alone and combined with tabular features) for Cormorant detection
4. Compare against tabular-only baseline

Input: 4-channel sequences [RCS, altitude, speed, bearing_change], fixed length 64.
Models are evaluated by 9-class SKF accuracy to select best architecture.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
CORM_IDX = CLASSES.index("Cormorants")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = len(CLASSES)
FIXED_LEN = 64

print("=" * 60, flush=True)
print("E43: PRETRAINED TRAJECTORY EMBEDDINGS", flush=True)
print(f"Device: {DEVICE}", flush=True)
print("=" * 60, flush=True)

# ======================================================================
# Load and prepare data
# ======================================================================
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
y_bin = (y == CORM_IDX).astype(int)

ts_train = pd.to_datetime(train_df["timestamp_start_radar_utc"])
months_train = ts_train.dt.month.values
unique_months = sorted(np.unique(months_train))

print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
print(f"  Classes: {N_CLASSES}, Cormorants: {y_bin.sum()}", flush=True)

# ======================================================================
# Extract raw 4-channel sequences for train + test
# ======================================================================
print("\nExtracting sequences...", flush=True)

def extract_channels(df):
    """Extract 4-channel time series from dataframe."""
    all_seqs = []
    for _, row in df.iterrows():
        pts = parse_ewkb_4d(row["trajectory"])
        times = parse_trajectory_time(row["trajectory_time"])

        rcs = np.array([p[3] for p in pts])
        alt = np.array([p[2] for p in pts])
        lats = np.array([p[1] for p in pts])
        lons = np.array([p[0] for p in pts])

        # Speed
        if len(pts) > 1 and len(times) > 1:
            dt = np.diff(times)
            dt[dt == 0] = 1e-6
            dx = np.diff(lons) * 67000
            dy = np.diff(lats) * 111000
            dalt = np.diff(alt)
            dist = np.sqrt(dx**2 + dy**2 + dalt**2)
            spd = dist / dt
            spd = np.concatenate([[spd[0]], spd])
        else:
            spd = np.array([row["airspeed"]] * len(pts))

        # Bearing change
        if len(pts) > 2:
            bearings = np.arctan2(np.diff(lats) * 111000, np.diff(lons) * 67000)
            bc = np.diff(bearings)
            bc = (bc + np.pi) % (2 * np.pi) - np.pi
            bc = np.concatenate([[0], [0], bc])
        else:
            bc = np.zeros(len(pts))

        all_seqs.append(np.column_stack([rcs, alt, spd, bc]))
    return all_seqs

train_seqs = extract_channels(train_df)
test_seqs = extract_channels(test_df)
print(f"  Train sequences: {len(train_seqs)}", flush=True)
print(f"  Test sequences: {len(test_seqs)}", flush=True)

# Resample to fixed length
def resample(seq, target_len=FIXED_LEN):
    n = len(seq)
    if n == target_len:
        return seq.copy()
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, seq.shape[1]))
    for c in range(seq.shape[1]):
        result[:, c] = np.interp(x_new, x_old, seq[:, c])
    return result

X_train_seq = np.array([resample(s) for s in train_seqs])  # (2601, 64, 4)
X_test_seq = np.array([resample(s) for s in test_seqs])    # (1872, 64, 4)

# Global normalization
all_seq = np.vstack([X_train_seq.reshape(-1, 4), X_test_seq.reshape(-1, 4)])
ch_means = all_seq.mean(axis=0)
ch_stds = all_seq.std(axis=0)
ch_stds[ch_stds < 1e-8] = 1.0

X_train_seq = (X_train_seq - ch_means) / ch_stds
X_test_seq = (X_test_seq - ch_means) / ch_stds
print(f"  Shapes: train {X_train_seq.shape}, test {X_test_seq.shape}", flush=True)

# ======================================================================
# Dataset
# ======================================================================
class SeqDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (seq_len, n_channels)
        if self.augment:
            if np.random.rand() < 0.7:
                x = x + torch.randn_like(x) * 0.05
            if np.random.rand() < 0.3:
                scale = torch.FloatTensor(x.shape[1]).uniform_(0.9, 1.1)
                x = x * scale
            if np.random.rand() < 0.3:
                x = x.flip(0)
        x = x.permute(1, 0)  # (channels, seq_len)
        if self.y is not None:
            return x, self.y[idx]
        return x

# ======================================================================
# Models
# ======================================================================
class CNN1D(nn.Module):
    def __init__(self, in_ch=4, emb_dim=32, n_layers=3, n_classes=9, dropout=0.3):
        super().__init__()
        layers = []
        ch = in_ch
        kernels = [7, 5, 3][:n_layers]
        for i, k in enumerate(kernels):
            out_ch = emb_dim
            layers.append(nn.Conv1d(ch, out_ch, kernel_size=k, padding=k//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if i < n_layers - 1:
                layers.append(nn.MaxPool1d(2))
            ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(emb_dim, n_classes)

    def get_embedding(self, x):
        x = self.backbone(x)
        x = x.mean(dim=2)  # global avg pool
        return x

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.head(self.dropout(emb))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, in_ch=4, emb_dim=32, n_heads=4, n_layers=1,
                 ff_dim=64, n_classes=9, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(emb_dim, n_classes)

    def get_embedding(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global avg pool
        return x

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.head(self.dropout(emb))

# ======================================================================
# Effective number class weights
# ======================================================================
def effective_number_weights(labels, n_classes, beta=0.999):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    eff[eff == 0] = 1.0
    weights = 1.0 / eff
    weights = weights / weights.sum() * n_classes
    return torch.FloatTensor(weights)

class_weights = effective_number_weights(y, N_CLASSES).to(DEVICE)
print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights.cpu().numpy()]}", flush=True)

# ======================================================================
# Training function
# ======================================================================
def train_model(model, train_idx, val_idx, epochs=150, lr=1e-3, patience=20, seed=42):
    """Train 9-class model and return OOF embeddings + predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SeqDataset(X_train_seq[train_idx], y[train_idx], augment=True)
    val_ds = SeqDataset(X_train_seq[val_idx], y[val_idx], augment=False)

    # Weighted sampler for class imbalance
    sample_weights = class_weights.cpu()[torch.LongTensor(y[train_idx])].numpy()
    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(yb.numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Load best and extract embeddings + predictions
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()

    embeddings = []
    probas = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            emb = model.get_embedding(xb)
            logits = model.head(emb)
            embeddings.append(emb.cpu().numpy())
            probas.append(F.softmax(logits, dim=1).cpu().numpy())

    embeddings = np.concatenate(embeddings)
    probas = np.concatenate(probas)
    return best_acc, embeddings, probas

# ======================================================================
# Extract test embeddings
# ======================================================================
def extract_test_embeddings(model, state_dict):
    """Extract embeddings for test set from a trained model."""
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    test_ds = SeqDataset(X_test_seq)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    embeddings = []
    probas = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE)
            emb = model.get_embedding(xb)
            logits = model.head(emb)
            embeddings.append(emb.cpu().numpy())
            probas.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(probas)

# ======================================================================
# Architecture configs to try
# ======================================================================
configs = {
    "CNN-d32-L3-do0.3": lambda: CNN1D(emb_dim=32, n_layers=3, dropout=0.3),
    "CNN-d32-L2-do0.3": lambda: CNN1D(emb_dim=32, n_layers=2, dropout=0.3),
    "CNN-d64-L3-do0.4": lambda: CNN1D(emb_dim=64, n_layers=3, dropout=0.4),
    "CNN-d16-L3-do0.2": lambda: CNN1D(emb_dim=16, n_layers=3, dropout=0.2),
    "TF-d32-L1-h4":     lambda: TransformerModel(emb_dim=32, n_heads=4, n_layers=1, dropout=0.3),
    "TF-d32-L2-h4":     lambda: TransformerModel(emb_dim=32, n_heads=4, n_layers=2, dropout=0.3),
    "TF-d64-L1-h4":     lambda: TransformerModel(emb_dim=64, n_heads=4, n_layers=1, dropout=0.4),
    "TF-d16-L1-h2":     lambda: TransformerModel(emb_dim=16, n_heads=2, n_layers=1, dropout=0.2),
}

# ======================================================================
# Phase 1: Train all configs, extract OOF embeddings
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PHASE 1: 9-CLASS MODEL COMPARISON", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_train_seq, y))

results = {}
N_SEEDS = 3

for config_name, model_fn in configs.items():
    model_sample = model_fn()
    emb_dim = model_sample.head.in_features
    n_params = sum(p.numel() for p in model_sample.parameters())
    print(f"\n  {config_name} ({n_params} params, emb_dim={emb_dim}):", flush=True)

    # OOF embeddings and predictions across seeds
    all_seed_embs = []
    all_seed_probas = []
    all_seed_accs = []

    for seed in range(N_SEEDS):
        oof_emb = np.zeros((len(y), emb_dim))
        oof_proba = np.zeros((len(y), N_CLASSES))
        fold_accs = []

        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            model = model_fn()
            acc, emb, proba = train_model(model, tr_idx, va_idx, seed=seed*100+fold_i)
            oof_emb[va_idx] = emb
            oof_proba[va_idx] = proba
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        all_seed_accs.append(mean_acc)
        all_seed_embs.append(oof_emb)
        all_seed_probas.append(oof_proba)

    # Average across seeds
    avg_emb = np.mean(all_seed_embs, axis=0)
    avg_proba = np.mean(all_seed_probas, axis=0)
    avg_acc = np.mean(all_seed_accs)

    # 9-class mAP from softmax predictions
    map_score, per_class = compute_map(y, avg_proba)

    # Cormorant AP from softmax P(Cormorant)
    corm_ap = average_precision_score(y_bin, avg_proba[:, CORM_IDX])

    print(f"    9-class acc: {avg_acc:.4f}, mAP: {map_score:.4f}, "
          f"Corm AP (from softmax): {corm_ap:.4f}", flush=True)

    results[config_name] = {
        "acc": avg_acc, "map": map_score, "corm_ap": corm_ap,
        "emb": avg_emb, "proba": avg_proba, "emb_dim": emb_dim,
        "model_fn": model_fn,
    }

# Rank by 9-class mAP
print("\n  RANKING (by 9-class mAP):", flush=True)
ranked = sorted(results.items(), key=lambda x: -x[1]["map"])
for i, (name, r) in enumerate(ranked):
    marker = " <-- BEST" if i == 0 else ""
    print(f"    {i+1}. {name}: acc={r['acc']:.4f}, mAP={r['map']:.4f}, "
          f"Corm={r['corm_ap']:.4f}{marker}", flush=True)

best_name = ranked[0][0]
best_result = ranked[0][1]

# ======================================================================
# Phase 2: Use best embeddings for Cormorant binary detection
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PHASE 2: CORMORANT DETECTION WITH EMBEDDINGS", flush=True)
print(f"Using: {best_name} (emb_dim={best_result['emb_dim']})", flush=True)
print("=" * 60, flush=True)

from catboost import CatBoostClassifier

best_emb = best_result["emb"]

# Also build tabular features for comparison
print("\n  Building tabular features...", flush=True)
X_tab_train = build_features(train_df)
drop_cols = [c for c in ALL_TEMPORAL if c in X_tab_train.columns]
X_tab_train = X_tab_train.drop(columns=drop_cols)
X_tab_train = X_tab_train.replace([np.inf, -np.inf], np.nan).fillna(0)

# Top-50 by t-stat
corm_mask = y_bin == 1
t_stats = []
for col in X_tab_train.columns:
    t, p = ttest_ind(X_tab_train.loc[corm_mask, col], X_tab_train.loc[~corm_mask, col], equal_var=False)
    t_stats.append((col, abs(t)))
t_stats.sort(key=lambda x: -x[1])
top50 = [x[0] for x in t_stats[:50]]
X_tab50 = X_tab_train[top50].values

# Feature sets to compare
feature_sets = {
    "tabular_top50": X_tab50,
    "embedding_only": best_emb,
    "tab50+embedding": np.hstack([X_tab50, best_emb]),
}

# Also try with second-best model's embeddings concatenated
if len(ranked) > 1:
    second_name = ranked[1][0]
    second_emb = ranked[1][1]["emb"]
    feature_sets["tab50+emb_best+emb_2nd"] = np.hstack([X_tab50, best_emb, second_emb])

# Evaluate each feature set with SKF + LOMO
print(f"\n  {'Feature Set':<30s} {'SKF AP':>8s} {'LOMO AP':>8s}", flush=True)
print(f"  {'-'*30} {'-'*8} {'-'*8}", flush=True)

for fs_name, X_fs in feature_sets.items():
    # SKF
    oof_skf = np.full(len(y_bin), np.nan)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf5.split(X_fs, y_bin):
        n_pos = y_bin[tr_idx].sum()
        n_neg = len(tr_idx) - n_pos
        cb = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg/max(n_pos,1),
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X_fs[tr_idx], y_bin[tr_idx])
        oof_skf[va_idx] = cb.predict_proba(X_fs[va_idx])[:, 1]
    skf_ap = average_precision_score(y_bin, oof_skf)

    # LOMO
    oof_lomo = np.full(len(y_bin), np.nan)
    for held_month in unique_months:
        val_mask = months_train == held_month
        train_mask = ~val_mask
        if y_bin[val_mask].sum() == 0:
            continue
        n_pos = y_bin[train_mask].sum()
        n_neg = train_mask.sum() - n_pos
        cb = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg/max(n_pos,1),
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X_fs[train_mask], y_bin[train_mask])
        oof_lomo[val_mask] = cb.predict_proba(X_fs[val_mask])[:, 1]
    valid = ~np.isnan(oof_lomo)
    lomo_ap = average_precision_score(y_bin[valid], oof_lomo[valid]) if y_bin[valid].sum() > 0 else 0.0

    print(f"  {fs_name:<30s} {skf_ap:>8.4f} {lomo_ap:>8.4f}", flush=True)

# ======================================================================
# Phase 3: Full 9-class evaluation with embeddings as extra features
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PHASE 3: FULL 9-CLASS MODEL WITH EMBEDDING FEATURES", flush=True)
print("=" * 60, flush=True)

# Build full feature set (114 tabular + embeddings)
X_tab_full = X_tab_train.values
X_combined = np.hstack([X_tab_full, best_emb])

print(f"  Tabular: {X_tab_full.shape[1]} features", flush=True)
print(f"  Combined: {X_combined.shape[1]} features ({X_tab_full.shape[1]} tab + {best_emb.shape[1]} emb)", flush=True)

# 9-class evaluation with CatBoost
from catboost import CatBoostClassifier as CBC

for feat_name, X_feat in [("tabular_only", X_tab_full), ("tabular+embedding", X_combined)]:
    oof_proba = np.zeros((len(y), N_CLASSES))
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (tr_idx, va_idx) in enumerate(skf5.split(X_feat, y)):
        ew = effective_number_weights(y[tr_idx], N_CLASSES, beta=0.999).numpy()
        sample_w = ew[y[tr_idx]]

        cb = CBC(
            iterations=1000, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0, task_type="GPU",
        )
        cb.fit(X_feat[tr_idx], y[tr_idx], sample_weight=sample_w)
        oof_proba[va_idx] = cb.predict_proba(X_feat[va_idx])

    map_score, per_class = compute_map(y, oof_proba)
    corm_ap = per_class[CORM_IDX]
    print(f"\n  {feat_name}:", flush=True)
    print(f"    9-class mAP: {map_score:.4f}", flush=True)
    print(f"    Per-class AP:", flush=True)
    for ci, cls in enumerate(CLASSES):
        marker = " <--" if cls == "Cormorants" else ""
        print(f"      {cls:<18s}: {per_class[ci]:.4f}{marker}", flush=True)

# ======================================================================
# Phase 4: Train best model on ALL train data, extract test embeddings
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("PHASE 4: EXTRACT TEST EMBEDDINGS", flush=True)
print("=" * 60, flush=True)

# Train best config on full training set with multiple seeds
test_embs = []
for seed in range(N_SEEDS):
    model = best_result["model_fn"]()
    torch.manual_seed(seed * 1000)
    np.random.seed(seed * 1000)

    train_ds = SeqDataset(X_train_seq, y, augment=True)
    sample_weights = class_weights.cpu()[torch.LongTensor(y)].numpy()
    sampler = WeightedRandomSampler(sample_weights, len(y), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Extract test embeddings
    model.eval()
    test_ds = SeqDataset(X_test_seq)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    embs = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE)
            embs.append(model.get_embedding(xb).cpu().numpy())
    test_embs.append(np.concatenate(embs))

test_emb_avg = np.mean(test_embs, axis=0)
print(f"  Test embeddings: {test_emb_avg.shape}", flush=True)

# Save
np.save(ROOT / "oof_e43_emb.npy", best_emb)
np.save(ROOT / "test_e43_emb.npy", test_emb_avg)
np.save(ROOT / "oof_e43_proba.npy", best_result["proba"])
print(f"  Saved: oof_e43_emb.npy, test_e43_emb.npy, oof_e43_proba.npy", flush=True)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Best architecture: {best_name}", flush=True)
print(f"  9-class mAP (embeddings model): {best_result['map']:.4f}", flush=True)
print(f"  Cormorant AP (softmax): {best_result['corm_ap']:.4f}", flush=True)
print(f"  Embeddings shape: ({len(y)}, {best_result['emb_dim']})", flush=True)
print(f"\n  See Phase 2 & 3 tables above for detection results.", flush=True)
print("\nDone!", flush=True)
