"""E118: Self-supervised SimCLR pretraining on trajectories -> tree model on embeddings.

Rationale
---------
Supervised deep models trained directly on our 2.6k labels overfit and fail on LB.
Instead, we:
  1) learn a trajectory embedding with *self-supervised contrastive learning*
     on ALL tracks (train+test) using augmentations that preserve flight style.
  2) freeze the encoder and train a LightGBM classifier on embeddings + tabular
     features (small-label regime friendly).
  3) optionally apply the proven E96 post-processing downstream (not included here
     to keep this experiment isolated; can be stacked later).

If this works, it introduces a new, robust trajectory-derived signal without
relying on supervised deep learning generalization.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.metrics import compute_map, print_results  # noqa: E402
from src.sequence import prepare_sequences  # noqa: E402
from src.submission import save_submission  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def standardize_channels(x: np.ndarray) -> np.ndarray:
    """Standardize per-channel over (N,T) using dataset stats."""
    x = x.astype(np.float32)
    mu = x.mean(axis=(0, 2), keepdims=True)
    sig = x.std(axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    return (x - mu) / sig


@dataclass(frozen=True)
class AugParams:
    jitter_std: float = 0.05
    scale_std: float = 0.10
    drop_prob: float = 0.10
    time_mask_frac: float = 0.12
    time_mask_count: int = 2


def augment(x: torch.Tensor, p: AugParams) -> torch.Tensor:
    """Trajectory-safe augmentations: channel scale, jitter, dropout, time masking."""
    # x: (B,C,T)
    b, c, t = x.shape
    out = x

    # Channel-wise scaling
    if p.scale_std > 0:
        s = torch.randn((b, c, 1), device=x.device) * p.scale_std + 1.0
        out = out * s

    # Additive jitter
    if p.jitter_std > 0:
        out = out + torch.randn_like(out) * p.jitter_std

    # Feature dropout (channel drop)
    if p.drop_prob > 0:
        drop = (torch.rand((b, c, 1), device=x.device) < p.drop_prob).float()
        out = out * (1.0 - drop)

    # Time masking (contiguous)
    if p.time_mask_frac > 0 and p.time_mask_count > 0:
        mlen = max(1, int(round(p.time_mask_frac * t)))
        for _ in range(p.time_mask_count):
            start = torch.randint(0, max(1, t - mlen), (b,), device=x.device)
            for i in range(b):
                out[i, :, start[i] : start[i] + mlen] = 0.0

    return out


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        z = self.proj(h)
        return z


def simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    """NT-Xent loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B,D)
    sim = torch.matmul(z, z.T) / temperature

    # mask self-similarity
    n = sim.shape[0]
    sim = sim - torch.eye(n, device=sim.device) * 1e9

    # positives are i<->i+B
    b = z1.shape[0]
    pos = torch.cat([torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)], dim=0)
    loss = F.cross_entropy(sim, pos)
    return loss


def build_tabular(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fast, conservative tabular features (no heavy per-track extraction)."""
    cols = ["airspeed", "min_z", "max_z", "radar_bird_size"]
    tr = train_df[cols].copy()
    te = test_df[cols].copy()

    tr["airspeed"] = pd.to_numeric(tr["airspeed"], errors="coerce").fillna(0.0)
    te["airspeed"] = pd.to_numeric(te["airspeed"], errors="coerce").fillna(0.0)
    tr["min_z"] = pd.to_numeric(tr["min_z"], errors="coerce").fillna(0.0)
    te["min_z"] = pd.to_numeric(te["min_z"], errors="coerce").fillna(0.0)
    tr["max_z"] = pd.to_numeric(tr["max_z"], errors="coerce").fillna(0.0)
    te["max_z"] = pd.to_numeric(te["max_z"], errors="coerce").fillna(0.0)

    tr = pd.get_dummies(tr, columns=["radar_bird_size"])
    te = pd.get_dummies(te, columns=["radar_bird_size"])
    for c in tr.columns:
        if c not in te.columns:
            te[c] = 0
    te = te[tr.columns]

    return tr.to_numpy(dtype=np.float32), te.to_numpy(dtype=np.float32)


def main() -> None:
    set_seed(42)
    print("=" * 78, flush=True)
    print("E118: SSL SIMCLR SEQUENCE EMBEDDINGS -> LGBM".center(78), flush=True)
    print("=" * 78, flush=True)

    train_df = load_train()
    test_df = load_test()

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"]).astype(int)

    # --- sequences (train+test) ---
    print("\nPreparing sequences...", flush=True)
    x_tr = prepare_sequences(train_df, seq_len=128)
    x_te = prepare_sequences(test_df, seq_len=128)

    x_all = np.concatenate([x_tr, x_te], axis=0)
    x_all = standardize_channels(x_all)

    # MPS can occasionally hard-crash in long loops (silent exit). Prefer CUDA when available,
    # otherwise use CPU for reliability.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    enc = ConvEncoder(in_ch=x_all.shape[1], emb_dim=96).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3, weight_decay=1e-4)

    ds = SeqDataset(x_all)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    aug_p = AugParams()

    # --- SSL pretraining ---
    epochs = 10
    for ep in range(epochs):
        enc.train()
        loss_sum = 0.0
        for xb in dl:
            xb = xb.to(device)
            x1 = augment(xb.clone(), aug_p)
            x2 = augment(xb.clone(), aug_p)
            z1 = enc(x1)
            z2 = enc(x2)
            loss = simclr_loss(z1, z2, temperature=0.15)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        if (ep + 1) % 1 == 0:
            print(f"  SSL epoch {ep+1:02d}/{epochs} | loss={loss_sum/len(dl):.4f}", flush=True)

    # --- Extract embeddings ---
    print("\nExtracting embeddings...", flush=True)
    enc.eval()
    with torch.no_grad():
        def _embed(x: np.ndarray) -> np.ndarray:
            out = []
            dl2 = DataLoader(SeqDataset(x), batch_size=256, shuffle=False)
            for xb in dl2:
                xb = xb.to(device)
                z = enc(xb)
                z = F.normalize(z, dim=1)
                out.append(z.detach().cpu().numpy())
            return np.vstack(out).astype(np.float32)

        z_tr = _embed(x_tr)
        z_te = _embed(x_te)

    # --- Tabular features ---
    print("\nBuilding tabular features...", flush=True)
    Xtr_tab, Xte_tab = build_tabular(train_df, test_df)

    Xtr = np.hstack([Xtr_tab, z_tr]).astype(np.float32)
    Xte = np.hstack([Xte_tab, z_te]).astype(np.float32)

    # --- Train LGBM with SKF ---
    print("\nTraining LGBM on (tabular + SSL embeddings)...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_classes = len(CLASSES)
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    test_pred = np.zeros((len(Xte), n_classes), dtype=np.float32)

    for k, (tr, va) in enumerate(skf.split(Xtr, y)):
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=1400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=60,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=0.4,
            class_weight="balanced",
            random_state=42 + k,
            verbose=-1,
            n_jobs=-1,
        )
        clf.fit(Xtr[tr], y[tr], eval_set=[(Xtr[va], y[va])])
        oof[va] = clf.predict_proba(Xtr[va]).astype(np.float32)
        test_pred += clf.predict_proba(Xte).astype(np.float32) / 5.0
        m, _ = compute_map(y[va], oof[va])
        print(f"  Fold {k+1}/5 mAP: {m:.4f}", flush=True)

    m_all, per = compute_map(y, oof)
    print_results(m_all, per, label="OOF (SKF) on tabular+SSL embeddings")

    test_pred = renorm_rows(test_pred)
    save_submission(test_pred, "e118_ssl_simclr_lgbm", cv_map=float(m_all))
    np.save(ROOT / "test_e118.npy", test_pred)
    np.save(ROOT / "oof_e118.npy", oof)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

