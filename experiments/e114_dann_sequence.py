"""E114: Domain-Adversarial Sequence Network (DANN) on Raw Trajectories.

Goal
----
Learn a *domain-invariant* sequence embedding that captures biomechanical rhythm (e.g.
bounding flight) while removing train↔test shift cues. Train with:
  - source: labeled train tracks
  - target: unlabeled test tracks
and a gradient-reversal domain head (binary: source vs target).

Key implementation details
--------------------------
- Uses `src.sequence.prepare_sequences()` to build (N, 8, T) channels including
  RCS/altitude derivatives (low-frequency wing motion / vertical speed cues).
- Drops `ALL_TEMPORAL` from tabular features.
- Runs on Apple MPS if available.
- Writes two submissions:
    - `e114_dann_raw_*`
    - `e114_dann_pp_heading_ac1_*` (optional PP to preserve our best unseen-month recipe)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.features import ALL_TEMPORAL, build_features  # noqa: E402
from src.sequence import prepare_sequences  # noqa: E402
from src.submission import save_submission  # noqa: E402

# Optional: keep our best PP as a final stage
from experiments.e96_nbalt_heading_ac1 import (  # noqa: E402
    BASE_ALPHA,
    GAMMA,
    TAU_NB,
    TAU_PRIOR,
    UNSEEN_MONTHS,
    apply_gated_ratio_priors,
    build_gbif_priors,
    build_nb_params,
    compute_log_p_u_given_c,
    extract_heading_ac1,
    apply_nb_poe,
    renorm_rows,
    top2_margin,
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42
SEQ_LEN = 128


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


def standardize_sequences(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel standardization using train stats only."""
    x_tr = x_tr.astype(np.float32)
    x_te = x_te.astype(np.float32)
    # mean/std over (N, T)
    mu = x_tr.mean(axis=(0, 2), keepdims=True)
    sig = x_tr.std(axis=(0, 2), keepdims=True)
    sig = np.where(sig > 1e-6, sig, 1.0)
    return (x_tr - mu) / sig, (x_te - mu) / sig


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class DANN(nn.Module):
    def __init__(self, seq_channels: int, tab_dim: int):
        super().__init__()

        self.seq_extractor = nn.Sequential(
            nn.Conv1d(seq_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.tab_extractor = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        emb_dim = 128 + 128

        self.class_head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, N_CLASSES),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor, alpha: float):
        seq_emb = self.seq_extractor(x_seq).squeeze(-1)
        tab_emb = self.tab_extractor(x_tab)
        emb = torch.cat([seq_emb, tab_emb], dim=1)

        y_logits = self.class_head(emb)
        d_logits = self.domain_head(grad_reverse(emb, alpha))
        return y_logits, d_logits


def build_tabular(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_sets = ["core", "tabular", "rcs_fft"]
    Xtr = build_features(train_df, feature_sets=feature_sets)
    Xte = build_features(test_df, feature_sets=feature_sets)

    # Drop temporal leakage features
    drop_cols = [c for c in ALL_TEMPORAL if c in Xtr.columns]
    if drop_cols:
        Xtr = Xtr.drop(columns=drop_cols, errors="ignore")
        Xte = Xte.drop(columns=drop_cols, errors="ignore")

    # Keep a conservative subset (avoid track-length features that behaved non-invariant as evidence)
    # We keep airspeed/alt/rcs stats etc, but drop duration/track_length explicitly if present.
    for c in ["duration", "track_length"]:
        if c in Xtr.columns:
            Xtr = Xtr.drop(columns=[c])
            Xte = Xte.drop(columns=[c])

    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # One-hot radar_bird_size if needed
    if "radar_bird_size" in Xtr.columns and Xtr["radar_bird_size"].dtype == "object":
        Xtr = pd.get_dummies(Xtr, columns=["radar_bird_size"])
        Xte = pd.get_dummies(Xte, columns=["radar_bird_size"])
        for c in Xtr.columns:
            if c not in Xte.columns:
                Xte[c] = 0
        Xte = Xte[Xtr.columns]

    scaler = StandardScaler()
    Xtr_np = scaler.fit_transform(Xtr.to_numpy(dtype=np.float32)).astype(np.float32)
    Xte_np = scaler.transform(Xte.to_numpy(dtype=np.float32)).astype(np.float32)
    return Xtr_np, Xte_np


def main() -> None:
    print("=" * 70, flush=True)
    print("E114 DOMAIN-ADVERSARIAL SEQUENCE NETWORK (DANN)".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    print("\nBuilding tabular features...", flush=True)
    Xtr_tab, Xte_tab = build_tabular(train_df, test_df)

    print("\nPreparing sequences...", flush=True)
    Xtr_seq = prepare_sequences(train_df, seq_len=SEQ_LEN)
    Xte_seq = prepare_sequences(test_df, seq_len=SEQ_LEN)

    Xtr_seq, Xte_seq = standardize_sequences(Xtr_seq, Xte_seq)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"\nUsing device: {device}", flush=True)

    model = DANN(seq_channels=Xtr_seq.shape[1], tab_dim=Xtr_tab.shape[1]).to(device)

    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    w_class = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    w_class_t = torch.tensor(w_class, dtype=torch.float32, device=device)

    crit_y = nn.CrossEntropyLoss(weight=w_class_t)
    crit_d = nn.CrossEntropyLoss()

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    src_ds = TensorDataset(
        torch.tensor(Xtr_seq, dtype=torch.float32),
        torch.tensor(Xtr_tab, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    tgt_ds = TensorDataset(
        torch.tensor(Xte_seq, dtype=torch.float32),
        torch.tensor(Xte_tab, dtype=torch.float32),
    )

    src_loader = DataLoader(src_ds, batch_size=64, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=64, shuffle=True, drop_last=True)

    epochs = 35
    for epoch in range(epochs):
        model.train()
        tgt_it = iter(tgt_loader)

        p = float(epoch) / max(1, epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        loss_y_sum = 0.0
        loss_d_sum = 0.0

        for x_seq_s, x_tab_s, y_s in src_loader:
            try:
                x_seq_t, x_tab_t = next(tgt_it)
            except StopIteration:
                tgt_it = iter(tgt_loader)
                x_seq_t, x_tab_t = next(tgt_it)

            x_seq_s = x_seq_s.to(device)
            x_tab_s = x_tab_s.to(device)
            y_s = y_s.to(device)
            x_seq_t = x_seq_t.to(device)
            x_tab_t = x_tab_t.to(device)

            opt.zero_grad()

            y_logits_s, d_logits_s = model(x_seq_s, x_tab_s, alpha=alpha)
            loss_y = crit_y(y_logits_s, y_s)

            d_y_s = torch.zeros(x_seq_s.shape[0], dtype=torch.long, device=device)
            loss_d_s = crit_d(d_logits_s, d_y_s)

            _, d_logits_t = model(x_seq_t, x_tab_t, alpha=alpha)
            d_y_t = torch.ones(x_seq_t.shape[0], dtype=torch.long, device=device)
            loss_d_t = crit_d(d_logits_t, d_y_t)

            loss_d = 0.5 * (loss_d_s + loss_d_t)

            loss = loss_y + loss_d
            loss.backward()
            opt.step()

            loss_y_sum += float(loss_y.item())
            loss_d_sum += float(loss_d.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:02d}/{epochs} | alpha={alpha:.3f} | "
                f"cls={loss_y_sum/len(src_loader):.4f} | dom={loss_d_sum/len(src_loader):.4f}",
                flush=True,
            )

    print("\nPredicting on test...", flush=True)
    model.eval()
    test_ds = TensorDataset(
        torch.tensor(Xte_seq, dtype=torch.float32),
        torch.tensor(Xte_tab, dtype=torch.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    preds = []
    with torch.no_grad():
        for x_seq_t, x_tab_t in test_loader:
            x_seq_t = x_seq_t.to(device)
            x_tab_t = x_tab_t.to(device)
            y_logits, _ = model(x_seq_t, x_tab_t, alpha=0.0)
            p = torch.softmax(y_logits, dim=1).cpu().numpy()
            preds.append(p)

    P_test = renorm_rows(np.vstack(preds))
    save_submission(P_test, "e114_dann_raw", cv_map=None)

    # Optional: apply our best unseen-month PP (priors + NB heading/ac1) to preserve gains.
    print("\nApplying E96 unseen-month PP...", flush=True)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_heading, train_ac1, train_ok = extract_heading_ac1(train_df)
    test_heading, test_ac1, test_ok = extract_heading_ac1(test_df)

    test_p0, _ = apply_gated_ratio_priors(P_test, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR)
    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)

    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, train_heading, train_ac1, train_ok, use_heading=True, use_ac1=True
    )
    loglike_test = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, mu, sig, test_heading, test_ac1, test_ok, use_heading=True, use_ac1=True
    )
    P_final = apply_nb_poe(test_p0, loglike_test, gamma=GAMMA, gate=gate)

    tag = f"e114_dann_pp_heading_ac1_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}"
    save_submission(P_final, tag, cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
