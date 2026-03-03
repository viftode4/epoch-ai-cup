"""E105: Custom ROCKET-style sequence model on raw trajectories (new signal).

Why this is the right pivot
---------------------------
Post-processing around E50 is saturating at 0.59 on public LB because the PP family differs
on only O(10) test rows; with a ~24% public slice those differences are often invisible.

E102 proved that *high-dimensional generative evidence* can collapse under shift.
So we switch to a different inductive bias: random convolutional kernel features (ROCKET),
trained discriminatively with logistic regression. This learns directly from the `trajectory`
sequence without relying on month cues.

Implementation notes
--------------------
- No external time-series libraries (the real `aeon` toolkit doesn't support our Python env).
- We use `src.sequence.prepare_sequences()` to build (N, 8, T) sequences at fixed length.
- We implement a ROCKET-like transform:
    for each random kernel -> compute conv output -> 2 features: max, PPV
  Biases are fit from training data as a random quantile of conv outputs (ROCKET trick).
- Evaluate with Leave-One-Month-Out (LOMO) on train months for a sanity check.
- Train on full train and write:
    - `oof_e105.npy`, `test_e105.npy`
    - submissions: `e105_rocket_raw`, `e105_rocket_geo_e50` (new signal + strong baseline)
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.metrics import compute_map, print_results  # noqa: E402
from src.sequence import prepare_sequences  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

SEQ_LEN = 128
N_KERNELS = 1024  # -> 2048 features
SUBSAMPLE_BIAS = 256  # samples used to set each kernel's bias
RANDOM_SEED = 42


def renorm_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, None)
    return p / p.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class KernelSpec:
    channel: int
    length: int
    dilation: int
    weights: np.ndarray  # (length,)
    bias_q: float
    bias: float = 0.0


class RocketLike:
    """ROCKET-style random conv feature transform for (N,C,T) float arrays."""

    def __init__(self, n_kernels: int = 1024, random_state: int = 42):
        self.n_kernels = int(n_kernels)
        self.random_state = int(random_state)
        self.kernels: list[KernelSpec] = []

    def _make_kernel_specs(self, n_channels: int, T: int) -> list[KernelSpec]:
        rng = np.random.RandomState(self.random_state)
        lengths = np.array([7, 9, 11], dtype=int)
        specs: list[KernelSpec] = []
        for _ in range(self.n_kernels):
            ch = int(rng.randint(0, n_channels))
            L = int(rng.choice(lengths))
            # dilation so that effective receptive field fits
            max_d = max(1, (T - 1) // (L - 1))
            # sample dilation log-uniform-ish
            d = int(np.exp(rng.uniform(0.0, np.log(max_d + 1e-9))))
            d = max(1, min(d, max_d))
            w = rng.normal(0.0, 1.0, size=L).astype(np.float32)
            w = w - w.mean()  # mean center helps stability
            q = float(rng.uniform(0.0, 1.0))
            specs.append(KernelSpec(channel=ch, length=L, dilation=d, weights=w, bias_q=q, bias=0.0))
        return specs

    @staticmethod
    def _conv_windows(x: np.ndarray, length: int, dilation: int) -> np.ndarray:
        """Build dilated windows for conv: x (N,T) -> (N, steps, length)."""
        N, T = x.shape
        steps = T - (length - 1) * dilation
        if steps <= 0:
            return np.zeros((N, 1, length), dtype=x.dtype)
        idx0 = np.arange(steps)[:, None] + dilation * np.arange(length)[None, :]
        return x[:, idx0]  # (N, steps, length)

    def fit(self, X: np.ndarray) -> "RocketLike":
        # X: (N,C,T)
        N, C, T = X.shape
        specs = self._make_kernel_specs(C, T)
        rng = np.random.RandomState(self.random_state + 1)
        sub_idx = rng.choice(N, size=min(SUBSAMPLE_BIAS, N), replace=False)

        # Group kernels by (channel,length,dilation) to reuse window extraction.
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, ks in enumerate(specs):
            key = (ks.channel, ks.length, ks.dilation)
            groups.setdefault(key, []).append(i)

        biases = np.zeros(len(specs), dtype=np.float32)
        for (ch, L, d), idxs in groups.items():
            x = X[sub_idx, ch, :].astype(np.float32)  # (nsub,T)
            win = self._conv_windows(x, length=L, dilation=d)  # (nsub,steps,L)
            # compute conv for all kernels in group at once: (nsub,steps,k)
            W = np.stack([specs[j].weights for j in idxs], axis=1)  # (L,k)
            out = np.tensordot(win, W, axes=([2], [0]))  # (nsub,steps,k)
            # set each bias to q-quantile of its outputs
            for k_pos, j in enumerate(idxs):
                q = specs[j].bias_q
                biases[j] = float(np.quantile(out[:, :, k_pos].ravel(), q))

        self.kernels = [
            KernelSpec(
                channel=ks.channel,
                length=ks.length,
                dilation=ks.dilation,
                weights=ks.weights,
                bias_q=ks.bias_q,
                bias=float(biases[i]),
            )
            for i, ks in enumerate(specs)
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N, C, T = X.shape
        feats = np.zeros((N, 2 * len(self.kernels)), dtype=np.float32)

        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, ks in enumerate(self.kernels):
            key = (ks.channel, ks.length, ks.dilation)
            groups.setdefault(key, []).append(i)

        for (ch, L, d), idxs in groups.items():
            x = X[:, ch, :].astype(np.float32)
            win = self._conv_windows(x, length=L, dilation=d)  # (N,steps,L)
            W = np.stack([self.kernels[j].weights for j in idxs], axis=1)  # (L,k)
            out = np.tensordot(win, W, axes=([2], [0]))  # (N,steps,k)

            b = np.array([self.kernels[j].bias for j in idxs], dtype=np.float32)[None, None, :]
            z = out - b
            mx = z.max(axis=1)  # (N,k)
            ppv = (z > 0).mean(axis=1)  # (N,k)

            # write into feature matrix
            for k_pos, j in enumerate(idxs):
                feats[:, 2 * j] = mx[:, k_pos]
                feats[:, 2 * j + 1] = ppv[:, k_pos]

        return feats


def geo_mean(ps: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.clip(p, 1e-15, 1.0) for p in ps], axis=0)
    logp = np.log(stack)
    m = np.mean(logp, axis=0)
    p = np.exp(m)
    return renorm_rows(p)


def main() -> None:
    print("=" * 70, flush=True)
    print("E105 CUSTOM ROCKET SEQUENCE MODEL".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values

    print("\nPreparing sequences...", flush=True)
    X_all = prepare_sequences(train_df, seq_len=SEQ_LEN)
    X_test = prepare_sequences(test_df, seq_len=SEQ_LEN)
    print(f"Train seq: {X_all.shape}  Test seq: {X_test.shape}", flush=True)

    # Global normalization (fit on train): preserves absolute level cues (altitude, RCS scale, etc.)
    ch_mean = X_all.mean(axis=(0, 2), keepdims=True)
    ch_std = X_all.std(axis=(0, 2), keepdims=True) + 1e-6
    X_all = (X_all - ch_mean) / ch_std
    X_test = (X_test - ch_mean) / ch_std

    # --- LOMO sanity check ---
    unique_months = sorted(set(train_months))
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    for m in unique_months:
        tr_idx = np.where(train_months != m)[0]
        va_idx = np.where(train_months == m)[0]
        print(f"\nLOMO month {m}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

        rocket = RocketLike(n_kernels=N_KERNELS, random_state=RANDOM_SEED)
        rocket.fit(X_all[tr_idx])
        Xtr_f = rocket.transform(X_all[tr_idx])
        Xva_f = rocket.transform(X_all[va_idx])

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_f = scaler.fit_transform(Xtr_f)
        Xva_f = scaler.transform(Xva_f)

        clf = LogisticRegression(
            C=1.0,
            max_iter=4000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        clf.fit(Xtr_f, y[tr_idx])
        p_va = clf.predict_proba(Xva_f).astype(np.float32)
        oof[va_idx] = p_va

        fold_map, _ = compute_map(y[va_idx], p_va)
        ll = log_loss(y[va_idx], p_va, labels=np.arange(N_CLASSES))
        print(f"  mAP={fold_map:.4f}  logloss={ll:.4f}", flush=True)

    m, per = compute_map(y, oof)
    print_results(m, per, "E105 custom ROCKET (LOMO)")

    np.save("oof_e105.npy", oof)
    print("Saved oof_e105.npy", flush=True)

    # --- Train full + predict test ---
    print("\nTraining on full train and predicting test...", flush=True)
    rocket = RocketLike(n_kernels=N_KERNELS, random_state=RANDOM_SEED)
    rocket.fit(X_all)
    Xtr_f = rocket.transform(X_all)
    Xte_f = rocket.transform(X_test)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_f = scaler.fit_transform(Xtr_f)
    Xte_f = scaler.transform(Xte_f)

    clf = LogisticRegression(
        C=1.0,
        max_iter=4000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(Xtr_f, y)
    p_test = clf.predict_proba(Xte_f).astype(np.float32)
    p_test = renorm_rows(p_test)
    np.save("test_e105.npy", p_test)
    print("Saved test_e105.npy", flush=True)

    save_submission(p_test, "e105_rocket_raw", cv_map=m)

    # New-signal ensemble with strong baseline posterior
    base_e50 = renorm_rows(np.load(ROOT / "test_e50.npy").astype(np.float32))
    ens = geo_mean([p_test, base_e50])
    save_submission(ens, "e105_rocket_geo_e50", cv_map=None)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

