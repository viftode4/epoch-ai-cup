# Design: 1D-CNN on Raw Trajectory + GBM Ensemble Blend (E05)

**Date:** 2026-02-13
**Status:** Approved
**Target:** Improve CV mAP beyond 0.7214 by adding a sequence-aware neural model to the ensemble

---

## Problem

The current LGB+XGB+CB ensemble (E02, mAP=0.7214) relies entirely on hand-crafted tabular statistics that lose temporal structure. The four weakest classes all have distinctive *temporal patterns* that aggregate statistics cannot capture:

| Class | AP | Key temporal pattern missed by tabular |
|-------|-----|----------------------------------------|
| Pigeons | 0.254 | Short tracks, rapid continuous flapping rhythm |
| Clutter | 0.610 | Erratic RCS sequence with no coherent trajectory |
| Ducks | 0.666 | Consistently very-low altitude across entire track |
| Songbirds | 0.640 | Bounding flight: periodic altitude oscillations + RCS dips |

---

## Solution: 1D-CNN on Raw Trajectory Sequences

Add a PyTorch 1D-CNN trained on raw trajectory sequences as a 4th model, then blend its OOF predictions with the existing GBM trio via optimized weights.

---

## Section 1: Sequence Preprocessing

**Fixed sequence length:** N=64 steps.

- Tracks shorter than 64: linear interpolation to 64 steps via `scipy.interpolate.interp1d` along time axis
- Tracks longer than 64: uniform subsampling (64 evenly-spaced indices)

**6 channels per step:**
1. `alt` — altitude (z-score normalized per-track)
2. `rcs` — radar cross section (z-score normalized per-track)
3. `speed` — instantaneous speed m/s
4. `bearing_change` — turning angle radians [-π, π]
5. `lat_delta` — latitude difference between consecutive points
6. `lon_delta` — longitude difference between consecutive points

**Output tensor shape:** `(B, 6, 64)` — channels-first for PyTorch `Conv1d`.

**New functions in `src/features.py`:**
- `extract_sequence(hex_str, traj_time_str, n_steps=64) -> np.ndarray` shape `(6, 64)`
- `build_sequences(df, n_steps=64) -> np.ndarray` shape `(N, 6, 64)`

---

## Section 2: Model Architecture

3-block 1D-CNN with residual connections (InceptionTime / FCN-style).

```
Input: (B, 6, 64)

ConvBlock1: Conv1d(6→64, k=7, pad=3) → BN → ReLU → Conv1d(64→64, k=7, pad=3) → BN → ReLU
  + residual skip: Conv1d(6→64, k=1)

ConvBlock2: Conv1d(64→128, k=5, pad=2) → BN → ReLU → Conv1d(128→128, k=5, pad=2) → BN → ReLU
  + residual skip: Conv1d(64→128, k=1)

ConvBlock3: Conv1d(128→256, k=3, pad=1) → BN → ReLU → Conv1d(256→256, k=3, pad=1) → BN → ReLU
  + residual skip: Conv1d(128→256, k=1)

GlobalAveragePooling → (B, 256)
Dropout(0.3)
Linear(256→128) → ReLU → Dropout(0.2)
Linear(128→9) → output logits
```

Increasing filter counts (64→128→256) and decreasing kernel sizes (7→5→3) follows the "Stronger Baseline" paper (Wang et al. 2019). Residual connections prevent vanishing gradients. GAP replaces Flatten to prevent overfitting on small dataset.

---

## Section 3: Training Strategy

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Loss | CrossEntropyLoss + class weights | Inverse-frequency weights, same as GBMs |
| Label smoothing | 0.1 | Prevents overconfidence on tiny classes |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 | Standard for tabular-scale networks |
| LR schedule | CosineAnnealingLR, T_max=100 | Smooth decay, no warm restarts needed |
| Epochs | 100 with early stopping (patience=15) | ~2-3 min/fold on GPU |
| Batch size | 64 | |
| Sampling | WeightedRandomSampler | Oversamples minority classes at batch level |
| CV | StratifiedKFold(n_splits=5, seed=42) | Same folds as GBM ensemble |

**No SMOTE** on sequences — SMOTE interpolates in feature space and would produce temporally incoherent synthetic tracks. Weighted sampler + weighted loss is the correct approach for sequence data.

---

## Section 4: Ensemble Blending

CNN OOF/test predictions are blended with GBM predictions via 4-model Nelder-Mead optimization:

```python
blend = w0*oof_lgb + w1*oof_xgb + w2*oof_cb + w3*oof_cnn
```

**Temperature scaling:** CNN logits are scaled by T=1.5 before softmax to correct neural net overconfidence before blending with better-calibrated GBMs.

The optimizer maximizes OOF mAP — if CNN adds no value on a class, its weight naturally goes to 0.

---

## Section 5: Codebase Structure

**Modified files:**
- `src/features.py` — add `extract_sequence()` and `build_sequences()`
- `requirements.txt` — add `torch`

**New files:**
- `experiments/e05_1dcnn.py` — full experiment script containing:
  - `BirdCNN` PyTorch model class
  - `BirdDataset` PyTorch Dataset
  - 5-fold CV loop (GBMs + CNN in same folds)
  - 4-model Nelder-Mead weight optimization
  - `print_results()` + `save_submission()` calls

**Manual step after run:** Log result to `EXPERIMENTS.md` per project rules.

---

## Expected Outcome

- CNN captures temporal patterns invisible to tabular features
- Target: lift Pigeons AP from 0.254, Songbirds from 0.640, push overall mAP > 0.74
- If CNN adds signal, optimal blend weight w3 > 0.15
