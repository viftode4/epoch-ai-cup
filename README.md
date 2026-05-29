# epoch-ai-cup

This repository contains two things:

1. **AI Cup 2026 competition solution**  -  Team Epoch's bird radar classification system, developed for the TNO/TU Delft AI Cup 2026 Performance Track. Classifies Robin Radar MAX tracks into 9 bird species using gradient-boosted trees, LambdaRank, TabPFN, and graph-based flock smoothing.

2. **Research OS Pilot**  -  A CLI-first applied-ML research operating system for technical research labs and solo power users, built on top of the competition codebase. Provides canonical `research-os` commands for running experiments, validating predictions, and managing memory.

---

# Part 1: AI Cup 2026 – Bird Radar Classification (Team Epoch / TU Delft)

**Competition:** AI Cup 2026 – Performance Track (Kaggle)  
**Task:** Classify bird species from Robin Radar MAX radar tracks into 9 classes  
**Metric:** Macro-averaged Mean Average Precision (mAP), sklearn implementation

## Submissions

| Submission | Public LB | Private LB | Notes |
|---|---|---|---|
| `e113_graph_smoothed_mega_geo5_thresh0.50` | **0.5999** | 0.5267 | Official competition submission |
| `e205_multi_restart_T09` | 0.5776 | **0.5453** | Best model weights (post-deadline) |

---

## Problem Description

Given 4D radar tracks (longitude, latitude, altitude, RCS) from a Robin Radar MAX system sampling at 1 Hz, classify each track into one of 9 bird categories:

| Class | Training Share |
|---|---|
| Gulls | 57.8% |
| Songbirds | 18.6% |
| Pigeons | 4.7% |
| Waders | 4.6% |
| Birds of Prey | 4.2% |
| Clutter | 3.2% |
| Geese | 3.2% |
| Ducks | 2.2% |
| Cormorants | 1.5% |

Training data: 2,601 labelled tracks across four months (January, April, September, October).  
Test set: February, May, September, October, December  -  including months with different species compositions.

The core challenge is a **month distribution shift**: unseen test months (Feb, May, Dec) have radically different species compositions that the model never observed during training.

---

## Repository Structure

```
epoch-ai-cup/
├── src/
│   ├── data.py             # Data loading, EWKB trajectory parsing, class constants
│   ├── features.py         # All feature extraction functions
│   ├── metrics.py          # compute_map(), print_results()
│   ├── submission.py       # save_submission() → versioned CSV in submissions/
│   ├── postprocessing.py   # 3-stage Naive Bayes post-processing pipeline
│   └── validate.py         # Importance-Weighted mAP with LB calibration
├── experiments/
│   ├── e205_e188_variations.py   # Best weights submission script
│   └── e205_extract_weights.py  # Extracts ensemble weights
├── data/
│   ├── best_features.txt   # 36 features from backward elimination (E79 base)
│   └── lb_calibration.csv  # IW-mAP → LB mapping
├── submissions/
│   ├── e205_multi_restart_T09.csv                          # Best model weights
│   └── e113_graph_smoothed_mega_geo5_thresh0.50_...csv    # Competition submission
├── FINAL_ARCHITECTURE.md   # Full architecture design rationale
└── EXPERIMENTS.md          # Full experiment log
```

---

## Architecture 1: Competition Submission (E113)

**Public LB 0.5999 | Private LB 0.5267**

E113 is a **graph-smoothed geometric mean ensemble** built on top of five independently post-processed models (E111 "Mega Geo5").

### Base: E111  -  Geometric Mean of 5 Post-Processed Models

Five models, each built on a 36-feature gradient-boosted tree ensemble (LightGBM 50% / XGBoost 40% / CatBoost 10%) and independently post-processed with different evidence channels, are combined via geometric mean in log-probability space:

- **E75**  -  altitude evidence (airspeed + alt_mid + alt_range)
- **E98**  -  flock evidence (RCS autocorrelation + flock size)
- **E100**  -  wind-compensated kinematic evidence
- **E101**  -  heading consistency evidence
- **E109**  -  targeted specialist corrections (BoP, Cormorants)

Each post-processing stage follows the same 3-stage Naive Bayes pipeline:
1. **GBIF ratio priors**  -  month-specific label-shift correction for unseen months using GBIF ecological occurrence counts, gated on prediction uncertainty (top-2 margin < τ)
2. **Physics evidence**  -  diagonal Gaussian NB likelihoods over stable physical channels (airspeed, altitude, heading consistency R, RCS lag-1 autocorrelation)
3. **Gated Product-of-Experts**  -  `p ∝ p_base · P(evidence|class)^γ` (γ=0.10), applied only on uncertain rows in unseen months {2, 5, 12}

### Graph Flock Smoothing (E113)

Data analysis revealed 74% of test tracks start within 60 seconds of another track. Birds in the same flock should receive consistent predictions. A pairwise **HistGradientBoosting link predictor** is trained on the training set to predict P(same flock) using only season-invariant relative features (Δtime, Δlon, Δlat, Δspeed, Δaltitude, size match). Test tracks where P(same flock) > 0.50 are linked; connected components define flocks, and E111 predictions are averaged within each flock.

### Feature Set (36 features, backward-eliminated from 139)

Trajectory kinematics, RCS statistics, spatial context, and 7 weather/solar features (KNMI + solar ephemeris) used as month-ecological proxies.

---

## Architecture 2: Best Model Weights (E205)

**Public LB 0.5776 | Private LB 0.5453**

E205 is a **diverse 11-model ensemble** with weights optimized via multi-restart Nelder-Mead on out-of-fold predictions, followed by temperature sharpening (T=0.9).

### Model Pool (E188)

| Model | Weight | Type |
|---|---|---|
| e186_ovo | 56.8% | One-vs-One multiclass ranker |
| e185_tabpfn_relabel | 25.2% | TabPFN (in-context learning transformer) |
| e175_lgb | 9.4% | OvR LambdaRank DART (month-grouped queries) |
| e185_tabpfn_all | 3.4% | TabPFN (full training set) |
| e180_cnn | 2.3% | 1D CNN on raw trajectory sequences |
| e79 | 1.7% | Tree ensemble (LGB/XGB/CB) |
| others | ~1.2% | CatBoost, blend, and ranker variants |

OOF macro-mAP: **0.8724**

### Key Model Types

**OvR LambdaRank (E175):** Nine binary LightGBM rankers with `objective='lambdarank'`, `metric='map'`. Query groups are defined by training month, forcing per-month MAP optimisation (Group DRO for ranking). DART boosting (`drop_rate=0.15`) regularises against the small dataset.

**TabPFN (E185):** In-context learning transformer that uses the entire training set as context at inference time with no gradient updates. Provides fundamentally different uncertainty estimates from tree models.

**OvO ranker (E186):** One-vs-One decomposition into 36 binary classifiers (one per class pair), each benefiting from more balanced labels than the original 9-class problem.

**CNN (E180):** 1D convolutional network on padded raw trajectory sequences; captures temporal structure without hand-crafted feature engineering.

### Feature Set (E175 base)

~100 cross-month stability-selected features from 324 candidates:
- 86 hand-crafted kinematic + RCS features
- 90 log-path signatures (depth-3, 3 sliding windows) capturing cross-channel trajectory geometry
- 22 catch22 time-series statistics per channel
- 60 environmental features (KNMI weather, solar ephemeris, GBIF priors, physics interaction scores)

Cross-month stability selection scores each feature by its *minimum* permutation importance across four leave-one-month-out folds, directly penalising month-specific proxies.

### Ensemble & Sharpening

```python
# 10 random Nelder-Mead restarts, keep best OOF weights
for seed in range(10):
    w0 = rng.dirichlet(np.ones(N))
    res = minimize(neg_map, w0, method='Nelder-Mead', ...)

# Temperature sharpening T=0.9
blend = sum(w_i * pred_i for w_i, pred_i in zip(weights, preds))
sharp = blend ** (1.0 / 0.9)
sharp /= sharp.sum(axis=1, keepdims=True)
```

---

## Reproducibility

### Requirements

```bash
pip install lightgbm xgboost catboost scikit-learn numpy pandas shapely tabpfn
```

### Running E205 (best weights)

```bash
python experiments/e205_e188_variations.py
```

Requires pre-trained OOF and test `.npy` files for each base model in the repo root.

### Running E113 (competition submission)

E113 builds on top of the E111 geometric mean ensemble. See `experiments/e113_graph_flock_smoothing.py`.

---

## Validation

Custom **Importance-Weighted mAP** reweights OOF predictions to match the estimated test month distribution via MLLS + GBIF priors, then applies linear calibration `LB = 5.52 × IW-mAP − 3.17` (RMSE = 0.006 on 6 known LB points) for ~0.01 LB prediction accuracy.

---

## Team

**Team Epoch**  -  Delft University of Technology, Faculty of EEMCS  
Alexandra Carutasu, Alexandru Ojica, Vlad Iftode, Daniel Popovici, Albert Sandu  
AI Cup 2026  -  Performance Track finalist

---

# Part 2: Research OS Pilot

During the competition, the team used [research-copilot](https://github.com/viftode4/research-copilot) as support infrastructure - a terminal-first, AI-powered research loop with a TUI, autonomous workflow commands, and persistent state. It handled triage, experiment planning, review, and next-step decisions alongside the ML work, but was not part of the core ML pipeline.

This repo contains a lighter, competition-specific spin-off: a plain `research-os` CLI (`research_os/` package) that wraps the same experiment state in canonical commands tailored to this codebase. It does not replicate the full research-copilot feature set (no TUI, no autonomous loop, no Rust scaffold).

It is **not**:
- a generic agent platform
- a UI-first product
- a full rewrite of the legacy research repo

## Install

Use a clean Python 3.11+ environment and install the pilot surface in editable mode:

```bash
python -m pip install -e .
research-os --root . status
```

## Workspace model

The repo is split conceptually into:

- **canonical pilot surface**  -  `research-os` commands and `.pilot/`
- **legacy research surface**  -  the historical `src/` + `experiments/` workflow
- **quarantine/generated outputs**  -  `.pilot/outputs`, `.pilot/reports`, `.pilot/quarantine`

If a command writes to the repo root, that is a **legacy escape hatch**, not the pilot default.

## Canonical commands

```bash
research-os --root . init
research-os --root . status
research-os --root . doctor
research-os --root . run baseline-run --base-test test_e175_best.npy --output-label pilot-baseline
research-os --root . validate-compare-runs --base-oof oof_e175_best.npy --base-test test_e175_best.npy
research-os --root . report-and-memory-update --experiment-id E900 --name pilot --cv-map 0.7000 --note "pilot note"
research-os --root . memory inspect --kind all
```

## Workflow contract

All 3 canonical workflows emit the same top-level result shape:
- `workflow`
- `spec`
- `artifacts`
- `outputs`
- `metrics`
- `decision`
- `memory_updates`
- `summary`
- `warnings`
- `generated_at`

## Artifact validation

`baseline-run` and `validate-compare-runs` validate prediction artifacts before loading them. The pilot surface:
- detects Git LFS pointer files explicitly
- records artifact metadata (path, shape, dtype, validity)
- fails with a readable pilot-specific error instead of exposing raw NumPy loader noise

## Recommendation policy

`validate-compare-runs` normalizes the raw validation output into one of four states:
- `submit`
- `safe-trial`
- `review`
- `reject`

These states are driven by a documented threshold policy (`loop-v1`) using estimated delta, shared-month safety, and prediction-shift magnitude.

## Memory contract

Each memory file has a distinct role:
- `EXPERIMENTS.md`  -  append-only run ledger
- `TRACKER.md`  -  mutable status / priority board
- `RESEARCH.md`  -  curated findings and synthesis
- `FINAL_ARCHITECTURE.md`  -  durable architecture snapshot
- `CLAUDE.md`  -  operator guide

External `~/.claude/.../memory/*` files are out of scope for v1 unless explicitly bridged later.

## Legacy caveats

Legacy scripts may still:
- use repo-local paths
- inject `sys.path`
- write `submission.csv` at repo root
- depend on cached `.npy/.pkl` artifacts

Those behaviors remain available only through the **legacy bridge** and are not the canonical pilot path.
