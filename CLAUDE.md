# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition

**AI Cup 2026 – Performance Track** (Kaggle, Team Epoch / TU Delft)
- **Task:** Classify bird species from radar tracks → 9 classes
- **Metric:** Macro-averaged mAP (mean Average Precision) over all 9 classes, sklearn implementation
- **Submission:** CSV with `track_id` + 9 probability columns (floats 0–1)
- **Deadline:** March 19, 2026

## Repo Structure

```
epoch-ai-cup/
├── src/                    # Shared library code — import, don't duplicate
│   ├── data.py             # Data loading, EWKB parsing, constants (CLASSES)
│   ├── features.py         # Feature extraction functions (core, rcs_fft, tabular)
│   ├── metrics.py          # compute_map(), print_results()
│   └── submission.py       # save_submission() → versioned file in submissions/
├── experiments/            # One script per experiment, named e{NN}_{name}.py
│   ├── e02_ensemble.py     # v2 LGB+XGB+CB ensemble (current best)
│   ├── e04_multiseed.py    # v4 multi-seed experiment
│   └── analysis_*.py       # Analysis/EDA scripts (not experiments)
├── notebooks/              # Jupyter notebooks for exploration
├── submissions/            # Versioned submission CSVs (auto-named by save_submission)
├── data/                   # Raw competition data (gitignored)
│   ├── train.csv           # 2601 labeled radar tracks
│   ├── test.csv            # 1872 tracks to predict
│   ├── sample_submission.csv
│   └── description.md
├── EXPERIMENTS.md           # Experiment log — ALWAYS update after running
├── RESEARCH.md              # Paper references and implementation plan
├── CLAUDE.md                # This file
└── Statement.md             # Competition statement
```

## Rules

### Experiments
1. **Every experiment gets a script** in `experiments/` named `e{NN}_{short_name}.py` where NN is the next number from EXPERIMENTS.md.
2. **Always use `src/` imports** — never copy-paste parsing, feature extraction, or metric code into experiments. If you need new features, add them to `src/features.py` first.
3. **Log every run** in `EXPERIMENTS.md` with: ID, date, name, CV mAP, all 9 per-class APs, and a short note. Even failed experiments get logged.
4. **Use `src/submission.py:save_submission()`** to save submissions. It auto-versions with timestamp + score in `submissions/`. It also writes `submission.csv` at root for quick Kaggle upload.
5. **Run from project root**: `python experiments/e05_whatever.py`. Scripts should use `src.data.ROOT` for paths.

### Features
6. **New feature sets go in `src/features.py`** as separate functions (e.g., `extract_wavelet_features()`). Experiments compose feature sets by calling multiple extractors.
7. **Don't add features blindly.** v3 proved 115 features < 75 features. New features must have a hypothesis for which class they help.

### Models
8. **5-fold Stratified CV is mandatory.** Never evaluate on train. Always report macro mAP.
9. **Class weights:** Use `is_unbalance=True` for LGB or inverse-frequency sample weights. Don't over-boost individual classes — it steals from neighbors.
10. **Ensemble:** Optimize weights on OOF predictions before applying to test.

### Code
11. **No print-heavy scripts.** Use `src/metrics.py:print_results()` for standardized output.
12. **Handle inf/nan:** Always run `.replace([np.inf, -np.inf], np.nan).fillna(0)` on feature matrices before training.

## Data Quick Reference

### Columns (both train & test)
- `track_id` — unique identifier
- `timestamp_start_radar_utc`, `timestamp_end_radar_utc` — track time range
- `trajectory` — EWKB hex: series of (Longitude, Latitude, Altitude_m, RCS_dBm2)
- `trajectory_time` — elapsed seconds per measurement (JSON list)
- `radar_bird_size` — categorical: Small bird, Medium, Large, Flock
- `airspeed` — average airspeed (m/s)
- `min_z`, `max_z` — altitude range relative to radar (m)

### Train-only (privileged)
- `bird_group` — **target label**
- `bird_species` — fine-grained species
- `observation_id`, `primary_observation_id`, `observer_position`, `observer_comment`, `n_birds_observed`

### Class Distribution
| Class | N | % | Best AP (v2) | Key signal |
|-------|---|---|-------------|------------|
| Gulls | 1503 | 57.8% | 0.956 | Dominant, easy |
| Songbirds | 483 | 18.6% | 0.640 | Bounding flight, low alt |
| Pigeons | 122 | 4.7% | 0.254 | 14:00 peak, Oct, short tracks |
| Waders | 120 | 4.6% | 0.816 | High alt variance, continuous flap |
| Birds of Prey | 108 | 4.2% | 0.885 | Slow (11.8 m/s), soaring |
| Clutter | 84 | 3.2% | 0.610 | RCS=-13.8 dB (birds: -24 to -30) |
| Geese | 83 | 3.2% | 0.728 | Large flocks, high alt, Oct |
| Ducks | 58 | 2.2% | 0.666 | Very low alt, overlaps Pigeons |
| Cormorants | 40 | 1.5% | 0.939 | Distinctive despite tiny sample |

## Current Status

**Best: E11 — stacking_4model — CV mAP 0.7396**

Heterogeneous stacking: 70% tree ensemble + 10% MiniRocket + 10% CNN + 10% SVM.
GPU enabled for all tree models (LGB device=gpu, XGB device=cuda, CB task_type=GPU).

See `EXPERIMENTS.md` for full history. See `RESEARCH.md` for paper references.

### Next experiments to try:
1. Augment CNN (window warping, slicing, jitter, mixup) to improve from 0.52 standalone
2. Improve MiniRocket (try Ridge classifier, more channels, longer sequences)
3. Per-class calibration / post-processing on stacked predictions
4. Pseudo-labeling: use E11 predictions on test as extra training data
