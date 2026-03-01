# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition

**AI Cup 2026 – Performance Track** (Kaggle, Team Epoch / TU Delft)
- **Task:** Classify bird species from radar tracks → 9 classes
- **Metric:** Macro-averaged mAP (mean Average Precision) over all 9 classes, sklearn implementation
- **Submission:** CSV with `track_id` + 9 probability columns (floats 0–1)
- **Deadline:** March 19, 2026

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

**Best LB: 0.59** — E75/E79/E96/E98/E99/E100/E101 all tie at 0.59. Hard ceiling.

- **E79**: 36-feat pruned tree ensemble (LGB=0.50, XGB=0.40, CB=0.10), SKF 0.7736, LB 0.59 raw.
- **E75/E96**: E50 base + NB PoE post-processing (GBIF priors + physics evidence), LB 0.59.
- Post-processing adds ~0.03 LB over raw on E50 base, but E79 matches without PP.

See `EXPERIMENTS.md` for full history (E01-E102+). See `RESEARCH.md` for paper references.

## Repo Structure (updated)

```
epoch-ai-cup/
├── src/                    # Shared library code — import, don't duplicate
│   ├── data.py             # Data loading, EWKB parsing, constants (CLASSES)
│   ├── features.py         # Feature extraction functions (core, rcs_fft, tabular)
│   ├── metrics.py          # compute_map(), print_results()
│   ├── submission.py       # save_submission() -> versioned file in submissions/
│   ├── postprocessing.py   # Canonical NB post-processing pipeline (priors + evidence + PoE)
│   └── validate.py         # IW-mAP validation with LB calibration (eval_pp)
├── experiments/            # One script per experiment, named e{NN}_{name}.py
│   ├── e103_template.py    # Template for new PP experiments (copy+edit)
│   ├── calibrate.py        # Build LB calibration from Kaggle submissions
│   └── analysis_*.py       # Analysis/EDA scripts (not experiments)
├── data/
│   ├── lb_calibration.csv  # IW-mAP -> LB mapping (6 fit points, use_in_fit column)
│   ├── best_features.txt   # 36 features from backward elimination
│   └── ...                 # Raw competition data (gitignored)
├── submissions/            # Versioned submission CSVs
├── EXPERIMENTS.md          # Experiment log — ALWAYS update after running
└── CLAUDE.md               # This file
```

## Validation System

**Problem**: LOMO correlates poorly with LB (~0.40) due to month shift. SKF is inflated. Neither predicts LB well.

**Solution**: `src/validate.py` — Importance-Weighted mAP with LB calibration.

```python
from src.validate import eval_pp

def my_pp(preds, test_df, test_months, train_df, y):
    # ... your post-processing ...
    return modified_preds

result = eval_pp(my_pp)           # full report
score  = result['calibrated_lb']  # predicted Kaggle LB (~0.01 accuracy)
```

**How it works**:
1. Temperature-scale OOF predictions (fix tree model overconfidence)
2. MLLS + GBIF priors estimate per-month class proportions on test
3. Importance-weight OOF mAP to match estimated test distribution
4. Linear calibration: `LB = 5.52 * IW-mAP - 3.17` (6 known LB points, RMSE=0.006)

**Key files**: `src/validate.py`, `src/postprocessing.py`, `data/lb_calibration.csv`, `experiments/calibrate.py`

**Limitation**: IW-mAP differentiates NB evidence strength (gamma) well, but NOT GBIF prior strength (OOF too confident for gating to fire). Calibration uses NB PP variants only.

## Post-Processing Pipeline (`src/postprocessing.py`)

Three stages, always in order:
1. **GBIF ratio priors** (`apply_gated_ratio_priors`) — month-specific label shift on unseen months
2. **Evidence extraction** — tabular NB channels: speed, alt_mid, alt_range, heading_R, rcs_ac1
3. **Gated PoE update** (`apply_nb_poe`) — multiply by P(u|c)^gamma, gated on uncertain rows

Template: `experiments/e103_template.py` — copy, rename, edit the starred sections.

### Next experiments to try:
1. Submit gamma sweep configs (g=0.20, g=0.30) to Kaggle to tighten LB calibration
2. Explore stronger evidence channels that maintain P(u|y) invariance across months
3. Better base model (architecture changes, not just PP)
