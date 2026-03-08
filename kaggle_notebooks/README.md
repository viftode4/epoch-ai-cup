# Kaggle Notebooks

## Setup

1. Create a Kaggle dataset called `epoch-src` containing:
   - `src/` folder (data.py, features.py, metrics.py, submission.py, sequence.py, postprocessing.py, validate.py)
   - `data/best_features.txt`
   - `data/train_weather.csv`, `data/test_weather.csv`
   - `data/train_solar.csv`, `data/test_solar.csv`

2. Add these as input datasets to your notebook:
   - `ai-cup-2026` (competition data)
   - `epoch-src` (your private dataset)

3. Run with GPU enabled.

## Notebooks
- `e156_kaggle.py` — E79 retrain with bearing fix (LGB+XGB+CB, ~30 min GPU)
- `e157_kaggle.py` — MultiRocket + LGB (~20 min GPU)
- `e158_kaggle.py` — 1D-CNN on raw padded sequences (~40 min GPU)
