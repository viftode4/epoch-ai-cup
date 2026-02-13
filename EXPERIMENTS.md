# Experiment Log

Track every experiment run. Add a row when you run something, even if it fails.

## Results Table

| ID | Date | Name | CV mAP | Clutter | Cormorants | Pigeons | Ducks | Geese | Gulls | BoP | Waders | Songbirds | Notes |
|----|------|------|--------|---------|------------|---------|-------|-------|-------|-----|--------|-----------|-------|
| E01 | 2026-02-13 | v1_baseline | 0.7030 | 0.582 | 0.941 | 0.305 | 0.605 | 0.665 | 0.957 | 0.880 | 0.810 | 0.583 | LGB only, 40 features |
| E02 | 2026-02-13 | v2_ensemble | **0.7214** | 0.610 | 0.939 | 0.254 | 0.666 | 0.728 | 0.956 | 0.885 | 0.816 | 0.640 | LGB+XGB+CB, 75 feats, class weights. **Current best.** |
| E03 | 2026-02-13 | v3_targeted | 0.7213 | 0.620 | 0.940 | 0.253 | 0.648 | 0.731 | 0.957 | 0.882 | 0.825 | 0.635 | 115 features — too many, diluted signal |
| E04 | 2026-02-13 | v4_multiseed | 0.7197 | 0.615 | 0.940 | 0.273 | 0.621 | 0.713 | 0.956 | 0.876 | 0.811 | 0.672 | 5-seed avg, extra Pigeon weight. Pigeons+Songbirds up, Ducks down |

## Key Learnings

- More features ≠ better (v3 proved this). Be selective.
- Boosting one minority class (Pigeons) steals from its neighbor (Ducks).
- Multi-seed averaging stabilizes but doesn't improve mAP if the features are the same.
- The ensemble weight search adds ~1-2% over any single model.
- RCS is the strongest signal for Clutter. Time-of-day is the strongest for Pigeons.
