# Experiment Log

Track every experiment run. Add a row when you run something, even if it fails.

## Results Table

| ID | Date | Name | CV mAP | Clutter | Cormorants | Pigeons | Ducks | Geese | Gulls | BoP | Waders | Songbirds | Notes |
|----|------|------|--------|---------|------------|---------|-------|-------|-------|-----|--------|-----------|-------|
| E01 | 2026-02-13 | v1_baseline | 0.7030 | 0.582 | 0.941 | 0.305 | 0.605 | 0.665 | 0.957 | 0.880 | 0.810 | 0.583 | LGB only, 40 features |
| E02 | 2026-02-13 | v2_ensemble | **0.7214** | 0.610 | 0.939 | 0.254 | 0.666 | 0.728 | 0.956 | 0.885 | 0.816 | 0.640 | LGB+XGB+CB, 75 feats, class weights. **Current best.** |
| E03 | 2026-02-13 | v3_targeted | 0.7213 | 0.620 | 0.940 | 0.253 | 0.648 | 0.731 | 0.957 | 0.882 | 0.825 | 0.635 | 115 features — too many, diluted signal |
| E04 | 2026-02-13 | v4_multiseed | 0.7197 | 0.615 | 0.940 | 0.273 | 0.621 | 0.713 | 0.956 | 0.876 | 0.811 | 0.672 | 5-seed avg, extra Pigeon weight. Pigeons+Songbirds up, Ducks down |
| E05 | 2026-02-16 | 1dcnn_ensemble | 0.7152 | — | — | — | — | — | — | — | — | — | 1D-CNN+GBM blend. CNN overfits (mAP=0.15). **LB=0.26 — confirmed CV leakage.** |
| E06 | 2026-02-18 | fix_cv | 0.5452 | — | — | — | — | — | — | — | — | — | StratifiedGroupKFold honest CV baseline. **Per-class labels were wrong (LabelEncoder bug).** |
| E07 | 2026-02-18 | bio_features | 0.6188 | — | — | — | — | — | — | — | — | — | +sun elevation/azimuth, shape, flap/glide. **Per-class labels were wrong (LabelEncoder bug). Submission CSV scrambled.** |
| E08 | 2026-02-18 | time_freq | **0.6168** | 0.900 | 0.227 | 0.811 | 0.572 | 0.607 | 0.918 | 0.577 | 0.223 | 0.717 | +STFT spectrogram + RCS-size consistency. **Fixed LabelEncoder bug — per-class APs now correct. Submit this.** |

## Key Learnings

- **E01–E05 CV scores are inflated due to primary_observation_id leakage.** 43% of validation data shared an observation with training data. StratifiedKFold cannot be trusted for this dataset.
- **StratifiedGroupKFold with primary_observation_id** gives honest CV scores (~0.54 vs 0.72 leaky).
- More features ≠ better (v3 proved this). Be selective.
- Boosting one minority class (Pigeons) steals from its neighbor (Ducks).
- Multi-seed averaging stabilizes but doesn't improve mAP if the features are the same.
- The ensemble weight search adds ~1-2% over any single model.
- RCS is the strongest signal for Clutter. Time-of-day features are **harmful** (train/test temporal mismatch).
- Temporal features (month, hour indicators) exploit train-specific patterns: 32.7% of test comes from months absent in train.
- Adversarial validation AUC=0.70: moderate train/test distribution shift exists even after dropping temporal features.
- Top discriminative features (honest CV): rcs_q75, bearing_change_mean, avg_ground_speed, speed_median, airspeed.
- **Sun elevation and azimuth** are #1 and #2 features by LGB gain (E07/E08) — they capture biological activity cycles generalising across months.
- **LabelEncoder bug (E06/E07):** `LabelEncoder.fit(CLASSES)` sorts alphabetically, but CLASSES uses submission order — 6/9 class columns were mismatched. E08 fixed this with direct class→index mapping. E06/E07 overall mAP is correct (it's the mean), but per-class labels and submission CSVs were wrong.
- **Real weak classes (E08, correct labels):** Waders 0.223, Cormorants 0.227, Birds of Prey 0.577, Ducks 0.572. Songbirds 0.717 and Pigeons 0.811 are actually strong.
- `rcs_size_residual` is #4 feature — RCS vs expected size mismatch catches Clutter effectively (Clutter AP=0.900).
