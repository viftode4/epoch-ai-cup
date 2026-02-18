# Experiment Log

Track every experiment run. Add a row when you run something, even if it fails.

## Results Table

| ID | Date | Name | CV mAP | BoP | Clutter | Cormorants | Ducks | Geese | Gulls | Pigeons | Songbirds | Waders | Notes |
|----|------|------|--------|-----|---------|------------|-------|-------|-------|---------|-----------|--------|-------|
| E01 | 2026-02-13 | v1_baseline | 0.7030 | 0.582 | 0.941 | 0.305 | 0.605 | 0.665 | 0.957 | 0.880 | 0.810 | 0.583 | LGB only, 40 features |
| E02 | 2026-02-13 | v2_ensemble | 0.7214 | 0.610 | 0.939 | 0.254 | 0.666 | 0.728 | 0.956 | 0.885 | 0.816 | 0.640 | LGB+XGB+CB, 75 feats, class weights |
| E03 | 2026-02-13 | v3_targeted | 0.7213 | 0.620 | 0.940 | 0.253 | 0.648 | 0.731 | 0.957 | 0.882 | 0.825 | 0.635 | 115 features -- too many, diluted signal |
| E04 | 2026-02-13 | v4_multiseed | 0.7197 | 0.615 | 0.940 | 0.273 | 0.621 | 0.713 | 0.956 | 0.876 | 0.811 | 0.672 | 5-seed avg. Cormorants+Waders up, Ducks down |
| E05 | 2026-02-13 | wavelet_flightmode | 0.7193 | 0.604 | 0.937 | 0.262 | 0.656 | 0.725 | 0.957 | 0.878 | 0.812 | 0.644 | CWT wavelet + flight mode feats, 90 feats |
| E06 | 2026-02-13 | 1dcnn | 0.5238 | 0.485 | 0.715 | 0.103 | 0.397 | 0.548 | 0.914 | 0.705 | 0.591 | 0.257 | 1D-CNN on raw trajectory (6ch x 64 steps) |
| E07 | 2026-02-13 | blend_e05+e06 | 0.7250 | 0.607 | 0.932 | 0.277 | 0.668 | 0.746 | 0.958 | 0.879 | 0.814 | 0.644 | 88% tabular + 12% CNN |
| E08 | 2026-02-13 | minirocket | 0.4799 | 0.396 | 0.786 | 0.135 | 0.258 | 0.478 | 0.890 | 0.548 | 0.606 | 0.221 | MiniRocket+LogReg on 8ch x 128 trajectory |
| E09 | 2026-02-13 | svm_wavelet | 0.5238 | 0.470 | 0.793 | 0.102 | 0.417 | 0.621 | 0.873 | 0.706 | 0.573 | 0.159 | SVM RBF C=100 on CWT+core+tab (136 feats) |
| E10 | 2026-02-13 | tree_ensemble_gpu | 0.7322 | 0.605 | 0.943 | 0.289 | 0.693 | 0.753 | 0.959 | 0.879 | 0.823 | 0.648 | LGB+XGB+CB on GPU, 105 feats, CB=60% |
| E11 | 2026-02-13 | stacking_4model | 0.7396 | 0.607 | 0.939 | 0.308 | 0.711 | 0.773 | 0.960 | 0.880 | 0.829 | 0.649 | 70% tree + 10% each Rocket/CNN/SVM |
| E12 | 2026-02-13 | logit_adjustment | **0.7451** | 0.607 | 0.939 | 0.342 | 0.725 | 0.775 | 0.960 | 0.879 | 0.836 | 0.643 | **T08.** Per-class tau on E11 OOF. +0.0056 free. |
| E13a | 2026-02-13 | quant | 0.5845 | 0.546 | 0.828 | 0.176 | 0.451 | 0.547 | 0.931 | 0.742 | 0.659 | 0.381 | **T01.** QUANT on 8ch x 128. +0.10 vs MiniRocket but no stacking gain. |
| E13b | 2026-02-13 | hydra | 0.3532 | 0.216 | 0.760 | 0.052 | 0.120 | 0.418 | 0.773 | 0.285 | 0.398 | 0.159 | **T02.** Hydra transform + LogReg. Much worse than MiniRocket. |
| E13c | 2026-02-13 | multirocket | 0.4704 | 0.405 | 0.753 | 0.072 | 0.233 | 0.451 | 0.895 | 0.587 | 0.600 | 0.237 | **T02.** MultiRocket + LogReg. Slightly worse than MiniRocket. |
| E14 | 2026-02-13 | calibration | 0.7451 | 0.607 | 0.939 | 0.342 | 0.725 | 0.775 | 0.960 | 0.879 | 0.836 | 0.643 | **T13+T14.** Isotonic hurts (-0.018), temp scaling negligible. E12 still best. |
| E15 | 2026-02-13 | effective_number | **0.7535** | 0.610 | 0.942 | 0.361 | 0.739 | 0.780 | 0.960 | 0.889 | 0.842 | 0.659 | **T09.** beta=0.999 trees (0.7451) + stack (0.7493) + logit adj = **0.7535 NEW BEST** |
| E16 | 2026-02-13 | improved_cnn | 0.5193 | 0.410 | 0.768 | 0.159 | 0.373 | 0.518 | 0.916 | 0.715 | 0.537 | 0.279 | **T15-17.** Label smooth+SWA+snapshots+augment. Worse than E06 (0.5238). |
| E17 | 2026-02-13 | pseudo_labeling | 0.7485 | 0.617 | 0.942 | 0.328 | 0.725 | 0.777 | 0.960 | 0.882 | 0.840 | 0.665 | **T20-23.** 400 pseudo-labels (21% test). Hurt: -0.005 vs E15. Minority classes got 0 labels. |
| E18 | 2026-02-13 | moment_fmb | 0.6461 | 0.547 | 0.922 | 0.185 | 0.541 | 0.556 | 0.941 | 0.851 | 0.762 | 0.510 | **T06.** MOMENT embeddings alone=0.23, +feats=0.65. Stack -0.002. TS pretrain doesn't transfer to radar. |
| E20 | 2026-02-14 | groupkfold_honest | 0.6898 | 0.618 | 0.906 | 0.280 | 0.684 | 0.682 | 0.949 | 0.868 | 0.793 | 0.428 | GroupKFold on primary_observation_id. Honest baseline -0.062 vs E15. |
| E21 | 2026-02-14 | smote_groupkfold | 0.6785 | -- | -- | -- | -- | -- | -- | -- | -- | -- | SMOTE on features within GroupKFold. All targets (100-300) hurt vs E20. Discarded. |
| E22 | 2026-02-14 | hierarchical | 0.6899 | 0.619 | 0.907 | 0.281 | 0.684 | 0.682 | 0.949 | 0.869 | 0.793 | 0.427 | Binary Gull/NonGull (87.7%) + 8-class. Blend=0% hier. Didn't beat flat. |
| E23 | 2026-02-14 | perclass_weights | 0.7557 | 0.614 | 0.942 | 0.379 | 0.742 | 0.784 | 0.960 | 0.889 | 0.843 | 0.652 | Per-class stacking weights on existing OOF. +0.002 vs E15. StratifiedKFold (inflated). |
| E24 | 2026-02-14 | weakclass_features | 0.6737 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 22 weakclass feats. GroupKFold. Cormorants +0.013 but net -0.010 (dilution). |
| E25 | 2026-02-14 | no_temporal_overfit | **0.7050** | 0.591 | 0.924 | 0.324 | 0.612 | 0.712 | 0.948 | 0.847 | 0.811 | 0.576 | **CRITICAL: removed 18 temporal overfit features + weakclass. LB was 0.52 because train months [1,4,9,10] != test [2,5,9,10,12]. Config D best.** |
| E27 | 2026-02-15 | lomo_baseline | 0.6965/0.3557 | 0.587/0.323 | 0.924/0.544 | 0.304/0.061 | 0.599/0.319 | 0.709/0.355 | 0.948/0.857 | 0.843/0.172 | 0.795/0.525 | 0.561/0.045 | SKF=0.6965 vs LOMO=0.3557. Massive 0.34 gap. LOMO predicts 0 Cormorants/Pigeons/Waders. |
| E28 | 2026-02-15 | adversarial_weights | 0.6738 | 0.584 | 0.914 | 0.250 | 0.579 | 0.688 | 0.939 | 0.826 | 0.782 | 0.503 | Adversarial weighting hurts: SKF-adv=0.6738 vs 0.6965 base. LOMO-adv=0.3497 vs 0.3560 base. Shift is biological. |
| E29 | 2026-02-15 | ovr_binary | 0.6499 | 0.581 | 0.908 | 0.143 | 0.570 | 0.603 | 0.950 | 0.850 | 0.780 | 0.465 | OvR binary (LGB+CB per-class). Worse than multiclass alone but adds diversity for blending. |
| E30 | 2026-02-15 | ovr_adversarial | 0.6248 | 0.536 | 0.896 | 0.099 | 0.522 | 0.591 | 0.944 | 0.811 | 0.768 | 0.455 | OvR + adversarial weights. Worse than E29 (no adv). Adversarial weighting consistently harmful. |
| E31 | 2026-02-15 | blend_ovr_multiclass | **0.7115** | 0.600 | 0.926 | 0.324 | 0.626 | 0.713 | 0.952 | 0.855 | 0.814 | 0.594 | Per-class blend of E25D+E29 OvR. **New best SKF CV.** OvR diversity helps Ducks+Pigeons+Gulls. |
| E32 | 2026-02-15 | honest_baseline | **0.6808** | 0.584 | 0.908 | 0.275 | 0.560 | 0.702 | 0.948 | 0.843 | 0.792 | 0.516 | **HONEST BASELINE.** 114 feats (23 temporal removed). Fixed weights 0.33/0.33/0.34. No post-proc. RSKF 5x5: 0.6754+/-0.0067. Bootstrap 95% CI: [0.6505-0.7143]. LOMO: 0.3321. |
| E33 | 2026-02-15 | feature_audit | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | Adversarial AUC=0.7469 on clean 114 feats. 0 PRUNE, 10 FLAG (top adv = top classification). Pruning top-5 costs -0.055 mAP. Shift is biological, keep all features. |
| E34 | 2026-02-15 | logit_adj_honest | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | Split-half honest delta = -0.0018. All fixed taus worse. Biased full-OOF opt = +0.006 (fake). **DROP logit adjustment.** |
| E35 | 2026-02-15 | ecological_priors | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Bayesian prior adjustment on E32 test preds.** Post-processing only, no retraining. OOF shared-month (Sep+Oct): best alpha=0.75 -> +0.011 mAP. Full OOF: +0.016. Ecological priors for unseen months (Feb/May/Dec) from Dutch ornithology. Saves 6 submissions (alpha 0/0.5/0.75/1.0/1.5 + nearest-month). LB: TBD. |
| E36-A | 2026-02-15 | gbif_postproc | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **GBIF data-driven post-processing on E32.** Same formula as E35 but priors from 96K real GBIF observations. OOF shared months: best alpha=0.75 -> +0.011. Full OOF: alpha=0.75 -> +0.016. Better than hand-crafted E35 priors. |
| E36-B | 2026-02-15 | gbif_features | **0.6984** | 0.598 | 0.921 | 0.294 | 0.578 | 0.709 | 0.952 | 0.846 | 0.804 | 0.585 | **10 GBIF seasonal features added to E32.** 9 class SIs + 1 Shannon diversity. 124 total feats. SKF CV=0.6984, **+0.0176 vs E32**. Waders +0.069, Ducks +0.018, Cormorants +0.020. |
| E36-AB | 2026-02-15 | gbif_combined | **0.7008** | 0.597 | 0.920 | 0.334 | 0.583 | 0.708 | 0.952 | 0.838 | 0.803 | 0.575 | **E36-B + GBIF post-processing (alpha=0.5).** Best combined: 0.7008, **+0.020 vs E32**. Cormorants +0.059 (0.275->0.334). |
| E37 | 2026-02-15 | species_traits | 0.7006 | 0.600 | 0.921 | 0.349 | 0.550 | 0.714 | 0.951 | 0.841 | 0.800 | 0.579 | **AVONET body mass + Bruderer wingbeat freq + flight speed traits.** 60 trait features (per-class distance + match scores) on E36-B. Delta: +0.002 vs E36-B. Cormorants +0.055 but Ducks -0.028. Within noise. |
| E38 | 2026-02-15 | weather_solar | SKF 0.7604 / LOMO 0.3615 | 0.618/0.365 | 0.953/0.533 | 0.381/0.054 | 0.728/0.385 | 0.807/0.410 | 0.965/0.856 | 0.890/0.149 | 0.858/0.463 | 0.644/0.039 | **KNMI weather + solar + GBIF.** 139 feats. LOMO +0.030 vs E32 base (0.332->0.362). Solar alone best single add (+0.029 LOMO). Weather helps Ducks/Geese LOMO. But SKF jumped to 0.76 = solar/weather features leak month identity in SKF! LOMO gap = 0.40. Pigeons LOMO crashed to 0.15. **LB = 0.53.** |
| E39 | 2026-02-15 | temporal_free_stacking | SKF 0.6900 / LOMO 0.3356 | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Tree+MiniRocket+CNN stacking (no temporal).** LOMO: Tree 0.332, Rocket 0.245, CNN 0.235. Best blend 80/10/10 = 0.3356 (+0.004 vs tree alone = noise). Sequence models too weak to help LOMO. |
| E40 | 2026-02-16 | balanced_augmentation | LOMO 0.3609 (best=A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Mixup augmentation + regularization sweep on E38 features.** 6 configs tested. A (E38 baseline): 0.3609. B (Mixup+base): 0.3592. C (mid reg): 0.3497. D (Mixup+mid): 0.3542. E (heavy reg): 0.3226. F (Mixup+heavy): 0.3452. **Nothing beats E38 baseline.** Augmentation and heavier reg both hurt. |
| E41 | 2026-02-16 | quick_ensembles | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Post-processing experiments on existing preds.** Month detection via daylight hours: perfect (176 Feb, 303 May, 457 Sep, 803 Oct, 133 Dec). E32+E38 blend, month-adaptive GBIF reweighting. Submissions: e41a (blend), e41b (month-adaptive), e41c (blend+adaptive). |
| E42 | 2026-02-16 | minority_specialists | LOMO 0.3799 | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Binary CB specialists per minority class.** Confusion analysis: Gulls absorb 44-54% of minority preds. Binary CB beats multiclass for Cormorants (+0.034), Waders (+0.040), Pigeons (+0.028). BoP/Ducks: multiclass better. Best blend alpha=0.4 LOMO=0.3799 (+0.019 vs E38 base 0.3609). Top features: Cormorants (lon_std, rcs_range), Waders (airspeed, wind_speed), Pigeons (lon_mean, total_turning). |
| E44 | 2026-02-16 | flight_physics | LOMO 0.3577 | 0.369 | 0.566 | 0.058 | 0.332 | 0.395 | 0.857 | 0.142 | 0.460 | 0.040 | **T27-T31: 24 physics features (cross-channel, biomechanics, RCS modulation, 3D geometry, complexity).** LOMO -0.003 vs E38 (0.361->0.358). Clutter +0.037, Cormorants +0.004, but Ducks -0.054. Feature dilution on LOMO. SKF unchanged at 0.76. **Discarded for now -- delta within noise but wrong direction.** |
| E45 | 2026-02-16 | path_signatures | LOMO 0.3606 (best=A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **T32: Path signatures via esig.** A: E38 base 139 feats = 0.3606 (baseline). B: +sig depth-2 lead-lag 212 feats = 0.3479 (-0.013). C: +sig depth-3 no-LL 224 feats = 0.3463 (-0.014). D: +phys+sig crashed (duplicate columns). **Signatures hurt tree models -- feature dilution on small LOMO folds. Discarded.** |
| E46 | 2026-02-16 | cwt_svm_stack | LOMO 0.3605 (best=5%SVM) | 0.371 | 0.518 | 0.050 | 0.374 | 0.421 | 0.859 | 0.145 | 0.469 | 0.038 | **T33: Zaugg CWT + SVM stacking.** SVM standalone LOMO=0.2493 (67 CWT feats). Tree LOMO=0.3597. Best blend 5%SVM: 0.3605 (+0.0008 = noise). SVM on CWT spectral features is much weaker than trees. No diversity gain. SKF=0.7619. **Discarded.** |
| E47 | 2026-02-16 | multirocket_tta | LOMO 0.3576 (best=C) | 0.354 | 0.543 | 0.047 | 0.368 | 0.404 | 0.857 | 0.135 | 0.460 | 0.040 | **T34-T35: MultiRocket+Ridge + tree TTA.** A: MultiRocket+Ridge LOMO=0.2064 (49728 feats, terrible). B: Tree LOMO=0.3565 (116 feats, no wx/sol/gbif). C: 15% MR blend=0.3576 (+0.001=noise). D: TTA LOMO=0.3448 (-0.012, noise hurts trees). **All discarded.** |

## CRITICAL: Temporal Overfitting Discovery (2026-02-14)

**ALL CV scores in E01-E23 are inflated** due to temporal feature overfitting.

- Train months: [1, 4, 9, 10]. Test months: [2, 5, 9, 10, 12].
- 33% of test data is from months NEVER seen in training (Feb, May, Dec).
- Temporal features (month, hour, is_october, oct_afternoon, etc.) learned spurious correlations.
- E15 best: CV=0.7535 but LB=0.52. The 0.23 gap is from temporal overfit + observation_id leakage.
- E25 Config A (with temporal): CV=0.7450, predicts **0 Pigeons** in test.
- E25 Config D (no temporal): CV=0.7050, predicts 7 Pigeons, 42 Waders -- more realistic.
- The CV drop from 0.75 to 0.70 is MOSTLY fake signal being removed, not real signal lost.

**Lesson**: Never include calendar features when train/test have different temporal distributions.
Cross-validation cannot catch this because all folds share the same months.

## Ablation Study (2026-02-13)

Systematic test of every feature group and model type in isolation.

### Feature ablation (LGB only, same hyperparams):
| Config | #Feats | mAP |
|--------|--------|------|
| core only | 53 | 0.6236 |
| core+tab | 69 | 0.6994 |
| core+fft+tab | 73 | 0.6963 |
| core+fft+tab+tgt | 93 | 0.6993 |
| core+tab+wav | 78 | 0.6900 |
| core+tab+flight | 81 | 0.6948 |
| core+fft+tab+tgt+flight | 105 | 0.7010 |
| kitchen_sink | 114 | 0.6925 |

### Model ablation (best feature set, 105 feats):
| Model | mAP |
|-------|------|
| XGB alone | 0.7094 |
| CatBoost alone | 0.7024 |
| LGB alone | 0.7010 |
| LGB+XGB+CB ensemble | 0.7239 |

### Pairwise ensembles:
| Pair | mAP |
|------|------|
| LGB+CB | 0.7226 |
| XGB+CB | 0.7209 |
| LGB+XGB | 0.7158 |
| LGB+XGB+CB | 0.7239 |

## Key Learnings

- **Features are saturated.** core+tabular (69 feats) = 0.6994. Best combo (105 feats) = 0.7010. Delta = 0.0016.
- **The ensemble is what matters.** Best single model (XGB 0.7094) -> 3-model ensemble (0.7239) = +0.0145.
- **CatBoost is the secret weapon.** LGB+CB (0.7226) > LGB+XGB (0.7158). CB at 45% weight, best at Pigeons (0.267) and Clutter (0.604).
- **Wavelet features hurt when added alone** (-0.009). Only neutral when diluted by other features.
- **FFT features also hurt** (-0.003 vs core+tab). The RCS spectral features add noise.
- **Kitchen sink is worst full config.** 114 features = 0.6925. Feature dilution confirmed.
- Boosting one minority class (Pigeons) steals from its neighbor (Ducks).
- Even a weak CNN (0.52) at 12% weight adds +0.006 via model diversity (E07).
- **Model diversity >> feature engineering** at this stage. Next gains come from more diverse models, not more features.
- **GPU confirmed working** for LGB (device=gpu), XGB (device=cuda), CB (task_type=GPU). GPU on E10 improved CB weight from 50% to 60% and mAP from 0.7276 to 0.7322.
- **Heterogeneous stacking works.** Even weak models (MiniRocket 0.48, CNN 0.52, SVM 0.52) add +0.0074 via diversity at 10% each.
- **LR meta-learner overfits** on 36 meta-features with 2601 samples (0.7020 < 0.7396 weighted avg). Simple weighted average is better here.
- **CWT-only SVM failed** (0.16 mAP) — radar sampling rate too low for pure wingbeat analysis. Combined with core+tab features: 0.52.
- **Effective Number weights (T09) are huge**: beta=0.999 → tree ensemble 0.7451 (+0.0129 vs E10). Downweights Gulls to 0.116, upweights Cormorants to 2.3. CB weight rose from 60% to 80%.
- **Post-hoc logit adjustment stacks with better base**: E15 trees + stack + logit adj = 0.7535, +0.0084 over E12's 0.7451.
- **TEMPORAL OVERFITTING IS THE #1 PROBLEM**: Train months [1,4,9,10] vs test [2,5,9,10,12]. 18 temporal features (month, hour, is_october, etc.) inflate CV by ~0.05 but HURT LB. All CV scores before E25 are inflated. E25 without temporal: CV 0.7050 is the honest StratifiedKFold number.
- **GroupKFold + no temporal = double penalty**: GroupKFold (E20) gives 0.6898, already below E25's 0.7050. The primary_observation_id leakage and temporal overfitting are partially overlapping issues.
- **SMOTE hurts tree models**: All targets (100-300) worse than baseline. Synthetic features confuse gradient boosting.
- **Hierarchical classification doesn't help**: Binary Gull/NonGull (87.7% acc) too noisy, errors propagate. Optimal blend = 0% hierarchical.
- **Weakclass features help Cormorants but dilute overall**: +0.013 Cormorants, -0.010 overall. BUT when combined with temporal removal (E25D), net positive: 0.7050 vs 0.6970 without them.
- **LOMO CV is far too harsh**: LOMO=0.3557 vs SKF=0.6965 (delta=0.34!). Model can barely generalize across months. LOMO predicts 0 Cormorants/Pigeons/Waders on test.
- **Adversarial sample weighting HURTS**: Both on multiclass (-0.006 LOMO, -0.023 SKF) and OvR (-0.025 SKF). The train/test shift is biological (seasonal bird behavior), not fixable by reweighting samples.
- **OvR binary classifiers worse standalone** (0.6499 vs 0.6965) but add diversity for blending: per-class blend of E25D+E29 OvR = 0.7115 (+0.0067 vs E25D alone).
- **Per-class blending > greedy blending**: Per-class optimal (0.7115) > greedy forward (0.7058) > single model (0.7048). OvR helps Ducks (+0.019), Pigeons (+0.011), Gulls (+0.002).
- **5 additional temporal leaks found** in add_weakclass_tabular(): is_oct_nov, migration_alt, migration_speed, is_night, night_high_alt. Now included in ALL_TEMPORAL filter.
- **E32 honest baseline = 0.6808** (RSKF 5x5 = 0.6754+/-0.0067). This is the REAL number after removing all tricks: 23 temporal features removed, fixed weights, no logit adj. E25D's 0.7050 was inflated by weight optimization + only 18 temporal removed.
- **Logit adjustment is NEGATIVE** when evaluated honestly: split-half delta = -0.0018. Per-class tau optimized on same OOF was +0.006 (fake). All fixed taus are also worse. **DROP IT.**
- **Feature pruning HURTS**: top adversarial features are also top classification features. Removing top-5 costs -0.055 mAP. The biological shift cannot be reduced without destroying classification signal.
- **Bootstrap std = 0.016** means a delta must exceed 0.032 to be meaningful. Most "gains" from E25-E31 are within noise.
