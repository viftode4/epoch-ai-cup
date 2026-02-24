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
| E42 | 2026-02-16 | minority_specialists | **LB 0.53** (LOMO 0.3799) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Binary CB specialists per minority class.** Confusion analysis: Gulls absorb 44-54% of minority preds. Binary CB beats multiclass for Cormorants (+0.034), Waders (+0.040), Pigeons (+0.028). BoP/Ducks: multiclass better. Best blend alpha=0.4 LOMO=0.3799 (+0.019 vs E38 base 0.3609). **Kaggle:** `e42_blend40_0.3799_20260216_0020.csv` scored **0.53** → LOMO does not translate to unseen months. Top features: Cormorants (lon_std, rcs_range), Waders (airspeed, wind_speed), Pigeons (lon_mean, total_turning). |
| E44 | 2026-02-16 | flight_physics | LOMO 0.3577 | 0.369 | 0.566 | 0.058 | 0.332 | 0.395 | 0.857 | 0.142 | 0.460 | 0.040 | **T27-T31: 24 physics features (cross-channel, biomechanics, RCS modulation, 3D geometry, complexity).** LOMO -0.003 vs E38 (0.361->0.358). Clutter +0.037, Cormorants +0.004, but Ducks -0.054. Feature dilution on LOMO. SKF unchanged at 0.76. **Discarded for now -- delta within noise but wrong direction.** |
| E45 | 2026-02-16 | path_signatures | LOMO 0.3606 (best=A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **T32: Path signatures via esig.** A: E38 base 139 feats = 0.3606 (baseline). B: +sig depth-2 lead-lag 212 feats = 0.3479 (-0.013). C: +sig depth-3 no-LL 224 feats = 0.3463 (-0.014). D: +phys+sig crashed (duplicate columns). **Signatures hurt tree models -- feature dilution on small LOMO folds. Discarded.** |
| E46 | 2026-02-16 | cwt_svm_stack | LOMO 0.3605 (best=5%SVM) | 0.371 | 0.518 | 0.050 | 0.374 | 0.421 | 0.859 | 0.145 | 0.469 | 0.038 | **T33: Zaugg CWT + SVM stacking.** SVM standalone LOMO=0.2493 (67 CWT feats). Tree LOMO=0.3597. Best blend 5%SVM: 0.3605 (+0.0008 = noise). SVM on CWT spectral features is much weaker than trees. No diversity gain. SKF=0.7619. **Discarded.** |
| E47 | 2026-02-16 | multirocket_tta | LOMO 0.3576 (best=C) | 0.354 | 0.543 | 0.047 | 0.368 | 0.404 | 0.857 | 0.135 | 0.460 | 0.040 | **T34-T35: MultiRocket+Ridge + tree TTA.** A: MultiRocket+Ridge LOMO=0.2064 (49728 feats, terrible). B: Tree LOMO=0.3565 (116 feats, no wx/sol/gbif). C: 15% MR blend=0.3576 (+0.001=noise). D: TTA LOMO=0.3448 (-0.012, noise hurts trees). **All discarded.** |
| E48 | 2026-02-18 | external_priors | SKF 0.7567 / LOMO 0.3526 | 0.347 | 0.518 | 0.038 | 0.321 | 0.423 | 0.864 | 0.152 | 0.467 | 0.044 | **AVONET + BirdWingData + Col de la Croix class priors.** 4-way ablation: base 0.3487, +morph 0.3433, +flight 0.3526, +both 0.3390. Only flight priors helped (+0.0039 LOMO). Saved submission `e48_external_priors_0.7567_20260218_2143.csv`. |
| E49 | 2026-02-18 | prior_specialists | LOMO 0.3591 | 0.353 | 0.489 | 0.041 | 0.300 | 0.442 | 0.861 | 0.219 | 0.479 | 0.048 | **Binary specialists on top of E48-C (flight-prior base).** Specialists helped Waders (+0.047 AP) and Pigeons (+0.014 AP), hurt Cormorants/Ducks/BoP. Global alpha sweep best at 0.8, improving LOMO +0.0065 vs E48-C. Saved `e49_prior_specialists_0.3591_20260218_2144.csv`. |
| E50 | 2026-02-18 | perclass_specialist_blend | LOMO 0.3625 | 0.352 | 0.497 | 0.041 | 0.319 | 0.440 | 0.862 | 0.224 | 0.479 | 0.049 | **Per-class alpha optimization (E49 extension).** Brute-force class-wise blend found best map `{Waders:1.0, Pigeons:0.6}` for improving specialists. LOMO +0.0099 vs E48-C and +0.0034 vs E49. Saved `e50_perclass_specialist_blend_0.3625_20260218_2146.csv`. |
| E51 | 2026-02-18 | e42_e50_blend | LOMO 0.3807 | 0.328 | 0.567 | 0.064 | 0.400 | 0.434 | 0.859 | 0.261 | 0.446 | 0.067 | **Post-processing blend of strong historical model + new external-prior model.** Blended `E42` and `E50` OOF/test predictions with alpha sweep. Best at `alpha_e50=0.35` -> **0.3807**, beating E42 alone (0.3743) and all new standalone runs. Saved `e51_e42_e50_blend_0.3807_20260218_2148.csv`. |
| E52 | 2026-02-18 | month_aware_blend | OOF 0.3904 (shared=0.3840) | 0.330 | 0.614 | 0.061 | 0.385 | 0.459 | 0.853 | 0.281 | 0.451 | 0.082 | **Kaggle-driven continuation after E50 > E51 on LB.** Kept E50 for unseen months and blended E42 into shared months only (Sep/Oct). Grid search found `w9=0.75, w10=0.55`; conservative variant also saved. Submissions: `e52_monthaware_w9_0.75_w10_0.55_0.3904_20260218_2156.csv`, `e52_monthaware_cons_w9_0.60_w10_0.44_0.3869_20260218_2156.csv`. |
| E53 | 2026-02-18 | unseen_month_prior_sweep | LB 0.55 (a0.15) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Unseen-month-only GBIF prior sweep on top of E50 and E52.** Applied Bayesian adjustment only for months {2,5,12} with alpha in {0.15,0.25,0.35,0.50}. **Kaggle result:** `e53_e50_unseenprior_a0.15_20260218_2157.csv` scored **0.55 (new best LB)**. Other E53 variants not evaluated on Kaggle yet. |
| E54 | 2026-02-18 | unseen_month_specific | LB 0.56 (winter) / 0.55 (spring) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Month-specific unseen adjustment on top of E50 (post-E53 success).** Diagnosis: uniform `a=0.15` mostly changed Feb/Dec and weakly touched May. Two targeted variants were tested on Kaggle: `spring_tilt {m2=0.15,m5=0.28,m12=0.15}` scored **0.55**, while `winter_tilt {m2=0.22,m5=0.12,m12=0.24}` scored **0.56 (new best LB)**. Saved `e54_e50_spring_tilt_m2_0.15_m5_0.28_m12_0.15_20260218_2229.csv` and `e54_e50_winter_tilt_m2_0.22_m5_0.12_m12_0.24_20260218_2229.csv`. |
| E55 | 2026-02-19 | winter_refinement | LB 0.56 (balanced/stronger) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Local refinement around winning E54 winter tilt.** Hypothesis: keep May correction low while increasing winter correction slightly. Kaggle: `winter_balanced {m2=0.24,m5=0.10,m12=0.26}` = **0.56**, `winter_stronger {m2=0.26,m5=0.10,m12=0.30}` = **0.56** (no improvement vs E54 winter). `winter_stronger_gated` not evaluated on Kaggle. Files: `e55_winter_balanced_m2_0.24_m5_0.10_m12_0.26_20260219_1211.csv`, `e55_winter_stronger_m2_0.26_m5_0.10_m12_0.30_20260219_1211.csv`, `e55_winter_stronger_gated_m2_0.26_m5_0.10_m12_0.30_gated_20260219_1211.csv`. |
| E56 | 2026-02-20 | may_off_probe | LB 0.55 | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Diagnostic probe (Kaggle tested):** keep the winning winter correction and turn **May correction off**. Result: **0.55** (worse than the 0.56 winter-tilt baseline), implying May adjustment is beneficial (or at least not safely removable). Variant: `{m2=0.22,m5=0.00,m12=0.24}` saved as `e56_e50_may_off_m2_0.22_m5_0.00_m12_0.24_20260220_1337.csv`. |
| E57 | 2026-02-20 | may_alpha_adaptive | LB 0.56 (all) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Adaptive May-alpha candidates (smart order, not grid).** Fixed winter to the winning values (`m2=0.22, m12=0.24`) and tested 4 informative May probes: `m5 ∈ {0.06, 0.09, 0.15, 0.18}`. **Kaggle:** all four scored **0.56** (no separation at 2-decimal LB precision), indicating a plateau in this May-alpha range. Files: `e57_e50_mayprobe_m2_0.22_m5_0.18_m12_0.24_20260220_1508.csv`, `...m5_0.06...`, `...m5_0.15...`, `...m5_0.09...`. |
| E58 | 2026-02-21 | winter_airspeed_gate | LB 0.55 (k135) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Next hypothesis tested (FAILED):** residual winter errors are within-month ambiguities (especially Gulls vs Waders), not month-prior magnitude. Applied an additional **airspeed-gated** boost to Waders only when `top1=Gulls`, `top2=Waders`, margin < 0.15, and airspeed exceeds a month-specific threshold (Feb >= 15.5, Dec >= 14.0). **Kaggle:** `k135` scored **0.55** (worse than 0.56 baseline), so this gate over-corrects; do NOT pursue larger k. Files: `e58_winter_airspeed_gate_k135_20260221_2117.csv`, `e58_winter_airspeed_gate_k155_20260221_2117.csv` (not evaluated). |
| E59 | 2026-02-21 | col_de_la_croix_transfer | LOMO 0.3470 | 0.345 | 0.487 | 0.054 | 0.315 | 0.417 | 0.863 | 0.152 | 0.450 | 0.042 | **Trained XGBoost auxiliary model on 1988 Col de la Croix dataset.** Mapped 4 kinematics features and blended 8 class probabilities into E38 full feature stack. Delta: **-0.0018 LOMO**. Minor improvements in Cormorants/Ducks/Pigeons outweighed by drops in Songbirds/Clutter/BoP. **Discarded.** |
| E60 | 2026-02-21 | optuna_lgbm_tuned | SKF 0.7524 | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Optuna tuned LightGBM** (E38 full feature stack, 139 feats, CPU). Best params: lr=0.0159, leaves=65, depth=9, mcs=26, subsample=0.673, colsample=0.503, reg_a=0.00338, reg_l=0.0585. Saved `e60_lgbm_tuned_0.7524_20260221_2328.csv`. |
| E61 | 2026-02-21 | focal_loss_lgbm | SKF 0.5076 | 0.425 | 0.790 | 0.058 | 0.295 | 0.343 | 0.887 | 0.715 | 0.656 | 0.399 | **Custom multi-class focal loss attempt** in LightGBM. Result collapsed; objective/hessian approximation likely wrong. **Do not submit.** Saved `e61_lgbm_focal_loss_0.5076_20260221_2148.csv`. |
| E62 | 2026-02-21 | soft_pseudolabels | **LB 0.50** (SKF 0.7711) | 0.658 | 0.947 | 0.393 | 0.746 | 0.823 | 0.969 | 0.892 | 0.867 | 0.645 | **Soft pseudo-labeling from E54 preds** (p>0.05) added 7030 pseudo samples (downweighted). SKF jumped, but test argmax distribution collapsed to Waders (0 Gulls/Songbirds top-1). **Kaggle: 0.50 → discard.** Saved `e62_soft_pseudolabels_0.7711_20260221_2150.csv`. |
| E63 | 2026-02-21 | blend_e54_e60_monthaware | **LB 0.55** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Month-aware blend**: keep E54 for unseen months, inject 10% of E60 on months 9/10. **Kaggle: 0.55 (worse than 0.56 baseline).** Saved `e63_blend_e54_e60_m9_0.10_m10_0.10_20260221_2338.csv`. |
| E64 | 2026-02-21 | blend_e54_e62_unseen | LB TBD | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Unseen-month blend**: inject 5% of E62 into E54 for months {2,5,12}. Saved `e64_blend_e54_e62_unseen_w0.05_20260221_2338.csv`. |
| E65 | 2026-02-21 | blend_e54_e60_monthaware_stronger | **LB 0.54** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Stronger month-aware blend**: inject 20% of E60 on months 9/10. **Kaggle: 0.54 (worse).** Saved `e65_blend_e54_e60_m9_0.20_m10_0.20_20260221_2339.csv`. |
| E66 | 2026-02-22 | gw_specialist_correction | LOMO 0.3608 | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Learned Gulls↔Waders correction layer** (binary specialist \(s(x)=P(\text{Waders}\mid x)\); redistribute pair mass \(S=p_G+p_W\): \(p'_W=S\cdot s,\; p'_G=S\cdot(1-s)\)). Specialist AP=0.1449 and net **-0.0017 LOMO** vs E50 base. Submissions saved (not Kaggle-tested): `e66_gw_specialist_then_priors_20260222_0013.csv`, `e66_priors_then_gw_specialist_20260222_0013.csv`. |
| E67 | 2026-02-22 | gatedpriors_margin | **LB 0.56** (tau=0.10/0.15) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Uncertainty-gated unseen-month priors (E54 alphas)**: apply GBIF prior tilt only when top-2 margin \(p_{top1}-p_{top2}<\\tau\). Kaggle: `tau=0.05` -> **0.55** (99 rows adjusted), `tau=0.10` -> **0.56** (190 rows), `tau=0.15` -> **0.56** (265 rows). Files: `e67_gatedpriors_tau0.05_20260222_0015.csv`, `...tau0.10...`, `...tau0.15...`. |
| E70 | 2026-02-22 | bop_song_specialists_unseeninj | **LB 0.56** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Specialists-only use of E68 features.** Trained LGB binary specialists for BoP + Songbirds on the 12 E68 features (8 `enhanced_bio_shape` + 4 solar-derived). Applied *only on test unseen months* (2/5/12) with gates: base uncertainty (margin<0.25) + BoP requires thermal window, Songbirds requires dawn/dusk and specialist>base+Δ. Injection counts: BoP 3 (May), Songbirds 76 (Feb 12, Dec 64). Then apply E67 gated priors (tau=0.15). Saved `e70_unseeninj_bop0.35_song0.25_marg0.25_priors_tau0.15_0.3625_20260222_2257.csv`. **Kaggle: 0.56 (no improvement vs E54/E67 plateau).** |
| E71 | 2026-02-23 | e52_plus_gated_priors | **LB 0.56** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Isolate shared-month bottleneck test.** Use `E52` month-aware blend (inject E42 into months 9/10 only; unseen months stay E50), then apply `E67` gated unseen-month priors (tau=0.15, E54 winter alphas). This changes *only* months 9/10 vs `e67_tau0.15` (1260/1872 rows changed; 71 top-1 label changes), while months 2/5/12 are identical. Saved `e71_e52_plus_gatedpriors_tau0.15_0.3904_20260223_1214.csv`. **Kaggle: 0.56 → shared-month blending does not break plateau.** |
| E72 | 2026-02-23 | e52_conservative_plus_gated_priors | LB TBD | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Conservative version of E71.** Rebuild `E52` conservative month-aware blend with weights `w9=0.60, w10=0.44` (20% shrink), then apply `E67` gated unseen-month priors (tau=0.15). Saved `e72_e52cons_w9_0.60_w10_0.44_plus_gatedpriors_tau0.15_0.3869_20260223_1217.csv`. |
| E73 | 2026-02-23 | unseen_nb_physics_correction | **LB 0.58** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **New unseen-month within-month correction (WINNER so far).** Start from `test_e50.npy`, apply E67 gated priors (tau=0.15; 265 rows), then apply a *Naive Bayes physics* product-of-experts using only `airspeed` + `radar_bird_size` (learned from train; Laplace=1.0; Gaussian speed per class; min σ=0.5). Apply only for unseen months (2/5/12) when margin<0.25 (451 rows). Compared to `e67_tau0.15`: 451 rows changed, 40 top-1 changes (Feb 18, May 6, Dec 16). Saved `e73_nbphys_unseen_tau0.25_g0.12_priortau0.15_20260223_1338.csv`. **Kaggle: 0.58 → confirms remaining headroom is within-unseen-month ranking, not month blending.** |
| E74 | 2026-02-24 | nbphys_tuning | **LB 0.58** (A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Tune E73 correction strength / coverage.** Same pipeline (E50 -> E67 gated priors tau=0.15 -> NB(size+airspeed) correction on unseen months only). Two variants generated: (A) `tau_nb=0.30, gamma=0.14` (560 rows gated; 44 top1 flips on unseen) scored **0.58** (no improvement vs E73). (B) `tau_nb=0.20, gamma=0.10` (355 rows; 37 flips) not Kaggle-tested yet. Files: `e74_nbphys_unseen_tau0.30_g0.14_priortau0.15_20260224_1522.csv`, `e74_nbphys_unseen_tau0.20_g0.10_priortau0.15_20260224_1522.csv`. |
| E75 | 2026-02-24 | nbphys_altitude_correction | **LB 0.59** (A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Extend NB correction with altitude cues (new best LB).** Same base pipeline (E50 -> E67 gated priors tau=0.15), then NB likelihood uses `radar_bird_size` + Gaussians for `(airspeed, alt_mid=(min_z+max_z)/2, alt_range=max_z-min_z)` on unseen months only with `tau_nb=0.30`. Candidate (A) `gamma=0.10` scored **0.59**. Candidate (B) `gamma=0.08` not Kaggle-tested yet. Files: `e75_nbalt_unseen_tau0.30_g0.10_priortau0.15_20260224_1529.csv`, `e75_nbalt_unseen_tau0.30_g0.08_priortau0.15_20260224_1529.csv`. |
| E76 | 2026-02-24 | nbalt_tracklen_correction | **LB 0.58** (A) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Add track length evidence (FAILED).** Same pipeline as E75 (E50 -> E67 gated priors tau=0.15), then NB likelihood extends continuous factors with `n_pts=len(trajectory_time)` (and optional `duration=trajectory_time[-1]-trajectory_time[0]`). Unseen-month only, `tau_nb=0.30`. Candidate (A) feats=`n_pts`, `gamma=0.06` scored **0.58** (worse than E75=0.59), indicating track-length cues likely violate the invariance assumption \(P(u\\mid y)\\) across train→test. Candidate (B) feats=`duration+n_pts`, `gamma=0.08` not Kaggle-tested yet. Files: `e76_nbalt_npts_tau0.30_g0.06_priortau0.15_20260224_1542.csv`, `e76_nbalt_dur_npts_tau0.30_g0.08_priortau0.15_20260224_1542.csv`. |
| E77 | 2026-02-24 | nbalt_month_gamma | **LB 0.58** | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Targeted May tempering (FAILED).** Same as E75 but apply month-specific NB exponent: `gamma_m2=0.10, gamma_m5=0.06, gamma_m12=0.10` (tau_nb=0.30). This only changes **May** vs E75 (259 rows differ; 5 top-1 changes). **Kaggle: 0.58** → reducing May correction harmed; keep a single gamma for all unseen months for now. Saved `e77_nbalt_monthgamma_m2_0.10_m5_0.06_m12_0.10_tau0.30_20260224_1553.csv`. |
| E78 | 2026-02-24 | nbalt_feature_weighting | LB TBD | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Refine E75 without adding new evidence.** Keep same NB-alt evidence but downweight `alt_range` inside the likelihood to reduce redundancy / noise (altitude “double-counting”). Two candidates generated (tau_nb=0.30, gamma=0.10): (A) `w_alt_range=0.50` saved as `e78_nbalt_weighted_ws1.00_wv1.00_wm1.00_wr0.50_tau0.30_g0.10_20260224_1602.csv` (vs E75: 3 top-1 changes), (B) `w_alt_range=0.00` saved as `e78_nbalt_weighted_ws1.00_wv1.00_wm1.00_wr0.00_tau0.30_g0.10_20260224_1602.csv` (vs E75: 7 top-1 changes). |

| E69 | 2026-02-22 | e42_with_priors | **LB 0.53** (A winter_tilt) | -- | -- | -- | -- | -- | -- | -- | -- | -- | **Apply E54 winter_tilt priors to E42 base (LOMO 0.3799) instead of E50 (0.3625).** 4 variants: A=winter_tilt(m2=0.22,m5=0.12,m12=0.24), B=stronger_winter(m2=0.26,m5=0.12,m12=0.28), C=spring_tilt(m2=0.15,m5=0.28,m12=0.15), D=moderate(m2=0.20,m5=0.10,m12=0.22). **Note:** E69 uses convex mixing \((1-\\alpha)p+\\alpha\\pi_{GBIF}\) (not the Bayes ratio tilt used in E53–E55). Kaggle result indicates E42 does not benefit from unseen-month priors the way E50 does; likely base posterior structure differs in unseen months. |
| E68 | 2026-02-22 | enhanced_features | LOMO 0.3452 | 0.355 | 0.446 | 0.069 | 0.316 | 0.404 | 0.854 | 0.122 | 0.495 | 0.047 | **New features: 8 enhanced bio-shape (turn_dir_consistency, max_sustained_turn_frac, rcs_dominant_ac_lag, rcs_flap_regularity, rcs_glide_flap_var_ratio, rcs_burst_fraction, path_loop_fraction, turn_reversal_rate) + 4 solar-derived (hours_from_solar_noon, is_thermal_window, is_dawn_dusk, is_afternoon_thermal) on E38 base.** CB multiclass alone=0.3556 (close to E38 base 0.3615), but LGB+XGB ensemble=0.3406 (hurt by new feats). After specialists: 0.3452. **Feature dilution: +12 feats to 151 total hurt ensemble. BoP +0.027, Songbirds +0.044 improved individually. Do NOT add these features to E42. Revert to E42 feature set.** |
| E70 | 2026-02-23 | optuna_lomo_flight_priors | LOMO 0.3507 (0.3556 tuned-w) | 0.348 | 0.549 | 0.044 | 0.299 | 0.412 | 0.862 | 0.140 | 0.465 | 0.040 | **Optuna-tuned LGB + E48-C flight prior features (AVONET, no BirdWingData) + E54 priors.** 30 Optuna LOMO trials, best LGB params found. 170 features total. LGB(tuned)=0.3324, XGB=0.3346, CB=0.3605. Equal-weight ensemble=0.3507. Tuned-weight sweep (LGB=0.30, XGB=0.20, CB=0.50) → 0.3556. **Optuna hurt LGB (overfitting to train fold); CB defaults still dominant.** Ensemble=0.3507 worse than E50=0.3625. Variant C (tuned weights, winter_tilt) saved as `e70_tunedw_lgb0.3_xgb0.2_cb0.5_0.3556`. Submit Variant C (0.3556 + E54 priors) to test if tuned-weight base beats E54 on LB. |
| E71 | 2026-02-23 | pergroup_gbif_priors | LOMO 0.3551 | 0.351 | 0.567 | 0.043 | 0.312 | 0.412 | 0.863 | 0.136 | 0.473 | 0.040 |**Fixed wing morphology + per-group GBIF alpha optimisation.** Two improvements vs E70: (1) BirdWingData absent → now uses hardcoded per-class wingspan/wing_area from literature (not global 0.70m/0.07m² default) → recovers wing_loading/aspect_ratio differentiation that E50 used. (2) Per-group alpha: alpha_rare (Waders/Ducks/Cormorants/Geese) tuned separately from alpha_common (Gulls/Songbirds/BoP/Pigeons/Clutter) via month-analog LOMO proxy (Feb→Jan, May→Apr, Dec→Oct). Grid: alpha_rare∈[0.1–0.5], alpha_common∈[0.0–0.20]. Ensemble CB=0.50/LGB=0.30/XGB=0.20, skip Optuna. 4 submissions: base, winter_tilt (E54 alphas), pergroup, pergroup_stronger. |

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

## Mathematical Formalization (2026-02-22)

This section captures the *mathematical structure* behind the experiments, so future iterations reuse the same assumptions and the post-processing math.

### Notation

- Dataset consists of tracks \(i=1..N\) with:
  - \(x_i\): features (trajectory-derived + non-privileged tabular),
  - \(m_i\): month extracted from timestamp,
  - \(y_i \in \{1,\dots,K\}\): class label (train only), with \(K=9\).
- Model outputs a probability vector \(p_i = f_\theta(x_i) \in \Delta^{K-1}\), where \(p_{i,c}\) is the score for class \(c\).

### Evaluation metric (macro mAP)

Competition score is macro-averaged Average Precision:

\[
\text{mAP}=\frac{1}{K}\sum_{c=1}^K \text{AP}_c
\]

where \(\text{AP}_c\) is computed by `sklearn.metrics.average_precision_score(y_true==c, p_{:,c})`.

**Important consequence:** \(\text{AP}_c\) depends on the *global ranking* of \(p_{i,c}\) across **all** test samples, including cross-month comparisons.

### Shift model (why SKF can mislead)

- Train months: \(\{1,4,9,10\}\)
- Test months: \(\{2,5,9,10,12\}\) (33% unseen months)

Standard StratifiedKFold estimates performance under the *training-month mixture*. The public test distribution uses a different month mixture (and includes unseen months), so SKF is an optimistic model-selection signal. LOMO and Kaggle LB are the only reliable selection criteria once temporal features are removed.

### Month-prior tilt post-processing (E53–E55, works on LB)

Let:
- \(\pi_{\text{train}}(c)\): train prior (empirical class frequency on train),
- \(\pi_{\text{ext}}(c\mid m)\): external month prior (GBIF seasonal indices).

We adjust predictions by a month-specific prior tilt:

\[
q_{i,c} \propto p_{i,c}\cdot\Big(\frac{\pi_{\text{ext}}(c\mid m_i)}{\pi_{\text{train}}(c)}\Big)^{\alpha_{m_i}}
\quad\text{then renormalize rows.}
\]

Equivalently (log-space):

\[
\log q_{i,c} = \log p_{i,c} + \alpha_{m_i}\log\Big(\frac{\pi_{\text{ext}}(c\mid m_i)}{\pi_{\text{train}}(c)}\Big) - \log Z_i.
\]

Mechanism: within each month, multiplying scores by a constant does not change within-month ranking, but it *does* re-rank examples **across months**, which affects \(\text{AP}_c\). This is why month priors can move the leaderboard.

### Uncertainty-gated priors (E67, next step)

Uniform month tilts can harm confident cases. Define per-sample uncertainty via the top-2 margin:

\[
\text{margin}_i = p_{i,\text{top1}} - p_{i,\text{top2}}.
\]

Apply the prior tilt only when the model is uncertain:

\[
q_{i,c} \propto p_{i,c}\cdot\Big(\frac{\pi_{\text{ext}}(c\mid m_i)}{\pi_{\text{train}}(c)}\Big)^{\alpha_{m_i}\,\mathbf{1}[\text{margin}_i<\tau]}
\quad\text{then renormalize.}
\]

This makes the correction *example-dependent* (can improve rankings within a month) while preserving confident predictions.

### Blending / ensembling predictions

Given two predictors \(P^A, P^B \in \mathbb{R}^{N\times K}\), a linear blend is:

\[
P = (1-w)P^A + wP^B,\quad w\in[0,1]
\]

with optional row-renormalization. Month-aware blending uses \(w=w(m)\) per month (e.g., adjust only for months 9/10).

### Pseudo-labeling (why it can collapse under shift)

Soft pseudo-labeling adds unlabeled test samples back into training using model probabilities as targets (or weighted hard-label expansions). This creates a feedback loop: if the pseudo-label distribution is biased (common under shift + imbalance), training moves the model toward a self-consistent but wrong fixed point. E62 is the canonical failure: very high SKF but LB collapse.

### Label-shift correction caveat

Class-prior estimation methods based on inverting a confusion matrix (e.g., BBSE / EM label-shift corrections) can be numerically unstable when the matrix is ill-conditioned or when assumptions (\(p(x\mid y)\) invariant) are violated. In our setting, naïve implementations can collapse to a single class (Waders), so any such method needs strong regularization and careful validation.

### Product-of-experts post-processing (E73–E77, LB ≥ 0.58)

The key discovery after the 0.56 plateau is that **month-prior tilt alone is not enough**: it mainly re-ranks examples *across months*, but does not reliably correct *within-month* confusions on the unseen months \(\{2,5,12\}\).

We therefore treat post-processing as **sequential Bayesian updates** of an existing posterior:

1) **Base model** (learned from trajectory features + non-privileged tabular):
\[
p_{i} = f_\theta(x_i)\in \Delta^{K-1}.
\]

2) **Month prior (label-shift) correction** (E54/E67):
\[
p^{(m)}_{i,c}\propto p_{i,c}\cdot\Big(\frac{\pi_{\text{GBIF}}(c\mid m_i)}{\pi_{\text{train}}(c)}\Big)^{\alpha_{m_i}}
\]
optionally **uncertainty gated** by \(\mathbf{1}[\text{margin}_i<\tau]\).

3) **Physics likelihood correction** (E73+): introduce per-sample “evidence” \(u_i\) consisting of stable physical cues, and apply a tempered product-of-experts update:
\[
q_{i,c}\propto p^{(m)}_{i,c}\cdot P(u_i\mid c)^{\gamma},
\quad \gamma\in[0,1].
\]

The exponent \(\gamma\) is essential because (a) the NB independence assumption is false and (b) \(u\) is not conditionally independent of \(x\) (double-counting). \(\gamma\) acts as a *temperature / trust* parameter.

#### Naive Bayes factorization used

We approximate class-conditional evidence as:
\[
P(u\mid c)\approx P(\text{size}\mid c)\cdot \prod_{j} \mathcal{N}(u_j\mid \mu_{c,j},\sigma_{c,j}^2)
\]
with Laplace smoothing for `radar_bird_size` and diagonal Gaussians for continuous cues.

Experiments:

- **E73 / E74 (LB 0.58)**: \(u=(\text{radar\_bird\_size},\;\text{airspeed})\).
- **E75 (LB 0.59, best)**: \(u=(\text{radar\_bird\_size},\;\text{airspeed},\;\text{alt\_mid},\;\text{alt\_range})\), where
  - \(\text{alt\_mid}=(\text{min\_z}+\text{max\_z})/2\),
  - \(\text{alt\_range}=\text{max\_z}-\text{min\_z}\).

These improvements are consistent with the claim:
> Remaining leaderboard headroom is dominated by **within-unseen-month ranking errors**, and stable physics cues provide extra information to re-rank those cases.

#### Uncertainty gating for the evidence update

To avoid perturbing confident predictions, we apply the \(P(u\mid c)^{\gamma}\) factor only when the base is uncertain:
\[
\text{margin}_i=p_{i,\text{top1}}-p_{i,\text{top2}},\qquad
q_i=
\begin{cases}
p^{(m)}_i & \text{if }\text{margin}_i\ge \tau_{NB}\\
\text{Renorm}\big(p^{(m)}_i\odot P(u_i\mid \cdot)^{\gamma}\big) & \text{if }\text{margin}_i< \tau_{NB}.
\end{cases}
\]

This gate is what allows the evidence update to change **within-month rankings** without destroying already-correct confident rankings.

#### When the invariance assumption breaks (E76, E77)

The evidence update implicitly assumes that \(P(u\mid y)\) is approximately **domain-invariant** across train→test months.

- **E76 (LB 0.58, worse than E75)** added `n_pts` / `duration` as evidence. These variables are strongly affected by the **tracking / segmentation process**, so \(P(u\mid y)\) is not stable across domains (violating the key assumption). The correction becomes mis-specified and hurts LB.
- **E77 (LB 0.58)** tried month-specific \(\gamma_m\) by *reducing* May correction. The LB drop implies either (a) May still benefits from the same magnitude of physics correction, or (b) the public test May subset differs from our “May≈April” similarity assumption. For now, keep a single \(\gamma\) across unseen months.
