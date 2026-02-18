# Research Notes -- Bird Radar Classification

## Table of Contents
1. [Paper References](#papers--references)
2. [Ablation Findings](#ablation-findings)
3. [Next Strategy: Heterogeneous Feature-Partitioned Stacking](#next-strategy)
4. [Rocket/MiniRocket for Time Series](#rocketminirocket)
5. [Data Augmentation for CNN](#data-augmentation-for-cnn)
6. [Feature-Model Interactions](#feature-model-interactions)

---

## Papers & References

### 1. Wingbeat Identification via CWT (Zaugg et al. 2008)
- **Paper**: "Automatic identification of bird targets with radar via patterns produced by wing flapping"
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC2607429/
- **Method**: CWT with Morlet wavelet on RCS signal, 32 frequency bands (0.31-65 Hz). Per-band mean AND std = 64 features + 3 signal stats = 67 total. Model-selected 43 of 67.
- **Model**: SVM with Laplace kernel (C=100, sigma=0.1). Laplace outperformed RBF, polynomial, and linear kernels.
- **Results**: AUC 0.965-0.995 within-dataset, 0.887-0.990 cross-dataset
- **Critical detail**: Used 32 frequency bands with per-band stats (64 features), NOT coarse 4-band energy summaries. Our E05 implementation was too coarse (9 CWT features).
- **Why SVM, not trees**: Laplace kernel (L1 distance) captures the entire spectral profile shape simultaneously. Trees split one feature at a time -- inefficient for correlated spectral bands.

### 2. Large vs Small Bird Radar Signatures (Gong et al. 2020)
- **Paper**: "Comparison of radar signatures based on flight morphology for large birds and small birds"
- **Link**: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2020.0064
- **Key finding**: Flight morphology differences (not raw RCS level) are the primary discriminator. RCS variance from wing flapping is more stable than absolute RCS.

### 3. Flight Mode Classification from Radar (Gong & Yan 2019)
- **Paper**: "Using Radar Signatures to Classify Bird Flight Modes Between Flapping and Gliding"
- **Link**: https://www.semanticscholar.org/paper/Using-Radar-Signatures-to-Classify-Bird-Flight-and-Gong-Yan/ac6d236c75a26e98ab7a57a72de40064d1b82b52
- **Species patterns**: Pigeons=continuous flap, Gulls=long glide, BoP=soaring, Songbirds=bounding (flap-pause), Waders=continuous wingbeats

### 4. Universal Wingbeat Frequency Scaling (PLOS ONE 2024)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11152310/
- **Law**: WBF = 2.4 * mass^(-0.38) Hz
- **Expected ranges**: Songbirds 8-20 Hz, Pigeons 5-7 Hz, Gulls 3-5 Hz, Geese 2.5-4 Hz, BoP minimal (soaring)

### 5. ML Algorithms in Radar Ornithology (Rosa et al. 2016)
- **Link**: https://onlinelibrary.wiley.com/doi/abs/10.1111/ibi.12333
- **Finding**: Random Forest held accuracy >0.80 for all tasks. SIX algorithms tested -- validates tree ensemble approach.

### 6. ROCKET: Random Convolutional Kernels for TSC (Dempster et al. 2020)
- **Paper**: "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels"
- **Link**: https://arxiv.org/abs/1910.13051
- **GitHub**: https://github.com/angus924/rocket
- **Method**: 10,000 random 1D convolutional kernels (lengths {7,9,11}, random weights/bias/dilation/padding). Two pooling ops per kernel (max + PPV). Linear classifier on 20,000 features.
- **Results**: State-of-the-art accuracy on UCR archive, fraction of computational cost of deep learning.

### 7. MiniRocket (Dempster et al. 2021)
- **Paper**: "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification"
- **Link**: https://arxiv.org/abs/2012.08791
- **GitHub**: https://github.com/angus924/minirocket
- **Method**: Fixed kernel length 9, 84 fixed weight patterns {-1,2}, only PPV pooling. ~9,996 features. Up to 75x faster than ROCKET.
- **Results**: Same accuracy as ROCKET. All 109 UCR datasets in <10 minutes.
- **Multivariate support**: Yes, via aeon/sktime libraries.

### 8. MultiRocket (Tan et al. 2022)
- **Paper**: "MultiRocket: multiple pooling operators and transformations for fast and effective time series classification"
- **Link**: https://link.springer.com/article/10.1007/s10618-022-00844-1
- **Method**: Extends MiniRocket with multiple pooling operators + input transformations. Competitive with HIVE-COTE 2.0 (most accurate TSC method).

### 9. Data Augmentation Survey for TSC (Iwana & Uchida 2021)
- **Paper**: "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks"
- **Link**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841
- **Code**: https://github.com/uchidalab/time_series_augmentation
- **Tested**: 12 augmentation methods on 128 UCR datasets across 6 architectures.
- **Best methods**: (1) Window Warping, (2) Window Slicing, (3) DGW (Discriminative Guided Warping)
- **Harmful methods**: Rotation (destroys temporal semantics), Permutation (destroys order), global Time Warping
- **Key finding**: CNNs (VGG) benefit most from augmentation. RNNs (LSTM-FCN) often hurt by augmentation.

### 10. Comprehensive Augmentation Survey (2023)
- **Paper**: "Data Augmentation for Time-Series Classification: An Extensive Empirical Study"
- **Link**: https://arxiv.org/abs/2310.10060
- **Tested**: 20 strategies on 15 UCR datasets with ResNet and LSTM.
- **Finding**: Combining jitter+scaling+magnitude warping+permutation boosted Parkinson's sensor classification from 77.52% to 86.88%.

### 11. Mixup for Time Series (Zhang et al. 2018 + extensions)
- **Mixup original**: https://arxiv.org/pdf/1710.09412
- **Remix for imbalanced data**: https://arxiv.org/abs/2007.03943 (Chou et al. 2020 -- shifts labels toward minority class)
- **Balanced Mixup**: https://github.com/agaldran/balanced_mixup (Galdran et al. MICCAI 2021 -- pairs minority with majority)
- **Multivariate TSC**: https://arxiv.org/abs/2201.11739 -- tested on 26 MTS datasets, accuracy gains 1-45%, strongest on small datasets.

### 12. Label Smoothing for TSC (2024)
- **Paper**: "Improving Time Series Classification with Representation Soft Label Smoothing"
- **Link**: https://arxiv.org/abs/2408.17010
- **Typical value**: alpha=0.1 (reduce to 0.05 if combined with Mixup)

### 13. Test-Time Augmentation
- **Paper**: Shanmugam et al. (ICCV 2021) "Better Aggregation in Test-Time Augmentation"
- **Link**: https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf
- **Expected gain**: +1-3% accuracy. Use geometric mean of softmax outputs.

### 14. Kaggle Ensemble Strategies
- **NVIDIA Grandmaster Stacking**: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
- **KDnuggets Ensembles Part 3**: https://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p3.html
- **H2O Stacked Ensembles**: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
- **Key principle**: Different feature sets per model multiplies diversity. An SVM on wavelet features + a tree on tabular features > both on the same features.

### 15. Deep Learning for Aviation Bird Safety (2025)
- **Link**: https://arxiv.org/html/2602.07019
- **Finding**: Image-based CNNs achieve 92.8% on 24 species. Paper states "avian radars cannot identify bird species" -- confirms radar-only classification is genuinely hard.

### 16. Radar Post-Processing (Erp et al. 2024)
- **Link**: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14249
- **Method**: birdR R package for Robin Radar 3D-Fix data quality control.

### 17. Robin Radar MAX System
- **Link**: https://www.robinradar.com/products/max-radar
- **Info**: The exact radar that collected our data. Multiple stacked beams for altitude.

---

## Ablation Findings

Systematic test run 2026-02-13 (see EXPERIMENTS.md for full table).

### Feature ablation (LGB only, same hyperparams)
| Config | #Feats | mAP | Delta vs core+tab |
|--------|--------|------|-----|
| core only | 53 | 0.6236 | -0.076 |
| **core+tab** | **69** | **0.6994** | **baseline** |
| core+fft+tab | 73 | 0.6963 | -0.003 (FFT hurts!) |
| core+fft+tab+tgt | 93 | 0.6993 | -0.000 |
| core+tab+wav | 78 | 0.6900 | -0.009 (wavelet hurts!) |
| core+tab+flight | 81 | 0.6948 | -0.005 (flight hurts!) |
| core+fft+tab+tgt+flight | 105 | 0.7010 | +0.002 (best) |
| kitchen_sink | 114 | 0.6925 | -0.007 |

### Model ablation (best feature set, 105 feats)
| Model | mAP |
|-------|------|
| XGB alone | 0.7094 |
| CatBoost alone | 0.7024 |
| LGB alone | 0.7010 |
| LGB+CB | 0.7226 |
| XGB+CB | 0.7209 |
| LGB+XGB | 0.7158 |
| **LGB+XGB+CB** | **0.7239** |

### Key conclusions
1. **Features saturated for trees**: core+tab (69) = 0.6994, best combo (105) = 0.7010. Delta = 0.0016.
2. **Ensemble is what matters**: best single model -> 3-model = +0.0145.
3. **CWT/FFT/flight features HURT trees when added alone** but become neutral in larger sets.
4. **HOWEVER**: features that hurt TREES may help OTHER model types (see next section).

---

## Next Strategy

### Heterogeneous Feature-Partitioned Stacking

The ablation proved that for tree models, features are saturated. But Zaugg 2008 achieved AUC 0.96+ with CWT features + SVM. The features aren't bad -- they're being used by the wrong model.

**Why SVM handles wavelet features better than trees** (Ref: [Nature Scientific Reports](https://www.nature.com/articles/s41598-023-33215-x)):
1. **Trees make axis-aligned splits**: inefficient for correlated spectral bands
2. **SVM kernels compute distances across the full spectral profile**: captures shape
3. **High-dim small-sample**: 64 CWT features + 2601 samples = SVM's sweet spot
4. **No feature dilution**: SVM uses all features simultaneously, trees have selection lottery

**Proposed architecture:**
```
Level 0 (Base Models, each with own feature set):
  A: LGB+XGB+CB ensemble  on core+tabular (69 feats)     -- proven 0.7239
  B: SVM (Laplace kernel)  on CWT wavelet feats (64)       -- Zaugg approach
  C: MiniRocket transform  on raw trajectory -> Ridge/LR   -- random kernel TSC
  D: 1D-CNN (augmented)    on raw time series (8ch x 128)  -- learned patterns

Level 1 (Meta-learner):
  Logistic Regression on OOF predictions: 9 classes x 4 models = 36 inputs
```

Each model sees a DIFFERENT view of the data. Diversity comes from features AND algorithms.

---

## Rocket/MiniRocket

### Why it fits our problem
- Designed for small TSC datasets (works on UCR datasets with 16-24,000 samples)
- No learned parameters in kernels -- only the linear classifier trains (low overfitting)
- Handles multivariate time series natively (our 6-8 channel radar tracks)
- Extremely fast: 5-fold CV on 2601 samples in <1 minute

### Implementation plan
- Library: `aeon` (recommended) or `sktime`
- Pad/interpolate trajectories to fixed length (128 steps)
- Input: (N, 8, 128) -- 8 channels: alt, RCS, speed, bearing_change, lon_delta, lat_delta, rcs_deriv, alt_deriv
- Transform: MiniRocket -> 9,996 features
- Classifier: LogisticRegressionCV for calibrated probabilities (needed for mAP metric)
- Alternative: feed Rocket features to LGB for probability outputs

### Speed estimate
| Step | Time |
|------|------|
| MiniRocket transform (per fold) | ~1-2s |
| LogisticRegressionCV fit | ~0.5s |
| **5-fold CV total** | **~10-30s** |

### Sources
- [ROCKET paper](https://arxiv.org/abs/1910.13051)
- [MiniRocket paper](https://arxiv.org/abs/2012.08791)
- [aeon MiniRocket API](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.convolution_based.MiniRocketClassifier.html)
- [sktime MiniRocketMultivariate](https://www.sktime.net/en/stable/examples/transformation/minirocket.html)

---

## Data Augmentation for CNN

### Recommended augmentations (ranked by evidence)

| Method | Parameters | Evidence | DO use? |
|--------|-----------|----------|---------|
| **Window Warping** | window_ratio=0.1, scales=[0.5, 2.0], p=0.5 | #1 in Iwana 2021 survey | YES |
| **Window Slicing** | reduce_ratio=0.9, p=0.5 | #2 in Iwana 2021 | YES |
| **Jittering** | sigma=0.02, p=0.3 | Standard, conservative for radar | YES |
| **Scaling** | sigma=0.1, p=0.3 | Simulates distance variation | YES |
| **Mixup** | alpha=0.2 (Beta distribution) | Regularizer, helps imbalanced | YES |
| **Label smoothing** | alpha=0.1 (0.05 if with Mixup) | Prevents overconfidence | YES |
| **TTA** | 5 window slices + 3 jitters, average | +1-3% accuracy | YES |
| Rotation/flipping | - | "Significantly degraded" in survey | NO |
| Permutation | - | "Severely detrimental" | NO |
| Global Time Warping | - | Over-transforms, hurts periodic signals | NO |

### Expected improvement
- Augmentation alone: CNN from 0.52 to ~0.60-0.67 mAP (based on +5-12% accuracy in comparable benchmarks)
- TTA on top: +0.01-0.03 mAP additional
- Better blend weight -> higher ensemble lift

### Sources
- [Iwana & Uchida 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841)
- [Augmentation parameter docs](https://github.com/uchidalab/time_series_augmentation/blob/master/docs/AugmentationMethods.md)
- [Comprehensive survey 2023](https://arxiv.org/abs/2310.10060)
- [Remix for imbalanced](https://arxiv.org/abs/2007.03943)

---

## Feature-Model Interactions

### Why wavelet features fail with trees but succeed with SVMs

| Property | Trees (LGB/XGB/CB) | SVM (RBF/Laplace kernel) |
|----------|--------------------|-----------------------|
| Decision boundary | Axis-aligned splits | Smooth hypersurface |
| Correlated features | Redundant splits, dilution | Natural via kernel distance |
| Spectral profiles | Must staircase-approximate | Computes shape similarity |
| Feature selection | Random lottery per split | Uses ALL features simultaneously |
| Small-sample high-dim | Overfits or ignores features | Margin maximization regularizes |

**References**:
- [SVM vs XGBoost](https://www.geeksforgeeks.org/support-vector-machine-vs-extreme-gradient-boosting/)
- [Axis-aligned vs oblique boundaries](https://www.researchgate.net/figure/A-decision-boundary-generated-by-a-an-axis-aligned-and-b-an-oblique-split-function_fig1_290508933)
- [SVM and Boosting comparison](https://www.cs.toronto.edu/~huang/courses/csc2515_2020f/readings/SVM-and-boosting.pdf)
- [SVM vs RF learning differences](https://www.nature.com/articles/s41598-023-33215-x)

### Heterogeneous stacking evidence from Kaggle
- KDD Cup winner: 7 feature sets x multiple algorithms = 64 base models ([KDnuggets](https://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p3.html))
- Otto Classification 1st: diverse trees + SVM + RF stacked with LR ([Toptal](https://www.toptal.com/machine-learning/ensemble-methods-kaggle-machine-learn))
- H2O docs explicitly recommend "different predictor columns across models" ([H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html))

---

## Current Model Performance

**Best: E11 stacking -- CV mAP 0.7396** (70% tree + 10% Rocket + 10% CNN + 10% SVM)

| Class | E11 AP | Samples | Key challenge |
|-------|--------|---------|---------------|
| Gulls | 0.960 | 1503 | Solved |
| Cormorants | 0.939 | 40 | Solved despite tiny sample |
| Birds of Prey | 0.880 | 108 | Good |
| Waders | 0.829 | 120 | Good |
| Geese | 0.773 | 83 | OK |
| Ducks | 0.711 | 58 | Improving, still overlaps Pigeons |
| Songbirds | 0.649 | 483 | Weak -- confused with Gulls |
| Clutter | 0.607 | 84 | Weak -- RCS is main signal |
| Pigeons | 0.308 | 122 | Very weak -- overlaps everything |

---

## Research Round 2 (2026-02-13): New Techniques

### A. TSC Architectures (2024-2026)

**18. QUANT — Quantile Interval Classifier (Dempster 2024)**
- Link: https://github.com/angus924/quant
- Method: Quantiles over dyadic intervals on raw + diff + diff2 + FFT representations. Extra-trees classifier.
- In aeon: `aeon.classification.interval_based.QUANTClassifier`
- Results: SOTA on 142 UCR datasets, < 15 min total compute. Matches HIVE-COTE 2.0.
- **Why use**: Completely different feature family from Rocket (quantile vs convolution). Trivial to add as 5th stacking model.

**19. Hydra+MultiRocket (Dempster 2023-2024)**
- Link: https://github.com/angus924/hydra
- Method: Competing convolutional kernel groups + MultiRocket features. Dictionary-based + convolution hybrid.
- Results: Not statistically different from HIVE-COTE 2.0, <0.5% of compute.
- **Why use**: Direct upgrade from MiniRocket (E08, 0.48). Available in aeon.

**20. InceptionTime (Fawaz et al. 2020, still top-tier)**
- Link: https://arxiv.org/abs/1909.04939 | GitHub: https://github.com/hfawaz/InceptionTime
- Architecture: 5-ensemble of Inception networks. Multi-scale kernels {39,19,9}, residual blocks, ~420K params each.
- In tsai: `InceptionTime`, `InceptionTimePlus`
- **Why use**: Multi-scale kernels capture different wingbeat frequencies simultaneously. Ensemble of 5 reduces overfitting.

**21. LITE / LITETime (Ismail Fawaz et al. 2023-2024)**
- Link: https://arxiv.org/abs/2409.02869 | GitHub: https://github.com/MSD-IRIMAS/LITE
- Architecture: Only **9,814 params** (2.34% of InceptionTime). DepthWise Separable Conv + dilated conv + 40 handcrafted filters + multiplexing.
- LITEMV for multivariate. LITETime = ensemble of 5.
- Results: Comparable to InceptionTime, 2.78x faster. Ranks 2nd on UEA multivariate.
- **Why use**: With 2601 samples, LITE's parameter efficiency is a huge advantage over InceptionTime. Less overfitting.

**22. ConvTran (Foumani et al. 2024)**
- Link: https://link.springer.com/article/10.1007/s10618-023-00948-2
- Method: CNN-Transformer hybrid with tAPE (time Absolute Position Encoding) + eRPE.
- Results: #1 on UEA multivariate archive.
- Caveat: Transformers are parameter-hungry, risky on 2601 samples.

**23. MOMENT Foundation Model (ICML 2024)**
- Link: https://arxiv.org/abs/2402.03885 | HuggingFace: AutonLab/MOMENT-1-large
- Method: Pretrained on "Time Series Pile". Linear probing or fine-tuning for classification.
- **Why use**: Pretrained backbone >> training from scratch on 2601 samples. Could massively improve CNN component.

**24. Series2Vec (Foumani et al. 2024)**
- Link: https://arxiv.org/abs/2312.03998 | GitHub: https://github.com/Navidfoumani/Series2Vec
- Method: Self-supervised pretraining predicting similarity in temporal + spectral domains.
- Results: 82.47% UCR accuracy (beats all SSL methods).
- **Why use**: Pretrain on ALL 4473 samples (2601 train + 1872 test, unsupervised), then fine-tune on labeled. Semi-supervised.

### B. Class Imbalance & Calibration

**25. Post-hoc Logit Adjustment (Menon et al. 2021)**
- Link: https://arxiv.org/abs/2007.07314
- Method: After training, shift logits: `adjusted = probs * (prior ** -tau)`, tune tau on OOF.
- **Why use**: Zero-cost post-processing. Boosts minority class predictions without retraining.

**26. Effective Number of Samples (Cui et al. 2019)**
- Link: https://arxiv.org/abs/1901.05555
- Method: Weight = (1-beta)/(1-beta^n). Beta=0.9999 gives: Gulls 1x, Pigeons 11.5x, Ducks 24.1x, Cormorants 34.9x.
- **Why use**: More principled than raw inverse-frequency. Tunable beta avoids Pigeon-steals-from-Ducks.

**27. SOAP — Direct AP Optimization (NeurIPS 2021)**
- Link: https://arxiv.org/abs/2104.08736 | Library: https://docs.libauc.org/examples/auprc.html
- Method: Directly optimizes AUPRC as loss for deep learning. LibAUC library.
- **Why use**: Directly optimizes our competition metric instead of cross-entropy.

**28. Dynamic-Recall Focal Loss (2024)**
- Link: https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2411845
- Method: Focal loss weighted by per-class recall. Low-recall classes (Pigeons) auto-get higher weight.
- **Why use**: Auto-adapts to class difficulty during training, unlike static class weights.

**29. Decoupled Training (Kang et al. 2020)**
- Link: https://arxiv.org/abs/1910.09217
- Method: Stage 1: learn representations with balanced sampling. Stage 2: retrain classifier with class-balanced weights.
- **Why use**: For trees: train without class weights, then apply logit adjustment post-hoc. Avoids minority noise overfitting.

**30. Per-Class Isotonic Calibration**
- Link: scikit-learn CalibratedClassifierCV
- Method: Fit isotonic regression per class on OOF predictions. Non-parametric, can change rankings.
- Caveat: May overfit on small classes (24 Pigeons per fold).

**31. GETS — Ensemble Temperature Scaling (ICLR 2025)**
- Link: https://openreview.net/pdf?id=qgsXsqahMq
- Method: Per-component temperature scaling for ensemble models.
- **Why use**: Each of our 4 base models has different calibration. Per-model temperature before blending.

### C. CNN Training Tricks

**32. Label Smoothing + Representation Soft Labels (2024)**
- Link: https://arxiv.org/abs/2408.17010
- Method: Soft labels based on L2 distance in encoder latent space. +7.14% on small InceptionTime.
- Parameters: gamma=0.001, temperature tau in [2,4,10], standard alpha=0.1 (0.05 with Mixup).

**33. SWA — Stochastic Weight Averaging**
- Link: https://arxiv.org/abs/1803.05407
- Method: Average weights from multiple SGD trajectory points. Finds flatter optima.
- PyTorch: `torch.optim.swa_utils.AveragedModel`
- Expected: +0.5-1.5% on small datasets. Practically free.

**34. Snapshot Ensembles (Huang et al. 2017)**
- Link: https://openreview.net/pdf?id=BJYwwY9ll
- Method: Save weights at each cosine annealing cycle minimum. "Train 1, get M for free."
- Expected: +1-3% over single model.

**35. KDCTime — Knowledge Distillation with Calibration (2022)**
- Link: https://arxiv.org/abs/2112.02291
- Method: InceptionTime teacher -> LITE student with calibrated soft labels.
- **Why use**: Teacher ensemble captures inter-class similarities (Pigeon-Duck overlap). Student learns soft boundaries with 10K params.

**36. TTA — Test-Time Augmentation**
- Link: https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf
- Protocol: 3x window slice + 3x jitter + 2x scale + original = 10 views, geometric mean.
- Expected: +0.01-0.03 mAP. Free at inference.

### D. Pseudo-Labeling

**37. DARP — Distribution Aligning Refinery (NeurIPS 2020)**
- Link: https://arxiv.org/abs/2007.08844
- Method: Aligns pseudo-label distribution to true class distribution via convex optimization.
- **Why use**: Prevents majority-class bias in pseudo-labels (model would over-predict Gulls).

**38. Multi-Model Agreement Filter**
- Source: UPS (https://arxiv.org/abs/2101.06329) + Kaggle Grandmasters Playbook
- Method: Only pseudo-label where all 4 model families agree + low entropy.
- **Why use**: Our heterogeneous stacking is the best defense against confirmation bias.

**39. Soft Pseudo-Labels + K-Fold Isolation**
- Source: NVIDIA Kaggle Grandmasters (https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Key: Use probability vectors not hard labels. Compute K separate pseudo-label sets for K-fold.

**40. Per-Class Adaptive Thresholds (FlexMatch/SEVAL)**
- Links: FlexMatch https://arxiv.org/abs/2110.08263 | SEVAL https://arxiv.org/abs/2407.05370
- Method: Different confidence thresholds per class. Pigeons 0.75, Gulls 0.98.

### E. Time Series Oversampling

**41. T-SMOTE (IJCAI 2022)**
- Link: https://www.ijcai.org/proceedings/2022/334
- Method: SMOTE adapted for time series preserving temporal structure.

**42. Evo-TFS (2026)**
- Link: https://arxiv.org/abs/2601.01150
- Method: Genetic programming to evolve synthetic samples in time+frequency domains.

**43. CFAMG — Counterfactual Minority Augmentation (KDD 2025)**
- Link: https://haoxuanli-pku.github.io/papers/KDD%2025%20-%20Mitigating%20Data%20Imbalance%20in%20Time%20Series%20Classification%20Based%20on%20Counterfactual%20Minority%20Samples%20Augmentation.pdf
- Method: VAE disentanglement of causal vs non-causal factors, generates counterfactual minority samples.
- Results: 18-67% improvement over best baseline oversampling.

### F. Augmentation Parameters (Radar/Sensor)

| Augmentation | Parameter | Value | Notes |
|-------------|-----------|-------|-------|
| Window Warping | window_ratio=0.1, scales=[0.5,2.0] | p=0.5 | #1 in survey |
| Window Slicing | reduce_ratio=0.9 | p=0.5 | #2 in survey |
| Jittering | sigma=0.02-0.03 | p=0.3 | Conservative for radar |
| Scaling | sigma=0.1 | p=0.3 | Distance variation |
| Manifold Mixup | alpha=0.2 at hidden layer | p=0.5 | Better than input Mixup |
| Label Smoothing | alpha=0.1 (0.05 with Mixup) | always | Prevents overconfidence |
| CutMix | alpha=1.0 | p=0.3 | Uniform segment selection |

Radar-specific: be conservative with RCS jittering (most informative channel). Altitude/position carry physical meaning — warping safer than jittering.

---

## Prioritized Implementation Plan (Round 2)

### Tier 1: Quick wins (minutes)
1. **Post-hoc logit adjustment** on E11 OOF — tune tau, zero retraining
2. **QUANT** as 5th stacking model — `aeon`, <1 min
3. **Hydra+MultiRocket** replacing MiniRocket — `aeon`, direct upgrade

### Tier 2: Medium effort (1-2 experiments each)
4. **InceptionTime/LITE** replacing Conv1D CNN — with augmentation + SWA + snapshot ensembles + TTA
5. **Pseudo-labeling** — multi-model agreement + soft labels + per-class thresholds
6. **Effective Number reweighting** — tune beta for class weights

### Tier 3: Higher effort, potentially high impact
7. **MOMENT fine-tuning** — pretrained TS backbone
8. **SOAP direct AP loss** — LibAUC for CNN
9. **CFAMG** counterfactual minority augmentation
10. **Decoupled training** — trees without class weights + post-hoc logit shift

---

## Research Round 3 (2026-02-16): Flight Behavior Physics & Novel Techniques

### Context

Our best model (E38, LB=0.53, LOMO=0.3615) suffers from temporal distribution shift -- train months [1,4,9,10] vs test months [2,5,9,10,12]. The key insight: we should focus on **how birds fly** (physics/biomechanics) rather than **when they fly** (temporal features that leak).

---

### Part 1: Radar Ornithology Literature

#### Wingbeat Frequency (WBF) -- The Gold Standard in Radar Bird ID

**Physics:** As a bird flaps, its wings act as a "corner reflector" -- when perpendicular to radar, RCS spikes by ~10dB. This creates a periodic modulation at the wingbeat frequency.

**Measured frequencies by species group:**

| Group | WBF (Hz) | Source |
|-------|----------|--------|
| Songbirds | 8-20 | Bruderer 2010 |
| Pigeons | 5.5-6.5 | Bruderer 2010, PLOS Bio 2019 |
| Geese | ~5.5 | JEB 2001 |
| Waders | 5-7 (continuous) | Multiple |
| Gulls | 3-4 | Multiple |
| Cormorants | 4-5 (est.) | High wing loading |
| Birds of Prey | Irregular/minimal | Soaring dominates |
| Ducks | 4-6 (est.) | Medium-large |
| Clutter | N/A | No wingbeat |

**Scaling law:** WBF = 2.4 * mass^(-0.38) Hz (allometric relationship)

**OUR LIMITATION:** Robin Radar MAX likely samples at 1-5 Hz -- BELOW Nyquist for most wingbeat frequencies. We CANNOT directly extract wingbeat frequency. BUT:
- RCS modulation depth (amplitude of fluctuation) still works at low sampling
- RCS autocorrelation captures periodicity even if aliased
- RCS variance distinguishes continuous flappers from gliders

**Key paper: Zaugg et al. 2008** -- 32-band CWT (0.31-65 Hz, log-spaced) + SVM achieved AUC 0.965-0.995. We have `extract_zaugg_cwt_features()` in src/features.py but never used it in stacking with SVM. Critical insight: **SVM is much better than trees for spectral features** (trees make axis-aligned splits, can't capture spectral shape).

#### RCS (Radar Cross Section) Patterns

**Absolute RCS as size proxy:**
- Clutter: -13.5 to -9.5 dBm2 (MUCH higher than birds)
- Large birds (Geese, Cormorants): -15 to -20 dBm2
- Medium (Gulls, Pigeons): -24 to -28 dBm2
- Small (Songbirds): -30 to -37 dBm2

**RCS temporal modulation (micro-Doppler):**
- Wings create ~10dB fluctuation during flapping
- Near 0dB during gliding (wings stationary)
- Bimodal RCS distribution common (body-only peak + wing-extended peak)
- Corner reflector effect: larger wings = larger modulation amplitude

**What we already capture:** rcs_mean, rcs_std, rcs_range, rcs_iqr, rcs_skew, rcs_cv, rcs_autocorr_lag1/3, rcs_stability, rcs_zero_cross_rate, rcs_n_peaks_per_sec

**What we're MISSING (high-value gaps):**
- **RCS modulation depth** (P90-P10, more robust than range)
- **RCS periodicity index** (max autocorr at specific lags relevant to bird wingbeats)
- **RCS bimodality** (wings create bimodal distribution -- Hartigan's dip test)
- **RCS fluctuation power** (total oscillation energy in detrended signal)

#### Flight Pattern Classification

**Five main flight modes (from Swiss Birdradar + biomechanics lit):**

1. **Continuous flapping** (Waders, Pigeons, Cormorants)
   - Signal: steady RCS modulation, no pauses
   - Detection: low variance in windowed RCS variance, high autocorrelation

2. **Bounding flight** (Songbirds ONLY, body mass <300g)
   - Pattern: rapid flap burst -> fold wings -> freefall -> repeat
   - Signal: periodic high-variance and near-zero-variance RCS segments
   - Altitude oscillates in phase with flap/fold cycle (0.5-2 Hz)
   - KEY: during bound phase, wings FOLDED -> RCS drops. Alt and RCS are positively correlated.

3. **Flap-gliding** (Gulls, Geese, some Ducks)
   - Pattern: flapping bursts -> extended glides on outstretched wings
   - Signal: long glide segments (low RCS variance)
   - Mean glide duration >> mean flap duration

4. **Soaring/thermal circling** (Birds of Prey)
   - Minimal flapping, circle upward in thermals
   - Near-constant low RCS, altitude gain while going slow
   - High trajectory curvature
   - Median thermal: ~90 seconds

5. **Non-biological** (Clutter)
   - No periodic RCS modulation at bird frequencies
   - Erratic trajectory, very high RCS, short duration

**What we already capture well:** flap_fraction, glide_fraction, n_mode_transitions, soaring_index, curvature, alt_osc_freq, alt_osc_amplitude

**What we're MISSING:**
- **Bounding flight index** (combining altitude oscillation WITH RCS phase correlation)
- **Glide ratio** (L/D estimation from descending segments)
- **Multi-segment analysis** (proportion of track in each flight mode)

---

### Part 2: Bird Flight Biomechanics

#### Species-Specific Flight Characteristics (Season-Invariant)

**What DOESN'T change with season (SAFE features):**
- Wingbeat frequency (set by body mass & wing morphology)
- Flight style (bounding, soaring, flap-glide)
- Glide ratio / lift-to-drag ratio (aspect ratio is fixed)
- RCS modulation pattern (wing shape & flapping kinematics)
- Airspeed range (aerodynamic constraints)
- Trajectory straightness (species-specific navigation)
- Speed variability (CV of speed)

**What DOES change (AVOID):**
- Hour, month, season
- Migration timing
- Absolute altitude preferences (seasonal weather patterns)
- Flock size (varies by season)

#### Aerodynamic Parameters Computable from Our Data

**Glide ratio estimation:**
- During descending + low-RCS-variance segments: horizontal_dist / vertical_loss = L/D proxy
- BoP: 10-20:1, Gulls: 15-25:1 (best gliders), Songbirds: 4-8:1, Cormorants: 6-10:1

**Wing loading proxy:**
- Wing loading = body weight / wing area, determines minimum flight speed
- Proxy: airspeed / (radar_bird_size + 1)
- High wing loading (Cormorants, diving Ducks): fast, continuous flapping
- Low wing loading (BoP, Gulls): slow, can soar/glide

**Flight efficiency:**
- Straight-line distance / total energy proxy
- High AR birds (BoP, Gulls): efficient, long straight glides
- Low AR birds (Songbirds): inefficient, must flap hard

#### Hardest Species Pairs to Distinguish

| Pair | Physical Overlap | Key Differentiator |
|------|------------------|--------------------|
| Pigeons vs Ducks | Both fast (15-20 m/s), medium size, continuous flap | Ducks: lower altitude, water proximity |
| Cormorants vs BoP | Both medium-large, long tracks | Cormorants: STRAIGHT + continuous flap; BoP: CIRCLING + soaring |
| Waders vs Songbirds | Both small-medium, migratory | Waders: CONTINUOUS flap; Songbirds: BOUNDING |
| Geese vs Waders | Both migratory, high altitude in Oct | Geese: FLOCK + slower; Waders: SOLO + faster |

#### Clutter Discrimination
- Clutter is NOT birds -- rain, insects, debris
- Key: RCS = -13.8 dB (birds: -24 to -30 dB) -- MUCH brighter
- No periodic wingbeat modulation
- Short duration, erratic movement
- Already our 2nd-best class (0.533 LOMO) -- less work needed here

---

### Part 3: ML Techniques for Trajectory Classification

#### Path Signatures -- MOST PROMISING

**What:** A mathematical framework that encodes a multi-dimensional path into a hierarchical feature vector. The signature of a path captures all "iterated integrals" up to a chosen depth.

**Why it's perfect for us:**
- **Provably invariant to time reparameterization** -- a cormorant's signature is the same whether the radar samples fast or slow, in January or October
- Works on multi-channel data (altitude + RCS + speed + bearing simultaneously)
- Captures nonlinear interactions between channels (e.g., altitude-RCS coupling)
- Proven effective on small datasets
- Paper: MultiPSCA (2025) beats elastic time series classifiers with minimal tuning

**Feature counts by depth:**
- 4 channels, depth 2: 4 + 16 = 20 features
- 4 channels, depth 3: 4 + 16 + 64 = 84 features
- With lead-lag augmentation (doubles channels to 8): depth 2 = 8 + 64 = 72 features

**Lead-lag embedding:** Doubles channels by appending lagged version. Captures autocorrelation and momentum effects. Systematically outperforms vanilla signatures.

**Libraries:** `iisignature` (Python, fast C++ backend), `esig`, or `signatory` (PyTorch)

#### Cross-Channel Correlation Features

**Currently missing entirely.** We compute features per-channel (altitude stats, RCS stats, speed stats) but never look at HOW channels relate to each other.

**Key correlations:**
- **Altitude-RCS:** Positive in bounding flight (wings fold = low RCS at altitude dip). Negative in soaring (higher altitude = farther from radar = lower RCS). ~0 in straight level flight.
- **Speed-altitude:** Negative for gliders (descend to gain speed). ~0 for powered flight.
- **Bearing-RCS:** High for circling birds (aspect angle changes rapidly).
- **Speed change vs altitude rate:** Captures dive behavior, pull-up maneuvers.

Note: We already have `rcs_alt_corr` in weakclass features (one cross-channel feature). The rest are missing.

#### Test-Time Augmentation (TTA)

**What:** At inference, create multiple augmented versions of each test sample, predict all of them, average the predictions.

**Augmentations for trajectories:**
- Time warping (stretch/compress time axis)
- Gaussian noise injection on channels
- Random crop + resample to original length
- Channel dropout (zero out one channel)
- Time reversal (fly path backwards)

**Expected: +0.005-0.015 LOMO.** Small but free -- no retraining needed.

#### SVM on Spectral Features (Zaugg Approach)

**Key insight from literature:** Tree models make axis-aligned splits that are inefficient for correlated spectral bands. SVM with RBF kernel computes distance across the full spectral profile, capturing spectral shape.

**Our situation:** We have `extract_zaugg_cwt_features()` producing 64 CWT band features. We tried adding these to tree models and they HURT (-0.009). But we never tried SVM specifically on them as a stacking component.

**Zaugg's results:** SVM on 64 CWT features -> AUC 0.965. This was for individual wingbeat extraction though, not track-level classification.

**Plan:** Use SVM as a stacking component alongside trees. Trees handle tabular features, SVM handles spectral features. Blend for diversity.

---

### Part 4: Feature Gap Analysis -- What We Have vs What's Missing

#### Features We Have (Well Covered)
- Core trajectory stats (53 features): alt, RCS, speed, acceleration, bearing, sinuosity, straightness
- Flight mode segmentation: flap/glide fraction, mode transitions, soaring index
- Altitude oscillation: alt_osc_freq, alt_osc_amplitude
- RCS autocorrelation: lag 1, lag 3, zero-crossing rate, peaks per second
- Weakclass composites: RCS stability, straightness, turn angles, altitude dynamics
- External data: weather (11), solar (4), GBIF seasonal indices (10)

#### Features MISSING (High-Value Gaps)

1. **Cross-channel correlations** (speed-altitude, bearing-RCS, speed-RCS coupling)
   - We only have rcs_alt_corr. Missing 5+ other important cross-channel interactions.

2. **Biomechanics composites** (bounding flight index, glide ratio, thermal soaring score, wing loading proxy)
   - Compound features that combine multiple signals to detect specific flight modes.

3. **RCS modulation depth** (P90-P10, more robust than raw range)
   - P90-P10 is less affected by outliers than min-max range.

4. **RCS periodicity index** (max autocorrelation strength at relevant lags)
   - Tests specific lags corresponding to bird-frequency wingbeats at our sampling rate.

5. **Path signatures** (mathematically invariant trajectory representation)
   - Time-reparameterization invariant, captures nonlinear channel interactions.

6. **3D trajectory geometry** (vertical/horizontal velocity ratio, altitude trend, trajectory aspect ratio)
   - How the bird uses 3D space -- separates soaring from level flight from diving.

7. **Multi-scale sinuosity** (sub-track sinuosity to capture behavior changes within a track)
   - A bird that flap-glides has different sinuosity at 10s vs 60s scales.

8. **Path complexity** (permutation entropy -- chaotic for clutter, regular for birds)
   - Information-theoretic measure of trajectory regularity.

---

### Part 5: Implementation Plan (E44-E47)

#### E44: Physics-Based Flight Behavior Features
**Priority: HIGHEST. Implement first.**

Add ~25 new features to `src/features.py` in a new `extract_flight_physics_features()` function:

**A. Cross-channel coupling (6 features):**
- speed_alt_corr: Pearson(speed, altitude) -- negative for gliders
- speed_rcs_corr: Pearson(speed, RCS) -- captures aspect angle effects
- bearing_rcs_corr: Pearson(abs(bearing_change), RCS) -- circling birds
- alt_rate_rcs_corr: Pearson(d(alt)/dt, RCS) -- bounding flight detection
- speed_alt_rate_corr: Pearson(speed, d(alt)/dt) -- dive/climb behavior
- rcs_speed_interaction: mean(RCS * speed) -- energy signature

**B. Biomechanics composites (6 features):**
- bounding_index: alt_osc_amplitude * alt_rcs_corr -- Songbird-specific
- glide_ratio: horizontal_dist / vertical_loss during glide segments
- thermal_score: curvature * alt_gain_rate / speed -- BoP-specific
- wing_loading_proxy: airspeed^2 / rcs_mean (proportional to wing loading)
- flap_regularity: std of flap-segment durations (low = continuous, high = irregular)
- continuous_flap_score: flap_fraction * rcs_autocorr_lag1 -- Wader/Pigeon/Cormorant

**C. Enhanced RCS modulation (4 features):**
- rcs_modulation_depth: P90 - P10 (robust amplitude of RCS fluctuation)
- rcs_periodicity_idx: max(autocorrelation) at lags 2-10 (wingbeat regularity)
- rcs_bimodality: Hartigan's dip test statistic (bimodal = flapping bird)
- rcs_fluctuation_power: variance of first-differenced RCS (total oscillation energy)

**D. 3D trajectory geometry (4 features):**
- vert_horiz_ratio: mean(|d(alt)/dt|) / mean(horizontal_speed) -- soaring vs level
- alt_trend_strength: R^2 of linear fit to altitude -- climbing/descending/level
- trajectory_aspect_ratio: vertical_extent / horizontal_extent
- altitude_entropy: Shannon entropy of altitude histogram (complex altitude use)

**E. Multi-scale features (4 features):**
- sinuosity_ratio: sinuosity(first_half) / sinuosity(second_half) -- behavior change
- rcs_var_ratio: rcs_var(first_half) / rcs_var(second_half) -- mode transitions
- speed_trend: slope of linear fit to speed -- accelerating/decelerating
- permutation_entropy: ordinal pattern complexity of RCS signal

**Evaluate with LOMO CV on E38 pipeline (LGB+XGB+CB).**

#### E45: Path Signatures
**Priority: HIGH. Novel approach, directly addresses temporal shift.**

Install `iisignature`. Extract depth-2 and depth-3 signatures from 4-channel trajectory (altitude, RCS, speed, bearing). Test with lead-lag augmentation. Evaluate as features for trees and as standalone + stacking.

#### E46: Zaugg CWT + SVM Stacking
**Priority: MEDIUM. Features already exist, just need SVM pipeline.**

Use existing `extract_zaugg_cwt_features()` (64 features). Train SVM (sklearn SVC with RBF kernel) as stacking component. Blend with tree ensemble.

#### E47: TTA + MultiRocket
**Priority: MEDIUM-LOW. Quick experiments.**

TTA: augment test trajectories, average predictions. MultiRocket: replace MiniRocket with `aeon.transformations.collection.convolution_based.MultiRocket`.

#### Execution Order
1. E44 -> evaluate LOMO -> if +0.01 or more, keep features
2. E45 -> evaluate LOMO -> if signatures help, integrate into best pipeline
3. E46 -> evaluate SVM diversity -> blend if positive
4. E47 -> evaluate TTA/MultiRocket -> stack if positive
5. Combine best of E44-E47 into a final "E48_combined" submission

#### Verification
- LOMO CV is primary metric for every experiment
- Per-class AP breakdown (especially Cormorants, Pigeons, Waders -- weakest)
- Bootstrap CI: delta > 0.032 to be meaningful
- Submit to Kaggle LB if LOMO beats E38 (0.3615)
