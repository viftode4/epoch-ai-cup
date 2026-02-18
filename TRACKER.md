# Technique Tracker

Baseline: **E25D no temporal + weakclass = 0.7050 mAP** (119 feats, LGB+XGB+CB, logit adj)
Previous "best" (overfit): ~~E15 stack + logit adj = 0.7535~~ (inflated by temporal features, LB = 0.52)
LB score with E25D: **TBD** (submitted 2026-02-14)

Status legend: `pending` | `discussing` | `testing` | `kept` | `discarded` | `skipped`

---

## Critical Bug Fix (2026-02-13)

`LabelEncoder.fit(CLASSES)` was sorting alphabetically, but `CLASSES` in `src/data.py` was
in a custom order. Result: 6 of 9 per-class labels were wrong in all print output, and all
submission CSVs had misaligned columns. **Overall mAP values were always correct** (internally
consistent), and all model training was correct (labels and predictions used the same encoding).

Fixed by sorting CLASSES alphabetically in `src/data.py` to match LabelEncoder.

### Corrected class picture (E11)

| Class | N | E11 AP | Was mislabeled as |
|---|---|---|---|
| Cormorants | 40 | 0.308 | "Pigeons" |
| Birds of Prey | 108 | 0.607 | "Clutter" |
| Waders | 120 | 0.649 | "Songbirds" |
| Ducks | 58 | 0.711 | same |
| Geese | 83 | 0.773 | same |
| Songbirds | 483 | 0.829 | "Waders" |
| Pigeons | 122 | 0.880 | "Birds of Prey" |
| Clutter | 84 | 0.939 | "Cormorants" |
| Gulls | 1503 | 0.960 | same |

Key insight: **Cormorants** (not Pigeons) is the real weakest class. Clutter is easy (RCS signature),
Pigeons are actually decent (0.88). This reframes what "hard" means for our problem.

---

## A. TSC Architectures

| # | Technique | Ref# | Status | Result | Delta | Notes |
|---|-----------|------|--------|--------|-------|-------|
| T01 | QUANT (quantile intervals) | 18 | discarded | 0.5845 | +0.10 standalone | E13a: see discussion |
| T02 | Hydra+MultiRocket | 19 | discarded | 0.47/0.35 | -0.01/-0.13 | E13b/c: see discussion |
| T03 | InceptionTime | 20 | pending | -- | -- | see discussion |
| T04 | LITE / LITETime | 21 | pending | -- | -- | see discussion |
| T05 | ConvTran (CNN-Transformer) | 22 | skipped | -- | -- | see discussion |
| T06 | MOMENT foundation model | 23 | discarded | 0.2275 | -0.30 standalone | E18: domain gap too large |
| T07 | Series2Vec (self-supervised) | 24 | skipped | -- | -- | If MOMENT fails, Series2Vec won't help |

### T01: QUANT -- DISCARDED

**What**: Quantile-interval features over dyadic intervals on raw + diff + FFT representations,
fed to an extra-trees classifier. Completely different feature family from Rocket (quantile vs
convolution). Available in aeon as `QUANTClassifier`.

**Hypothesis**: Different feature extraction = diversity in stacking. Could be a 5th base model.

**Result (E13a)**: Standalone 0.5845 -- much better than MiniRocket (0.48) and on par with CNN
(0.52). However, replacing MiniRocket in the E11 stack **hurt** by -0.003, and adding as a 5th
model didn't help either.

**Why it failed for stacking**: QUANT's quantile features likely correlate with the tree ensemble's
feature-based view. MiniRocket's random convolutions, despite lower standalone mAP, provide more
**orthogonal errors** to the tree ensemble -- exactly what stacking rewards. This confirms the
ablation finding: **diversity matters more than standalone accuracy** for stacking.

### T02: Hydra + MultiRocket -- DISCARDED

**What**: Hydra = competing kernel groups (dictionary-based + convolution hybrid). MultiRocket =
multiple pooling operators + input transforms extending MiniRocket. Both in aeon.

**Hypothesis**: Direct upgrade from MiniRocket (E08, 0.48). Better standalone should improve stack.

**Result (E13b/c)**: Hydra+LogReg = 0.35 (terrible), MultiRocket+LogReg = 0.47 (slightly worse
than MiniRocket). Neither improved the stack when substituted for E08.

**Why**: Hydra's 4096 features may be too few for our multivariate 8-channel data (vs MiniRocket's
~10K). MultiRocket's 49,728 features may cause overfitting with LogReg on 2601 samples. The
LogReg classifier is the bottleneck -- these transforms might work better with Ridge or LightGBM
on top. But given the stacking lesson from T01, even better standalone may not translate.

### T03: InceptionTime -- PENDING

**What**: 5-ensemble of Inception networks with multi-scale kernels {39, 19, 9}. ~420K params each.
Available in tsai library.

**Discussion**: Multi-scale kernels are conceptually similar to what we built in E16 (improved CNN
with kernel sizes 3, 7, 15). InceptionTime ensembles 5 networks with different random seeds,
which is the same idea as snapshot ensembles (T17). With 420K params x 5 = 2.1M total params on
2601 samples, overfitting is the main risk. However, the official 5-ensemble + early stopping is
well-tested on small UCR datasets. Worth testing after E16 results are in -- if our improved CNN
already reaches 0.55+, InceptionTime may not add enough diversity.

### T04: LITE / LITETime -- PENDING

**What**: Only 9,814 params -- 2.34% of InceptionTime. DepthWise Separable Conv + dilated conv +
40 handcrafted filters. LITEMV for multivariate.

**Discussion**: The parameter efficiency is very appealing for 2601 samples. If InceptionTime
overfits (likely), LITE is the fallback. Could also serve as the "student" for knowledge
distillation (T18) if we build a good teacher first. Priority depends on E16 CNN results.

### T05: ConvTran -- SKIPPED

**What**: CNN-Transformer hybrid, #1 on UEA multivariate.

**Discussion**: Skipping for now. Transformers need many samples to learn attention patterns.
With 2601 training samples, the attention mechanism will almost certainly overfit. ConvTran's
reported results are on datasets with much more data. If we had 10x more samples (via pseudo-
labeling), revisit.

### T06: MOMENT Foundation Model -- DISCARDED

**What**: Pretrained on "Time Series Pile" (large-scale TS corpus). MOMENT-1-large (d_model=1024).
Linear probing, LGB on embeddings, or combined with handcrafted features.

**Result (E18)**: Catastrophic. Ridge on MOMENT embeddings = 0.2275 mAP. LGB = 0.2199.
Combined MOMENT+handcrafted = 0.6461 (worse than handcrafted alone at 0.7451).
Neither replacing CNN in stack (-0.003) nor adding as 5th model (-0.002) helped.

**Why it failed**: MOMENT was pretrained on fundamentally different data (weather, ECG, finance).
Radar bird trajectory has domain-specific patterns (RCS signatures, flight altitude profiles,
bearing changes) that are nothing like general time series. The 1024-dim embeddings capture
zero useful information for our task -- Gulls AP=0.79 (majority class memorization) while every
other class is below 0.33. The "pretrained backbone sidesteps small-sample" promise only works
when the pretraining domain overlaps with the target domain.

**Lesson**: Foundation models require domain overlap. For niche signals like radar, domain-specific
feature engineering (our 105 features) vastly outperforms general pretrained representations.

### T07: Series2Vec -- SKIPPED

**What**: Self-supervised pretraining predicting temporal + spectral similarity.

**Discussion**: Skipped. If MOMENT (pretrained on millions of time series) can't learn useful
representations for our radar data, training from scratch on 4473 samples won't help either.
The fundamental issue is domain specificity, not pretraining scale.

---

## B. Class Imbalance & Calibration

| # | Technique | Ref# | Status | Result | Delta | Notes |
|---|-----------|------|--------|--------|-------|-------|
| T08 | Post-hoc logit adjustment | 25 | kept | 0.7451 | +0.0056 | E12: see discussion |
| T09 | Effective Number of Samples | 26 | kept | 0.7535 | +0.0084 | E15: beta=0.999 NEW BEST |
| T10 | SOAP direct AP optimization | 27 | pending | -- | -- | see discussion |
| T11 | Dynamic-Recall Focal Loss | 28 | pending | -- | -- | see discussion |
| T12 | Decoupled training | 29 | pending | -- | -- | see discussion |
| T13 | Per-class isotonic calibration | 30 | discarded | 0.7270 | -0.018 | E14: see discussion |
| T14 | GETS ensemble temperature scaling | 31 | discarded | 0.7398 | +0.0002 | E14: see discussion |

### T08: Post-hoc Logit Adjustment -- KEPT (NEW BEST)

**What**: After training, shift probabilities: `adjusted = probs * (prior ** -tau)`, then
renormalize. Sweep tau per class on OOF predictions. Zero retraining cost.

**Hypothesis**: Minority classes (Cormorants, Ducks) get systematically under-ranked by models
trained on imbalanced data. Logit adjustment boosts their predictions in the ranking.

**Result (E12)**: Global tau=-0.24 gives +0.0035. Per-class greedy tau optimization gives
**+0.0056** (0.7396 -> 0.7451). Biggest gains: Cormorants +0.035 (0.308 -> 0.342),
Ducks +0.014 (0.711 -> 0.725), Songbirds +0.007. Slight losses on Waders (-0.006).

**Why it works**: The per-class tau allows different boost levels per class. Cormorants (40
samples, prior=0.015) gets the strongest adjustment. The negative tau (unusual -- typically
positive in the literature) is because our models with `is_unbalance` already over-correct for
class frequency, so we're actually **dialing back** the over-correction slightly for majority
classes while still boosting the smallest minorities.

**Application**: Applied as the final post-processing step on any stack. Will re-optimize tau
whenever the base stack changes.

### T09: Effective Number of Samples -- KEPT (NEW BEST)

**What**: Class weight = (1-beta) / (1-beta^n_c). Tunable beta controls smoothness: beta=0.9
is gentle, beta=0.99999 is close to inverse-frequency. Applied during tree ensemble training.

**Hypothesis**: is_unbalance (inverse-frequency) may be too aggressive for Cormorants (40
samples, getting 34.9x weight). Effective Number smoothing finds a better balance that doesn't
amplify noise in tiny classes.

**Result (E15)**:
- Beta sweep: 0.9=0.7359, 0.99=0.7445, **0.999=0.7447**, 0.9999=0.7374, 0.99999=0.7376
- inv_freq baseline: 0.7307 (slightly worse than E10's 0.7322 due to variance)
- Best beta=0.999: tree standalone 0.7451 (+0.0129 vs E10)
- Optimal tree weights shifted: CB=80% (was 60%), LGB=15%, XGB=5%
- Stack with E08+E06+E09: 0.7493 (tree=75%, cnn=10%, svm=10%, rocket=5%)
- + logit adjustment: **0.7535** (new overall best, +0.0084 vs E12)

**Why it works**: beta=0.999 gives Gulls weight=0.116 (down from ~1.0 with inv-freq),
Cormorants weight=2.3 (moderate boost), Ducks weight=1.6. This is gentler than raw
inverse-frequency (which gave Cormorants ~35x). The smoother weighting prevents the models
from amplifying noise in the smallest classes while still rebalancing attention from Gulls.
CatBoost benefits most — its 80% weight (up from 60%) suggests it's best at leveraging
the reweighted samples.

### T10: SOAP Direct AP Optimization -- PENDING

**What**: LibAUC library. Optimizes AUPRC (our metric) as the loss function for deep learning.

**Discussion**: Only applicable to CNN/neural network models. If E16 improved CNN produces a
decent standalone score, replacing cross-entropy with SOAP could give another bump. But SOAP
adds training complexity (surrogate loss, momentum, specific optimizer). Try after E16 baseline.

### T11: Dynamic-Recall Focal Loss -- PENDING

**What**: Focal loss weighted by per-class recall. Classes with low recall (Cormorants ~0.3)
auto-get higher focus during training.

**Discussion**: Interesting but only for neural networks. Similar intent to T09 for trees.
Could combine with E16 CNN. Lower priority than T10 (SOAP) since SOAP directly targets our
metric while focal loss targets classification difficulty.

### T12: Decoupled Training -- PENDING

**What**: Stage 1: learn representations with instance-balanced (no class weights). Stage 2:
retrain classifier head with class-balanced weights. For trees: train without is_unbalance,
then apply T08 logit adjustment post-hoc.

**Discussion**: We already partially do this -- E11 stacking + E12 logit adjustment is
effectively "learn ensemble predictions, then adjust calibration post-hoc." The specific test
would be: train E10 trees WITHOUT is_unbalance, check if T08 logit adjustment compensates.
If T09 beta sweep already shows that lighter weights are better, this is redundant.

### T13: Per-class Isotonic Calibration -- DISCARDED

**What**: Fit isotonic regression per class on OOF to recalibrate probabilities.

**Result (E14)**: -0.018 mAP. Catastrophic for Cormorants (-0.058) and Ducks (-0.025).

**Why it failed**: Isotonic regression is non-parametric and needs enough calibration data per
class. With 5-fold CV, each fold has only ~8 Cormorants and ~12 Ducks for fitting. The isotonic
function overfits to noise in these tiny bins. For mAP (which cares about ranking), isotonic
regression can actually swap rankings of borderline samples, destroying the signal.

**Lesson**: Non-parametric calibration methods are dangerous when any class has <50 samples per fold.

### T14: GETS Ensemble Temperature Scaling -- DISCARDED

**What**: Per-model temperature T_i on logits before blending. T > 1 smooths, T < 1 sharpens.

**Result (E14)**: Best = CNN T=1.5, SVM T=0.7, trees T=1.0, MiniRocket T=1.0. Only +0.0002
improvement. When T08 logit adjustment is layered on top, lands at exactly 0.7451 -- same as
E12 alone.

**Why negligible**: Temperature scaling adjusts calibration (confidence vs accuracy), but mAP
depends on **ranking** which is preserved under monotone transforms. Temperature only helps if
it changes the relative ordering between models' contributions, which barely happens when the
tree ensemble dominates at 70% weight.

---

## C. CNN Training Tricks

| # | Technique | Ref# | Status | Result | Delta | Notes |
|---|-----------|------|--------|--------|-------|-------|
| T15 | Label smoothing + repr soft labels | 32 | discarded | 0.5193 | -0.005 | E16: combined with T16+T17, see discussion |
| T16 | SWA (stochastic weight averaging) | 33 | discarded | 0.5193 | -0.005 | E16: combined, see discussion |
| T17 | Snapshot ensembles | 34 | discarded | 0.5193 | -0.005 | E16: combined, see discussion |
| T18 | KDCTime knowledge distillation | 35 | pending | -- | -- | see discussion |
| T19 | TTA (test-time augmentation) | 36 | pending | -- | -- | see discussion |

### T15+T16+T17: Combined CNN Improvements -- DISCARDED

**What**: E16 combines all three plus data augmentation into one improved CNN:
- T15: Label smoothing alpha=0.1 (prevents overconfident predictions)
- T16: SWA in final 50 epochs (finds flatter optima)
- T17: 5 snapshots from cosine annealing restarts (free ensemble)
- Plus: jitter, scaling, window warping augmentation
- Plus: multi-scale first layer (kernels 3, 7, 15 -- InceptionTime-inspired)
- Plus: 8 channels x 128 timesteps (up from E06's 6ch x 64)

**Result (E16)**: Overall 0.5193 -- slightly WORSE than E06 baseline (0.5238, -0.005).
Per-fold: snapshot ensemble won fold 0 (0.5452), SWA won fold 2 (0.5128), best_single won
folds 1,3,4. No consistent winner among the three methods.

**Why it failed**: 2601 samples is simply too few for a CNN to learn effectively, regardless
of training tricks. The deeper multi-scale architecture (more params) combined with 200 epochs
likely still overfits despite augmentation + label smoothing. The original E06 with its simpler
architecture (3 blocks, 6ch x 64) and early stopping at 120 epochs was actually better tuned
for this data size.

**Stacking impact**: Replacing E06 at 10% gives 0.7400 (+0.0004 vs E11, within noise). Higher
CNN weight hurts. The CNN slot in the stack is capped by the data size.

**Lesson**: For 2601 samples, training tricks can't overcome the fundamental data limitation.
Need a pretrained backbone (T06 MOMENT) or more data (T20-23 pseudo-labeling) to meaningfully
improve the CNN slot.

### T18: Knowledge Distillation -- PENDING

**What**: Train a large "teacher" (InceptionTime or our best CNN), then distill into a small
"student" (LITE) using calibrated soft labels.

**Discussion**: Requires a good teacher first. If E16 produces a strong CNN (0.55+) or if we
implement InceptionTime (T03), distillation into LITE could give a parameter-efficient model
that generalizes better. Deferred until we have a teacher worth distilling from.

### T19: TTA (Test-Time Augmentation) -- PENDING (HIGH PRIORITY)

**What**: At inference, create 10 augmented views (3 window slices + 3 jitter + 2 scale +
original), average predictions via geometric mean. Free +0.01-0.03 mAP.

**Discussion**: Completely free at inference -- no retraining needed. Apply to any CNN model's
test predictions. Should be done AFTER we finalize the best CNN (E16 or later), as the last
step before submission. Very high priority due to zero cost.

---

## D. Pseudo-Labeling

| # | Technique | Ref# | Status | Result | Delta | Notes |
|---|-----------|------|--------|--------|-------|-------|
| T20 | DARP (distribution-aligning refinery) | 37 | discarded | 0.7487 | -0.0048 | E17: see discussion |
| T21 | Multi-model agreement filter | 38 | discarded | 0.7487 | -0.0048 | E17: see discussion |
| T22 | Soft pseudo-labels + K-fold isolation | 39 | discarded | 0.7487 | -0.0048 | E17: see discussion |
| T23 | Per-class adaptive thresholds | 40 | discarded | 0.7487 | -0.0048 | E17: see discussion |

### T20-T23 Combined: Pseudo-Labeling -- DISCARDED

**What**: Multi-model agreement filter (3+/4 models agree), per-class adaptive confidence
thresholds, DARP distribution cap (2x class prior), soft labels from stacked model.

**Result (E17)**: 400 pseudo-labeled samples (21.4% of test). Overall **-0.0048** vs E15.
- Tree alone: 0.7401 (-0.0050 vs E15's 0.7451)
- Stack: 0.7454 (-0.0039 vs E15's 0.7493)
- Stack + logit adj: 0.7487 (-0.0048 vs E15's 0.7535)

**Why it failed**: The fundamental Catch-22 of pseudo-labeling on imbalanced data:
- Classes that NEED more data (Cormorants=0, Pigeons=0, Waders=0, BoP=5, Ducks=3) got
  almost no pseudo-labels because models disagree on them (low agreement + low confidence).
- Classes already well-served (Gulls=293, Songbirds=58, Clutter=41) dominated pseudo-labels.
- Adding 293 more Gulls to an already 57.8% Gull dataset made imbalance WORSE.
- Even with DARP distribution cap and effective number weights, the noise from uncertain
  pseudo-labels (especially Clutter false positives) outweighed the benefit.
- 4/4 model agreement: 0 for BoP, Cormorants, Pigeons, Waders -- the models fundamentally
  disagree on minority classes because they're genuinely hard to classify.

**Lesson**: Pseudo-labeling requires the model to already be good at the classes it needs
help with. Circular dependency. Would need a fundamentally different approach for minority
classes (e.g., active learning with expert annotation, or synthetic oversampling T24-T26).

---

## E. Time Series Oversampling

| # | Technique | Ref# | Status | Result | Delta | Notes |
|---|-----------|------|--------|--------|-------|-------|
| T24 | T-SMOTE | 41 | pending | -- | -- | see discussion |
| T25 | Evo-TFS | 42 | skipped | -- | -- | see discussion |
| T26 | CFAMG counterfactual augmentation | 43 | pending | -- | -- | see discussion |

### T24: T-SMOTE -- PENDING (LOW PRIORITY)

**What**: SMOTE adapted for time series, preserving temporal structure.

**Discussion**: Could help Cormorants (40 samples) and Ducks (58 samples). But Cormorants
already has 0.939 AP for Clutter... wait, with corrected labels, Cormorants is at 0.308.
So augmenting Cormorant training data could be valuable. However, SMOTE on radar time series
may generate unrealistic trajectories. Lower priority than pseudo-labeling (T20-23) which
uses real test data.

### T25: Evo-TFS -- SKIPPED

**What**: Genetic programming to evolve synthetic samples.

**Discussion**: Implementation complexity is very high for uncertain benefit. The genetic
programming framework would need custom fitness functions for radar data. Skip unless T24
shows oversampling works at all.

### T26: CFAMG -- PENDING (LOW PRIORITY)

**What**: VAE disentangling causal vs non-causal factors for counterfactual augmentation.
18-67% improvement over baseline oversampling reported.

**Discussion**: Requires training a VAE on our 2601 samples. The VAE itself may overfit. If
pseudo-labeling (T20-23) gives enough extra data, this becomes unnecessary. Try only if
pseudo-labeling fails or is insufficient.

---

## Overall Strategy Notes

### What we've learned so far

1. **Post-processing has a ceiling**: Logit adjustment (+0.006) was the only winner among 4
   calibration techniques tested. Isotonic, temperature scaling all failed or tied. The predictions
   from E11 are already well-calibrated for ranking -- there's little left to squeeze out post-hoc.

2. **Stacking diversity > standalone accuracy**: QUANT (0.58) lost to MiniRocket (0.48) in the
   stack because MiniRocket's errors are more orthogonal to the tree ensemble. New base models
   must provide genuinely different error patterns, not just higher standalone mAP.

3. **Base model improvement is the path forward**: The tree ensemble (0.73) dominates the stack
   at 70%. Even a small improvement in tree mAP translates almost 1:1 to the stack. Similarly,
   improving the CNN from 0.52 to 0.60+ would increase its useful weight in the blend.

4. **Cormorants is the real bottleneck**: With 40 samples and 0.308 AP, Cormorants drags the
   macro-average down by ~0.08 vs if it were at the mean (0.74). This is the single highest-
   leverage class to improve.

### CRITICAL: Temporal Overfitting Discovery (2026-02-14)

**Root cause of 0.52 LB score identified.** Train/test have different month distributions:
- Train months: [1, 4, 9, 10] (Jan, Apr, Sep, Oct)
- Test months: [2, 5, 9, 10, 12] (Feb, May, Sep, Oct, Dec)
- 33% of test data from months NEVER in training (Feb, May, Dec)

18 temporal features (month, hour, dayofweek, is_october, oct_afternoon, is_migration, etc.)
were learning spurious correlations like "Pigeons = October afternoon" that don't exist in test.

**Impact on all previous results:**
- ALL E01-E23 CV scores are inflated by ~0.05 from temporal overfitting
- StratifiedKFold cannot detect this because all folds share the same months
- E15's 0.7535 CV -> 0.52 LB = 0.23 gap (temporal overfit + observation_id leakage + distribution shift)
- E25 Config A (with temporal): CV=0.7450, predicts **0 Pigeons** in test (clearly broken)
- E25 Config D (no temporal+weakclass): CV=0.7050, predicts 7 Pigeons, 42 Waders (more realistic)

**What this means going forward:**
- All experiments must exclude the 18 TEMPORAL_OVERFIT features
- CV scores ~0.70 are the new honest baseline (StratifiedKFold, no temporal)
- Previous technique comparisons (T08, T09, etc.) should still be relatively valid since the temporal bias affects all configs equally
- Stacking experiments need to be re-run without temporal features

### Priority stack (what to do next)

1. ~~Finish E15 (T09)~~ DONE: 0.7535 (overfit CV)
2. ~~E16 (T15-17)~~ DONE: 0.5193, discarded (worse than E06)
3. ~~T20-23 Pseudo-labeling~~ DONE: 0.7487, discarded (-0.0048)
4. ~~T06 MOMENT~~ DONE: 0.2275 standalone, discarded
5. ~~E25 Temporal overfitting fix~~ DONE: 0.7050 (no temporal + weakclass + logit adj)
6. ~~Submit E25D to Kaggle~~ DONE: LB = 0.51
7. ~~E32-E34 Honest evaluation sprint~~ DONE: E32=0.6808, logit adj NEGATIVE
8. ~~E36-E38 External features~~ DONE: E38=0.3615 LOMO, LB=0.53 (BEST)
9. ~~E39 Stacking~~ DONE: +0.004 LOMO (noise), sequence models can't generalize
10. ~~E40 Augmentation~~ DONE: nothing beats baseline
11. ~~E41 Month-adaptive post-processing~~ DONE: submitted alpha=0.2

---

## F. Flight Behavior Physics & Novel Techniques (Round 3, 2026-02-16)

| # | Technique | Status | Result | Delta | Notes |
|---|-----------|--------|--------|-------|-------|
| T27 | Cross-channel correlation features | discarded | 0.3577 | -0.003 | E44: feature dilution on LOMO, Clutter +0.037 but Ducks -0.054 |
| T28 | Biomechanics composite features | discarded | 0.3577 | -0.003 | E44: combined with T27-T31 as 24 features |
| T29 | Enhanced RCS modulation features | discarded | 0.3577 | -0.003 | E44: combined with T27-T31 |
| T30 | 3D trajectory geometry features | discarded | 0.3577 | -0.003 | E44: combined with T27-T31 |
| T31 | Multi-scale & complexity features | discarded | 0.3577 | -0.003 | E44: combined with T27-T31 |
| T32 | Path signatures (iisignature) | pending | -- | -- | E45: time-invariant trajectory encoding |
| T33 | Zaugg CWT + SVM stacking | pending | -- | -- | E46: SVM on spectral features |
| T34 | TTA (test-time augmentation) | pending | -- | -- | E47: free inference boost |

### T27-T31 Combined: Physics Features -- DISCARDED (E44)

**What**: 24 physics-based features in 5 groups:
- A. Cross-channel coupling (6): speed_alt_corr, speed_rcs_corr, bearing_rcs_corr, etc.
- B. Biomechanics composites (6): bounding_index, glide_ratio, thermal_score, wing_loading_proxy
- C. Enhanced RCS modulation (4): rcs_mod_depth, periodicity_idx, bimodality, fluctuation_power
- D. 3D trajectory geometry (4): vert_horiz_ratio, alt_trend_r2, traj_aspect_ratio, alt_entropy
- E. Multi-scale & complexity (4): sinuosity_ratio, rcs_var_ratio, speed_trend, perm_entropy

**Hypothesis**: Season-invariant features based on aerodynamics should help LOMO generalization.

**Result (E44)**: LOMO 0.3577, delta = **-0.003** vs E38 base (0.3611).
- Per-class: Clutter +0.037, BoP +0.002, Cormorants +0.004, Waders +0.001
- But: Ducks -0.054, Geese -0.014, Pigeons -0.006
- Feature stats: all 24 features have good variance and unique values (no degenerate features)

**Why it failed**: The extra 24 features (139->163) cause feature dilution in the tree ensemble.
With only 4 LOMO folds and 2601 samples, the models can't effectively use 163 features without
overfitting. The Clutter improvement (+0.037) suggests some features have signal, but the noise
from 20+ extra features overwhelms it. Consistent with the ablation finding that features are
saturated for trees (core+tab 69 feats = 0.6994, best 105 = 0.7010).

**Lesson**: Adding more features to tree models on 2601 samples with LOMO evaluation doesn't work.
The model capacity for generalization across months is already maxed out. These features might
work better with a feature-selection step or as inputs to a different model type (SVM, Ridge).

### T32: Path Signatures -- DISCARDED (E45)

**What**: Mathematical framework encoding multi-dimensional paths as hierarchical feature
vectors via iterated integrals. Time-reparameterization invariant by construction.

**Hypothesis**: Since signatures are invariant to sampling rate and temporal stretching, a
bird's signature should be the same regardless of when in the year it was recorded. This
directly addresses our temporal shift problem.

**Implementation**: `esig` library (iisignature failed on numpy 2.x). 4 channels (alt, RCS,
speed, bearing_change). `extract_path_signature_features()` added to src/features.py.

**Results (E45 LOMO)**:
- A: E38 base (139 feats): 0.3606
- B: +sig depth-2 lead-lag (212 feats): 0.3479 (-0.013)
- C: +sig depth-3 no-LL (224 feats): 0.3463 (-0.014)
- D: +phys+sig d2 (261 feats): CRASHED (duplicate feature columns)

**Verdict**: DISCARDED. Path signatures are mathematically elegant but add 73-85 features
to a tree model trained on ~2000 LOMO samples. Feature dilution dominates. The invariance
property doesn't compensate for the curse of dimensionality on small folds.

### T33: Zaugg CWT + SVM Stacking -- DISCARDED (E46)

**What**: Use existing `extract_zaugg_cwt_features()` (67 features) with SVM classifier
as a stacking component.

**Hypothesis**: CWT features HURT when given to tree models (-0.009) but the literature
shows SVM achieves AUC 0.965+ on the same features. The features aren't bad -- they need
the right model. SVM with RBF/Laplace kernel captures spectral shape that trees can't.

**Results (E46 LOMO)**:
- SVM standalone (67 CWT features): 0.2493
- Tree standalone (E38, 139 features): 0.3597
- Best blend (5% SVM): 0.3605 (+0.0008 = noise)
- SVM per-month: Month 1: 0.287, Month 4: 0.334, Month 9: 0.220, Month 10: 0.327

**Verdict**: DISCARDED. SVM on CWT features is much weaker than trees (0.25 vs 0.36).
Zaugg (2008) worked on single wingbeat extraction at >100Hz sampling -- our 1-5Hz radar
can't resolve wingbeat modulation. CWT features at our sampling rate capture only coarse
spectral structure, not the fine wingbeat patterns that SVM excels at. No diversity gain.

### T34: Test-Time Augmentation -- DISCARDED (E47)

**What**: At inference, create multiple augmented views of each test trajectory, predict
all, average predictions. Free boost with no retraining.

**Implementation**: 10 augmented views with 2% Gaussian multiplicative noise on tabular
features. Average predictions across augmentations.

**Results (E47 LOMO)**:
- Tree baseline: 0.3565
- Tree TTA (10 views, 2% noise): 0.3448 (-0.012)
- TTA improved Month 1 (0.294->0.330) but hurt Month 10 (0.387->0.390 minimal) and
  overall was worse due to Clutter dropping 0.543->0.497 and Gulls 0.857->0.837.

**Verdict**: DISCARDED. Noise injection hurts tree models. Trees make axis-aligned splits
on exact feature values -- small perturbations push samples across decision boundaries
unpredictably. TTA works for neural nets (smooth decision surfaces) not trees (discontinuous).

### T35: MultiRocket + Stacking -- DISCARDED (E47)

**What**: MultiRocket (49728 features from convolution kernels) + Ridge classifier as
stacking component alongside tree ensemble.

**Results (E47 LOMO)**:
- MultiRocket+Ridge standalone: 0.2064 (terrible -- extreme overfit with 49728 features)
- Tree standalone: 0.3565
- Best blend (15% MultiRocket): 0.3576 (+0.001 = noise)
- MiniRocket (E39) was 0.245 LOMO, MultiRocket is even worse at 0.206

**Verdict**: DISCARDED. With only 2601 samples, 49728 features is absurd. Ridge
regularization can't compensate for this extreme dimensionality mismatch. MultiRocket
also can't generalize across months -- same fundamental limitation as CNN/MiniRocket.
