# Design: Adversarial Weighting + One-vs-Rest Pipeline

Date: 2026-02-15
Status: Draft
Baseline: E25D CV=0.7050, LB~0.40-0.52

## Problem Statement

The CV-LB gap (0.18-0.30) is caused by fundamental distribution shift between train and test:
- Train months [1,4,9,10], test months [2,5,9,10,12]. 33% of test from unseen months.
- Train hours 6-15, test hours 6-12. No afternoon data in test.
- Adversarial AUC = 0.71-0.82 depending on features (0.5 = no shift).
- Even core trajectory features (speed, altitude, sinuosity) shift because bird behavior changes across seasons.
- 5 temporal features still leak in E25D (`is_oct_nov`, `migration_alt`, `migration_speed`, `is_night`, `night_high_alt`).

The macro mAP metric evaluates 9 classes independently, each worth 1/9 regardless of sample count. Current multi-class softmax compresses rare class probabilities because sum-to-1 constraint lets Gulls dominate.

## Design

### Phase 1: Fix Temporal Leaks

Add 5 features to TEMPORAL_OVERFIT filter:
- `is_oct_nov`, `migration_alt`, `migration_speed`, `is_night`, `night_high_alt`

Expected impact: AUC drops from 0.82 to 0.75 (confirmed by adversarial validation).

### Phase 2: Adversarial Sample Weighting

Use the adversarial validation model to reweight training samples:

1. Train adversarial model: binary classifier (train=0, test=1) on feature matrix
2. Get P(test | x) for each training sample
3. Weight_i = P(test | x_i) / P(train | x_i) = p / (1-p)
4. Clip weights to [0.1, 10] to prevent extreme values
5. Apply these weights during tree model training (multiply with existing class weights)

This upweights training samples that look like test data (e.g., morning observations, birds at similar altitudes to test) and downweights train-specific patterns (afternoon observations, April clutter).

Expected impact: CV will drop (we're downweighting easy-to-classify training patterns), but LB should improve because the model focuses on test-like patterns.

### Phase 3: Leave-One-Month-Out CV (LOMO)

Replace StratifiedKFold with Leave-One-Month-Out for honest evaluation:

| Fold | Train months | Val month | Train N | Val N |
|------|-------------|-----------|---------|-------|
| 0 | 4, 9, 10 | 1 | 2380 | 221 |
| 1 | 1, 9, 10 | 4 | 2128 | 473 |
| 2 | 1, 4, 10 | 9 | 2134 | 467 |
| 3 | 1, 4, 9 | 10 | 1161 | 1440 |

This simulates the test scenario where the model must predict on unseen months. CV scores from LOMO should correlate better with LB.

Caveat: fold 3 has 55% of data in validation (October). Consider also running with 3 folds plus "pseudo month 5" (held-out random subset of April+Sep to simulate unseen spring month).

### Phase 4: One-vs-Rest Binary Classifiers

For each of 9 classes, train an independent binary classifier:

```
For class c:
  y_c = (y == c).astype(int)
  ratio = n_neg / n_pos  (for scale_pos_weight)

  Train LGB binary:
    objective: binary
    metric: average_precision  (directly optimizes AP!)
    scale_pos_weight: ratio * adversarial_weight_adjustment

  Train CatBoost binary:
    loss_function: Logloss
    eval_metric: PRAUC
    scale_pos_weight: ratio

  Ensemble: optimize LGB/CB weight per class on OOF

  Output: independent score in [0, 1] for class c
```

Advantages over multi-class:
1. Directly optimizes AUCPR (our metric) instead of log-loss proxy
2. No sum-to-1 constraint -- rare classes get full score range
3. Per-class hyperparameter optimization
4. Per-class feature selection possible

Per-class settings:
| Class | N | Ratio | Regularization | Notes |
|-------|---|-------|----------------|-------|
| Gulls | 1503 | 1:0.7 | Low | Nearly balanced |
| Songbirds | 483 | 1:4.4 | Low-med | |
| Pigeons | 122 | 1:20 | Medium | 89% October, shift risk |
| Waders | 120 | 1:21 | Medium | |
| BoP | 108 | 1:23 | Medium | 45% April, shift risk |
| Clutter | 84 | 1:30 | High | 83% April, major shift risk |
| Geese | 83 | 1:30 | High | |
| Ducks | 58 | 1:44 | High | |
| Cormorants | 40 | 1:64 | Very high | Main bottleneck |

### Phase 5: Per-Class Post-Processing

After OvR training:
1. Per-class logit adjustment on OOF (already proven, +0.006)
2. Probability floor: ensure min score >= 0.001 for all rare classes
3. Score normalization per class to [0, 1] range (if needed)

### Phase 6: Test-Time Prior Estimation (Optional)

Use external ornithological knowledge about Eemshaven bird populations by month:
- Feb: winter residents (Gulls, Ducks, Geese), fewer migrants
- May: spring migration (Waders, Songbirds), breeding BoP
- Sep/Oct: autumn migration (Geese, Songbirds, Waders)
- Dec: winter (Gulls, Ducks, Geese)

Adjust class priors in test predictions based on month composition.

### Phase 7: Ensemble Multi-class + OvR

If OvR and multi-class have complementary errors:
- Blend: alpha * OvR_scores + (1-alpha) * multiclass_scores
- Optimize alpha per class on OOF
- Expected small additional gain from diversity

## Experiments

| ID | Description | Key metric |
|----|-------------|------------|
| E27 | LOMO CV baseline (E25D config, no temporal, StratKF vs LOMO) | CV gap diagnostic |
| E28 | Adversarial sample weighting on E25D config | LB improvement |
| E29 | OvR binary classifiers (9 independent) | Per-class AP, overall mAP |
| E30 | OvR + adversarial weights + LOMO | Best honest CV, submit to LB |
| E31 | Blend OvR + multi-class | Final submission |

## Success Criteria

1. LOMO CV should be lower than StratifiedKFold CV but closer to LB
2. Adversarial weighting should reduce CV-LB gap
3. OvR should improve minority class APs (Cormorants, Ducks, Clutter)
4. LB score > 0.52 (beating the temporal-overfit baseline)
5. Target: LB > 0.60

## Risks

1. Adversarial weighting might make the model focus too much on the overlap months (Sep+Oct) since those most resemble test
2. OvR loses multi-class ranking information ("this is not a Gull, so maybe Cormorant")
3. LOMO CV has high variance (only 4 folds, unbalanced fold sizes)
4. With 40 Cormorants, even binary classification is very noisy
