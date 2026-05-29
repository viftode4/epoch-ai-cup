# Final Validated Architecture — Every Decision Justified by Research

## Research-to-Design Mapping

Every component in this architecture traces back to a specific research finding.
Components that FAILED validation are explicitly marked as rejected with the reason.

---

## DECISION 1: How to Represent the Trajectory

### Research says:
- Path signatures (Agent 3): Log-signatures at depth 3 for 4D paths = 30 features.
  Multi-scale (3 windows) = 90 features. Captures cross-channel interactions
  (RCS×altitude = aspect angle), trajectory area (circling vs straight).
  MUST normalize channels first. Novel for radar but proven on trajectory tasks.
- catch22 (timeseries agent): 22 canonical features per channel. Complements
  signatures (statistical vs geometric). 4 channels × 22 = 88 features.
- Our hand-crafted features: 76 from v2 + 10 new (alt_curvature, rcs_lags 2-5,
  speed_cv, speed_ac1, speed_trend, alt_r2) = 86 trajectory features.
- MiniRocket: 10,000 features. Too many for 2601 samples (overfitting).
  REJECTED for base model. Could use for diversity ensemble member if time permits.

### Design:
```
Trajectory features:
  86  hand-crafted (existing v2 + alt shape + RCS multi-lag + speed profile)
  90  log-signatures (depth 3, 3 windows, normalized channels)
  88  catch22 (4 channels × 22)
────
 264  total trajectory features
```

### What each captures that the others miss:
- Hand-crafted: domain-specific physics (RCS_per_alt, slow_flight_frac, airspeed_vs_ground)
- Log-signatures: cross-channel geometry (altitude-RCS coupling, trajectory area)
- catch22: statistical properties (entropy, time-reversibility, nonlinearity)

---

## DECISION 2: How to Represent the Environment

### Research says:
- Data audit: 95 of 138 external columns were unused. 46 validated, 0% NaN.
- Domain validation: Each feature checked against per-class means.
  Confirmed: cloud_cover_low separates Clutter (63.8) from Pigeon (5.0).
  wind_at_bird_alt separates Clutter (29.2) from Pigeon (17.4).
  Caught: era5_wind_shear_100_500 = -wind_speed_100m (redundant, dropped).
  wind_80m = wind_at_bird_alt (r=0.987, dropped).
  grassland_fraction_2km too weak (spread 0.42-0.49, dropped).

### Design:
```
External features (validated, non-redundant):
  46  new external columns
   8  physics score features (cormorant_wind, wader_tidal, bop_soaring, etc.)
   5  derived (true_airspeed, temp_dewpoint_spread, cormorant_residual,
      insect_drift_ratio, wind_support→renamed headwind_at_alt)
   1  predicted flock size (from flock size regressor)
────
  60  total environment + derived features
```

---

## DECISION 3: How to Select Features (MOST IMPORTANT)

### Research says (Agent 6, cross-check):
- E79's 36 features were backward-eliminated optimizing SKF (within-month).
  "The top teams at 0.63 likely found the feature set that balances
  within-month discrimination against cross-month generalization."
- LOMO-optimized feature selection has NEVER been tried.
- This is identified as the "single highest-leverage experiment remaining."

### Design: Cross-Month Stability Selection
Instead of expensive backward elimination, use FEATURE STABILITY RANKING:

```python
# For each of 4 training months (Jan=1, Apr=4, Sep=9, Oct=10):
#   Hold out that month entirely
#   Train on remaining 3 months
#   Compute permutation importance on held-out month
#
# Features that are important in ALL 4 held-out months = month-stable
# Features important in only 1-2 months = month-specific proxies
#
# Selection criterion: min(importance across 4 months)
# This directly selects features whose predictive power GENERALIZES

importance_jan = permutation_importance(model_trained_on_apr_sep_oct, X_jan, y_jan)
importance_apr = permutation_importance(model_trained_on_jan_sep_oct, X_apr, y_apr)
importance_sep = permutation_importance(model_trained_on_jan_apr_oct, X_sep, y_sep)
importance_oct = permutation_importance(model_trained_on_jan_apr_sep, X_oct, y_oct)

# Stability score = minimum importance across all months
stability_score = np.minimum.reduce([importance_jan, importance_apr, importance_sep, importance_oct])

# Select top N features by stability score
selected_features = top_n_by_stability(stability_score, n=80)
```

Why min instead of mean: A feature that's rank-1 in October but rank-200 in January
has a high mean but terrible cross-month stability. Min penalizes this correctly.

Target: ~80-100 features from ~324 candidates. Ratio 1:26-33, comfortable for GBDT.

---

## DECISION 4: What Loss Function to Train With

### Research says (Agent 1, focal loss):
- mAP is a ranking metric. Standard logloss optimizes calibration, not ranking.
- Focal loss helps vs CE (+0.5-3%) but is suboptimal for mAP specifically.
- AP-Loss variants outperform focal loss by +1.5-3.0 AP for ranking.
- OvR decomposition is PROVABLY OPTIMAL for macro mAP:
  macro mAP = average of 9 independent binary APs.
  Each binary AP can be independently optimized.
- LambdaRank directly optimizes pairwise ranking. LightGBM supports it
  natively with eval_metric='map'.
- Per-class gamma strongly supported (+2-3% AP from Equalized Focal Loss).

### Research says (Agent 5, threshold/calibration):
- mAP is invariant to monotonic transforms. ANY monotonic post-hoc
  adjustment (threshold, temperature, isotonic) CANNOT change AP.
- The ONLY post-hoc methods that change rankings inject NEW information.

### Design: Two complementary training approaches

**Primary: 9× OvR LambdaRank (directly optimizes the metric)**
```python
for class_idx in range(9):
    y_binary = (y == class_idx).astype(int)

    # Query groups by month: forces month-invariant ranking optimization
    qids = train_months  # each month = separate query group

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='map',                    # directly optimize MAP
        boosting_type='dart',            # dropout regularization
        n_estimators=1000,
        learning_rate=0.03,              # slower for small data
        num_leaves=31,
        colsample_bytree=0.6,
        subsample=0.7,
        drop_rate=0.15,                  # DART dropout rate
        lambdarank_truncation_level=30,  # focus on top-30 ranking
    )
    ranker.fit(X_train, y_binary, group=month_group_sizes,
               eval_set=[(X_val, y_val_binary)], eval_group=[val_month_sizes])

    test_scores[:, class_idx] = ranker.predict(X_test)
```

Why query groups by month: LambdaRank with MAP metric optimizes the
average MAP across query groups. With months as groups, it optimizes
per-month AP and averages — forcing the ranker to produce good
rankings in EVERY month, not just on average. This is Group DRO for
ranking, built into the training objective.

**Secondary: Multi-class CatBoost with focal loss (for PP compatibility)**
```python
# CatBoost multiclass — needed because NB PP requires probability inputs
# Focal loss via custom objective, per-class gamma
# DART-like: CatBoost doesn't have DART natively but has
# `model_shrink_rate` and `model_shrink_mode='Constant'` for similar effect

cb_model = CatBoostClassifier(
    loss_function='MultiClass',
    auto_class_weights='Balanced',   # macro-mAP = equal class weight
    depth=6,
    l2_leaf_reg=5.0,                 # heavy regularization
    learning_rate=0.03,              # slow
    iterations=2000,
    rsm=0.6,                         # random subspace (like colsample)
    subsample=0.7,
    model_shrink_rate=0.1,           # regularization similar to DART
    early_stopping_rounds=100,
)
```

Why both: The OvR rankers produce ranking scores (not probabilities).
NB PP requires calibrated probabilities to multiply by P(features|class).
The multiclass CatBoost provides these probabilities for PP.
The final submission blends both.

---

## DECISION 5: How to Handle Month Shift During Training

### Research says (Agent 6, cross-check):
- "You have been treating the shift as a post-processing problem (170+ PP
  experiments) when the literature says it should be addressed during training."
- Group DRO: minimize worst-month loss. Sagawa 2020: 10-40pp improvement.
- Density-ratio importance weighting: use p_test(x)/p_train(x) as training
  sample weights. Different from adversarial weighting (which failed in E28).
- Noise-corrected loss: Robust-GBDT shows +10.72% under noise + imbalance.

### Design: Three training-time shift mitigations

**A. Month-grouped LambdaRank (already built into Decision 4)**
Query groups by month = per-month MAP optimization = Group DRO for ranking.

**B. Group DRO sample weighting for multiclass CatBoost**
```python
# After each epoch, compute per-month loss
# Upweight samples from worst-performing month
for fold in cv_folds:
    for epoch in boosting_epochs:
        month_losses = {m: compute_loss(model, X[month==m], y[month==m])
                       for m in [1, 4, 9, 10]}
        worst_month = max(month_losses, key=month_losses.get)
        sample_weights[month == worst_month] *= 1.5  # boost worst month

# Practical: CatBoost doesn't support per-epoch reweighting.
# Alternative: Run 2-3 rounds of training:
#   Round 1: train with uniform weights, identify worst month
#   Round 2: train with 2× weight on worst month
#   Round 3: train with 3× weight on worst month
#   Select round with best LOMO mAP
```

**C. Cross-month feature selection (already built into Decision 3)**
Stability-ranked features inherently resist month-specific overfitting.

---

## DECISION 6: How to Regularize (Prevent Overfitting on 2601 Samples)

### Research says (Agent 6):
- DART boosting: dropout applied to boosting, specifically designed for small
  datasets. Simple config change. Never tried in our 170+ experiments.
- Very low learning rate (0.01-0.03) + more trees + early stopping.
- Heavy colsample (0.5-0.7) more important with small data.
- CatBoost's ordered boosting designed for small datasets.

### Design:
```
All models use:
  - DART boosting (LGB: boosting_type='dart', drop_rate=0.15)
  - Low learning rate: 0.03 (was 0.05)
  - High colsample: 0.6 (was 0.8)
  - High subsample: 0.7 (was 0.8)
  - More min_samples: 20 (was 10)
  - Fewer leaves: 31 (was 63) — simpler trees, more features handle interactions
  - Early stopping: 100 rounds on validation MAP (not logloss!)
```

---

## DECISION 7: How to Ensemble

### Research says (Agent 4, Q10 trick):
- Q10 as direct prediction INVALIDATED. Reverses rankings, shrinks TP-FP gap.
- Mean seed averaging: validated free lunch.
- Rank averaging: better than probability averaging for ranking metrics.
- Power averaging (Laurae): raise predictions to power p, empirically validated
  for AUC/ranking. p > 1 amplifies confident predictions.
- Seed variance as meta-features: validated for stacking.

### Research says (Agent 5, calibration):
- Probability averaging across models works but is sensitive to calibration.
- Rank averaging is immune to calibration differences.
- For mAP specifically, only methods that produce NEW rankings help.

### Design: Rank + Power Averaging
```python
from scipy.stats import rankdata

def rank_power_ensemble(model_preds_list, weights, power=1.5):
    """
    Rank-based ensemble with power averaging for ranking metrics.

    For each class column:
    1. Convert each model's predictions to ranks
    2. Apply power transform (amplifies confident rankings)
    3. Weighted average of powered ranks
    """
    n_samples = model_preds_list[0].shape[0]
    n_classes = model_preds_list[0].shape[1]
    final = np.zeros((n_samples, n_classes))

    for c in range(n_classes):
        powered_ranks = []
        for preds, w in zip(model_preds_list, weights):
            ranks = rankdata(preds[:, c])  # 1 to N
            ranks_norm = ranks / n_samples  # 0 to 1
            powered_ranks.append(w * (ranks_norm ** power))
        final[:, c] = np.sum(powered_ranks, axis=0)

    return final

# Power p > 1: amplifies top rankings (good for AP which cares about precision@top)
# Tune p on OOF: grid search [1.0, 1.25, 1.5, 2.0, 3.0]
```

Why power averaging for mAP: AP is dominated by precision at the TOP of the
ranking. Power > 1 makes the top-ranked samples stand out more, increasing
the gap between rank 1 and rank 2. This directly helps AP by making the
ranking more decisive.

---

## DECISION 8: How to Post-Process

### Research says (Agent 5):
- Monotonic transforms (threshold, temperature, isotonic) CANNOT change mAP.
- ONLY methods that inject NEW information can change rankings.
- NB PP works because P(features|class) provides new ranking signal.

### Research says (Agent 6):
- 20+ different PP strategies all converge to 0.59 on top of E79 base.
- PP is fully explored. Further PP innovation has zero expected value.

### Design: Standard NB PP, applied to multiclass component only
```
NB PP (on CatBoost multiclass probabilities only):
  1. GBIF ratio priors for unseen months (Feb, May, Dec)
  2. 3-channel NB evidence: speed, alt_mid, alt_range
  3. Gated PoE (gamma=0.10, tau=0.25)

NOT applied to OvR ranker scores (rankers already optimize ranking directly,
PP could hurt by introducing miscalibrated probability adjustments)
```

---

## DECISION 9: How Many Seeds and Which CV

### Research says:
- Multi-seed averaging: validated, variance reduction. Diminishing returns after ~20.
- StratifiedGroupKFold: honest evaluation, prevents same-bird leakage.
  E173 showed CB dominates under SGKF.
- StratifiedKFold: E79 used this and got LB 0.59. Has leakage.

### Design:
- 20 seeds for OvR rankers (9 classes × 20 seeds × 5 folds = 900 model trains)
- 10 seeds for multiclass CatBoost (10 × 5 = 50 model trains)
- Use SGKF (honest) for primary evaluation
- Also generate SKF predictions for safety submission (E79 style)

---

## DECISION 10: What NOT to Do (Explicitly Rejected)

| Rejected Component | Research Source | Reason |
|---|---|---|
| Per-class threshold optimization | Agent 5 | mAP invariant to monotonic transforms |
| Per-class temperature scaling | Agent 5 | Same — monotonic can't change rankings |
| Isotonic calibration | Agent 5 | Creates ties, harmful to ranking |
| Q10 as direct prediction | Agent 4 | Reverses rankings, not a known technique |
| GMM on seed predictions | Agent 4 | Statistically dubious with 20-50 points |
| Full species-level (28-class) | Agent 2 | Probability leakage, gradient dilution, 42/68 species <20 samples |
| MiniRocket (10K features) | Timeseries agent | Too many features for 2601 samples |
| End-to-end deep learning | Timeseries agent, E06/E16/E18 | Not enough samples |
| Adversarial reweighting | Agent 6, E28/E30 | Coarse binary weights, consistently harmful |
| MLLS/BBSE label shift | Agent 6, E91/E92/E115 | Unstable, collapses |
| More PP innovation | Agent 6 | 20+ variants all hit 0.59, fully explored |

---

## COMPLETE PIPELINE

```
RAW DATA
    │
    ├── Trajectory → hand-crafted (86) + log-signatures (90) + catch22 (88) = 264
    ├── External CSVs → 46 validated features
    ├── Derived → physics scores (8) + true_airspeed + temp_dewpoint + flock_pred + etc = 14
    │
    Total: ~324 candidate features
    │
    ▼
CROSS-MONTH FEATURE STABILITY SELECTION
    Train on 3 months, importance on held-out month, ×4 months
    Select by min-month importance
    → ~80-100 stable features
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
9× OvR LambdaRank                  CatBoost Multi-class
(LGB, DART, query-by-month)        (Balanced, DART-like, Group DRO)
Directly optimizes per-class AP     Produces calibrated probabilities
20 seeds                            10 seeds
    │                                  │
    ▼                                  ▼
Mean seed average (9 scores)        Mean seed average (9 probs)
    │                                  │
    │                                  ├── NB PP (GBIF + 3-channel evidence)
    │                                  │
    ▼                                  ▼
Rank + Power averaging of OvR scores + CatBoost PP'd probabilities
    │
    ▼
SUBMIT (multiple variants: ranker-only, CB+PP, blended)
```

---

## IMPLEMENTATION ORDER

| Phase | What | Models | Time Est |
|-------|------|--------|----------|
| 0 | Feature extraction (signatures + catch22 + physics + flock) | 0 | 1-2 hours |
| 1 | Cross-month stability feature selection | 4 models | 30 min |
| 2 | OvR LambdaRank (9 classes × 20 seeds × 5 folds) | 900 | 3-4 hours |
| 3 | CatBoost multiclass (10 seeds × 5 folds) | 50 | 1 hour |
| 4 | Rank+power ensemble + NB PP | 0 | 30 min |
| 5 | Submit variants | 0 | 15 min |
| **Total** | | **~950 models** | **~7 hours** |

---

## EXPECTED OUTCOMES

| Submission | Components | Expected LB |
|---|---|---|
| `e175_ranker_raw` | OvR LambdaRank, rank ensemble, no PP | 0.60-0.62 |
| `e175_cb_pp` | CatBoost multiclass + NB PP | 0.59-0.60 |
| `e175_blend_pp` | Ranker + CB blend + PP | 0.60-0.63 |
| `e175_e79_seeds` | E79 features, 20 seeds, rank avg (safety) | 0.59-0.60 |

The ranker approach is the most likely to break 0.59 because:
1. It directly optimizes the evaluation metric
2. Month-grouped queries handle the shift during training
3. DART prevents overfitting
4. Log-signatures add genuinely new trajectory information
5. Stability-selected features resist month proxies
