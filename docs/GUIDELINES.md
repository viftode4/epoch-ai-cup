# Experiment Guidelines

Rules for running methodologically sound experiments. Written after discovering
temporal overfitting (E15: CV=0.7535, LB=0.52) and audit of E25D-E31 inflation sources.

## 1. Feature Hygiene

- **Import `ALL_TEMPORAL` from `src/features.py`**. Never hardcode temporal feature lists.
  ```python
  from src.features import ALL_TEMPORAL
  keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
  ```
- Any new timestamp-derived feature requires an adversarial AUC check before use.
  If adversarial importance rank <= 10 AND classification rank > 50, it is a leak.
- **Print feature count** in every experiment output. No silent feature addition.
- Train months: [1, 4, 9, 10]. Test months: [2, 5, 9, 10, 12].
  33% of test comes from unseen months. Calendar features cannot generalize.

## 2. Cross-Validation Rules

- **Primary metric**: StratifiedKFold 5-fold (seed=42) + bootstrap 95% CI.
  Use `bootstrap_map_ci()` from `src/metrics.py`.
- **Secondary**: LOMO (Leave-One-Month-Out, 4 folds) as lower bound.
  LOMO is harsh (typically ~0.35) but reflects real temporal generalization.
- **A delta must exceed 2x bootstrap std to be considered real.**
  E32 bootstrap std = 0.016. So a delta must exceed 0.032 to be real.
  Most "gains" from E25-E31 are within noise.
- Use RepeatedStratifiedKFold (5x5) for variance estimation on important experiments.
- StratifiedKFold **cannot** detect temporal overfitting. All folds share the same months.

## 3. No Optimization on Evaluation Data

- **Ensemble weights**: Fixed (0.33/0.33/0.34 for LGB/XGB/CB) unless split-half
  evaluation shows > 0.005 gain from tuning. Always report the fixed-weight score.
- **Logit adjustment**: Use fixed tau (e.g. rarity-scaled) OR split-half optimized.
  **Never** optimize tau on the same OOF used for evaluation. That is data leakage.
- **Per-class blend optimization is BANNED**. It overfits to OOF noise,
  especially for classes with < 50 samples (Cormorants=40).
- Always report the raw (no post-processing) score alongside any adjusted score.

## 4. Known Data Properties

| Property | Value |
|----------|-------|
| Train months | [1, 4, 9, 10] |
| Test months | [2, 5, 9, 10, 12] |
| Unseen test months | ~33% (Feb, May, Dec) |
| Adversarial AUC (clean features) | 0.7469 (E33, biological, not fixable) |
| Smallest class | Cormorants = 40 samples |
| SKF vs LOMO gap | ~0.35 (E32: 0.6808 vs 0.3321) |

- The distribution shift is biological (different species migrate in different months).
  Adversarial reweighting does not help (E28: -0.006 to -0.025 everywhere).
- Cormorants with 40 samples means any per-class optimization on this class is noise.
- LOMO is a lower bound on real test performance, not an expected score.

## 5. Reporting in EXPERIMENTS.md

Every experiment entry must include:

1. **Feature count** and which temporal features were removed
2. **CV method** (SKF, RSKF, LOMO, or all)
3. **Bootstrap 95% CI** on the primary metric
4. **Raw score** (no post-processing) listed separately from any adjusted score
5. Whether ensemble weights were **fixed** or **optimized**
6. For any post-processing: was it optimized on eval data? If so, was split-half used?

Example entry (E32 actual):
```
| E32 | 2026-02-15 | honest_baseline | 0.6808 | ... | 114 feats, 23 temporal removed.
  Fixed weights (0.33/0.33/0.34). No post-proc. RSKF 5x5: 0.6754 +/- 0.0067.
  Bootstrap 95% CI: [0.6505 - 0.7143]. LOMO: 0.3321. |
```

## 6. What NOT To Do (Lessons Learned)

| Mistake | What happened | Rule |
|---------|--------------|------|
| Temporal features | CV=0.7535, LB=0.52 | Never use calendar features |
| Weight optimization on OOF | Inflated E25D by ~0.01 | Fixed weights or split-half |
| Per-class blend on OOF | E31 inflated by ~0.007 | Banned |
| Logit adj on eval OOF | E34: honest delta = -0.002 | DROP entirely |
| Isotonic calibration | -0.018, overfits small classes | Never with < 50 samples/fold |
| Pseudo-labeling | -0.005, minority gets 0 labels | Not viable with class imbalance |
| Adversarial reweighting | -0.006 to -0.025 | Biological shift, not fixable |
| SMOTE | All variants hurt trees | Don't synthesize for GBDT |
