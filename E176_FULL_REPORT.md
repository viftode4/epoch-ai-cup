# E176 Full Report: Research & Experiments

**Date:** 2026-03-21
**Baseline:** E175 blend SKF=0.7043, LOMO=0.5461, LB=0.59
**Target:** Break 0.59 LB ceiling (top teams at 0.63)

---

## Part 1: Experimental Results

### Phase A — Post-Processing (on E175 OOF)

All techniques evaluated on BOTH SKF and LOMO. LOMO is the honest metric.

| Technique | SKF | LOMO | LOMO Delta | Verdict |
|-----------|-----|------|------------|---------|
| **Baseline (E175 blend)** | **0.7043** | **0.5461** | — | — |
| A6 Isotonic (non-CV, fit all) | 0.7260 | **0.5655** | **+0.0194** | Best LOMO but overfit risk |
| A1(LOMO) + A5(K=15) combo | 0.6727 | **0.5546** | **+0.0085** | Decent |
| A5 + A1(LOMO) | 0.6864 | **0.5493** | +0.0032 | Small |
| A2(0.1) + A5(K=15) | 0.7103 | **0.5487** | +0.0026 | Small |
| A5 KNN (K=15, a=0.15) | 0.7106 | **0.5475** | +0.0014 | Real signal |
| A5 KNN (K=10, a=0.10) | 0.7096 | 0.5468 | +0.0007 | Marginal |
| A2 Day Priors (alpha=0.05) | 0.7045 | 0.5461 | 0.0000 | No effect |
| A3 Flock Prop (conf=0.95) | 0.7043 | 0.5461 | 0.0000 | No effect |
| A1 Power (non-CV) | 0.7088 | 0.5419 | -0.0042 | Hurts LOMO |
| A4 MLLS (a=0.05, t=0.15) | 0.7026 | 0.5353 | -0.0108 | Harmful |
| A6 Isotonic (LOMO CV) | 0.6932 | 0.5261 | -0.0200 | Doesn't generalize across months |
| A6 Isotonic (SKF CV) | 0.6933 | 0.5205 | -0.0256 | Worst |
| NB PP (all gammas) | 0.7043 | 0.5461 | 0.0000 | No effect (UNSEEN gate never fires on train) |

**Key insight:** SKF-CV and LOMO tell OPPOSITE stories. Isotonic (non-CV) is the best LOMO technique despite being "overfit" on SKF. A5 KNN is the only technique that consistently helps both.

### Gaussian/Mixture Calibration (New Idea)

| Technique | SKF | LOMO | LOMO Delta |
|-----------|-----|------|------------|
| **GMM (K=50, alpha=0.5)** | 0.7183 | **0.5633** | **+0.0172** |
| **GMM (K=50, alpha=0.3)** | 0.7148 | **0.5585** | **+0.0124** |
| GMM (K=50, alpha=0.2) | 0.7121 | 0.5539 | +0.0078 |
| GMM (K=30, alpha=0.5) | 0.7090 | 0.5510 | +0.0049 |
| GMM (K=20, alpha=0.5) | 0.7098 | 0.5508 | +0.0047 |
| GMM (K=10, alpha=0.5) | 0.7022 | 0.5508 | +0.0047 |
| GMM LOMO-CV (K=10, a=0.2) | 0.6992 | 0.5462 | +0.0001 |
| Beta calibration (non-CV) | 0.7008 | 0.5372 | -0.0089 |
| GDA (non-CV) | 0.6949 | 0.5380 | -0.0081 |
| GDA month-specific priors | 0.6962 | 0.5364 | -0.0097 |

**GMM archetype correction is the best post-processing technique we found.** It clusters prediction vectors into archetypes, computes per-cluster true class distributions, and blends. Improves every month, especially Month 9 (0.411 → 0.433).

### Phase C — Model Architectures

All models individually worse than E175. Small diversity gains from blending.

| Model | SKF | LOMO |
|-------|-----|------|
| E175 best (baseline) | 0.7043 | 0.5461 |
| LGB GBDT | 0.6681 | 0.4802 |
| Per-class specialists | 0.6519 | 0.4693 |
| BalancedRandomForest | 0.5802 | 0.4248 |
| Smooth-AP MLP | 0.4895 | 0.3385 |
| Focal loss | FAILED | — |

**Best blends with E175 (LOMO):**

| Blend | SKF | LOMO | LOMO Delta |
|-------|-----|------|------------|
| E175 + GBDT@10% | 0.7065 | **0.5473** | **+0.0012** |
| E175 + GBDT@20% | 0.7081 | 0.5469 | +0.0008 |
| E175 + BRF@10% | 0.7051 | 0.5420 | -0.0041 |
| E175 + specialists@10% | 0.7082 | 0.5408 | -0.0053 |

Only GBDT provides real LOMO diversity gain.

### Per-Month LOMO Breakdown (E175 baseline)

| Month | N | mAP | Weakest Classes |
|-------|---|-----|-----------------|
| Jan (1) | 221 | 0.536 | BoP=0.03 (n=2), Clutter=0.14 (n=1), Cormorants=0.36 (n=6) |
| Apr (4) | 473 | 0.569 | Cormorants=0.01 (n=1), Pigeons=0.04 (n=5), Waders=0.50 (n=50) |
| **Sep (9)** | **467** | **0.411** | **Pigeons=0.16 (n=5), Cormorants=0.35 (n=9), Waders=0.44 (n=27)** |
| Oct (10) | 1440 | 0.669 | Waders=0.23 (n=32), Cormorants=0.31 (n=24) |

**Month 9 is catastrophic.** Cormorants and Waders are consistently the weakest across ALL months.

---

## Part 2: Research Findings

### Agent 1: Kaggle Winning Solutions

**Tier 1 — Proven, practical:**

1. **Seed Averaging (20+ seeds)** — Free variance reduction. Just average DART predictions across 20 random seeds. Standard practice in Kaggle. We use 5 seeds; going to 20 is guaranteed improvement at no overfit risk.

2. **Hill Climbing Ensemble Weight Optimization** — Start with best model, greedily add others keeping only additions that improve OOF macro-mAP. Use scipy.optimize. Better than heuristic 50/50 blends.

3. **Rank Averaging instead of Probability Averaging** — mAP is a ranking metric. Rank averaging normalizes different model calibration scales. We partially do this (rank_power_ensemble) but haven't systematically tested pure rank averaging.

4. **Per-Class Power Transform (p^alpha per class)** — Optimize alpha per class on OOF macro-mAP. Sharpening (alpha>1) amplifies ranking gaps.

5. **Diverse Hyperparameter Ensemble** — Train 5-8 models with INTENTIONALLY different configs (depth=3 vs 12, subsampling=0.5 vs 0.9). Structural diversity > seed diversity.

6. **Test-Time Augmentation (TTA)** — Add Gaussian noise to test features, predict 10-20 copies, average. Free variance reduction on test predictions. Low risk.

**Tier 2 — Worth trying:**

7. **Focal Loss for LGB** — Down-weight easy Gull negatives, focus on hard minority class gradients. API fixed but still failing in our implementation.

8. **CReST Pseudo-Labeling** — Class-aware thresholds: lower confidence threshold for minority classes. Fixes the failure mode of standard pseudo-labeling (minority gets 0 labels).

**Confirmed dead ends:** SMOTE, standard pseudo-labeling, adversarial reweighting, TabPFN, stacking with external features.

### Agent 2: Domain Adaptation & Covariate Shift

**Tier 1 — High probability of gain:**

1. **MAPLS (MAP Label Shift)** — MLLS with Dirichlet prior regularization using GBIF. Fixes MLLS collapse (E91/E92). Key formula:
   ```
   w_new ∝ (ml_counts + alpha_gbif - 1) / (N + sum(alpha) - K)
   ```
   Use GBIF monthly counts as informative Dirichlet priors.

2. **Label Proportion Matching** — Constrain test predictions so per-month marginal class proportions match MAPLS estimates. Minimize KL divergence. More principled than ratio-tilt.

3. **Conformal Prediction Sets as Uncertainty Gate** — Replace heuristic `top2_margin < tau` with conformal set size > 1. Statistically grounded, adapts to actual calibration quality. Uses weighted conformal prediction for shift-robustness.

**Tier 2 — Moderate probability:**

4. **Group DRO via Month-Based Sample Reweighting** — Iteratively upweight months with worst per-class performance. Combined with class weights and DART boosting.
   ```
   for each DRO round:
     train model with current weights
     compute per-month loss
     q[m] *= exp(eta * loss[m])  # upweight worst months
   ```

5. **CVaR-DRO** — Optimize for worst 20-30% of training samples. Simpler than full Group DRO. Naturally upweights hard minority samples.

6. **Domain Classifier Importance Weights (with clipping)** — Train propensity model, clip weights to [0.2, 5.0] to prevent ESS collapse. Different from E28/E30 because of aggressive clipping.

7. **AdapTable (ICLR 2025)** — First TTA method for tabular data. Shift-aware uncertainty calibrator + label distribution handler. Post-processing, no retraining. [GitHub](https://github.com/drumpt/AdapTable)

**Tier 3 — Experimental:**

8. **FtaT (AAAI 2025)** — Fully Test-time Adaptation for Tabular. Confident Distribution Optimizer + Local Consistent Weighter + Dynamic Model Ensembler. [GitHub](https://github.com/WNJXYK/FTTA)

9. **Fourier Month Embedding** — Encode month as sin/cos harmonics (cyclical). Only if combined with DRO to prevent month leakage.

10. **KMM (Kernel Mean Matching)** — Kernel-based importance weights. Feasible at n=2600 but use only 36 features.

**Key references:**
- TabReD (ICLR 2025): GBDT still best under temporal shift
- MAPLS (WACV 2024): Bayesian label shift for imbalanced data
- AdapTable (ICLR 2025): Test-time adaptation for tabular

### Agent 3: AP Optimization & Ranking

**Tier 1 — Highest practical impact:**

1. **XGBoost `rank:map` OvR** — XGBoost has a dedicated `rank:map` objective that directly optimizes MAP gradients. LightGBM's `lambdarank` optimizes NDCG gradients even when eval metric is MAP. This is potentially a significant difference. Drop-in replacement for our OvR rankers.
   ```python
   params = {'objective': 'rank:map', 'eval_metric': 'map', ...}
   dtrain.set_group(group_sizes)
   ```

2. **Per-Class Ensemble Weight Optimization for Macro-mAP** — Different weights per class. Minority classes (Cormorants) may want different model mix than majority (Gulls).
   ```python
   # weights_matrix: (n_models, 9) — different weights per class
   for cls in range(9):
       blended[:, cls] = sum(w[i, cls] * model_preds[i][:, cls] for i in range(n_models))
   ```

3. **Two-Stage Ranker** — Train Stage 1 classifiers normally. Stage 2: train per-class rankers (XGBoost rank:map) on Stage 1 probabilities + raw features. Corrects systematic ranking errors.

4. **LGB `rank_xendcg`** — Cross-entropy NDCG approximation. Different gradient dynamics than lambdarank. Quick to test as diversity model.

5. **`label_gain=[0,1]` fix** — For binary relevance (our OvR setup), explicitly set label_gain to [0,1]. Default is graded relevance [0,1,3,7,...] which is wrong for AP.

**Tier 2:**

6. **Rank-based blending** — Convert each model's predictions to per-class ranks, then blend ranks instead of probabilities. More robust to calibration differences.

7. **Per-class AP-optimal power transform** — Optimize alpha per class using scipy.optimize to maximize per-class AP independently (not overall mAP).

8. **Isotonic regression BEFORE ensembling** — Better-calibrated per-class scores lead to better blending. Isotonic preserves ranking within each class.

### Agent 4: Semi-Supervised & Pseudo-Labeling

**Tier 1 — Most likely to help:**

1. **CReST (Class-Rebalancing Self-Training)** — CVPR 2021, Google Research. Per-class adaptive thresholds:
   ```
   threshold_c = base_threshold * (class_count / max_count)^0.5
   ```
   Cormorants (n=40) get threshold ~0.50, Gulls (n=1503) get ~0.95. Multi-round with soft pseudo-labels (probability vectors, not hard 0/1). [GitHub](https://github.com/google-research/crest)

2. **Label Propagation (sklearn LabelSpreading)** — Build k-NN graph over all 4473 samples (train+test), propagate labels through graph. Zero retraining. Bridges month gap through feature-space similarity.
   ```python
   from sklearn.semi_supervised import LabelSpreading
   model = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
   all_labels = np.concatenate([y_train, np.full(len(test), -1)])
   model.fit(np.vstack([X_train, X_test]), all_labels)
   test_probs = model.label_distributions_[len(train):]
   ```

3. **FtaT (Fully Test-time Adaptation for Tabular, AAAI 2025)** — Three modules: Confident Distribution Optimizer, Local Consistent Weighter, Dynamic Model Ensembler. Designed for EXACTLY our problem (tabular + covariate shift + label shift). [GitHub](https://github.com/WNJXYK/FTTA)

**Tier 2:**

4. **Co-Training with Trajectory vs RCS views** — Natural feature split: View 1 = trajectory features (kinematics), View 2 = RCS features (electromagnetic). Each model labels its confident test samples for the other. Cross-view validation reduces pseudo-label noise.

5. **Feature-Space Perturbation** — Add Gaussian noise to minority class samples (3-5 copies per Cormorant/Duck/Geese). Simpler than SMOTE, less overfitting risk.

6. **FlexMatch per-class adaptive thresholds** — Dynamic confidence threshold per class that adjusts based on learning progress.

**Critical guardrails:**
- Never use hard pseudo-labels for minority classes (use soft probability vectors)
- Monitor per-class AP after each round — revert if any drops >0.02
- Cap pseudo-label proportions with MLLS/MAPLS estimates
- Only pseudo-label using month-invariant features

---

## Part 3: Prioritized Implementation Plan

Based on ALL research + experiments, ranked by **expected LOMO impact × feasibility:**

### Immediate (can run now, no retraining):

| # | Technique | Est. LOMO Gain | Time | Notes |
|---|-----------|---------------|------|-------|
| 1 | GMM archetype correction (K=50, a=0.5) | +0.017 | 1 min | Already tested, apply to test |
| 2 | Isotonic (non-CV) on test | +0.019 | 1 min | Already tested, apply to test |
| 3 | A5 KNN on test | +0.001 | 2 min | Proven real signal |
| 4 | Stack: GMM + isotonic + KNN | +0.02? | 5 min | Combine winners |
| 5 | Test-time augmentation | +0.001-0.005? | 5 min | Free variance reduction |
| 6 | Per-class ensemble weight optimization | +0.002-0.010? | 10 min | scipy.optimize on OOF |
| 7 | Label proportion matching (MAPLS) | +0.005? | 15 min | Fix MLLS with Dirichlet prior |

### Short-term (need training, <1 hour):

| # | Technique | Est. LOMO Gain | Time |
|---|-----------|---------------|------|
| 8 | Seed averaging (20 seeds) | +0.002-0.005 | 30 min |
| 9 | XGBoost rank:map OvR | +0.005-0.015? | 45 min |
| 10 | Label propagation (transductive) | +0.005? | 15 min |
| 11 | Rank-based blending | +0.001-0.003 | 10 min |

### Medium-term (need training, 1-3 hours):

| # | Technique | Est. LOMO Gain | Time |
|---|-----------|---------------|------|
| 12 | CReST pseudo-labeling | +0.005-0.020? | 2 hours |
| 13 | Group DRO (month-weighted) | +0.005-0.010? | 1 hour |
| 14 | Two-stage ranker | +0.003-0.010? | 1.5 hours |
| 15 | Diverse hyperparameter ensemble | +0.002-0.005 | 2 hours |
| 16 | AdapTable / FtaT test-time adaptation | +0.005-0.015? | 2 hours |

### Do NOT retry:
- SMOTE (E21, E40, E163b)
- Standard pseudo-labeling without class rebalancing (E17, E62, E163)
- Adversarial reweighting without clipping (E28, E30)
- BBSE/MLLS without regularization (E91, E92, E115)
- TabPFN (E83)
- More features beyond 100 without pruning (E44, E166)

---

## Part 4: Pending Results

Still running:
- **Phase B features** (flock intensity, glide ratio, wing loading, session time) — training with DART
- **Focal loss** (LGB with custom focal objective) — API issues being resolved

These will add to the picture but the research above is independent of their results.
