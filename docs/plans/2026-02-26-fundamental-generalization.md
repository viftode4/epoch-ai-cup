# Fundamental Generalization Research Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break the LB 0.59 ceiling by improving predictions on unseen months (Feb, May, Dec = 33% of test) through training-time modeling changes rather than post-processing.

**Architecture:** Three independent experiments (E93-E95) attack the generalization problem from different angles: (1) replace raw weather features with physics-derived interactions that are month-invariant, (2) Optuna-tune ALL model hyperparameters on LOMO objective, (3) train separate models per radar_bird_size group. Each experiment uses LOMO as primary validation (proxy for unseen months) with SKF as secondary. Best variants combine into E96.

**Tech Stack:** LightGBM (GPU), XGBoost (CUDA), CatBoost (GPU), Optuna, NumPy, scikit-learn

---

## Background & Motivation

**The problem:** SKF mAP = 0.77, LOMO = 0.36. The 0.41 gap is temporal distribution shift. Shared months (Sep/Oct = 67% of test) work fine; unseen months (Feb/May/Dec = 33%) estimated mAP ~0.18.

**What failed:**
- Post-processing (NB physics, GBIF priors, label shift): LB capped at 0.59, stronger PP hurts
- Feature addition (physics, TS, wavelets): ALL hurt LOMO via feature dilution (E88)
- Domain adaptation (monthly normalization, adversarial reweighting, pseudo-labeling): marginal to harmful (E89)
- Label shift correction (BBSE/MLLS): LOMO marginal, LB 0.51 (E91)

**What hasn't been tried:**
1. REPLACING (not adding) weather features with physics interactions
2. LOMO-tuning ALL model HPs (currently only CatBoost is LOMO-tuned)
3. Size-stratified modeling (radar_bird_size is month-invariant and strongly class-correlated)

**Success criteria:** LOMO > 0.38 (current best: E79 = 0.3798) AND/OR LB > 0.59

---

## Task 1: E93 — Physics-Derived Weather Features

**Hypothesis:** Raw weather/solar features (11 of 36) encode month identity as a side effect. Replacing them with physics *interactions* (bird_behavior x conditions) should preserve useful signal while reducing month leakage. Key: same feature count (36), just different features.

**Files:**
- Create: `experiments/e93_physics_features.py`
- Read: `src/features.py`, `src/data.py`, `data/best_features.txt`
- Read: `data/train_weather.csv`, `data/test_weather.csv`, `data/train_solar.csv`, `data/test_solar.csv`

### Step 1: Create experiment script

Create `experiments/e93_physics_features.py` with the following structure:

```python
"""E93: Physics-derived weather feature replacement.

Replace 11 raw weather/solar features with 11 physics interaction features.
Hypothesis: interactions encode bird-behavior-in-context (month-invariant)
rather than raw conditions (month-proxy).

Features replaced:
  OUT: wx_wind_speed, wx_wind_gust, wx_wind_u, wx_wind_v, wx_temp_c,
       wx_dewpoint_c, wx_humidity, sol_solar_elevation, sol_daylight_hours,
       sol_hours_since_sunrise, sol_daylight_fraction
  IN:  11 physics interactions (see PHYSICS_FEATURES below)

Variants:
  A. 25 intrinsic+spatial only (no weather at all — E90 reference)
  B. 25 + 11 physics interactions (REPLACEMENT — key test)
  C. 36 original E79 features (BASELINE)
  D. 25 + 11 physics + 11 raw weather (47 features — test if physics ADD value)
"""
```

**Physics features to compute** (11 total):

1. `ph_airspeed_over_wind` = airspeed / max(wx_wind_speed, 0.5)
   - Ratio of bird speed to wind. Soaring birds ~1.0, flapping birds >>1.0. Month-invariant per species.

2. `ph_ground_wind_ratio` = avg_ground_speed / max(wx_wind_speed, 0.5)
   - Ground speed relative to wind. Captures headwind/tailwind effect per species.

3. `ph_gust_fraction` = wx_wind_gust / max(wx_wind_speed, 0.5)
   - Turbulence indicator. Affects flight mode (flapping vs gliding). Ratio is month-invariant.

4. `ph_thermal_potential` = max(wx_temp_c - wx_dewpoint_c, 0) * max(sol_solar_elevation, 0)
   - Thermal soaring conditions. High = sunny + dry = thermals. BoP/Gulls exploit thermals.

5. `ph_density_altitude` = wx_temp_c * (1 + 0.00378 * wx_humidity)
   - Air density proxy affecting lift. Higher = thinner air = different flight dynamics.

6. `ph_wind_altitude_ratio` = wx_wind_speed / max(alt_median, 1.0)
   - Wind strength relative to flight altitude. Low-altitude + high wind = different species mix.

7. `ph_speed_thermal` = speed_median * max(sol_solar_elevation, 0.01) / 45.0
   - Speed normalized by solar position. Captures activity-time interaction.

8. `ph_rcs_wind_adjusted` = rcs_mean + 10 * np.log10(max(wx_wind_speed, 0.5))
   - RCS adjusted for wind-induced scattering. Wind affects radar returns.

9. `ph_altitude_wind_effort` = alt_median * wx_wind_speed / max(airspeed, 1.0)
   - Combined altitude-wind-speed interaction capturing flight effort.

10. `ph_activity_phase` = sol_hours_since_sunrise / max(sol_daylight_hours, 1.0)
    - Fraction of daylight elapsed (0=dawn, 1=dusk). More invariant than raw hours.
    Note: similar to sol_daylight_fraction but computed differently.

11. `ph_precip_flight` = slow_flight_frac * wx_humidity / 100.0
    - Slow flight in humid conditions. Species-specific response to bad weather.

**Evaluation:** For each variant (A-D):
1. LOMO (4 months) with LGB+CB ensemble
2. SKF (5-fold) with LGB+XGB+CB ensemble (only if LOMO improves)
3. Per-month LOMO breakdown
4. Save submissions for any variant with LOMO > 0.3798

### Step 2: Run experiment

Run: `python experiments/e93_physics_features.py`
Expected: ~15 min with GPU. 4 variants x (LOMO + possibly SKF).

### Step 3: Analyze results

Key questions:
- Does variant B (physics replacement) improve LOMO over C (raw weather)?
- Does the SKF-LOMO gap shrink with physics features?
- Which individual physics features have highest importance?
- Per-month: do unseen months (held-out in LOMO) improve?

### Step 4: Log results in EXPERIMENTS.md

Add E93 entry with LOMO and SKF scores, per-class breakdown for best variant.

---

## Task 2: E94 — Full LOMO Hyperparameter Optimization

**Hypothesis:** LGB (50% ensemble weight) and XGB (40% weight) use generic hyperparameters that were never tuned for LOMO. Since LOMO penalizes temporal overfitting, LOMO-tuned HPs should be more regularized (fewer leaves, lower colsample, stronger L1/L2). This is the lowest-hanging fruit — no feature changes, just better HPs.

**Files:**
- Create: `experiments/e94_lomo_hpopt.py`
- Read: `experiments/e79_pruned_tuned_base.py` (current HP reference)
- Read: `src/features.py`, `src/data.py`

### Step 1: Create experiment script

Create `experiments/e94_lomo_hpopt.py`:

```python
"""E94: Full LOMO hyperparameter optimization.

Optuna-tune ALL model HPs (LGB, XGB, CB) on LOMO objective.
Currently: only CB is LOMO-tuned (E79). LGB uses generic HPs.
LGB has 50% ensemble weight — optimizing it matters most.

Pipeline:
  1. Optuna-tune LGB (30 trials, LOMO objective)
  2. Optuna-tune XGB (30 trials, LOMO objective)
  3. Optuna-tune CB (20 trials, LOMO objective) — verify E79 result
  4. Grid-search ensemble weights on LOMO OOF
  5. Compare LOMO and SKF vs E79 baseline
"""
```

**HP search spaces:**

LGB:
```python
{
    "n_estimators": 1500,  # fixed
    "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
    "max_depth": trial.suggest_int("max_depth", 3, 8),
    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
    "colsample_bytree": trial.suggest_float("colsample", 0.3, 0.8),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    "min_child_samples": trial.suggest_int("min_child", 5, 50),
}
```

XGB:
```python
{
    "n_estimators": 1500,
    "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 8),
    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
    "colsample_bytree": trial.suggest_float("colsample", 0.3, 0.8),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
}
```

**Evaluation:**
1. Report LOMO for each model individually + ensemble
2. Compare to E79 baseline (LGB LOMO, XGB LOMO, CB LOMO, ensemble LOMO)
3. If LOMO improves: run SKF and save submissions
4. Key diagnostic: do LOMO-tuned HPs have fewer leaves / more regularization?

### Step 2: Run experiment

Run: `python experiments/e94_lomo_hpopt.py`
Expected: ~30-45 min (80 Optuna trials x 4 LOMO folds each). GPU required.

### Step 3: Analyze HP differences

Compare LOMO-optimal HPs to current defaults:
- Current LGB: num_leaves=63, max_depth=7, reg_alpha=0.01, reg_lambda=0.1
- If LOMO-optimal has num_leaves=20, max_depth=4, etc. -> confirms overfitting hypothesis

### Step 4: Log results in EXPERIMENTS.md

---

## Task 3: E95 — Size-Stratified Ensemble

**Hypothesis:** `radar_bird_size` (Small/Medium/Large/Flock) is month-invariant and strongly correlated with species. Training separate models per size group simplifies each sub-task, reducing the need for month-specific patterns.

Key data insight:
- Medium bird ONLY appears in months 1,4 (train) and 2,5,12 (test) — never in Sep/Oct!
- Small bird: 1657 train (63%), dominated by Gulls (1104) + Songbirds (294) + BoP (101)
- Large bird: 449 train, more balanced across 7 classes
- Flock: 345 train, Gulls (151) + Songbirds (128) + Geese (26)
- Medium bird: 150 train, very mixed — smallest group

**Files:**
- Create: `experiments/e95_size_stratified.py`
- Read: `src/features.py`, `src/data.py`

### Step 1: Create experiment script

Create `experiments/e95_size_stratified.py`:

```python
"""E95: Size-stratified ensemble.

Train separate LGB+CB models per radar_bird_size group.
Each sub-model faces a simpler classification task.

Groups:
  A. Small bird (n=1657): 5 main classes (Gulls, Songbirds, BoP, Waders, Pigeons)
  B. Large bird (n=449): 7 classes (Gulls, Clutter, Pigeons, Songbirds, Geese, Ducks, Cormorants)
  C. Flock (n=345): 5 main classes (Gulls, Songbirds, Geese, Pigeons, Waders)
  D. Medium bird (n=150): mixed — may need to merge with Large

Variants:
  1. 4-way split (Small / Medium / Large / Flock) — separate models
  2. 3-way split (Small / Medium+Large / Flock) — merge small groups
  3. Global model with size as ordinal feature (baseline comparison)
  4. Hybrid: size-stratified LOMO + global SKF

All use 36 E79 features. Evaluate with LOMO and SKF.
"""
```

**Key design decisions:**
- Each sub-model still predicts ALL 9 classes (some will be zero/rare in the subset, but we need predictions for all)
- Use the same E79 feature set (36 features) for each sub-model
- For LOMO: stratify by month within each size group (some months may have very few samples in some groups — handle gracefully)
- For SKF: stratify by class within each size group
- Final predictions: for each test sample, use the prediction from the model matching its radar_bird_size

### Step 2: Run experiment

Run: `python experiments/e95_size_stratified.py`
Expected: ~10 min (4 smaller models instead of 1 big one).

### Step 3: Analyze stratification impact

Key questions:
- Does each sub-model have better LOMO than the global model?
- Which size groups improve most? (Hypothesis: Medium bird, since it only appears in unseen-type months)
- Per-class: which species benefit from stratification?
- Is the merged 3-way better than 4-way? (Medium bird has only 150 samples)

### Step 4: Log results in EXPERIMENTS.md

---

## Task 4: E96 — Combined Best Approaches

**Prerequisite:** Tasks 1-3 completed with results analyzed.

**Files:**
- Create: `experiments/e96_combined.py`

### Step 1: Identify best components from E93-E95

From each experiment, take the component that improved LOMO:
- E93: If physics features helped → use physics feature set
- E94: If LOMO-tuned HPs helped → use tuned HPs for all models
- E95: If size-stratified helped → use stratified ensemble

### Step 2: Create combined experiment

Combine winning components. E.g., if all three helped:
- Size-stratified models
- With physics features replacing raw weather
- With LOMO-optimized hyperparameters
- Evaluate with both LOMO and SKF
- Apply conservative PP (gamma=0.10, tau=0.30 from E75) to best variant

### Step 3: Generate submissions

Save multiple variants for LB validation:
1. Combined raw (no PP)
2. Combined + E75 PP (gamma=0.10)
3. Best individual improvement + E75 PP

### Step 4: Update EXPERIMENTS.md

---

## Task 5: Update EXPERIMENTS.md and MEMORY.md

After all experiments complete:
1. Add E93-E96 entries to EXPERIMENTS.md with full results
2. Update MEMORY.md with key findings about which approaches helped/hurt
3. Record LB results when available

---

## Decision Points

- **After Task 1:** If physics features show LOMO > 0.39, proceed with them in Task 4. If LOMO <= 0.38, physics features don't help — use original 36 features.
- **After Task 2:** If LOMO-tuned HPs improve LOMO by > 0.005, use them. Otherwise, current HPs are fine.
- **After Task 3:** If any stratification variant shows LOMO > 0.39, include in Task 4. Key diagnostic: does the Medium bird sub-model specifically improve?
- **After Task 4:** Submit top 2-3 variants to LB. If LB > 0.59, we've broken the ceiling.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Physics features also leak month | Compute adversarial AUC on physics feats vs raw — should be lower |
| LOMO tuning overfits to 4 LOMO folds | Use cross-validated Optuna (inner CV) |
| Size stratification: too few samples per group | Merge Medium+Large; minimum 200 samples per group |
| Combined approach doesn't stack | Test components independently first |
| LOMO still doesn't predict LB | Submit conservatively; E79+PP remains fallback |
