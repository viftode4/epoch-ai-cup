# E174+ Modeling Plan: Breaking the 0.59 Ceiling

## Status
- **Current best LB:** 0.59 (6th place, 170+ experiments)
- **Top LB:** 0.63 (3 teams)
- **Gap to close:** 0.04 macro mAP
- **Deadline:** March 24, 2026

---

## Part 1: WHY We're Stuck at 0.59

### 1.1 The Wrong Loss Function

We train all models with **multi-class logloss** (cross-entropy). This optimizes:
- P(class | features) — probability calibration

The competition measures **macro-averaged mAP**. This measures:
- For each class independently: can you rank the TRUE examples above the FALSE ones?

Logloss and mAP are correlated but not the same. A model can have perfect logloss but mediocre mAP if it assigns moderate probabilities to everything. mAP rewards **sharp separation** — pushing true positives to the top of each class's ranking.

**Impact:** Every model we've trained wastes gradient on calibrating Gull probabilities (58% of data, already AP=0.955) instead of focusing on ranking Cormorants (40 samples, AP=0.296 on honest CV) or Waders (120 samples, AP=0.505).

### 1.2 Within-Class Month Heterogeneity

Each class contains 2-3 species sub-populations with completely different radar signatures by month:

| Class | Train months | Dominant species | Test unseen months | Different species |
|-------|-------------|-----------------|-------------------|-------------------|
| Songbirds | Oct: thrushes | Nocturnal, bounding, high alt, small RCS | Feb: corvids/finches, May: warblers | Diurnal, continuous flap, bigger RCS |
| Pigeons | Oct: wood pigeon migration | 17-22 m/s, 100-500m, heading 220° SW | Feb: feral pigeons | 8-12 m/s, low alt, erratic |
| Geese | Oct: barnacle + greylag | Barnacle 163m median, greylag 56m | Feb: different proportions, Dec: departure | Species ratio changes altitude distribution |
| Waders | Sep-Oct: pre-migration | RCS doubles (fattening 7→15 cm²) | Feb: winter residents, May: spring return | Different body condition, different behavior |
| Ducks | Oct: autumn staging | Sea ducks, very low altitude | Feb: winter residents, Dec: ice-driven movement | Different species composition |

Our model learns ONE decision boundary per class. For Pigeons, it compromises between "fast high migrant" and "slow low local" — and does poorly on both, especially the unseen-month variant.

### 1.3 Feature Dilution vs Month Proxies

E79's 36 backward-eliminated features include 11 weather/solar features that act as month proxies. These features are useful (removing them drops SKF from 0.7736 to 0.6821) because they encode real seasonal biology. BUT they don't transfer to unseen months where the weather patterns are different.

Adding MORE features (E166: 76 features, E174: 121 features) gives the model even more month-proxy opportunities, making it LESS likely to learn month-invariant patterns.

---

## Part 2A: DATA ARCHITECTURE — Extracting Everything from What We Have

### 2A.1 Species-Level Training (Biggest Untapped Opportunity)

We have **68 species labels** in training. We use NONE of them — we train on the 9 coarse groups.

**The heterogeneity problem in numbers:**
- Songbirds: 20 species. Oct = Starling (104) + Crow (48) + Jackdaw (37). Jan = Jackdaw (8) + Crow (4). These have DIFFERENT radar signatures.
- Pigeons: 81% Stock Dove, 7% Wood Pigeon. Stock Dove and Wood Pigeon behave differently.
- Waders: April = Curlew (18) + Oystercatcher (16). Sep = Golden Plover (18). Oct = Lapwing (17). Complete species turnover by month.
- Geese: Greylag (34, median 56m altitude) vs Barnacle (9, median 163m) vs Bean Goose (25) — different altitude ranges.

**Approach: Train species-cluster classifier → aggregate to group**

1. Merge rare species (< 5 samples) into "Other_{Group}" clusters → ~25-30 clusters
2. Train a 25-30 class CatBoost classifier at species-cluster level
3. Sum species probabilities within each group: `P(Songbird) = P(Starling) + P(Crow) + P(Jackdaw) + ...`
4. The species-level model learns TIGHTER decision boundaries per species
5. Aggregation naturally handles month heterogeneity — October predictions weight Starling/Fieldfare, January predictions weight Crow/Jackdaw

**Why this solves the heterogeneity problem:** The model no longer needs ONE Songbird boundary. It learns separate boundaries for Starling (fast, flocking, Oct-Sep), Crow (slow, solitary, Jan-Oct), Jackdaw (medium, small flocks, all months). When these boundaries are summed to get P(Songbird), the model is implicitly ADAPTING to whatever species mix exists in the test month.

**Risk:** Some species have very few samples (Blue Tit: 1, Brambling: 1). These get merged into "Other_Songbirds". The top 10-15 species cover 80%+ of samples.

### 2A.2 Flock Size Predictor (Recover Privileged Feature)

`n_birds_observed` is the most discriminative feature we found — but it's train-only:
- BoP: always 1 (solitary)
- Clutter: always 1
- Waders: mean 37.0 (large flocks!)
- Songbirds: mean 19.4
- Pigeons: mean 13.6

**Approach:** Train a regression model to predict `n_birds_observed` from available features, then apply to both train and test.

```python
# Features correlated with flock size:
# radar_bird_size (Flock/Large/Medium/Small)
# rcs_scintillation (multi-scatterer interference)
# rcs_linear_mean (proportional to body count × individual RCS)
# rcs_kurtosis_linear (flock interference patterns)
# rcs_deep_fade_frac (multi-path fading in flocks)

flock_features = ['radar_bird_size', 'rcs_scintillation', 'rcs_linear_mean',
                  'rcs_kurtosis_linear', 'rcs_deep_fade_frac', 'rcs_std_dB']

flock_model = LGBMRegressor(n_estimators=200, max_depth=4)
flock_model.fit(X_train[flock_features], np.log1p(n_birds_observed))  # log-transform count
predicted_flock = np.expm1(flock_model.predict(X_all[flock_features]))
```

Log-transform because flock sizes are heavily right-skewed (1 to 400).

### 2A.3 New Trajectory-Derived Features

Features we're NOT extracting from the raw trajectory:

**Altitude profile shape (3 features):**
```python
# Fit quadratic: alt = a*t² + b*t + c
coeffs = np.polyfit(times, alts, 2)
alt_curvature = coeffs[0]    # a<0: inverted-U (departure), a>0: U-shape (circling descent)
alt_trend = coeffs[1]         # b: overall climb/descent rate
alt_r2 = 1 - residual_var/total_var  # Smooth=purposeful flight, noisy=clutter
```

**RCS multi-lag autocorrelation (4 features):**
```python
# At 1 Hz, wingbeats alias to different lag patterns per species
# Gulls 3-5 Hz → lags 2-4 capture aliased periodicity
for lag in [2, 3, 4, 5]:
    rcs_ac_lag[lag] = mean(rcs_centered[:-lag] * rcs_centered[lag:]) / var(rcs)
```

**Speed profile (3 features):**
```python
speed_cv = speed_std / max(speed_mean, 0.01)     # Cormorant: <0.15, Clutter: high
speed_ac1 = autocorrelation(speeds, lag=1)        # Bounding: negative, steady: positive
speed_trend = polyfit(times[:-1], speeds, 1)[0]   # Accelerating vs decelerating
```

**Total new trajectory features: 10** (3 altitude shape + 4 RCS lags + 3 speed profile)

### 2A.4 Pairwise Confusion Specialists

Cleanlab identified the top confusion pairs:
- Gulls → Waders: 49 mislabels
- Gulls → Songbirds: 46
- Songbirds → Gulls: 25
- Gulls → BoP: 21

For each major pair, train a dedicated binary classifier:
```python
# Gull vs Wader specialist
mask = (y == GULL) | (y == WADER)
specialist = CatBoostClassifier(...)
specialist.fit(X[mask], (y[mask] == WADER))
# Output: P(Wader | Gull_or_Wader) for every sample
```

Key: each specialist uses features MOST RELEVANT to its confusion:
- Gull↔Wader: tidal phase, speed (Waders faster), altitude variance
- Gull↔Songbird: RCS level (Songbirds lower), speed pattern
- Songbird↔BoP: speed (BoP slower), curvature (BoP higher)
- Pigeon↔Duck: crepuscular index, rain tolerance, crop distance

The specialist outputs become META-FEATURES for the final ensemble.

### 2A.5 Per-Class Temperature Scaling

Optimize temperature T_c independently per class to maximize that class's AP:
```python
for c in range(9):
    best_T, best_AP = 1.0, 0
    for T in np.linspace(0.1, 5.0, 50):
        scaled = sigmoid(logit(probs[:, c]) / T)
        ap = average_precision_score(y == c, scaled)
        if ap > best_AP: best_T, best_AP = T, ap
    T_per_class[c] = best_T
```

High T = compress probabilities toward 0.5 (reduce overconfidence).
Low T = sharpen probabilities (increase separation).
Minority classes (Cormorants) likely need LOW T (sharpen the few confident predictions).
Majority classes (Gulls) can handle HIGH T (already well-separated).

---

## Part 2B: THE ARCHITECTURE

### 2B.1 Layer 1 — Per-Class Physics Score Features

Instead of hoping the tree discovers domain patterns from raw features, we PRE-COMPUTE physics-informed scores. Each score encodes a specific domain hypothesis as a continuous value.

**Cormorant Wind Model Score:**
- Literature: ground_speed = 0.70 × wind_speed + 14.4 m/s (Alerstam, SD=0.31)
- Feature: `cormorant_score = exp(-(speed - (0.70×wind + 14.4))² / (2×3.0²))`
- Rationale: Cormorants have the tightest speed-wind relationship of any class. The residual from this model is nearly diagnostic. Existing `cormorant_wind_residual` is the linear version; this is the Gaussian likelihood version.
- Expected class separation: Cormorants should have highest score (~0.9), others < 0.5

**Clutter Drift Score:**
- Physics: Insects/clutter drift at wind speed. True birds fly under own power.
- Feature: `clutter_drift_score = exp(-((speed / max(wind_at_bird_alt, 0.5)) - 1.0)² / (2×0.3²))`
- Rationale: When airspeed/wind ≈ 1, the track is drifting. Score peaks at ratio=1.
- Expected: Clutter highest (ratio near 1), birds lower (ratio > 1 or < 1)
- Note: E174 validation showed Clutter ratio = 0.78 (blade flash pulls it below 1), BoP = 0.70. Score still discriminates.

**Birds of Prey Soaring Score:**
- Physics: BoP soar in thermals. Requires: slow speed, positive CAPE, clear sky, shear.
- Feature: `bop_score = norm.pdf(speed, 11.8, 3.0) × sigmoid((cape - 0) / 5.0) × (1 - cloud_cover_low/100)`
- Rationale: Combines speed match + thermal conditions + visibility. BoP need ALL THREE.
- Expected: BoP highest when conditions are right, near 0 for non-soaring conditions.

**Wader Tidal Score:**
- Physics: Waders feed on mudflats at low tide, fly to roost ~3h before high tide.
- Feature: `wader_tidal_score = norm.pdf(hours_since_high_tide, 9.4, 2.0)`
- Rationale: Tidal cycle is 12.4h. Peak feeding flight = 12.4 - 3 = 9.4h since last high tide. This is gravitationally driven and PERFECTLY month-invariant.
- Expected: Wader flights cluster around this tidal phase.

**Duck Crepuscular Score:**
- Physics: Ducks are crepuscular. They fly at dawn/dusk. They tolerate rain.
- Feature: `duck_score = crepuscular_index × (1 + 0.5 × rain_occurring)`
- Rationale: Crepuscular_index already in v2 pipeline. Combining with rain tolerance gives a Duck-specific signature that Pigeons don't match (Pigeons avoid rain).
- Expected: Ducks get boost from rain+dusk, Pigeons penalized.

**Pigeon Weather Score:**
- Physics: Pigeons avoid rain. Wood pigeons are crop foragers (near arable land).
- Feature: `pigeon_score = (1 - rain_occurring) × (1 / max(dist_to_arable_m, 100)) × 100`
- Rationale: No-rain + near-crops = Pigeon habitat. Simple multiplicative.
- Expected: Pigeons highest in clear weather near farmland.

**Geese Flock Score:**
- Physics: Geese fly in V-formation. High RCS scintillation (multi-scatterer interference). Large/Flock radar size.
- Feature: `geese_score = I(radar_bird_size ∈ {Flock, Large}) × rcs_scintillation`
- Rationale: Only flocking species have high scintillation. Geese + Flocks have the highest.
- Expected: Geese flocks get highest score.

**Songbird Nocturnal Migration Score:**
- Physics: Songbird (thrush) migration is nocturnal with consistent altitude.
- Feature: `songbird_migration = (1 - is_day) × (alt_std < threshold) × I(speed ∈ [10, 16])`
- Note: All our data is daytime (is_day=1 always), so this score would be 0 for everything. SKIP — our data doesn't capture nocturnal migration.
- Alternative: `songbird_score = norm.pdf(speed, 13.1, 2.2) × I(rcs_mean_dB < -22)`
- Rationale: Songbirds are small (low RCS) and fly at moderate speeds.

**Summary:** 8 physics scores (skip songbird nocturnal). Each is a continuous [0, 1] value encoding "how well does this track match class X's physics?" The tree doesn't need to discover these patterns — it just needs to learn the right WEIGHTS and THRESHOLDS on pre-computed physics.

---

### 2B.2 Layer 2 — Multiple Complementary Models

We train SEVEN different model types. Each approaches the problem differently.

#### Model A: CatBoost Multi-Class with Focal Loss

**What:** Standard multi-class CatBoost but with focal loss instead of logloss.

**Focal loss formula:** FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

Where:
- p_t = predicted probability of the TRUE class
- α_t = class weight (inverse frequency)
- γ = focusing parameter (higher = ignore easy examples more)

**Per-class gamma:**
| Class | N | % | Current AP | γ | Rationale |
|-------|---|---|-----------|---|-----------|
| Gulls | 1503 | 57.8% | 0.955 | 0.5 | Already easy, minimal gradient needed |
| Songbirds | 483 | 18.6% | 0.789 | 1.0 | Moderate difficulty |
| Pigeons | 122 | 4.7% | 0.864 | 1.5 | Fixed by v2 features but still hard on unseen months |
| Waders | 120 | 4.6% | 0.505 | 2.5 | Very hard, needs focused gradient |
| Birds of Prey | 108 | 4.2% | 0.590 | 2.5 | Hard, confused with Gulls |
| Clutter | 84 | 3.2% | 0.910 | 1.0 | Actually easy (high RCS) |
| Geese | 83 | 3.2% | 0.621 | 2.0 | Hard, confused with Waders/Ducks |
| Ducks | 58 | 2.2% | 0.728 | 2.0 | Hard, confused with Pigeons |
| Cormorants | 40 | 1.5% | 0.296 | 3.0 | Hardest class, tiny sample |

**Implementation:** Custom LightGBM objective function (existing GitHub implementation: jrzaurin/LightGBM-with-Focal-Loss). CatBoost has built-in `Logloss` with `class_weights` but no native focal — would use LGB for this variant.

**Hyperparameters (adapted for 130 features):**
- num_leaves: 31 (simpler trees, more features handle interactions)
- colsample_bytree: 0.6 (each tree sees 60% of features — prevents overfitting)
- min_child_samples: 20 (more regularization)
- subsample: 0.7
- n_estimators: 2000, early_stopping: 100
- 50 random seeds

**Expected output:** 50 × 5-fold = 250 OOF probability matrices + 50 test probability matrices

#### Model B: 9 × One-vs-Rest Binary Rankers

**What:** Train 9 independent binary ranking models. Each one asks: "for class X, rank all tracks from most-likely-X to least-likely-X."

**Why this directly optimizes mAP:** macro mAP = average of 9 per-class APs. Each per-class AP depends ONLY on the ranking of that class's probability column. By training a dedicated ranker per class, we directly optimize each class's AP independently.

**Implementation options (in priority order):**

1. **CatBoost YetiRank** — `loss_function='YetiRank:mode=MAP'`, `eval_metric='MAP'`
   - Binary labels: 1 = target class, 0 = everything else
   - CatBoost handles the ranking objective natively
   - Produces relevance scores (not probabilities), higher = more likely

2. **LightGBM LambdaRank** — `objective='lambdarank'`, `eval_metric='map'`
   - Requires query groups. Use: one giant query group (all samples in one "query")
   - Or: group by month (each month is a query) — this makes the ranker learn month-invariant rankings

3. **XGBoost rank:map** — `objective='rank:map'`
   - Similar to LGB but XGBoost's implementation

**Hyperparameters:**
- Per-class: same features but alpha/class_weight adjusted to handle 1:20-1:65 imbalance
- Each ranker trained with 50 seeds
- Early stopping on MAP metric

**Expected output:** 9 relevance score columns per seed. Normalize to [0,1] range per column.

**Key insight:** The OvR rankers capture different information than the multi-class model. The multi-class model forces competition between classes (raising P(Gull) lowers P(Cormorant)). The OvR rankers are INDEPENDENT — a track can score high on both Gull AND Cormorant rankings. This independence helps when the multi-class model is "confused" between two similar classes.

#### Model C: Weather-Regime Gated Experts

**What:** Instead of one model for all conditions, train separate specialist models for different weather regimes.

**Regime definitions (from data):**
- **Regime 1: Clear + Warm** — cloud_cover < 50%, temp > 12°C, no rain
  - Species active: BoP (soaring), Clutter (insects), resident Gulls
  - Key features: CAPE, sunshine, thermal indicators

- **Regime 2: Overcast + Wet** — cloud_cover > 70%, rain_occurring = 1 OR precip > 0.01
  - Species active: Ducks (rain-tolerant), Cormorants, feeding Gulls
  - Key features: visibility, precipitation intensity, wave conditions

- **Regime 3: Strong Wind** — wind_at_bird_alt > 15 m/s OR wind_speed_100m > 20 m/s
  - Species active: Clutter (blade flash), migrants (using wind), Geese
  - Key features: wind shear, headwind, wind support

- **Regime 4: Calm + Low Light** — wind < 8 m/s, solar_elevation < 20°
  - Species active: Songbirds, Pigeons (local feeding), evening Ducks
  - Key features: crepuscular index, soil temp, arable distance

**Implementation:**
- Route each sample to its regime(s) using soft gating (a sample can belong to multiple regimes with different weights)
- Train a CatBoost per regime using only the samples in that regime
- Final prediction: weighted average of regime experts, weighted by gating function

**Risk:** With 2601 samples split across 4 regimes, each expert gets ~650 samples. Might overfit. Mitigation: use the regime experts as ADDITIONAL features (their predictions), not as replacement for the main model.

**Alternative (simpler):** Don't split the data. Instead, train 4 models on ALL data but with regime-specific sample weighting. Regime 1 expert: upweight clear+warm samples 3×. This keeps full data access while focusing each expert.

#### Model D: Tabular Foundation Model (Mitra or TabPFN)

**What:** Add a non-tree model that uses fundamentally different inductive bias.

**Mitra (preferred):**
- Amazon's tabular foundation model, built into AutoGluon 1.4+
- Uses 2D attention across BOTH rows and features (trees only split on single features)
- Excels on datasets < 5,000 samples with < 100 features
- Outperforms TabPFNv2, CatBoost in benchmarks
- NOT a tree — provides genuine ensemble diversity

**TabPFN v2.5:**
- Nature 2024 paper. Foundation model for small tabular data.
- Matches 4-hour AutoGluon ensemble in a single forward pass
- CAUTION: E83 showed TabPFN collapsed on unseen months. The month shift may defeat it.

**Implementation:**
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='bird_group', eval_metric='average_precision_macro')
predictor.fit(train_data, hyperparameters={'MITRA': {}})
probs = predictor.predict_proba(test_data)
```

**Expected role:** NOT the primary model. Used as a diverse ensemble member that captures feature INTERACTIONS that trees miss. Even 5-10% weight in the final ensemble could help.

#### Model E: Species-Level Classifier → Group Aggregation

**What:** Train at species-cluster level (~25-30 classes), then sum probabilities within each group.

**Species clusters:**
- Starling, Jackdaw, Crow, Fieldfare+Redwing (thrushes), Meadow Pipit, Barn Swallow, Chaffinch, Other_Songbird
- Stock Dove, Wood Pigeon, Other_Pigeon
- Greylag, Bean Goose, Barnacle, Other_Goose
- Lapwing, Golden Plover, Curlew, Oystercatcher, Other_Wader
- Mallard, Other_Duck
- Great Cormorant (only 1 species)
- Kestrel, Marsh Harrier, Buzzard, Other_BoP
- Black-headed Gull, Common Gull, Herring Gull, LBB Gull, Other_Gull
- Clutter, Turbine, Other_Clutter

Total: ~28 clusters (species with ≥5 samples get own cluster, rest merged)

**Implementation:**
```python
# Map species to cluster
species_to_cluster = {'Common Starling': 'Starling', 'Carrion Crow': 'Crow', ...}
y_species = train_df['bird_species'].map(species_to_cluster)

# Train multi-class model at species level
species_model = CatBoostClassifier(loss_function='MultiClass', ...)
species_model.fit(X, y_species)

# Aggregate to group level
species_probs = species_model.predict_proba(X_test)  # shape (N, 28)
group_probs = np.zeros((N, 9))
for species_idx, group_idx in cluster_to_group_mapping.items():
    group_probs[:, group_idx] += species_probs[:, species_idx]
```

**Why this is powerful:** It directly solves the within-class heterogeneity problem. The model learns that October Songbirds = Starling+Fieldfare+Chaffinch, while January Songbirds = Crow+Jackdaw. When it encounters a February test sample that looks like a Jackdaw, it assigns it to the Jackdaw cluster, which maps to Songbirds. The model ADAPTS to unseen-month species mixes automatically.

**Risk:** 28 classes with 2601 samples means small classes (Barnacle: 9, Buzzard: 12). Focal loss helps focus on these. Also, species labels may be noisy (cleanlab found 12.5% noise at group level — species level may be worse).

#### Model F: Pairwise Confusion Specialists

**What:** 4 dedicated binary classifiers for the top confusion pairs.

| Specialist | Pair | N samples | Key discriminative features |
|-----------|------|-----------|---------------------------|
| S1 | Gull vs Wader | 1623 | tidal_phase, speed (Waders 17.2 vs Gulls 14.5), wader_tidal_score |
| S2 | Gull vs Songbird | 1986 | rcs_mean_dB (Songbirds lower), alt_std, speed_cv |
| S3 | Gull vs BoP | 1611 | speed (BoP 11.8 vs Gulls 14.5), curvature, bop_soaring_score |
| S4 | Pigeon vs Duck | 180 | crepuscular_index, rain_occurring, dist_to_arable |

**How specialists integrate with ensemble:**
Each specialist outputs P(class_B | class_A_or_B). These become meta-features:
- When main model is confused between Gull and Wader (both ~0.3), the S1 specialist breaks the tie
- When main model is confident (Gull=0.9), specialist output doesn't matter (meta-learner learns to ignore it)

**Small sample risk for S4:** Pigeon vs Duck has only 180 samples. Use simple model (max_depth=3, few features) to prevent overfitting.

#### Model G: Flock Size Regressor → Feature Injection

**What:** Train a regression model to predict `n_birds_observed` from radar features, inject predicted flock size as a new feature for all other models.

**Why:** n_birds_observed is the single most discriminative variable:
- BoP/Clutter: always 1
- Cormorants: 3.0
- Ducks: 3.9
- Geese: 12.5
- Pigeons: 13.6
- Songbirds: 19.4
- Waders: 37.0

But it's train-only. A predicted version makes it available on test data.

**Implementation:** LGBMRegressor on log1p(n_birds), using radar features (size, RCS scintillation, RCS stats). Use OOF predictions for training to avoid leakage, direct predictions for test.

---

### 2B.3 Layer 3 — Distributional Aggregation (The Quantile Trick)

**What:** Instead of averaging predictions from 50 seeds, model the DISTRIBUTION of predictions and extract the 10th quantile.

**Why this matters for mAP:**

Consider a Cormorant track that 50 seeds predict as:
- 30 seeds: P(Cormorant) ≈ 0.6 (correctly confident)
- 20 seeds: P(Cormorant) ≈ 0.1 (confused, wrong)

Mean: 0.6×30/50 + 0.1×20/50 = 0.40
10th quantile: ~0.08 (the low end of the distribution)

Wait — that HURTS for a true Cormorant. So Q10 on a true positive is bad.

Now consider a Gull track that 50 seeds predict as:
- 50 seeds: P(Cormorant) ≈ 0.02 (correctly low, never confused)

Mean: 0.02
10th quantile: ~0.01

The Q10 compresses the false positive MORE than the true positive (0.40→0.08 vs 0.02→0.01). The KEY is that for AP, what matters is the GAP between true positives and false positives. Let's compute:

**With mean:**
- True Cormorant: 0.40
- False Cormorant (Gull): 0.02
- Gap: 0.38

**With Q10:**
- True Cormorant: 0.08
- False Cormorant (Gull): 0.01
- Gap: 0.07

Hmm, the gap shrinks. So Q10 doesn't help here.

**REVISED understanding:** The Q10 trick works best when:
- TRUE positives have NARROW distributions (consistently high across seeds)
- FALSE positives have WIDE distributions (high in some seeds, low in others)

In that case:
- True positive: narrow, Q10 ≈ mean
- False positive: wide, Q10 << mean

This is the realistic scenario: the model is CONSISTENTLY right about true Cormorants but INCONSISTENTLY wrong about false Cormorants (confused in some seeds, not others).

**Better approach: Use BOTH mean and Q10 as separate features for a stacking meta-learner:**
- `mean_pred[class]`: standard ensemble prediction
- `q10_pred[class]`: conservative prediction (suppresses uncertain FPs)
- `std_pred[class]`: uncertainty measure
- Let the meta-learner learn the optimal combination per class

**Implementation:**
```python
# For each sample i, class c:
# seed_preds[i, c, :] has shape (50,) — one value per seed
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

mean_preds = seed_preds.mean(axis=2)  # shape (N, 9)
q10_preds = np.percentile(seed_preds, 10, axis=2)  # shape (N, 9)
std_preds = seed_preds.std(axis=2)  # shape (N, 9)

# For 2-component mixture (more sophisticated):
for i in range(N):
    for c in range(9):
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(seed_preds[i, c, :].reshape(-1, 1))
        # Q10 from mixture CDF
        samples = gmm.sample(10000)[0].flatten()
        q10_mixture[i, c] = np.percentile(samples, 10)
```

---

### 2B.4 Layer 4 — Rank-Based Ensemble + Threshold Optimization

#### Step 1: Per-Class Isotonic Calibration

For each model's OOF predictions, apply isotonic regression PER CLASS to fix the probability-rank mapping.

```python
from sklearn.isotonic import IsotonicRegression

for c in range(9):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_probs[:, c], (y == c).astype(float))
    oof_calibrated[:, c] = iso.predict(oof_probs[:, c])
    test_calibrated[:, c] = iso.predict(test_probs[:, c])
```

This doesn't change within-model rankings (isotonic = monotonic) but fixes the SCALE so that probability averaging across models works better.

#### Step 2: Rank-Based Blending

Instead of averaging probabilities (which depends on calibration), average RANKS:

```python
for c in range(9):
    # For each model, rank all samples by P(class c)
    rank_A = rankdata(model_A_probs[:, c])
    rank_B = rankdata(model_B_probs[:, c])
    rank_C = rankdata(model_C_probs[:, c])
    rank_D = rankdata(model_D_probs[:, c])

    # Average ranks (optionally weighted)
    final_rank[:, c] = w_A * rank_A + w_B * rank_B + w_C * rank_C + w_D * rank_D

# Convert ranks back to pseudo-probabilities
final_probs = softmax(final_rank, axis=1)
```

Rank averaging is immune to calibration differences. It directly combines RANKINGS.

**Weight optimization:** Grid search over (w_A, w_B, w_C, w_D) on OOF to maximize macro mAP.

#### Step 3: Per-Class Threshold Optimization on the Simplex

After blending, optimize per-class thresholds τ_j to maximize macro mAP:

```python
from scipy.optimize import minimize

def neg_macro_map(thresholds, probs, y_true):
    """Negative macro mAP with per-class threshold adjustment."""
    adjusted = probs - thresholds  # shift each class's predictions
    # Compute macro mAP on adjusted predictions
    return -compute_macro_map(y_true, adjusted)

# Optimize 9 thresholds on OOF
result = minimize(neg_macro_map, x0=np.zeros(9), args=(oof_blended, y),
                  method='Nelder-Mead')
optimal_thresholds = result.x

# Apply to test
test_final = test_blended - optimal_thresholds
```

Recent paper (arXiv:2505.11276) showed +0.16 improvement on 9-class imbalanced data. Even +0.01-0.02 would be significant for us.

---

### 2B.5 Layer 5 — Post-Processing (unchanged)

Standard proven 3-stage pipeline:
1. GBIF ratio priors for unseen months (Feb, May, Dec)
2. NB evidence: speed, alt_mid, alt_range
3. Gated PoE update (gamma=0.10, tau=0.25)

NO new evidence channels (E172 proved they hurt).

---

## Part 2C: TIMESERIES ARCHITECTURE — Extracting the Trajectory Properly

We have 4-channel variable-length trajectories (lon, lat, alt, RCS × 5-200 timesteps at 1 Hz) that we reduce to 76 summary statistics. The raw sequence contains information our statistics CANNOT capture:

- The **SHAPE** of the altitude profile (inverted-U vs flat vs circling)
- **Cross-channel coupling** (RCS changes when altitude changes = aspect angle effect)
- **Heading change sequences** (regular turning = BoP, random = Clutter, straight = migration)
- The **speed-altitude interaction** over time (BoP: speed drops as altitude rises = soaring entry)

Previous sequence-only attempts (E06 1D-CNN: 0.5238, E08 MiniRocket: 0.4799) failed because they REPLACED tabular features. The correct approach is to ADD sequence representations TO tabular features.

### 2C.1 Path Signatures (highest priority — naturally fits our data)

**What:** A mathematically rigorous transform that maps a variable-length multivariate path to a fixed-size vector. At truncation depth L for d-dimensional input, the signature has Σ(k=0..L) d^k terms.

For our 4D trajectory (lon, lat, alt, RCS):
- Depth 2: 1 + 4 + 16 = **21 features**
- Depth 3: + 64 = **85 features**
- Depth 4: + 256 = **341 features**

**Why this is perfect for bird trajectories:**
- **Naturally handles variable length** — defined for any continuous path regardless of length
- **Invariant to time reparameterization** — robust to 1 Hz sampling irregularities
- **Captures cross-channel interactions** — how RCS co-varies with altitude changes (aspect angle), how speed changes with heading (turning dynamics)
- **Mathematically complete** — at sufficient depth, the signature uniquely determines the path up to reparameterization
- **Low dimensional** — 85 features at depth 3 for 4D paths. Very manageable.

**What it captures that our features miss:**
- Iterated integral of alt × RCS = how altitude and RCS change TOGETHER (aspect angle modulation)
- Iterated integral of lon × lat = trajectory AREA (circling encloses area, straight doesn't)
- Higher-order terms capture sequential patterns (climb-then-descend vs descend-then-climb)

**Implementation:**
```python
import iisignature

def extract_path_signature(hex_str, traj_time_str, depth=3):
    pts = parse_ewkb_4d(hex_str)
    times = parse_trajectory_time(traj_time_str)
    if len(pts) < 3:
        return np.zeros(iisignature.siglength(4, depth))

    # Normalize channels to [0,1] for numerical stability
    path = np.array([[p[0], p[1], p[2], p[3]] for p in pts])
    path = (path - path.mean(axis=0)) / (path.std(axis=0) + 1e-8)

    return iisignature.sig(path, depth)  # returns fixed-size vector

# For multi-scale: compute on first half, second half, full path
sig_full = extract_path_signature(hex_str, time_str, depth=3)  # 85 features
sig_first = extract_path_signature(first_half, ...)             # 85 features
sig_second = extract_path_signature(second_half, ...)           # 85 features
# Total: 255 signature features
```

**Multi-scale signatures:** Computing signatures on first half + second half of the trajectory captures whether the bird's behavior CHANGES mid-track (migration departure = different first vs second half, steady commute = similar halves).

**Library:** `pip install iisignature` — pure C, very fast.

### 2C.2 catch22 Per Channel (complementary to signatures)

**What:** 22 canonical time series characteristics per channel, covering distribution shape, autocorrelation, entropy, nonlinearity. Applied per channel: alt, RCS, speed_series, heading_series → 4 × 22 = **88 features**.

**Why it complements signatures:** Signatures capture geometric/interaction properties. catch22 captures STATISTICAL properties (entropy, autocorrelation structure, nonlinearity). Together they cover both geometry and statistics.

**What it captures that we miss:**
- `CO_trev_1_num` — time reversibility (asymmetric flight patterns like bounding)
- `FC_LocalSimple_mean3_stderr` — forecast error (how predictable is the trajectory?)
- `SB_BinaryStats_diff_longstretch1` — longest stretch above mean (sustained high-altitude phases)
- `DN_HistogramMode_5` — mode of distribution (most common altitude/RCS)

**Implementation:**
```python
from pycatch22 import catch22_all

def extract_catch22(trajectory_channel):
    result = catch22_all(trajectory_channel)
    return np.array(result['values'])

# Per channel
catch22_alt = extract_catch22(altitudes)
catch22_rcs = extract_catch22(rcs_values)
catch22_speed = extract_catch22(step_speeds)
catch22_heading = extract_catch22(step_headings)
# Total: 88 features
```

**Library:** `pip install pycatch22` — very fast (milliseconds per series).

### 2C.3 What NOT to Use (and why)

- **MiniRocket** — Produces ~10,000 features. With 2601 samples, this is severe overfitting risk. Feature selection to ~200 is needed, and at that point signatures + catch22 are cleaner. E08 already showed MiniRocket alone = 0.4799.
- **1D CNN** — E06 (0.5238), E16 (0.5193). Not enough samples for end-to-end learning. CNN features are learnable but not with 2601 samples.
- **MOMENT/foundation models** — E18 showed domain gap too large (0.6461). Radar time series ≠ the pre-training distribution.
- **DTW distances** — O(n²) computation, doesn't outperform feature extraction with trees.
- **Temporal Fusion Transformer** — Needs tens of thousands of samples minimum.

### 2C.4 Integration into the Architecture

The sequence features slot into the pipeline as an EXTENSION of Step 1 (Flight Behavior Extraction):

```
Trajectory (lon, lat, alt, RCS × time)
    ├── Hand-crafted features: 76 existing + 10 new (alt shape, RCS lags, speed profile) = 86
    ├── Path signatures: depth 3, full + first_half + second_half = 255
    ├── catch22: 4 channels × 22 = 88
    └── Total trajectory features: ~429

+ Tabular (weather, environment, external): ~60
+ Physics scores: 8
+ Derived (true_airspeed, temp_dewpoint, cormorant_residual, etc.): ~5
+ Predicted flock size: 1

GRAND TOTAL: ~503 features
```

**Feature selection is CRITICAL at this count.** With 503 features and 2601 samples (ratio 1:5), use:
1. CatBoost's built-in feature importance (train once, drop bottom 50%)
2. Or: backward elimination on LOMO (expensive but proven in E79)
3. Target: ~150-200 features after selection

This gives the species-level model rich behavioral + contextual + physics input while keeping the feature/sample ratio manageable.

---

## Part 3: IMPLEMENTATION PLAN

### Phase 1: Foundation (Day 1 — March 21)

| Step | What | Time | Expected Gain |
|------|------|------|---------------|
| 1a | New trajectory features (alt_curvature, rcs_lags, speed_cv) — 10 features | 30 min | Better sequence capture |
| 1b | 8 physics score features | 30 min | Domain knowledge injection |
| 1c | Flock size predictor (Model G) | 30 min | Recover most discriminative feature |
| 1d | Multi-seed CB ensemble (20 seeds, ~140 features, SGKF) | 2 hours | +0.01-0.02 (seed averaging) |
| 1e | Per-class temperature scaling + threshold optimization on OOF | 30 min | +0.01 (free post-processing) |
| 1f | Submit best variant | 5 min | Check LB |

### Phase 2: Ranking + Species (Day 2 — March 22)

| Step | What | Time | Expected Gain |
|------|------|------|---------------|
| 2a | Species-cluster classifier (Model E, ~28 classes) → aggregate to group | 2 hours | Solves heterogeneity |
| 2b | 9× OvR CatBoost YetiRank rankers (Model B, 20 seeds) | 3 hours | +0.01-0.02 (direct mAP opt) |
| 2c | 4× Pairwise confusion specialists (Model F) | 1 hour | Breaks tie on confusions |
| 2d | Rank-based blending of all models | 1 hour | +0.005-0.01 |
| 2e | Submit best variant | 5 min | Check LB |

### Phase 3: Advanced Ensemble (Day 3 — March 23)

| Step | What | Time | Expected Gain |
|------|------|------|---------------|
| 3a | LGB focal loss variant (Model A, per-class gamma) | 2 hours | +0.005-0.01 |
| 3b | Mitra/AutoGluon foundation model (Model D) | 1 hour | +0.005 (diversity) |
| 3c | Weather-regime experts (Model C, soft-gated) | 2 hours | +0.005-0.01 |
| 3d | Distributional aggregation (multi-seed → Q10/std meta features) | 1 hour | +0.005 |
| 3e | Full meta-learner: combine all model outputs + physics + uncertainty | 1 hour | Optimal combination |
| 3f | Submit best variant | 5 min | Check LB |

### Phase 4: Final Optimization (Day 4 — March 24, deadline)

| Step | What | Time | Expected Gain |
|------|------|------|---------------|
| 4a | Per-class threshold optimization on full ensemble OOF | 30 min | +0.005-0.01 |
| 4b | Standard PP (GBIF + NB) on best ensemble | 15 min | Unseen month adjustment |
| 4c | Generate all submission variants | 30 min | Options |
| 4d | Select and submit final 2-3 based on OOF evaluation | 15 min | Final LB |

**Submission variants (prioritized):**
1. `e174_full_ensemble_pp_threshold` — All models, rank-blend, PP, threshold opt
2. `e174_species_ranker_blend` — Species model + OvR rankers, rank-blend, PP
3. `e174_cb_multiseed_pp_threshold` — CB-only multi-seed, PP, threshold opt (safe fallback)
4. `e174_full_ensemble_raw` — All models, no PP (in case PP hurts on this base)
5. `e174_e79_multiseed_threshold` — E79's 36 features, multi-seed, threshold opt (safest)

---

## Part 4: RISK ANALYSIS

### What could go WRONG

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Physics scores overfit to train months | Medium | Based on month-invariant physics (tidal, wind model, RCS) |
| OvR rankers lose inter-class competition signal | Medium | Blend with multi-class model, never use alone |
| Multi-seed expensive (20×7 models×5 folds = 700 trains) | High | Start with 10 seeds, scale up if time allows |
| Threshold optimization overfits OOF | Medium | Nested CV for threshold selection |
| Mitra/TabPFN collapses on unseen months | High | Max 5-10% ensemble weight |
| Weather regime experts: too few samples each | Medium | Soft gating (weighted, not hard split) |
| Species-level model: noisy labels at species level | Medium | Merge rare species into clusters; focal loss |
| Flock size predictor: regression errors propagate | Low | Use predicted flock as one feature among 140+, tree robust to noise |
| Confusion specialists: overfit on small pairwise samples | Medium | Simple models (max_depth=3) for small pairs (S4: 180 samples) |
| Total pipeline complexity: integration bugs | High | Build incrementally, test each model independently first |
| OOF gains don't transfer to LB | Medium | Submit after EACH phase, don't wait until full pipeline |

### What we're CONFIDENT about

- Multi-seed averaging: free lunch, no downside risk
- Per-class threshold/temperature: pure post-processing, no retraining
- OvR rankers: directly optimize competition metric (theoretically sound)
- Species-level training: proven concept (hierarchical classification literature)
- Physics scores: domain-grounded, not arbitrary engineering
- Flock size proxy: n_birds_observed is overwhelmingly discriminative (mean 1 vs 37)
- New trajectory features: based on physics (altitude shape, RCS autocorrelation)

---

## Part 5: FULL MODEL INVENTORY

| Model | Type | Training Target | # Outputs | Seeds | Key Advantage |
|-------|------|----------------|-----------|-------|---------------|
| A | LGB focal loss multi-class | 9 groups | 9 probs | 20 | Focus on hard classes |
| B | 9× OvR CatBoost YetiRank | Binary per class | 9 scores | 20 | Direct mAP optimization |
| C | 4× Weather-regime experts | 9 groups (regime-weighted) | 4×9 probs | 10 | Regime-specific boundaries |
| D | Mitra/AutoGluon | 9 groups | 9 probs | 1 | Non-tree diversity |
| E | Species-cluster classifier | ~28 species clusters | 9 aggregated probs | 20 | Solves month heterogeneity |
| F | 4× Pairwise specialists | Binary per pair | 4 scores | 10 | Breaks confusion ties |
| G | Flock size regressor | n_birds_observed | 1 predicted count | 5 | Recovers privileged feature |

**Total model outputs per sample:** 9+9+36+9+9+4+1 = **77 meta-features** for the stacking layer.

---

## Part 6: SUCCESS CRITERIA

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| LB mAP | 0.59 | 0.61 | 0.63 |
| OOF mAP (SGKF) | 0.6954 | 0.72 | 0.74 |
| LOMO mAP | 0.5093 | 0.53 | 0.55 |
| Weak class min AP | 0.296 (Cormorants) | 0.40 | 0.50 |

**Pivot strategy:** If Phase 1 doesn't move LB above 0.59, pivot to the safest approach:
- E79's 36 features + multi-seed (20) + per-class threshold optimization + PP
- This combines the proven feature set with the free-lunch improvements
- No risk of feature dilution, just better aggregation of existing signal
