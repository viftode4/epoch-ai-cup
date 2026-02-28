# Radar bird classification: a complete research toolkit for 1Hz track data

**The most effective pipeline for classifying bird species from low-frequency radar combines physics-based feature engineering (wind-decoupled airspeed, flock discrimination via RCS statistics) with domain adaptation methods (CORAL, V-REx) and Bayesian prior correction (MLLS + ecological priors).** This synthesis draws from 80+ papers, 10+ datasets, and proven Kaggle strategies to address every layer of the classification problem — from raw feature extraction through distribution-shift-robust inference. The radar's 1Hz update rate precludes micro-Doppler wingbeat analysis, but track-level kinematics, RCS statistics, and aerodynamic invariants provide strong discriminative power across species groups. The key insight: **airspeed has a narrow, species-characteristic distribution** (~10.5 ± 2 m/s for passerines vs. 17–20 m/s for geese), while ground speed varies wildly with wind — making wind correction the single highest-value feature engineering step.

---

## 1. Extracting discriminative features from 1Hz radar tracks

At 1Hz, the Nyquist limit of 0.5Hz makes direct wingbeat extraction impossible (typical wingbeats: 3–15 Hz). However, track-level aggregate statistics carry substantial classification power, particularly when organized into five feature families.

**Speed features** (~15 features) are the most accessible: mean, standard deviation, coefficient of variation, quantiles, autocorrelation, and trend. Species-specific ranges are well-documented — small passerines cruise at **8–13 m/s** airspeed, waders at **14–18 m/s**, geese at **17–20 m/s**, and diving ducks at **18–23 m/s** (Alerstam et al. 2007, *PLoS Biology*). Speed variance partially captures bounding flight oscillations for larger birds whose flap-glide cycles exceed 2 seconds.

**Heading features** require circular statistics to avoid wraparound artifacts. The mean resultant length ρ̄ = √(C̄² + S̄²) measures directional consistency (ρ̄→1 for straight migration, ρ̄→0 for circling/foraging). Angular velocity (heading change rate), computed via `atan2(sin(Δθ), cos(Δθ))`, distinguishes soaring raptors (large consistent turns) from straight-flying migrants. Van Erp et al. (2023, *Marine Ecology Progress Series*) successfully identified thermal soaring from both GPS and radar tracks using precisely these metrics.

**RCS features** (~15 features) serve as body-size proxies and flight-activity indicators. Mean RCS separates large birds (swan/eagle: ~−20 dBsm) from small passerines (~−30 to −40 dBsm). RCS variance reflects wing flapping intensity (~10 dB fluctuation during active flapping vs. ~0 dB during gliding). Even at 1Hz, the coefficient of variation, skewness, kurtosis, and autocorrelation structure of RCS encode information about flight mode and target type.

**Trajectory shape features** include straightness index (net displacement / path length), sinuosity (Benhamou 2004), and fractal dimension. These reliably separate directed migration (high straightness) from foraging or soaring behavior.

**Automated extraction** complements domain-specific features. **catch22** (Lubba et al. 2019) provides 22 diverse features per channel — apply to speed, altitude, RCS, and heading-change series for 88 total features with minimal computation. **ROCKET/MiniROCKET** (Dempster et al. 2020, *Data Mining and Knowledge Discovery*) generates 20,000 features via random convolutional kernels and trains a ridge classifier, achieving state-of-the-art time-series classification accuracy. **tsfresh** with `EfficientFCParameters` offers comprehensive automated search including FFT coefficients and entropy measures.

**Key foundational papers**: Rosa et al. (2016, *Ibis*) achieved >80% accuracy classifying radar trajectories into five groups using Random Forest on speed, bearing, and echo features. Van Erp et al. (2024, *Methods in Ecology and Evolution*) developed a three-module post-processing framework for Robin Radar 3D-Fix data, filtering by airspeed range 8–23 m/s. Nilsson et al. (2018, *Journal of Applied Ecology*) confirmed that species-level identification from any single radar system generally requires wingbeat patterns, but broad group classification is feasible from track kinematics alone.

---

## 2. Wind-decoupled airspeed as the highest-value engineered feature

The physics is straightforward: **V̄_air = V̄_ground − V̄_wind**. Nussbaumer et al. (2022, *Ecology and Evolution*) showed definitively that nocturnal passerine migrants maintain airspeed of ~10.5 m/s with a narrow ±2 m/s spread across seasons, altitudes, and geography — while ground speed ranges from 5 to 18 m/s depending on wind. This makes airspeed the single most discriminative continuous feature.

The implementation decomposes ground velocity and wind velocity into East-North components:

```
va_E = vg × sin(track) − u_wind
va_N = vg × cos(track) − v_wind  
airspeed = √(va_E² + va_N²)
```

**ERA5 reanalysis** (ECMWF Copernicus) provides hourly wind at 0.25° × 0.25° resolution across 37 pressure levels, accessible via the `cdsapi` Python package. For each radar track, match the observation timestamp and altitude to the nearest ERA5 grid point and pressure level. Key levels for bird migration: **1000 hPa** (~surface), **950 hPa** (~500m), **925 hPa** (~760m), **850 hPa** (~1500m). Convert radar altitude to pressure using P ≈ P₀ × (1 − 2.26×10⁻⁵ × h)^5.256, then interpolate between bracketing levels.

**No temporal leakage concern exists**: wind is an exogenous physical variable independent of bird behavior. Dozens of published radar ornithology studies (Schekler et al. 2024; Safi et al. 2013; Shamoun-Baranes et al. 2007) use contemporaneous ERA5 wind data. Using the hour matching or immediately preceding the radar observation is standard practice.

Beyond airspeed itself, the **drift angle** (track direction minus heading) reveals species-specific compensation behavior. Geese and waterfowl exhibit full to partial wind compensation; nocturnal songbirds mostly drift over land but compensate near coastlines (Van Doren & Horton 2018, *Scientific Reports*). Adult raptors compensate ~71% while juveniles show full drift (Thorup et al. 2003). These behavioral differences produce additional discriminative features: drift angle mean/std, crosswind component, and the correlation between airspeed and headwind strength.

**Pennycuick's flight mechanics** (2001, 2008) provide theoretical speed priors. Minimum power speed Vmp and maximum range speed Vmr can be computed from morphology (mass, wingspan, wing area) using the **Flight 1.24** program. For any candidate species, expected airspeed falls in [Vmp, Vmr], creating bounded likelihood functions for Bayesian classification.

---

## 3. Flock discrimination through RCS statistics and centroid jitter

A flock of waders and a single goose may share identical mean RCS and speed, but their **scattering statistics are fundamentally different**. A single bird acts as one dominant scatterer with secondary reflections from wings (Swerling Case III, chi-squared 4 DOF), while a flock of N birds acts as N independent scatterers (Swerling Case I, chi-squared 2 DOF / exponential).

The **scintillation index** SI = var(σ_linear) / mean(σ_linear)² is the most direct discriminator. For a Swerling I flock, SI→**1.0** (exponential distribution has std = mean). For a Swerling III single bird, SI→**0.5**. A threshold of SI > 0.8 indicates a likely flock. The related coefficient of variation (CV ≈ 1.0 for flocks vs. ≈ 0.707 for singles) and Nakagami m-parameter (m ≈ 1 for flocks vs. m ≈ 2 for singles) provide corroborating evidence.

Additional RCS distribution features strengthen the discrimination:
- **Kurtosis**: Excess kurtosis ~6 for exponential (flock) vs. ~3 for chi-squared 4-DOF (single)
- **Deep fade fraction**: P(σ < 0.1 × μ) > 0.15 → likely flock (destructive interference causes near-zero returns)
- **RCS range**: Flocks show 15–30+ dB dynamic range vs. 5–15 dB for singles
- **Exponential fit p-value**: High Kolmogorov-Smirnov p-value against exponential → flock

**Centroid oscillation analysis** provides a complementary detection channel. When a radar tracks a flock as a single target, the reported position is the power-weighted centroid of all scatterers. As birds shift within the flock, this centroid jitters. Fitting a smooth trajectory (spline or Kalman smoother) and computing residuals reveals flock signatures: elevated RMS jitter, **temporal autocorrelation of residuals** (flock geometry changes slowly vs. white-noise radar errors for singles), and excess low-frequency power in the residual spectrum.

Urmy & Warren (2017, *Methods in Ecology and Evolution*) confirmed that flock RCS scales linearly with N (σ_flock ≈ N × σ_individual) in the incoherent regime. Flock size can be estimated as N̂ = mean(σ_track) / σ_species_expected, providing a continuous feature rather than a binary flock/single indicator.

Species-flock associations serve as strong priors: waders (Dunlin, Knot) form tight flocks of 10–10,000+; geese fly in V-formation (10–100s); raptors and herons are predominantly solitary. Combining RCS statistics with flock/single classification creates a two-level hierarchy that dramatically sharpens species-group identification.

---

## 4. Domain adaptation strategies that survive seasonal distribution shift

Training on summer/autumn months and predicting February, May, and December creates both covariate shift (different flight conditions) and label shift (different species compositions). Pseudo-labeling collapses under this regime because the source-trained model assigns confident but systematically wrong labels to target samples, particularly for species absent from training data. Error accumulates with each iteration, leading to mode collapse onto a few dominant classes (Rusak et al. 2021).

**The recommended three-tier approach**, ordered by implementation priority:

**Tier 1 — Feature-level domain normalization** (zero-cost, immediate gains). Z-score features per month to remove calendar-dependent shifts. Run adversarial validation (train a classifier to distinguish train vs. test features) to identify and remove/downweight domain-leaking features. Focus on physically invariant features: airspeed (not ground speed), RCS (body-size proxy), flight pattern periodicity.

**Tier 2 — Deep CORAL + V-REx** (moderate complexity, proven). CORAL (Sun et al. 2016) aligns second-order statistics (covariance matrices) between source and target feature distributions via a differentiable loss: L_CORAL = (1/4d²) × ||C_S − C_T||²_F. Adding this to the penultimate layer of an MLP encoder is straightforward. V-REx (Krueger et al. 2021, ICML) penalizes variance of per-environment losses: L = Σ_e R^e + β × Var({R^e}), where each training month is one environment. Combined loss: L_cls + λ×L_CORAL + β×Var(R_month).

**Tier 3 — Test-time adaptation** (free performance at inference). TENT (Wang et al. 2021, ICLR) updates batch normalization statistics and affine parameters on test data by minimizing prediction entropy. For neural networks with BN layers, this is nearly free and orthogonal to all training-time methods. Even simpler: just replacing BN running statistics with test-batch statistics often captures 30–50% of TENT's benefit.

**RAINCOAT** (He et al. 2023, ICML) merits special attention — it's the first UDA method for time series handling both feature and label shifts simultaneously, using time-frequency decomposition and Sinkhorn divergence alignment. It outperformed 13 SOTA methods by up to 16.33% and directly addresses the competition's dual shift problem. The DANN architecture with gradient reversal (Ganin et al. 2016) is more powerful than CORAL but harder to train; use the CDAN variant for class-conditional alignment if attempting it.

**Validation strategy is critical**: use leave-one-month-out cross-validation from training months, holding out months most similar to test months (e.g., November to approximate December).

---

## 5. Calibrated label shift correction and evidence fusion

The most robust label shift correction pipeline combines BCTS calibration → MLLS estimation → logarithmic pooling with ecological priors. This avoids the failure modes of both naive Bayes updating and BBSE.

**Step 1: Calibrate with Bias-Corrected Temperature Scaling** (Alexandari et al. 2020, ICML). Standard temperature scaling applies p_cal = softmax(z/T), but BCTS adds per-class biases: p_cal = softmax(z/T + b). The bias terms compensate for training-set class imbalance that the base model internalizes. Only K+1 parameters, fit on held-out validation data via NLL minimization. Alexandari et al. showed BCTS + MLLS **uniformly dominates** BBSE, RLLS, and uncalibrated MLLS across all tested settings.

**Step 2: Estimate target priors via MLLS** (Saerens et al. 2002). The EM algorithm iterates: E-step reweights calibrated posteriors by the ratio π̂_target(y)/p_source(y), then M-step averages the reweighted posteriors to update π̂_target. This maximizes a **concave** log-likelihood (Garg et al. 2020, NeurIPS), converging in 10–50 iterations. Initialize with ecological monthly abundance rather than uniform for faster convergence with limited target data.

**Step 3: Fuse with ecological prior via logarithmic pooling**. Rather than naive Bayes multiplication (which assumes conditional independence), use weighted logarithmic pooling:

```
log p_combined(y) ∝ α × log p_MLLS(y) + β × log p_ecological(y)
```

This is the only pooling operator satisfying **external Bayesianity** (Genest & Zidek 1984) — the unique method where combining posteriors yields the same result as combining priors then updating with likelihood. Weights α, β are tunable via cross-validation on held-out months with known distributions. Set floor probabilities π(y) ≥ ε to prevent any class from being zeroed out by incomplete ecological databases.

**BBSE failure modes** explain why previous approaches collapsed: the confusion matrix C becomes ill-conditioned with many classes, amplifying estimation noise through C⁻¹. RLLS (Azizzadenesheli et al. 2019, ICLR) adds ℓ₂ regularization, but Garg et al. showed the core issue is **information loss through confusion-matrix aggregation** — BBSE effectively bins probabilities into K categories, discarding fine-grained posterior information.

**For ranking metrics** (AP/PR-AUC): label shift correction IS needed. Unlike ROC-AUC (which is prior-invariant), precision depends explicitly on class prevalence: Precision = (TPR × π) / (TPR × π + FPR × (1−π)). Correcting for shifted priors directly improves AP by matching predicted score distributions to true class frequencies.

---

## 6. Ecological priors for Eemshaven across February, May, and December

Eemshaven sits on the Groningen coast at the edge of the Wadden Sea — one of Europe's most important bird areas. The species composition shifts dramatically across the three unseen test months.

**February** is dominated by massive wintering goose flocks (~2.4 million geese winter in the Netherlands). Greater White-fronted Goose, Barnacle Goose, Greylag Goose, and Brent Goose produce the majority of large, fast radar tracks. Winter waders (Oystercatcher, Dunlin, Bar-tailed Godwit, Curlew) persist along the coast, while large gulls (Herring, Great Black-backed) are constant year-round. Passerine presence is minimal — Snow Bunting and Twite on dike slopes.

**May** brings peak diversity with massive nocturnal passerine migration (warblers, flycatchers, thrushes crossing from the UK). Breeding species include 675 pairs of Black-headed Gulls on Eemshaven roofs, Common Terns, Avocets, and Western Marsh Harriers. Migrant waders (Whimbrel, Ruff, Greenshank) pass through. Late-departing geese head north. Radar tracks span the full size and speed spectrum.

**December** resembles February but with autumn arrivals still settling. Geese numbers build, winter duck assemblages (Eider, Scoter, Merganser) establish in harbour basins, and reduced but persistent wader flocks work the mudflats.

Approximate monthly prior vectors for radar classification:

| Group | Feb | May | Dec |
|-------|-----|-----|-----|
| Small passerines | 0.02 | 0.30 | 0.02 |
| Waders (all) | 0.20 | 0.23 | 0.17 |
| Small gulls | 0.10 | 0.07 | 0.10 |
| Large gulls | 0.12 | 0.05 | 0.15 |
| Geese | 0.25 | 0.05 | 0.22 |
| Ducks | 0.08 | 0.03 | 0.10 |
| Cormorants | 0.05 | 0.04 | 0.05 |
| Raptors | 0.03 | 0.02 | 0.03 |
| Terns | 0.00 | 0.08 | 0.00 |

**Key data sources for refining priors**: eBird Status & Trends (https://science.ebird.org/en/status-and-trends) provides weekly abundance maps. Sovon Vogelatlas (https://www.vogelatlas.nl/) gives Dutch breeding and winter distributions. AVONET (Tobias et al. 2022, *Ecology Letters*) provides morphological traits for 11,009 species — body mass predicts RCS, and hand-wing index predicts flight efficiency. BirdWingData (Shiomi et al. 2025, figshare) covers wingspan and wing area for 856 species. Movebank hosts directly relevant tracking datasets including Western Marsh Harriers in Groningen, Oystercatchers on Schiermonnikoog, and Lesser Black-backed Gulls along the North Sea coast. Alerstam et al. (2007, *PLoS Biology*) remains the definitive airspeed reference for 138 species.

**Radar studies at Eemshaven**: Bureau Waardenburg deployed the Robin Radar MAX 3D system at Eemshaven in 2018 — the first such deployment by an ecological consultancy. The RWE Black Blade study (2021–2025) at Westereems wind farm in Eemshaven used TNO's WT-Bird system with cameras and dedicated 3D bird radar. Bradarić et al. (2024) analyzed 5 years of Robin Radar data from nearby Borssele and Luchterduinen, finding spring migrants fly higher (median **286m**) than autumn migrants (median **169m**).

---

## 7. External datasets ranked by relevance to the competition

Three datasets stand out for direct applicability:

**Col de la Croix 1988** (DOI: 10.5281/zenodo.10209093) — **OPEN ACCESS, HIGHEST PRIORITY**. Individual tracks of nocturnal migrants from a Swiss tracking radar, with ground speed, airspeed, heading, climb rate, and wingbeat-derived classification into 9 groups (wader-type, passerine-type, swift-type, raptor, flock, etc.). This is the closest publicly available analog to competition data. A single 438 KB CSV. Use it to validate feature engineering and build class-conditional speed/heading distributions.

**BirdScan Community Reference Dataset** (DOI: 10.5281/zenodo.5734961) — **RESTRICTED, HIGH PRIORITY**. Labeled radar echoes from BirdScan MR1 with hierarchical classification (passerine/wader/swift/large bird/flock/bat/insect). ~1.9 GB including raw echo signatures. Request access from birgen.haest@vogelwarte.ch for non-commercial research. The `birdscanR` R package (CRAN) processes these data.

**WFIP3 DeTect Avian Radar Track Data** (DOI: 10.21947/2349403) — **A2e PORTAL, HIGH PRIORITY**. Processed track data from DeTect's S-band MERLIN radar during offshore deployment, June–September 2024. Three related datasets on the DOE A2e portal (OSTI IDs: 2349403, 2476343, 3007247). Register at https://a2e.energy.gov/ for access.

**Medium-relevance datasets**: The Aloft platform (https://aloftdata.eu/) provides vertical profiles of biological targets from ~151 European weather radars via the vol2bird algorithm — useful for seasonal migration pattern priors but aggregated, not individual tracks. The `bioRad` R package is the primary analysis tool. The 77 GHz FMCW drone/bird dataset (DOI: 10.5281/zenodo.5845259, open access) contains 75,868 measurements but at micro-Doppler scales irrelevant to 1Hz data — valuable only for conceptual understanding of RCS patterns.

**Pre-training strategy**: Use Col de la Croix for supervised feature validation and class-conditional distribution fitting. Apply self-supervised contrastive learning on WFIP3 unlabeled tracks to learn general "bird track" representations. Use Aloft data to build temporal/seasonal context features and validate speed/direction distributions.

---

## 8. Advanced modeling: conformal prediction leads, TabPFN and V-REx follow

**Conformal prediction is the highest-priority addition** — it wraps around any base model with zero accuracy penalty, providing calibrated uncertainty estimates that directly identify OOD examples from unseen months. Implementation via MAPIE takes ~20 lines:

```python
from mapie.classification import MapieClassifier
mapie = MapieClassifier(estimator=clf, cv="prefit", method="cumulated_score")
mapie.fit(X_calib, y_calib)
y_pred, y_sets = mapie.predict(X_test, alpha=0.05)
```

**Weighted conformal prediction** (Tibshirani et al. 2019, NeurIPS) explicitly handles covariate shift by assigning likelihood ratio weights w(x) = P_test(x)/P_train(x) to calibration scores, estimated via a domain classifier trained on unlabeled test features. Barber et al. (2023, *Annals of Statistics*) generalized this beyond exchangeability. For 2025, Wasserstein-Regularized CP (Xu et al., ICLR 2025) handles general distribution shift, reducing coverage gaps to ~3.2% with 37% smaller prediction sets.

**TabPFN v2.5** (Hollmann et al., November 2025) achieves 100% win rate vs. default XGBoost on datasets up to 10K samples. It performs in-context learning — the entire training set is processed in a single forward pass, approximating Bayesian posterior prediction. **Drift-Resilient TabPFN** (Helli et al., NeurIPS 2024) specifically addresses temporal distribution shifts, improving accuracy from 0.688→0.744 on shifted datasets. Key limitation: designed for ≤10 classes, with the `many_class` extension degrading on >10 categories.

**V-REx** (Krueger et al. 2021, ICML) is the simplest effective invariant learning method. Adding `+ β × Var(per_month_losses)` to any training objective penalizes unequal performance across months, encouraging the model to rely on month-invariant features. It requires no target data and avoids the documented failure modes of IRM (overparameterization collapses the penalty; too few environments undermine guarantees — Rosenfeld et al. 2020).

**Anchor regression** (Rothenhäusler et al. 2021, JRSS-B) provides formal distributional robustness by using month indicators as "anchors" and penalizing correlation between anchors and residuals. The γ parameter interpolates between OLS (γ=0) and causal estimates (γ→∞).

---

## Conclusion: the integrated pipeline

The complete system combines physics-based invariants, statistical discrimination, and robust inference in five stages. First, **wind-correct all tracks** using ERA5 to compute airspeed, drift angle, and compensation features — this single step transforms the noisiest feature (ground speed) into the most discriminative one (airspeed). Second, **extract ~100 domain features** (speed, altitude, heading, RCS, trajectory shape statistics) plus catch22/ROCKET automated features. Third, **compute flock indicators** (scintillation index, Nakagami m, centroid jitter) to create a binary flock/single split that sharpens all downstream classification. Fourth, **train with domain adaptation**: CORAL loss on shared encoder features + V-REx variance penalty across training months + TENT adaptation at test time. Fifth, **calibrate and adjust**: BCTS calibration → MLLS target prior estimation initialized with ecological priors → logarithmic pooling → conformal prediction sets for uncertainty quantification.

The most neglected opportunity is wind correction — it requires only ERA5 data and basic vector arithmetic but converts a weather-confounded signal into a species-specific invariant. The most common pitfall is naive pseudo-labeling under label shift, which collapses reliably. The Col de la Croix dataset (freely available) provides immediate validation of the entire feature engineering pipeline against labeled bird group tracks with known kinematics.