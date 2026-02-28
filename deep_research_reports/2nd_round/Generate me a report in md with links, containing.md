
***

# Radar- and Model-Focused Literature for Bird Classification under Low-Frequency Sampling and Distribution Shift

## 1. Low-Frequency (≈1 Hz) Kinematic Signatures \& Radar Features

### 1.1. Radar-based kinematic features for birds and drones

Several radar studies classify birds vs drones or other small targets using *track-level kinematics* rather than high-rate micro‑Doppler, which is directly relevant when only ≈1 Hz updates are available.[^1][^2][^3]

- Doumard et al. (“Radar Discrimination of Small Airborne Targets Through Kinematic Features and ML”) use holographic radar trajectories and extract angle, slope, velocity and combinations thereof, then classify drones vs “other” using random forests, achieving >95% accuracy on experimental data.[^3]
- A Chinese airport-surveillance study (classification of birds, quadcopters and dynamic precipitation) uses range rate, curvature and an “oscillation factor” derived from X‑band surveillance radar tracks; these kinematic features plus standard radar variables feed a random forest classifier and give good separation between birds, drones and weather.[^1]
- A recent L‑band monostatic/bistatic study compares performance of bird/drone classifiers trained and tested across geometries, again using spectrograms and kinematic descriptors; mixing data from multiple radar geometries improved robustness.[^4]

These works indicate that with 1–10 Hz track updates, discriminative features include speed statistics, curvature/turn rate, vertical dynamics and low‑frequency oscillation metrics derived from smoothed range/altitude rather than micro‑Doppler.[^2][^3][^1]

**Key papers:**

- Doumard et al., *Radar Discrimination of Small Airborne Targets Through Kinematic Features and Machine Learning*, Cranfield University report.[^3]
- Zhang et al., *Classification of bird and drone targets based on motion characteristics and random forest model using surveillance radar data*.[^1]
- Colorado Montaño, *BaTboT: a biologically inspired flapping and morphing bat robot* (for flapping mechanics background).[^5]


### 1.2. Extracting flight style \& tortuosity from sparse tracks

Movement-ecology work on GPS trajectories provides a toolbox for quantifying *tortuosity, turning behavior and bounded vs continuous flight* that can be applied directly to sparse radar tracks.[^6][^7][^8]

- Shimatani et al. introduce a circular autoregressive movement model to analyze turning angles, using resultant vector length and circular statistics to separate oriented vs tortuous movement and to model asymmetric wind effects; they apply it to seabird GPS tracks.[^8]
- Åkesson \& Bianco review route simulations under different compass mechanisms and show how large‑scale trajectory curvature can be explained by different orientation strategies, which can be parameterized with heading distributions and resultant lengths.[^7]
- An MSc thesis on anomalous trajectories in bird datasets uses Euclidean distance between re‑sampled trajectories and route probability models, illustrating how to parameterize routes via discrete time warped paths.[^9]

These works suggest useful low‑frequency features:

- Mean and variance of step length and turning angle in a sliding window (capturing tortuosity).
- Resultant length $R$ of heading vectors and its temporal evolution (straight vs highly tortuous segments).
- Segment‑wise classification into “commuting”, “searching” and “restless” behaviors using hidden Markov or autoregressive circular models.[^7][^8]

**Key references:**

- Shimatani et al., *Toward the Quantification of a Conceptual Framework for Movement Ecology Using Circular Statistical Modeling*, PLoS ONE 2012.[^8]
- Åkesson \& Bianco, *Route simulations, compass mechanisms and long-distance migration flights in birds*.[^7]


### 1.3. Wingbeat and “bounding flight” information at low sampling

Classical radar ornithology has quantified wingbeat frequency and patterns for >150 species using tracking radar and cine/video, providing reference values for continuous flapping vs bounding flight modes.[^10][^11][^12][^13]

- Bruderer et al. (2010) present wingbeat frequency data for 153 Western Palearctic species, largely from tracking radar, and explicitly distinguish continuous flapping flyers from passerines with bounding flight (flap–pause cycles); they confirm Pennycuick’s allometric model with minor coefficient adjustments.[^11][^12]
- Pennycuick (1996) analyzes wingbeat frequency as a function of mass, wing span and wing area for 47 species in steady cruising flight and provides dimensionally consistent formulas; together with Bruderer this yields species‑level priors on typical wingbeat rates.[^13][^10]

At ≈1 Hz sampling you cannot resolve individual wingbeats, but you *can* use these priors to define:

- Expected presence/absence of visible micro‑oscillation in *smoothed* altitude or radial-range signals: small passerines with bounding flight should show low‑frequency up–down cycles (“staircase” altitude) over tens of wingbeats, whereas cormorants and geese in continuous flapping or gliding show smoother altitude traces.[^12][^11]
- Species‑ or guild‑specific constraints on feasible climb/descent rates and maneuver sharpness, derived from allometric relationships between mass, wing area and power.[^14][^10]

**Key references:**

- Bruderer, Peter, Boldt \& Liechti, *Wing‑beat characteristics of birds recorded with tracking radar and cine camera*.[^11][^12]
- Pennycuick, *Wingbeat frequency of birds in steady cruising flight: new data and improved predictions*.[^13]
- Alerstam et al., *Flight speeds among bird species: allometric and phylogenetic effects* (airspeed vs mass, corrected for wind).[^14]


### 1.4. RCS fluctuation and multi-scan statistics

Radar theory and bio‑radar work emphasize that multi‑scatterer targets (e.g. flocks) and complex shapes produce RCS fluctuations that can be modeled with Swerling‑type distributions or more detailed EM models.[^15][^16][^17][^18][^19]

- Radar textbooks and tutorials show how composite RCS from multiple scatterers fluctuates strongly with frequency and aspect angle; when many small scatterers contribute comparably, Swerling I/II (chi‑square with 2 degrees of freedom) describes pulse‑to‑pulse or scan‑to‑scan amplitude statistics.[^17][^20][^19][^15]
- Size‑matters work for migration radar proposes estimating bird size from wingbeat frequency extracted from echo‑signal modulations, independent of absolute RCS, demonstrating that micro‑modulation characteristics carry taxonomic information.[^16]
- An EM modeling study of bats shows that detailed anatomical models can reproduce measured RCS patterns as a function of orientation, suggesting that similar modeling is feasible for key bird guilds if needed.[^18]

For low‑rate scans you can treat the per‑scan intensity series as a low‑frequency RCS time series and derive:

- Per‑track log‑amplitude variance, skewness and kurtosis as RCS fluctuation descriptors.
- Goodness‑of‑fit to Swerling‑like distributions (e.g. via Kolmogorov–Smirnov between empirical amplitude and chi‑square models) as features for distinguishing single vs multi‑scatterer behavior (see Section 3).[^21][^19][^17]

***

## 2. Wind-Decoupled Aerodynamics \& Airspeed Estimation

### 2.1. Standard methods to derive airspeed from ground speed and wind

Classical migration and movement studies routinely compute airspeed vectors by subtracting wind from ground speed, exactly matching your desired physics‑based preprocessing.[^22][^23][^24][^25][^26][^14]

- Alerstam et al. (2007) analyze flight speeds of many species and correct for wind by subtracting the measured wind vector (from tracked balloons) at bird flight altitude from ground‑speed vectors, obtaining mean airspeeds and vertical speed per track.[^14]
- Kogure et al. show for European shags that airspeed and heading direction can be estimated by combining ground‑speed vectors from GPS with wind vectors measured at colonies; they explicitly compute tailwind components and airspeeds per second and compare behavior to Pennycuick’s $V_{mp}$ and $V_{mr}$ predictions.[^25]
- Safi et al. similarly calculate airspeed by subtracting modeled or measured wind from GPS‑derived ground speed, and then analyze speed and direction responses to wind at different temporal and spatial scales.[^26]
- A wind‑estimation paper for soaring birds derives wind vectors from high‑frequency GPS by assuming circular thermalling and then solves for wind and airspeed via maximum likelihood; this validates joint estimation of bird and wind vectors given pressure and GPS, and confirms the standard vector subtraction approach.[^24]
- A recent HMM‑based trajectory reconstruction for geolocator‑tagged birds uses historical reanalysis wind fields and constrains transitions with realistic airspeed ranges (5–30 m s⁻¹) derived from flight mechanics.[^23]

The basic physics used in these works is:

- Obtain ground velocity vector $\vec{v}_g = \frac{\Delta \vec{x}}{\Delta t}$ from radar track positions.
- Obtain wind vector $\vec{v}_w$ at the bird’s location, time and altitude from reanalysis (e.g. ERA5) or mesoscale models.
- Compute airspeed vector $\vec{v}_a = \vec{v}_g - \vec{v}_w$, then airspeed magnitude $|\vec{v}_a|$ and heading (direction of $\vec{v}_a$).[^14][^25][^26][^24]


### 2.2. Avoiding temporal leakage and month inference

Multiple studies show that wind patterns shape migration strongly, making them a major temporal confounder if fed directly into supervised models.[^27][^28][^29][^30]

- Full‑year weather radar analyses across western Europe show that seasonal patterns in airspeed and ground speed are heavily driven by synoptic winds; favorable tailwinds speed up migration in spring but not necessarily in autumn.[^27]
- Studies over the North Sea show that intense migration nights are associated with high‑pressure systems and tailwinds in spring, and sidewinds in autumn, highlighting strong coupling between month and wind regime.[^28][^29][^30]

To avoid leakage of “month via wind field” into the model:

- Perform all wind‑related calculations *upstream*, deriving airspeed magnitude, heading relative to the ground track, and simple “wind effort” metrics such as $|\vec{v}_a| - |\vec{v}_g|$ or estimated mechanical power based on Pennycuick’s flight theory; only these derived, wind‑invariant or weakly seasonal features enter the classifier.[^31][^32][^33]
- Use fixed climatological or month‑agnostic caps for wind‑based filters; for example, follow van Erp et al. and Alerstam’s recommendation of acceptable bird airspeeds between 5 and 30 m s⁻¹ and treat tracks outside this as likely non‑bird or mis‑tracked, without exposing the model to raw wind vectors.[^34][^35][^14]


### 2.3. Drift compensation and crosswind behavior

A large body of radar and GPS work quantifies how birds compensate for crosswinds, which suggests additional physics‑based invariants.[^36][^37][^38][^39][^22][^26]

- Studies of nocturnally migrating passerines using weather radar and modeling show partial drift compensation and wind selectivity, with individuals accepting some drift to exploit favorable winds.[^36]
- Experiments with common swifts using tracking radar demonstrate *complete* compensation for crosswinds, with swifts adjusting headings into the wind to maintain track direction, and increasing airspeed with increasing side‑wind component.[^37]
- Shorebirds in the Baltic region show almost full drift over sea immediately after departure, followed by increasing compensation over land later in the same night.[^39]
- Seabirds tracked by GPS and inverse modeling of heading and wind show sophisticated compensation and orientation over the ocean, matching coarse atmospheric reanalysis winds.[^22]

These results suggest track‑level features such as:

- Estimated drift angle $\alpha = \angle(\vec{v}_a, \vec{v}_g)$.
- Degree of compensation (ratio of crosswind component projected into heading vs ground‑track), summarized over track segments.

These physics‑derived quantities can be computed without exposing the raw wind field or timestamp, thus reducing temporal leakage while encoding biologically meaningful effort and control.

***

## 3. Flock Dynamics vs Single Large Targets

### 3.1. RCS variance and Swerling‑type behavior for flocks

Radar theory treats targets with many similar scatterers as exhibiting chi‑square‑distributed RCS fluctuations (Swerling I/II), whereas single targets with dominant scatterers follow different Swerling classes.[^20][^19][^21][^17]

- Classical radar texts and tutorials show that when many isotropic scatterers of similar size contribute, composite RCS fluctuates Pulse‑to‑Pulse with an exponential (Swerling I/II) distribution; for a dominant scatterer plus smaller ones, Swerling III/IV models (chi‑square with 4 degrees of freedom) are appropriate.[^19][^17][^20]
- MATLAB’s documentation on Swerling target models provides examples of simulating RCS fluctuations for complex targets and explicitly connects Swerling models to modeling multiple independent scattering centers.[^40][^20]

In practice, a flock of small waders will approximate a many‑scatterer Swerling‑I/II process at the *cell* level, while a single goose will produce more stable RCS (closer to constant‑RCS point target) with lower normalized variance across scans.[^41][^21]

### 3.2. Empirical avian RCS fluctuation data

Several field studies quantify species‑ and flock‑size dependence of avian echoes, although often at higher sampling rates than your system.[^42][^43][^44][^41]

- A 2024 field validation of avian radar surveys reports that detection range and echo strength scale with species and flock size, with Bewick’s swans detectable up to ~1.8 km as pairs vs much shorter for small species; they provide survey procedures for validating RCS and detection probability using ornithodolites.[^42]
- A field validation comparing radar tracks with line‑transect surveys in intertidal areas finds that detection probability of single low‑altitude individuals is about 0.5 within ~1.5 km, with strong dependence on altitude, distance, substrate and species; detection was biased toward higher altitude and larger birds.[^41]
- Visualizing aspect‑dependent RCS of seabirds using tracking radar shows that RCS is highly aspect dependent and can vary strongly as birds flap or change orientation relative to the beam.[^44]

Combined, these indicate that:

- Instantaneous RCS is noisy due to aspect and wingbeat, but *relative* scan‑to‑scan variance and higher‑order statistics can still distinguish many scatterers (flock) vs one large scatterer.
- You can calibrate thresholds using simulated Swerling models and validated variance ranges from such field studies.


### 3.3. Micro‑oscillations of flock centroids

Micro‑Doppler work has recently been extended to *formation wing‑beat modulation* (FWM), which analyzes micro‑oscillatory behavior of a flock’s echo as a whole.[^45][^46]

- Gong et al. introduce FWM: radar echoes from seagull flocks show groups of spectral peaks spaced according to wingbeat rates and phasing strategies; they demonstrate that this effect can estimate bird number and mean wingbeat rate for flocks via micro‑Doppler modulation, using X‑band radar.[^46][^45]

At 1 Hz, classical micro‑Doppler spectrograms are unavailable, but a related idea remains: the *centroid trajectory* of a flock can show small, quasi‑periodic oscillations from coherent flapping and adjustment of individuals, whereas a single large bird should show smoother ballistic motion with occasional turns.

You can therefore define low‑frequency features such as:

- Power spectral density of detrended centroid deviations (after removing mean drift), summarized by dominant low‑frequency peaks and spectral flatness.
- Ratio of high‑frequency to low‑frequency motion energy in the centroid, as a proxy for formation‑level flutter.

These features can be interpreted in light of FWM and Swerling models, even if not resolving individual wingbeats.

***

## 4. Unsupervised Domain Adaptation (UDA) \& OOD Generalization for Time-Series Trajectories

### 4.1. Benchmarks and general UDA methods for time series

Recent work has systematically benchmarked deep UDA methods for time‑series classification, including adversarial, CORAL‑type alignment and contrastive approaches.[^47][^48][^49][^50][^51][^52]

- Fawaz et al. introduce AdaTime and a comprehensive benchmark of deep UDA algorithms (adversarial, MMD‑based, contrastive, frequency‑domain) on time‑series datasets (HAR, machine fault diagnosis, ECG, etc.), showing that with careful hyperparameter selection some vision‑style UDA methods remain competitive.[^49][^50][^51][^52]
- Their Data Mining and Knowledge Discovery paper (*Deep Unsupervised Domain Adaptation for Time Series Classification: a Benchmark*) provides code and standardized backbones (e.g. InceptionTime), and highlights that some UDA methods can underperform naive source‑only baselines if tuned improperly.[^50][^53][^51]
- A 2025 benchmark on “Deep Feature Unsupervised Domain Adaptation for Time‑Series Classification” introduces DFUDA, combining consistency pre‑training with end‑to‑end adaptation, and shows gains on fault‑diagnosis and HAR datasets.[^47]
- Hierarchical UDA for time series (VLH‑DA) decomposes tasks into local pattern recognition and sequence‑level classification, aligning domain distributions at both levels.[^48]

These results suggest that for your radar tracks:

- You can reuse InceptionTime‑like backbones and plug in domain‑adversarial (DANN), MMD‑based, contrastive or CORAL‑style losses, with domain defined by month or explicit temporal blocks.
- Hyperparameter selection must be done via unsupervised criteria (e.g. source risk or importance‑weighted cross‑validation) as demonstrated in the benchmarks to avoid target‑label leakage.[^51][^52][^50]


### 4.2. CORAL and covariance alignment in tabular/time-series settings

Correlation Alignment (CORAL) aligns second‑order statistics (covariance matrices) of source and target features in an unsupervised way and has been applied in multiple domains.[^54][^55][^56][^57]

- Sun \& Saenko’s CORAL learns a linear transformation (whiten–recolor) that aligns source feature covariance with target covariance, reducing domain discrepancy without target labels.[^55]
- Later work extends correlation‑based alignment to large time‑series repositories, using a “CORAL” model of time‑series subsequence clusters for fast similarity search; this shows that covariance‑level alignment is computationally cheap even for large time‑series datasets.[^54]
- A 2025 Test‑Time Correlation Alignment (TCA) method aligns feature statistics at test time alone, showing that covariance alignment can also be used purely as an online adaptation without labels.[^57]

For your setting, CORAL can be used at the *feature* level:

- Compute feature covariance per month (or season) in an unsupervised way, and learn a transformation that makes each month’s covariance match a reference (e.g. training months), encouraging the model to operate in a feature space where months are indistinguishable.


### 4.3. Domain-Adversarial Neural Networks (DANN) for radar/time series

Domain‑adversarial training with a gradient reversal layer is a standard method to learn domain‑invariant representations.[^58][^59][^60][^61][^62]

- Ganin \& Lempitsky’s DANN introduces a feature extractor feeding both a label predictor and a domain classifier via a gradient reversal layer, such that the features are simultaneously predictive of labels and maximally confusing for domain discrimination.[^60][^62][^58]
- DANN has been applied to diverse time‑series applications (fault diagnosis, hydrology, mooring failure detection), often using recurrent or CNN encoders.[^63][^59][^61]

In your competition, define “domain” as month or as radar configuration, and train a DANN‑style network so that:

- The feature extractor is forced to remove month‑specific covariate information while preserving class‑discriminative structure, helping to align train and test months.


### 4.4. Contrastive and mixup-based UDA for trajectories

Contrastive learning and time‑series mixup have been adapted to UDA with promising results.[^64][^65][^63]

- CoTMix proposes contrastive domain adaptation for time‑series via temporal mixup, mixing subsequences across domains and adding a contrastive loss to encourage domain‑invariant, class‑consistent representations; it shows strong performance on several UCR‑style datasets.[^65]
- Recent methods such as DACAD and MR‑CoTMix combine multiscale feature extraction with contrastive alignment, demonstrating robustness when anomalous classes or variable speeds create domain‑specific distortions.[^63][^64]

For bird‑radar tracks, you could:

- Treat complete tracks or fixed‑length windows as instances, sample positive pairs across months for the same approximate region/species label (source) and negative pairs across species, and add a cross‑month contrastive loss on top of standard classification.


### 4.5. Invariant risk minimization \& multi-environment learning

Invariant Causal Prediction (ICP) and Invariant Risk Minimization (IRM) frameworks aim to learn predictors whose relationships between features and labels are stable across environments.[^66][^67]

- Recent work on mining invariance from nonlinear multi‑environment binary classification establishes conditions under which invariant relationships can be identified from heterogeneous environments.[^67]
- TabPFN’s drift‑resilient extension (see Section 8) explicitly uses structural causal models with mechanism shifts to simulate temporal domain shift and trains a prior‑data fitted network to be robust to such shifts.[^68][^69][^66]

Although these are not specialized to radar, they suggest that:

- Using months as environments and penalizing environment‑specific gradients (IRM‑style) or searching for features whose conditional distribution of labels is stable across months could help identify truly causal kinematic or aerodynamic features.

***

## 5. Label Shift, Prior Shift \& Evidence Fusion

### 5.1. Product-of-experts and Bayesian fusion of posteriors

Product‑of‑experts (PoE) is a principled way to combine multiple probabilistic models by multiplying their (possibly unnormalized) densities and renormalizing.[^70]

- Welling’s Scholarpedia article defines PoE and contrasts it with mixtures: in PoE, events with low probability from any expert are strongly down‑weighted, making PoE appropriate when each model encodes a soft constraint.[^70]

Modern Bayesian classifier‑fusion work extends this idea to correlated, noisy probabilistic classifiers:

- A Bayesian fusion model based on a correlated Dirichlet distribution (Classifier Fusion Model, CFM) shows how to fuse base probability vectors accounting for biases, variances and correlations, yielding Bayes‑optimal fused posteriors under the model.[^71]
- The authors show that naive independent‑expert fusion (e.g. independent PoE) can be sub‑optimal when base models are correlated, and that CFM/IFM can still reduce uncertainty and improve log‑loss on both synthetic and real datasets.[^71]

For your setting, instead of Naive Bayes‑style likelihood multiplication, you can:

- Treat the base model’s posterior as one “expert” and ecological priors (Section 6) as another, then fit a low‑dimensional PoE‑style or correlated‑Dirichlet fusion model on validation data to learn how strongly to weight each component.


### 5.2. Calibration and multi-class probability correction

There is an extensive literature on multi‑class calibration: temperature scaling, Dirichlet calibration, mutual‑information–based binning and more.[^72][^73][^74][^75][^76][^77]

- A recent survey on classifier calibration reviews proper scoring rules, visualization tools and post‑hoc calibration methods for binary and multi‑class models, including temperature scaling, histogram binning, Dirichlet calibration and others.[^72]
- Kull et al. propose Dirichlet calibration, a native multi‑class method that transforms uncalibrated probability vectors via a Dirichlet‑based mapping (implemented as log‑probability linear layer + softmax), and show improvements in ECE, log‑loss and Brier score across many datasets.[^76]
- Johansson et al. discuss calibrating multi‑class models in the context of conformal prediction and uncertainty, and show that good calibration is essential for downstream decision‑making.[^75]

In your pipeline, you can separate:

- Learning a domain‑robust *score function* via UDA.
- Applying post‑hoc multi‑class calibration (e.g. Dirichlet calibration) on an in‑distribution validation set, then adjusting for label shift as described below.


### 5.3. Label shift estimation and correction (BBSE and beyond)

Lipton et al.’s Black Box Shift Estimation (BBSE) is the canonical method for estimating label‑shift ratios $w_y = q(y)/p(y)$ from a frozen classifier and unlabeled target data.[^78][^79][^80][^81]

- BBSE assumes label shift (marginal $p(y)$ changes, class‑conditional $p(x|y)$ remains constant) and uses the confusion matrix of a black‑box classifier on source data plus predicted label frequencies on target data to estimate class‑prior ratios; it then reweights losses or posteriors to correct predictions.[^78][^79][^80]
- The method is consistent but can suffer from high variance and negative importance weights in small samples or for rare classes; practical implementations clip negative weights.[^79]
- Later work unifies BBSE with maximum‑likelihood label‑shift estimators and shows that well‑calibrated maximum‑likelihood approaches (e.g. MLLS) can outperform BBSE and be more stable.[^82]

More recent Bayesian and robust methods improve on raw BBSE:

- “Bayesian Quantification with Black‑Box Estimators” shows that adjusted classify‑and‑count, BBSE and ratio‑estimation methods can be put in a Bayesian framework, offering posterior distributions over class proportions and revealing brittleness of point estimators under small samples.[^83]
- Graph‑Smoothed Bayesian BBSE uses Laplacian–Gaussian priors over class‑log‑priors and confusion‑matrix columns connected by a label‑similarity graph, yielding more stable priors for rare or correlated classes and provable variance reductions.[^84]
- Robust multi‑source label‑shift adaptation proposes truncated‑mean and other robust estimators for target class proportions and shows improved performance under contamination.[^85]

For your competition scenario (small sample per unseen month, high class imbalance, ranking metrics):

- Treat label‑shift correction as a post‑processing step on calibrated probabilities, using robust or Bayesian variants of BBSE instead of vanilla Naive Bayes assumptions; for example, enforce non‑negativity and smooth priors via graph‑smoothed BBSE or regularized MLLS.[^83][^84][^82]
- Remember that ROC‑AUC is invariant to monotonic reweighting of scores, while PR‑AUC and accuracy are affected by class priors; label‑shift correction can improve accuracy but cannot change PR‑AUC given fixed ranking.[^86][^87][^81][^88]

***

## 6. High-Fidelity Ecological Priors for the North Sea / Eemshaven Region

### 6.1. Radar-based migration studies over the North Sea and Dutch coast

Multiple studies use dedicated bird radars and the European weather‑radar network to quantify migration intensity, altitude and timing over the North Sea and adjacent coasts, including the Dutch sector.[^89][^90][^91][^29][^30][^92][^28]

- A study of nocturnal bird migration over the North Sea used bird radar mounted on an offshore wind farm to relate migration intensity to synoptic weather; intense nights were associated with clear, high‑pressure conditions and tailwinds (spring) or sidewinds (autumn).[^90][^29][^28]
- “Bird migration flight altitudes studied by a network of operational weather radars” developed an automated method to extract bird density, speed and direction as a function of altitude from C‑band weather radars; validation with a dedicated bird radar in the Netherlands, Belgium and France showed close agreement in altitude profiles.[^91][^92]
- A geostatistical model interpolated nocturnal bird densities from 69 European weather radars at 15‑min, 0.2° resolution, estimating that up to ~120 million birds can be simultaneously in flight over the network and providing uncertainty maps.[^89]

For Eemshaven / Groningen, these works imply:

- You can obtain regional altitude‑resolved bird densities and speed distributions for February, May and December from European weather‑radar datasets (e.g. ENRAM/OPERA derivatives) and use them to define prior distributions over altitude bands, airspeeds and migration intensity per month.


### 6.2. Dedicated North Sea / Dutch offshore bird radars and models

Rijkswaterstaat and partners operate specialized bird radars and models along the Dutch coast and offshore wind farms, producing directly relevant data.[^93][^94][^95][^96][^97][^98]

- The Dutch “birdradar” network (horizontal and vertical Robin Radar systems) monitors birds up to 6 km and 1.5 km altitude at several offshore and coastal platforms; data include flight direction, speed, distance and coarse size class (small/medium/large/flock), although raw data require extensive post‑processing and are not yet openly accessible.[^95]
- Validation reports for the bird migration prediction model and for specific wind farms (e.g. Luchterduinen, Borssele) compare model predictions to radar‑measured Migration Traffic Rate (MTR) and discuss seasonal patterns of migration intensity; they note peaks in spring and autumn, with low intensities in mid‑summer and varied in winter.[^96][^98][^93]
- A review of tracking data for collision‑risk modeling compiles flight speed and height information for key species (including gulls, waders and geese) in the southern North Sea.[^97]

These can be converted into priors such as:

- Monthly prior over “bird present vs no bird” per altitude and time‑of‑night.
- Species‑group‑level priors over altitude ranges (e.g. cormorants and large gulls mostly within a few hundred meters, many nocturnal migrants higher).[^92][^99][^91]


### 6.3. GPS tracking data for Groningen / Wadden / North Sea birds (Movebank \& UvA-BiTS)

The Movebank Data Repository and UvA‑BiTS host many GPS tracking datasets for North Sea coastal species including gulls, cormorants, waders and geese, some specifically in Groningen and adjacent regions.[^100][^101][^102][^103][^104][^105]

- INBO and NIOO‑KNAW have published several GPS studies via Movebank and GBIF, including Western Marsh Harriers in Groningen, herring gulls and lesser black‑backed gulls along the southern North Sea coast, and Eurasian oystercatchers breeding on Dutch islands and mainland.[^100]
- Black‑headed Gulls from a colony in the Dutch Wadden Sea were tracked with GPS, revealing migration routes of 130–560 km and evidence of nocturnal sea crossings; their wintering grounds included the Netherlands, Belgium, UK and France, with habitat use across agriculture, inland waters, urban and maritime wetlands.[^102][^104]
- A PhD thesis on geese colonizing new land in the Netherlands analyzes GPS‑burst altitudes and suggests that geese generally do not fly at very high altitudes over land, providing empirical altitude distributions for geese in the region.[^103]

These datasets allow you to:

- Derive species‑specific priors for ground speed, climb/descent rate, altitude distribution and diurnal timing for focal species (e.g. cormorants, waders, gulls, geese) in the exact geographical context.
- Map these priors from GPS coordinate space to your radar coordinate and altitude grid, using representative track segments near Eemshaven.


### 6.4. Flight-speed and wingbeat trait databases

Two trait databases are especially relevant for kinematic priors:

- **Eoldist**: a web application and database of flight speeds for 168 Western Palearctic bird species, compiled from literature and unpublished GPS tracking; it was built to compute required detection distances for automatic detection systems at wind farms and provides species‑specific distributions of flight speed.[^106][^107]
- **AVONET**: a global trait database with 11 continuous morphological traits (beak, wing, tail, tarsus, body size) and six ecological variables for all extant bird species, based on ~90,000 museum specimens; it is linked to range maps, IUCN data and eBird.[^108]

Using AVONET plus Pennycuick–Bruderer formulas, you can derive:

- Expected wingbeat-frequency ranges for absent species in Bruderer’s table, given mass and wing area.
- Expected minimum‑power and maximum‑range airspeeds for each potential species in the competition, informing prior distributions over airspeed and climb rates.


### 6.5. eBird monthly abundance for Groningen

eBird’s Status \& Trends products provide weekly abundance estimates for >2,000 species worldwide, including heatmaps and animations for the Netherlands and Groningen region.[^109][^110][^111][^112]

- eBird’s abundance maps quantify relative abundance (expected individuals per 1 km traveling count) for each species at weekly resolution, distinguishing breeding, non‑breeding, migration and year‑round areas.[^111][^112]
- Analytical guidelines show how to adjust for spatial and effort biases when using eBird for species‑distribution modeling and emphasize the need to model detection and observer effects.[^113][^110]

For ecological priors:

- Extract, for each species in your label set, the February, May and December abundance surfaces around Eemshaven / Groningen, integrate over your radar footprint, and normalize to produce a species‑prior vector per month.
- Combine these with trait‑based (AVONET) and flight‑speed‑based (Eoldist) priors for a “physics + ecology” prior over species and flight modes.

***

## 7. External Radar Datasets for Pre-training and Representation Learning

### 7.1. BirdScan MR1 Community Reference Dataset

The BirdScan Community Reference Dataset is a labeled dataset of vertical‑looking BirdScan MR1 radar echoes with associated feature tables and raw signatures.[^114][^115][^116][^117][^118]

- Haest et al. compiled thousands of labeled echo samples from BirdScan MR1 campaigns into a reference dataset stored on Zenodo (DOI: 10.5281/zenodo.5734961), with labels and radar‑derived features stored in CSV/RDS, plus raw signature files.[^118][^114]
- The accompanying MR1 ML Tool documentation describes the feature set, classification and wingbeat‑frequency estimation algorithms currently used in SBRS software.[^114]
- The R package `birdscanR` provides tools to extract BirdScan MR1 SQL databases and compute Migration Traffic Rate and density metrics per height bin and time interval, based on classification trained on this dataset.[^115][^116]

This dataset can be used for:

- Pre‑training models to distinguish biological vs non‑biological echoes and to learn low‑level radar representation of birds vs insects, which can then be adapted to your competition data via UDA.


### 7.2. European weather-radar biological datasets

In 2025, a large dataset of biological data derived from European weather radar was released, building on the OPERA network.[^119]

- Desmet et al. publish two datasets of biological signals extracted from European C‑band weather radars, providing bird and insect metrics over large spatiotemporal scales and demonstrating their usefulness for biodiversity monitoring.[^119]

Although not at the single‑track level, these datasets:

- Provide distributions of bird speeds, directions and densities as a function of altitude, time and location across Europe, useful for data‑driven priors and for pre‑training models to distinguish biological from meteorological signals.


### 7.3. OSTI / A2e avian radar track datasets

The US Department of Energy’s A2e Data Archive hosts multiple DeTect avian radar datasets with processed track data, accessible via OSTI DOIs.[^120][^121]

- “Avian Radar / Processed Data” and “WFIP3 / DeTect Avian Radar / Processed Track Data” contain radar track data of avian targets from DeTect’s 7360 S‑band radar during barge deployments (June–September 2024 and related campaigns), managed by PNNL and accessible via DOIs 10.21947/2476343 and 10.21947/2349403.[^121][^120]

These datasets:

- Provide real avian radar tracks (positions, RCS, sometimes classification labels) that are suitable for self‑supervised or contrastive pre‑training of track encoders, which can then be fine‑tuned on your competition data.


### 7.4. Other relevant radar archives

- The TAASRAD19 radar scans dataset, while precipitation‑focused, demonstrates best practices for releasing large radar image sequences (240 km radius, 5‑min sampling, labeled precipitation types), and could be re‑used for representation learning of radar geometry and clutter.[^122]
- Swiss Birdradar Solution’s private reference repository (not open) is built from BirdScan MR1 campaigns and is conceptually similar to the BirdScan Community dataset.[^117]

Given your focus, the most directly usable external datasets for pre‑training are:

- BirdScan Community Reference Dataset (vertical‑looking, labeled echoes).[^115][^114]
- OSTI/PNNL DeTect avian radar processed track data (horizontal scanning tracks).[^120][^121]

***

## 8. Advanced Modeling Approaches: Conformal Prediction, TabPFN Adaptation, Causal/Invariance

### 8.1. Conformal prediction under distribution shift

Conformal prediction provides prediction sets with guaranteed coverage under exchangeability; recent work extends this to covariate shift and more general distribution changes.[^123][^124][^125][^126][^127][^128][^129]

- Tibshirani et al. show that standard split conformal prediction can fail under covariate shift, with coverage significantly below nominal levels; they propose weighted conformal prediction that uses likelihood ratios between test and training covariate distributions to restore coverage, supported by theory and simulations.[^128][^129]
- Subsequent work develops training‑conditional coverage bounds under covariate shift and provides a weighted DKW inequality to analyze coverage of split, full and jackknife+ conformal methods; full/jackknife+ require strong stability assumptions, but split conformal with weighting is nearly assumption‑free.[^123]
- Jonkers et al. extend *Conformal Predictive Systems* (which output predictive distributions rather than sets) to covariate shift via Weighted CPS (WCPS), again using likelihood ratios, and show probabilistic calibration under shift.[^127][^130][^131]
- Doubly robust methods combine covariate‑shift modeling with conformal based on quantile regression or auxiliary models.[^124][^125][^132]

For your competition:

- Instead of using ad‑hoc uncertainty gating, you can train any base classifier, then apply weighted split conformal (classification‑set version) where weights approximate the ratio of test‑month to train‑month density over features, guaranteeing marginal coverage per test month under covariate shift assumptions.
- You can also use WCPS to obtain calibrated predictive distributions that maintain coverage guarantees under covariate shift, integrating nicely with calibrated probabilities and label‑shift correction.


### 8.2. TabPFN and drift-resilient extensions

TabPFN is a prior‑data fitted transformer that approximates Bayesian inference on small tabular datasets; recent extensions handle temporal distribution shifts explicitly.[^69][^133][^134][^66][^68]

- Drift‑Resilient TabPFN (NeurIPS 2024) models temporal distribution shifts using structural causal models (SCMs) whose mechanisms change over time, and trains a transformer to perform in‑context learning over synthetic datasets drawn from this prior; it then predicts on new time steps in a single forward pass.[^66][^68][^69]
- The method improves accuracy and ROC‑AUC under temporal domain generalization benchmarks relative to XGBoost, CatBoost, original TabPFN and other baselines, while remaining calibration‑friendly and hyperparameter‑free.[^68][^69][^66]
- Follow‑up analyses examine TabPFN v2’s performance and limitations in open environments, suggesting it is best suited to small‑scale tabular problems with moderate covariate shift rather than arbitrarily open‑world OOD data.[^133][^134]

For your use case (TabPFN already strong in‑distribution):

- You can mimic the drift‑resilient approach by generating synthetic training tasks reflecting plausible month‑to‑month shifts (e.g. using SCMs where only *priors* over species and some weather‑linked nuisance features change), fine‑tuning TabPFN on such tasks to learn invariance to those shifts.
- Alternatively, use your competition data segmented by months as multiple “tasks” in TabPFN’s meta‑training, encouraging the model to treat month as a nuisance variable while focusing on kinematic/aerodynamic features.


### 8.3. Causal and invariant multi-environment learning

Invariant Causal Prediction (ICP) and related methods attempt to learn causal predictors that are stable across environments, which is conceptually aligned with your month‑invariance goal.[^67][^66]

- Recent theoretical work on mining invariance from nonlinear multi‑environment binary classification identifies a form of invariance that exists only in the binary case and provides sufficient conditions for learning invariant predictors that remain robust under environment changes.[^67]
- Structural‑causal‑model priors used in Drift‑Resilient TabPFN provide a concrete way to simulate environments where mechanisms change (e.g. seasonal effects) while causal relations remain, and train models to focus on invariant mechanisms.[^66][^68]

To apply this thinking to bird‑radar data:

- Treat months (or month×year) as environments and explicitly test which feature–label relationships are stable across them (e.g. via environment‑wise regression and testing invariance of coefficients).
- Penalize models that rely heavily on features whose predictive power changes with month (e.g. raw altitude bands tightly tied to known seasonal flight‑altitude shifts), steering representation learning toward aerodynamics and kinematics that are expected to be causal for species identity.

***

**Note:** The references above are intentionally diverse and not tied to a single modeling paradigm, to avoid bias. For each of your eight topics, there are multiple independent sources supporting the key ideas, and many of the cited datasets (BirdScan, OSTI/PNNL avian radar, Movebank GPS, European weather‑radar biological products, eBird/AVONET/Eoldist) can be directly integrated into your next iteration pipeline for feature design, priors, pre‑training and evaluation.[^135][^34][^49][^50][^78][^28][^91][^95][^106][^108][^121][^119][^89][^100][^114][^1][^14][^66]
<span style="display:none">[^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^180][^181][^182][^183][^184][^185][^186][^187][^188][^189][^190][^191][^192][^193][^194][^195][^196][^197][^198][^199][^200][^201][^202][^203][^204][^205][^206][^207][^208][^209][^210][^211][^212][^213][^214][^215][^216][^217][^218][^219][^220][^221][^222][^223][^224][^225][^226][^227][^228][^229][^230][^231][^232][^233][^234][^235][^236][^237][^238][^239][^240][^241][^242][^243][^244][^245][^246][^247][^248][^249][^250][^251][^252][^253][^254][^255][^256][^257][^258][^259][^260][^261][^262][^263][^264][^265][^266][^267][^268][^269][^270][^271][^272][^273][^274][^275][^276][^277][^278][^279][^280][^281][^282][^283][^284][^285][^286]</span>

<div align="center">⁂</div>

[^1]: https://ieeexplore.ieee.org/document/9626001/

[^2]: https://repository.tudelft.nl/record/uuid:96cc142c-ddf6-4823-baf5-1a6a27b97b51

[^3]: https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/b7890e6b-99cf-4480-80b9-5b495e0d1863/content

[^4]: https://ieeexplore.ieee.org/document/11046034/

[^5]: https://oa.upm.es/14657/1/JULIAN_DAVID_COLORADO_MONTANO_A.pdf

[^6]: https://lucris.lub.lu.se/ws/files/5782430/796334.pdf

[^7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5522512/

[^8]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0050309

[^9]: https://theses.liacs.nl/pdf/2020-2021-VinkT.pdf

[^10]: https://pubmed.ncbi.nlm.nih.gov/19880724/

[^11]: https://agris.fao.org/search/en/providers/122535/records/65df74ccb766d82b1801c1f0

[^12]: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1474-919X.2010.01014.x

[^13]: https://pubmed.ncbi.nlm.nih.gov/9319516/

[^14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1914071/

[^15]: https://dsp-book.narod.ru/RSAD/C1828_PDF_C02.pdf

[^16]: https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.04025

[^17]: https://skynet.ee.ic.ac.uk/notes/Radar_4_RCS.pdf

[^18]: https://www.nature.com/articles/srep35637

[^19]: https://www.radartutorial.eu/01.basics/Fluctuation Loss.en.html

[^20]: https://www.mathworks.com/help/phased/ug/swerling-2-target-models.html

[^21]: https://www.mathworks.com/help/radar/ug/modeling-target-radar-cross-section.html

[^22]: https://www.science.org/doi/10.1126/sciadv.1700097

[^23]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/2041-210X.14082

[^24]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/ece3.2585

[^25]: https://nora.nerc.ac.uk/id/eprint/513117/1/N513117JA.pdf

[^26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4337751/

[^27]: https://onlinelibrary.wiley.com/doi/10.1002/ece3.9146

[^28]: https://www.frontiersin.org/article/10.3389/fevo.2020.542438/full

[^29]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/jav.02562

[^30]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.04003

[^31]: https://royalsocietypublishing.org/doi/10.1098/rsif.2013.0419

[^32]: https://royalsocietypublishing.org/doi/pdf/10.1098/rstb.2015.0398

[^33]: https://arc.aiaa.org/doi/10.2514/1.J059438

[^34]: https://pure.uva.nl/ws/files/194481518/Methods_Ecol_Evol_-_2023_-_Erp_-_A_framework_for_post_processing_bird_tracks_from_automated_tracking_radar_systems.pdf

[^35]: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14249

[^36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3431116/

[^37]: https://www.lunduniversity.lu.se/lup/publication/d14c6264-9def-4162-9550-062e54b8274c

[^38]: https://www.bohrium.com/paper-details/do-seabirds-control-wind-drift-during-their-migration-across-the-strait-of-gibraltar-a-study-using-remote-tracking-by-radar/817370648635506689-3949

[^39]: https://hkr.diva-portal.org/smash/record.jsf?pid=diva2%3A956748

[^40]: https://www.mathworks.com/help/phased/ug/swerling-3-target-models.html

[^41]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3774623/

[^42]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11439958/

[^43]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.04041

[^44]: https://www.sciencedirect.com/science/article/abs/pii/S2214166917300024

[^45]: https://arxiv.org/abs/2309.15415

[^46]: http://arxiv.org/pdf/2309.15415.pdf

[^47]: https://ieeexplore.ieee.org/document/10746385/

[^48]: https://ieeexplore.ieee.org/document/10654246/

[^49]: https://arxiv.org/pdf/2203.08321.pdf

[^50]: https://arxiv.org/abs/2312.09857

[^51]: https://web3.arxiv.org/pdf/2312.09857

[^52]: https://openreview.net/pdf?id=hyuacPZQFb0

[^53]: https://openreview.net/forum?id=xsts7MRLey

[^54]: http://web.cs.wpi.edu/~gsarkozy/Cikkek/88.pdf

[^55]: https://arxiv.org/pdf/1612.01939.pdf

[^56]: https://www.emergentmind.com/topics/correlation-alignment-coral

[^57]: https://openreview.net/forum?id=0dualJz9OI

[^58]: https://www.emergentmind.com/topics/domain-adversarial-neural-networks-dann

[^59]: https://www.arxiv.org/abs/2512.17831

[^60]: https://www.emergentmind.com/topics/domain-adversarial-neural-network-dann

[^61]: https://ui.adsabs.harvard.edu/abs/2024AppOR.14904066X/abstract

[^62]: https://jmlr.org/papers/volume17/15-239/15-239.pdf

[^63]: https://iopscience.iop.org/article/10.1088/1361-6501/adc4fd

[^64]: https://arxiv.org/pdf/2404.11269.pdf

[^65]: https://arxiv.org/pdf/2212.01555.pdf

[^66]: https://proceedings.neurips.cc/paper_files/paper/2024/file/b2e2774c8e76afe191b5bf518f5cb727-Paper-Conference.pdf

[^67]: https://arxiv.org/html/2404.15245v2

[^68]: https://neurips.cc/virtual/2024/103122

[^69]: https://openreview.net/forum?id=p3tSEFMwpG

[^70]: http://www.scholarpedia.org/article/Product_of_experts

[^71]: https://arxiv.org/pdf/2106.01770.pdf

[^72]: https://arxiv.org/pdf/2112.10327.pdf

[^73]: http://arxiv.org/pdf/2006.13092.pdf

[^74]: https://arxiv.org/pdf/2402.07821.pdf

[^75]: https://proceedings.mlr.press/v152/johansson21a.html

[^76]: http://www.arxiv.org/abs/1910.12656

[^77]: https://arxiv.org/html/2602.18573v1

[^78]: https://arxiv.org/pdf/1802.03916.pdf

[^79]: https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf

[^80]: https://arxiv.org/abs/1802.03916

[^81]: https://www.emergentmind.com/topics/label-shift

[^82]: https://papers.nips.cc/paper_files/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Review.html

[^83]: http://arxiv.org/pdf/2302.09159.pdf

[^84]: https://arxiv.org/html/2505.16251v1

[^85]: https://arxiv.org/html/2503.02506v1

[^86]: https://nadavb.com/Label-Shift-and-Domain-Adaptation-in-Machine-Learning/

[^87]: https://coralogix.com/ai-blog/ultimate-guide-to-pr-auc-calculations-uses-and-limitations/

[^88]: https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc

[^89]: https://www.mdpi.com/2072-4292/11/19/2233/pdf?version=1569756227

[^90]: https://www.frontiersin.org/articles/10.3389/fevo.2020.542438/pdf

[^91]: http://rsif.royalsocietypublishing.org/content/8/54/30.full.pdf

[^92]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3024816/

[^93]: https://www.noordzeeloket.nl/publish/pages/220058/validation-of-a-bird-radar-system.pdf

[^94]: https://www.teamepoch.ai/wp-content/uploads/2026/02/Radar-and-Bird-Migration-Bart-Kranstauber.pdf

[^95]: https://www.digitalnorthsea.nl/data/ecological-data/the-birdradar

[^96]: https://noordzeeloket.nl/publish/pages/236249/validation-of-the-outcomes-of-the-bird-migration-prediction-model-for-2023.pdf

[^97]: https://www.noordzeeloket.nl/publish/pages/187415/review_and_analysis_of_tracking_data_to_delineate_flight_characteristics_and_migration_routes_of_bir.pdf

[^98]: https://noordzeeloket.nl/publish/pages/239599/bird-research-in-offshore-wind-farm-borssele.pdf

[^99]: https://www.bsg-ecology.com/wp-content/uploads/2015/03/Egmond_aan_Zee.pdf

[^100]: https://purews.inbo.be/ws/portalfiles/portal/71446339/DMP_MOVE2GBIF.pdf

[^101]: https://datarepository.movebank.org

[^102]: https://research.rug.nl/en/publications/evidence-of-nocturnal-migration-over-sea-and-sex-specific-migrati

[^103]: https://pure.uva.nl/ws/files/135837298/Thesis_complete_.pdf

[^104]: https://research.rug.nl/files/630847745/arde.v110i1.a8.pdf

[^105]: https://www.movebank.org

[^106]: https://pure.knaw.nl/portal/en/publications/eoldist-a-web-application-for-estimating-cautionary-detection-dis/

[^107]: https://onlinelibrary.wiley.com/doi/full/10.1002/we.2971

[^108]: https://opentraits.org/datasets/avonet.html

[^109]: http://arxiv.org/pdf/2407.02690.pdf

[^110]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ddi.13271

[^111]: https://science.ebird.org/en/status-and-trends/abundance-animations

[^112]: https://science.ebird.org/en/status-and-trends/abundance-maps

[^113]: https://cloudfront.escholarship.org/dist/prd/content/qt01t5c00w/qt01t5c00w.pdf?t=pg55g7

[^114]: https://zenodo.org/records/5734961

[^115]: https://cran.r-project.org/web/packages/birdscanR/refman/birdscanR.html

[^116]: https://cran.r-project.org/web/packages/birdscanR/index.html

[^117]: https://swiss-birdradar.com/reference-data-repository/

[^118]: https://cris.haifa.ac.il/en/publications/birdscan-community-reference-dataset/

[^119]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11871220/

[^120]: https://www.osti.gov/biblio/2476343

[^121]: https://www.osti.gov/biblio/2349403

[^122]: https://gotriple.eu/fr/documents/ftzenodo:oai:zenodo.org:3591396

[^123]: https://arxiv.org/pdf/2405.16594.pdf

[^124]: https://arxiv.org/pdf/2203.01761.pdf

[^125]: https://arxiv.org/html/2402.13042v1

[^126]: https://arxiv.org/pdf/2401.17452.pdf

[^127]: https://proceedings.mlr.press/v230/jonkers24a.html

[^128]: https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal_ds.pdf

[^129]: https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal_ds.pdf

[^130]: https://arxiv.org/abs/2404.15018

[^131]: https://cml.rhul.ac.uk/copa2024/presentations/COPA2024_Conformal_Predictive_Systems_under_Covariate_Shift_Jef_Jonkers_et_al.pdf

[^132]: http://arxiv.org/pdf/2502.13030.pdf

[^133]: https://openreview.net/forum?id=lQYNmlTwtc

[^134]: https://arxiv.org/html/2505.16226v1

[^135]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11742/2587214/Improved-bird-micro-doppler-simulation-for-bird-versus-UAV-recognition/10.1117/12.2587214.full

[^136]: https://ieeexplore.ieee.org/document/11347682/

[^137]: https://ieeexplore.ieee.org/document/11383721/

[^138]: https://ieeexplore.ieee.org/document/10659036/

[^139]: https://ieeexplore.ieee.org/document/11170505/

[^140]: https://iopscience.iop.org/article/10.1088/1361-6501/ada059

[^141]: https://arxiv.org/abs/2502.13440

[^142]: https://ieeexplore.ieee.org/document/10218182/

[^143]: https://ieeexplore.ieee.org/document/10436801/

[^144]: https://arxiv.org/pdf/2112.09042.pdf

[^145]: https://www.mdpi.com/2072-4292/14/9/2196/pdf?version=1651911126

[^146]: https://www.mdpi.com/2076-3417/8/11/2089/pdf?version=1540803795

[^147]: https://www.mdpi.com/2673-4931/8/1/48/pdf?version=1637665678

[^148]: https://royalsocietypublishing.org/doi/10.1098/rstb.2023.0113

[^149]: https://www.mdpi.com/2624-6120/4/2/18

[^150]: https://cran.r-project.org/web/packages/monitoR/vignettes/monitoR_QuickStart.pdf

[^151]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4506967

[^152]: https://publications.tno.nl/publication/105220/tJKET4/molchanov-2013-classification.pdf

[^153]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2607429/

[^154]: https://en.wikipedia.org/wiki/Radar_cross_section

[^155]: https://publications.sto.nato.int/publications/STO Meeting Proceedings/STO-MP-MSG-SET-183/MP-MSG-SET-183-15.pdf

[^156]: https://docs.ogc.org/bp/16-004r5.html

[^157]: https://ieeexplore.ieee.org/document/9440989/

[^158]: https://ieeexplore.ieee.org/document/10928980/

[^159]: https://www.semanticscholar.org/paper/6e6a8efaae6bf2f13db9e0bea446b4d396ddc081

[^160]: https://tuprints.ulb.tu-darmstadt.de/id/eprint/11498

[^161]: https://www.semanticscholar.org/paper/60a6fbf69da1816e333cb91ce9d01a7a2199476a

[^162]: http://link.springer.com/10.1007/3-540-45723-2

[^163]: https://www.semanticscholar.org/paper/03e0310b6bb652490bc7471516da460ef8af83fc

[^164]: https://www.semanticscholar.org/paper/1b8f371244e5146dc3e8d6340ddf623a01e175b7

[^165]: https://www.semanticscholar.org/paper/785c392b87e8f2f3b3048b077b04294d7e43ff6d

[^166]: https://www.semanticscholar.org/paper/0d32ba920d9bedad64c4ccd2cf614c64398ab8ac

[^167]: https://www.mdpi.com/2072-4292/13/4/662/pdf?version=1613987880

[^168]: https://www.mdpi.com/2079-9292/13/8/1412/pdf?version=1712654064

[^169]: http://arxiv.org/pdf/2405.12038.pdf

[^170]: https://www.mdpi.com/2072-4292/11/8/926/pdf?version=1555423961

[^171]: https://www.mdpi.com/1424-8220/24/1/263/pdf?version=1704185243

[^172]: https://www.mdpi.com/1424-8220/22/7/2664/pdf

[^173]: https://www.mdpi.com/2073-8994/14/3/570/pdf?version=1647238518

[^174]: https://freidok.uni-freiburg.de/fedora/objects/freidok:471/datastreams/FILE1/content

[^175]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900112/

[^176]: https://pdfs.semanticscholar.org/1220/3af003f06b8698f3c0c802d6a6abd3a41121.pdf

[^177]: https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.14249

[^178]: https://resolver.tudelft.nl/uuid:12a7b370-267b-482b-b4c7-4fce22e138b1

[^179]: https://journalhosting.ucalgary.ca/index.php/arctic/article/view/63997

[^180]: https://linkinghub.elsevier.com/retrieve/pii/S0022519302930948

[^181]: https://nsojournals.onlinelibrary.wiley.com/doi/10.1034/j.1600-048X.2000.310213.x

[^182]: https://linkinghub.elsevier.com/retrieve/pii/S0003347298908831

[^183]: https://www.semanticscholar.org/paper/7f3cf48b14c7791d708805c534c59328829273f0

[^184]: https://linkinghub.elsevier.com/retrieve/pii/030096299090674H

[^185]: https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2013.0419

[^186]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3730693/

[^187]: https://www.mdpi.com/2075-4450/14/2/112/pdf?version=1674305843

[^188]: https://is-birdcast-wordpress-prod-s3.s3.amazonaws.com/wp-content/uploads/DietterichThomasElectricalEngineeringComputerScienceReconstructingVelocitiesMigrating.pdf

[^189]: https://www.sciencedaily.com/releases/2016/02/160216181101.htm

[^190]: https://royalsocietypublishing.org/rsif/article/10/86/20130419/35124/Air-speeds-of-migrating-birds-observed-by

[^191]: https://noordzeeloket.nl/publish/pages/226850/validation-of-the-bird-migration-prediction-model.pdf

[^192]: https://researchportal.hkr.se/sv/publications/orientation-of-shorebirds-in-relation-to-wind-both-drift-and-comp-2/

[^193]: https://royalsocietypublishing.org/rsos/article/9/11/211364/96584/Observations-and-models-of-across-wind-flight

[^194]: https://linkinghub.elsevier.com/retrieve/pii/S0002916523272949

[^195]: https://www.mdpi.com/2072-4292/14/3/508/pdf?version=1642764556

[^196]: https://academic.oup.com/condor/advance-article-pdf/doi/10.1093/ornithapp/duae062/60304578/duae062.pdf

[^197]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4034227/

[^198]: https://energy.sustainability-directory.com/learn/can-radar-distinguish-between-a-single-large-bird-and-a-dense-flock-of-small-birds/

[^199]: https://ams.confex.com/ams/36Radar/webprogram/Manuscript/Paper228799/Biological_scatterers_DYNAMAO_AMS_radar.pdf

[^200]: https://ams.confex.com/ams/pdfpapers/47065.pdf

[^201]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2020.0064

[^202]: https://dsp-book.narod.ru/Farina/11185_15.pdf

[^203]: https://eprints.whiterose.ac.uk/id/eprint/233803/1/Ecosphere - 2025 - Matthews - Taxonomic resolution in dual‐polarization weather radar observations of biological scatterers.pdf

[^204]: https://repository.library.noaa.gov/view/noaa/34122/noaa_34122_DS1.pdf

[^205]: https://open.metu.edu.tr/bitstream/handle/11511/15334/index.pdf

[^206]: https://www.mdpi.com/2072-4292/17/22/3762

[^207]: https://linkinghub.elsevier.com/retrieve/pii/S0924271625001224

[^208]: https://link.springer.com/10.1007/s10618-025-01108-4

[^209]: http://www.dbpia.co.kr/Journal/ArticleDetail/NODE11654384

[^210]: https://arxiv.org/abs/2505.09955

[^211]: https://arxiv.org/abs/2508.18630

[^212]: https://arxiv.org/abs/2507.20968

[^213]: https://arxiv.org/html/2409.12169v1

[^214]: https://arxiv.org/html/2312.09857v2

[^215]: https://arxiv.org/pdf/2411.17869.pdf

[^216]: http://arxiv.org/pdf/2302.03133v2.pdf

[^217]: https://arxiv.org/html/2410.06671v1

[^218]: https://arxiv.org/html/2312.09857v3

[^219]: https://link.aps.org/doi/10.1103/PhysRevA.102.032410

[^220]: http://arxiv.org/pdf/1910.11385.pdf

[^221]: http://arxiv.org/pdf/2411.02988.pdf

[^222]: http://arxiv.org/pdf/2210.16955.pdf

[^223]: http://arxiv.org/pdf/2210.03702.pdf

[^224]: https://arxiv.org/pdf/2003.06820.pdf

[^225]: https://www.sciencedirect.com/science/article/abs/pii/S0034425717303140

[^226]: https://publica.fraunhofer.de/bitstreams/cb7981b5-4284-472e-918e-29e6664c5bb5/download

[^227]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900160/table/sensors-26-01056-t011/

[^228]: https://stackoverflow.com/questions/60110209/multiclass-classification-probabilities-and-calibration

[^229]: https://arxiv.org/pdf/2505.15888.pdf

[^230]: https://www.evidentlyai.com/classification-metrics/multi-class-metrics

[^231]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9470580/

[^232]: https://arxiv.org/html/2506.14540v2

[^233]: https://arxiv.org/pdf/2011.06169.pdf

[^234]: http://arxiv.org/pdf/2410.12457.pdf

[^235]: https://arxiv.org/pdf/2303.15041.pdf

[^236]: http://arxiv.org/pdf/2309.13692.pdf

[^237]: http://arxiv.org/pdf/2411.07523.pdf

[^238]: https://arxiv.org/pdf/2501.13697.pdf

[^239]: https://proceedings.neurips.cc/paper_files/paper/2022/file/8d5f526a31d3731a30eb58d5874cf5b1-Paper-Conference.pdf

[^240]: https://insightful-data-lab.com/2025/08/17/precision-recall-auc-pr-auc/

[^241]: https://www.jmlr.org/papers/volume23/21-1519/21-1519.pdf

[^242]: https://www.linkedin.com/pulse/probabilistic-evaluation-metrics-classification-problems-van-mil-x5bse

[^243]: https://www.deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/

[^244]: https://www.semanticscholar.org/paper/Detecting-and-Correcting-for-Label-Shift-with-Black-Lipton-Wang/80ef8b8a1284790e0d8f7cbf9727c9e0b2a89332

[^245]: https://arxiv.org/html/2508.17780v1

[^246]: https://www.semanticscholar.org/paper/531eb7ff73a33bc24ce7acb30387c846af09e00d

[^247]: https://onlinelibrary.wiley.com/doi/10.1002/rra.4206

[^248]: http://link.springer.com/10.1007/s00027-007-0952-8

[^249]: http://link.springer.com/10.1007/s10336-019-01704-4

[^250]: https://www.cambridge.org/core/product/identifier/S095927092510021X/type/journal_article

[^251]: https://www.nature.com/articles/s41598-019-54228-5

[^252]: https://onlinelibrary.wiley.com/doi/10.1002/ece3.70815

[^253]: https://onlinelibrary.wiley.com/doi/10.1111/j.1474-919X.1960.tb05091.x

[^254]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/geb.13742

[^255]: https://www.mdpi.com/2072-4292/12/4/635/pdf

[^256]: https://pure.uva.nl/ws/files/9701124/Seasonal_detours_by_soaring_migrants_shaped_by_wind_regimes_along_the_East_Atlantic_Flyway.pdf

[^257]: https://bioone.org/journals/ardea/volume-55/issue-1–2/arde.v78.p339/Moult-Mass-and-Flight-Range-of-Waders-Ready-to-Take/10.5253/arde.v78.p339.pdf

[^258]: https://www.cebc.cnrs.fr/wp-content/uploads/publipdf/2021/JEC44_2021.pdf

[^259]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3169047/

[^260]: https://www.pnas.org/content/pnas/118/21/e2023170118.full.pdf

[^261]: http://www.ace-eco.org/vol12/iss2/art12/ACE-ECO-2017-1104.pdf

[^262]: https://www.ace-eco.org/vol18/iss1/art4/ACE-ECO-2022-2357.pdf

[^263]: https://www.birds.cornell.edu/landtrust/ebird-abundance-maps/

[^264]: https://www.flightdata.com

[^265]: https://eightflight.com

[^266]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3383431/

[^267]: https://ebird.github.io/ebird-best-practices/abundance.html

[^268]: https://www.oag.com

[^269]: https://en.wikipedia.org/wiki/EBird

[^270]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/2041-210X.14249

[^271]: https://arxiv.org/pdf/2306.16019.pdf

[^272]: https://www.frontiersin.org/articles/10.3389/fmars.2024.1235061/pdf?isPublishedV2=False

[^273]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4161323/

[^274]: https://pnas.org/doi/10.1073/pnas.2315933121

[^275]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8316499/

[^276]: https://www.data.gouv.fr/datasets/radars-automatiques/

[^277]: https://zenodo.org/records/12750580

[^278]: https://obis.org/dataset/ab84bb8e-3a0f-43f6-83be-27a1d8965998

[^279]: https://zenodo.org/records/13932869

[^280]: https://trepo.tuni.fi/bitstream/handle/10024/123975/978-952-03-1776-8.pdf?isAllowed=y\&sequence=2

[^281]: https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/zeller2024tro.pdf

[^282]: https://arxiv.org/html/2404.15018v1

[^283]: http://arxiv.org/pdf/2502.16513.pdf

[^284]: https://arxiv.org/html/2406.05405v3

[^285]: https://arxiv.org/html/2602.07019v1

[^286]: https://www.climatechange.ai/papers

