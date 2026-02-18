# Deep Research Plan

File reset check.
# Deep Research Plan: External Data to Improve OOD Radar Bird Classification

This document translates newly collected external datasets into an executable plan on top of the current temporal-safe baseline.

---

## 1) Dataset Acquisition Status (2026-02-18)

Local manifest: `data/other_datasets/dataset_manifest.json`

### Already available (local)

- `data/other_datasets/Col de la Croix 1988.csv` (Zenodo `10.5281/zenodo.10209093`)
- `data/other_datasets/5845259/data_SAAB_SIRS_77GHz_FMCW.npy` (Zenodo `10.5281/zenodo.5845259`)
- `data/other_datasets/figshare_16586228_avonet/` (AVONET core files)
- `data/other_datasets/figshare_23537892_birdwingdata/` (full BirdWingData package incl. `Alerstam_2017.csv`, `Bruderer_2010.csv`)
- Metadata captures:
  - `data/other_datasets/dryad_3n5tb2rd2_target_tracking/metadata.json`
  - `data/other_datasets/lat_birddrone/metadata.json`
  - `data/other_datasets/zenodo_5734961_birdscan_reference/record.json`

### Access constrained

- BirdScan Community Reference Dataset (`10.5281/zenodo.5734961`) is `restricted` on Zenodo.
- LAT-BirdDrone public page exists (`https://www.scidb.cn/anonymous/ZkFWRlJ2`) but does not expose a stable non-interactive direct file endpoint.
- Dryad large archive endpoint currently returns anti-bot `403` from scripted download path.

---

## 2) Core Principle for This Competition

The main failure mode is month/domain shift, not in-domain fit.  
Therefore, external data must be used to improve:

1. **Season-shift robustness** (LOMO/generalization),
2. **Minority class ranking** (macro-AP),
3. **Physics consistency** (class-conditional plausibility of speed/size/wingbeat proxies).

Anything that only increases in-fold SKF but hurts LOMO is rejected.

---

## 3) Mathematical Integration Strategy

Let:

- `x` = track features,
- `m` = month,
- `c` = class in `{0..8}`,
- `f_j(x)` = model `j` score/probability for class `c`.

### 3.1 Final score decomposition

Use class-wise fused logits:

`z_c(x,m) = sum_j w_{c,j} * logit(clip(f_{j,c}(x), eps, 1-eps)) + beta_c * log(pi_ext(c|m) / pi_train(c))`

`p_c = sigmoid(z_c)` (macro-AP optimization is per-class ranking; no forced row renormalization required during tuning).

Where:

- `w_{c,j}` are per-class blend weights (fit on OOF),
- `pi_ext(c|m)` is external month prior (GBIF + external ecology),
- `beta_c` is class-specific prior strength.

### 3.2 External feature families to add

From AVONET + BirdWingData:

- class prototype stats: `mass`, `wingspan`, `wing_area`, `aspect_ratio`, `wing_loading`.
- observed-to-expected mismatch features:
  - `z_speed_c = (v_obs - mu_speed_c) / sigma_speed_c`
  - `size_speed_residual = v_obs - g(rcs_mean, radar_bird_size)`
  - `flight_efficiency_gap = observed_glide_proxy - expected_glide_c`

From Alerstam + Bruderer tables:

- class expected speed and wingbeat priors:
  - `delta_speed_to_class_prior`
  - `delta_rcs_periodicity_to_wbf_prior` (proxy only; no direct high-rate wingbeat assumption).

From Col de la Croix:

- auxiliary radar-track transfer branch:
  - learn coarse movement embedding,
  - distill into current model via pseudo-targets and OOF-safe blending.

From 77GHz FMCW:

- use only as **regularization/representation pretraining** for clutter-vs-bio separation patterns (domain-shift aware, low blend weight).

---

## 4) Experiment Ladder (Strictly LOMO-Gated)

All experiments:

- remove `ALL_TEMPORAL`,
- evaluate `SKF + LOMO`,
- report per-class AP,
- keep only if LOMO improves beyond noise threshold.

### E48: Reproduce best temporal-safe anchor

- Baseline: E38/E42 family (temporal-safe).
- Goal: exact reproducibility and stable seed variance.
- Keep artifacts for fair A/B comparisons.

### E49: Morphology prior features (AVONET + BirdWingData)

- Add class-prototype mismatch features only.
- No architecture change.
- Hypothesis: improves Ducks/Geese/Pigeons/Cormorants separation.

### E50: Flight-parameter priors (Alerstam + Bruderer)

- Add speed and wingbeat-prior mismatch features.
- Focus on Songbirds/Waders/BoP ambiguity reduction.

### E51: External transfer from Col de la Croix

- Train auxiliary model on Col de la Croix.
- Map to competition coarse groups.
- Blend as weak specialist branch (`<=15%` class-wise unless validated).

### E52: Clutter specialist with 77GHz regularization

- Binary specialist: `Clutter vs non-Clutter`.
- Distill or blend this specialist only into clutter logit.

### E53: Per-class robust blending + external priors

- Optimize `w_{c,j}` and `beta_c` using nested OOF protocol.
- Apply logit-level class-wise prior shift.

### E54: Constrained pseudo-labeling (optional)

- Only if E53 is stable.
- Per-class/month quota constraints to avoid majority collapse.

### E55: Final robust ensemble

- Multi-seed of top 2-3 robust configs.
- Selection criterion: maximize `mean(LOMO) - lambda * std(LOMO)`.

---

## 5) Acceptance Rules

Reject any change if:

- `LOMO` drops,
- minority AP gains come with severe collateral collapse,
- gain is within noise band and unstable across seeds.

Prefer improvements with:

- positive LOMO delta,
- better minority AP tails,
- realistic test distribution (no pathological class starvation).

---

## 6) Implementation Notes

1. Build one canonical external-priors table with columns:
   `class`, `mu_speed`, `sigma_speed`, `mu_wbf`, `mass`, `wingspan`, `wing_area`, `aspect_ratio`, `wing_loading`.
2. Add a feature builder module that consumes this table and current track features.
3. Keep all blending and prior-shift tuning inside OOF-only pipelines (no leakage).
4. Track each run in `EXPERIMENTS.md` with SKF/LOMO and per-class APs.

---

## 7) Immediate Next Step

Start with **E49 (morphology priors)** because it is low-risk, directly available from downloaded data, and aligns with known class confusion structure.
# Deep Research Plan: External Data to Improve OOD Radar Bird Classification

This document translates newly collected external datasets into an executable plan on top of the current temporal-safe baseline.

---

## 1) Dataset Acquisition Status (2026-02-18)

Local manifest: `data/other_datasets/dataset_manifest.json`

### Already available (local)

- `data/other_datasets/Col de la Croix 1988.csv` (Zenodo `10.5281/zenodo.10209093`)
- `data/other_datasets/5845259/data_SAAB_SIRS_77GHz_FMCW.npy` (Zenodo `10.5281/zenodo.5845259`)
- `data/other_datasets/figshare_16586228_avonet/` (AVONET core files)
- `data/other_datasets/figshare_23537892_birdwingdata/` (full BirdWingData package incl. `Alerstam_2017.csv`, `Bruderer_2010.csv`)
- Metadata captures:
  - `data/other_datasets/dryad_3n5tb2rd2_target_tracking/metadata.json`
  - `data/other_datasets/lat_birddrone/metadata.json`
  - `data/other_datasets/zenodo_5734961_birdscan_reference/record.json`

### Access constrained

- BirdScan Community Reference Dataset (`10.5281/zenodo.5734961`) is `restricted` on Zenodo.
- LAT-BirdDrone public page exists (`https://www.scidb.cn/anonymous/ZkFWRlJ2`) but does not expose a stable non-interactive direct file endpoint.
- Dryad large archive endpoint currently returns anti-bot `403` from scripted download path.

---

## 2) Core Principle for This Competition

The main failure mode is month/domain shift, not in-domain fit.  
Therefore, external data must be used to improve:

1. **Season-shift robustness** (LOMO/generalization),
2. **Minority class ranking** (macro-AP),
3. **Physics consistency** (class-conditional plausibility of speed/size/wingbeat proxies).

Anything that only increases in-fold SKF but hurts LOMO is rejected.

---

## 3) Mathematical Integration Strategy

Let:

- `x` = track features,
- `m` = month,
- `c` = class in `{0..8}`,
- `f_j(x)` = model `j` score/probability for class `c`.

### 3.1 Final score decomposition

Use class-wise fused logits:

`z_c(x,m) = sum_j w_{c,j} * logit(clip(f_{j,c}(x), eps, 1-eps)) + beta_c * log(pi_ext(c|m) / pi_train(c))`

`p_c = sigmoid(z_c)` (macro-AP optimization is per-class ranking; no forced row renormalization required during tuning).

Where:

- `w_{c,j}` are per-class blend weights (fit on OOF),
- `pi_ext(c|m)` is external month prior (GBIF + external ecology),
- `beta_c` is class-specific prior strength.

### 3.2 External feature families to add

From AVONET + BirdWingData:

- class prototype stats: `mass`, `wingspan`, `wing_area`, `aspect_ratio`, `wing_loading`.
- observed-to-expected mismatch features:
  - `z_speed_c = (v_obs - mu_speed_c) / sigma_speed_c`
  - `size_speed_residual = v_obs - g(rcs_mean, radar_bird_size)`
  - `flight_efficiency_gap = observed_glide_proxy - expected_glide_c`

From Alerstam + Bruderer tables:

- class expected speed and wingbeat priors:
  - `delta_speed_to_class_prior`
  - `delta_rcs_periodicity_to_wbf_prior` (proxy only; no direct high-rate wingbeat assumption).

From Col de la Croix:

- auxiliary radar-track transfer branch:
  - learn coarse movement embedding,
  - distill into current model via pseudo-targets and OOF-safe blending.

From 77GHz FMCW:

- use only as **regularization/representation pretraining** for clutter-vs-bio separation patterns (domain-shift aware, low blend weight).

---

## 4) Experiment Ladder (Strictly LOMO-Gated)

All experiments:

- remove `ALL_TEMPORAL`,
- evaluate `SKF + LOMO`,
- report per-class AP,
- keep only if LOMO improves beyond noise threshold.

### E48: Reproduce best temporal-safe anchor

- Baseline: E38/E42 family (temporal-safe).
- Goal: exact reproducibility and stable seed variance.
- Keep artifacts for fair A/B comparisons.

### E49: Morphology prior features (AVONET + BirdWingData)

- Add class-prototype mismatch features only.
- No architecture change.
- Hypothesis: improves Ducks/Geese/Pigeons/Cormorants separation.

### E50: Flight-parameter priors (Alerstam + Bruderer)

- Add speed and wingbeat-prior mismatch features.
- Focus on Songbirds/Waders/BoP ambiguity reduction.

### E51: External transfer from Col de la Croix

- Train auxiliary model on Col de la Croix.
- Map to competition coarse groups.
- Blend as weak specialist branch (`<=15%` class-wise unless validated).

### E52: Clutter specialist with 77GHz regularization

- Binary specialist: `Clutter vs non-Clutter`.
- Distill or blend this specialist only into clutter logit.

### E53: Per-class robust blending + external priors

- Optimize `w_{c,j}` and `beta_c` using nested OOF protocol.
- Apply logit-level class-wise prior shift.

### E54: Constrained pseudo-labeling (optional)

- Only if E53 is stable.
- Per-class/month quota constraints to avoid majority collapse.

### E55: Final robust ensemble

- Multi-seed of top 2-3 robust configs.
- Selection criterion: maximize `mean(LOMO) - lambda * std(LOMO)`.

---

## 5) Acceptance Rules

Reject any change if:

- `LOMO` drops,
- minority AP gains come with severe collateral collapse,
- gain is within noise band and unstable across seeds.

Prefer improvements with:

- positive LOMO delta,
- better minority AP tails,
- realistic test distribution (no pathological class starvation).

---

## 6) Implementation Notes

1. Build one canonical external-priors table with columns:
   `class`, `mu_speed`, `sigma_speed`, `mu_wbf`, `mass`, `wingspan`, `wing_area`, `aspect_ratio`, `wing_loading`.
2. Add a feature builder module that consumes this table and current track features.
3. Keep all blending and prior-shift tuning inside OOF-only pipelines (no leakage).
4. Track each run in `EXPERIMENTS.md` with SKF/LOMO and per-class APs.

---

## 7) Immediate Next Step

Start with **E49 (morphology priors)** because it is low-risk, directly available from downloaded data, and aligns with known class confusion structure.
Advanced Methodologies and Extrinsic Datasets for Radar-Based Avian Trajectory ClassificationIntroduction and Contextual BackgroundThe intersection of artificial intelligence, aeroecology, and renewable energy infrastructure represents a critical frontier in modern conservation and industrial optimization. The global transition toward clean energy has precipitated the rapid expansion of offshore and nearshore windfarms. However, facilities such as the Eemshaven windfarm, located in the coastal region of Groningen in the Netherlands, introduce substantial collision risks for diverse avian populations traversing these ecologically sensitive corridors. To mitigate this environmental impact while maintaining the operational and economic efficiency of the wind turbines, continuous, highly accurate, and real-time classification of biological targets within the airspace is strictly required. The implementation of targeted mitigation strategies, particularly shutdown-on-demand (SDOD) protocols, relies entirely on the precision of the underlying detection systems to halt rotors temporarily only when high-impact species or dense migratory flocks enter the immediate rotor-swept zone.The AI Cup 2026 Performance Track, hosted by Team Epoch—a competitive machine learning Dreamteam from the Technical University of Delft—and partnered with the Netherlands Organisation for Applied Scientific Research (TNO) and the AI Coalition for the Netherlands (AIC4NL), isolates this exact analytical challenge. The objective is to develop a robust machine learning model capable of classifying bird species based entirely on radar track data gathered at the Eemshaven windfarm. The target taxonomy encompasses nine distinct categories: Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, and Songbirds.The evaluation metric for this competition is the Mean Average Precision (mAP), macro-averaged across all nine classes. The final score represents the arithmetic mean of the Average Precision (AP) calculated independently for each of the nine columns representing the predicted probability of the classes. This metric introduces a profound mathematical challenge: macro-averaging ensures that the classification performance on minority classes, such as the ecologically sensitive Birds of Prey, carries the exact same statistical weight as the performance on hyper-abundant coastal species like Gulls. Consequently, models that simply predict the majority class will fail to achieve competitive scores. Submissions require the output of a single file mapping a unique track_id to nine float probabilities bounded between 0.0 and 1.0.Solving this multivariate time-series classification problem requires moving far beyond the provided baseline training datasets. It necessitates the integration of advanced deep learning architectures capable of modeling complex spatio-temporal kinematics, the application of signal processing techniques to extract micro-Doppler signatures, and the strategic ingestion of extrinsic datasets for pre-training, domain adaptation, and synthetic data generation. This comprehensive report provides an exhaustive analysis of the biological radar signatures of the target classes, evaluates state-of-the-art trajectory classification models, and identifies the most valuable external datasets and repositories available to maximize the Mean Average Precision metric.Radar Aeroecology and the Physics of DetectionTo develop a robust predictive algorithm for trajectory classification, it is imperative to first understand the physical constraints, capabilities, and underlying physics of the sensing hardware generating the tracking sequences. The reference data from the Eemshaven site is deeply intertwined with the operational mechanics of advanced 3D avian radar systems, specifically those developed by Robin Radar Systems, which are extensively deployed by TNO for ecological monitoring and collision risk modeling.Hardware Parameters and Signal AcquisitionModern avian radars, such as the Robin MAX system, operate on Frequency Modulated Continuous Wave (FMCW) technology, typically utilizing the X-band or S-band frequency spectrum. These systems are designed to provide full three-dimensional, 360-degree spatial awareness continuously, regardless of diurnal cycles or adverse weather conditions. A standard X-band avian radar configuration might utilize a power output of 44 Watts and a rapid rotation speed of 60 revolutions per minute, which guarantees an extremely high-frequency track update rate of one second per target.This continuous tracking generates highly granular time-series data encompassing spatial coordinates, target velocity, directional heading, and estimated altitude. The hardware pipelines tracking algorithms—such as Global Nearest Neighbor (GNN), Joint Probabilistic Data Association (JPDA), or modern neural-network-based multiple object tracking (MOT) systems like DeepSORT—to associate sequential radar echoes into a coherent, continuous flight path assigned a unique track_id.The preliminary classification and filtering of targets by the raw radar hardware is predominantly deterministic, relying heavily on the Radar Cross Section (RCS). The RCS is a fundamental electromagnetic measurement of a target's detectability; it represents the theoretical area intercepting the radar power that, if radiated isotropically in all directions, would produce the same received power at the radar antenna. In the context of biological targets, the RCS is highly dynamic. It fluctuates rapidly due to the changing physical conformation of the bird's body and wings during the flapping cycle, but its mean and maximum values serve as primary proxies for avian biomass and physical dimensions.The standard operational thresholds for target categorization based on RCS in highly calibrated FMCW avian radar systems are empirically defined to partition the airspace. Table 1 outlines the general detection boundaries and RCS categorizations typical of these systems.Target ClassificationMaximum Detection RangeAltitude LimitApparent RCS ThresholdLarge Targets10.0 km700 m-13 dBm²Medium Targets8.0 km600 m-16 dBm²Small Targets4.0 km400 m-25 dBm²Micro Targets3.3 km300 m-30 dBm²Table 1: Operating thresholds, altitude limits, and RCS classifications characteristic of advanced FMCW avian radar systems.While the RCS provides a highly effective initial heuristic for filtering out massive anomalies or micro-insects, it is fundamentally insufficient for fine-grained, species-level classification. Two completely different species of identical mass—for example, a large Herring Gull and a similarly sized Bird of Prey—will exhibit nearly identical mean RCS values. Furthermore, the RCS of a single bird depends heavily on its orientation relative to the radar beam, known as the aspect angle. A bird flying tangentially to the radar returns a vastly different scattering profile than one flying directly radially toward the receiver.Recent advancements in electromagnetic scattering modeling, such as the T-matrix method, demonstrate that anatomically representative models of large birds produce highly complex RCS signatures that spherical simplifications fail to capture. Therefore, advanced machine learning models must look beyond the static mean RCS and extract dynamic kinematic features, spatial geometries, and temporal micro-Doppler signatures directly from the trajectory sequences.Micro-Doppler Signatures and Wingbeat KinematicsThe most discriminative physical feature embedded within high-resolution radar tracks is the micro-Doppler signature generated by the periodic amplitude modulation of the radar echo. As an avian target traverses the airspace, the continuous forward motion of its central body mass provides a base Doppler shift that translates to the radial velocity. Simultaneously, the cyclical, highly energetic movement of its wings introduces secondary frequency modulations superimposed on the main signal.By applying advanced signal processing techniques, such as the Continuous Wavelet Transform (CWT) or the Short-Time Fourier Transform (STFT), to the raw track data or the high-resolution Automatic Gain Control (AGC) signal, it is possible to extract the fundamental wingbeat frequency of the target. The CWT is particularly effective in this domain because it provides a robust, multi-resolution time-frequency representation, accommodating the inherently non-stationary nature of bird flight where periods of active flapping may be seamlessly interspersed with bounding, gliding, or soaring maneuvers.Empirical studies utilizing tracking radars in coastal European regions have established firm baseline wingbeat frequencies that classification algorithms must learn to implicitly isolate. For instance, coastal observations indicate that Cormorants maintain a highly steady, uninterrupted wingbeat frequency of approximately 4.4 Hz. Conversely, Gulls exhibit a slower, more varied flapping frequency typically ranging between 3.0 and 3.1 Hz. At the extreme end of the spectrum, smaller, highly agile birds operate at rapid frequencies approaching or exceeding 8.0 Hz. Machine learning models that can implicitly learn these oscillatory factors from the sequential variance in track speed, acceleration, or altitude will achieve vastly superior Average Precision metrics.Kinematic and Morphological Profiling of Target ClassesThe nine target classes defined in the AI Cup 2026 Performance Track demand rigorous biological and aeroecological profiling. A successful predictive model will not merely identify statistical correlations; it will implicitly map the geometric trajectory data to the aerodynamic capabilities, energetic constraints, and behavioral ecology of these specific biological groups. Understanding the scaling laws of flight speed relative to body mass and wing morphology is essential for feature engineering.Coastal Transients: Gulls and CormorantsGulls (Laridae) represent the overwhelmingly dominant coastal species in the Eemshaven and broader North Sea regions. Historical radar studies conducted in the Dutch coastal area during the summer months demonstrate that gulls can comprise up to 89% of all identified biological tracks. They are morphologically adapted for versatile flight, characterized by moderate cruising airspeeds ranging between 8 and 15 m/s. Gulls frequently engage in highly localized, non-migratory movements, commuting between marine foraging sites and terrestrial roosts at extremely low altitudes. Empirical data indicates that approximately 55% of tracked gulls fly below 25 meters, and 75% remain below 50 meters, placing them in direct conflict with the lower sweep of offshore wind turbine blades. Their flight paths are highly tortuous and erratic when actively foraging, but transition to highly linear trajectories when commuting long distances.Cormorants (Phalacrocoracidae), while possessing a biomass comparable to large gulls, display markedly different kinematics and aerodynamic profiles. They fly at intermediate speeds averaging roughly 15 m/s. Unlike gulls, which frequently utilize thermal soaring and dynamic gliding to conserve energy, cormorants possess a significantly lower wing aspect ratio and higher wing loading. Consequently, they rely on continuous, heavy, and energetically demanding flapping to remain airborne, resulting in a highly stable, continuous altitude profile and the aforementioned signature 4.4 Hz wingbeat frequency. Radar data from regional windfarms such as Horns Rev indicate that cormorants often migrate at higher altitudes but display strong resident and staging behaviors in coastal waters, leading to dense, localized tracking events.Heavy Waterfowl: Ducks and GeeseDucks and Geese (Anseriformes) constitute the extreme high-velocity spectrum of the biological dataset. Equivalent airspeeds for these groups frequently fall strictly within the 15 to 20 m/s range, with specific species of diving ducks capable of reaching extraordinary continuous cruising speeds of 20 to 23 m/s. Aeroecological analyses of flight speeds among bird species reveal that swans, geese, and ducks exhibit a remarkable negative scaling exponent relative to body mass, driving them toward these high-velocity flight envelopes.Geese possess massive biological biomass, returning the largest RCS signatures among the target classes, and are frequently tracked at much higher altitudes during major migratory events. Their trajectory sequences are distinctly marked by extreme spatial linearity and minimal deviation. Because their high wing loading requires steady, energy-efficient straight-line flight to maintain aerodynamic lift, they do not engage in erratic maneuvers unless actively avoiding a collision. When ducks and geese appear on a radar scope, they typically present as large, high-speed, highly uniform tracks that completely lack the sharp turning angles or vertical agility seen in smaller species. They also frequently travel in distinct V-formations or large flocks, creating composite radar echoes that algorithms must disambiguate.Terrestrial and Aerial Specialists: Birds of Prey and PigeonsBirds of Prey (raptors), including various species of falcons, hawks, and eagles, rely heavily on thermal soaring and orographic lift to minimize the metabolic cost of flight. Their radar tracks are highly distinctive and geometrically complex, characterized by relatively slow forward airspeeds (typically 8 to 15 m/s) coupled with continuous circular, spiraling flight paths as they gain altitude within rising columns of warm air (thermals). This spiraling ascent is usually followed by a long, straight, downward-angled glide toward the next thermal. An algorithm computing the curvature, heading variance, and vertical ascent rate of a track sequence will easily isolate raptor trajectories from the linear, powered flight of waterfowl.Pigeons (Columbiformes), conversely, are powerful, direct flyers adapted for rapid evasion and sustained cruising. They achieve high flight speeds of 15 to 20 m/s and possess a highly direct flight pattern consisting of steady, unwavering flight with rapid, uninterrupted wing beats. Their trajectories will show high velocities statistically similar to ducks, but they will present with substantially lower RCS values due to their smaller physical size. Furthermore, studies utilizing high-precision GPS and accelerometer bio-loggers have shown that pigeons flying in pairs or flocks will actively increase their wingbeat frequency by up to 18% to improve aerodynamic stability and navigational accuracy, introducing measurable variance in their micro-Doppler signatures depending on their flocking state.The Micro-Targets: Waders and SongbirdsWaders (shorebirds) and Songbirds (passerines) represent the smallest biological targets in the dataset, frequently brushing the absolute -25 to -30 dBm² RCS detection limits of the radar hardware. Songbirds generally exhibit slower cruising speeds (8 to 15 m/s) and predominantly employ a highly distinct bounding flight pattern—a brief, intense burst of flapping followed by a ballistic, wings-folded glide that minimizes drag. This bounding behavior results in a micro-oscillating altitude profile and a highly periodic, high-amplitude fluctuation in RCS that a sequential neural network can detect and map to the Songbird class.Waders are generally faster than songbirds, achieving speeds of 15 to 20 m/s, and engage in rapid, tightly coordinated flocking maneuvers. These flocks act as a single super-organism, producing collective radar echoes that fluctuate violently in intensity as the flock banks in unison, exposing entirely different aspect angles and cross-sectional areas to the radar beam.Clutter and Non-Biological InterferenceThe "Clutter" class serves as a vital catch-all aggregation of non-avian phenomena, encompassing ground reflections, sea wave backscatter, heavy weather artifacts such as rain bands, and anthropogenic objects like unmanned aerial vehicles (UAVs/drones) or light aircraft. Drones overlap significantly with birds in terms of operating altitude, general velocity, and mean RCS, representing a highly complex, well-documented edge case in airspace surveillance known universally as the "bird-drone problem".However, drone trajectories fundamentally lack the micro-Doppler wingbeat signatures inherent to avian flight. Furthermore, mechanical UAVs exhibit unnatural kinematic smoothness, instantaneous turning behaviors that defy biological inertia, and rigid velocity maintenance that diverge sharply from the organically variable kinematics of a living organism combating variable wind currents. Accurate classification of the Clutter class is essential, as false positives will trigger unnecessary and costly wind turbine shutdowns.Avian ClassExpected Mean VelocityFlight Style / Kinematic SignatureRelative RCS ClassCormorants~15 m/sContinuous, heavy flapping, highly linearLargeGulls8 - 15 m/sFlapping, thermal soaring, highly variableMedium / LargeDucks15 - 23 m/sHigh-speed, highly linear, low altitudeMediumGeese15 - 20 m/sHigh-speed, highly linear, high altitudeLargePigeons15 - 20 m/sRapid uninterrupted wingbeat, directSmall / MediumBirds of Prey8 - 15 m/sThermal soaring, spiraling, glidingMedium / LargeWaders15 - 20 m/sDense flocking, rapid collective maneuveringSmallSongbirds8 - 15 m/sBounding flight, oscillating altitudeMicro / SmallTable 2: Kinematic and radar scattering profiles of the 8 target avian classes based on historical aeroecology tracking studies.State-of-the-Art Modeling Architectures for Trajectory ClassificationGiven that the submission format for the AI Cup strictly requires outputting predicted probabilities corresponding to a specific sequential track_id, the core computational challenge is multi-class multivariate time-series classification. The traditional paradigm of utilizing flattened feature vectors fed into shallow machine learning algorithms—such as Support Vector Machines (SVM) or Random Forests (RF)—while capable of achieving respectable >90% accuracy on highly curated, localized datasets, ultimately fails to capture the deep temporal dependencies inherent in complex, noisy trajectories. Consequently, modern solutions for radar track disambiguation have shifted entirely toward deep sequence models.Bi-Directional Long Short-Term Memory (Bi-LSTM) NetworksTrajectory data, when encoded as a discrete time-series of image-plane coordinates, range, azimuth, apparent size, and instantaneous velocity, is highly suitable for Recurrent Neural Networks (RNNs), specifically Bi-Directional Long Short-Term Memory (Bi-LSTM) networks. A standard RNN processes data sequentially, which can lead to vanishing gradients over long bird tracks. An LSTM solves this via its complex architecture of input, output, and forget gates, which regulate the flow of information and maintain long-term context.A Bi-LSTM enhances this by processing the radar track sequence both forward and backward in time. This allows the network to understand a given spatial coordinate or velocity drop within the context of the entire past and future flight path. In empirical studies specifically addressing the classification of drones versus birds using radar trajectories, Bi-LSTMs have demonstrated a profound ability to learn the characteristic differences in track smoothness, turning behavior, and localized velocity fluctuations without relying on appearance-based visual features. For the AI Cup 2026 dataset, a Bi-LSTM can implicitly learn the rhythmic bounding flight oscillations of a Songbird or the steady, powerful acceleration of a departing Cormorant, effectively mapping these temporal patterns directly to the predicted class probabilities.Transformer Models and Self-Attention MechanismsWhile LSTMs excel at strictly sequential data, they still struggle with extremely long tracks due to inherent information bottlenecking at the final hidden state. Transformer-based architectures, originally developed for complex natural language processing tasks, have recently been adapted and have achieved state-of-the-art results in physical trajectory prediction, spatial-temporal classification, and maritime tracking.Transformers discard recurrence entirely, utilizing multi-head self-attention mechanisms to weigh the importance of every spatial point in a radar track against every other point, regardless of their temporal distance from one another. This architectural paradigm is highly advantageous for classifying Birds of Prey, where the defining feature—a massive circular thermal climb spanning thousands of meters—might unfold over several minutes of radar tracking. A Transformer can attend to the entire macro-structure of the circle simultaneously, recognizing the global geometric shape rather than just the point-to-point sequence.The Hybrid LSTM-Transformer ArchitectureThe absolute frontier in radar-based avian and UAV classification is the hybrid LSTM-Transformer network. This architecture leverages the complementary strengths of both paradigms to effectively distinguish between highly similar tracks, drastically reducing false positive rates when separating mechanical drones from biological birds, or fast-flying waders from ducks.In this advanced pipeline, the pre-processed radar track sequence is first fed through an LSTM layer (or occasionally a 1D Convolutional Neural Network). The LSTM extracts local, short-term dynamic characteristics—such as the micro-fluctuations in speed and altitude caused by individual wingbeats. These locally enriched feature embeddings are then passed as sequential tokens into a Transformer encoder block. The Transformer applies multi-head self-attention to capture the global spatial-sensitive information, evaluating the macro-trajectory, including the overall compass heading, long-term altitude changes, and geographic waypoints relative to the wind turbine locations.A global average pooling layer compresses this deep spatial-temporal representation, which is finally passed through a fully connected multi-layer perceptron (MLP) with a softmax activation function to output the 9 specific class probabilities required by the Kaggle metric. Empirical tests utilizing real collected radar track data show that this hybrid approach substantially outperforms standalone LSTM or RNN models across multiple metrics, including overall accuracy, the Matthews Correlation Coefficient (MCC), and class recall.2D CNN Spectrogram Conversion and Transfer LearningAn alternative deep learning approach, which is highly relevant if raw micro-Doppler profiles, high-resolution AGC signals, or structural point-cloud data are accessible alongside the localized tracks, involves converting the 1D time-series data into 2D pseudo-images. By generating spectrograms (time-frequency plots) via Short-Time Fourier Transforms of the track's velocity or RCS variance over time, the sequential data is mapped cleanly into a visual domain.Once in the visual domain, standard, highly optimized Convolutional Neural Networks (CNNs)—such as ResNet, VGGish, or EfficientNet architectures—can be applied to these spectrograms to perform image-based classification. In complex multi-label bird classification tasks, pre-trained CNNs fused with secondary LSTM modules have shown extraordinary, state-of-the-art performance in isolating specific target classes amidst heavy acoustic or radar background noise. This method essentially treats the radar tracking problem as a computer vision problem, unlocking the vast array of pre-trained image weights available in the open-source community.Architecture TypeTemporal HandlingSpatial/Macro HandlingOptimal Target Use-CaseRandom Forest / SVMPoor (Requires manual feature engineering)PoorBaseline establishment, high-clutter filtering.Bi-LSTMExcellent (captures sequential wingbeat variance)ModerateSongbirds, Pigeons, separating smooth drones from flapping birds.TransformerModerate (computationally heavy for long sequences)Excellent (attends to global track geometry)Birds of Prey (thermal spiraling), Geese (long linear migrations).LSTM-Transformer HybridExcellentExcellentComprehensive 9-class probability generation.CNN (Spectrograms)Visualized via time-frequency plotsMinimalMicro-Doppler analysis, acoustic transfer learning.Table 3: Comparison of state-of-the-art machine learning architectures applicable to the multivariate radar trajectory classification task.Exhaustive Analysis of Extrinsic DatasetsThe pursuit of a robust, winning mAP score in a competitive machine learning environment invariably requires looking beyond the provided baseline training data to prevent overfitting, improve the model's generalization capabilities, and handle the inherent class imbalances present in the real world. Radar track data is notoriously difficult to annotate accurately, leading to extreme disparities in data volume. For example, the Eemshaven coastal site will natively produce hundreds of thousands of mundane Gull tracks, but potentially only a handful of perfectly annotated Bird of Prey or distinct Songbird tracks. To rectify this mathematical imbalance, the ingestion of external datasets for pre-training, transfer learning, and synthetic data augmentation is strictly necessary. The following data repositories and models represent the most critical resources currently available for this specific aeroecological problem space.1. The LAT-BirdDrone DatasetThe LAT-BirdDrone (Low-Altitude Target) dataset is a dedicated, open-source academic repository designed explicitly for the high-precision classification of low-altitude small target trajectories enhanced by hybrid neural networks. Unlike standard computer vision datasets (e.g., CUB-200-2011 or Birdsnap) that focus entirely on high-resolution visual pixel data to differentiate plumage or beak shape, LAT-BirdDrone is fundamentally concerned with the temporal trajectory, providing continuous bounding box sequences and track IDs over time.This dataset fills a massive scarcity in trajectory classification resources for micro-targets. By pre-training the aforementioned hybrid LSTM-Transformer model on LAT-BirdDrone, the network learns the foundational physics of biological versus mechanical flight. The dataset contains hundreds of track sequences specifically contrasting varied bird flight against multicopter and fixed-wing drone flight. Transferring these learned weights to the AI Cup dataset will profoundly enhance the model's ability to isolate the competition's "Clutter" class, ensuring that UAVs, localized weather anomalies, and wind turbine blade reflections are mathematically segregated and not falsely predicted as avian species.2. OSTI Avian Radar Processed Data (A2e Archive)Hosted by the U.S. Department of Energy's Atmosphere to Electrons (A2e) Data Archive, the "Avian Radar / Processed Data" repository (OSTI ID: 2476343) contains highly detailed radar track data of various avian targets. Published recently in late 2024 by researchers affiliated with the Pacific Northwest National Laboratory (PNNL) and DeTect Inc., this data was collected using advanced 7360 s-band radar systems during extensive offshore and large barge deployments spanning June through September 2024.Because this dataset natively consists of continuous radar track data derived directly from marine and coastal environments, its statistical distribution of background noise, sea clutter, and wind-blown interference perfectly mirrors the harsh operating conditions at the coastal Eemshaven windfarm. Utilizing this vast dataset allows for the implementation of self-supervised contrastive learning. The machine learning model can be pre-trained on millions of unlabelled OSTI tracks to construct a highly accurate latent space of generic bird movement. Once this latent space is established, the model can be fine-tuned on the labeled 9-class Eemshaven dataset, drastically reducing the volume of labeled data required for convergence and preventing overfitting on the local geography.3. Dryad: Avian Radar Target-Tracking Performance DataAuthored by S. Urmy and J. Warren and hosted on the Dryad Data Repository, this massive 5.17 GB dataset contains X-band marine radar tracks collected at a major breeding colony of common and roseate terns on Great Gull Island, New York. Crucially, rather than just providing passive observations, this dataset actively investigates radar tracking probabilities in environments with highly variable background clutter and includes echoes from simulated bird tracks mathematically overlaid onto real radar scans.This dataset provides an exact, peer-reviewed blueprint for synthetic data generation and data augmentation. Because rare biological classes (such as Waders or Birds of Prey) may be severely underrepresented in the AI Cup dataset, competitors can use the sophisticated methodologies outlined in the Dryad repository to synthetically generate mathematically sound bird trajectories and inject them into the training data. Furthermore, the dataset demonstrates exactly how automated tracking algorithms behave at specific distances (ranging from optimal performance at 0.5 km out to degraded performance at 3.0 km). This allows modelers to accurately model signal degradation over distance and apply corresponding noise profiles to the training batches, thereby making the final Transformer model invariant to range-based signal attenuation.4. ENRAM and Continental Weather Radar ArchivesThe European Network for the Radar Surveillance of Animal Movement (ENRAM) is an extensive collaborative initiative that utilizes a massive network of continental weather surveillance radars (WSR) to extract biological targets from meteorological data. The ENRAM repositories, which encompass data from 141 radar stations across 18 countries spanning from 2008 to 2023, include highly detailed case studies on intense bird migration across the Netherlands and Belgium. These datasets provide comprehensive profiles of bird migration altitudes, reflectivity-ppi (Plan Position Indicator), and projected forward trajectories.While weather radar operates at a vastly different spatial scale and lower resolution than the specialized high-resolution Robin MAX 3D radar, the ENRAM datasets offer a powerful macro-level prior that can be deeply integrated into the classification logic. Radar track classification does not exist in a vacuum; the probability of a given track belonging to a Songbird versus a Pigeon is heavily dependent on the time of day, the season, and the regional migratory flow. By merging the temporal metadata (the exact timestamp of the track_id) with historical ENRAM migration flow data, models can utilize Bayesian priors. For example, high-altitude nocturnal tracks exhibiting mass directional flow over the North Sea in autumn have a statistically overwhelming probability of belonging to migrating waterfowl (Geese/Ducks) or songbirds, rather than diurnal Raptors or localized Gulls.5. Movebank GPS TrajectoriesMovebank is a massive, free online database of animal tracking data hosted and maintained by the Max Planck Institute of Animal Behavior. It contains over 46 million high-precision locations of thousands of individual animals, with deep, specialized repositories specifically tracking migratory birds such as storks, waterfowl, raptors, and gulls across their annual life cycles.While Movebank data consists of GPS telemetry rather than radar echoes, the underlying kinematic truths—including maximum velocity, cruising heading, track tortuosity, and terrain avoidance behaviors—are fundamentally identical. The strategic approach here involves cross-domain translation: transforming high-resolution GPS tracks into synthetic radar tracks. By computing the 3D derivatives (velocity and acceleration) of Movebank GPS tracks for specific target classes (e.g., Geese, Ducks, Birds of Prey), and subsequently injecting Gaussian noise to simulate the radar hardware's discrete sampling rates, modelers can generate an infinite supply of class-perfect, biologically accurate training data. This cross-domain translation is highly effective for training the spatial attention heads of a Transformer model, ensuring it understands the absolute biomechanical limits of specific species.6. Bioacoustic Datasets (BirdCLEF / xeno-canto)While the AI Cup Kaggle competition relies exclusively on radar tracking rather than audio recordings, the massive datasets generated by global audio classification challenges—such as the annual BirdCLEF competition and the xeno-canto audio archive—have spurred the development of highly optimized, domain-specific deep learning architectures for avian discrimination. Over 85 GB of audio data encompassing thousands of species have been used to train Audio Spectrogram Transformers (AST) and specialized Convolutional Neural Networks.Because radar micro-Doppler signatures can be converted via Short-Time Fourier Transforms into spectrograms that visually resemble audio frequency plots, pre-trained weights from BirdCLEF winning models serve as vastly superior starting points for fine-tuning compared to generic computer vision weights trained on ImageNet. Transfer learning from audio-based bird classification models directly into radar-based micro-Doppler classification is a highly proven, sophisticated strategy for accelerating model convergence and improving classification accuracy on difficult tracks.Dataset / RepositoryPrimary Data ModalityStrategic Application for the Classification ProblemSource Identity / AuthorityLAT-BirdDroneTrajectory Bounding BoxesClutter isolation, drone vs. bird discrimination, LSTM pre-training.Kang et al. OSTI Avian RadarS-band Radar TracksDomain adaptation, self-supervised pre-training on marine radar noise.USDOE A2e / PNNL Dryad Target-TrackingX-band Radar / SimulatedRange-based signal attenuation modeling, synthetic trajectory generation.Urmy & Warren ENRAM ArchivesWeather Radar ProfilingMacro-environmental Bayesian priors, seasonal migration probabilities.ENRAM / OPERA MovebankGPS TelemetryHigh-fidelity kinematic boundaries, cross-domain trajectory synthesis.Max Planck Inst. BirdCLEF / CUB-200Audio / ImageTransfer learning, Audio Spectrogram Transformer (AST) weight initialization.Cornell Lab / Kaggle Table 4: Summary and strategic application of the most critical extrinsic datasets applicable to the AI Cup 2026 radar classification task.Strategic Optimization and Implementation PathwaysTo synthesize the complex biological realities of the nine target classes with the mathematical architectures and extrinsic datasets identified above, a deliberate, multi-staged methodology is required. The following strategies are recommended for achieving superior evaluation metrics on the AI Cup 2026 Performance Track, specifically tailored to maximize the Mean Average Precision metric.1. Multi-Modal Feature EngineeringFeeding raw positional data coordinates (x, y, z) directly into a neural network is computationally inefficient and forces the model to learn basic physics from scratch. The track data must be augmented with explicitly engineered physical features calculated prior to model ingestion. From the raw time-series, algorithms must compute the first and second derivatives of position to extract instantaneous velocity and acceleration vectors.Furthermore, geometric and kinematic descriptors such as path tortuosity, range rate, turning angle variance, and oscillation factors must be explicitly defined as distinct input channels. By feeding the LSTM or Transformer encoder a multi-dimensional tensor that explicitly includes these physical properties, the network is freed from having to infer basic kinematics, allowing its millions of parameters to focus entirely on the subtle, species-level deviations that separate a fast-flying pigeon from a diving duck.2. Addressing Class Imbalance with Focal Loss and AugmentationCoastal windfarms like Eemshaven are overwhelmingly dominated by resident Gulls and localized wader movements, with massive migratory spikes of Geese and Songbirds occurring only during highly specific seasonal windows. Consequently, the training dataset will inevitably exhibit massive, natural class imbalance. Because the competition utilizes macro-averaged Mean Average Precision (mAP), poor predictive performance on a rare minority class (such as Birds of Prey) will mathematically devastate the final score, regardless of near-perfect accuracy on the Gull class.Standard Cross-Entropy loss functions are fundamentally inadequate for this evaluation metric. The neural network must be trained utilizing Focal Loss, a dynamic function that scales the loss based on prediction confidence, heavily penalizing the model for misclassifying difficult, rare examples while aggressively reducing the weight of easily classified majorities (e.g., standard linear Gulls). Furthermore, advanced oversampling techniques—specifically utilizing the synthetic trajectory generation methods derived from the Dryad  and Movebank datasets —should be employed to artificially balance the class distributions during the construction of training batches.3. Implementation of the Probabilistic PipelineThe optimal predictive architecture is a highly integrated, sequential pipeline. The pre-processed track data, enriched with kinematic derivatives and augmented with synthetic minority classes, should first pass through a stacked Bi-Directional LSTM. The Bi-LSTM outputs a dense sequence of hidden states that encapsulate the localized aerodynamic variations of the bird (e.g., the rhythmic bounding flight of a songbird versus the continuous flapping of a cormorant).These enriched hidden states are then treated as sequential tokens and passed into a Multi-Head Self-Attention Transformer Encoder. The Transformer learns the global context of the flight path—evaluating whether the track represents a straight, high-altitude commute from a breeding colony or an erratic, low-altitude foraging pattern over the water. A final Multi-Layer Perceptron (MLP) head outputs the 9 float probabilities required by the submission format. To maximize the mAP score, these raw output logits should be further calibrated using Platt Scaling or Isotonic Regression, ensuring that the predicted probabilities strictly represent true likelihoods, which heavily optimizes the Area Under the Precision-Recall Curve.4. Leveraging Macro-Priors for Final CalibrationFinally, because the objective is to classify real-world biological phenomena, the output of the deep learning model should be subjected to a final layer of environmental logic. If the track_id includes or can be mapped to temporal metadata, meteorological and geographic priors should be integrated at the end of the network. Wind speed, time of day (diurnal versus nocturnal), and current season absolutely dictate biological realities.For example, songbirds heavily migrate at night, whereas birds of prey are strictly diurnal as they require solar heating to generate the thermals necessary to soar. Injecting time and weather embeddings alongside the Transformer's output into the final classification layer provides a massive statistical advantage. This acts as a mathematical reality-check mechanism against physically improbable predictions, ensuring that the model does not predict a thermal-soaring raptor in the middle of a cold, nocturnal rainstorm.ConclusionThe AI Cup 2026 Performance Track challenges the machine learning community to untangle the complex, noisy, and highly dynamic reality of biological radar echoes. Differentiating nine distinct classes—from massive flocks of Geese traveling in highly linear, high-speed formations to individual Songbirds bounding at low altitudes—requires a fundamental understanding of aeroecology mapped directly onto advanced deep learning architectures.By moving beyond simple tabular machine learning models and adopting deep, hybrid LSTM-Transformer pipelines, developers can mathematically capture both the micro-kinematics of wingbeats and the macro-geometry of flight paths. More importantly, the strategic ingestion of extrinsic datasets—leveraging LAT-BirdDrone for strict clutter discrimination , OSTI S-band data for environmental domain adaptation , Dryad for synthetic attenuation modeling , and Movebank for high-fidelity kinematic boundaries —provides the necessary volume of highly varied data to train these complex networks without overfitting. Combining these cutting-edge models with rigorous physical feature engineering and macro-environmental priors will yield highly precise, generalizable classifications capable of significantly advancing automated wildlife protection and maintaining the operational efficiency of global wind energy facilities.