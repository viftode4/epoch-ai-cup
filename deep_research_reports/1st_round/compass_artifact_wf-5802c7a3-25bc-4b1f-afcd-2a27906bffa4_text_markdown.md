# Datasets and models for radar-based bird classification

**No pretrained model exists for this exact task, but a rich ecosystem of radar datasets, flight parameter databases, and proven ML approaches can give competitors a significant edge.** The AI Cup 2026 competition — classifying radar tracks into 9 bird classes at Eemshaven wind farm — appears genuinely novel in the ML competition landscape. However, two directly relevant radar classification datasets, several species-specific flight parameter databases, and a well-established literature on radar ornithology ML provide substantial pretraining, feature engineering, and prior-knowledge resources. The dataset was provided by TNO, which originated the ROBIN radar project in 1980 and spun out Robin Radar Systems commercially in 2010, making this a rare opportunity to work with real Dutch bird radar data.

## Two radar datasets directly match the competition's structure

The **Col de la Croix 1988 Radar Tracking Dataset** (Zenodo, DOI: 10.5281/zenodo.10209093) is the single most immediately useful external dataset. It contains individual tracks of free-flying nocturnal migrants recorded by a military tracking radar in Switzerland, stored as a single CSV with per-track tabular features: mean flight altitude, ground speed, airspeed, heading, climb rate, and flight direction. Tracks are labeled into **9 classes** based on wingbeat patterns — wader-type large/small, passerine-type large/small, swift-type, raptor, single large bird, flock, and unknown. This classification scheme maps closely to the competition's categories (waders, songbirds ≈ passerine-type, birds of prey ≈ raptor). The dataset is **freely downloadable under CC-BY 4.0** and could serve as pretraining or auxiliary data for models that operate on track-level features.

The **BirdScan Community Reference Dataset** (Zenodo, DOI: 10.5281/zenodo.5734961) is the most feature-rich radar bird classification dataset available. Collected using a BirdScan MR1 dedicated X-band bird radar, it contains **6,361 labeled radar echoes** with radar-derived features, wingbeat pattern signatures, and hierarchical biological group labels (large-bird, passerine-type, wader-type, insect, non-biological scatterer). The dataset includes both tabular features (`TrainingData.csv`) and raw echo signatures. Access is **restricted** — researchers must request permission from the Swiss Ornithological Institute (birgen.haest@vogelwarte.ch) for non-commercial use. Despite the access barrier, this dataset's feature set likely overlaps substantially with the competition data, making it worth pursuing.

A third dataset of peripheral relevance is the **77 GHz FMCW Radar Dataset** (Zenodo, DOI: 10.5281/zenodo.5845259) containing **75,868 samples** of birds, humans, and six drone types in `.npy` format. While it treats "bird" as a single undifferentiated class, the micro-Doppler classification methodology demonstrated in accompanying papers transfers directly to bird species discrimination.

## Flight parameter databases provide powerful species-specific priors

Several published databases compile species-specific flight characteristics that can serve as classification priors or engineered features, effectively encoding domain knowledge into the model.

**Alerstam et al. (2007)** published tracking-radar measurements of equivalent airspeeds for **138 bird species** during migration (PLoS Biology, open access with supplementary data). Species are grouped into six phylogenetic categories — swans/geese/ducks, waders, gulls/auks, raptors/owls, falcons/crows/songbirds, and pigeons/swifts/woodpeckers — that align with competition classes. Speed ranges vary substantially: **8–23 m/s across species**, with ducks among the fastest and raptors showing distinctive soaring patterns.

The **Eoldist database** (Fluhr et al. 2025, Wind Energy, DOI: 10.1002/we.2971) provides flight speeds for **168 Western Palearctic bird species** compiled from 25+ publications and GPS-tracking data, distinguishing local from migratory flights. Being Western Palearctic–focused, it covers the Netherlands' bird fauna precisely. The electronic supplementary material contains per-species speed distributions (mean ± SD) directly usable as Bayesian priors for radar track speed-based classification.

**Bruderer et al. (2010)** compiled wingbeat frequency data for **155 species** measured by tracking radar and cine camera (published in Ibis). Wingbeat frequency is one of the strongest radar classification features — cormorants flap at ~5–6 Hz, gulls at ~3–4 Hz, and songbirds at ~10–15 Hz. If the competition data includes any wingbeat-related features (which is likely given TNO's radar capabilities), this database provides authoritative species-level ground truth.

**AVONET** (Tobias et al. 2022, Ecology Letters) supplies morphological traits for all **11,009 extant bird species** including body mass, wingspan, wing area, and hand-wing index. Combined with the **BirdWingData** dataset (Shiomi et al. 2025, figshare, 856 species), competitors can compute wing loading and aspect ratio — the primary aerodynamic parameters governing flight speed and radar cross-section. The R package **afpt** implements Pennycuick's aerodynamic models to convert these morphological measurements into predicted flight speed ranges for any species.

## GPS tracking data covers every competition species class

Movebank (movebank.org) hosts **over 4 billion animal location records** with several datasets directly relevant to each competition class. Most valuable are the INBO/LifeWatch GPS tracking datasets, published on Zenodo with DOIs:

- **H_GRONINGEN**: Western marsh harriers (Birds of Prey) tracked in **Groningen, Netherlands** — the exact competition region
- **DELTATRACK**: Herring gulls and lesser black-backed gulls (Gulls) at Neeltje Jans, Netherlands
- **O_BALGZAND**: Eurasian oystercatchers (Waders) at Balgzand, Netherlands
- **CURLEW_VLAANDEREN**: Eurasian curlews (Waders) in Flanders, Belgium
- **LifeTrack Geese**: Greater white-fronted geese with GPS + acceleration data
- Multiple **homing pigeon** GPS tracking studies
- **Great cormorant** GPS studies (Fijn et al. 2022, Ardea)

These GPS tracks enable extraction of species-specific flight speed distributions, trajectory tortuosity, altitude profiles, turning rates, and acceleration patterns. While GPS resolution differs from radar resolution, the **relative differences between species groups** in these kinematic features are consistent across measurement methods and directly applicable as classification priors.

**eBird** provides species occurrence data for the Netherlands that can establish temporal prior probabilities — which species are present at Eemshaven during which months. **NatureScot** publishes a ready-to-use lookup table of body length, wingspan, and flight speed values for species at risk of wind turbine collision in Northern Europe.

## ML approaches that work best on radar bird data

The radar ornithology literature converges on a consistent set of findings about which models and features perform best. **Random Forest and gradient boosting (XGBoost, LightGBM) consistently outperform other methods** on tabular radar track data. Rosa et al. (2016) tested six algorithms on marine radar ornithological data and found RF achieved >0.80 AUC across all classification tasks — bird vs. non-bird, species group discrimination, and clutter filtering. Zaugg et al. (2017) used RF ensembles on BirdScan MR1 features for 6-class echo classification with strong results.

A **hierarchical/cascade classification strategy** emerges as the recommended approach from multiple papers. A 2025 study (arXiv:2602.07019) demonstrated that cascading classifiers — first separating clutter, then classifying by size (small/medium/large), then identifying species within each size class — often outperforms unified single-model classification. This maps naturally to the competition's 9 classes: clutter is separable first; then geese and cormorants (large), ducks and gulls (medium), songbirds and waders (small-medium), with birds of prey distinguished by soaring flight patterns rather than size alone.

The **most discriminative features** identified across the literature, ranked by importance, are:

- **Radar cross-section (RCS)**: The strongest single feature, serving as a proxy for body size. Geese have dramatically larger RCS than songbirds.
- **Wingbeat frequency**: Extracted via continuous wavelet transform or FFT from echo intensity fluctuations. Species-diagnostic across groups.
- **Flight speed**: Varies significantly (ducks ~21 m/s, songbirds ~10–13 m/s, raptors often <10 m/s when soaring).
- **Track kinematics**: Trajectory smoothness, turning angles, acceleration patterns, and flight altitude preferences.
- **Temporal features**: Time of day and season strongly constrain which species are present.
- **Flapping pattern**: Flap-pause ratio distinguishes passerine bounding flight from continuous wader flapping and raptor soaring.

A key physical insight from Gong et al. (2020) explains why radar signatures differ: large waders and herons stretch their feet behind them in flight, creating additional scattering centers visible to radar, while small passerines and pigeons tuck their feet, producing simpler signatures.

## No existing competition matches this task, but one comes close

**No Kaggle competition has previously addressed multi-class bird species classification from radar track data.** The AI Cup 2026 appears genuinely novel in this respect. The closest analog is the **ICMCIS Drone Detection/Tracking competition** on Kaggle, which involves classifying and tracking radar targets (drones) — the feature engineering and track-level classification approaches from that competition transfer directly.

The annual **BirdCLEF series** (2021–2025) handles bird species identification from audio, not radar. Several drone-vs-bird visual classification datasets exist on Kaggle but use image data. The **MSTAR benchmark** (20,000+ SAR images, 10 vehicle classes) is the classic radar classification benchmark but operates in an entirely different domain (ground vehicles in synthetic aperture radar imagery).

The **Drone vs Bird Challenge (WOSDETC, 2017–2022)** used sequence classification on track data with 3D CNNs, LSTMs, and Transformers — these temporal modeling approaches could transfer to radar track sequences if the competition data includes per-detection time series rather than only aggregated track statistics.

## Practical resource inventory for competitors

The ENRAM/Aloft data repository (aloftdata.eu) provides aggregate bird migration density from European weather radars, including Dutch KNMI stations. While it lacks species labels, it supplies **migration timing context** — knowing when peak migration occurs at Groningen helps calibrate seasonal species priors. The Figshare dataset for the Netherlands predictive bird migration model combines KNMI weather radar data with ECMWF ERA5 weather variables.

On GitHub, several repositories offer relevant code and methodology. The **aloftdata** organization hosts processing pipelines for weather radar bird data. The **INBO bird-tracking** repository (github.com/inbo/bird-tracking) publishes GPS tracking data processing code. The **Open Radar Initiative** (github.com/openradarinitiative/open_radar_datasets) provides micro-Doppler recognition benchmarks. A synthetic micro-Doppler bird/drone SVM classifier exists at github.com/Keerthi2134/doppler-radar, and a Naive Bayes radar bird/aircraft classifier using trajectory smoothness at github.com/EvanDietrich/Naive-Bayes-for-Radar demonstrates simple but effective track-level features.

No publicly downloadable pretrained model weights exist for radar bird species classification. The BirdScan MR1 system's classification algorithms are proprietary to Swiss-BirdRadar Solution AG. Robin Radar's MAX system classifies by size class only, and its algorithms are not public. **The most practical path to leveraging external data is therefore not transfer learning from pretrained weights, but rather using the flight parameter databases as engineered features or Bayesian priors** — encoding species-specific speed distributions, wingbeat frequencies, and body size relationships directly into the classification pipeline.

## Conclusion

The competition sits at a genuine frontier — no public pretrained model or exact-match dataset exists for 9-class bird species classification from radar tracks. However, competitors can assemble a powerful toolkit from available resources. The **Col de la Croix dataset** provides directly usable labeled radar track data for pretraining. The **Eoldist, Alerstam, and Bruderer databases** supply species-specific flight speed and wingbeat priors that encode decades of radar ornithology expertise. **Movebank GPS tracks** from Dutch bird populations — including data from Groningen itself — enable construction of species-group flight behavior profiles. The literature strongly favors **gradient-boosted tree models with hierarchical classification** as the optimal approach for tabular radar features, with RCS, wingbeat frequency, and flight speed as the three most diagnostic features. Competitors who combine these external knowledge sources as engineered features or informative priors with modern tabular ML methods (LightGBM/XGBoost with careful feature engineering, or TabPFN/TabNet for deep tabular learning) will likely hold a significant advantage over those working with the competition data alone.