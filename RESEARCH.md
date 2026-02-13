# Research Notes — Bird Radar Classification

## Papers & References

### 1. Wingbeat Identification from Radar via Wing Flapping Patterns
- **Paper**: Zaugg et al. (2008) — "Automatic identification of bird targets with radar via patterns produced by wing flapping"
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC2607429/
- **Key method**: Continuous Wavelet Transform (CWT) with Morlet wavelet on RCS signal (0.3–65 Hz range)
- **Features**: Wavelet coefficient mean/std across frequencies, peak frequency, signal intensity SD, max signal intensity
- **Model**: SVM with Laplace kernel — 43 of 67 extracted features used
- **Results**: AUC 0.96–0.99 for bird vs non-bird classification
- **Takeaway**: Wing flapping produces periodic RCS oscillations. CWT captures this better than FFT because it handles non-stationary signals (birds that alternate flapping/gliding)
- **Use in our project**: Replace our basic FFT features with CWT-based wavelet energy features at multiple frequency bands. Apply LOWESS detrending before wavelet analysis.

### 2. Large vs Small Bird Radar Signatures
- **Paper**: Gong et al. (2020) — "Comparison of radar signatures based on flight morphology for large birds and small birds"
- **Link**: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2020.0064
- **Key finding**: Wingbeat frequency decreases with increasing bird size. FFT peak frequency correlates with body mass.
- **Takeaway**: RCS oscillation frequency is a proxy for bird size/species group
- **Use in our project**: Extract dominant wingbeat frequency from RCS, use as feature. Cross-reference with radar_bird_size for consistency features.

### 3. Flight Mode Classification from Radar
- **Paper**: Referenced in ResearchGate — "Using Radar Signatures to Classify Bird Flight Modes Between Flapping and Gliding"
- **Link**: https://www.researchgate.net/publication/337041034_Using_Radar_Signatures_to_Classify_Bird_Flight_Modes_Between_Flapping_and_Gliding
- **Key method**: Classify radar segments into flapping vs gliding phases based on RCS variance
- **Species patterns**:
  - Pigeons: continuous fast flapping, direct flight
  - Gulls: long gliding with occasional flaps
  - Birds of Prey: soaring/circling with altitude gain
  - Songbirds: bounding flight (flap-pause-flap, altitude oscillation)
  - Waders/waterfowl: continuous wingbeats, no pauses
- **Use in our project**: Segment each trajectory into flap/glide/pause phases. Compute ratios (flap_fraction, glide_fraction, n_phase_transitions). This directly targets Pigeon vs Songbird discrimination.

### 4. Bird & Bat Classification Using Flight Tracks (PNNL)
- **Paper**: PNNL — "Classification of birds and bats using flight tracks"
- **Link**: https://www.pnnl.gov/sites/default/files/media/file/Classification%20of%20birds%20and%20bats%20using%20flight%20tracks.pdf
- **Also**: https://www.sciencedirect.com/science/article/abs/pii/S1574954115000692
- **Key method**: Track sinuosity and wingbeat frequency from thermal video as primary discriminants
- **Key finding**: Larger birds (gulls) fly straighter than small birds (swallows). Track shape + wingbeat = 82% classification accuracy.
- **Use in our project**: We already have sinuosity. Add finer trajectory shape features: fractal dimension, direction autocorrelation, curvature distribution.

### 5. ML Algorithms in Radar Ornithology
- **Paper**: "Classification success of six machine learning algorithms in radar ornithology"
- **Link**: https://www.researchgate.net/publication/283984520_Classification_success_of_six_machine_learning_algorithms_in_radar_ornithology
- **Key finding**: Random Forest held accuracy >0.80 for ALL classification tasks (bird vs clutter, bird group separation). Other algorithms (SVM, NN, LDA) dropped when doing species-group classification.
- **Takeaway**: Ensemble tree methods are the right base approach. Species-level is harder than bird-vs-clutter.
- **Use in our project**: Validates our LGB/XGB/CB ensemble approach.

### 6. 1D-CNN for Time Series Classification
- **Paper**: "Rethinking 1D-CNN for Time Series Classification: A Stronger Baseline"
- **Link**: https://www.researchgate.net/publication/339471768_Rethinking_1D_CNN_for_Time_Series_Classification_A_Stronger_Baseline
- **Key method**: Simple 1D-CNN with batch norm outperforms complex architectures on many TSC benchmarks
- **Use in our project**: Feed raw (alt, RCS, speed, bearing_change) as 4-channel time series into 1D-CNN. Pad/truncate to fixed length. Train as separate model, blend with tabular ensemble.

### 7. Transformer for Time Series Classification
- **Link**: https://keras.io/examples/timeseries/timeseries_classification_transformer/
- **Key method**: Multi-head self-attention on time series. Good for variable-length sequences with positional encoding.
- **Use in our project**: Alternative to 1D-CNN for the raw trajectory model. May capture long-range dependencies (e.g., circling patterns for Birds of Prey).

### 8. Deep Learning for Aviation Bird Safety
- **Paper**: "Deep Learning–Based Multi-Level Classification for Aviation Safety" (2025)
- **Link**: https://arxiv.org/html/2602.07019
- **Key finding**: Image-based CNNs (ResNet50V2) achieved 92.8% on 24 species. But the paper explicitly states "avian radars cannot identify bird species" — motivating why radar-only classification is hard.
- **Takeaway**: Confirms radar-only species classification is a genuinely hard problem. Our 0.72 mAP is reasonable.

### 9. Radar Post-Processing Framework (2024)
- **Paper**: Erp et al. (2024) — "A framework for post-processing bird tracks from automated tracking radar systems"
- **Link**: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14249
- **Key method**: birdR R package for filtering and quality control of Robin Radar 3D-Fix data
- **Takeaway**: Our data comes from MAX Avian Radar (Robin Radar). Similar post-processing principles apply.

### 10. Robin Radar MAX System
- **Link**: https://www.robinradar.com/products/max-radar
- **Key info**: This is the exact radar that collected our data. Multiple stacked beams for altitude, market-leading rotation speed for fast track updates.
- **Takeaway**: Understanding the radar's capabilities helps interpret data quality and limitations.

---

## Implementation Plan

### Phase 1: Wavelet + Flight Mode Features (add to tabular ensemble)
- [ ] CWT with Morlet wavelet on RCS time series → energy in 4-5 frequency bands
- [ ] LOWESS detrending of RCS before analysis (per Paper #1)
- [ ] Flight mode segmentation: detect flap/glide/pause from RCS variance in sliding windows
- [ ] Compute: flap_fraction, glide_fraction, n_phase_transitions, mean_flap_duration
- [ ] Altitude oscillation frequency (bounding flight detection for Songbirds)
- [ ] Trajectory curvature profile (circling detection for Birds of Prey)
- [ ] Direction autocorrelation (straight flight vs erratic)

### Phase 2: 1D-CNN / Transformer on Raw Trajectory
- [ ] Preprocess: interpolate trajectories to fixed-length (e.g., 64 or 128 steps)
- [ ] Channels: altitude, RCS, speed, bearing_change, lat_delta, lon_delta
- [ ] Architecture: 1D-CNN with batch norm (Paper #6) or small Transformer (Paper #7)
- [ ] Train with same 5-fold CV, blend predictions with tabular ensemble

### Phase 3: Advanced Features
- [ ] Fractal dimension of trajectory path
- [ ] RCS autocorrelation features
- [ ] Trajectory shape descriptors (Fourier descriptors of 2D path)
- [ ] Cross-feature: wingbeat_freq × radar_bird_size consistency

---

## Current Model Performance

| Version | Model | CV mAP | Notes |
|---------|-------|--------|-------|
| v1 | LightGBM only | 0.7030 | 40 features, basic |
| v2 | LGB+XGB+CB ensemble | 0.7214 | 75 features, class weights, best so far |
| v3 | v2 + targeted features | 0.7213 | 115 features — too many, diluted signal |
| v4 | Multi-seed ensemble | 0.7197 | Extra Pigeon weight hurt Ducks |

**Best submission: v2 (0.7214 mAP)**

### Per-Class AP (v2 — current best)
| Class | AP | Samples | Key challenge |
|-------|-----|---------|---------------|
| Gulls | 0.956 | 1503 | Solved |
| Cormorants | 0.939 | 40 | Solved despite tiny sample |
| Birds of Prey | 0.885 | 108 | Good — distinctive slow flight |
| Waders | 0.816 | 120 | Good |
| Geese | 0.728 | 83 | OK — overlaps with Ducks in size |
| Ducks | 0.666 | 58 | Weak — tiny sample, overlaps Pigeons |
| Songbirds | 0.640 | 483 | Weak — confused with Gulls |
| Clutter | 0.610 | 84 | Weak — high RCS is main signal |
| Pigeons | 0.254 | 122 | Very weak — overlaps everything |
