# AI Architecture Descriptions  -  Team Epoch (AI Cup 2026)

Two models are described below: the official competition submission and the best model weights.

---

## Model 1: Competition Submission  -  E113 (Public LB 0.5999)

Our competition submission is a **graph-smoothed geometric mean ensemble** of five independently post-processed gradient-boosted tree models. Each base model is a weighted ensemble of LightGBM (50%), XGBoost (40%), and CatBoost (10%) trained on 36 features selected via backward elimination from 139 candidates, covering trajectory kinematics (speed, altitude, curvature, RCS modulation), spatial context, and KNMI weather/solar variables used as ecological month proxies.

The five base models are post-processed independently using a 3-stage Naive Bayes pipeline: (1) GBIF-based label-shift correction for unseen months, (2) physics evidence extraction via diagonal Gaussian NB over airspeed, altitude, heading consistency, and RCS autocorrelation, and (3) a gated Product-of-Experts update (γ=0.10) applied only to uncertain predictions in unseen months {February, May, December}. Their outputs are combined via geometric mean in log-probability space. Finally, graph-based flock smoothing is applied: a HistGradientBoosting pairwise link predictor (trained on season-invariant relative features: Δtime, Δposition, Δspeed, Δaltitude, size match) identifies co-occurring tracks from the same flock, and predictions are averaged within each connected component.

---

## Model 2: Best Model Weights  -  E205 multi_restart_T09 (Private LB 0.5453)

Our best model weights come from a **diverse 11-model ensemble** whose blend weights are optimised via multi-restart Nelder-Mead on out-of-fold predictions, with temperature sharpening (T=0.9) applied to the final blend. The pool combines fundamentally different model families: a One-vs-One multiclass ranker (56.8%), TabPFN in-context learning (25.2%), OvR LambdaRank with month-grouped queries (9.4%), a 1D CNN on raw trajectory sequences (2.3%), gradient-boosted trees (1.7%), and several CatBoost and blend variants. Ensemble diversity  -  not individual model strength  -  drives the improvement over the tree-only baseline.

The OvR LambdaRank component directly optimises per-class average precision using calendar months as query groups (Group DRO for ranking), forcing good rankings in every deployment month rather than on average. Features for this component include 86 hand-crafted kinematic and RCS features, 90 log-path signatures capturing cross-channel trajectory geometry (altitude–RCS coupling, trajectory area), and 22 catch22 time-series statistics per channel, with cross-month stability selection ensuring features generalise across months rather than acting as calendar proxies.

---

## Research Infrastructure

Alongside the ML work, the team used [research-copilot](https://github.com/viftode4/research-copilot) - a terminal-first AI research loop with a TUI and autonomous workflow commands - to manage experiment triage, architecture decisions, and review cycles. This repo also contains a lighter competition-specific spin-off (`research-os` CLI) that wraps the same experiment state in canonical commands for running baselines, validating predictions, and updating the experiment log. Neither tool is part of the ML pipeline itself; both served as research operations infrastructure.
