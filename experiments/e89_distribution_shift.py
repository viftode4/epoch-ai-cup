"""E89: Distribution shift countermeasures -- LOMO validation.

Three techniques to handle train-test temporal shift (AUC 0.75):
  A. Per-month feature normalization (speed/altitude z-scored per month)
  B. Adversarial sample reweighting (upweight test-like training samples)
  C. Pseudo-labeling (add high-confidence E79 test predictions to training)

All validated with LOMO. Baseline = E79's 36-feature LGB+CB ensemble.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42

KEEP_FEATURES = [
    f.strip() for f in (ROOT / "data" / "best_features.txt").read_text().splitlines()
    if f.strip()
]

# Features that shift most (from adversarial validation)
SPEED_FEATS = ["airspeed_vs_ground", "avg_ground_speed", "speed_median",
               "airspeed", "accel_std", "slow_flight_frac", "speed_x_alt"]
ALT_FEATS = ["alt_max", "alt_median", "alt_q75", "alt_change_halves", "alt_rate_mean"]
NORMALIZE_FEATS = SPEED_FEATS + ALT_FEATS


def load_and_build():
    """Load data, build 36 features, return X, X_test, y, months."""
    train_df = load_train()
    test_df = load_test()
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(train_df["bird_group"])

    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
    train_feats = build_features(train_df, feature_sets=feat_sets)
    test_feats = build_features(test_df, feature_sets=feat_sets)

    # Remove temporal
    keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
    train_feats = train_feats[keep]
    test_feats = test_feats[keep]

    # Add weather + solar
    for prefix, fname_pat in [("wx_", "{}_weather.csv"), ("sol_", "{}_solar.csv")]:
        for split, feats in [("train", train_feats), ("test", test_feats)]:
            path = ROOT / "data" / fname_pat.format(split)
            if path.exists():
                df = pd.read_csv(path)
                for c in df.columns:
                    feats[f"{prefix}{c}"] = df[c].values

    # Prune to 36
    available = [f for f in KEEP_FEATURES if f in train_feats.columns]
    train_feats = train_feats[available]
    test_feats = test_feats[available]

    feature_names = list(train_feats.columns)

    X = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X_test = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

    return X, X_test, y, train_months, test_months, feature_names, train_df, test_df


def class_weights(y):
    """Effective number class weights (beta=0.999)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    beta = 0.999
    eff_n = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff_n, 1e-6)
    w /= w.sum() / N_CLASSES
    return w[y]


def train_ensemble_lomo(X, y, train_months, sample_weight=None):
    """Train LGB+CB with LOMO, return OOF predictions and per-fold scores."""
    unique_months = sorted(np.unique(train_months))
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    sw = sample_weight if sample_weight is not None else class_weights(y)

    for month in unique_months:
        va = np.where(train_months == month)[0]
        tr = np.where(train_months != month)[0]

        # LGB
        lgb = LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
            subsample=0.7, colsample_bytree=0.5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight="balanced", random_state=SEED, verbose=-1, device="gpu",
        )
        lgb.fit(X[tr], y[tr], sample_weight=sw[tr],
                eval_set=[(X[va], y[va])])
        oof_lgb = lgb.predict_proba(X[va])

        # CatBoost
        cb = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, loss_function="MultiClass",
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type="GPU",
        )
        cb.fit(X[tr], y[tr], eval_set=(X[va], y[va]), verbose=0)
        oof_cb = cb.predict_proba(X[va])

        # Blend (50/50 default for LOMO)
        oof[va] = 0.5 * oof_lgb + 0.5 * oof_cb

    lomo_map, per_class = compute_map(y, oof)
    return lomo_map, per_class, oof


def monthly_normalize(X, months, feature_names):
    """Z-score normalize shifting features per-month."""
    X_norm = X.copy()
    norm_idx = [feature_names.index(f) for f in NORMALIZE_FEATS if f in feature_names]

    for idx in norm_idx:
        for m in np.unique(months):
            mask = months == m
            vals = X_norm[mask, idx]
            mu = np.mean(vals)
            std = np.std(vals) + 1e-8
            X_norm[mask, idx] = (vals - mu) / std
    return X_norm


def compute_adversarial_weights(X_train, X_test, boost_factor=3.0):
    """Train adversarial classifier, return sample weights for training."""
    import lightgbm as lgbm
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_all))

    for _, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        dtrain = lgbm.Dataset(X_all[tr_idx], label=y_all[tr_idx])
        dval = lgbm.Dataset(X_all[va_idx], label=y_all[va_idx])
        mdl = lgbm.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
             "num_leaves": 20, "max_depth": 4, "verbose": -1, "seed": 42,
             "subsample": 0.7, "colsample_bytree": 0.7, "device": "gpu"},
            dtrain, num_boost_round=500, valid_sets=[dval],
            callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(0)]
        )
        oof[va_idx] = mdl.predict(X_all[va_idx])

    train_scores = oof[:len(X_train)]
    # Higher score = more test-like = higher weight
    # Scale: samples with high adversarial score get boosted
    weights = 1.0 + boost_factor * train_scores
    # Normalize to mean=1
    weights /= weights.mean()
    return weights, roc_auc_score(y_all, oof)


# =====================================================================
print("=" * 70)
print("E89: DISTRIBUTION SHIFT COUNTERMEASURES (LOMO VALIDATION)".center(70))
print("=" * 70, flush=True)

# -- Load data --------------------------------------------------------
print("\nLoading data and building features...", flush=True)
X, X_test, y, train_months, test_months, feature_names, train_df, test_df = load_and_build()
print(f"  Train: {X.shape}, Test: {X_test.shape}, Features: {X.shape[1]}", flush=True)

# -- A. BASELINE (E79 equivalent with LOMO) ---------------------------
print("\n" + "=" * 60)
print("A. BASELINE (E79 pipeline, LOMO)")
print("=" * 60, flush=True)

base_lomo, base_per, base_oof = train_ensemble_lomo(X, y, train_months)
print(f"  LOMO mAP: {base_lomo:.4f}", flush=True)
print_results(base_lomo, base_per, label="Baseline LOMO")

# -- B. PER-MONTH FEATURE NORMALIZATION --------------------------------
print("\n" + "=" * 60)
print("B. PER-MONTH FEATURE NORMALIZATION")
print("=" * 60, flush=True)

norm_feats = [f for f in NORMALIZE_FEATS if f in feature_names]
print(f"  Normalizing {len(norm_feats)} features: {norm_feats}", flush=True)

X_norm = monthly_normalize(X, train_months, feature_names)
norm_lomo, norm_per, norm_oof = train_ensemble_lomo(X_norm, y, train_months)
print(f"  LOMO mAP: {norm_lomo:.4f} (delta: {norm_lomo - base_lomo:+.4f})", flush=True)
print_results(norm_lomo, norm_per, label="Monthly-normalized LOMO")

# Per-class comparison
print("\n  Per-class comparison (norm - base):")
for cls in CLASSES:
    d = norm_per.get(cls, 0) - base_per.get(cls, 0)
    if abs(d) > 0.001:
        print(f"    {cls:20s}: {d:+.4f}")

# -- C. ADVERSARIAL SAMPLE REWEIGHTING --------------------------------
print("\n" + "=" * 60)
print("C. ADVERSARIAL SAMPLE REWEIGHTING")
print("=" * 60, flush=True)

# Compute adversarial weights
print("  Computing adversarial weights...", flush=True)
adv_weights, adv_auc = compute_adversarial_weights(X, X_test)
print(f"  Adversarial AUC: {adv_auc:.4f}", flush=True)
print(f"  Weight stats: min={adv_weights.min():.2f} max={adv_weights.max():.2f} "
      f"mean={adv_weights.mean():.2f} std={adv_weights.std():.2f}", flush=True)

# Combine adversarial weights with class weights
base_sw = class_weights(y)
for boost in [1.0, 2.0, 3.0, 5.0]:
    adv_w, _ = compute_adversarial_weights(X, X_test, boost_factor=boost)
    combined_w = base_sw * adv_w
    combined_w /= combined_w.mean()  # normalize

    rw_lomo, rw_per, rw_oof = train_ensemble_lomo(X, y, train_months, sample_weight=combined_w)
    print(f"  boost={boost:.1f}: LOMO mAP = {rw_lomo:.4f} (delta: {rw_lomo - base_lomo:+.4f})", flush=True)

    if boost == 3.0:  # default
        rw3_lomo, rw3_per = rw_lomo, rw_per

# Per-class for boost=3
print("\n  Per-class comparison (reweight boost=3 - base):")
for cls in CLASSES:
    d = rw3_per.get(cls, 0) - base_per.get(cls, 0)
    if abs(d) > 0.001:
        print(f"    {cls:20s}: {d:+.4f}")

# -- D. PSEUDO-LABELING -----------------------------------------------
print("\n" + "=" * 60)
print("D. PSEUDO-LABELING")
print("=" * 60, flush=True)

# Load E79 test predictions
test_pred_path = ROOT / "test_e79.npy"
if test_pred_path.exists():
    test_preds_e79 = np.load(test_pred_path)
    print(f"  Loaded E79 test predictions: {test_preds_e79.shape}", flush=True)

    # Test various confidence thresholds
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        max_probs = test_preds_e79.max(axis=1)
        confident_mask = max_probs >= threshold
        n_pseudo = confident_mask.sum()
        pseudo_labels = test_preds_e79[confident_mask].argmax(axis=1)

        if n_pseudo < 10:
            print(f"  threshold={threshold:.2f}: only {n_pseudo} samples, skipping", flush=True)
            continue

        # Class distribution of pseudo-labels
        pseudo_counts = np.bincount(pseudo_labels, minlength=N_CLASSES)
        print(f"  threshold={threshold:.2f}: {n_pseudo} pseudo-labeled samples", flush=True)
        dist_str = ", ".join(f"{CLASSES[i]}:{pseudo_counts[i]}" for i in range(N_CLASSES) if pseudo_counts[i] > 0)
        print(f"    distribution: {dist_str}", flush=True)

        # Get pseudo-labeled test features and months
        X_pseudo = X_test[confident_mask]
        pseudo_months = test_months[confident_mask]

        # Augmented training set
        X_aug = np.vstack([X, X_pseudo])
        y_aug = np.concatenate([y, pseudo_labels])
        months_aug = np.concatenate([train_months, pseudo_months])

        # Augmented sample weights (pseudo-labels get lower weight)
        for pseudo_weight in [0.3, 0.5, 1.0]:
            sw_base = class_weights(y_aug)
            # Scale pseudo-label weights
            sw_aug = sw_base.copy()
            sw_aug[len(y):] *= pseudo_weight

            pl_lomo, pl_per, pl_oof = train_ensemble_lomo(
                X_aug, y_aug, months_aug, sample_weight=sw_aug
            )
            print(f"    pw={pseudo_weight:.1f}: LOMO mAP = {pl_lomo:.4f} "
                  f"(delta: {pl_lomo - base_lomo:+.4f})", flush=True)
else:
    print("  WARNING: test_e79.npy not found, skipping pseudo-labeling", flush=True)

# -- E. COMBINED: best of each approach --------------------------------
print("\n" + "=" * 60)
print("E. COMBINED APPROACHES")
print("=" * 60, flush=True)

# Normalization + adversarial reweighting
print("  E1. Monthly normalization + adversarial reweighting (boost=3)...", flush=True)
adv_w3, _ = compute_adversarial_weights(X_norm, X_test, boost_factor=3.0)
combined_sw = class_weights(y) * adv_w3
combined_sw /= combined_sw.mean()
e1_lomo, e1_per, _ = train_ensemble_lomo(X_norm, y, train_months, sample_weight=combined_sw)
print(f"  E1 LOMO mAP: {e1_lomo:.4f} (delta: {e1_lomo - base_lomo:+.4f})", flush=True)

# Pseudo-label + adversarial reweighting (if available)
if test_pred_path.exists():
    # Use threshold=0.8 and pw=0.5 as default combo
    max_probs = test_preds_e79.max(axis=1)
    confident_mask = max_probs >= 0.8
    n_pseudo = confident_mask.sum()

    if n_pseudo >= 10:
        pseudo_labels = test_preds_e79[confident_mask].argmax(axis=1)
        X_pseudo = X_test[confident_mask]
        pseudo_months = test_months[confident_mask]

        X_aug = np.vstack([X, X_pseudo])
        y_aug = np.concatenate([y, pseudo_labels])
        months_aug = np.concatenate([train_months, pseudo_months])

        # Adversarial weights for augmented set
        adv_w_aug, _ = compute_adversarial_weights(X_aug, X_test, boost_factor=3.0)
        sw_aug = class_weights(y_aug) * adv_w_aug
        sw_aug[len(y):] *= 0.5  # pseudo-label discount
        sw_aug /= sw_aug.mean()

        print("  E2. Pseudo-label (t=0.8, pw=0.5) + adversarial reweighting...", flush=True)
        e2_lomo, e2_per, _ = train_ensemble_lomo(X_aug, y_aug, months_aug, sample_weight=sw_aug)
        print(f"  E2 LOMO mAP: {e2_lomo:.4f} (delta: {e2_lomo - base_lomo:+.4f})", flush=True)

# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  A. Baseline (E79 LOMO):          {base_lomo:.4f}")
print(f"  B. Monthly normalization:        {norm_lomo:.4f}  ({norm_lomo - base_lomo:+.4f})")
print(f"  C. Adversarial reweight (b=3):   {rw3_lomo:.4f}  ({rw3_lomo - base_lomo:+.4f})")
print(f"  E1. Norm + reweight:             {e1_lomo:.4f}  ({e1_lomo - base_lomo:+.4f})")
print(flush=True)
print("=" * 60)
print("Done!", flush=True)
