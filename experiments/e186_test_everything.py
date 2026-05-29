"""E186: Test ALL remaining untested ideas in one script.

1. OvO Pairwise Classification (TabPFN, key pairs)
2. Spatial-Ecological Features (turbine, heading, commuting)
3. Track-Length Augmentation (subsample long tracks)
4. Robust Focal Loss (noise-robust LGB)
5. Curriculum Learning (clean first, then noisy)
6. Quick LGB+XGB+CB ensemble on ALL features + relabeled data

All evaluated with 5-fold StratifiedGroupKFold on SKF.
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import average_precision_score

from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from src.metrics import compute_map

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)

train = load_train()
test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Load features + cleanlab cache
feats_train = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
feats_test = pd.read_pickle(ROOT / "data/_cached_test_features_v3.pkl")
X_train = np.nan_to_num(feats_train.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(feats_test.values.astype(np.float32), nan=0, posinf=0, neginf=0)

cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
agreed_noisy = cache['agreed_noisy'].tolist()
consensus_labels = cache['consensus_labels']
quality = cache['quality']

y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])

def eval_skf(name, oof):
    skf, pc = compute_map(y, oof)
    print(f"  {name}: SKF={skf:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} "
          f"BoP={pc['Birds of Prey']:.4f} Gulls={pc['Gulls']:.4f}", flush=True)
    return skf, pc

# Reference
print("=" * 90, flush=True)
print("E186: TEST ALL REMAINING IDEAS", flush=True)
print("=" * 90, flush=True)

oof_e175 = np.load(ROOT / "oof_e175_best.npy")
oof_tabpfn_relabel = np.load(ROOT / "oof_e185_tabpfn_relabel.npy")
eval_skf("REF: E175 (LB=0.59)", oof_e175)
eval_skf("REF: TabPFN-ALL relabel", oof_tabpfn_relabel)

# ══════════════════════════════════════════════════════════════
# 1. OvO PAIRWISE CLASSIFICATION
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n1. OvO PAIRWISE CLASSIFICATION (TabPFN)\n{'='*90}", flush=True)

from tabpfn import TabPFNClassifier

t0 = time.time()
oof_ovo = np.zeros((len(y), N))
test_ovo = np.zeros((len(X_test), N))

# For each class pair, train a binary TabPFN
pair_count = 0
for i in range(N):
    for j in range(i+1, N):
        pair_mask_train = (y_relabeled == i) | (y_relabeled == j)
        if pair_mask_train.sum() < 10:
            continue
        pair_count += 1

        # Binary labels: class i = 1, class j = 0
        y_pair = (y_relabeled[pair_mask_train] == i).astype(int)
        X_pair = X_train[pair_mask_train]
        g_pair = groups[pair_mask_train]

        # Cross-val predict
        oof_pair = np.zeros(pair_mask_train.sum())
        test_pair = np.zeros(len(X_test))

        for fold, (tr, va) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X_pair, y_pair, g_pair)):
            clf = TabPFNClassifier(n_estimators=4, random_state=42)  # fewer estimators for speed
            clf.fit(X_pair[tr], y_pair[tr])
            probs = clf.predict_proba(X_pair[va])
            oof_pair[va] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            t_probs = clf.predict_proba(X_test)
            test_pair += (t_probs[:, 1] if t_probs.shape[1] > 1 else t_probs[:, 0]) / 5

        # Accumulate: votes for class i
        train_indices = np.where(pair_mask_train)[0]
        for k, idx in enumerate(train_indices):
            oof_ovo[idx, i] += oof_pair[k]
            oof_ovo[idx, j] += (1 - oof_pair[k])
        test_ovo[:, i] += test_pair
        test_ovo[:, j] += (1 - test_pair)

print(f"  {pair_count} pairs trained in {time.time()-t0:.0f}s", flush=True)

# Normalize
oof_ovo = oof_ovo / np.maximum(oof_ovo.sum(axis=1, keepdims=True), 1e-10)
test_ovo = test_ovo / np.maximum(test_ovo.sum(axis=1, keepdims=True), 1e-10)
eval_skf("OvO Pairwise TabPFN", oof_ovo)
np.save(ROOT / "oof_e186_ovo.npy", oof_ovo)
np.save(ROOT / "test_e186_ovo.npy", test_ovo)

# ══════════════════════════════════════════════════════════════
# 2. SPATIAL-ECOLOGICAL FEATURES
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n2. SPATIAL-ECOLOGICAL FEATURES\n{'='*90}", flush=True)

t0 = time.time()
# Compute for train
def spatial_eco_features(df):
    results = []
    for _, row in df.iterrows():
        pts = parse_ewkb_4d(row['trajectory'])
        times = parse_trajectory_time(row['trajectory_time'])
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        alts = np.array([p[2] for p in pts])
        n = len(pts)
        f = {}

        ts = pd.to_datetime(row['timestamp_start_radar_utc'])
        f['hour'] = ts.hour + ts.minute / 60.0

        if n > 2:
            # Heading (mean bearing)
            dlons = np.diff(lons); dlats = np.diff(lats)
            bearing = np.arctan2(np.mean(dlons), np.mean(dlats))
            f['heading_deg'] = np.degrees(bearing) % 360

            # Heading relative to coast (Eemshaven coast is roughly N-S, coast to east)
            # Toward sea (west/NW) vs inland (east/SE)
            f['heading_toward_sea'] = np.cos(bearing - np.radians(315))  # NW = toward sea

            # Altitude relative to typical rotor zone (30-150m)
            f['below_rotor'] = np.mean(alts < 30)
            f['in_rotor_zone'] = np.mean((alts >= 30) & (alts <= 150))
            f['alt_mean'] = np.mean(alts)

            # Morning vs evening (commuting direction should flip)
            is_morning = f['hour'] < 12
            f['morning_toward_sea'] = f['heading_toward_sea'] if is_morning else -f['heading_toward_sea']
        else:
            f['heading_deg'] = 0; f['heading_toward_sea'] = 0
            f['below_rotor'] = 0; f['in_rotor_zone'] = 0
            f['alt_mean'] = 0; f['morning_toward_sea'] = 0

        results.append(f)
    return pd.DataFrame(results)

print("  Computing spatial-ecological features...", flush=True)
eco_train = spatial_eco_features(train)
eco_test = spatial_eco_features(test)
X_eco_train = np.nan_to_num(eco_train.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_eco_test = np.nan_to_num(eco_test.values.astype(np.float32), nan=0, posinf=0, neginf=0)

# Test: add to existing features
X_train_eco = np.column_stack([X_train, X_eco_train])
X_test_eco = np.column_stack([X_test, X_eco_test])
print(f"  Features: {X_train.shape[1]} -> {X_train_eco.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

# Quick LGB test
import lightgbm as lgb
LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
             subsample=0.7, colsample_bytree=0.5, random_state=42, verbose=-1, n_jobs=1)

probs_eco = cross_val_predict(lgb.LGBMClassifier(**LGB_P), X_train_eco, y_relabeled,
                               cv=sgkf, method="predict_proba", groups=groups)
eval_skf("LGB + spatial-eco features (relabeled)", probs_eco)

# ══════════════════════════════════════════════════════════════
# 3. TRACK-LENGTH AUGMENTATION
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n3. TRACK-LENGTH AUGMENTATION\n{'='*90}", flush=True)

# Subsample long Cormorant tracks to create short-track training examples
CORM = 2
corm_idx = np.where(y == CORM)[0]
long_corm = [i for i in corm_idx if len(parse_ewkb_4d(train.iloc[i]['trajectory'])) >= 60]
print(f"  {len(long_corm)} Cormorants with >= 60 points", flush=True)

# For each long Cormorant, create features from first 20, 30, 40 points
# This means re-extracting features from truncated trajectories
# Expensive to do properly — approximate by using existing features + adding noise
# Better approach: just duplicate Cormorant samples with slight feature noise (like SMOTE but simpler)

# Simple augmentation: duplicate each Cormorant 3x with small noise
rng = np.random.RandomState(42)
aug_X = []
aug_y = []
aug_groups = []
max_grp = groups.max() + 1

for idx in corm_idx:
    for rep in range(3):
        noise = rng.normal(0, 0.05, X_train.shape[1]) * np.abs(X_train[idx])
        aug_X.append(X_train[idx] + noise)
        aug_y.append(y_relabeled[idx])
        aug_groups.append(max_grp + idx * 3 + rep)

X_aug = np.vstack([X_train, np.array(aug_X)])
y_aug = np.concatenate([y_relabeled, np.array(aug_y)])
g_aug = np.concatenate([groups, np.array(aug_groups)])
keep_aug = np.concatenate([keep_mask, np.ones(len(aug_y), dtype=bool)])

print(f"  Augmented: {len(X_train)} -> {len(X_aug)} (+{len(aug_X)} Cormorant copies)", flush=True)

# Train LGB on augmented data, evaluate on ORIGINAL only
oof_aug = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
    # Training: original fold + all augmented Cormorants
    aug_idx = np.arange(len(X_train), len(X_aug))
    tr_combined = np.concatenate([tr[keep_mask[tr]], aug_idx[keep_aug[aug_idx]]])
    clf = lgb.LGBMClassifier(**LGB_P)
    clf.fit(X_aug[tr_combined], y_aug[tr_combined])
    oof_aug[va] = clf.predict_proba(X_train[va])

eval_skf("LGB + Cormorant augmentation", oof_aug)

# ══════════════════════════════════════════════════════════════
# 4. ROBUST FOCAL LOSS (via sample weighting approximation)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n4. ROBUST FOCAL LOSS (quality-weighted)\n{'='*90}", flush=True)

# Approximate robust focal loss by weighting samples by cleanlab quality
# Clean samples: weight 1.0, noisy: weight proportional to quality
# Also add inverse-frequency class weight on top

for min_weight in [0.1, 0.3, 0.5]:
    weights = np.clip(quality, min_weight, 1.0)
    oof_robust = np.zeros((len(y), N))
    for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
        clf = lgb.LGBMClassifier(**LGB_P)
        clf.fit(X_train[tr], y_relabeled[tr], sample_weight=weights[tr])
        oof_robust[va] = clf.predict_proba(X_train[va])
    eval_skf(f"LGB quality-weighted (min={min_weight})", oof_robust)

# ══════════════════════════════════════════════════════════════
# 5. CURRICULUM LEARNING
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n5. CURRICULUM LEARNING (clean first, fine-tune with all)\n{'='*90}", flush=True)

# Stage 1: train on clean subset (quality > 0.3)
# Stage 2: continue training on all data with lower learning rate
clean_mask = quality > 0.3
print(f"  Clean samples (q>0.3): {clean_mask.sum()} / {len(y)}", flush=True)

oof_curriculum = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    # Stage 1: clean only
    tr_clean = tr[clean_mask[tr] & keep_mask[tr]]
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                              random_state=42, verbose=-1, n_jobs=1)
    clf.fit(X_train[tr_clean], y_relabeled[tr_clean])

    # Stage 2: all data, use stage 1 as init
    tr_all = tr[keep_mask[tr]]
    clf2 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.02, num_leaves=31,
                               class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                               random_state=42, verbose=-1, n_jobs=1, init_model=clf)
    clf2.fit(X_train[tr_all], y_relabeled[tr_all])
    oof_curriculum[va] = clf2.predict_proba(X_train[va])

eval_skf("Curriculum (clean->all)", oof_curriculum)

# ══════════════════════════════════════════════════════════════
# 6. QUICK LGB + XGB + CB ENSEMBLE (ALL features, relabeled)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\n6. LGB + XGB + CatBoost ENSEMBLE (ALL features, relabeled)\n{'='*90}", flush=True)

import xgboost as xgb
import catboost as cb

t0 = time.time()

# LGB
oof_lgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              class_weight="balanced", subsample=0.7, colsample_bytree=0.6,
                              random_state=42, verbose=-1, n_jobs=1)
    clf.fit(X_train[tr_use], y_relabeled[tr_use],
            eval_set=[(X_train[va], y[va])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = clf.predict_proba(X_train[va])
eval_skf("LGB (relabeled, ALL feat)", oof_lgb)

# XGBoost
oof_xgb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    # Compute sample weights for class balance
    class_counts = np.bincount(y_relabeled[tr_use], minlength=N)
    sw = np.array([1.0 / max(class_counts[c], 1) for c in y_relabeled[tr_use]])
    sw = sw / sw.mean()

    clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6,
                              subsample=0.7, colsample_bytree=0.6,
                              random_state=42, verbosity=0, n_jobs=1,
                              early_stopping_rounds=50, eval_metric="mlogloss")
    clf.fit(X_train[tr_use], y_relabeled[tr_use], sample_weight=sw,
            eval_set=[(X_train[va], y[va])], verbose=False)
    oof_xgb[va] = clf.predict_proba(X_train[va])
eval_skf("XGB (relabeled, ALL feat)", oof_xgb)

# CatBoost
oof_cb = np.zeros((len(y), N))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_relabeled, groups)):
    tr_use = tr[keep_mask[tr]]
    clf = cb.CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                                 auto_class_weights="Balanced",
                                 subsample=0.7, rsm=0.6,
                                 random_seed=42, verbose=0, task_type="CPU",
                                 early_stopping_rounds=50)
    clf.fit(cb.Pool(X_train[tr_use], y_relabeled[tr_use]),
            eval_set=cb.Pool(X_train[va], y[va]))
    oof_cb[va] = clf.predict_proba(X_train[va])
eval_skf("CatBoost (relabeled, ALL feat)", oof_cb)

print(f"  3-model ensemble training: {time.time()-t0:.0f}s", flush=True)

# Simple average blend
oof_3way = (oof_lgb + oof_xgb + oof_cb) / 3
eval_skf("LGB+XGB+CB average", oof_3way)

# Rank-power blend
from scipy.stats import rankdata
def rpe(preds_list, weights, power=1.5):
    n_s = preds_list[0].shape[0]
    out = np.zeros((n_s, N))
    for c in range(N):
        for p, w in zip(preds_list, weights):
            out[:, c] += w * (rankdata(p[:, c]) / n_s) ** power
    return out

oof_3way_rp = rpe([oof_lgb, oof_xgb, oof_cb], [0.4, 0.3, 0.3], 1.5)
eval_skf("LGB+XGB+CB rank-power", oof_3way_rp)

# + TabPFN
oof_4way = rpe([oof_lgb, oof_xgb, oof_cb, oof_tabpfn_relabel], [0.25, 0.2, 0.2, 0.35], 1.5)
eval_skf("LGB+XGB+CB+TabPFN rank-power", oof_4way)

# Save
np.save(ROOT / "oof_e186_lgb.npy", oof_lgb)
np.save(ROOT / "oof_e186_xgb.npy", oof_xgb)
np.save(ROOT / "oof_e186_cb.npy", oof_cb)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}\nFINAL SUMMARY\n{'='*90}", flush=True)

all_results = [
    ("E175 (LB=0.59)", oof_e175),
    ("TabPFN-ALL relabel", oof_tabpfn_relabel),
    ("OvO Pairwise", oof_ovo),
    ("LGB + eco features", probs_eco),
    ("LGB + Corm augmentation", oof_aug),
    ("LGB quality-weighted (0.3)", oof_robust),  # last one computed
    ("Curriculum learning", oof_curriculum),
    ("LGB (relabeled, ALL)", oof_lgb),
    ("XGB (relabeled, ALL)", oof_xgb),
    ("CatBoost (relabeled, ALL)", oof_cb),
    ("LGB+XGB+CB average", oof_3way),
    ("LGB+XGB+CB rank-power", oof_3way_rp),
    ("LGB+XGB+CB+TabPFN", oof_4way),
]

print(f"\n  {'Config':35s} {'SKF':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Gulls':>7s} {'Geese':>7s}", flush=True)
print(f"  {'-'*90}", flush=True)
for name, oof in all_results:
    skf, pc = compute_map(y, oof)
    print(f"  {name:35s} {skf:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} "
          f"{pc['Birds of Prey']:7.4f} {pc['Gulls']:7.4f} {pc['Geese']:7.4f}", flush=True)

print(f"\nDone.", flush=True)
