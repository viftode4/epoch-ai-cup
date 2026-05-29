"""Find the MAXIMUM separability for Cormorants.

The KNN analysis showed 6.4x clustering — signal EXISTS in high-dimensional space.
PCA KNN gives only 0.583 AUC — signal is NOT in top PCA components.
This means the signal is in SPECIFIC feature combinations. Find them.

Approaches:
1. LDA: find the linear combination that MAXIMIZES Cormorant separation
2. Upper bound: what's the max AUC achievable with a flexible model?
3. Overlap-zone specialized model: within Cormorant-possible region
4. Interaction features from conditional separators
5. Distance-to-Cormorant-cluster as meta-feature
6. Supervised feature selection (RFE)
7. UMAP embedding to visualize where Cormorants actually live
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from src.metrics import compute_map
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
CORM = 2; GULL = 5
y_corm = (y == CORM).astype(int)

feats = pd.read_pickle("G:/Projects/epoch-ai-cup/data/_cached_train_features_v3.pkl")
X = np.nan_to_num(feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)
col_names = list(feats.columns)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 90, flush=True)
print("FINDING MAXIMUM CORMORANT SEPARABILITY", flush=True)
print("=" * 90, flush=True)

# ══════════════════════════════════════════════════════════════════════
# 1. LDA: optimal linear projection for Cormorant separation
# ══════════════════════════════════════════════════════════════════════
print("\n--- 1. LDA: Optimal Linear Projection ---", flush=True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Binary: Cormorant vs rest
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_s, y_corm)
auc_lda = roc_auc_score(y_corm, X_lda.ravel())
ap_lda = average_precision_score(y_corm, X_lda.ravel())
print(f"  LDA (Corm vs rest): AUC={auc_lda:.4f}, AP={ap_lda:.4f}", flush=True)

# Also multiclass LDA (all 9 classes, project to 8 dims, check Cormorant separation)
lda_multi = LinearDiscriminantAnalysis()
X_lda_multi = lda_multi.fit_transform(X_s, y)
# Check Cormorant AUC in LDA space
from sklearn.neighbors import KNeighborsClassifier
knn_lda = cross_val_predict(KNeighborsClassifier(n_neighbors=5, weights="distance"),
                             X_lda_multi, y_corm, cv=sgkf, method="predict_proba")
auc_knn_lda = roc_auc_score(y_corm, knn_lda[:, 1])
ap_knn_lda = average_precision_score(y_corm, knn_lda[:, 1])
print(f"  KNN in multiclass LDA space: AUC={auc_knn_lda:.4f}, AP={ap_knn_lda:.4f}", flush=True)

# Top LDA coefficients (what features contribute most?)
coefs = np.abs(lda.coef_[0])
top_lda = np.argsort(-coefs)[:20]
print(f"\n  Top 20 features in LDA projection:")
for rank, idx in enumerate(top_lda):
    print(f"    {rank+1:2d}. {col_names[idx]:35s} coef={lda.coef_[0][idx]:+.4f} (|{coefs[idx]:.4f}|)")

# ══════════════════════════════════════════════════════════════════════
# 2. UPPER BOUND: flexible model on binary Cormorant detection
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 2. Upper Bound: Best Achievable AUC/AP ---", flush=True)
import lightgbm as lgb

# LGB with extreme class weight
for cw in [10, 50, 100, 200]:
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              scale_pos_weight=cw, subsample=0.7, colsample_bytree=0.5,
                              random_state=42, verbose=-1, n_jobs=1)
    probs = cross_val_predict(clf, X, y_corm, cv=sgkf, method="predict_proba")
    auc = roc_auc_score(y_corm, probs[:, 1])
    ap = average_precision_score(y_corm, probs[:, 1])
    print(f"  LGB binary (pos_weight={cw:3d}): AUC={auc:.4f}, AP={ap:.4f}", flush=True)

# LGB multiclass — Cormorant AP from full 9-class model
clf9 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                            random_state=42, verbose=-1, n_jobs=1)
probs9 = cross_val_predict(clf9, X, y, cv=sgkf, method="predict_proba")
_, pc9 = compute_map(y, probs9)
print(f"  LGB 9-class (balanced): Corm AP={pc9['Cormorants']:.4f}, overall={_:.4f}", flush=True)

# Random Forest (different inductive bias)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample",
                             random_state=42, n_jobs=1)
probs_rf = cross_val_predict(rf, X, y_corm, cv=sgkf, method="predict_proba")
auc_rf = roc_auc_score(y_corm, probs_rf[:, 1])
ap_rf = average_precision_score(y_corm, probs_rf[:, 1])
print(f"  RandomForest binary: AUC={auc_rf:.4f}, AP={ap_rf:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 3. OVERLAP-ZONE SPECIALIZED MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 3. Overlap-Zone Specialized Model ---", flush=True)

# Define overlap zone
speed_col = col_names.index("speed_median") if "speed_median" in col_names else col_names.index("airspeed")
straight_cols = [c for c in col_names if "straight" in c.lower()]
if straight_cols:
    straight_col = col_names.index(straight_cols[0])
    oz = (X[:, speed_col] >= 10) & (X[:, speed_col] <= 20) & (X[:, straight_col] > 0.6)
    print(f"  Overlap zone: {oz.sum()} samples ({(oz & (y==CORM)).sum()} Corm, {(oz & (y==GULL)).sum()} Gull)")

    # Within zone: LGB binary
    X_oz = X[oz]; y_oz = y_corm[oz]; g_oz = groups[oz]
    if y_oz.sum() >= 5:
        sgkf_oz = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        clf_oz = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                     scale_pos_weight=50, random_state=42, verbose=-1, n_jobs=1)
        try:
            probs_oz = cross_val_predict(clf_oz, X_oz, y_oz, cv=sgkf_oz, method="predict_proba",
                                          groups=g_oz)
            auc_oz = roc_auc_score(y_oz, probs_oz[:, 1])
            ap_oz = average_precision_score(y_oz, probs_oz[:, 1])
            print(f"  LGB within overlap zone: AUC={auc_oz:.4f}, AP={ap_oz:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Top features within overlap zone (by importance)
    clf_oz_full = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                      scale_pos_weight=50, random_state=42, verbose=-1, n_jobs=1)
    clf_oz_full.fit(X_oz, y_oz)
    imp = clf_oz_full.feature_importances_
    top_oz = np.argsort(-imp)[:15]
    print(f"\n  Top 15 features within overlap zone:")
    for rank, idx in enumerate(top_oz):
        print(f"    {rank+1:2d}. {col_names[idx]:35s} importance={imp[idx]}")

# ══════════════════════════════════════════════════════════════════════
# 4. INTERACTION FEATURES from conditional separators
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 4. Interaction Features ---", flush=True)

# The conditional analysis found these top separators within overlap zone:
# rcs_for_size, rcs_q75_dB, phys_flock_signal, speed_median, alt_diff_start_end, rcs_deep_fade_frac
key_feats = ["rcs_for_size", "rcs_q75_dB", "phys_flock_signal", "speed_median",
             "alt_diff_start_end", "rcs_deep_fade_frac", "rcs_mean_dB", "rcs_std_dB",
             "rcs_linear_mean", "slow_flight_frac", "lon_std", "radar_bird_size"]
key_feats = [f for f in key_feats if f in col_names]

# Create interaction features (ratios, products)
interactions = {}
for i, f1 in enumerate(key_feats):
    c1 = col_names.index(f1)
    for f2 in key_feats[i+1:]:
        c2 = col_names.index(f2)
        v1 = X[:, c1]; v2 = X[:, c2]
        # Product
        interactions[f"{f1}_x_{f2}"] = v1 * v2
        # Ratio (with safety)
        safe_v2 = np.where(np.abs(v2) > 0.001, v2, 0.001)
        interactions[f"{f1}_div_{f2}"] = v1 / safe_v2

# Test each interaction
print(f"  Testing {len(interactions)} interaction features...")
inter_results = []
for name, vals in interactions.items():
    vals = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
    vals = np.clip(vals, -1e6, 1e6)
    try:
        ap_p = average_precision_score(y_corm, vals)
        ap_n = average_precision_score(y_corm, -vals)
        ap = max(ap_p, ap_n)
        auc = roc_auc_score(y_corm, vals if ap_p >= ap_n else -vals)
        inter_results.append((name, ap, auc))
    except:
        pass

inter_results.sort(key=lambda x: -x[1])
print(f"\n  Top 15 interaction features:")
for name, ap, auc in inter_results[:15]:
    print(f"    {name:50s} AP={ap:.4f} AUC={auc:.3f}")

# ══════════════════════════════════════════════════════════════════════
# 5. DISTANCE-TO-CORMORANT-CLUSTER as meta-feature
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 5. Distance to Cormorant Cluster ---", flush=True)

from sklearn.neighbors import NearestNeighbors

# For each sample, compute distance to nearest K Cormorants in feature space
# Use CV to prevent leakage
oof_dist = np.zeros(len(y))
for fold, (tr, va) in enumerate(sgkf.split(X_s, y, groups)):
    corm_in_train = np.where((y[tr] == CORM))[0]
    corm_X = X_s[tr[corm_in_train]]
    if len(corm_X) < 3:
        continue
    for k in [1, 3, 5]:
        nn = NearestNeighbors(n_neighbors=min(k, len(corm_X)))
        nn.fit(corm_X)
        dists_va, _ = nn.kneighbors(X_s[va])
        mean_dist = dists_va.mean(axis=1)
        if k == 3:  # use k=3 for the main metric
            oof_dist[va] = -mean_dist  # negative = closer = more Cormorant-like

auc_dist = roc_auc_score(y_corm, oof_dist)
ap_dist = average_precision_score(y_corm, oof_dist)
print(f"  Distance to nearest 3 Cormorants (CV): AUC={auc_dist:.4f}, AP={ap_dist:.4f}")

# Also: distance to Cormorant centroid vs Gull centroid RATIO
oof_ratio = np.zeros(len(y))
for fold, (tr, va) in enumerate(sgkf.split(X_s, y, groups)):
    corm_centroid = X_s[tr][y[tr] == CORM].mean(axis=0)
    gull_centroid = X_s[tr][y[tr] == GULL].mean(axis=0)
    for i in va:
        d_corm = np.linalg.norm(X_s[i] - corm_centroid)
        d_gull = np.linalg.norm(X_s[i] - gull_centroid)
        oof_ratio[i] = d_gull / max(d_corm, 0.001)  # high = closer to Cormorant

auc_ratio = roc_auc_score(y_corm, oof_ratio)
ap_ratio = average_precision_score(y_corm, oof_ratio)
print(f"  Distance ratio (d_gull/d_corm): AUC={auc_ratio:.4f}, AP={ap_ratio:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 6. COMBINED: LDA + interactions + distance as new features
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 6. Combined Meta-Features ---", flush=True)

# Combine: original features + top interactions + distance features + LDA score
top_inter_names = [name for name, _, _ in inter_results[:10]]
X_inter = np.column_stack([interactions[n] for n in top_inter_names])
X_meta = np.column_stack([X, X_inter, X_lda, oof_dist.reshape(-1,1), oof_ratio.reshape(-1,1)])

# Test with LGB 9-class
clf_meta = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                class_weight="balanced", subsample=0.7, colsample_bytree=0.5,
                                random_state=42, verbose=-1, n_jobs=1)
probs_meta = cross_val_predict(clf_meta, X_meta, y, cv=sgkf, method="predict_proba")
map_meta, pc_meta = compute_map(y, probs_meta)
print(f"  LGB 9-class with meta-features: mAP={map_meta:.4f}, Corm={pc_meta['Cormorants']:.4f}")

# Compare to baseline
print(f"  vs LGB 9-class baseline:        mAP={_:.4f}, Corm={pc9['Cormorants']:.4f}")
print(f"  Delta: mAP={map_meta - _:+.4f}, Corm={pc_meta['Cormorants'] - pc9['Cormorants']:+.4f}")

# ══════════════════════════════════════════════════════════════════════
# 7. WHAT'S THE ACTUAL CEILING?
# ══════════════════════════════════════════════════════════════════════
print("\n\n--- 7. Ceiling Analysis ---", flush=True)

# If we had a PERFECT Cormorant detector, what would the 9-class mAP be?
# Replace Cormorant column with perfect scores
probs_perfect = probs9.copy()
probs_perfect[:, CORM] = y_corm.astype(float)  # perfect Cormorant predictions
map_ceiling, pc_ceiling = compute_map(y, probs_perfect)
print(f"  With PERFECT Cormorant predictions: mAP={map_ceiling:.4f} (current {_:.4f})")
print(f"  Cormorant AP ceiling: {pc_ceiling['Cormorants']:.4f}")
print(f"  Maximum mAP gain from fixing Cormorants: {map_ceiling - _:+.4f}")

# Same for Waders
probs_perfect_w = probs9.copy()
probs_perfect_w[:, CLASSES.index('Waders')] = (y == CLASSES.index('Waders')).astype(float)
map_ceil_w, _ = compute_map(y, probs_perfect_w)
print(f"  With PERFECT Wader predictions: mAP={map_ceil_w:.4f} (gain {map_ceil_w - _:+.4f})")

# Both perfect
probs_both = probs9.copy()
probs_both[:, CORM] = y_corm.astype(float)
probs_both[:, CLASSES.index('Waders')] = (y == CLASSES.index('Waders')).astype(float)
map_both, _ = compute_map(y, probs_both)
print(f"  With BOTH perfect: mAP={map_both:.4f} (gain {map_both - _:+.4f})")

print("\nDone.", flush=True)
