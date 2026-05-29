"""DEEP CORMORANT MODELING ANALYSIS.

Stop looking at individual features. Instead:
1. Do Cormorants cluster together in feature space? (vs scattered among Gulls)
2. What CONDITIONAL patterns separate them? (within the overlap zone)
3. What INTERACTIONS matter? (feature pairs/triples that jointly separate)
4. What does the RAW trajectory sequence look like? (not just summary stats)
5. Can we find a SUBSPACE where Cormorants are separable?
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from src.data import load_train, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

train = load_train()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
CORM = 2; GULL = 5
corm_idx = np.where(y == CORM)[0]
gull_idx = np.where(y == GULL)[0]

# Load ALL features
feats = pd.read_pickle("G:/Projects/epoch-ai-cup/data/_cached_train_features_v3.pkl")
X = np.nan_to_num(feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 90, flush=True)
print("PART 1: DO CORMORANTS CLUSTER TOGETHER?", flush=True)
print("=" * 90, flush=True)

# KNN analysis: for each Cormorant, how many of its K nearest neighbors are also Cormorants?
nn = NearestNeighbors(n_neighbors=20, metric="euclidean")
nn.fit(X_scaled)
dists, indices = nn.kneighbors(X_scaled[corm_idx])

print("\nFor each Cormorant, class distribution of 20 nearest neighbors:", flush=True)
total_neighbor_classes = np.zeros(len(CLASSES))
corm_in_neighbors = []
for i in range(len(corm_idx)):
    neighbor_labels = y[indices[i]]
    n_corm = (neighbor_labels == CORM).sum()
    corm_in_neighbors.append(n_corm)
    for c in range(len(CLASSES)):
        total_neighbor_classes[c] += (neighbor_labels == c).sum()

# Expected by chance: 40/2601 = 1.5% -> in 20 neighbors, ~0.3 Cormorants
print(f"  Mean Cormorant neighbors: {np.mean(corm_in_neighbors):.1f} / 20 (expected by chance: 0.3)")
print(f"  Median: {np.median(corm_in_neighbors):.0f}, Max: {np.max(corm_in_neighbors)}")
print(f"  Cormorants with 0 Corm neighbors: {(np.array(corm_in_neighbors) == 0).sum()}")
print(f"  Cormorants with 3+ Corm neighbors: {(np.array(corm_in_neighbors) >= 3).sum()}")

print(f"\n  Total neighbor distribution (all 40 Cormorants x 20 neighbors):")
for c in range(len(CLASSES)):
    pct = total_neighbor_classes[c] / (40 * 20) * 100
    expected = (y == c).sum() / len(y) * 100
    ratio = pct / max(expected, 0.01)
    print(f"    {CLASSES[c]:15s}: {pct:5.1f}% (expected {expected:5.1f}%, ratio {ratio:.2f}x)")

# Per-Cormorant: show which ones are isolated vs clustered
print(f"\n  Per-Cormorant clustering:", flush=True)
for i, idx in enumerate(corm_idx):
    n_corm = corm_in_neighbors[i]
    # What's the nearest Cormorant?
    corm_mask = np.isin(indices[i], corm_idx)
    nearest_corm_dist = dists[i][corm_mask].min() if corm_mask.any() else 999
    # Nearest Gull
    gull_mask = np.isin(indices[i], gull_idx)
    nearest_gull_dist = dists[i][gull_mask].min() if gull_mask.any() else 999
    ratio = nearest_corm_dist / max(nearest_gull_dist, 0.001)
    label = "CLUSTERED" if n_corm >= 3 else "MODERATE" if n_corm >= 1 else "ISOLATED"
    print(f"    Corm#{i+1:2d}: {n_corm:2d} corm neighbors, "
          f"nearest_corm={nearest_corm_dist:.2f}, nearest_gull={nearest_gull_dist:.2f}, "
          f"ratio={ratio:.2f} [{label}] m={months[idx]}")

print("\n\n" + "=" * 90, flush=True)
print("PART 2: CONDITIONAL ANALYSIS (within the overlap zone)", flush=True)
print("=" * 90, flush=True)

# Define overlap zone: speed 10-20, straightness > 0.6
# Use feature columns
speed_col = feats.columns.get_loc("speed_median") if "speed_median" in feats.columns else feats.columns.get_loc("airspeed")
straight_col = None
for c in feats.columns:
    if "straight" in c.lower():
        straight_col = feats.columns.get_loc(c)
        break

if straight_col is not None:
    speed_vals = X[:, speed_col]
    straight_vals = X[:, straight_col]
    overlap_mask = (speed_vals >= 10) & (speed_vals <= 20) & (straight_vals > 0.6)
    print(f"\n  Overlap zone (speed 10-20, straight>0.6): {overlap_mask.sum()} samples")
    print(f"    Cormorants in zone: {(overlap_mask & (y == CORM)).sum()} / 40")
    print(f"    Gulls in zone: {(overlap_mask & (y == GULL)).sum()} / {(y == GULL).sum()}")

    # Within overlap zone, which features NOW separate?
    oz_corm = overlap_mask & (y == CORM)
    oz_gull = overlap_mask & (y == GULL)
    if oz_corm.sum() >= 5 and oz_gull.sum() >= 5:
        print(f"\n  Features that separate WITHIN the overlap zone:")
        oz_results = []
        for col_idx, col_name in enumerate(feats.columns):
            cv = X[oz_corm, col_idx]
            gv = X[oz_gull, col_idx]
            try:
                from scipy.stats import mannwhitneyu
                stat, pval = mannwhitneyu(cv, gv)
                effect = stat / (len(cv) * len(gv))
                oz_results.append((col_name, pval, abs(effect - 0.5), np.median(cv), np.median(gv)))
            except:
                pass
        oz_results.sort(key=lambda x: x[1])
        print(f"\n  {'Feature':35s} {'p-value':>10s} {'Effect':>8s} {'Corm_med':>10s} {'Gull_med':>10s}")
        print(f"  {'-'*80}")
        for name, pval, eff, cm, gm in oz_results[:25]:
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {name:35s} {pval:10.6f} {eff:8.3f} {cm:10.4f} {gm:10.4f} {sig}")

print("\n\n" + "=" * 90, flush=True)
print("PART 3: FEATURE INTERACTIONS (pairs that jointly separate)", flush=True)
print("=" * 90, flush=True)

# Find feature PAIRS where the 2D box captures Cormorants with high purity
# Use a subset of promising features
promising = ["speed_median", "airspeed", "rcs_deep_fade_frac", "rcs_std_dB",
             "alt_mean", "duration", "radar_bird_size", "rcs_kurtosis_linear",
             "rcs_scintillation", "speed_trend", "climb_frac", "phys_flock_signal"]
promising = [f for f in promising if f in feats.columns]

# Add any columns with "straight" or "ac" or "cv"
for c in feats.columns:
    if any(k in c.lower() for k in ["straight", "rcs_ac", "speed_cv", "alt_diff", "alt_c22_15"]):
        if c not in promising:
            promising.append(c)

print(f"\n  Testing {len(promising)} features in pairs...", flush=True)
pair_results = []
for i, f1 in enumerate(promising):
    c1 = feats.columns.get_loc(f1)
    for f2 in promising[i+1:]:
        c2 = feats.columns.get_loc(f2)
        # Define Cormorant box as 25-75 percentile of Cormorant values
        cv1 = X[y == CORM, c1]; cv2 = X[y == CORM, c2]
        lo1, hi1 = np.percentile(cv1, [25, 75])
        lo2, hi2 = np.percentile(cv2, [25, 75])
        # How many Cormorants vs Gulls in this box?
        in_box = (X[:, c1] >= lo1) & (X[:, c1] <= hi1) & (X[:, c2] >= lo2) & (X[:, c2] <= hi2)
        n_corm_box = (in_box & (y == CORM)).sum()
        n_gull_box = (in_box & (y == GULL)).sum()
        n_other_box = (in_box & (y != CORM)).sum()
        if n_corm_box > 0:
            purity = n_corm_box / (n_corm_box + n_other_box)
            corm_recall = n_corm_box / 40
            if purity > 0.05:  # at least 5% purity
                pair_results.append((f1, f2, purity, corm_recall, n_corm_box, n_gull_box, n_other_box))

pair_results.sort(key=lambda x: -x[2])
print(f"\n  Top 20 feature pairs by Cormorant purity:")
print(f"  {'F1':>25s} x {'F2':>25s} {'Purity':>8s} {'Recall':>8s} {'Corm':>5s} {'Gull':>5s} {'Other':>6s}")
print(f"  {'-'*95}")
for f1, f2, pur, rec, nc, ng, no in pair_results[:20]:
    print(f"  {f1:>25s} x {f2:>25s} {pur:8.3f} {rec:8.3f} {nc:5d} {ng:5d} {no:6d}")

print("\n\n" + "=" * 90, flush=True)
print("PART 4: RAW TRAJECTORY PATTERNS", flush=True)
print("=" * 90, flush=True)

# Parse raw trajectories for Cormorants and nearest Gulls
# Compare the SHAPE of the multivariate time series
print("\n  Comparing raw trajectory SHAPES (speed, altitude, RCS profiles)...", flush=True)

# For each Cormorant, compute the "trajectory signature":
# Normalize to fixed length (20 points) and compare
def trajectory_signature(row, n_bins=20):
    pts = parse_ewkb_4d(row["trajectory"])
    times = parse_trajectory_time(row["trajectory_time"])
    alts = np.array([p[2] for p in pts])
    rcs = np.array([p[3] for p in pts])
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    n = len(pts)
    if n < 5:
        return np.zeros(n_bins * 3)
    # Compute speed
    dists = np.array([haversine(lons[j], lats[j], lons[j+1], lats[j+1]) for j in range(n-1)])
    dt = np.maximum(np.diff(times), 0.001)
    speeds = dists / dt

    # Normalize time to [0, 1] and bin
    t_norm = np.linspace(0, 1, n)
    t_bins = np.linspace(0, 1, n_bins + 1)

    # Bin altitude (normalized), speed (normalized), RCS (centered)
    alt_norm = (alts - np.mean(alts)) / max(np.std(alts), 0.001)
    rcs_norm = (rcs - np.mean(rcs)) / max(np.std(rcs), 0.001)
    spd_interp = np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(speeds)), speeds)
    spd_norm = (spd_interp - np.mean(spd_interp)) / max(np.std(spd_interp), 0.001)
    alt_interp = np.interp(np.linspace(0, 1, n_bins), t_norm, alt_norm)
    rcs_interp = np.interp(np.linspace(0, 1, n_bins), t_norm, rcs_norm)

    return np.concatenate([spd_norm, alt_interp, rcs_interp])

# Compute signatures for Cormorants and nearest Gulls
print("  Computing trajectory signatures...", flush=True)
corm_sigs = np.array([trajectory_signature(train.iloc[i]) for i in corm_idx])
# Get 100 nearest Gulls to any Cormorant
nn_gull = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn_gull.fit(X_scaled[gull_idx])
_, nearest_gull_idx = nn_gull.kneighbors(X_scaled[corm_idx])
nearest_gulls = np.unique(gull_idx[nearest_gull_idx.ravel()])[:100]
gull_sigs = np.array([trajectory_signature(train.iloc[i]) for i in nearest_gulls])

# Compare: distance between Cormorant signatures vs Gull signatures
from scipy.spatial.distance import cdist
corm_corm_dist = cdist(corm_sigs, corm_sigs, metric="euclidean")
corm_gull_dist = cdist(corm_sigs, gull_sigs, metric="euclidean")

# Average within-class vs between-class distance
np.fill_diagonal(corm_corm_dist, np.nan)
within_dist = np.nanmean(corm_corm_dist)
between_dist = np.mean(corm_gull_dist)
ratio = between_dist / max(within_dist, 0.001)

print(f"\n  Trajectory signature distances:")
print(f"    Cormorant-Cormorant (within-class): {within_dist:.3f}")
print(f"    Cormorant-Gull (between-class):     {between_dist:.3f}")
print(f"    Ratio (>1 = separable):             {ratio:.3f}")

# Also: do the trajectory signatures separate with a simple classifier?
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X_sig = np.vstack([corm_sigs, gull_sigs])
y_sig = np.concatenate([np.ones(len(corm_sigs)), np.zeros(len(gull_sigs))])
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
scores = cross_val_score(lr, X_sig, y_sig, cv=5, scoring="roc_auc")
print(f"    Logistic regression AUC on signatures: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

print("\n\n" + "=" * 90, flush=True)
print("PART 5: PCA/EMBEDDING — WHERE DO CORMORANTS LIVE?", flush=True)
print("=" * 90, flush=True)

# PCA on all features — where are Cormorants?
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print(f"\n  PCA explained variance: {pca.explained_variance_ratio_[:5]}")
print(f"  Cormorant centroids vs class centroids in PC1-PC5:")
corm_centroid = X_pca[y == CORM].mean(axis=0)
for c in range(len(CLASSES)):
    centroid = X_pca[y == c].mean(axis=0)
    dist_to_corm = np.linalg.norm(centroid[:5] - corm_centroid[:5])
    print(f"    {CLASSES[c]:15s}: dist_to_corm_centroid = {dist_to_corm:.3f}")

# Cormorant spread in PCA space
corm_std = X_pca[y == CORM].std(axis=0)
gull_std = X_pca[y == GULL].std(axis=0)
print(f"\n  Cormorant spread (std in PC space): {corm_std[:5]}")
print(f"  Gull spread (std in PC space):      {gull_std[:5]}")
print(f"  Corm/Gull spread ratio:             {corm_std[:5] / gull_std[:5]}")

# Are Cormorants a TIGHT cluster or scattered?
from scipy.spatial.distance import pdist
corm_pca = X_pca[y == CORM, :5]
gull_pca = X_pca[y == GULL, :5]
corm_internal = np.mean(pdist(corm_pca))
gull_internal = np.mean(pdist(gull_pca))
print(f"\n  Mean internal distance:")
print(f"    Cormorants: {corm_internal:.3f}")
print(f"    Gulls:      {gull_internal:.3f}")
print(f"    Ratio (Corm/Gull, <1 = tighter cluster): {corm_internal/gull_internal:.3f}")

# KNN accuracy in PCA space
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
# Binary: Cormorant vs Gull in PCA space
cg_mask = (y == CORM) | (y == GULL)
y_cg = (y[cg_mask] == CORM).astype(int)
X_cg = X_pca[cg_mask, :10]
from sklearn.model_selection import cross_val_score
knn_scores = cross_val_score(knn, X_cg, y_cg, cv=5, scoring="roc_auc")
print(f"\n  KNN (k=5) AUC in 10-dim PCA space (Corm vs Gull): {np.mean(knn_scores):.3f}")

print("\nDone.", flush=True)
