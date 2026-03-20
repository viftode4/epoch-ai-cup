"""
Spatiotemporal smoothing parameter sweep.
The main analysis found ST smoothing (500m+60s) gives +0.0039 mAP.
Sweep over parameters to find optimal configuration.
"""
import sys
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_train, parse_ewkb_4d

def get_first_point(hex_str):
    raw = bytes.fromhex(hex_str)
    offset = 5
    if struct.unpack_from('<I', raw, 1)[0] & 0x20000000:
        offset += 4
    offset += 4
    lon, lat = struct.unpack_from('<2d', raw, offset)
    return lon, lat

# Load data
train = load_train()
oof_preds = np.load(ROOT / "oof_e79.npy")

label_map = {c: i for i, c in enumerate(CLASSES)}
train['label'] = train['bird_group'].map(label_map)
train_labels = train['label'].values

# Parse positions and times
lons, lats = zip(*[get_first_point(h) for h in train['trajectory']])
train['lon'] = list(lons)
train['lat'] = list(lats)
train['ts_epoch'] = pd.to_datetime(train['timestamp_start_radar_utc']).astype(np.int64) // 10**9
train_epochs = train['ts_epoch'].values

cos_lat = np.cos(np.radians(53.4))
train_x = train['lon'].values * 111320 * cos_lat
train_y = train['lat'].values * 111320
train_coords = np.column_stack([train_x, train_y])

def compute_macro_map(y_true, y_proba):
    n_classes = len(CLASSES)
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    aps = []
    for c in range(n_classes):
        if y_onehot[:, c].sum() > 0:
            aps.append(average_precision_score(y_onehot[:, c], y_proba[:, c]))
        else:
            aps.append(0)
    return np.mean(aps), aps

baseline_map, baseline_aps = compute_macro_map(train_labels, oof_preds)
print(f"Baseline OOF macro mAP: {baseline_map:.4f}")
print(f"Per-class: {', '.join(f'{c[:4]}={a:.3f}' for c, a in zip(CLASSES, baseline_aps))}")

# ── Sweep ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SPATIOTEMPORAL SMOOTHING PARAMETER SWEEP")
print("=" * 80)

radii = [200, 300, 500, 750]
time_windows = [30, 60, 90, 120]
self_weights = [0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95]
min_neighbors_list = [1, 2, 3]

# Pre-compute spatial trees at different radii
best_map = baseline_map
best_params = None

for radius in radii:
    tree = cKDTree(train_coords)
    pairs = tree.query_pairs(r=radius)
    spatial_nb = defaultdict(set)
    for i, j in pairs:
        spatial_nb[i].add(j)
        spatial_nb[j].add(i)

    for tw in time_windows:
        # Pre-compute ST neighbors for this radius+time combo
        st_neighbors = {}
        for i in range(len(train)):
            st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= tw]
            if st_nb:
                st_neighbors[i] = np.array(st_nb)

        n_with_nb = len(st_neighbors)

        for min_nb in min_neighbors_list:
            for sw in self_weights:
                smoothed = oof_preds.copy()
                n_smoothed = 0
                for i, nb in st_neighbors.items():
                    if len(nb) >= min_nb:
                        neighbor_mean = oof_preds[nb].mean(axis=0)
                        smoothed[i] = sw * oof_preds[i] + (1 - sw) * neighbor_mean
                        n_smoothed += 1

                if n_smoothed == 0:
                    continue

                m, aps = compute_macro_map(train_labels, smoothed)
                delta = m - baseline_map

                if m > best_map:
                    best_map = m
                    best_params = (radius, tw, min_nb, sw, n_smoothed)

                if delta > 0.001:
                    print(f"  r={radius:4d}m, t={tw:3d}s, min_nb={min_nb}, self_w={sw:.2f}: "
                          f"mAP={m:.4f} ({delta:+.4f}), smoothed={n_smoothed}")

print(f"\nBest: r={best_params[0]}m, t={best_params[1]}s, min_nb={best_params[2]}, "
      f"self_w={best_params[3]:.2f} -> mAP={best_map:.4f} (+{best_map - baseline_map:.4f}), "
      f"smoothed={best_params[4]}")

# ── Detailed analysis of best config ─────────────────────────────────
print("\n" + "=" * 80)
print("DETAILED ANALYSIS OF BEST CONFIG")
print("=" * 80)

r, tw, mn, sw, _ = best_params
tree = cKDTree(train_coords)
pairs = tree.query_pairs(r=r)
spatial_nb = defaultdict(set)
for i, j in pairs:
    spatial_nb[i].add(j)
    spatial_nb[j].add(i)

smoothed_best = oof_preds.copy()
for i in range(len(train)):
    st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= tw]
    if len(st_nb) >= mn:
        neighbor_mean = oof_preds[np.array(st_nb)].mean(axis=0)
        smoothed_best[i] = sw * oof_preds[i] + (1 - sw) * neighbor_mean

_, best_aps = compute_macro_map(train_labels, smoothed_best)
print(f"\nPer-class changes:")
for ci, cname in enumerate(CLASSES):
    d = best_aps[ci] - baseline_aps[ci]
    print(f"  {cname:20s}: {baseline_aps[ci]:.4f} -> {best_aps[ci]:.4f} ({d:+.4f})")

# Show how many hard-class predictions change
old_hard = oof_preds.argmax(axis=1)
new_hard = smoothed_best.argmax(axis=1)
changed = old_hard != new_hard
print(f"\nHard predictions changed: {changed.sum()}/{len(train)}")
print(f"  Correct -> Wrong: {np.sum(changed & (old_hard == train_labels))}")
print(f"  Wrong -> Correct: {np.sum(changed & (new_hard == train_labels))}")
print(f"  Wrong -> Wrong:   {np.sum(changed & (old_hard != train_labels) & (new_hard != train_labels))}")

# ── Cluster feature importance ───────────────────────────────────────
print("\n" + "=" * 80)
print("CLUSTER-BASED FEATURES FOR MODEL TRAINING")
print("=" * 80)

# The n_neighbors_st feature is available at test time (no label leakage)
# Compute correlation with class
print("\nST neighbor count as feature (no leakage - uses only timestamps + positions):")

# Also compute features from neighbors' non-label attributes
# (airspeed, altitude, radar_bird_size - all available at test time)
airspeeds = train['airspeed'].values
min_zs = train['min_z'].values
max_zs = train['max_z'].values

size_map = {"Small bird": 1, "Medium bird": 2, "Large bird": 3, "Flock": 4}
sizes = train['radar_bird_size'].map(size_map).fillna(0).values

print("\nCluster aggregate features (500m + 60s, using neighbor attributes):")
print(f"{'Feature':40s} {'Unique_per_class':>16s}")

# For each track, compute cluster features
cluster_feats = np.zeros((len(train), 8))
for i in range(len(train)):
    st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
    n = len(st_nb)
    cluster_feats[i, 0] = n  # n_neighbors
    if n > 0:
        nb = np.array(st_nb)
        cluster_feats[i, 1] = airspeeds[nb].mean()  # mean airspeed of cluster
        cluster_feats[i, 2] = airspeeds[nb].std()    # airspeed variability
        cluster_feats[i, 3] = min_zs[nb].mean()      # mean min_z
        cluster_feats[i, 4] = max_zs[nb].mean()      # mean max_z
        cluster_feats[i, 5] = sizes[nb].mean()        # mean radar_bird_size
        cluster_feats[i, 6] = (sizes[nb] == sizes[i]).mean()  # size homogeneity
        cluster_feats[i, 7] = np.abs(airspeeds[nb] - airspeeds[i]).mean()  # speed diff from neighbors

feat_names = ['n_neighbors', 'cluster_mean_speed', 'cluster_speed_std',
              'cluster_mean_min_z', 'cluster_mean_max_z', 'cluster_mean_size',
              'cluster_size_homogeneity', 'cluster_speed_diff']

for fi, fn in enumerate(feat_names):
    # Per-class means
    means = []
    for ci in range(len(CLASSES)):
        mask = train_labels == ci
        means.append(cluster_feats[mask, fi].mean())
    cv = np.std(means) / max(np.mean(means), 0.001)
    print(f"  {fn:40s} CV={cv:.3f}  range=[{min(means):.2f}, {max(means):.2f}]")

# ── Same-class smoothing with NEIGHBOR PREDICTIONS (not labels) ──────
print("\n" + "=" * 80)
print("NEIGHBOR PREDICTION CONSISTENCY CHECK")
print("=" * 80)

# For tracks where model is uncertain, check if neighbors agree
max_probs = oof_preds.max(axis=1)
uncertain = max_probs < 0.5
print(f"Uncertain tracks (max_prob < 0.5): {uncertain.sum()}/{len(train)}")

# Among uncertain tracks, how often do neighbors' majority prediction match true label?
correct_by_nb = 0
total_uncertain_with_nb = 0
for i in np.where(uncertain)[0]:
    st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
    if len(st_nb) >= 2:
        total_uncertain_with_nb += 1
        nb_preds = oof_preds[np.array(st_nb)]
        ensemble_pred = nb_preds.mean(axis=0).argmax()
        if ensemble_pred == train_labels[i]:
            correct_by_nb += 1

print(f"Uncertain tracks with 2+ ST neighbors: {total_uncertain_with_nb}")
if total_uncertain_with_nb > 0:
    print(f"Neighbor ensemble gives correct class: {correct_by_nb}/{total_uncertain_with_nb} ({100*correct_by_nb/total_uncertain_with_nb:.1f}%)")

# Compare: own prediction correct rate for uncertain tracks
own_correct = 0
for i in np.where(uncertain)[0]:
    if oof_preds[i].argmax() == train_labels[i]:
        own_correct += 1
print(f"Own prediction correct (uncertain): {own_correct}/{uncertain.sum()} ({100*own_correct/uncertain.sum():.1f}%)")

# ── Confidence-gated smoothing ───────────────────────────────────────
print("\n" + "=" * 80)
print("CONFIDENCE-GATED SPATIOTEMPORAL SMOOTHING")
print("=" * 80)

# Only smooth when model is uncertain AND neighbors are confident
confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for conf_thresh in confidence_thresholds:
    smoothed_gated = oof_preds.copy()
    n_gated = 0

    for i in range(len(train)):
        if oof_preds[i].max() > conf_thresh:
            continue  # confident, skip

        st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
        if len(st_nb) < 2:
            continue

        nb_preds = oof_preds[np.array(st_nb)]
        nb_mean = nb_preds.mean(axis=0)

        if nb_mean.max() > 0.5:  # neighbors are reasonably confident
            smoothed_gated[i] = 0.5 * oof_preds[i] + 0.5 * nb_mean
            n_gated += 1

    m, aps = compute_macro_map(train_labels, smoothed_gated)
    delta = m - baseline_map
    print(f"  conf_thresh={conf_thresh:.1f}: mAP={m:.4f} ({delta:+.4f}), gated={n_gated}")

# ── BEST: Only smooth uncertain + use neighbor confidence weights ────
print("\n--- Weighted by neighbor confidence ---")
for conf_thresh in [0.4, 0.5, 0.6]:
    for nb_weight in [0.3, 0.4, 0.5]:
        smoothed_gated = oof_preds.copy()
        n_gated = 0

        for i in range(len(train)):
            if oof_preds[i].max() > conf_thresh:
                continue

            st_nb = [j for j in spatial_nb[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
            if len(st_nb) < 2:
                continue

            # Weight each neighbor by their max confidence
            nb_arr = np.array(st_nb)
            nb_preds = oof_preds[nb_arr]
            nb_confs = nb_preds.max(axis=1)
            weights = nb_confs / nb_confs.sum()
            nb_weighted = (nb_preds * weights[:, None]).sum(axis=0)

            smoothed_gated[i] = (1 - nb_weight) * oof_preds[i] + nb_weight * nb_weighted
            n_gated += 1

        m, _ = compute_macro_map(train_labels, smoothed_gated)
        delta = m - baseline_map
        if abs(delta) > 0.0005:
            print(f"    conf={conf_thresh}, nb_w={nb_weight}: mAP={m:.4f} ({delta:+.4f}), n={n_gated}")

print("\nDone.")
