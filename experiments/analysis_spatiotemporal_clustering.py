"""
Spatiotemporal Clustering Analysis for Bird Radar Classification
================================================================
Quantify clustering signal in train and test data.
Determine if majority-vote smoothing within clusters can improve predictions.
"""
import sys
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_train, load_test, parse_ewkb_4d

# ── Helpers ──────────────────────────────────────────────────────────

def get_first_point(hex_str):
    """Extract first (lon, lat) from EWKB hex trajectory."""
    raw = bytes.fromhex(hex_str)
    offset = 5  # skip byte order + geom type
    if struct.unpack_from('<I', raw, 1)[0] & 0x20000000:
        offset += 4  # skip SRID
    offset += 4  # skip n_points
    lon, lat = struct.unpack_from('<2d', raw, offset)
    return lon, lat


def parse_timestamp(ts_str):
    """Parse timestamp string to pandas Timestamp."""
    return pd.to_datetime(ts_str)


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in meters."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Load data ────────────────────────────────────────────────────────

print("=" * 80)
print("SPATIOTEMPORAL CLUSTERING ANALYSIS")
print("=" * 80)

print("\nLoading data...")
train = load_train()
test = load_test()

# Load OOF and test predictions
oof_preds = np.load(ROOT / "oof_e79.npy")
test_preds = np.load(ROOT / "test_e79.npy")

print(f"Train: {len(train)} tracks, Test: {len(test)} tracks")
print(f"OOF preds shape: {oof_preds.shape}, Test preds shape: {test_preds.shape}")

# ── Parse positions and timestamps ───────────────────────────────────

print("\nParsing positions and timestamps...")

# Train positions and times
train_lons, train_lats = [], []
for hex_str in train['trajectory']:
    lon, lat = get_first_point(hex_str)
    train_lons.append(lon)
    train_lats.append(lat)
train['lon'] = train_lons
train['lat'] = train_lats
train['ts_start'] = pd.to_datetime(train['timestamp_start_radar_utc'])
train['ts_epoch'] = train['ts_start'].astype(np.int64) // 10**9  # seconds

# Test positions and times
test_lons, test_lats = [], []
for hex_str in test['trajectory']:
    lon, lat = get_first_point(hex_str)
    test_lons.append(lon)
    test_lats.append(lat)
test['lon'] = test_lons
test['lat'] = test_lats
test['ts_start'] = pd.to_datetime(test['timestamp_start_radar_utc'])
test['ts_epoch'] = test['ts_start'].astype(np.int64) // 10**9

# Encode labels
label_map = {c: i for i, c in enumerate(CLASSES)}
train['label'] = train['bird_group'].map(label_map)
train['oof_pred'] = oof_preds.argmax(axis=1)

print(f"Train time range: {train['ts_start'].min()} to {train['ts_start'].max()}")
print(f"Test time range:  {test['ts_start'].min()} to {test['ts_start'].max()}")
print(f"Train lon range:  [{train['lon'].min():.4f}, {train['lon'].max():.4f}]")
print(f"Train lat range:  [{train['lat'].min():.4f}, {train['lat'].max():.4f}]")
print(f"Test lon range:   [{test['lon'].min():.4f}, {test['lon'].max():.4f}]")
print(f"Test lat range:   [{test['lat'].min():.4f}, {test['lat'].max():.4f}]")

# ======================================================================
# 1. TEMPORAL CLUSTERING IN TRAIN DATA
# ======================================================================
print("\n" + "=" * 80)
print("1. TEMPORAL CLUSTERING IN TRAIN DATA")
print("=" * 80)

time_windows = [30, 60, 120, 300]
train_epochs = train['ts_epoch'].values
train_labels = train['label'].values

for window in time_windows:
    same_class_fracs = []
    n_with_neighbors = 0
    neighbor_counts = []

    for i in range(len(train)):
        # Find temporal neighbors (excluding self)
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= window)
        neighbors = np.where(mask)[0]

        neighbor_counts.append(len(neighbors))
        if len(neighbors) > 0:
            n_with_neighbors += 1
            same = np.sum(train_labels[neighbors] == train_labels[i])
            same_class_fracs.append(same / len(neighbors))

    nc = np.array(neighbor_counts)
    print(f"\n--- Window: {window}s ---")
    print(f"  Tracks with >= 1 neighbor: {n_with_neighbors}/{len(train)} ({100*n_with_neighbors/len(train):.1f}%)")
    print(f"  Neighbor count: mean={nc.mean():.1f}, median={np.median(nc):.0f}, "
          f"max={nc.max()}, p90={np.percentile(nc, 90):.0f}")
    if same_class_fracs:
        scf = np.array(same_class_fracs)
        print(f"  Same-class fraction: mean={scf.mean():.3f}, median={np.median(scf):.3f}")
        print(f"  Tracks with >50% same-class neighbors: {np.sum(scf > 0.5)}/{n_with_neighbors} ({100*np.sum(scf > 0.5)/max(n_with_neighbors,1):.1f}%)")
        print(f"  Tracks with 100% same-class neighbors: {np.sum(scf == 1.0)}/{n_with_neighbors} ({100*np.sum(scf == 1.0)/max(n_with_neighbors,1):.1f}%)")

# Per-class analysis at 60s window
print("\n--- Per-class temporal clustering (60s window) ---")
for ci, cname in enumerate(CLASSES):
    cls_mask = train_labels == ci
    cls_indices = np.where(cls_mask)[0]
    if len(cls_indices) == 0:
        continue

    neighbor_counts_cls = []
    same_class_fracs_cls = []

    for i in cls_indices:
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= 60)
        neighbors = np.where(mask)[0]
        neighbor_counts_cls.append(len(neighbors))
        if len(neighbors) > 0:
            same = np.sum(train_labels[neighbors] == ci)
            same_class_fracs_cls.append(same / len(neighbors))

    nc = np.array(neighbor_counts_cls)
    has_nb = len([x for x in neighbor_counts_cls if x > 0])
    if same_class_fracs_cls:
        scf = np.array(same_class_fracs_cls)
        print(f"  {cname:20s}: n={len(cls_indices):4d}, "
              f"has_neighbor={has_nb:4d} ({100*has_nb/len(cls_indices):5.1f}%), "
              f"mean_neighbors={nc.mean():5.1f}, "
              f"same_class_frac={scf.mean():.3f}")
    else:
        print(f"  {cname:20s}: n={len(cls_indices):4d}, has_neighbor={has_nb} (0%)")


# ======================================================================
# 2. CAN CLUSTERING RESCUE ERRORS?
# ======================================================================
print("\n" + "=" * 80)
print("2. CAN CLUSTERING RESCUE OOF ERRORS?")
print("=" * 80)

wrong_mask = train['oof_pred'] != train['label']
n_wrong = wrong_mask.sum()
print(f"Wrongly classified tracks: {n_wrong}/{len(train)} ({100*n_wrong/len(train):.1f}%)")

for window in [60, 120, 300]:
    rescuable = 0
    rescuable_by_majority = 0
    no_neighbors = 0

    wrong_indices = np.where(wrong_mask.values)[0]

    for i in wrong_indices:
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= window)
        neighbors = np.where(mask)[0]

        if len(neighbors) == 0:
            no_neighbors += 1
            continue

        # True labels of neighbors
        neighbor_labels = train_labels[neighbors]
        true_label = train_labels[i]

        # Is true class present among neighbors?
        if true_label in neighbor_labels:
            rescuable += 1

        # Would majority vote give correct answer?
        label_counts = Counter(neighbor_labels)
        majority_label = label_counts.most_common(1)[0][0]
        if majority_label == true_label:
            rescuable_by_majority += 1

    print(f"\n--- Window: {window}s ---")
    print(f"  Wrong tracks with no neighbors: {no_neighbors}/{n_wrong}")
    print(f"  True class present in neighbors: {rescuable}/{n_wrong} ({100*rescuable/n_wrong:.1f}%)")
    print(f"  Majority vote gives correct class: {rescuable_by_majority}/{n_wrong} ({100*rescuable_by_majority/n_wrong:.1f}%)")

# Detailed per-class rescue analysis at 60s
print("\n--- Per-class rescue analysis (60s window) ---")
for ci, cname in enumerate(CLASSES):
    cls_wrong = wrong_mask.values & (train_labels == ci)
    n_cls_wrong = cls_wrong.sum()
    if n_cls_wrong == 0:
        print(f"  {cname:20s}: 0 errors")
        continue

    rescuable = 0
    majority_correct = 0
    for i in np.where(cls_wrong)[0]:
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= 60)
        neighbors = np.where(mask)[0]
        if len(neighbors) == 0:
            continue
        neighbor_labels = train_labels[neighbors]
        if ci in neighbor_labels:
            rescuable += 1
        label_counts = Counter(neighbor_labels)
        if label_counts.most_common(1)[0][0] == ci:
            majority_correct += 1

    print(f"  {cname:20s}: {n_cls_wrong:3d} errors, "
          f"rescuable={rescuable:3d} ({100*rescuable/max(n_cls_wrong,1):5.1f}%), "
          f"majority_correct={majority_correct:3d} ({100*majority_correct/max(n_cls_wrong,1):5.1f}%)")


# ======================================================================
# 3. TEST DATA CLUSTERING
# ======================================================================
print("\n" + "=" * 80)
print("3. TEST DATA CLUSTERING")
print("=" * 80)

test_epochs = test['ts_epoch'].values
test_pred_labels = test_preds.argmax(axis=1)

for window in [60, 120]:
    neighbor_counts = []
    consistent_fracs = []

    for i in range(len(test)):
        dt = np.abs(test_epochs - test_epochs[i])
        mask = (dt > 0) & (dt <= window)
        neighbors = np.where(mask)[0]
        neighbor_counts.append(len(neighbors))

        if len(neighbors) > 0:
            same_pred = np.sum(test_pred_labels[neighbors] == test_pred_labels[i])
            consistent_fracs.append(same_pred / len(neighbors))

    nc = np.array(neighbor_counts)
    print(f"\n--- Test data, window: {window}s ---")
    has_nb = np.sum(nc > 0)
    print(f"  Tracks with >= 1 neighbor: {has_nb}/{len(test)} ({100*has_nb/len(test):.1f}%)")
    print(f"  Tracks with >= 3 neighbors: {np.sum(nc >= 3)}/{len(test)} ({100*np.sum(nc >= 3)/len(test):.1f}%)")
    print(f"  Neighbor count: mean={nc.mean():.1f}, median={np.median(nc):.0f}, max={nc.max()}")
    if consistent_fracs:
        cf = np.array(consistent_fracs)
        print(f"  Prediction consistency: mean={cf.mean():.3f}, median={np.median(cf):.3f}")
        print(f"  >50% consistent: {np.sum(cf > 0.5)}/{has_nb} ({100*np.sum(cf > 0.5)/max(has_nb,1):.1f}%)")


# ======================================================================
# 4. SPATIAL + SPATIOTEMPORAL CLUSTERING
# ======================================================================
print("\n" + "=" * 80)
print("4. SPATIAL + SPATIOTEMPORAL CLUSTERING")
print("=" * 80)

# Convert to approximate meters for KDTree (at ~53.4N latitude)
# 1 degree lon ~ 111320 * cos(53.4) ~ 66,300 m
# 1 degree lat ~ 111320 m
cos_lat = np.cos(np.radians(53.4))

# --- Train spatial ---
train_x = train['lon'].values * 111320 * cos_lat
train_y = train['lat'].values * 111320
train_coords = np.column_stack([train_x, train_y])
tree_train = cKDTree(train_coords)

for dist_m in [200, 500, 1000]:
    pairs = tree_train.query_pairs(r=dist_m)

    # Count neighbors per track
    nc = np.zeros(len(train), dtype=int)
    same_class = np.zeros(len(train), dtype=int)
    for i, j in pairs:
        nc[i] += 1
        nc[j] += 1
        if train_labels[i] == train_labels[j]:
            same_class[i] += 1
            same_class[j] += 1

    has_nb = np.sum(nc > 0)
    scf_vals = same_class[nc > 0] / nc[nc > 0]

    print(f"\n--- Train spatial, radius: {dist_m}m ---")
    print(f"  Tracks with >= 1 neighbor: {has_nb}/{len(train)} ({100*has_nb/len(train):.1f}%)")
    print(f"  Neighbor count: mean={nc.mean():.1f}, median={np.median(nc):.0f}, max={nc.max()}")
    if len(scf_vals) > 0:
        print(f"  Same-class fraction: mean={scf_vals.mean():.3f}")

# --- Spatiotemporal (combined) ---
print("\n--- SPATIOTEMPORAL: 500m AND 60s combined (TRAIN) ---")
spatial_pairs_500 = tree_train.query_pairs(r=500)

# Build spatial neighbor dict
spatial_neighbors = defaultdict(set)
for i, j in spatial_pairs_500:
    spatial_neighbors[i].add(j)
    spatial_neighbors[j].add(i)

st_neighbor_counts = []
st_same_class_fracs = []
for i in range(len(train)):
    # Spatiotemporal: within 500m AND within 60s
    st_neighbors = []
    for j in spatial_neighbors[i]:
        if abs(train_epochs[i] - train_epochs[j]) <= 60:
            st_neighbors.append(j)

    st_neighbor_counts.append(len(st_neighbors))
    if len(st_neighbors) > 0:
        same = sum(1 for j in st_neighbors if train_labels[j] == train_labels[i])
        st_same_class_fracs.append(same / len(st_neighbors))

st_nc = np.array(st_neighbor_counts)
has_st = np.sum(st_nc > 0)
print(f"  Tracks with >= 1 ST neighbor: {has_st}/{len(train)} ({100*has_st/len(train):.1f}%)")
print(f"  Tracks with >= 3 ST neighbors: {np.sum(st_nc >= 3)}/{len(train)} ({100*np.sum(st_nc >= 3)/len(train):.1f}%)")
if st_same_class_fracs:
    scf = np.array(st_same_class_fracs)
    print(f"  Same-class fraction: mean={scf.mean():.3f}, median={np.median(scf):.3f}")
    print(f"  100% same class: {np.sum(scf == 1.0)}/{has_st} ({100*np.sum(scf == 1.0)/max(has_st,1):.1f}%)")

# Per-class spatiotemporal analysis
print("\n  Per-class spatiotemporal (500m + 60s):")
for ci, cname in enumerate(CLASSES):
    cls_mask = train_labels == ci
    cls_indices = np.where(cls_mask)[0]
    if len(cls_indices) == 0:
        continue
    cls_st_counts = [st_neighbor_counts[i] for i in cls_indices]
    has_nb_cls = sum(1 for c in cls_st_counts if c > 0)
    cls_scf = []
    for i in cls_indices:
        if st_neighbor_counts[i] > 0:
            st_nb = [j for j in spatial_neighbors[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
            same = sum(1 for j in st_nb if train_labels[j] == ci)
            cls_scf.append(same / len(st_nb))

    mean_scf = np.mean(cls_scf) if cls_scf else 0
    print(f"    {cname:20s}: n={len(cls_indices):4d}, "
          f"has_ST_neighbor={has_nb_cls:4d} ({100*has_nb_cls/len(cls_indices):5.1f}%), "
          f"mean_same_class={mean_scf:.3f}")

# --- Test spatiotemporal ---
print("\n--- SPATIOTEMPORAL: 500m AND 60s combined (TEST) ---")
test_x = test['lon'].values * 111320 * cos_lat
test_y = test['lat'].values * 111320
test_coords = np.column_stack([test_x, test_y])
tree_test = cKDTree(test_coords)

spatial_pairs_test = tree_test.query_pairs(r=500)
spatial_neighbors_test = defaultdict(set)
for i, j in spatial_pairs_test:
    spatial_neighbors_test[i].add(j)
    spatial_neighbors_test[j].add(i)

test_st_counts = []
test_consistent_fracs = []
for i in range(len(test)):
    st_neighbors = [j for j in spatial_neighbors_test[i] if abs(test_epochs[i] - test_epochs[j]) <= 60]
    test_st_counts.append(len(st_neighbors))
    if len(st_neighbors) > 0:
        same_pred = sum(1 for j in st_neighbors if test_pred_labels[j] == test_pred_labels[i])
        test_consistent_fracs.append(same_pred / len(st_neighbors))

tst_nc = np.array(test_st_counts)
has_tst = np.sum(tst_nc > 0)
print(f"  Tracks with >= 1 ST neighbor: {has_tst}/{len(test)} ({100*has_tst/len(test):.1f}%)")
print(f"  Tracks with >= 3 ST neighbors: {np.sum(tst_nc >= 3)}/{len(test)} ({100*np.sum(tst_nc >= 3)/len(test):.1f}%)")
if test_consistent_fracs:
    cf = np.array(test_consistent_fracs)
    print(f"  Prediction consistency: mean={cf.mean():.3f}")


# ======================================================================
# 5. CLASS-SPECIFIC CLUSTERING PATTERNS
# ======================================================================
print("\n" + "=" * 80)
print("5. CLASS-SPECIFIC CLUSTERING PATTERNS")
print("=" * 80)

print("\nCluster size distribution per class (temporal 60s):")
print(f"{'Class':20s} {'P(0)':>6s} {'P(1-2)':>6s} {'P(3-5)':>6s} {'P(6-10)':>7s} {'P(11+)':>6s} {'Mean':>6s}")
print("-" * 60)

for ci, cname in enumerate(CLASSES):
    cls_indices = np.where(train_labels == ci)[0]
    cls_nb_counts = []
    for i in cls_indices:
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= 60)
        cls_nb_counts.append(mask.sum())

    nc = np.array(cls_nb_counts)
    p0 = np.mean(nc == 0)
    p12 = np.mean((nc >= 1) & (nc <= 2))
    p35 = np.mean((nc >= 3) & (nc <= 5))
    p610 = np.mean((nc >= 6) & (nc <= 10))
    p11 = np.mean(nc >= 11)
    print(f"{cname:20s} {p0:6.1%} {p12:6.1%} {p35:6.1%} {p610:7.1%} {p11:6.1%} {nc.mean():6.1f}")


# ======================================================================
# 6. SIMULATED MAJORITY-VOTE SMOOTHING IMPACT
# ======================================================================
print("\n" + "=" * 80)
print("6. SIMULATED MAJORITY-VOTE SMOOTHING (TRAIN OOF)")
print("=" * 80)

from sklearn.metrics import average_precision_score

def compute_macro_map(y_true, y_proba, classes=CLASSES):
    """Compute macro mAP."""
    n_classes = len(classes)
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

# Baseline
baseline_map, baseline_aps = compute_macro_map(train_labels, oof_preds)
print(f"\nBaseline OOF macro mAP: {baseline_map:.4f}")

# Strategy: For tracks with neighbors, average the probability predictions
for window in [60, 120]:
    for min_neighbors in [1, 3]:
        smoothed = oof_preds.copy()
        n_smoothed = 0

        for i in range(len(train)):
            dt = np.abs(train_epochs - train_epochs[i])
            mask = (dt > 0) & (dt <= window)
            neighbors = np.where(mask)[0]

            if len(neighbors) >= min_neighbors:
                # Average own prediction with neighbors' predictions
                all_preds = np.vstack([oof_preds[i:i+1]] + [oof_preds[j:j+1] for j in neighbors])
                smoothed[i] = all_preds.mean(axis=0)
                n_smoothed += 1

        smooth_map, smooth_aps = compute_macro_map(train_labels, smoothed)
        delta = smooth_map - baseline_map
        print(f"\n  Window={window}s, min_neighbors={min_neighbors}: "
              f"mAP={smooth_map:.4f} (delta={delta:+.4f}), "
              f"smoothed {n_smoothed}/{len(train)} tracks")

        # Show per-class deltas
        for ci, cname in enumerate(CLASSES):
            d = smooth_aps[ci] - baseline_aps[ci]
            if abs(d) > 0.005:
                print(f"    {cname}: {baseline_aps[ci]:.3f} -> {smooth_aps[ci]:.3f} ({d:+.3f})")

# Strategy 2: Weighted average (own prediction weighted 2x)
print("\n--- Weighted smoothing (self-weight=2x) ---")
for window in [60, 120]:
    smoothed = oof_preds.copy()
    n_smoothed = 0

    for i in range(len(train)):
        dt = np.abs(train_epochs - train_epochs[i])
        mask = (dt > 0) & (dt <= window)
        neighbors = np.where(mask)[0]

        if len(neighbors) >= 1:
            # Self gets 2x weight
            neighbor_preds = oof_preds[neighbors].mean(axis=0)
            smoothed[i] = 0.67 * oof_preds[i] + 0.33 * neighbor_preds
            n_smoothed += 1

    smooth_map, smooth_aps = compute_macro_map(train_labels, smoothed)
    delta = smooth_map - baseline_map
    print(f"  Window={window}s: mAP={smooth_map:.4f} (delta={delta:+.4f})")

# Strategy 3: Spatiotemporal smoothing (500m + 60s)
print("\n--- Spatiotemporal smoothing (500m + 60s) ---")
smoothed_st = oof_preds.copy()
n_smoothed_st = 0

for i in range(len(train)):
    st_neighbors = [j for j in spatial_neighbors[i] if abs(train_epochs[i] - train_epochs[j]) <= 60]
    if len(st_neighbors) >= 1:
        neighbor_preds = oof_preds[np.array(st_neighbors)].mean(axis=0)
        smoothed_st[i] = 0.67 * oof_preds[i] + 0.33 * neighbor_preds
        n_smoothed_st += 1

smooth_st_map, smooth_st_aps = compute_macro_map(train_labels, smoothed_st)
delta_st = smooth_st_map - baseline_map
print(f"  mAP={smooth_st_map:.4f} (delta={delta_st:+.4f}), smoothed {n_smoothed_st}/{len(train)} tracks")
for ci, cname in enumerate(CLASSES):
    d = smooth_st_aps[ci] - baseline_aps[ci]
    if abs(d) > 0.005:
        print(f"    {cname}: {baseline_aps[ci]:.3f} -> {smooth_st_aps[ci]:.3f} ({d:+.3f})")


# Strategy 4: Label propagation - use neighbor TRUE labels as soft targets
print("\n--- Oracle: neighbor label distribution as feature (60s temporal) ---")
# This shows the CEILING - if we had true labels for neighbors
n_correct_before = np.sum(oof_preds.argmax(axis=1) == train_labels)
oracle_preds = oof_preds.copy()
for i in range(len(train)):
    dt = np.abs(train_epochs - train_epochs[i])
    mask = (dt > 0) & (dt <= 60)
    neighbors = np.where(mask)[0]
    if len(neighbors) >= 2:
        # Create soft label from neighbor true labels
        neighbor_dist = np.zeros(len(CLASSES))
        for j in neighbors:
            neighbor_dist[train_labels[j]] += 1
        neighbor_dist /= neighbor_dist.sum()
        # Blend with model prediction
        oracle_preds[i] = 0.5 * oof_preds[i] + 0.5 * neighbor_dist

oracle_map, oracle_aps = compute_macro_map(train_labels, oracle_preds)
print(f"  Oracle mAP: {oracle_map:.4f} (delta={oracle_map - baseline_map:+.4f})")
n_correct_after = np.sum(oracle_preds.argmax(axis=1) == train_labels)
print(f"  Accuracy: {n_correct_before}/{len(train)} -> {n_correct_after}/{len(train)}")


# ======================================================================
# 7. CROSS-SET CLUSTERING: TRAIN-TEST NEIGHBORS
# ======================================================================
print("\n" + "=" * 80)
print("7. TRAIN-TEST CROSS-SET CLUSTERING")
print("=" * 80)

# For each test track, find train neighbors within 60s and 500m
all_epochs = np.concatenate([train_epochs, test_epochs])
all_coords = np.vstack([train_coords, test_coords])
n_train = len(train)

# Build combined KDTree
tree_all = cKDTree(all_coords)
all_spatial_pairs = tree_all.query_pairs(r=500)

# Build neighbor dict (only train-test pairs)
train_test_neighbors = defaultdict(list)  # test_idx -> list of train_idx
for i, j in all_spatial_pairs:
    # We want pairs where one is train and one is test
    if i < n_train and j >= n_train:
        test_idx = j - n_train
        if abs(all_epochs[i] - all_epochs[j]) <= 60:
            train_test_neighbors[test_idx].append(i)
    elif j < n_train and i >= n_train:
        test_idx = i - n_train
        if abs(all_epochs[i] - all_epochs[j]) <= 60:
            train_test_neighbors[test_idx].append(j)

n_test_with_train_nb = sum(1 for v in train_test_neighbors.values() if len(v) > 0)
print(f"  Test tracks with train ST neighbors (500m+60s): {n_test_with_train_nb}/{len(test)} ({100*n_test_with_train_nb/len(test):.1f}%)")

# What classes do these train neighbors provide?
if n_test_with_train_nb > 0:
    print("\n  Class distribution of train neighbors for test tracks:")
    all_train_nb_labels = []
    for test_idx, train_indices in train_test_neighbors.items():
        for ti in train_indices:
            all_train_nb_labels.append(train_labels[ti])

    if all_train_nb_labels:
        lc = Counter(all_train_nb_labels)
        for ci, cname in enumerate(CLASSES):
            cnt = lc.get(ci, 0)
            print(f"    {cname}: {cnt}")

# Temporal-only cross-set
print("\n  --- Temporal only (60s, no spatial constraint) ---")
n_cross_temporal = 0
cross_neighbor_counts = []
for i in range(len(test)):
    dt = np.abs(train_epochs - test_epochs[i])
    mask = dt <= 60
    n_nb = mask.sum()
    cross_neighbor_counts.append(n_nb)
    if n_nb > 0:
        n_cross_temporal += 1

cnc = np.array(cross_neighbor_counts)
print(f"  Test tracks with train temporal neighbors: {n_cross_temporal}/{len(test)} ({100*n_cross_temporal/len(test):.1f}%)")
print(f"  Neighbor count: mean={cnc.mean():.1f}, median={np.median(cnc):.0f}, max={cnc.max()}")

# Strategy: Use train neighbor labels to adjust test predictions
print("\n--- Cross-set label propagation (test using train neighbors, 60s temporal) ---")
test_smoothed = test_preds.copy()
n_test_smoothed = 0
for i in range(len(test)):
    dt = np.abs(train_epochs - test_epochs[i])
    mask = dt <= 60
    neighbors = np.where(mask)[0]
    if len(neighbors) >= 2:
        neighbor_dist = np.zeros(len(CLASSES))
        for j in neighbors:
            neighbor_dist[train_labels[j]] += 1
        neighbor_dist /= neighbor_dist.sum()
        test_smoothed[i] = 0.7 * test_preds[i] + 0.3 * neighbor_dist
        n_test_smoothed += 1

n_changed = np.sum(test_smoothed.argmax(axis=1) != test_preds.argmax(axis=1))
print(f"  Test tracks smoothed: {n_test_smoothed}/{len(test)}")
print(f"  Test predictions changed: {n_changed}/{len(test)} ({100*n_changed/len(test):.1f}%)")

# Which classes gain/lose?
old_pred_dist = Counter(test_preds.argmax(axis=1))
new_pred_dist = Counter(test_smoothed.argmax(axis=1))
print("\n  Class prediction changes:")
for ci, cname in enumerate(CLASSES):
    old_c = old_pred_dist.get(ci, 0)
    new_c = new_pred_dist.get(ci, 0)
    if old_c != new_c:
        print(f"    {cname}: {old_c} -> {new_c} ({new_c - old_c:+d})")


# ======================================================================
# 8. FEATURE ENGINEERING: CLUSTER-BASED FEATURES
# ======================================================================
print("\n" + "=" * 80)
print("8. POTENTIAL CLUSTER-BASED FEATURES")
print("=" * 80)

print("\nFeatures that could be extracted from spatiotemporal clusters:")
print("  1. n_neighbors_60s — count of tracks within 60s")
print("  2. n_neighbors_st — count of tracks within 500m AND 60s")
print("  3. cluster_size — connected component size in ST graph")
print("  4. cluster_mean_speed — average airspeed of ST neighbors")
print("  5. cluster_mean_alt — average altitude of ST neighbors")
print("  6. cluster_rcs_std — RCS variability across cluster (flock vs solo)")
print("  7. cluster_size_homogeneity — fraction of neighbors with same radar_bird_size")

# Compute some of these for train
print("\n--- Correlation of n_neighbors_60s with class ---")
nb60 = np.array(st_neighbor_counts)  # reusing from spatiotemporal
for ci, cname in enumerate(CLASSES):
    cls_vals = nb60[train_labels == ci]
    print(f"  {cname:20s}: mean={cls_vals.mean():5.1f}, "
          f"median={np.median(cls_vals):4.0f}, "
          f"max={cls_vals.max():3d}, "
          f"pct_with_nb={100*np.mean(cls_vals > 0):5.1f}%")


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
