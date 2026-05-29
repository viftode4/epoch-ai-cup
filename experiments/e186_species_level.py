"""E186: Species-level prediction aggregated to bird groups.

Trains LightGBM on bird_species (68 classes, rare merged to parent group),
predicts species probabilities, then sums to 9 group probabilities.

Key hypothesis: species-level training gives the model finer-grained decision
boundaries. Cormorants especially benefit since all 40 are Great Cormorant
(clean single-species target vs mixed groups like Gulls with 10 species).

Evaluation: LOMO (4-fold leave-one-month-out) as primary metric.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map

# ── Load data ────────────────────────────────────────────────────
train_df = load_train()
test_df = load_test()

y_group = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
TRAIN_MONTHS = [1, 4, 9, 10]
N = len(y_group)
CORM_IDX = CLASSES.index("Cormorants")

# Load cached v3 features (327 features)
with open(ROOT / "data" / "_cached_train_features_v3.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(ROOT / "data" / "_cached_test_features_v3.pkl", "rb") as f:
    X_test = pickle.load(f)

# Clean inf/nan
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Features: {X_train.shape[1]} columns")
print(f"Train: {N}, Test: {len(X_test)}")

# ── Build species labels with merging ──────────────────────────────
species = train_df["bird_species"].values.copy()
species_counts = pd.Series(species).value_counts()

# Species -> group mapping
species_to_group = dict(zip(train_df["bird_species"], train_df["bird_group"]))

# Merge rare species (< MIN_COUNT) into their parent group name
MIN_COUNT = 5
merged_species = []
for sp in species:
    if species_counts[sp] < MIN_COUNT:
        # Use group name as merged label
        group = species_to_group[sp]
        merged_species.append(f"_merged_{group}")
    else:
        merged_species.append(sp)

merged_species = np.array(merged_species)
unique_species = sorted(set(merged_species))
n_species = len(unique_species)
print(f"\nSpecies labels: {n_species} (after merging rare < {MIN_COUNT})")

# Create species -> group mapping for aggregation
species_label_to_group_idx = {}
for sp_label in unique_species:
    if sp_label.startswith("_merged_"):
        group_name = sp_label.replace("_merged_", "")
        species_label_to_group_idx[sp_label] = CLASSES.index(group_name)
    else:
        # Find this species' group
        mask = train_df["bird_species"] == sp_label
        if mask.sum() > 0:
            group_name = train_df.loc[mask, "bird_group"].iloc[0]
            species_label_to_group_idx[sp_label] = CLASSES.index(group_name)
        else:
            # Fallback — shouldn't happen
            species_label_to_group_idx[sp_label] = 0

# Species index encoding
sp_to_idx = {sp: i for i, sp in enumerate(unique_species)}
y_species = np.array([sp_to_idx[sp] for sp in merged_species])

# Build aggregation matrix: (n_species, 9) — species_probs @ agg_matrix = group_probs
agg_matrix = np.zeros((n_species, 9))
for sp_label, sp_idx in sp_to_idx.items():
    group_idx = species_label_to_group_idx[sp_label]
    agg_matrix[sp_idx, group_idx] = 1.0

print(f"\nAggregation matrix shape: {agg_matrix.shape}")
print("Species per group:")
for g_idx, g_name in enumerate(CLASSES):
    sp_in_group = [sp for sp, gi in species_label_to_group_idx.items() if gi == g_idx]
    print(f"  {g_name:15s}: {len(sp_in_group)} species ({', '.join(sp_in_group[:5])}{'...' if len(sp_in_group) > 5 else ''})")

# ── LOMO evaluation function ──────────────────────────────────────
def lomo_map(oof_preds):
    fold_maps = []
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        mAP, _ = compute_map(y_group[mask], oof_preds[mask])
        fold_maps.append(mAP)
    return np.mean(fold_maps)

def lomo_map_perclass(oof_preds):
    per_class = {c: [] for c in CLASSES}
    for m in TRAIN_MONTHS:
        mask = months == m
        if mask.sum() == 0:
            continue
        _, pc = compute_map(y_group[mask], oof_preds[mask])
        for c in CLASSES:
            per_class[c].append(pc[c])
    return {c: np.mean(v) for c, v in per_class.items()}

# ── LightGBM parameters ──────────────────────────────────────────
lgb_params = {
    "objective": "multiclass",
    "num_class": n_species,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

N_ROUNDS = 2000
EARLY_STOP = 50

# ── Approach 1: LOMO species-level ────────────────────────────────
print("\n" + "=" * 70)
print("  APPROACH 1: LOMO SPECIES-LEVEL PREDICTION")
print("=" * 70)

oof_species = np.zeros((N, n_species))
oof_group = np.zeros((N, 9))

for held_month in TRAIN_MONTHS:
    val_mask = months == held_month
    tr_mask = ~val_mask

    X_tr = X_train[tr_mask]
    X_va = X_train[val_mask]
    y_tr = y_species[tr_mask]
    y_va = y_species[val_mask]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )

    # Predict species probabilities
    sp_probs = model.predict(X_va)  # (n_val, n_species)
    oof_species[val_mask] = sp_probs

    # Aggregate to group probabilities
    group_probs = sp_probs @ agg_matrix  # (n_val, 9)
    oof_group[val_mask] = group_probs

    # Check this fold
    mAP, pc = compute_map(y_group[val_mask], group_probs)
    corm_ap = pc.get("Cormorants", 0)
    print(f"  Month {held_month}: n={val_mask.sum()}, mAP={mAP:.4f}, Corm_AP={corm_ap:.4f}, "
          f"n_rounds={model.best_iteration}")

lomo_species = lomo_map(oof_group)
pc_species = lomo_map_perclass(oof_group)
print(f"\n  Species-level LOMO: {lomo_species:.4f}")
print(f"  Per-class APs:")
for c in CLASSES:
    marker = " <-- weak" if pc_species[c] < 0.5 else ""
    print(f"    {c:15s}: {pc_species[c]:.4f}{marker}")

# ── Approach 2: Direct group-level baseline (for comparison) ──────
print("\n" + "=" * 70)
print("  APPROACH 2: DIRECT GROUP-LEVEL BASELINE")
print("=" * 70)

lgb_params_group = lgb_params.copy()
lgb_params_group["num_class"] = 9

oof_direct = np.zeros((N, 9))

for held_month in TRAIN_MONTHS:
    val_mask = months == held_month
    tr_mask = ~val_mask

    X_tr = X_train[tr_mask]
    X_va = X_train[val_mask]
    y_tr = y_group[tr_mask]
    y_va = y_group[val_mask]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        lgb_params_group,
        dtrain,
        num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )

    preds = model.predict(X_va)
    oof_direct[val_mask] = preds

    mAP, pc = compute_map(y_group[val_mask], preds)
    corm_ap = pc.get("Cormorants", 0)
    print(f"  Month {held_month}: n={val_mask.sum()}, mAP={mAP:.4f}, Corm_AP={corm_ap:.4f}, "
          f"n_rounds={model.best_iteration}")

lomo_direct = lomo_map(oof_direct)
pc_direct = lomo_map_perclass(oof_direct)
print(f"\n  Direct group LOMO: {lomo_direct:.4f}")
print(f"  Per-class APs:")
for c in CLASSES:
    marker = " <-- weak" if pc_direct[c] < 0.5 else ""
    print(f"    {c:15s}: {pc_direct[c]:.4f}{marker}")

# ── Approach 3: Species-level with different merge thresholds ─────
print("\n" + "=" * 70)
print("  APPROACH 3: VARY MERGE THRESHOLD")
print("=" * 70)

for min_count in [3, 10, 15, 20]:
    merged = []
    for sp in species:
        if species_counts[sp] < min_count:
            group = species_to_group[sp]
            merged.append(f"_merged_{group}")
        else:
            merged.append(sp)
    merged = np.array(merged)
    uniq = sorted(set(merged))
    n_sp = len(uniq)
    sp2i = {sp: i for i, sp in enumerate(uniq)}
    y_sp = np.array([sp2i[sp] for sp in merged])

    # Aggregation matrix
    agg = np.zeros((n_sp, 9))
    for sp_label, sp_i in sp2i.items():
        if sp_label.startswith("_merged_"):
            g = CLASSES.index(sp_label.replace("_merged_", ""))
        else:
            mask = train_df["bird_species"] == sp_label
            g = CLASSES.index(train_df.loc[mask, "bird_group"].iloc[0])
        agg[sp_i, g] = 1.0

    params = lgb_params.copy()
    params["num_class"] = n_sp

    oof_sp = np.zeros((N, 9))
    for held_month in TRAIN_MONTHS:
        val_mask = months == held_month
        tr_mask = ~val_mask
        dtrain = lgb.Dataset(X_train[tr_mask], label=y_sp[tr_mask])
        dval = lgb.Dataset(X_train[val_mask], label=y_sp[val_mask], reference=dtrain)
        model = lgb.train(
            params, dtrain, num_boost_round=N_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        sp_probs = model.predict(X_train[val_mask])
        oof_sp[val_mask] = sp_probs @ agg

    score = lomo_map(oof_sp)
    pc = lomo_map_perclass(oof_sp)
    print(f"  min_count={min_count:2d}: n_species={n_sp:2d}, LOMO={score:.4f}, "
          f"Corm={pc['Cormorants']:.4f}")

# ── Approach 4: Blend species-level with existing best ────────────
print("\n" + "=" * 70)
print("  APPROACH 4: BLEND SPECIES-LEVEL WITH EXISTING PREDICTIONS")
print("=" * 70)

# Load existing best predictions
e166 = np.load(ROOT / "oof_e166.npy")
e175_best = np.load(ROOT / "oof_e175_best.npy")
e176_igk = np.load(ROOT / "oof_e176_iso_gmm_knn.npy")
e183_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")

existing = {
    "e166": e166,
    "e175_best": e175_best,
    "e176_igk": e176_igk,
    "e183_tabpfn": e183_tabpfn,
}

# Save species OOF for potential future use
np.save(ROOT / "oof_e186_species.npy", oof_group)

for name, ext_oof in existing.items():
    best_blend_score = 0
    best_alpha = 0
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * oof_group + (1.0 - alpha) * ext_oof
        score = lomo_map(blend)
        if score > best_blend_score:
            best_blend_score = score
            best_alpha = alpha

    pc_blend = lomo_map_perclass(best_alpha * oof_group + (1.0 - best_alpha) * ext_oof)
    print(f"  species + {name:15s}: alpha={best_alpha:.2f}, LOMO={best_blend_score:.4f}, "
          f"Corm={pc_blend['Cormorants']:.4f}")

# ── Approach 5: Per-class replacement — use species for Cormorant only ──
print("\n" + "=" * 70)
print("  APPROACH 5: CORMORANT COLUMN FROM SPECIES MODEL, REST FROM E166")
print("=" * 70)

for name, ext_oof in existing.items():
    # Replace only Cormorant column
    hybrid = ext_oof.copy()
    best_score = lomo_map(hybrid)
    best_config = "no replacement"

    for alpha in np.arange(0.0, 1.01, 0.1):
        test_preds = ext_oof.copy()
        test_preds[:, CORM_IDX] = alpha * oof_group[:, CORM_IDX] + (1 - alpha) * ext_oof[:, CORM_IDX]
        # Renormalize
        row_sums = test_preds.sum(axis=1, keepdims=True)
        test_preds = test_preds / np.maximum(row_sums, 1e-12)

        score = lomo_map(test_preds)
        if score > best_score:
            best_score = score
            best_config = f"alpha={alpha:.1f}"

    pc = lomo_map_perclass(test_preds) if best_config != "no replacement" else lomo_map_perclass(ext_oof)
    print(f"  {name:15s}: {best_config}, LOMO={best_score:.4f}")

# ── Final comparison ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)

print(f"\n  {'Method':40s}  {'LOMO':>7}  {'Corm_AP':>8}")
print(f"  {'-'*40}  {'-'*7}  {'-'*8}")
print(f"  {'Direct group-level (v3 features)':40s}  {lomo_direct:.4f}  {pc_direct['Cormorants']:.4f}")
print(f"  {'Species-level (merged >=5)':40s}  {lomo_species:.4f}  {pc_species['Cormorants']:.4f}")
print(f"  {'e166 alone':40s}  {lomo_map(e166):.4f}  {lomo_map_perclass(e166)['Cormorants']:.4f}")

delta = lomo_species - lomo_direct
print(f"\n  Species vs Direct delta: {delta:+.4f}")
print(f"  Cormorant AP delta: {pc_species['Cormorants'] - pc_direct['Cormorants']:+.4f}")
print(f"\n{'='*70}")
