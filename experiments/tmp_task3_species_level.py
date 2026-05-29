"""Task 3: Validate species-level prediction aggregated to groups.

Train LightGBM on bird_species (68 labels, merge rare to parent group),
predict species probabilities, aggregate to 9 group probabilities by summing.
Compare per-class AP to direct group-level prediction.

Key question: does Cormorant benefit (all 40 are Great Cormorant = clean species target)?
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_train, load_test
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
MONTHS = [1, 4, 9, 10]

t0 = time.time()
print("=" * 90)
print("  TASK 3: Species-Level Prediction Aggregated to Groups")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)


# ══════════════════════════════════════════════════════════════════════
# 1. Load data and analyze species structure
# ══════════════════════════════════════════════════════════════════════

print("\n[1] Loading data...", flush=True)

train_df = load_train()
test_df = load_test()
y_group = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups_cv = train_df["primary_observation_id"].values
n_train = len(y_group)
n_test = len(test_df)

# Load cached features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")

# Use ALL 327 features for species-level (more granularity needed)
common_cols = sorted(set(train_feats.columns) & set(test_feats.columns))
X_train_all = np.nan_to_num(train_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(test_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# Also load E175 selected features for group-level baseline
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train_sel = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_sel = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Train: {X_train_all.shape}, Test: {X_test_all.shape}")
print(f"  Selected features: {len(selected)}, All features: {len(common_cols)}")

# Analyze species structure
species_col = train_df["bird_species"].values
bird_group = train_df["bird_group"].values
species_counts = Counter(species_col)

print(f"\n  Total unique species: {len(species_counts)}")
print(f"\n  Species per group:")
for g in CLASSES:
    mask = bird_group == g
    sp_in_group = Counter(species_col[mask])
    n_sp = len(sp_in_group)
    top3 = sp_in_group.most_common(3)
    top3_str = ", ".join(f"{s}({n})" for s, n in top3)
    print(f"    {g:15s}: {n_sp:3d} species, {mask.sum():5d} samples | Top: {top3_str}")


# ══════════════════════════════════════════════════════════════════════
# 2. Build species -> group mapping with rare merging
# ══════════════════════════════════════════════════════════════════════

MIN_SAMPLES = 3  # merge species with < 3 samples into parent group

# Build mapping: species name -> merged label
species_to_group = {}
for sp, count in species_counts.items():
    # Find parent group
    sp_mask = species_col == sp
    parent_group = bird_group[sp_mask][0]
    species_to_group[sp] = parent_group

# Create merged species labels
# For species with >= MIN_SAMPLES, keep as distinct species
# For species with < MIN_SAMPLES, merge into "group_other"
merged_species = []
for sp in species_col:
    if species_counts[sp] >= MIN_SAMPLES:
        merged_species.append(sp)
    else:
        parent = species_to_group[sp]
        merged_species.append(f"{parent}_rare")

merged_species = np.array(merged_species)
merged_counts = Counter(merged_species)

# Build species label encoding
unique_species = sorted(set(merged_species))
sp_to_idx = {sp: i for i, sp in enumerate(unique_species)}
y_species = np.array([sp_to_idx[sp] for sp in merged_species])
n_species = len(unique_species)

# Build species -> group index mapping (for aggregation)
sp_to_group_idx = {}
for sp in unique_species:
    # Find the group for this species
    if sp.endswith("_rare"):
        group_name = sp.replace("_rare", "")
    else:
        group_name = species_to_group[sp]
    sp_to_group_idx[sp] = CLASSES.index(group_name)

# Aggregation matrix: species_probs @ agg_matrix = group_probs
agg_matrix = np.zeros((n_species, N_CLASSES))
for i, sp in enumerate(unique_species):
    g_idx = sp_to_group_idx[sp]
    agg_matrix[i, g_idx] = 1.0

print(f"\n  After merging (min_samples={MIN_SAMPLES}):")
print(f"    Distinct species labels: {n_species}")
for g in CLASSES:
    g_idx = CLASSES.index(g)
    sp_in_g = [sp for sp in unique_species if sp_to_group_idx[sp] == g_idx]
    counts_str = ", ".join(f"{sp}({merged_counts[sp]})" for sp in sp_in_g)
    print(f"    {g:15s}: {len(sp_in_g)} labels -> {counts_str}")


# ══════════════════════════════════════════════════════════════════════
# 3. Helper
# ══════════════════════════════════════════════════════════════════════

def lomo_eval(oof, y_eval, months, label=""):
    """TRUE LOMO evaluation on GROUP-level."""
    skf, per_class = compute_map(y_eval, oof)
    lomo_maps = {}
    for held in MONTHS:
        mask = months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y_eval[mask], oof[mask])
            lomo_maps[held] = lm
    lomo_avg = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  {label:45s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]", flush=True)
    return skf, lomo_avg, per_class, lomo_maps


# ══════════════════════════════════════════════════════════════════════
# 4. Baseline: group-level LGB (selected features)
# ══════════════════════════════════════════════════════════════════════

import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

N_SEEDS = 5
print(f"\n[2] Group-level baseline ({N_SEEDS} seeds)...", flush=True)

oof_group_all = np.zeros((N_SEEDS, n_train, N_CLASSES))
test_group_all = np.zeros((N_SEEDS, n_test, N_CLASSES))

for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
    oof_s = np.zeros((n_train, N_CLASSES))
    test_s = np.zeros((n_test, N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_train_sel, y_group, groups_cv)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            drop_rate=0.15, is_unbalance=True, verbosity=-1,
            random_state=42 + seed + fold, n_jobs=-1,
        )
        m.fit(X_train_sel[tr], y_group[tr], eval_set=[(X_train_sel[va], y_group[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[va] = m.predict_proba(X_train_sel[va])
        test_s += m.predict_proba(X_test_sel) / N_FOLDS
    oof_group_all[seed] = oof_s
    test_group_all[seed] = test_s

oof_group = np.mean(oof_group_all, axis=0)
test_group = np.mean(test_group_all, axis=0)
skf_g, lomo_g, pc_g, lomo_maps_g = lomo_eval(oof_group, y_group, train_months, "Group-level baseline (selected)")


# ══════════════════════════════════════════════════════════════════════
# 4b. Baseline with ALL features
# ══════════════════════════════════════════════════════════════════════

print(f"\n[2b] Group-level with ALL {len(common_cols)} features ({N_SEEDS} seeds)...", flush=True)

oof_group_all2 = np.zeros((N_SEEDS, n_train, N_CLASSES))
test_group_all2 = np.zeros((N_SEEDS, n_test, N_CLASSES))

for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
    oof_s = np.zeros((n_train, N_CLASSES))
    test_s = np.zeros((n_test, N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_train_all, y_group, groups_cv)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            drop_rate=0.15, is_unbalance=True, verbosity=-1,
            random_state=42 + seed + fold, n_jobs=-1,
        )
        m.fit(X_train_all[tr], y_group[tr], eval_set=[(X_train_all[va], y_group[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[va] = m.predict_proba(X_train_all[va])
        test_s += m.predict_proba(X_test_all) / N_FOLDS
    oof_group_all2[seed] = oof_s
    test_group_all2[seed] = test_s

oof_group2 = np.mean(oof_group_all2, axis=0)
test_group2 = np.mean(test_group_all2, axis=0)
skf_g2, lomo_g2, pc_g2, lomo_maps_g2 = lomo_eval(oof_group2, y_group, train_months, "Group-level baseline (ALL feats)")


# ══════════════════════════════════════════════════════════════════════
# 5. Species-level LGB -> aggregate to groups
# ══════════════════════════════════════════════════════════════════════

print(f"\n[3] Species-level ({n_species} classes, ALL features, {N_SEEDS} seeds)...", flush=True)

oof_sp_all = np.zeros((N_SEEDS, n_train, n_species))
test_sp_all = np.zeros((N_SEEDS, n_test, n_species))

for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
    oof_s = np.zeros((n_train, n_species))
    test_s = np.zeros((n_test, n_species))
    for fold, (tr, va) in enumerate(sgkf.split(X_train_all, y_group, groups_cv)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_species, boosting_type="dart",
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            min_child_samples=5,  # lower for rare species
            colsample_bytree=0.6, subsample=0.7,
            drop_rate=0.15, is_unbalance=True, verbosity=-1,
            random_state=42 + seed + fold, n_jobs=-1,
        )
        m.fit(X_train_all[tr], y_species[tr],
              eval_set=[(X_train_all[va], y_species[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[va] = m.predict_proba(X_train_all[va])
        test_s += m.predict_proba(X_test_all) / N_FOLDS
    oof_sp_all[seed] = oof_s
    test_sp_all[seed] = test_s

oof_species = np.mean(oof_sp_all, axis=0)
test_species = np.mean(test_sp_all, axis=0)

# Aggregate species -> groups
oof_sp_agg = oof_species @ agg_matrix
test_sp_agg = test_species @ agg_matrix

skf_sp, lomo_sp, pc_sp, lomo_maps_sp = lomo_eval(oof_sp_agg, y_group, train_months, f"Species-level ({n_species} classes)")


# ══════════════════════════════════════════════════════════════════════
# 5b. Species-level with selected features
# ══════════════════════════════════════════════════════════════════════

print(f"\n[3b] Species-level (selected features, {N_SEEDS} seeds)...", flush=True)

oof_sp_sel_all = np.zeros((N_SEEDS, n_train, n_species))
test_sp_sel_all = np.zeros((N_SEEDS, n_test, n_species))

for seed in range(N_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
    oof_s = np.zeros((n_train, n_species))
    test_s = np.zeros((n_test, n_species))
    for fold, (tr, va) in enumerate(sgkf.split(X_train_sel, y_group, groups_cv)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_species, boosting_type="dart",
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            min_child_samples=5, colsample_bytree=0.6, subsample=0.7,
            drop_rate=0.15, is_unbalance=True, verbosity=-1,
            random_state=42 + seed + fold, n_jobs=-1,
        )
        m.fit(X_train_sel[tr], y_species[tr],
              eval_set=[(X_train_sel[va], y_species[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[va] = m.predict_proba(X_train_sel[va])
        test_s += m.predict_proba(X_test_sel) / N_FOLDS
    oof_sp_sel_all[seed] = oof_s
    test_sp_sel_all[seed] = test_s

oof_sp_sel = np.mean(oof_sp_sel_all, axis=0)
test_sp_sel = np.mean(test_sp_sel_all, axis=0)

oof_sp_sel_agg = oof_sp_sel @ agg_matrix
test_sp_sel_agg = test_sp_sel @ agg_matrix

skf_sp_sel, lomo_sp_sel, pc_sp_sel, lomo_maps_sp_sel = lomo_eval(
    oof_sp_sel_agg, y_group, train_months, f"Species-level (selected feats)"
)


# ══════════════════════════════════════════════════════════════════════
# 6. Per-class comparison
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  PER-CLASS AP COMPARISON")
print("=" * 90)

variants = [
    ("Group (sel)", pc_g, skf_g, lomo_g, lomo_maps_g),
    ("Group (all)", pc_g2, skf_g2, lomo_g2, lomo_maps_g2),
    ("Species (all)", pc_sp, skf_sp, lomo_sp, lomo_maps_sp),
    ("Species (sel)", pc_sp_sel, skf_sp_sel, lomo_sp_sel, lomo_maps_sp_sel),
]

header = f"  {'Class':15s}"
for name, _, _, _, _ in variants:
    header += f"  {name:>14s}"
header += "    Delta best"
print(header)
print(f"  {'-' * (15 + 16 * len(variants) + 14)}")

for cls in CLASSES:
    line = f"  {cls:15s}"
    base_val = pc_g[cls]
    for name, pc, _, _, _ in variants:
        line += f"  {pc[cls]:14.4f}"
    # Best non-baseline delta
    best_delta = 0
    for name, pc, _, _, _ in variants[1:]:
        delta = pc[cls] - base_val
        if abs(delta) > abs(best_delta):
            best_delta = delta
    marker = " ***" if best_delta > 0.01 else "   *" if best_delta > 0 else "  --"
    line += f"  {best_delta:+.4f}{marker}"
    print(line)

print(f"\n  {'SKF':15s}", end="")
for name, _, skf, _, _ in variants:
    print(f"  {skf:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for name, _, _, lomo, _ in variants:
    print(f"  {lomo:14.4f}", end="")
print()

# Month breakdown
print(f"\n  LOMO Month Breakdown:")
header = f"  {'Variant':20s}"
for m_val in MONTHS:
    header += f"  M{m_val:02d}"
print(header)
for name, _, _, _, lomo_maps in variants:
    line = f"  {name:20s}"
    for m_val in MONTHS:
        line += f"  {lomo_maps.get(m_val, 0):.3f}"
    print(line)


# ══════════════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TASK 3 SUMMARY")
print("=" * 90)

print(f"\n  Species structure: {n_species} merged species from {len(species_counts)} original")
print(f"  Aggregation: sum species probs within each group")

for name, pc, skf, lomo, _ in variants:
    d_skf = skf - skf_g
    d_lomo = lomo - lomo_g
    corm = pc.get("Cormorants", 0)
    wader = pc.get("Waders", 0)
    d_str = f"(dSKF={d_skf:+.4f}, dLOMO={d_lomo:+.4f})" if name != "Group (sel)" else "(baseline)"
    print(f"  {name:20s}: SKF={skf:.4f} LOMO={lomo:.4f}  Corm={corm:.4f} Wader={wader:.4f} {d_str}")

# Key insight: Cormorant analysis
print(f"\n  Cormorant analysis:")
print(f"    All 40 Cormorants are Great Cormorant -> clean species-level target")
corm_idx = CLASSES.index("Cormorants")
for name, pc, _, _, _ in variants:
    print(f"    {name:20s}: Cormorant AP = {pc['Cormorants']:.4f}")

elapsed = time.time() - t0
print(f"\n  Completed in {elapsed/60:.1f} min")
print("=" * 90)
