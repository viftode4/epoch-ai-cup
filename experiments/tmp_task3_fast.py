"""Task 3: Species-level prediction (fast version).

Train LGB on bird_species (merged rare), aggregate to groups.
Uses gbdt with 3 seeds.
"""

from __future__ import annotations
import sys, time, warnings
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
N_SEEDS = 3

t0 = time.time()
print("=" * 90)
print("  TASK 3: Species-Level Prediction (FAST)")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90, flush=True)

# ── Load data ──
train_df = load_train()
test_df = load_test()
y_group = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups_cv = train_df["primary_observation_id"].values
n_train = len(y_group)
n_test = len(test_df)

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
common_cols = sorted(set(train_feats.columns) & set(test_feats.columns))
X_train_all = np.nan_to_num(train_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(test_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train_sel = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_sel = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Train: {n_train}, Test: {n_test}, All feats: {len(common_cols)}, Sel feats: {len(selected)}", flush=True)

# ── Species structure ──
species_col = train_df["bird_species"].values
bird_group = train_df["bird_group"].values
species_counts = Counter(species_col)
species_to_group = {}
for sp, _ in species_counts.items():
    sp_mask = species_col == sp
    species_to_group[sp] = bird_group[sp_mask][0]

MIN_SAMPLES = 3
merged_species = []
for sp in species_col:
    if species_counts[sp] >= MIN_SAMPLES:
        merged_species.append(sp)
    else:
        merged_species.append(f"{species_to_group[sp]}_rare")
merged_species = np.array(merged_species)
merged_counts = Counter(merged_species)

unique_species = sorted(set(merged_species))
sp_to_idx = {sp: i for i, sp in enumerate(unique_species)}
y_species = np.array([sp_to_idx[sp] for sp in merged_species])
n_species = len(unique_species)

sp_to_group_idx = {}
for sp in unique_species:
    if sp.endswith("_rare"):
        group_name = sp.replace("_rare", "")
    else:
        group_name = species_to_group[sp]
    sp_to_group_idx[sp] = CLASSES.index(group_name)

agg_matrix = np.zeros((n_species, N_CLASSES))
for i, sp in enumerate(unique_species):
    agg_matrix[i, sp_to_group_idx[sp]] = 1.0

print(f"  Species: {len(species_counts)} original -> {n_species} merged (min={MIN_SAMPLES})", flush=True)
for g in CLASSES:
    g_idx = CLASSES.index(g)
    sp_in_g = [sp for sp in unique_species if sp_to_group_idx[sp] == g_idx]
    n_total = sum(merged_counts[sp] for sp in sp_in_g)
    print(f"    {g:15s}: {len(sp_in_g)} labels, {n_total} samples", flush=True)


# ── Training helper ──
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

def train_lgb_cv(X_tr_full, X_te_full, y_target, n_classes, n_seeds=N_SEEDS, label="", min_child=20):
    """Train LGB multiclass with CV. Returns (oof, test_preds)."""
    oof_seeds = np.zeros((n_seeds, n_train, n_classes))
    test_seeds = np.zeros((n_seeds, n_test, n_classes))
    for seed in range(n_seeds):
        t_s = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((n_train, n_classes))
        test_s = np.zeros((n_test, n_classes))
        for fold, (tr, va) in enumerate(sgkf.split(X_tr_full, y_group, groups_cv)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=n_classes, boosting_type="gbdt",
                n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=min_child, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            m.fit(X_tr_full[tr], y_target[tr],
                  eval_set=[(X_tr_full[va], y_target[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_tr_full[va])
            test_s += m.predict_proba(X_te_full) / N_FOLDS
        oof_seeds[seed] = oof_s
        test_seeds[seed] = test_s
        print(f"    Seed {seed+1}/{n_seeds}: {time.time()-t_s:.1f}s", flush=True)
    return np.mean(oof_seeds, axis=0), np.mean(test_seeds, axis=0)

def eval_group(oof_group, label=""):
    """Evaluate at group level (SKF + LOMO)."""
    skf, per_class = compute_map(y_group, oof_group)
    lomo_maps = {}
    for held in MONTHS:
        mask = train_months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y_group[mask], oof_group[mask])
            lomo_maps[held] = lm
    lomo = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  >> {label:40s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return skf, lomo, per_class, lomo_maps


# ── Run variants ──
results = {}

print(f"\n[2] Group-level baseline (selected features)...", flush=True)
oof_g, _ = train_lgb_cv(X_train_sel, X_test_sel, y_group, N_CLASSES, label="Group (selected)")
skf_g, lomo_g, pc_g, lm_g = eval_group(oof_g, "Group (selected)")
results["Group (sel)"] = (skf_g, lomo_g, pc_g, lm_g)

print(f"\n[2b] Group-level (ALL features)...", flush=True)
oof_g2, _ = train_lgb_cv(X_train_all, X_test_all, y_group, N_CLASSES, label="Group (all)")
skf_g2, lomo_g2, pc_g2, lm_g2 = eval_group(oof_g2, "Group (all feats)")
results["Group (all)"] = (skf_g2, lomo_g2, pc_g2, lm_g2)

print(f"\n[3] Species-level (ALL features, {n_species} classes)...", flush=True)
oof_sp, _ = train_lgb_cv(X_train_all, X_test_all, y_species, n_species, label="Species", min_child=5)
oof_sp_agg = oof_sp @ agg_matrix
skf_sp, lomo_sp, pc_sp, lm_sp = eval_group(oof_sp_agg, f"Species ({n_species} cls)")
results["Species (all)"] = (skf_sp, lomo_sp, pc_sp, lm_sp)

print(f"\n[3b] Species-level (selected features)...", flush=True)
oof_sp2, _ = train_lgb_cv(X_train_sel, X_test_sel, y_species, n_species, label="Species (sel)", min_child=5)
oof_sp2_agg = oof_sp2 @ agg_matrix
skf_sp2, lomo_sp2, pc_sp2, lm_sp2 = eval_group(oof_sp2_agg, "Species (sel feats)")
results["Species (sel)"] = (skf_sp2, lomo_sp2, pc_sp2, lm_sp2)


# ── Results table ──
print("\n" + "=" * 90)
print("  PER-CLASS AP COMPARISON")
print("=" * 90, flush=True)

vnames = list(results.keys())
print(f"\n  {'Class':15s}", end="")
for n in vnames:
    print(f"  {n:>14s}", end="")
print()
for cls in CLASSES:
    print(f"  {cls:15s}", end="")
    for n in vnames:
        print(f"  {results[n][2][cls]:14.4f}", end="")
    # Delta: species vs group (same features)
    d1 = results["Species (all)"][2][cls] - results["Group (all)"][2][cls]
    d2 = results["Species (sel)"][2][cls] - results["Group (sel)"][2][cls]
    print(f"  all:{d1:+.3f} sel:{d2:+.3f}")

print(f"\n  {'SKF':15s}", end="")
for n in vnames:
    print(f"  {results[n][0]:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for n in vnames:
    print(f"  {results[n][1]:14.4f}", end="")
print()

# Month breakdown
print(f"\n  LOMO Month Breakdown:")
print(f"  {'Variant':20s}", end="")
for m_val in MONTHS:
    print(f"  {'M'+str(m_val):>6s}", end="")
print()
for n in vnames:
    print(f"  {n:20s}", end="")
    for m_val in MONTHS:
        print(f"  {results[n][3].get(m_val,0):6.3f}", end="")
    print()


# ── Summary ──
print("\n" + "=" * 90)
print("  TASK 3 FINAL SUMMARY")
print("=" * 90, flush=True)

print(f"  Species: {len(species_counts)} -> {n_species} merged labels")
print(f"  Key: Cormorant = 1 species (Great Cormorant), 40 samples -> clean target")
print()
for n in vnames:
    skf, lomo, pc, _ = results[n]
    d_skf = skf - results["Group (sel)"][0]
    d_lomo = lomo - results["Group (sel)"][1]
    print(f"  {n:20s}: SKF={skf:.4f} LOMO={lomo:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}  dSKF={d_skf:+.4f} dLOMO={d_lomo:+.4f}")

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print("=" * 90, flush=True)
