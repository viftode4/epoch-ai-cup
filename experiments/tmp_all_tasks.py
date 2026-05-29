"""All 3 tasks: combined fast run.

Task 1 Variant C (extreme noise removal) + Task 2 (pseudo-labeling) + Task 3 (species-level).
Uses gbdt with 3 seeds, sequential execution.
"""

from __future__ import annotations
import sys, time, warnings, gc
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from src.data import CLASSES, load_train, load_test
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
MONTHS = [1, 4, 9, 10]
N_SEEDS = 3

t_total = time.time()

# ══════════════════════════════════════════════════════════════════════
# Shared data loading
# ══════════════════════════════════════════════════════════════════════
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
n_train = len(y)
n_test = len(test_df)

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
common_cols = sorted(set(train_feats.columns) & set(test_feats.columns))

X_sel = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_sel = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_all = np.nan_to_num(train_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(test_feats[common_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Train: {n_train}, Test: {n_test}, Sel: {len(selected)}, All: {len(common_cols)}", flush=True)


def eval_lomo(oof, y_eval, months, label=""):
    skf, per_class = compute_map(y_eval, oof)
    lomo_maps = {}
    for held in MONTHS:
        mask = months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y_eval[mask], oof[mask])
            lomo_maps[held] = lm
    lomo = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  >> {label:45s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return skf, lomo, per_class, lomo_maps


def train_lgb(X_tr_full, X_te_full, y_target, X_va_full=None, y_va_target=None,
              noise_idx=None, noise_mode="baseline", relabel_probs=None,
              pseudo_X=None, pseudo_y=None, pseudo_w=0.5,
              n_classes=N_CLASSES, n_seeds=N_SEEDS, min_child=20, label=""):
    """Unified LGB trainer handling noise + pseudo-labeling."""
    noise_set = set(noise_idx or [])
    y_mod = y_target.copy()
    if noise_mode == "relabel" and relabel_probs is not None:
        for idx in (noise_idx or []):
            y_mod[idx] = relabel_probs[idx].argmax()

    has_pseudo = pseudo_X is not None and len(pseudo_X) > 0
    n_pseudo = len(pseudo_X) if has_pseudo else 0
    if has_pseudo:
        X_comb = np.vstack([X_tr_full, pseudo_X])
        y_comb = np.concatenate([y_mod, pseudo_y])
        w_comb = np.concatenate([np.ones(len(y_target)), np.full(n_pseudo, pseudo_w)])
    else:
        X_comb = X_tr_full
        y_comb = y_mod
        w_comb = np.ones(len(y_target))

    # For evaluation - use original labels if different from training target
    y_eval = y_va_target if y_va_target is not None else y_target

    oof_seeds = np.zeros((n_seeds, len(y_target), n_classes))
    test_seeds = np.zeros((n_seeds, X_te_full.shape[0], n_classes))

    for seed in range(n_seeds):
        t_s = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((len(y_target), n_classes))
        test_s = np.zeros((X_te_full.shape[0], n_classes))

        for fold, (tr, va) in enumerate(sgkf.split(X_tr_full, y, groups)):
            # Build training set
            if noise_mode == "remove":
                tr = np.array([i for i in tr if i not in noise_set])
            if has_pseudo:
                pseudo_idx = np.arange(len(y_target), len(y_target) + n_pseudo)
                comb_tr = np.concatenate([tr, pseudo_idx])
            else:
                comb_tr = tr

            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=n_classes, boosting_type="gbdt",
                n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=min_child, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            # Eval set always uses original X and target
            m.fit(X_comb[comb_tr], y_comb[comb_tr], sample_weight=w_comb[comb_tr],
                  eval_set=[(X_tr_full[va], y_target[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_tr_full[va])
            test_s += m.predict_proba(X_te_full) / N_FOLDS

        oof_seeds[seed] = oof_s
        test_seeds[seed] = test_s
        elapsed = time.time() - t_s
        print(f"    Seed {seed+1}/{n_seeds}: {elapsed:.1f}s", flush=True)

    oof = np.mean(oof_seeds, axis=0)
    test_out = np.mean(test_seeds, axis=0)
    return oof, test_out


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Noise Cleaning - Variant C only (A & B already done)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TASK 1: Noise Cleaning (Variant C: extreme noise removal)")
print("=" * 90, flush=True)

from cleanlab.rank import get_label_quality_scores
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")
q_e175 = get_label_quality_scores(labels=y, pred_probs=oof_e175, method="self_confidence")
q_tabpfn = get_label_quality_scores(labels=y, pred_probs=oof_tabpfn, method="self_confidence")
agreed_extreme = sorted(set(np.where(q_e175 < 0.05)[0]) & set(np.where(q_tabpfn < 0.05)[0]))
agreed_noisy = sorted(set(np.where(q_e175 < 0.5)[0]) & set(np.where(q_tabpfn < 0.5)[0]))

print(f"  Agreed noisy (q<0.5): {len(agreed_noisy)}")
print(f"  Extreme (q<0.05): {len(agreed_extreme)}", flush=True)

# Baseline (needed for comparison)
print("\n  Baseline...", flush=True)
oof_base, _ = train_lgb(X_sel, X_test_sel, y, label="Baseline")
skf_0, lomo_0, pc_0, lm_0 = eval_lomo(oof_base, y, train_months, "Baseline")

# Variant C
print(f"\n  Variant C: Remove {len(agreed_extreme)} extreme...", flush=True)
oof_c, _ = train_lgb(X_sel, X_test_sel, y, noise_idx=agreed_extreme, noise_mode="remove", label="Extreme")
skf_c, lomo_c, pc_c, lm_c = eval_lomo(oof_c, y, train_months, f"Remove {len(agreed_extreme)} extreme")

gc.collect()

print("\n  TASK 1 COMBINED RESULTS (from this + earlier run):")
print(f"  Baseline:      SKF=0.6686 LOMO=0.4890  [1=0.456 4=0.517 9=0.364 10=0.619]")
print(f"  A:Remove 423:  SKF=0.6450 LOMO=0.4760  [1=0.467 4=0.497 9=0.378 10=0.562] -- WORSE")
print(f"  B:Relabel 423: SKF=0.6750 LOMO=0.4922  [1=0.434 4=0.540 9=0.389 10=0.605] -- BEST")
print(f"  C:Remove {len(agreed_extreme)}:   SKF={skf_c:.4f} LOMO={lomo_c:.4f}  [{' '.join(f'{m}={v:.3f}' for m,v in sorted(lm_c.items()))}]", flush=True)

print(f"\n  Key per-class (Baseline -> Relabel -> Extreme):")
for cls in ["Cormorants", "Waders", "Ducks", "Pigeons", "Birds of Prey"]:
    print(f"    {cls:15s}: {pc_0[cls]:.4f} -> relabel:TBD -> extreme:{pc_c[cls]:.4f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Pseudo-Labeling
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TASK 2: Pseudo-Labeling")
print("=" * 90, flush=True)

test_tabpfn = np.load(ROOT / "test_e183_tabpfn.npy")
test_e175_preds = np.load(ROOT / "test_e175_best.npy")
test_ensemble = 0.5 * test_tabpfn + 0.5 * test_e175_preds
test_max_prob = test_ensemble.max(axis=1)
test_hard_labels = test_ensemble.argmax(axis=1)

print(f"  Ensemble confidence: mean={test_max_prob.mean():.3f}", flush=True)
for thresh in [0.5, 0.6, 0.7, 0.8]:
    n_above = (test_max_prob >= thresh).sum()
    print(f"    >= {thresh:.1f}: {n_above} ({100*n_above/n_test:.1f}%)", flush=True)

t2_results = {}
t2_results["Baseline"] = (skf_0, lomo_0, pc_0, lm_0)  # reuse from above

for thresh in [0.5, 0.6, 0.7]:
    mask = test_max_prob >= thresh
    n_p = mask.sum()
    if n_p == 0:
        continue
    dist = np.bincount(test_hard_labels[mask], minlength=N_CLASSES)
    print(f"\n  Pseudo thresh={thresh:.1f} ({n_p} samples):", flush=True)
    print(f"    {dict(zip([c[:4] for c in CLASSES], dist))}", flush=True)
    oof_p, _ = train_lgb(X_sel, X_test_sel, y,
                          pseudo_X=X_test_sel[mask], pseudo_y=test_hard_labels[mask],
                          pseudo_w=0.5, label=f"Pseudo {thresh:.1f}")
    s, l, pc, lm = eval_lomo(oof_p, y, train_months, f"Pseudo {thresh:.1f} (n={n_p})")
    t2_results[f"Pseudo {thresh:.1f}"] = (s, l, pc, lm)
    gc.collect()

# Iterative pseudo-labeling
print(f"\n  Iterative pseudo-labeling (0.6, 2 rounds)...", flush=True)
mask_r1 = test_max_prob >= 0.6
n_r1 = mask_r1.sum()

# Train round 1
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X_comb_r1 = np.vstack([X_sel, X_test_sel[mask_r1]])
y_comb_r1 = np.concatenate([y, test_hard_labels[mask_r1]])
w_r1 = np.concatenate([np.ones(n_train), np.full(n_r1, 0.5)])
test_preds_r2 = np.zeros((n_test, N_CLASSES))
for fold, (tr, va) in enumerate(sgkf.split(X_sel, y, groups)):
    pseudo_idx = np.arange(n_train, n_train + n_r1)
    comb_tr = np.concatenate([tr, pseudo_idx])
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
        n_estimators=800, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
        is_unbalance=True, verbosity=-1, random_state=42+fold, n_jobs=-1,
    )
    m.fit(X_comb_r1[comb_tr], y_comb_r1[comb_tr], sample_weight=w_r1[comb_tr],
          eval_set=[(X_sel[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    test_preds_r2 += m.predict_proba(X_test_sel) / N_FOLDS

mask_r2 = test_preds_r2.max(axis=1) >= 0.6
labels_r2 = test_preds_r2.argmax(axis=1)
n_r2 = mask_r2.sum()
print(f"  R1: {n_r1} -> R2: {n_r2} pseudo samples", flush=True)

oof_it, _ = train_lgb(X_sel, X_test_sel, y,
                       pseudo_X=X_test_sel[mask_r2], pseudo_y=labels_r2[mask_r2],
                       pseudo_w=0.5, label="Iterative 2R")
s_it, l_it, pc_it, lm_it = eval_lomo(oof_it, y, train_months, "Iterative 2R")
t2_results["Iterative 2R"] = (s_it, l_it, pc_it, lm_it)
gc.collect()

print("\n  TASK 2 RESULTS:")
for n, (s, l, pc, lm) in t2_results.items():
    d_s = s - skf_0 if n != "Baseline" else 0
    d_l = l - lomo_0 if n != "Baseline" else 0
    print(f"  {n:25s}: SKF={s:.4f} LOMO={l:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}  d={d_s:+.4f}/{d_l:+.4f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Species-Level
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TASK 3: Species-Level Prediction")
print("=" * 90, flush=True)

species_col = train_df["bird_species"].values
bird_group = train_df["bird_group"].values
species_counts = Counter(species_col)
species_to_group = {}
for sp in species_counts:
    sp_mask = species_col == sp
    species_to_group[sp] = bird_group[sp_mask][0]

MIN_SAMPLES = 3
merged = []
for sp in species_col:
    if species_counts[sp] >= MIN_SAMPLES:
        merged.append(sp)
    else:
        merged.append(f"{species_to_group[sp]}_rare")
merged = np.array(merged)
unique_sp = sorted(set(merged))
sp_to_idx = {sp: i for i, sp in enumerate(unique_sp)}
y_sp = np.array([sp_to_idx[sp] for sp in merged])
n_sp = len(unique_sp)

sp_to_gidx = {}
for sp in unique_sp:
    gn = sp.replace("_rare", "") if sp.endswith("_rare") else species_to_group[sp]
    sp_to_gidx[sp] = CLASSES.index(gn)

agg_matrix = np.zeros((n_sp, N_CLASSES))
for i, sp in enumerate(unique_sp):
    agg_matrix[i, sp_to_gidx[sp]] = 1.0

print(f"  {len(species_counts)} species -> {n_sp} merged labels", flush=True)
for g in CLASSES:
    gidx = CLASSES.index(g)
    sp_in_g = [sp for sp in unique_sp if sp_to_gidx[sp] == gidx]
    print(f"    {g:15s}: {len(sp_in_g)} labels", flush=True)

t3_results = {}

# Group baseline (selected)
t3_results["Group (sel)"] = (skf_0, lomo_0, pc_0, lm_0)

# Group with all features
print(f"\n  Group (ALL features)...", flush=True)
oof_ga, _ = train_lgb(X_all, X_test_all, y, label="Group (all)")
s_ga, l_ga, pc_ga, lm_ga = eval_lomo(oof_ga, y, train_months, "Group (all feats)")
t3_results["Group (all)"] = (s_ga, l_ga, pc_ga, lm_ga)
gc.collect()

# Species (all features)
print(f"\n  Species-level ({n_sp} classes, ALL features)...", flush=True)
oof_sp, _ = train_lgb(X_all, X_test_all, y_sp, y_va_target=y_sp, n_classes=n_sp, min_child=5, label="Species (all)")
oof_sp_agg = oof_sp @ agg_matrix
s_sp, l_sp, pc_sp, lm_sp = eval_lomo(oof_sp_agg, y, train_months, f"Species ({n_sp} cls, all)")
t3_results["Species (all)"] = (s_sp, l_sp, pc_sp, lm_sp)
gc.collect()

# Species (selected features)
print(f"\n  Species-level (selected features)...", flush=True)
oof_sp2, _ = train_lgb(X_sel, X_test_sel, y_sp, y_va_target=y_sp, n_classes=n_sp, min_child=5, label="Species (sel)")
oof_sp2_agg = oof_sp2 @ agg_matrix
s_sp2, l_sp2, pc_sp2, lm_sp2 = eval_lomo(oof_sp2_agg, y, train_months, "Species (sel)")
t3_results["Species (sel)"] = (s_sp2, l_sp2, pc_sp2, lm_sp2)
gc.collect()

print("\n  TASK 3 RESULTS:")
for n, (s, l, pc, lm) in t3_results.items():
    d_s = s - skf_0
    d_l = l - lomo_0
    print(f"  {n:20s}: SKF={s:.4f} LOMO={l:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}  d={d_s:+.4f}/{d_l:+.4f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# GRAND SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  GRAND SUMMARY: ALL TASKS")
print("=" * 90, flush=True)

print(f"\n  BASELINE: SKF={skf_0:.4f} LOMO={lomo_0:.4f}", flush=True)

print(f"\n  TASK 1 - Noise Cleaning:")
print(f"    Agreed noisy: {len(agreed_noisy)}, Extreme: {len(agreed_extreme)}")
print(f"    Cormorants: 34/40 noisy (85%), 13 extreme")
print(f"    A: Remove 423:  SKF=0.6450 LOMO=0.4760  (WORSE - loses minority signal)")
print(f"    B: Relabel 423: SKF=0.6750 LOMO=0.4922  (BEST noise handling)")
print(f"    C: Remove {len(agreed_extreme)}:   SKF={skf_c:.4f} LOMO={lomo_c:.4f}", flush=True)

print(f"\n  TASK 2 - Pseudo-Labeling:")
for n, (s, l, pc, _) in t2_results.items():
    if n == "Baseline":
        continue
    d = l - lomo_0
    print(f"    {n:25s}: LOMO={l:.4f} ({d:+.4f}) Corm={pc['Cormorants']:.4f}", flush=True)

print(f"\n  TASK 3 - Species-Level:")
for n, (s, l, pc, _) in t3_results.items():
    if n == "Group (sel)":
        continue
    d = l - lomo_0
    print(f"    {n:20s}: LOMO={l:.4f} ({d:+.4f}) Corm={pc['Cormorants']:.4f}", flush=True)

# Per-class comparison for key classes
print(f"\n  KEY CLASS COMPARISON (AP):")
print(f"  {'Method':25s}  {'Corm':>6s}  {'Wader':>6s}  {'Duck':>6s}  {'Pigeon':>6s}  {'BoP':>6s}  {'Clut':>6s}", flush=True)
all_results = {}
all_results["Baseline"] = pc_0
all_results["T1:Remove extreme"] = pc_c
for n, (_, _, pc, _) in t2_results.items():
    if n != "Baseline":
        all_results[f"T2:{n}"] = pc
for n, (_, _, pc, _) in t3_results.items():
    if n != "Group (sel)":
        all_results[f"T3:{n}"] = pc

for name, pc in all_results.items():
    print(f"  {name:25s}  {pc['Cormorants']:6.4f}  {pc['Waders']:6.4f}  {pc['Ducks']:6.4f}  {pc['Pigeons']:6.4f}  {pc['Birds of Prey']:6.4f}  {pc['Clutter']:6.4f}", flush=True)

elapsed = time.time() - t_total
print(f"\n  Total elapsed: {elapsed/60:.1f} min")
print("=" * 90, flush=True)
