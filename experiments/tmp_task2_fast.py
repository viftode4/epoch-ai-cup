"""Task 2: Validate pseudo-labeling (fast version).

Uses gbdt with 3 seeds. Tests confidence thresholds + iterative PL.
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path
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
print("  TASK 2: Pseudo-Labeling Validation (FAST)")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90, flush=True)

# ── Load data ──
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
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

test_tabpfn = np.load(ROOT / "test_e183_tabpfn.npy")
test_e175 = np.load(ROOT / "test_e175_best.npy")

# Ensemble for pseudo-labels
test_ensemble = 0.5 * test_tabpfn + 0.5 * test_e175
test_max_prob = test_ensemble.max(axis=1)
test_hard_labels = test_ensemble.argmax(axis=1)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
print(f"  Ensemble confidence: mean={test_max_prob.mean():.3f}, max={test_max_prob.max():.3f}", flush=True)
for thresh in [0.5, 0.6, 0.7, 0.8]:
    n = (test_max_prob >= thresh).sum()
    print(f"    >= {thresh:.1f}: {n} samples ({100*n/n_test:.1f}%)", flush=True)


# ── Training helper ──
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

def train_with_pseudo(pseudo_X, pseudo_y, pseudo_weight=0.5, n_seeds=N_SEEDS, label=""):
    """Train LGB with pseudo data added to training only."""
    n_pseudo = len(pseudo_y)
    if n_pseudo > 0:
        X_comb = np.vstack([X_train, pseudo_X])
        y_comb = np.concatenate([y, pseudo_y])
        w_comb = np.concatenate([np.ones(n_train), np.full(n_pseudo, pseudo_weight)])
    else:
        X_comb = X_train
        y_comb = y
        w_comb = np.ones(n_train)

    oof_seeds = np.zeros((n_seeds, n_train, N_CLASSES))
    test_seeds = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        t_s = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((n_train, N_CLASSES))
        test_s = np.zeros((n_test, N_CLASSES))
        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            if n_pseudo > 0:
                pseudo_idx = np.arange(n_train, n_train + n_pseudo)
                comb_tr = np.concatenate([tr, pseudo_idx])
            else:
                comb_tr = tr

            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
                n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            m.fit(X_comb[comb_tr], y_comb[comb_tr], sample_weight=w_comb[comb_tr],
                  eval_set=[(X_train[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_train[va])
            test_s += m.predict_proba(X_test) / N_FOLDS
        oof_seeds[seed] = oof_s
        test_seeds[seed] = test_s
        print(f"    Seed {seed+1}/{n_seeds}: {time.time()-t_s:.1f}s", flush=True)

    oof = np.mean(oof_seeds, axis=0)
    test_out = np.mean(test_seeds, axis=0)
    skf, per_class = compute_map(y, oof)
    lomo_maps = {}
    for held in MONTHS:
        mask = train_months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
    lomo = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  >> {label:45s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return oof, test_out, skf, lomo, per_class, lomo_maps


# ── Run variants ──
results = {}

print(f"\n[2] Baseline...", flush=True)
oof_base, test_base, skf_0, lomo_0, pc_0, lm_0 = train_with_pseudo(
    np.zeros((0, X_train.shape[1])), np.zeros(0, dtype=int), label="Baseline (no pseudo)")
results["Baseline"] = (skf_0, lomo_0, pc_0, lm_0)

for thresh in [0.5, 0.6, 0.7, 0.8]:
    mask = test_max_prob >= thresh
    n_p = mask.sum()
    if n_p == 0:
        print(f"\n  Skipping thresh={thresh}: no samples", flush=True)
        continue
    dist = np.bincount(test_hard_labels[mask], minlength=N_CLASSES)
    print(f"\n[3] Pseudo thresh={thresh:.1f} ({n_p} samples)", flush=True)
    print(f"    Class dist: {dict(zip([c[:4] for c in CLASSES], dist))}", flush=True)
    # Show month dist
    pm = test_months[mask]
    for m_val in sorted(set(pm)):
        print(f"    Month {m_val:2d}: {(pm==m_val).sum()}", flush=True)

    _, _, skf_p, lomo_p, pc_p, lm_p = train_with_pseudo(
        X_test[mask], test_hard_labels[mask], pseudo_weight=0.5,
        label=f"Pseudo {thresh:.1f} (n={n_p})")
    results[f"Pseudo {thresh:.1f}"] = (skf_p, lomo_p, pc_p, lm_p)

# ── Iterative (2 rounds) at thresh 0.6 ──
print(f"\n[4] Iterative pseudo-labeling (thresh=0.6, 2 rounds)...", flush=True)
mask_r1 = test_max_prob >= 0.6
n_r1 = mask_r1.sum()
print(f"  Round 1: {n_r1} samples", flush=True)

# Train round 1 to get updated test preds
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X_comb_r1 = np.vstack([X_train, X_test[mask_r1]])
y_comb_r1 = np.concatenate([y, test_hard_labels[mask_r1]])
w_r1 = np.concatenate([np.ones(n_train), np.full(n_r1, 0.5)])
test_preds_r2 = np.zeros((n_test, N_CLASSES))
for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
    pseudo_idx = np.arange(n_train, n_train + n_r1)
    comb_tr = np.concatenate([tr, pseudo_idx])
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
        n_estimators=800, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
        is_unbalance=True, verbosity=-1, random_state=42+fold, n_jobs=-1,
    )
    m.fit(X_comb_r1[comb_tr], y_comb_r1[comb_tr], sample_weight=w_r1[comb_tr],
          eval_set=[(X_train[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    test_preds_r2 += m.predict_proba(X_test) / N_FOLDS

# Round 2
test_max_r2 = test_preds_r2.max(axis=1)
test_labels_r2 = test_preds_r2.argmax(axis=1)
mask_r2 = test_max_r2 >= 0.6
n_r2 = mask_r2.sum()
changed = (test_hard_labels[mask_r1 & mask_r2] != test_labels_r2[mask_r1 & mask_r2]).sum() if (mask_r1 & mask_r2).sum() > 0 else 0
print(f"  Round 2: {n_r2} samples, {changed} labels changed", flush=True)

_, _, skf_it, lomo_it, pc_it, lm_it = train_with_pseudo(
    X_test[mask_r2], test_labels_r2[mask_r2], pseudo_weight=0.5,
    label="Iterative 2R (thresh=0.6)")
results["Iterative 2R"] = (skf_it, lomo_it, pc_it, lm_it)


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
    print()

print(f"\n  {'SKF':15s}", end="")
for n in vnames:
    print(f"  {results[n][0]:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for n in vnames:
    print(f"  {results[n][1]:14.4f}", end="")
print()

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
print("  TASK 2 FINAL SUMMARY")
print("=" * 90, flush=True)
for n in vnames:
    skf, lomo, pc, _ = results[n]
    d_skf = skf - skf_0 if n != "Baseline" else 0
    d_lomo = lomo - lomo_0 if n != "Baseline" else 0
    print(f"  {n:25s}: SKF={skf:.4f} LOMO={lomo:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}  dSKF={d_skf:+.4f} dLOMO={d_lomo:+.4f}")

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print("=" * 90, flush=True)
