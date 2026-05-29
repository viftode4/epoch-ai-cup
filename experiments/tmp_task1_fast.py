"""Task 1: Validate noise cleaning impact (fast version).

Uses gbdt (not dart) with 3 seeds for speed.
Tests: baseline, remove agreed noisy, relabel, remove extreme.
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

t0 = time.time()
print("=" * 90)
print("  TASK 1: Noise Cleaning Validation (FAST)")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90, flush=True)

# ── Load data ──
train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

oof_e175 = np.load(ROOT / "oof_e175_best.npy")
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")

print(f"  Train: {X_train.shape}, Features: {len(selected)}", flush=True)

# ── Cleanlab noise detection ──
print("\n[1] Cleanlab noise detection...", flush=True)
from cleanlab.rank import get_label_quality_scores

quality_e175 = get_label_quality_scores(labels=y, pred_probs=oof_e175, method="self_confidence")
quality_tabpfn = get_label_quality_scores(labels=y, pred_probs=oof_tabpfn, method="self_confidence")

noisy_e175 = set(np.where(quality_e175 < 0.5)[0])
noisy_tabpfn = set(np.where(quality_tabpfn < 0.5)[0])
agreed_noisy = sorted(noisy_e175 & noisy_tabpfn)

extreme_e175 = set(np.where(quality_e175 < 0.05)[0])
extreme_tabpfn = set(np.where(quality_tabpfn < 0.05)[0])
agreed_extreme = sorted(extreme_e175 & extreme_tabpfn)

print(f"  Agreed noisy: {len(agreed_noisy)}, Extreme: {len(agreed_extreme)}", flush=True)

# Per-class breakdown
print(f"\n  {'Class':15s}  {'N':>5s}  {'Noisy':>6s}  {'%':>5s}  {'Extreme':>8s}  {'TabPFN says':>20s}", flush=True)
for c in range(N_CLASSES):
    n_class = (y == c).sum()
    noisy_in_c = [i for i in agreed_noisy if y[i] == c]
    n_noisy = len(noisy_in_c)
    n_ext = sum(1 for i in agreed_extreme if y[i] == c)
    if noisy_in_c:
        preds = oof_tabpfn[noisy_in_c].argmax(axis=1)
        cnts = np.bincount(preds, minlength=N_CLASSES)
        top = CLASSES[cnts.argmax()]
    else:
        top = "-"
    print(f"  {CLASSES[c]:15s}  {n_class:5d}  {n_noisy:6d}  {100*n_noisy/max(n_class,1):4.1f}%  {n_ext:8d}  -> {top}", flush=True)


# ── Training helper ──
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

def train_variant(noise_idx, mode="remove", relabel_probs=None, n_seeds=3, label=""):
    """Train LGB with noise handling. Returns (oof, skf, lomo, per_class, lomo_maps)."""
    noise_set = set(noise_idx)
    y_mod = y.copy()
    if mode == "relabel" and relabel_probs is not None:
        for idx in noise_idx:
            y_mod[idx] = relabel_probs[idx].argmax()
        changed = sum(1 for i in noise_idx if y_mod[i] != y[i])
        print(f"    Relabeled {changed}/{len(noise_idx)} samples", flush=True)

    n = len(y)
    oof_seeds = np.zeros((n_seeds, n, N_CLASSES))

    for seed in range(n_seeds):
        t_s = time.time()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_s = np.zeros((n, N_CLASSES))
        for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
            # Training: handle noise
            if mode == "remove":
                tr = np.array([i for i in tr if i not in noise_set])
                X_tr, y_tr = X_train[tr], y[tr]
            elif mode == "relabel":
                X_tr, y_tr = X_train[tr], y_mod[tr]
            else:
                X_tr, y_tr = X_train[tr], y[tr]

            # Validation: always original labels
            X_va, y_va = X_train[va], y[va]

            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
                n_estimators=800, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1,
                random_state=42 + seed + fold, n_jobs=-1,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_va)
        oof_seeds[seed] = oof_s
        print(f"    Seed {seed+1}/{n_seeds}: {time.time()-t_s:.1f}s", flush=True)

    oof = np.mean(oof_seeds, axis=0)
    skf, per_class = compute_map(y, oof)
    lomo_maps = {}
    for held in MONTHS:
        mask = train_months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof[mask])
            lomo_maps[held] = lm
    lomo = np.mean(list(lomo_maps.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_maps.items()))
    print(f"  >> {label:40s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return oof, skf, lomo, per_class, lomo_maps


# ── Run variants ──
N_SEEDS = 3
print(f"\n[2] Training variants ({N_SEEDS} seeds, gbdt)...", flush=True)

print("\n  --- Baseline ---", flush=True)
_, skf_0, lomo_0, pc_0, lm_0 = train_variant([], mode="baseline", n_seeds=N_SEEDS, label="Baseline")

print(f"\n  --- A: Remove {len(agreed_noisy)} agreed noisy ---", flush=True)
_, skf_a, lomo_a, pc_a, lm_a = train_variant(agreed_noisy, mode="remove", n_seeds=N_SEEDS, label=f"Remove {len(agreed_noisy)}")

print(f"\n  --- B: Relabel {len(agreed_noisy)} to TabPFN ---", flush=True)
_, skf_b, lomo_b, pc_b, lm_b = train_variant(agreed_noisy, mode="relabel", relabel_probs=oof_tabpfn, n_seeds=N_SEEDS, label=f"Relabel {len(agreed_noisy)}")

print(f"\n  --- C: Remove {len(agreed_extreme)} extreme (q<0.05) ---", flush=True)
_, skf_c, lomo_c, pc_c, lm_c = train_variant(agreed_extreme, mode="remove", n_seeds=N_SEEDS, label=f"Remove {len(agreed_extreme)} extreme")


# ── Results table ──
print("\n" + "=" * 90)
print("  PER-CLASS AP COMPARISON")
print("=" * 90, flush=True)

variants = [
    ("Baseline", pc_0, skf_0, lomo_0, lm_0),
    (f"Remove {len(agreed_noisy)}", pc_a, skf_a, lomo_a, lm_a),
    (f"Relabel {len(agreed_noisy)}", pc_b, skf_b, lomo_b, lm_b),
    (f"Extreme ({len(agreed_extreme)})", pc_c, skf_c, lomo_c, lm_c),
]

print(f"\n  {'Class':15s}", end="")
for name, _, _, _, _ in variants:
    print(f"  {name:>14s}", end="")
print(f"  {'Best dlt':>8s}")
for cls in CLASSES:
    print(f"  {cls:15s}", end="")
    base = pc_0[cls]
    best_d = 0
    for name, pc, _, _, _ in variants:
        print(f"  {pc[cls]:14.4f}", end="")
        if name != "Baseline":
            d = pc[cls] - base
            if abs(d) > abs(best_d):
                best_d = d
    mk = "***" if best_d > 0.01 else " * " if best_d > 0 else " - "
    print(f"  {best_d:+7.4f} {mk}")

print(f"\n  {'SKF':15s}", end="")
for _, _, skf, _, _ in variants:
    print(f"  {skf:14.4f}", end="")
print()
print(f"  {'LOMO':15s}", end="")
for _, _, _, lomo, _ in variants:
    print(f"  {lomo:14.4f}", end="")
print()

print(f"\n  LOMO Month Breakdown:")
print(f"  {'Variant':20s}", end="")
for m in MONTHS:
    print(f"  {'M'+str(m):>6s}", end="")
print()
for name, _, _, _, lm in variants:
    print(f"  {name:20s}", end="")
    for m in MONTHS:
        print(f"  {lm.get(m,0):6.3f}", end="")
    print()

# ── Summary ──
print("\n" + "=" * 90)
print("  TASK 1 FINAL SUMMARY")
print("=" * 90, flush=True)
print(f"  Noise: {len(agreed_noisy)} agreed (q<0.5), {len(agreed_extreme)} extreme (q<0.05)")
print(f"  Cormorants: {sum(1 for i in agreed_noisy if y[i]==2)}/40 noisy ({sum(1 for i in agreed_extreme if y[i]==2)} extreme)")
print(f"  Waders: {sum(1 for i in agreed_noisy if y[i]==8)}/120 noisy ({sum(1 for i in agreed_extreme if y[i]==8)} extreme)")
for name, pc, skf, lomo, _ in variants:
    d_skf = skf - skf_0 if name != "Baseline" else 0
    d_lomo = lomo - lomo_0 if name != "Baseline" else 0
    print(f"  {name:25s}: SKF={skf:.4f} LOMO={lomo:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}  dSKF={d_skf:+.4f} dLOMO={d_lomo:+.4f}")

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print("=" * 90, flush=True)
