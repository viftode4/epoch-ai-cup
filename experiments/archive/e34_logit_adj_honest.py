"""E34: Honest Logit Adjustment Evaluation

Load E32 OOF predictions, then:
1. Split-half evaluation: optimize tau on A, evaluate on B (and vice versa)
2. Fixed tau baselines: tau=0, tau=0.5, tau=1.0, rarity-scaled
3. Bootstrap CIs on all variants
4. If honest delta < 0.005 -> not worth the complexity

Depends on: oof_e32.npy (run E32 first)
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, bootstrap_map_ci

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)


def apply_logit_adj_with_tau(preds, priors, tau_vec):
    """Apply logit adjustment: multiply by prior^(-tau), renormalize."""
    adj = priors ** (-tau_vec)
    out = preds * adj[None, :]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def optimize_tau(oof, y, priors, label=""):
    """Per-class tau optimization. Returns (best_tau, best_map)."""
    tau = np.zeros(N_CLASSES)
    base_map, _ = compute_map(y, oof)
    best_map = base_map

    for iteration in range(3):
        improved = False
        for c in range(N_CLASSES):
            best_t = tau[c]
            best_m = best_map
            for t in np.arange(-0.5, 1.51, 0.02):
                tau[c] = t
                adj_preds = apply_logit_adj_with_tau(oof, priors, tau)
                m, _ = compute_map(y, adj_preds)
                if m > best_m:
                    best_m = m
                    best_t = t
            tau[c] = best_t
            if best_m > best_map:
                best_map = best_m
                improved = True
        if label:
            print(f"    [{label}] Round {iteration+1}: {best_map:.4f}", flush=True)
        if not improved:
            break

    return tau.copy(), best_map


# ======================================================================
print("=" * 60, flush=True)
print("E34 HONEST LOGIT ADJUSTMENT EVALUATION", flush=True)
print("=" * 60, flush=True)

# Load E32 OOF
oof_path = ROOT / "oof_e32.npy"
if not oof_path.exists():
    print("ERROR: oof_e32.npy not found. Run E32 first!", flush=True)
    sys.exit(1)

oof = np.load(oof_path)
print(f"  Loaded: oof_e32.npy, shape={oof.shape}", flush=True)

# Labels
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)
priors = counts / counts.sum()

# Baseline
base_map, base_per = compute_map(y, oof)
print(f"  E32 baseline mAP: {base_map:.4f}", flush=True)

# ======================================================================
# 1. Split-Half Evaluation
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("SPLIT-HALF EVALUATION", flush=True)
print("=" * 60, flush=True)
print("  Optimize tau on half A, evaluate on half B (and vice versa)", flush=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
idx_a, idx_b = next(sss.split(np.zeros(len(y)), y))

print(f"  Half A: {len(idx_a)} samples, Half B: {len(idx_b)} samples", flush=True)

# Optimize on A, evaluate on B
print("\n  --- Optimize on A, evaluate on B ---", flush=True)
tau_a, map_opt_a = optimize_tau(oof[idx_a], y[idx_a], priors, label="A->B")
oof_adj_b = apply_logit_adj_with_tau(oof[idx_b], priors, tau_a)
map_eval_b, per_eval_b = compute_map(y[idx_b], oof_adj_b)
map_base_b, _ = compute_map(y[idx_b], oof[idx_b])
print(f"  Optimized on A: {map_opt_a:.4f}", flush=True)
print(f"  Evaluated on B: {map_eval_b:.4f} (base B: {map_base_b:.4f}, delta: {map_eval_b - map_base_b:+.4f})", flush=True)

# Optimize on B, evaluate on A
print("\n  --- Optimize on B, evaluate on A ---", flush=True)
tau_b, map_opt_b = optimize_tau(oof[idx_b], y[idx_b], priors, label="B->A")
oof_adj_a = apply_logit_adj_with_tau(oof[idx_a], priors, tau_b)
map_eval_a, per_eval_a = compute_map(y[idx_a], oof_adj_a)
map_base_a, _ = compute_map(y[idx_a], oof[idx_a])
print(f"  Optimized on B: {map_opt_b:.4f}", flush=True)
print(f"  Evaluated on A: {map_eval_a:.4f} (base A: {map_base_a:.4f}, delta: {map_eval_a - map_base_a:+.4f})", flush=True)

# Average honest score
honest_avg = (map_eval_a + map_eval_b) / 2
honest_base_avg = (map_base_a + map_base_b) / 2
honest_delta = honest_avg - honest_base_avg
print(f"\n  Honest split-half average: {honest_avg:.4f} (delta: {honest_delta:+.4f})", flush=True)

# Print optimized taus
print(f"\n  Tau from A: {np.array2string(tau_a, precision=2, suppress_small=True)}", flush=True)
print(f"  Tau from B: {np.array2string(tau_b, precision=2, suppress_small=True)}", flush=True)
tau_diff = np.abs(tau_a - tau_b)
print(f"  Tau agreement (|A-B|): {np.array2string(tau_diff, precision=2)}", flush=True)
print(f"  Mean |tau_A - tau_B|: {tau_diff.mean():.3f}", flush=True)

# ======================================================================
# 2. Fixed Tau Baselines
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("FIXED TAU BASELINES", flush=True)
print("=" * 60, flush=True)

# tau = 0 (no adjustment)
map_t0, _ = compute_map(y, oof)

# tau = 0.5 uniform
tau_05 = np.full(N_CLASSES, 0.5)
oof_t05 = apply_logit_adj_with_tau(oof, priors, tau_05)
map_t05, _ = compute_map(y, oof_t05)

# tau = 1.0 uniform
tau_10 = np.full(N_CLASSES, 1.0)
oof_t10 = apply_logit_adj_with_tau(oof, priors, tau_10)
map_t10, _ = compute_map(y, oof_t10)

# Rarity-scaled: tau_c = log(N_max/N_c) / log(N_max/N_min)
n_max = counts.max()
n_min = counts.min()
tau_rarity = np.log(n_max / counts) / np.log(n_max / max(n_min, 1))
oof_rarity = apply_logit_adj_with_tau(oof, priors, tau_rarity)
map_rarity, _ = compute_map(y, oof_rarity)

# Per-class optimized on FULL OOF (biased -- for comparison only)
print("\n  Per-class optimized on full OOF (BIASED, reference only):", flush=True)
tau_full, map_full_opt = optimize_tau(oof, y, priors, label="FULL")
oof_full_adj = apply_logit_adj_with_tau(oof, priors, tau_full)
map_full_adj, per_full_adj = compute_map(y, oof_full_adj)

print(f"\n  {'Method':<35s} {'mAP':>7s} {'Delta':>7s} {'Note'}", flush=True)
print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*30}", flush=True)
print(f"  {'tau=0 (no adjustment)':<35s} {map_t0:>7.4f} {0:>+7.4f} Baseline", flush=True)
print(f"  {'tau=0.5 (uniform)':<35s} {map_t05:>7.4f} {map_t05-map_t0:>+7.4f}", flush=True)
print(f"  {'tau=1.0 (uniform)':<35s} {map_t10:>7.4f} {map_t10-map_t0:>+7.4f}", flush=True)
print(f"  {'tau=rarity-scaled':<35s} {map_rarity:>7.4f} {map_rarity-map_t0:>+7.4f} tau=log(Nmax/Nc)/log(Nmax/Nmin)", flush=True)
print(f"  {'Split-half honest avg':<35s} {honest_avg:>7.4f} {honest_delta:>+7.4f} Unbiased", flush=True)
print(f"  {'Per-class opt (BIASED)':<35s} {map_full_adj:>7.4f} {map_full_adj-map_t0:>+7.4f} OPTIMISTIC -- do not trust", flush=True)

# ======================================================================
# 3. Bootstrap CIs on best methods
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("BOOTSTRAP 95% CIs", flush=True)
print("=" * 60, flush=True)

methods = [
    ("tau=0 (baseline)", oof),
    ("tau=0.5 (uniform)", oof_t05),
    ("tau=1.0 (uniform)", oof_t10),
    ("tau=rarity-scaled", oof_rarity),
]

for name, preds in methods:
    bs = bootstrap_map_ci(y, preds, n_bootstrap=2000, ci=0.95, seed=42)
    print(f"  {name:<25s}: {bs['mean']:.4f} +/- {bs['std']:.4f}  "
          f"[{bs['ci_lo']:.4f} - {bs['ci_hi']:.4f}]", flush=True)

# ======================================================================
# 4. Recommendation
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("RECOMMENDATION", flush=True)
print("=" * 60, flush=True)

# Compare best fixed tau vs baseline
best_fixed_name = "tau=0"
best_fixed_map = map_t0
for name, m in [("tau=0.5", map_t05), ("tau=1.0", map_t10), ("rarity-scaled", map_rarity)]:
    if m > best_fixed_map:
        best_fixed_map = m
        best_fixed_name = name

fixed_delta = best_fixed_map - map_t0

if honest_delta < 0.005 and fixed_delta < 0.005:
    print("  VERDICT: Logit adjustment provides < 0.005 gain.", flush=True)
    print("  DROP IT. Use raw E32 predictions.", flush=True)
elif fixed_delta >= honest_delta - 0.002:
    print(f"  VERDICT: Use fixed {best_fixed_name} (delta={fixed_delta:+.4f}).", flush=True)
    print("  Fixed tau matches or exceeds split-half optimized. Simpler = better.", flush=True)
else:
    print(f"  VERDICT: Split-half optimized tau gives +{honest_delta:.4f}.", flush=True)
    print(f"  Best fixed tau ({best_fixed_name}) gives +{fixed_delta:.4f}.", flush=True)
    print("  Use split-half if the delta justifies the complexity.", flush=True)

print(f"\n  Rarity-scaled tau per class:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"    {cls:<15s}: tau={tau_rarity[i]:.3f} (N={counts[i]})", flush=True)

print("\nDone!", flush=True)
