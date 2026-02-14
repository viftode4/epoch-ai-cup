"""E12: Post-hoc Logit Adjustment (T08)

Zero-cost post-processing on existing E11 stacking predictions.
Method: adjusted = probs * (prior ** -tau), then renormalize.
Sweep tau on OOF to find optimal boost for minority classes.

Reference: Menon et al. 2021, "Long-tail learning via logit adjustment"
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

# ── Load E11 OOF + test predictions ─────────────────────────────
print("Loading E11 predictions...", flush=True)
oof_e11 = np.load(ROOT / "oof_e11.npy")
test_e11 = np.load(ROOT / "test_e11.npy")

train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Baseline
base_map, base_per = compute_map(y, oof_e11)
print_results(base_map, base_per, "E11 Baseline (no adjustment)")

# ── Compute class priors ─────────────────────────────────────────
counts = np.bincount(y, minlength=N_CLASSES)
priors = counts / counts.sum()
print("\nClass priors:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:15s}: {priors[i]:.4f} (n={counts[i]})", flush=True)


def logit_adjust(probs, priors, tau):
    """Apply post-hoc logit adjustment: probs * prior^(-tau), renormalize."""
    adjustment = priors ** (-tau)
    adjusted = probs * adjustment[np.newaxis, :]
    # Renormalize to sum to 1
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted


# ── Sweep tau ────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Sweeping tau on OOF predictions", flush=True)
print(f"{'='*60}", flush=True)

best_map = base_map
best_tau = 0.0
results = []

for tau in np.arange(-0.5, 1.01, 0.02):
    adjusted = logit_adjust(oof_e11, priors, tau)
    m, per = compute_map(y, adjusted)
    results.append((tau, m, per))
    if m > best_map:
        best_map = m
        best_tau = tau

print(f"\nBest tau: {best_tau:.2f} -> mAP: {best_map:.4f} (baseline: {base_map:.4f}, delta: {best_map - base_map:+.4f})",
      flush=True)

# Show results around the optimum
print(f"\nTau sweep results (top 10):", flush=True)
results.sort(key=lambda x: -x[1])
for tau, m, per in results[:10]:
    print(f"  tau={tau:+.2f}: mAP={m:.4f} ({m - base_map:+.4f})", flush=True)

# ── Per-class analysis at best tau ───────────────────────────────
best_adjusted = logit_adjust(oof_e11, priors, best_tau)
best_adj_map, best_adj_per = compute_map(y, best_adjusted)
print_results(best_adj_map, best_adj_per, f"E12 Logit Adjustment (tau={best_tau:.2f})")

# Compare per-class
print(f"\nPer-class delta vs E11:", flush=True)
for cls in CLASSES:
    delta = best_adj_per[cls] - base_per[cls]
    marker = " ***" if abs(delta) > 0.01 else ""
    print(f"  {cls:15s}: {base_per[cls]:.4f} -> {best_adj_per[cls]:.4f} ({delta:+.4f}){marker}",
          flush=True)

# ── Also try per-class tau (different adjustment per class) ──────
print(f"\n{'='*60}", flush=True)
print("Per-class tau optimization (greedy)", flush=True)
print(f"{'='*60}", flush=True)

# Start from the global best
per_class_tau = np.full(N_CLASSES, best_tau)
current_best = best_map

for iteration in range(3):  # 3 rounds of greedy optimization
    improved = False
    for c in range(N_CLASSES):
        orig_tau = per_class_tau[c]
        best_c_tau = orig_tau
        best_c_map = current_best

        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adjustment = priors ** (-per_class_tau)
            adjusted = oof_e11 * adjustment[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c

        per_class_tau[c] = best_c_tau
        if best_c_map > current_best:
            current_best = best_c_map
            improved = True

    print(f"  Round {iteration + 1}: mAP={current_best:.4f} ({current_best - base_map:+.4f})",
          flush=True)
    if not improved:
        break

print(f"\nPer-class tau values:", flush=True)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:15s}: tau={per_class_tau[i]:.2f}", flush=True)

# Apply per-class tau
adjustment_pc = priors ** (-per_class_tau)
oof_pc = oof_e11 * adjustment_pc[np.newaxis, :]
oof_pc = oof_pc / oof_pc.sum(axis=1, keepdims=True)
test_pc = test_e11 * adjustment_pc[np.newaxis, :]
test_pc = test_pc / test_pc.sum(axis=1, keepdims=True)

pc_map, pc_per = compute_map(y, oof_pc)
print_results(pc_map, pc_per, "E12 Per-Class Logit Adjustment")

# ── Pick best method ─────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E11 Baseline:       {base_map:.4f}", flush=True)
print(f"  Global tau={best_tau:.2f}:    {best_adj_map:.4f} ({best_adj_map - base_map:+.4f})", flush=True)
print(f"  Per-class tau:      {pc_map:.4f} ({pc_map - base_map:+.4f})", flush=True)

if pc_map > best_adj_map and pc_map > base_map:
    print(f"\nPer-class tau wins!", flush=True)
    final_oof = oof_pc
    final_test = test_pc
    final_map = pc_map
    final_per = pc_per
    method = "per_class_tau"
elif best_adj_map > base_map:
    print(f"\nGlobal tau wins!", flush=True)
    final_oof = best_adjusted
    final_test = logit_adjust(test_e11, priors, best_tau)
    final_map = best_adj_map
    final_per = best_adj_per
    method = f"global_tau_{best_tau:.2f}"
else:
    print(f"\nNo improvement from logit adjustment.", flush=True)
    final_oof = oof_e11
    final_test = test_e11
    final_map = base_map
    final_per = base_per
    method = "none"

print_results(final_map, final_per, f"E12 Best ({method})")

np.save(ROOT / "oof_e12.npy", final_oof)
np.save(ROOT / "test_e12.npy", final_test)
print(f"\nSaved oof_e12.npy and test_e12.npy", flush=True)

if final_map > base_map:
    save_submission(final_test, f"e12_logit_adj_{method}", cv_map=final_map)
else:
    print("No submission saved (no improvement).", flush=True)
