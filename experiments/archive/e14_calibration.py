"""E14: Calibration Techniques (T13 + T14)

T13: Per-class isotonic calibration on OOF predictions.
T14: Per-model temperature scaling before blending (GETS-style).

Both are post-processing on existing base model predictions.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, CLASSES
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5

# Load labels
train_df = load_train()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

# Load E11 stacking OOF
oof_e11 = np.load(ROOT / "oof_e11.npy")
test_e11 = np.load(ROOT / "test_e11.npy")

# Load E12 (logit-adjusted) OOF
oof_e12 = np.load(ROOT / "oof_e12.npy")
test_e12 = np.load(ROOT / "test_e12.npy")

# Load individual base model predictions
oof_e10 = np.load(ROOT / "oof_e10.npy")  # tree ensemble
oof_e08 = np.load(ROOT / "oof_e08.npy")  # MiniRocket
oof_e06 = np.load(ROOT / "oof_e06.npy")  # 1D-CNN
oof_e09 = np.load(ROOT / "oof_e09.npy")  # SVM

test_e10 = np.load(ROOT / "test_e10.npy")
test_e08 = np.load(ROOT / "test_e08.npy")
test_e06 = np.load(ROOT / "test_e06.npy")
test_e09 = np.load(ROOT / "test_e09.npy")

base_map, base_per = compute_map(y, oof_e12)
print_results(base_map, base_per, "E12 Baseline (logit-adjusted)")

# ══════════════════════════════════════════════════════════════════
# T13: Per-class Isotonic Calibration
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("T13: Per-Class Isotonic Calibration", flush=True)
print(f"{'='*60}", flush=True)

# Nested CV to avoid data leakage — calibrate within held-out folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Apply to E12 (our current best)
oof_iso = np.zeros_like(oof_e12)
test_iso = np.zeros_like(test_e12)

for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_e12, y)):
    for c in range(N_CLASSES):
        # Fit isotonic on train fold, predict on val fold
        ir = IsotonicRegression(out_of_bounds="clip")
        y_binary = (y[tr_idx] == c).astype(float)
        ir.fit(oof_e12[tr_idx, c], y_binary)

        oof_iso[va_idx, c] = ir.predict(oof_e12[va_idx, c])
        test_iso[:, c] += ir.predict(test_e12[:, c]) / N_FOLDS

    fold_map, _ = compute_map(y[va_idx], oof_iso[va_idx])
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

# Renormalize
row_sums = oof_iso.sum(axis=1, keepdims=True)
oof_iso = oof_iso / np.maximum(row_sums, 1e-10)
row_sums = test_iso.sum(axis=1, keepdims=True)
test_iso = test_iso / np.maximum(row_sums, 1e-10)

iso_map, iso_per = compute_map(y, oof_iso)
print_results(iso_map, iso_per, "T13 Isotonic Calibration on E12")

print(f"\nDelta vs E12:", flush=True)
for cls in CLASSES:
    d = iso_per[cls] - base_per[cls]
    marker = " ***" if abs(d) > 0.01 else ""
    print(f"  {cls:15s}: {base_per[cls]:.4f} -> {iso_per[cls]:.4f} ({d:+.4f}){marker}",
          flush=True)

# Also try on E11 directly (without logit adjustment)
oof_iso_e11 = np.zeros_like(oof_e11)
test_iso_e11 = np.zeros_like(test_e11)

for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_e11, y)):
    for c in range(N_CLASSES):
        ir = IsotonicRegression(out_of_bounds="clip")
        y_binary = (y[tr_idx] == c).astype(float)
        ir.fit(oof_e11[tr_idx, c], y_binary)
        oof_iso_e11[va_idx, c] = ir.predict(oof_e11[va_idx, c])
        test_iso_e11[:, c] += ir.predict(test_e11[:, c]) / N_FOLDS

row_sums = oof_iso_e11.sum(axis=1, keepdims=True)
oof_iso_e11 = oof_iso_e11 / np.maximum(row_sums, 1e-10)
row_sums = test_iso_e11.sum(axis=1, keepdims=True)
test_iso_e11 = test_iso_e11 / np.maximum(row_sums, 1e-10)

iso_e11_map, iso_e11_per = compute_map(y, oof_iso_e11)
print(f"\nIsotonic on E11 (no logit adj): mAP={iso_e11_map:.4f} (E11 was 0.7396)", flush=True)


# ══════════════════════════════════════════════════════════════════
# T14: Per-Model Temperature Scaling (GETS-style)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("T14: Per-Model Temperature Scaling", flush=True)
print(f"{'='*60}", flush=True)

# The idea: each base model has different calibration.
# Apply temperature T_i to each model's logits before blending.
# logits_i = log(probs_i + eps), scaled_probs_i = softmax(logits_i / T_i)

def apply_temperature(probs, T):
    """Apply temperature scaling to probability predictions."""
    eps = 1e-10
    logits = np.log(probs + eps)
    scaled = logits / max(T, 0.01)
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def temp_ensemble_loss(params, model_oofs, y):
    """Negative mAP for temperature + weight optimization."""
    n_models = len(model_oofs)
    temps = params[:n_models]
    weights = np.abs(params[n_models:])
    weights = weights / weights.sum()

    blended = np.zeros_like(model_oofs[0])
    for i, oof in enumerate(model_oofs):
        blended += weights[i] * apply_temperature(oof, temps[i])

    m, _ = compute_map(y, blended)
    return -m  # minimize negative mAP


model_oofs = [oof_e10, oof_e08, oof_e06, oof_e09]
model_tests = [test_e10, test_e08, test_e06, test_e09]
model_names = ["E10_tree", "E08_rocket", "E06_cnn", "E09_svm"]

# Grid search for temperatures (simpler and more robust than scipy.optimize)
print("\nGrid search over per-model temperatures...", flush=True)
best_map_t14 = 0
best_params = None

temp_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
weight_configs = [
    (0.70, 0.10, 0.10, 0.10),  # E11 original
    (0.65, 0.15, 0.10, 0.10),
    (0.60, 0.15, 0.15, 0.10),
    (0.75, 0.10, 0.05, 0.10),
    (0.65, 0.10, 0.15, 0.10),
    (0.70, 0.15, 0.10, 0.05),
]

# First: optimize temperatures at fixed weights
for t0 in temp_values:
    for t1 in temp_values:
        for t2 in temp_values:
            for t3 in temp_values:
                temps = [t0, t1, t2, t3]
                blended = np.zeros_like(oof_e10)
                weights = [0.70, 0.10, 0.10, 0.10]
                for i, oof in enumerate(model_oofs):
                    blended += weights[i] * apply_temperature(oof, temps[i])
                m, _ = compute_map(y, blended)
                if m > best_map_t14:
                    best_map_t14 = m
                    best_params = (temps, weights)

print(f"\nBest temps (fixed E11 weights): mAP={best_map_t14:.4f}", flush=True)
print(f"  Temps: {best_params[0]}", flush=True)

# Second: also optimize weights at best temperatures
best_temps = best_params[0]
for weights in weight_configs:
    blended = np.zeros_like(oof_e10)
    for i, oof in enumerate(model_oofs):
        blended += weights[i] * apply_temperature(oof, best_temps[i])
    m, _ = compute_map(y, blended)
    if m > best_map_t14:
        best_map_t14 = m
        best_params = (best_temps, list(weights))

# Also try wider weight search at best temps
for w0 in np.arange(0.50, 0.85, 0.05):
    for w1 in np.arange(0.05, 0.30, 0.05):
        for w2 in np.arange(0.05, 0.30, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.05:
                continue
            weights = [w0, w1, w2, w3]
            blended = np.zeros_like(oof_e10)
            for i, oof in enumerate(model_oofs):
                blended += weights[i] * apply_temperature(oof, best_temps[i])
            m, _ = compute_map(y, blended)
            if m > best_map_t14:
                best_map_t14 = m
                best_params = (best_temps, weights)

print(f"\nBest overall: mAP={best_map_t14:.4f}", flush=True)
print(f"  Temps: {best_params[0]}", flush=True)
print(f"  Weights: {best_params[1]}", flush=True)
for i, name in enumerate(model_names):
    print(f"    {name}: T={best_params[0][i]:.1f}, w={best_params[1][i]:.2f}", flush=True)

# Apply best params
oof_t14 = np.zeros_like(oof_e10)
test_t14 = np.zeros_like(test_e10)
for i in range(len(model_oofs)):
    oof_t14 += best_params[1][i] * apply_temperature(model_oofs[i], best_params[0][i])
    test_t14 += best_params[1][i] * apply_temperature(model_tests[i], best_params[0][i])

t14_map, t14_per = compute_map(y, oof_t14)
print_results(t14_map, t14_per, "T14 Temperature-Scaled Ensemble")

# ── Now apply T08 logit adjustment on top of T14 ─────────────────
print(f"\n{'='*60}", flush=True)
print("T14 + T08: Temperature scaling + logit adjustment", flush=True)
print(f"{'='*60}", flush=True)

counts = np.bincount(y, minlength=N_CLASSES)
priors = counts / counts.sum()

best_combo_map = t14_map
best_combo_tau = np.zeros(N_CLASSES)

# Per-class tau sweep on top of T14
per_class_tau = np.zeros(N_CLASSES)
current_best = t14_map

for iteration in range(3):
    improved = False
    for c in range(N_CLASSES):
        best_c_map = current_best
        best_c_tau = per_class_tau[c]

        for tau_c in np.arange(-0.5, 1.51, 0.02):
            per_class_tau[c] = tau_c
            adjustment = priors ** (-per_class_tau)
            adjusted = oof_t14 * adjustment[np.newaxis, :]
            adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
            m, _ = compute_map(y, adjusted)
            if m > best_c_map:
                best_c_map = m
                best_c_tau = tau_c

        per_class_tau[c] = best_c_tau
        if best_c_map > current_best:
            current_best = best_c_map
            improved = True

    print(f"  Round {iteration + 1}: mAP={current_best:.4f}", flush=True)
    if not improved:
        break

adjustment = priors ** (-per_class_tau)
oof_combo = oof_t14 * adjustment[np.newaxis, :]
oof_combo = oof_combo / oof_combo.sum(axis=1, keepdims=True)
test_combo = test_t14 * adjustment[np.newaxis, :]
test_combo = test_combo / test_combo.sum(axis=1, keepdims=True)

combo_map, combo_per = compute_map(y, oof_combo)
print_results(combo_map, combo_per, "T14+T08 Combined")

# ══════════════════════════════════════════════════════════════════
# Final comparison
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("FINAL COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  E11 Baseline:              0.7396", flush=True)
print(f"  E12 Logit adjustment:      {base_map:.4f} (+{base_map - 0.7396:.4f})", flush=True)
print(f"  T13 Isotonic on E12:       {iso_map:.4f} ({iso_map - 0.7396:+.4f})", flush=True)
print(f"  T13 Isotonic on E11:       {iso_e11_map:.4f} ({iso_e11_map - 0.7396:+.4f})", flush=True)
print(f"  T14 Temp scaling:          {t14_map:.4f} ({t14_map - 0.7396:+.4f})", flush=True)
print(f"  T14+T08 Combined:         {combo_map:.4f} ({combo_map - 0.7396:+.4f})", flush=True)

# Pick best
candidates = [
    ("E12_logit", base_map, oof_e12, test_e12, base_per),
    ("T13_isotonic_e12", iso_map, oof_iso, test_iso, iso_per),
    ("T13_isotonic_e11", iso_e11_map, oof_iso_e11, test_iso_e11, iso_e11_per),
    ("T14_temp", t14_map, oof_t14, test_t14, t14_per),
    ("T14+T08_combo", combo_map, oof_combo, test_combo, combo_per),
]
candidates.sort(key=lambda x: -x[1])
best_name, best_map, best_oof, best_test, best_per = candidates[0]

print(f"\nBest: {best_name} ({best_map:.4f})", flush=True)
print_results(best_map, best_per, f"E14 Best ({best_name})")

np.save(ROOT / "oof_e14.npy", best_oof)
np.save(ROOT / "test_e14.npy", best_test)
print(f"\nSaved oof_e14.npy and test_e14.npy", flush=True)

if best_map > base_map:
    save_submission(best_test, f"e14_{best_name}", cv_map=best_map)
else:
    print(f"No improvement over E12 ({base_map:.4f}). No submission saved.", flush=True)
