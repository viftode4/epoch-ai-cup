"""E184 Section 4 fix — Physics prior on probability-based blends."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.data import CLASSES, load_train, parse_ewkb_4d, parse_trajectory_time
from src.metrics import compute_map
from src.features import haversine

N_CLASSES = len(CLASSES)
CORM_IDX = CLASSES.index("Cormorants")
WADER_IDX = CLASSES.index("Waders")

# --- Data ---
train_df = load_train()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
n = len(y)


def lomo_map(y, preds, months):
    scores = []
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() < 5 or len(set(y[mask])) < 2:
            continue
        score, _ = compute_map(y[mask], preds[mask])
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def lomo_per_class(y, preds, months):
    class_aps = {c: [] for c in range(N_CLASSES)}
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() < 5:
            continue
        y_bin = np.eye(N_CLASSES)[y[mask]]
        for c in range(N_CLASSES):
            if y_bin[:, c].sum() > 0:
                ap = average_precision_score(y_bin[:, c], preds[mask, c])
                class_aps[c].append(ap)
    return {CLASSES[c]: float(np.mean(class_aps[c])) if class_aps[c] else 0.0 for c in range(N_CLASSES)}


# --- Load models ---
models = {}
for name in ["e175_best", "e175_ranker", "e175_cb", "e183_tabpfn", "e182_cnn_v3",
             "e175_lgb", "e175_xgb", "e175_dro", "e176_iso", "e180_spatial", "e180_rcs_linear"]:
    path = ROOT / f"oof_{name}.npy"
    if path.exists():
        arr = np.load(path).astype(float)
        if arr.shape == (n, N_CLASSES):
            models[name] = arr

igk = np.load(ROOT / "oof_e176_iso_gmm_knn.npy").astype(float)

# --- Build Frankenstein (per-class optimal from Section 3) ---
frank_config = {
    0: ("e176_iso", "e180_spatial", 0.5),
    1: ("e183_tabpfn", "e176_iso", 0.5),
    2: ("e176_iso", "e180_spatial", 0.4),
    3: ("e182_cnn_v3", "e176_iso", 0.3),
    4: ("e183_tabpfn", "e176_iso", 0.6),
    5: ("e183_tabpfn", "e176_iso", 0.6),
    6: ("e175_ranker", "e180_rcs_linear", 0.1),
    7: ("e183_tabpfn", "e176_iso", 0.4),
    8: ("e183_tabpfn", "e176_iso", 0.7),
}
frank = np.zeros((n, N_CLASSES))
for c, (a, b, w) in frank_config.items():
    frank[:, c] = w * models[a][:, c] + (1 - w) * models[b][:, c]
frank = np.clip(frank, 1e-8, None)
frank /= frank.sum(axis=1, keepdims=True)

# --- Parse trajectories ---
print("Parsing trajectories for physics features...")
speed_vals = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
min_z = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
max_z = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
alt_mid = 0.5 * (min_z + max_z)
alt_range = max_z - min_z

straightness = np.zeros(n)
rcs_mean_arr = np.zeros(n)
rcs_std_arr = np.zeros(n)
rcs_ac1_arr = np.zeros(n)
speed_cv_arr = np.zeros(n)

for idx in range(n):
    try:
        pts = parse_ewkb_4d(train_df.iloc[idx]["trajectory"])
        times = parse_trajectory_time(train_df.iloc[idx]["trajectory_time"])
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        rcs = np.array([p[3] for p in pts])
        if len(pts) > 1:
            total_dist = sum(
                haversine(lons[i], lats[i], lons[i + 1], lats[i + 1])
                for i in range(len(pts) - 1)
            )
            straight_dist = haversine(lons[0], lats[0], lons[-1], lats[-1])
            straightness[idx] = straight_dist / max(total_dist, 1e-6)
        rcs_mean_arr[idx] = np.mean(rcs)
        rcs_std_arr[idx] = np.std(rcs)
        if len(rcs) > 2:
            rcs_c = rcs - np.mean(rcs)
            v = np.var(rcs)
            rcs_ac1_arr[idx] = np.mean(rcs_c[:-1] * rcs_c[1:]) / v if v > 1e-10 else 0
        if len(pts) > 2:
            dists = np.array([
                haversine(lons[i], lats[i], lons[i + 1], lats[i + 1])
                for i in range(len(pts) - 1)
            ])
            dt = np.maximum(np.diff(times), 0.001)
            seg_speeds = dists / dt
            if np.mean(seg_speeds) > 1e-6:
                speed_cv_arr[idx] = np.std(seg_speeds) / np.mean(seg_speeds)
    except Exception:
        pass

print(f"Parsed {n} trajectories.\n")

# --- Cormorant physics likelihood ---
def corm_lik(speed, straight, alt, rcs_m, rcs_s, rcs_ac1, spd_cv):
    s = []
    s.append(np.exp(-0.5 * ((speed - 15.5) / 3.0) ** 2))
    s.append(1.0 / (1.0 + np.exp(-10 * (straight - 0.80))))
    s.append(1.0 / (1.0 + np.exp(0.1 * (alt - 60))))
    s.append(1.0 / (1.0 + np.exp(-0.5 * (rcs_m - (-27)))))
    s.append(1.0 / (1.0 + np.exp(-1.0 * (rcs_s - 2.0))))
    s.append(1.0 / (1.0 + np.exp(-10 * (rcs_ac1 - 0.30))))
    s.append(1.0 / (1.0 + np.exp(10 * (spd_cv - 0.30))))
    return np.prod(s, axis=0)


phys = corm_lik(speed_vals, straightness, alt_mid, rcs_mean_arr, rcs_std_arr, rcs_ac1_arr, speed_cv_arr)
corm_mask = y == CORM_IDX
print(f"Cormorant physics likelihood:")
print(f"  Cormorant mean: {phys[corm_mask].mean():.4f}")
print(f"  Non-Cormorant:  {phys[~corm_mask].mean():.4f}")
print(f"  Ratio:          {phys[corm_mask].mean() / max(phys[~corm_mask].mean(), 1e-8):.2f}x")

# --- Test on multiple base predictions ---
bases = {
    "Frankenstein": frank,
    "e176_iso_gmm_knn": igk,
    "e175_best": models["e175_best"],
    "e176_iso": models["e176_iso"],
}

gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

for base_name, base_preds in bases.items():
    base_lomo = lomo_map(y, base_preds, months)
    base_pc = lomo_per_class(y, base_preds, months)
    print(f"\n--- Cormorant physics prior on {base_name} (LOMO={base_lomo:.4f}, Corm={base_pc['Cormorants']:.4f}) ---")
    print(f"  {'Gamma':>7}  {'LOMO':>7}  {'dLOMO':>7}  {'Corm_AP':>8}  {'dCorm':>7}")
    for gamma in gammas:
        boosted = base_preds.copy()
        boosted[:, CORM_IDX] *= (1.0 + gamma * phys)
        boosted = np.clip(boosted, 1e-8, None)
        boosted /= boosted.sum(axis=1, keepdims=True)
        new_lomo = lomo_map(y, boosted, months)
        new_pc = lomo_per_class(y, boosted, months)
        d_lomo = new_lomo - base_lomo
        d_corm = new_pc["Cormorants"] - base_pc["Cormorants"]
        print(f"  {gamma:7.1f}  {new_lomo:.4f}  {d_lomo:+.4f}  {new_pc['Cormorants']:.4f}  {d_corm:+.4f}")

# --- Wader physics likelihood ---
def wader_lik(speed, straight, alt, altr, rcs_m):
    s = []
    s.append(np.exp(-0.5 * ((speed - 15) / 5.0) ** 2))
    s.append(1.0 / (1.0 + np.exp(-0.05 * (altr - 30))))
    s.append(1.0 / (1.0 + np.exp(-8 * (straight - 0.70))))
    s.append(1.0 / (1.0 + np.exp(0.5 * (rcs_m - (-25)))))
    return np.prod(s, axis=0)


w_phys = wader_lik(speed_vals, straightness, alt_mid, alt_range, rcs_mean_arr)
w_mask = y == WADER_IDX
print(f"\nWader physics likelihood:")
print(f"  Wader mean:     {w_phys[w_mask].mean():.4f}")
print(f"  Non-Wader:      {w_phys[~w_mask].mean():.4f}")
print(f"  Ratio:          {w_phys[w_mask].mean() / max(w_phys[~w_mask].mean(), 1e-8):.2f}x")

for base_name, base_preds in bases.items():
    base_lomo = lomo_map(y, base_preds, months)
    base_pc = lomo_per_class(y, base_preds, months)
    print(f"\n--- Wader physics prior on {base_name} (LOMO={base_lomo:.4f}, Wader={base_pc['Waders']:.4f}) ---")
    print(f"  {'Gamma':>7}  {'LOMO':>7}  {'dLOMO':>7}  {'Wader_AP':>8}  {'dWader':>7}")
    for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
        boosted = base_preds.copy()
        boosted[:, WADER_IDX] *= (1.0 + gamma * w_phys)
        boosted = np.clip(boosted, 1e-8, None)
        boosted /= boosted.sum(axis=1, keepdims=True)
        new_lomo = lomo_map(y, boosted, months)
        new_pc = lomo_per_class(y, boosted, months)
        d_lomo = new_lomo - base_lomo
        d_wader = new_pc["Waders"] - base_pc["Waders"]
        print(f"  {gamma:7.1f}  {new_lomo:.4f}  {d_lomo:+.4f}  {new_pc['Waders']:.4f}  {d_wader:+.4f}")

# --- Combined Cormorant + Wader boost ---
print("\n--- Combined Cormorant + Wader boost on Frankenstein ---")
base_lomo = lomo_map(y, frank, months)
base_pc = lomo_per_class(y, frank, months)
print(f"  Baseline LOMO={base_lomo:.4f}")
for cg in [0.5, 1.0, 2.0, 5.0]:
    for wg in [0.5, 1.0, 2.0]:
        boosted = frank.copy()
        boosted[:, CORM_IDX] *= (1.0 + cg * phys)
        boosted[:, WADER_IDX] *= (1.0 + wg * w_phys)
        boosted = np.clip(boosted, 1e-8, None)
        boosted /= boosted.sum(axis=1, keepdims=True)
        new_lomo = lomo_map(y, boosted, months)
        new_pc = lomo_per_class(y, boosted, months)
        print(f"  cg={cg:.1f} wg={wg:.1f}  LOMO={new_lomo:.4f}  d={new_lomo-base_lomo:+.4f}  "
              f"Corm={new_pc['Cormorants']:.4f} Wader={new_pc['Waders']:.4f}")

print("\nDone.")
