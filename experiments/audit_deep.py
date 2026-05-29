"""DEEP PIPELINE AUDIT — trace every step, validate every assumption."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from src.data import load_train, load_test, CLASSES
from src.features_v2 import _load_external_csv
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

print("=" * 80)
print("  DEEP PIPELINE AUDIT")
print("=" * 80)

# ═══ 1. DATA LOADING ═══
print("\n[1] DATA LOADING")
print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")
print(f"  Train months: {dict(zip(*np.unique(months, return_counts=True)))}")
print(f"  Test months:  {dict(zip(*np.unique(test_months, return_counts=True)))}")
print(f"  Class dist: {dict(zip(CLASSES, np.bincount(y)))}")
n_groups = len(set(groups))
gc = Counter(groups)
multi = sum(1 for v in gc.values() if v > 1)
print(f"  Unique groups: {n_groups}, multi-track groups: {multi}")

# ═══ 2. FEATURE ALIGNMENT ═══
print("\n[2] FEATURE ALIGNMENT")
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_v3 = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
print(f"  Train: {train_v3.shape}, Test: {test_v3.shape}")

shared = sorted(set(train_v3.columns) & set(test_v3.columns))
const_either = {c for c in shared if train_v3[c].std() < 1e-10 or test_v3[c].std() < 1e-10}
feature_cols = sorted(set(shared) - const_either)
print(f"  Shared: {len(shared)}, Constant: {len(const_either)}, Usable: {len(feature_cols)}")

# Leaked train-only cols?
TRAIN_ONLY = {"bird_group", "bird_species", "n_birds_observed", "observation_id",
              "primary_observation_id", "observer_position", "observer_comment"}
leaked = [c for c in feature_cols if c in TRAIN_ONLY]
print(f"  Leaked train-only: {leaked if leaked else 'NONE'}")

# NaN/Inf
X_train = train_v3[feature_cols].values.astype(np.float64)
X_test = test_v3[feature_cols].values.astype(np.float64)
print(f"  NaN: train={np.isnan(X_train).sum()}, test={np.isnan(X_test).sum()}")
print(f"  Inf: train={np.isinf(X_train).sum()}, test={np.isinf(X_test).sum()}")

# ═══ 3. EXTERNAL DATA ROW ALIGNMENT ═══
print("\n[3] EXTERNAL DATA ROW ALIGNMENT")
ext_names = ["water", "tidal", "turbines", "cape", "moon", "pressure", "landuse",
             "altitude_winds", "era5_winds", "solar", "weather", "weather_extra",
             "soil", "marine", "photoperiod", "natura2000", "visibility", "insect"]
issues = []
for name in ext_names:
    tr = _load_external_csv(name, "train")
    te = _load_external_csv(name, "test")
    status = "OK"
    if not tr.empty and len(tr) != len(train_df):
        status = f"TRAIN MISMATCH ({len(tr)} vs {len(train_df)})"
        issues.append(f"{name}: {status}")
    if not te.empty and len(te) != len(test_df):
        status = f"TEST MISMATCH ({len(te)} vs {len(test_df)})"
        issues.append(f"{name}: {status}")
    if tr.empty:
        status = "MISSING"
    print(f"    {name:20s}: {status}")
if issues:
    print(f"  !! ISSUES: {issues}")
else:
    print(f"  All external CSVs aligned correctly.")

# ═══ 4. FEATURE GROUP ANALYSIS ═══
print("\n[4] FEATURE GROUP ANALYSIS")
groups_map = {}
for c in feature_cols:
    if c.startswith("lsig_"):
        g = "log_sig"
    elif "_c22_" in c:
        g = "catch22"
    elif c.startswith("phys_"):
        g = "physics"
    elif c in ["alt_curvature", "alt_r2", "speed_cv", "speed_ac1", "speed_trend",
               "predicted_flock_size"] or c.startswith("rcs_ac_lag"):
        g = "new_traj"
    else:
        g = "v2_ext"
    groups_map.setdefault(g, []).append(c)

for gname, cols in groups_map.items():
    # Near-zero features
    nz = [c for c in cols if train_v3[c].abs().max() < 1e-8]
    lv = [c for c in cols if train_v3[c].std() < 0.001 and c not in nz]
    print(f"  {gname:15s}: {len(cols):3d} feats, {len(nz)} near-zero, {len(lv)} low-var")
    if nz:
        print(f"    Near-zero: {nz[:5]}")
    if lv:
        print(f"    Low-var: {lv[:5]}")

# ═══ 5. DISTRIBUTION SHIFT ANALYSIS ═══
print("\n[5] TRAIN vs TEST DISTRIBUTION SHIFT")
for gname, cols in groups_map.items():
    shifts = []
    for c in cols:
        tr_std = train_v3[c].std()
        if tr_std > 1e-10:
            shifts.append(abs(train_v3[c].mean() - test_v3[c].mean()) / tr_std)
    if shifts:
        print(f"  {gname:15s}: avg_shift={np.mean(shifts):.3f}, max={np.max(shifts):.3f}")

# ═══ 6. FEATURE IMPORTANCE & CONTRIBUTION ═══
print("\n[6] FEATURE IMPORTANCE (1-fold quick LGB)")
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
X = np.nan_to_num(X_train.astype(np.float32))
Xt = np.nan_to_num(X_test.astype(np.float32))
tidx, vidx = next(iter(sgkf.split(X, y, groups)))

m = lgb.LGBMClassifier(objective="multiclass", num_class=9, n_estimators=500,
                        verbosity=-1, random_state=42, n_jobs=-1)
m.fit(X[tidx], y[tidx], eval_set=[(X[vidx], y[vidx])],
      callbacks=[lgb.early_stopping(50, verbose=False)])
imp = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)

print(f"\n  Importance by group (% of total):")
for gname, cols in sorted(groups_map.items()):
    total = imp[cols].sum()
    pct = 100 * total / imp.sum()
    n_zero = (imp[cols] == 0).sum()
    print(f"    {gname:15s}: {pct:5.1f}%  ({len(cols)} feats, {n_zero} zero-imp)")

print(f"\n  Zero-importance: {(imp == 0).sum()} / {len(feature_cols)}")

print(f"\n  Top 30 features:")
for i, (feat, val) in enumerate(imp.head(30).items()):
    g = next((gn for gn, cols in groups_map.items() if feat in cols), "?")
    print(f"    {i+1:2d}. {feat:40s} {val:6d}  [{g[:8]}]")

# ═══ 7. ABLATION: NEW FEATURES vs OLD ═══
print("\n[7] ABLATION: Do new features actually help?")
v2_cols = groups_map.get("v2_ext", [])
X_v2 = np.nan_to_num(train_v3[v2_cols].values.astype(np.float32))

m_v2 = lgb.LGBMClassifier(objective="multiclass", num_class=9, n_estimators=500,
                           verbosity=-1, random_state=42, n_jobs=-1)
m_v2.fit(X_v2[tidx], y[tidx], eval_set=[(X_v2[vidx], y[vidx])],
         callbacks=[lgb.early_stopping(50, verbose=False)])
map_v2, pc_v2 = compute_map(y[vidx], m_v2.predict_proba(X_v2[vidx]))

map_all, pc_all = compute_map(y[vidx], m.predict_proba(X[vidx]))
print(f"  v2+ext only ({len(v2_cols)}f): {map_v2:.4f}")
print(f"  All features ({len(feature_cols)}f): {map_all:.4f}")
print(f"  Delta: {map_all - map_v2:+.4f}")

print(f"\n  Per-class delta (ALL - v2only):")
for cls in CLASSES:
    d = pc_all.get(cls, 0) - pc_v2.get(cls, 0)
    marker = " ***" if abs(d) > 0.03 else ""
    print(f"    {cls:15s}: v2={pc_v2[cls]:.4f}  all={pc_all[cls]:.4f}  delta={d:+.4f}{marker}")

# ═══ 8. CATCH22 DEEP CHECK ═══
print("\n[8] CATCH22 DEEP CHECK — are they computing correctly?")
# Sample a few tracks and verify catch22 manually
from src.data import parse_ewkb_4d
from src.features_v3 import _catch22_single

row = train_df.iloc[0]
pts = parse_ewkb_4d(row.trajectory)
alts = np.array([p[2] for p in pts])
c22 = _catch22_single(alts)
print(f"  Sample track (idx=0, n={len(pts)} pts):")
print(f"    alt range: [{alts.min():.1f}, {alts.max():.1f}], std={alts.std():.2f}")
for k, v in sorted(c22.items()):
    print(f"    {k}: {v:.6f}")

# Check if catch22 values are reasonable across all tracks
c22_cols_alt = [c for c in feature_cols if c.startswith("alt_c22_")]
print(f"\n  Altitude catch22 ({len(c22_cols_alt)} feats) stats:")
for c in c22_cols_alt[:5]:
    print(f"    {c}: mean={train_v3[c].mean():.4f}, std={train_v3[c].std():.4f}, "
          f"min={train_v3[c].min():.4f}, max={train_v3[c].max():.4f}")

# ═══ 9. LOG-SIGNATURE DEEP CHECK ═══
print("\n[9] LOG-SIGNATURE DEEP CHECK")
from src.features_v3 import extract_log_signature_features

row = train_df.iloc[0]
lsig = extract_log_signature_features(row.trajectory, row.trajectory_time)
print(f"  Sample track (idx=0): {len(lsig)} features")
# Check for dominated features (all near zero)
lsig_vals = np.array(list(lsig.values()))
print(f"  Range: [{lsig_vals.min():.4f}, {lsig_vals.max():.4f}]")
print(f"  Near-zero (<1e-6): {(np.abs(lsig_vals) < 1e-6).sum()} / {len(lsig_vals)}")

# Across all tracks
lsig_cols = [c for c in feature_cols if c.startswith("lsig_")]
lsig_stds = train_v3[lsig_cols].std()
print(f"\n  Log-sig std distribution:")
print(f"    Min std: {lsig_stds.min():.6f} ({lsig_stds.idxmin()})")
print(f"    Max std: {lsig_stds.max():.6f} ({lsig_stds.idxmax()})")
print(f"    Median std: {lsig_stds.median():.6f}")
print(f"    Feats with std < 0.01: {(lsig_stds < 0.01).sum()}")

# ═══ 10. SUBMISSION FORMAT CHECK ═══
print("\n[10] SUBMISSION FORMAT CHECK")
sample_sub = pd.read_csv(ROOT / "data" / "sample_submission.csv")
print(f"  Sample sub columns: {list(sample_sub.columns)}")
print(f"  Expected order: track_id + {[c for c in sample_sub.columns if c != 'track_id']}")
print(f"  CLASSES order:  {CLASSES}")
# Verify they match
sub_classes = [c for c in sample_sub.columns if c != "track_id"]
print(f"  Match: {sub_classes == CLASSES}")

# ═══ 11. CV HONESTY CHECK ═══
print("\n[11] CV HONESTY CHECK (SGKF group leakage)")
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (ti, vi) in enumerate(sgkf2.split(X, y, groups)):
    tr_groups = set(groups[ti])
    va_groups = set(groups[vi])
    overlap = tr_groups & va_groups
    print(f"  Fold {fold+1}: train_groups={len(tr_groups)}, val_groups={len(va_groups)}, "
          f"overlap={len(overlap)}")

# ═══ 12. PP VALIDATION ═══
print("\n[12] POST-PROCESSING VALIDATION")
from src.postprocessing import UNSEEN_MONTHS, BASE_ALPHA
print(f"  UNSEEN_MONTHS: {UNSEEN_MONTHS}")
print(f"  BASE_ALPHA: {BASE_ALPHA}")
print(f"  Test months in UNSEEN: {sum(np.isin(test_months, UNSEEN_MONTHS))} / {len(test_months)}")
print(f"  Test months shared (9,10): {sum(np.isin(test_months, [9, 10]))} / {len(test_months)}")
# PP should ONLY change unseen months
print(f"  PP affects {100*sum(np.isin(test_months, UNSEEN_MONTHS))/len(test_months):.1f}% of test")

print(f"\n{'='*80}")
print("  AUDIT COMPLETE")
print(f"{'='*80}")
