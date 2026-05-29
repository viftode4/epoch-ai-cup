"""E176 Phase B FAST: Quick validation of new features with LOMO.

Just trains 1 LGB GBDT (fast) with 3 seeds to check if new features help.
No DART, no ablations, no blending. Pure signal check.
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.metrics import compute_map
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent

print("=" * 70)
print("  E176 Phase B FAST: New Feature Validation")
print("=" * 70)
t0 = time.time()

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

X_base_tr = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_base_te = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def eval_skf_lomo(oof, name=""):
    skf, _ = compute_map(y, oof)
    lomo = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            lomo[m] = s
    lomo_avg = np.mean(list(lomo.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo.items()))
    print(f"  {name:<40s} SKF={skf:.4f} LOMO={lomo_avg:.4f} [{month_str}]")
    return skf, lomo_avg


def fast_train(X_tr, X_te, n_seeds=3):
    """Fast GBDT training, 3 seeds, returns OOF + test."""
    oof_all = np.zeros((n_seeds, len(y), N_CLASSES))
    test_all = np.zeros((n_seeds, X_te.shape[0], N_CLASSES))
    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((len(y), N_CLASSES))
        test_s = np.zeros((X_te.shape[0], N_CLASSES))
        for fold, (tr, va) in enumerate(sgkf.split(X_tr, y, groups)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                is_unbalance=True, verbosity=-1, random_state=42+seed+fold, n_jobs=-1,
            )
            m.fit(X_tr[tr], y[tr], eval_set=[(X_tr[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_s[va] = m.predict_proba(X_tr[va])
            test_s += m.predict_proba(X_te) / 5
        oof_all[seed] = oof_s
        test_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"    seed {seed+1}: {s:.4f}", flush=True)
    return np.mean(oof_all, axis=0), np.mean(test_all, axis=0)


# ── Extract new features (fast, ~10s total) ──
print("\nExtracting new features...", flush=True)

# B1: Flock intensity
print("  B1 flock...", flush=True)
ts_tr = pd.to_datetime(train_df["timestamp_start_radar_utc"]).values.astype(np.int64)/1e9
ts_te = pd.to_datetime(test_df["timestamp_start_radar_utc"]).values.astype(np.int64)/1e9

def flock_count(df, ts):
    n = len(df)
    lons, lats = np.zeros(n), np.zeros(n)
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(r["trajectory"])
            if pts:
                lons[i] = np.mean([p[0] for p in pts])
                lats[i] = np.mean([p[1] for p in pts])
        except: pass
    order = np.argsort(ts)
    ts_s = ts[order]
    counts = np.zeros(n)
    for ii in range(n):
        i = order[ii]
        lo = np.searchsorted(ts_s, ts[i]-60)
        hi = np.searchsorted(ts_s, ts[i]+60, side='right')
        for jj in range(lo, hi):
            j = order[jj]
            if j == i: continue
            d = np.sqrt(((lats[i]-lats[j])*111000)**2 + ((lons[i]-lons[j])*67000)**2)
            if d <= 500: counts[i] += 1
    return counts

fc_tr = flock_count(train_df, ts_tr)
fc_te = flock_count(test_df, ts_te)

# B2: Glide ratio
print("  B2 glide...", flush=True)
def glide(df):
    n = len(df)
    gr = np.zeros(n)
    gf = np.zeros(n)
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(r["trajectory"])
            if len(pts) < 3: continue
            alts = np.array([p[2] for p in pts])
            lons = np.array([p[0] for p in pts])
            lats = np.array([p[1] for p in pts])
            da = np.diff(alts)
            desc = da < -0.5
            if desc.sum() < 2: continue
            dx = np.diff(lons)*67000; dy = np.diff(lats)*111000
            hd = np.sqrt(dx**2+dy**2)
            dh = hd[desc].sum(); dv = np.abs(da[desc]).sum()
            if dv > 1.0: gr[i] = dh/dv
            gf[i] = desc.sum()/len(da)
        except: continue
    return gr, gf

gr_tr, gf_tr = glide(train_df)
gr_te, gf_te = glide(test_df)

# B3: Session time
print("  B3 session...", flush=True)
def sess_time(df):
    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    hod = (ts.dt.hour + ts.dt.minute/60.0).values
    dates = ts.dt.date
    srt = np.zeros(len(df))
    for d in dates.unique():
        m = dates == d
        srt[m.values] = (ts[m] - ts[m].min()).dt.total_seconds().values / 3600.0
    return hod, srt

hod_tr, srt_tr = sess_time(train_df)
hod_te, srt_te = sess_time(test_df)

# B4: Wing loading
print("  B4 wing loading...", flush=True)
def wing_load(df):
    n = len(df)
    mp = np.zeros(n); wl = np.zeros(n)
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(r["trajectory"])
            if len(pts) < 2: continue
            rcs = np.mean([p[3] for p in pts])
            rl = 10**(rcs/10.0)
            mass = max(rl**1.5, 1e-6)
            mp[i] = mass; wl[i] = mass**0.28 * 25.3
        except: continue
    return mp, wl

mp_tr, wl_tr = wing_load(train_df)
mp_te, wl_te = wing_load(test_df)

# Stack new features
new_tr = np.column_stack([fc_tr, gr_tr, gf_tr, hod_tr, srt_tr, mp_tr, wl_tr])
new_te = np.column_stack([fc_te, gr_te, gf_te, hod_te, srt_te, mp_te, wl_te])
new_names = ["flock_count", "glide_ratio", "glide_frac", "hour_of_day", "session_rel_hours", "mass_proxy", "wing_loading"]

X_aug_tr = np.hstack([X_base_tr, np.nan_to_num(new_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)])
X_aug_te = np.hstack([X_base_te, np.nan_to_num(new_te, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)])

print(f"\n  Base: {X_base_tr.shape[1]} features, New: {len(new_names)}, Total: {X_aug_tr.shape[1]}")
print(f"  New features: {new_names}")
print(f"  Feature extraction: {time.time()-t0:.0f}s\n")

# ── Train & Compare ──
print("=" * 70)
print("  Training GBDT (fast, 3 seeds)")
print("=" * 70)

print("\n[1] Base features (100):")
oof_base, _ = fast_train(X_base_tr, X_base_te)
eval_skf_lomo(oof_base, "Base (100 features)")

print("\n[2] Augmented features (100 + 7 new):")
oof_aug, test_aug = fast_train(X_aug_tr, X_aug_te)
eval_skf_lomo(oof_aug, "Augmented (107 features)")

print("\n[3] New features only (7):")
oof_new, _ = fast_train(new_tr.astype(np.float32), new_te.astype(np.float32))
eval_skf_lomo(oof_new, "New only (7 features)")

# Quick feature importance
print("\n--- New Feature Importance ---")
m = lgb.LGBMClassifier(
    objective="multiclass", num_class=N_CLASSES, boosting_type="gbdt",
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    is_unbalance=True, verbosity=-1, n_jobs=-1,
)
m.fit(X_aug_tr, y)
imp = m.feature_importances_
all_names = selected + new_names
for i, name in enumerate(all_names):
    if name in new_names:
        print(f"  {name:25s}: {imp[i]:6d} (rank {sorted(imp, reverse=True).index(imp[i])+1}/{len(all_names)})")

# Blend with E175
print("\n--- Blend with E175 ---")
oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
for alpha in [0.2, 0.3, 0.5]:
    blend = (1-alpha) * oof_e175 + alpha * renorm_rows(oof_aug)
    eval_skf_lomo(renorm_rows(blend), f"E175 + aug@{alpha}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
