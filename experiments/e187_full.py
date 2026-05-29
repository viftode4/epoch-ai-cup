"""E187: E175 architecture (full) + ALL improvements we found.

Architecture (from E175, proven LB=0.59):
  - OvR LambdaRank (DART, query-by-month) — 5 seeds
  - CatBoost DRO (worst-month on VALIDATION, not train) — 3 seeds
  - Rank-power ensemble

ALL improvements applied:
  1. ALL 327 features + physics features + interaction features
  2. Noise relabeling (133 agreed noisy -> consensus labels)
  3. Lower min_child_samples=5 for rare classes
  4. Cormorant augmentation (3x copies with noise)
  5. Quality-weighted training (cleanlab scores as sample weights)
  6. DRO worst-month evaluated on VALIDATION (bug fix)
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata
from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from src.metrics import compute_map
from src.submission import save_submission
import lightgbm as lgb
import catboost as cb

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES); FOLDS = 5; SEEDS_OVR = 5; SEEDS_CB = 3

# ══════════════════════════════════════════════════════════════
# DATA + ALL IMPROVEMENTS
# ══════════════════════════════════════════════════════════════
print("=" * 80, flush=True)
print("E187: E175 FULL ARCHITECTURE + ALL IMPROVEMENTS", flush=True)
print("=" * 80, flush=True)

train = load_train(); test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values

# Base features (327)
tf = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
xf = pd.read_pickle(ROOT / "data/_cached_test_features_v3.pkl")

# Add physics + interaction features
print("[1] Computing physics + interaction features...", flush=True)

def compute_extra(df):
    extras = []
    for _, row in df.iterrows():
        pts = parse_ewkb_4d(row['trajectory'])
        times = parse_trajectory_time(row['trajectory_time'])
        rcs = np.array([p[3] for p in pts])
        alts = np.array([p[2] for p in pts])
        lons = np.array([p[0] for p in pts])
        lats = np.array([p[1] for p in pts])
        n = len(pts)
        f = {}
        if n > 4:
            dists = np.array([haversine(lons[j],lats[j],lons[j+1],lats[j+1]) for j in range(n-1)])
            dt = np.maximum(np.diff(times), 0.001)
            speeds = dists / dt
            rcs_lin = 10 ** (rcs / 10.0)

            f['energy_proxy'] = np.mean(speeds ** 3)
            f['alt_gain_total'] = alts[-1] - alts[0]
            f['corner_ref'] = np.mean(np.abs(np.diff(rcs)) > 2.0)

            rcs_c = rcs - np.mean(rcs); rv = np.var(rcs)
            f['rcs_ac1'] = np.sum(rcs_c[:-1]*rcs_c[1:])/(rv*(n-1)) if rv > 0 else 0

            size_map = {'Small bird':1,'Medium bird':2,'Large bird':3,'Flock':4}
            ds = size_map.get(row.get('radar_bird_size', 'Medium bird'), 2)
            peaks = [j for j in range(1,n-1) if rcs[j]>rcs[j-1] and rcs[j]>rcs[j+1]]
            troughs = [j for j in range(1,n-1) if rcs[j]<rcs[j-1] and rcs[j]<rcs[j+1]]
            md = [abs(rcs[p]-rcs[troughs[np.argmin(np.abs(np.array(troughs)-p))]]) for p in peaks] if peaks and troughs else [0]
            implied = 1 + min(3, np.mean(md)/5.0)
            f['size_inconsistency'] = abs(implied - ds)

            # Interaction features
            slow_frac = np.mean(speeds < np.median(speeds) * 0.7) if len(speeds) > 0 else 0.5
            safe_sf = max(slow_frac, 0.01)
            rcs_dff = tf.iloc[len(extras)]['rcs_deep_fade_frac'] if 'rcs_deep_fade_frac' in tf.columns and len(extras) < len(tf) else 0
            f['rcs_fade_div_slow'] = rcs_dff / safe_sf
            f['energy_div_slow'] = f['energy_proxy'] / safe_sf
        else:
            f = {k: 0 for k in ['energy_proxy','alt_gain_total','corner_ref','rcs_ac1',
                                 'size_inconsistency','rcs_fade_div_slow','energy_div_slow']}
        extras.append(f)
    return pd.DataFrame(extras)

extra_train = compute_extra(train)
extra_test = compute_extra(test)

X_train = np.nan_to_num(
    np.column_stack([tf.values.astype(np.float32), extra_train.values.astype(np.float32)]),
    nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(
    np.column_stack([xf.values.astype(np.float32), extra_test.values.astype(np.float32)]),
    nan=0, posinf=0, neginf=0)
X_train = np.clip(X_train, -1e6, 1e6)
X_test = np.clip(X_test, -1e6, 1e6)

print(f"  Features: {X_train.shape[1]} ({tf.shape[1]} base + {extra_train.shape[1]} new)", flush=True)

# Noise relabeling
cache = np.load(ROOT / "data/_cleanlab_cache.npz", allow_pickle=True)
y_r = y.copy()
for idx in cache['agreed_noisy'].tolist(): y_r[idx] = cache['consensus_labels'][idx]
quality = cache['quality']
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])
print(f"  Relabeled: {np.sum(y_r != y)}, Removed extreme: {len(extreme)}", flush=True)

# Cormorant augmentation (3x copies with noise)
CORM = CLASSES.index("Cormorants")
corm_idx = np.where(y_r == CORM)[0]
rng = np.random.RandomState(42)
aug_X, aug_y, aug_g, aug_m, aug_q = [], [], [], [], []
max_grp = groups.max() + 1
for idx in corm_idx:
    if not keep_mask[idx]: continue
    for rep in range(3):
        noise = rng.normal(0, 0.03, X_train.shape[1]) * np.abs(X_train[idx])
        aug_X.append(X_train[idx] + noise)
        aug_y.append(y_r[idx])
        aug_g.append(max_grp + idx * 3 + rep)
        aug_m.append(months[idx])
        aug_q.append(quality[idx])

X_aug = np.vstack([X_train, np.array(aug_X)])
y_aug = np.concatenate([y_r, np.array(aug_y)])
g_aug = np.concatenate([groups, np.array(aug_g)])
m_aug = np.concatenate([months, np.array(aug_m)])
q_aug = np.concatenate([quality, np.array(aug_q)])
km_aug = np.concatenate([keep_mask, np.ones(len(aug_y), dtype=bool)])
print(f"  Augmented: {len(X_train)} -> {len(X_aug)} (+{len(aug_X)} Corm copies)", flush=True)

# Quality weights
weights_aug = np.clip(q_aug, 0.3, 1.0)

# ══════════════════════════════════════════════════════════════
# OvR LambdaRank (with augmented data + quality weights)
# ══════════════════════════════════════════════════════════════
print(f"\n[2] OvR LambdaRank ({SEEDS_OVR} seeds)...", flush=True)
t0 = time.time()

oof_ranker = np.zeros((len(y), N))
test_ranker = np.zeros((len(X_test), N))

for seed in range(SEEDS_OVR):
    ts = time.time()
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42 + seed)
    oof_seed = np.zeros((len(y), N))
    test_seed = np.zeros((len(X_test), N))

    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        # Training: original fold (with keep_mask) + augmented Cormorants
        tr_clean = np.array([i for i in tr if keep_mask[i]])
        aug_idx = np.arange(len(X_train), len(X_aug))
        tr_combined = np.concatenate([tr_clean, aug_idx[km_aug[aug_idx]]])

        X_t = X_aug[tr_combined]; y_t = y_aug[tr_combined]
        m_t = m_aug[tr_combined]; w_t = weights_aug[tr_combined]
        X_v = X_train[va]; m_v = months[va]

        tr_o = np.argsort(m_t); va_o = np.argsort(m_v)
        tg = [int((m_t[tr_o] == m).sum()) for m in sorted(set(m_t[tr_o]))]
        vg = [int((m_v[va_o] == m).sum()) for m in sorted(set(m_v[va_o]))]

        for c in range(N):
            ybt = (y_t[tr_o] == c).astype(int)
            ybv = (y[va][va_o] == c).astype(int)
            if ybt.sum() < 2 or ybv.sum() < 1: continue

            mc = 5 if c in [CORM, CLASSES.index("Ducks"), CLASSES.index("Geese"),
                            CLASSES.index("Waders")] else 15

            ranker = lgb.LGBMRanker(
                objective="lambdarank", metric="map", boosting_type="dart",
                n_estimators=800, learning_rate=0.03, num_leaves=31,
                min_child_samples=mc, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, lambdarank_truncation_level=30,
                verbosity=-1, random_state=42 + seed + c, n_jobs=1)
            ranker.fit(X_t[tr_o], ybt, group=tg, sample_weight=w_t[tr_o],
                       eval_set=[(X_v[va_o], ybv)], eval_group=[vg],
                       callbacks=[lgb.early_stopping(80, verbose=False)])

            vp = ranker.predict(X_v[va_o])
            iv = np.empty_like(va_o); iv[va_o] = np.arange(len(va_o))
            oof_seed[va, c] = vp[iv]
            test_seed[:, c] += ranker.predict(X_test) / FOLDS

    oof_ranker += oof_seed / SEEDS_OVR
    test_ranker += test_seed / SEEDS_OVR
    s, _ = compute_map(y, oof_seed)
    print(f"    Seed {seed+1}/{SEEDS_OVR}: mAP={s:.4f} ({time.time()-ts:.0f}s)", flush=True)

sr, pr = compute_map(y, oof_ranker)
print(f"  OvR Ranker: SKF={sr:.4f} Corm={pr['Cormorants']:.4f} Wader={pr['Waders']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ══════════════════════════════════════════════════════════════
# CatBoost DRO (worst-month on VALIDATION, with augmentation)
# ══════════════════════════════════════════════════════════════
print(f"\n[3] CatBoost DRO ({SEEDS_CB} seeds)...", flush=True)
t0 = time.time()

oof_cat = np.zeros((len(y), N))
test_cat = np.zeros((len(X_test), N))

for seed in range(SEEDS_CB):
    ts = time.time()
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42 + seed)
    oof_seed = np.zeros((len(y), N))
    test_seed = np.zeros((len(X_test), N))

    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        tr_clean = np.array([i for i in tr if keep_mask[i]])
        aug_idx = np.arange(len(X_train), len(X_aug))
        tr_combined = np.concatenate([tr_clean, aug_idx])

        X_t = X_aug[tr_combined]; y_t = y_aug[tr_combined]
        m_t = m_aug[tr_combined]; w_t = weights_aug[tr_combined]
        X_v = X_train[va]

        # Round 1: find worst month (on VALIDATION, not train — bug fix)
        r1 = cb.CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=300,
            rsm=0.6, bootstrap_type="MVS", subsample=0.7,
            early_stopping_rounds=30, random_seed=42+seed+fold, verbose=0, task_type="CPU")
        r1.fit(cb.Pool(X_t, y_t, weight=w_t), eval_set=cb.Pool(X_v, y[va]))

        preds_va = r1.predict_proba(X_v)
        month_maps = {}
        for m in sorted(set(months[va])):
            vm = months[va] == m
            if vm.sum() >= 5:
                mm, _ = compute_map(y[va][vm], preds_va[vm])
                month_maps[m] = mm

        sw = w_t.copy()
        if month_maps:
            worst = min(month_maps, key=month_maps.get)
            sw[m_t == worst] *= 2.0

        # Round 2: train with DRO weights
        r2 = cb.CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=1500,
            rsm=0.6, bootstrap_type="MVS", subsample=0.7,
            model_shrink_rate=0.1, early_stopping_rounds=80,
            random_seed=42+seed+fold, verbose=0, task_type="CPU")
        r2.fit(cb.Pool(X_t, y_t, weight=sw), eval_set=cb.Pool(X_v, y[va]))

        oof_seed[va] = r2.predict_proba(X_v)
        test_seed += r2.predict_proba(X_test) / FOLDS

    oof_cat += oof_seed / SEEDS_CB
    test_cat += test_seed / SEEDS_CB
    s, _ = compute_map(y, oof_seed)
    print(f"    Seed {seed+1}/{SEEDS_CB}: mAP={s:.4f} ({time.time()-ts:.0f}s)", flush=True)

sc, pc = compute_map(y, oof_cat)
print(f"  CatBoost DRO: SKF={sc:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ══════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════
print(f"\n[4] Ensemble tuning...", flush=True)

best_s = -1; best_cfg = None
for pw in [1.0, 1.5, 2.0, 3.0]:
    for wr in np.arange(0, 1.05, 0.1):
        wc = 1 - wr
        bl = np.zeros((len(y), N))
        for c in range(N):
            bl[:, c] = wr * (rankdata(oof_ranker[:, c])/len(y))**pw + wc * (rankdata(oof_cat[:, c])/len(y))**pw
        s, pc = compute_map(y, bl)
        if s > best_s: best_s = s; best_cfg = (wr, wc, pw, pc)

wr, wc, pw, pc_best = best_cfg
oof_blend = np.zeros((len(y), N))
test_blend = np.zeros((len(X_test), N))
for c in range(N):
    oof_blend[:, c] = wr * (rankdata(oof_ranker[:, c])/len(y))**pw + wc * (rankdata(oof_cat[:, c])/len(y))**pw
    test_blend[:, c] = wr * (rankdata(test_ranker[:, c])/len(X_test))**pw + wc * (rankdata(test_cat[:, c])/len(X_test))**pw

sb, _ = compute_map(y, oof_blend)
ms = {}
for m in sorted(set(months)):
    mk = months == m
    if mk.sum() >= 10: ms[m], _ = compute_map(y[mk], oof_blend[mk])
lo = np.mean(list(ms.values()))

print(f"  Blend: w_ranker={wr:.1f} w_cb={wc:.1f} power={pw}", flush=True)
print(f"  SKF={sb:.4f}  LOMO={lo:.4f}", flush=True)
month_str = " ".join(f"m{m}={v:.3f}" for m, v in sorted(ms.items()))
print(f"  Months: {month_str}", flush=True)
for cls in CLASSES:
    mk = " ***" if pc_best[cls] < 0.4 else ""
    print(f"    {cls:15s}: {pc_best[cls]:.4f}{mk}", flush=True)

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
print(f"\n[5] Saving...", flush=True)
save_submission(test_blend, "e187_full_blend", cv_map=sb)
save_submission(test_ranker, "e187_full_ranker", cv_map=sr)
save_submission(test_cat, "e187_full_cb", cv_map=sc)

np.save(ROOT / "oof_e187_ranker.npy", oof_ranker)
np.save(ROOT / "oof_e187_cb.npy", oof_cat)
np.save(ROOT / "oof_e187_blend.npy", oof_blend)
np.save(ROOT / "test_e187_blend.npy", test_blend)

# Compare to E175
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
se, pe = compute_map(y, oof_e175)

print(f"\n{'='*80}", flush=True)
print(f"  COMPARISON", flush=True)
print(f"{'='*80}", flush=True)
print(f"  E175 original (LB=0.59): SKF={se:.4f}", flush=True)
print(f"  E187 improved:           SKF={sb:.4f} ({sb-se:+.4f})", flush=True)
print(f"\n  {'Class':15s} {'E175':>8s} {'E187':>8s} {'Delta':>8s}", flush=True)
for cls in CLASSES:
    d = pc_best[cls] - pe[cls]
    mk = " ***" if abs(d) > 0.02 else ""
    print(f"  {cls:15s} {pe[cls]:8.4f} {pc_best[cls]:8.4f} {d:+8.4f}{mk}", flush=True)

print(f"\nDone.", flush=True)
