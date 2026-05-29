"""E179: Full E175 Pipeline with Augmented Features — 20+10 seeds.

The winning combination:
- 100 stability-selected features + 4 new (glide_ratio, glide_frac, mass_proxy, hour_of_day)
- LGB DART multiclass (20 seeds)
- CatBoost DRO (10 seeds) with worst-month upweighting on VAL (fixed)
- Rank-power ensemble (tuned weights + power)
- CReST@0.2 blend (10 Cormorant pseudo-labels from shared months)
- Save submissions
"""

from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train, parse_ewkb_4d
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
MONTHS = [1, 4, 9, 10]

print("=" * 90)
print("  E179: FULL RUN — Aug Features, 20+10 Seeds")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)
t0 = time.time()

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
n_train = len(y)

# Base features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_base_tr = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_base_te = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# New features
print("  Extracting new features...", flush=True)
def extract_new(df):
    n = len(df)
    gr = np.zeros(n); gf = np.zeros(n); mp = np.zeros(n)
    hod = (pd.to_datetime(df["timestamp_start_radar_utc"]).dt.hour +
           pd.to_datetime(df["timestamp_start_radar_utc"]).dt.minute / 60.0).values
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(r["trajectory"])
            if len(pts) >= 3:
                alts = np.array([p[2] for p in pts])
                da = np.diff(alts); desc = da < -0.5
                if desc.sum() >= 2:
                    lons = np.array([p[0] for p in pts]); lats = np.array([p[1] for p in pts])
                    dx = np.diff(lons)*67000; dy = np.diff(lats)*111000
                    hd = np.sqrt(dx**2 + dy**2); dv = np.abs(da[desc]).sum()
                    if dv > 1: gr[i] = hd[desc].sum() / dv
                    gf[i] = desc.sum() / len(da)
            if len(pts) >= 2:
                rcs = np.mean([p[3] for p in pts])
                mp[i] = max(10**(rcs/10.0)**1.5, 1e-6)
        except: pass
    return np.column_stack([gr, gf, mp, hod]).astype(np.float32)

new_tr = np.nan_to_num(extract_new(train_df), nan=0.0, posinf=0.0, neginf=0.0)
new_te = np.nan_to_num(extract_new(test_df), nan=0.0, posinf=0.0, neginf=0.0)
X_train = np.hstack([X_base_tr, new_tr])
X_test = np.hstack([X_base_te, new_te])
n_test = X_test.shape[0]
print(f"  Features: {X_train.shape[1]} ({X_base_tr.shape[1]} base + {new_tr.shape[1]} new)")


def true_lomo(oof, name=""):
    skf, _ = compute_map(y, oof)
    scores = {}
    for m in MONTHS:
        mask = train_months == m
        s, _ = compute_map(y[mask], oof[mask])
        scores[m] = s
    lomo = np.mean(list(scores.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()))
    print(f"  {name:<50s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return skf, lomo


# ══════════════════════════════════════════════════════════════════════
# Phase 1: LGB DART — 20 seeds
# ══════════════════════════════════════════════════════════════════════

N_LGB_SEEDS = 20
print(f"\n  Phase 1: LGB DART ({N_LGB_SEEDS} seeds)", flush=True)

oof_lgb_all = np.zeros((N_LGB_SEEDS, n_train, N_CLASSES))
test_lgb_all = np.zeros((N_LGB_SEEDS, n_test, N_CLASSES))

for seed in range(N_LGB_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
    oof_s = np.zeros((n_train, N_CLASSES))
    test_s = np.zeros((n_test, N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
            n_estimators=1500, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
            drop_rate=0.15, is_unbalance=True, verbosity=-1,
            random_state=42+seed+fold, n_jobs=-1,
        )
        m.fit(X_train[tr], y[tr], eval_set=[(X_train[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_s[va] = m.predict_proba(X_train[va])
        test_s += m.predict_proba(X_test) / N_FOLDS
    oof_lgb_all[seed] = oof_s
    test_lgb_all[seed] = test_s
    s, _ = compute_map(y, oof_s)
    if (seed + 1) % 5 == 0:
        running = np.mean(oof_lgb_all[:seed+1], axis=0)
        rs, _ = compute_map(y, running)
        print(f"    Seed {seed+1:2d}: single={s:.4f}, running_avg={rs:.4f}", flush=True)
    else:
        print(f"    Seed {seed+1:2d}: {s:.4f}", flush=True)

oof_lgb = np.mean(oof_lgb_all, axis=0)
test_lgb = np.mean(test_lgb_all, axis=0)
true_lomo(oof_lgb, f"LGB DART ({N_LGB_SEEDS} seeds)")

np.save(ROOT / "oof_e179_lgb.npy", oof_lgb)
np.save(ROOT / "test_e179_lgb.npy", test_lgb)


# ══════════════════════════════════════════════════════════════════════
# Phase 2: CatBoost DRO — 10 seeds
# ══════════════════════════════════════════════════════════════════════

N_CB_SEEDS = 10
print(f"\n  Phase 2: CatBoost DRO ({N_CB_SEEDS} seeds)", flush=True)

oof_cb_all = np.zeros((N_CB_SEEDS, n_train, N_CLASSES))
test_cb_all = np.zeros((N_CB_SEEDS, n_test, N_CLASSES))

for seed in range(N_CB_SEEDS):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
    oof_s = np.zeros((n_train, N_CLASSES))
    test_s = np.zeros((n_test, N_CLASSES))
    for fold, (tr, va) in enumerate(sgkf.split(X_train, y, groups)):
        # Round 1: find worst month on VAL
        m1 = cb.CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            depth=6, l2_leaf_reg=5.0, learning_rate=0.05, iterations=300,
            rsm=0.6, early_stopping_rounds=50,
            random_seed=42+seed+fold, verbose=0, task_type="CPU",
        )
        m1.fit(X_train[tr], y[tr], eval_set=cb.Pool(X_train[va], y[va]))
        preds_va = m1.predict_proba(X_train[va])
        m_va = train_months[va]
        month_maps = {}
        for mo in sorted(set(m_va)):
            mask = m_va == mo
            if mask.sum() >= 5:
                mm, _ = compute_map(y[va][mask], preds_va[mask])
                month_maps[mo] = mm

        sw = np.ones(len(tr))
        if month_maps:
            worst = min(month_maps, key=month_maps.get)
            m_tr_fold = train_months[tr]
            sw[m_tr_fold == worst] *= 2.0

        # Round 2: train with DRO weights
        m2 = cb.CatBoostClassifier(
            loss_function="MultiClass", auto_class_weights="Balanced",
            depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
            rsm=0.6, early_stopping_rounds=100,
            random_seed=42+seed+fold, verbose=0, task_type="CPU",
        )
        m2.fit(X_train[tr], y[tr], sample_weight=sw, eval_set=cb.Pool(X_train[va], y[va]))
        oof_s[va] = m2.predict_proba(X_train[va])
        test_s += m2.predict_proba(X_test) / N_FOLDS

    oof_cb_all[seed] = oof_s
    test_cb_all[seed] = test_s
    s, _ = compute_map(y, oof_s)
    if (seed + 1) % 5 == 0:
        running = np.mean(oof_cb_all[:seed+1], axis=0)
        rs, _ = compute_map(y, running)
        print(f"    CB Seed {seed+1:2d}: single={s:.4f}, running_avg={rs:.4f}", flush=True)
    else:
        print(f"    CB Seed {seed+1:2d}: {s:.4f}", flush=True)

oof_cb = np.mean(oof_cb_all, axis=0)
test_cb = np.mean(test_cb_all, axis=0)
true_lomo(oof_cb, f"CatBoost DRO ({N_CB_SEEDS} seeds)")

np.save(ROOT / "oof_e179_cb.npy", oof_cb)
np.save(ROOT / "test_e179_cb.npy", test_cb)


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Rank-Power Ensemble
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Phase 3: Tuning Rank-Power Blend", flush=True)

best_score, best_cfg = -1, (0.5, 0.5, 1.5)
for power in [1.0, 1.5, 2.0]:
    for w_l in np.arange(0, 1.05, 0.05):
        w_c = 1.0 - w_l
        blend = np.zeros((n_train, N_CLASSES))
        for c in range(N_CLASSES):
            r1 = rankdata(oof_lgb[:, c]) / n_train
            r2 = rankdata(oof_cb[:, c]) / n_train
            blend[:, c] = w_l * (r1 ** power) + w_c * (r2 ** power)
        s, _ = compute_map(y, blend)
        if s > best_score:
            best_score = s
            best_cfg = (round(w_l, 2), round(w_c, 2), power)

w_l, w_c, power = best_cfg
print(f"  Best: w_lgb={w_l}, w_cb={w_c}, power={power}, SKF={best_score:.4f}", flush=True)

oof_blend = np.zeros((n_train, N_CLASSES))
test_blend = np.zeros((n_test, N_CLASSES))
for c in range(N_CLASSES):
    oof_blend[:, c] = w_l * (rankdata(oof_lgb[:, c]) / n_train) ** power + \
                      w_c * (rankdata(oof_cb[:, c]) / n_train) ** power
    test_blend[:, c] = w_l * (rankdata(test_lgb[:, c]) / n_test) ** power + \
                       w_c * (rankdata(test_cb[:, c]) / n_test) ** power

true_lomo(oof_lgb, "LGB DART alone")
true_lomo(oof_cb, "CatBoost DRO alone")
true_lomo(oof_blend, "E179 Rank-Power Blend")

np.save(ROOT / "oof_e179_best.npy", oof_blend)
np.save(ROOT / "test_e179_best.npy", test_blend)


# ══════════════════════════════════════════════════════════════════════
# Phase 4: CReST@0.2 Blend
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Phase 4: CReST@0.2 Blend", flush=True)

oof_crest_path = ROOT / "oof_e178_d2_crest.npy"
test_crest_path = ROOT / "test_e178_d2_crest.npy"
if oof_crest_path.exists() and test_crest_path.exists():
    oof_crest = np.load(oof_crest_path)
    test_crest = np.load(test_crest_path)
    for alpha in [0.1, 0.2, 0.3]:
        blend_c = renorm_rows((1-alpha) * oof_blend + alpha * renorm_rows(oof_crest))
        true_lomo(blend_c, f"E179 + CReST@{alpha}")
else:
    print("  CReST OOF not found, skipping blend")


# ══════════════════════════════════════════════════════════════════════
# Compare with E175
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Comparison with E175:", flush=True)
oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
true_lomo(oof_e175, "E175 (original)")
true_lomo(oof_blend, "E179 (augmented, 20+10 seeds)")

# Also blend E179 with E175
for alpha in [0.3, 0.5, 0.7]:
    blend_e = renorm_rows(alpha * oof_blend + (1-alpha) * oof_e175)
    true_lomo(blend_e, f"{alpha:.0%} E179 + {1-alpha:.0%} E175")


# ══════════════════════════════════════════════════════════════════════
# Save Submissions
# ══════════════════════════════════════════════════════════════════════

print(f"\n  Saving submissions...", flush=True)

save_submission(renorm_rows(test_blend), "e179_raw", cv_map=best_score)
save_submission(renorm_rows(test_lgb), "e179_lgb_raw", cv_map=compute_map(y, oof_lgb)[0])

# E179 + E175 blend
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
for alpha in [0.3, 0.5, 0.7]:
    blend_t = renorm_rows(alpha * test_blend + (1-alpha) * test_e175)
    save_submission(blend_t, f"e179_e175_blend_{int(alpha*100)}", cv_map=best_score)

# With CReST
if test_crest_path.exists():
    test_crest = np.load(test_crest_path)
    blend_final = renorm_rows(0.8 * test_blend + 0.2 * renorm_rows(test_crest))
    save_submission(blend_final, "e179_crest_blend", cv_map=best_score)

elapsed = time.time() - t0
print(f"\n{'='*90}")
print(f"  E179 COMPLETE in {elapsed/3600:.1f} hours")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*90}")
