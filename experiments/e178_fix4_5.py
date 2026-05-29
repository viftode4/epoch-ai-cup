"""E178 Fix 4+5 only — Full E175 pipeline with aug features + per-class weights TRUE CV.
Fixes 1-3 already completed. Skip to avoid 30min redo.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
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
print("  E178 Fix 4+5: Full Pipeline + Per-Class Weights")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)
t0 = time.time()

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]
X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

oof_best = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))


def true_lomo(oof, name=""):
    skf, _ = compute_map(y, oof)
    scores = {}
    for m in MONTHS:
        mask = train_months == m
        s, _ = compute_map(y[mask], oof[mask])
        scores[m] = s
    lomo = np.mean(list(scores.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()))
    print(f"  {name:<55s} SKF={skf:.4f} LOMO={lomo:.4f} [{month_str}]", flush=True)
    return skf, lomo

print("\n--- Baseline ---")
true_lomo(oof_best, "E175 best")


# ══════════════════════════════════════════════════════════════════════
# FIX 4: Full E175 pipeline (LGB DART + CatBoost DRO + rank-power)
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(X_tr, X_te, tag, n_lgb=5, n_cb=3):
    import lightgbm as lgb
    import catboost as cb
    n_train, n_test = X_tr.shape[0], X_te.shape[0]

    # LGB DART
    print(f"  [{tag}] LGB DART ({n_lgb} seeds)...", flush=True)
    oof_lgb_all = np.zeros((n_lgb, n_train, N_CLASSES))
    test_lgb_all = np.zeros((n_lgb, n_test, N_CLASSES))
    for seed in range(n_lgb):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((n_train, N_CLASSES))
        test_s = np.zeros((n_test, N_CLASSES))
        for fold, (tr, va) in enumerate(sgkf.split(X_tr, y, groups)):
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
                n_estimators=1500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
                drop_rate=0.15, is_unbalance=True, verbosity=-1,
                random_state=42+seed+fold, n_jobs=-1,
            )
            m.fit(X_tr[tr], y[tr], eval_set=[(X_tr[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_s[va] = m.predict_proba(X_tr[va])
            test_s += m.predict_proba(X_te) / N_FOLDS
        oof_lgb_all[seed] = oof_s
        test_lgb_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"    LGB seed {seed+1}: {s:.4f}", flush=True)
    oof_lgb_m = np.mean(oof_lgb_all, axis=0)
    test_lgb_m = np.mean(test_lgb_all, axis=0)

    # CatBoost DRO
    print(f"  [{tag}] CatBoost DRO ({n_cb} seeds)...", flush=True)
    oof_cb_all = np.zeros((n_cb, n_train, N_CLASSES))
    test_cb_all = np.zeros((n_cb, n_test, N_CLASSES))
    for seed in range(n_cb):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed)
        oof_s = np.zeros((n_train, N_CLASSES))
        test_s = np.zeros((n_test, N_CLASSES))
        for fold, (tr, va) in enumerate(sgkf.split(X_tr, y, groups)):
            # Round 1: find worst month on VAL (not train)
            m1 = cb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.05, iterations=300,
                rsm=0.6, early_stopping_rounds=50,
                random_seed=42+seed+fold, verbose=0, task_type="CPU",
            )
            m1.fit(X_tr[tr], y[tr], eval_set=cb.Pool(X_tr[va], y[va]))
            preds_va = m1.predict_proba(X_tr[va])
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

            # Round 2
            m2 = cb.CatBoostClassifier(
                loss_function="MultiClass", auto_class_weights="Balanced",
                depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
                rsm=0.6, early_stopping_rounds=100,
                random_seed=42+seed+fold, verbose=0, task_type="CPU",
            )
            m2.fit(X_tr[tr], y[tr], sample_weight=sw, eval_set=cb.Pool(X_tr[va], y[va]))
            oof_s[va] = m2.predict_proba(X_tr[va])
            test_s += m2.predict_proba(X_te) / N_FOLDS
        oof_cb_all[seed] = oof_s
        test_cb_all[seed] = test_s
        s, _ = compute_map(y, oof_s)
        print(f"    CB seed {seed+1}: {s:.4f}", flush=True)
    oof_cb_m = np.mean(oof_cb_all, axis=0)
    test_cb_m = np.mean(test_cb_all, axis=0)

    # Rank-power blend
    print(f"  [{tag}] Tuning blend...", flush=True)
    best_score, best_cfg = -1, (0.5, 0.5, 1.5)
    for power in [1.0, 1.5, 2.0]:
        for w_l in np.arange(0, 1.05, 0.1):
            w_c = 1.0 - w_l
            blend = np.zeros_like(oof_lgb_m)
            for c in range(N_CLASSES):
                r1 = rankdata(oof_lgb_m[:, c]) / n_train
                r2 = rankdata(oof_cb_m[:, c]) / n_train
                blend[:, c] = w_l * (r1 ** power) + w_c * (r2 ** power)
            s, _ = compute_map(y, blend)
            if s > best_score:
                best_score = s
                best_cfg = (round(w_l, 1), round(w_c, 1), power)

    w_l, w_c, power = best_cfg
    print(f"    Best: w_lgb={w_l}, w_cb={w_c}, power={power}, SKF={best_score:.4f}", flush=True)

    oof_blend = np.zeros_like(oof_lgb_m)
    test_blend = np.zeros_like(test_lgb_m)
    for c in range(N_CLASSES):
        oof_blend[:, c] = w_l * (rankdata(oof_lgb_m[:, c]) / n_train) ** power + w_c * (rankdata(oof_cb_m[:, c]) / n_train) ** power
        test_blend[:, c] = w_l * (rankdata(test_lgb_m[:, c]) / n_test) ** power + w_c * (rankdata(test_cb_m[:, c]) / n_test) ** power

    true_lomo(oof_lgb_m, f"[{tag}] LGB DART alone")
    true_lomo(oof_cb_m, f"[{tag}] CatBoost DRO alone")
    true_lomo(oof_blend, f"[{tag}] Rank-power blend")
    return oof_blend, test_blend

# Base 100 features
print("\n--- Base 100 features (replicate E175) ---")
oof_base, test_base = run_pipeline(X_train, X_test, "base100")

# Augmented features
print("\n--- Augmented features (100 + new) ---")
def quick_feats(df):
    n = len(df)
    gr = np.zeros(n); gf = np.zeros(n); mp = np.zeros(n)
    hod = (pd.to_datetime(df["timestamp_start_radar_utc"]).dt.hour + pd.to_datetime(df["timestamp_start_radar_utc"]).dt.minute/60.0).values
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(r["trajectory"])
            if len(pts) >= 3:
                alts = np.array([p[2] for p in pts])
                da = np.diff(alts); desc = da < -0.5
                if desc.sum() >= 2:
                    lons = np.array([p[0] for p in pts]); lats = np.array([p[1] for p in pts])
                    dx = np.diff(lons)*67000; dy = np.diff(lats)*111000
                    hd = np.sqrt(dx**2+dy**2); dv = np.abs(da[desc]).sum()
                    if dv > 1: gr[i] = hd[desc].sum()/dv
                    gf[i] = desc.sum()/len(da)
            if len(pts) >= 2:
                rcs = np.mean([p[3] for p in pts])
                mp[i] = max(10**(rcs/10.0)**1.5, 1e-6)
        except: pass
    return np.column_stack([gr, gf, mp, hod]).astype(np.float32)

print("  Extracting features...", flush=True)
new_tr = np.nan_to_num(quick_feats(train_df), nan=0.0, posinf=0.0, neginf=0.0)
new_te = np.nan_to_num(quick_feats(test_df), nan=0.0, posinf=0.0, neginf=0.0)
X_aug_tr = np.hstack([X_train, new_tr])
X_aug_te = np.hstack([X_test, new_te])
print(f"  {X_aug_tr.shape[1]} features ({X_train.shape[1]} + {new_tr.shape[1]})")

oof_aug, test_aug = run_pipeline(X_aug_tr, X_aug_te, "aug104")


# ══════════════════════════════════════════════════════════════════════
# FIX 5: Per-class weights TRUE LOMO-CV
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("  FIX 5: Per-class weights (TRUE LOMO-CV)")
print(f"{'='*90}", flush=True)

from sklearn.metrics import average_precision_score

def pcw_true_lomo(oof_list, names):
    oof_out = oof_list[0].copy()
    for held in MONTHS:
        mask_h = train_months == held
        mask_t = ~mask_h
        for c in range(N_CLASSES):
            y_bin = (y[mask_t] == c).astype(float)
            if y_bin.sum() < 2: continue
            best_ap, best_w = -1, 0.5
            for w0 in np.arange(0, 1.05, 0.05):
                bl = w0 * oof_list[0][mask_t, c] + (1-w0) * oof_list[1][mask_t, c]
                ap = average_precision_score(y_bin, bl)
                if ap > best_ap:
                    best_ap = ap; best_w = w0
            oof_out[mask_h, c] = best_w * oof_list[0][mask_h, c] + (1-best_w) * oof_list[1][mask_h, c]
    return renorm_rows(oof_out)

oof_pcw = pcw_true_lomo([oof_best, oof_lgb], ["best", "lgb"])
true_lomo(oof_pcw, "Per-class weights best+lgb (TRUE LOMO-CV)")

# Also try with the new pipeline outputs
if 'oof_base' in dir():
    # per-class between new base and new aug
    oof_pcw2 = pcw_true_lomo([oof_base, oof_aug], ["base100", "aug104"])
    true_lomo(oof_pcw2, "Per-class weights base+aug (TRUE LOMO-CV)")

print(f"\nCompleted in {(time.time()-t0)/60:.1f} min")
print("=" * 90)
