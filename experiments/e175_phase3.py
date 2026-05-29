"""E175 Phase 3: Deep dive — 4 models + ensemble on ALL 316 features."""

import sys
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# ALL 316 features
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_v3 = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
shared = sorted(set(train_v3.columns) & set(test_v3.columns))
const = {c for c in shared if train_v3[c].std() < 1e-10 or test_v3[c].std() < 1e-10}
feature_cols = sorted(set(shared) - const)

X_train = np.nan_to_num(train_v3[feature_cols].values.astype(np.float32))
X_test = np.nan_to_num(test_v3[feature_cols].values.astype(np.float32))

print(f"Features: {len(feature_cols)}, Train: {X_train.shape}, Test: {X_test.shape}")


def eff_weights(y_arr, beta=0.999):
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-6)
    w = w / w.sum() * N_CLASSES
    return w[y_arr]


def eval_oof(oof, label):
    skf, pc = compute_map(y, oof)
    lomo = {}
    for held in sorted(set(months)):
        mask = months == held
        lm, _ = compute_map(y[mask], oof[mask])
        lomo[held] = lm
    lomo_avg = np.mean(list(lomo.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo.items()))
    print(f"  [{label}] SKF={skf:.4f}  LOMO={lomo_avg:.4f}  ({month_str})")
    for cls in CLASSES:
        if pc[cls] < 0.4:
            print(f"    WEAK: {cls}={pc[cls]:.4f}")
    return skf, lomo_avg, pc


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("  PHASE 3 DEEP DIVE - ALL 316 features")
print("  Models: LGB DART, CB Balanced, CB+DRO, XGB")
print("=" * 70)

# === A: LGB DART ===
print("\n--- A: LGB DART ---")
t = time.time()
oof_lgb = np.zeros((len(y), N_CLASSES))
test_lgb = np.zeros((len(test_df), N_CLASSES))
for fold, (tidx, vidx) in enumerate(sgkf.split(X_train, y, groups)):
    w_tr = eff_weights(y[tidx])
    w_va = eff_weights(y[vidx])
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES, boosting_type="dart",
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, colsample_bytree=0.6, subsample=0.7,
        drop_rate=0.15, is_unbalance=False,
        verbosity=-1, random_state=42, n_jobs=-1,
    )
    m.fit(X_train[tidx], y[tidx], sample_weight=w_tr,
          eval_set=[(X_train[vidx], y[vidx])], eval_sample_weight=[w_va],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[vidx] = m.predict_proba(X_train[vidx])
    test_lgb += m.predict_proba(X_test) / 5
print(f"  [{time.time()-t:.0f}s]")
skf_lgb, lomo_lgb, pc_lgb = eval_oof(oof_lgb, "LGB DART")

# === B: CatBoost Balanced ===
print("\n--- B: CatBoost Balanced ---")
t = time.time()
oof_cb = np.zeros((len(y), N_CLASSES))
test_cb = np.zeros((len(test_df), N_CLASSES))
for fold, (tidx, vidx) in enumerate(sgkf.split(X_train, y, groups)):
    mc = cb.CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
        rsm=0.6, model_shrink_rate=0.1,
        early_stopping_rounds=100, random_seed=42 + fold, verbose=0, task_type="CPU",
    )
    mc.fit(X_train[tidx], y[tidx],
           eval_set=cb.Pool(X_train[vidx], y[vidx]))
    oof_cb[vidx] = mc.predict_proba(X_train[vidx])
    test_cb += mc.predict_proba(X_test) / 5
print(f"  [{time.time()-t:.0f}s]")
skf_cb, lomo_cb, pc_cb = eval_oof(oof_cb, "CB Balanced")

# === C: CatBoost + Group DRO ===
print("\n--- C: CatBoost + Group DRO ---")
t = time.time()
oof_dro = np.zeros((len(y), N_CLASSES))
test_dro = np.zeros((len(test_df), N_CLASSES))
for fold, (tidx, vidx) in enumerate(sgkf.split(X_train, y, groups)):
    m_tr = months[tidx]
    # Quick round to find worst month
    m1 = cb.CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        depth=6, l2_leaf_reg=5.0, learning_rate=0.05, iterations=300,
        rsm=0.6, early_stopping_rounds=30,
        random_seed=42 + fold, verbose=0, task_type="CPU",
    )
    m1.fit(X_train[tidx], y[tidx],
           eval_set=cb.Pool(X_train[vidx], y[vidx]))
    preds_tr = m1.predict_proba(X_train[tidx])
    worst_m, worst_v = 9, 1.0
    for um in sorted(set(m_tr)):
        mask = m_tr == um
        if mask.sum() >= 5:
            mm, _ = compute_map(y[tidx][mask], preds_tr[mask])
            if mm < worst_v:
                worst_m, worst_v = um, mm
    sw = np.ones(len(tidx))
    sw[m_tr == worst_m] *= 2.0

    mc = cb.CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=2000,
        rsm=0.6, model_shrink_rate=0.1,
        early_stopping_rounds=100, random_seed=42 + fold, verbose=0, task_type="CPU",
    )
    mc.fit(X_train[tidx], y[tidx], sample_weight=sw,
           eval_set=cb.Pool(X_train[vidx], y[vidx]))
    oof_dro[vidx] = mc.predict_proba(X_train[vidx])
    test_dro += mc.predict_proba(X_test) / 5
print(f"  [{time.time()-t:.0f}s]")
skf_dro, lomo_dro, pc_dro = eval_oof(oof_dro, "CB + DRO")

# === D: XGBoost ===
print("\n--- D: XGBoost ---")
t = time.time()
oof_xgb = np.zeros((len(y), N_CLASSES))
test_xgb = np.zeros((len(test_df), N_CLASSES))
for fold, (tidx, vidx) in enumerate(sgkf.split(X_train, y, groups)):
    w_tr = eff_weights(y[tidx])
    w_va = eff_weights(y[vidx])
    mx = xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES, n_estimators=2000,
        learning_rate=0.05, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        verbosity=0, random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=100,
    )
    mx.fit(X_train[tidx], y[tidx], sample_weight=w_tr,
           eval_set=[(X_train[vidx], y[vidx])],
           sample_weight_eval_set=[w_va], verbose=False)
    oof_xgb[vidx] = mx.predict_proba(X_train[vidx])
    test_xgb += mx.predict_proba(X_test) / 5
print(f"  [{time.time()-t:.0f}s]")
skf_xgb, lomo_xgb, pc_xgb = eval_oof(oof_xgb, "XGBoost")

# === ENSEMBLE SWEEP (optimize LOMO) ===
print("\n--- ENSEMBLE SWEEP (optimize LOMO) ---")
best_skf = best_lomo = 0
best_w = best_label = None

for wl in np.arange(0, 1.05, 0.1):
    for wc in np.arange(0, 1.05 - wl, 0.1):
        for wx in np.arange(0, 1.05 - wl - wc, 0.1):
            wd = round(1.0 - wl - wc - wx, 2)
            if wd < -0.01:
                continue
            wd = max(wd, 0)
            oof_e = wl * oof_lgb + wc * oof_cb + wx * oof_xgb + wd * oof_dro
            lvals = []
            for held in sorted(set(months)):
                mask = months == held
                lm, _ = compute_map(y[mask], oof_e[mask])
                lvals.append(lm)
            la = np.mean(lvals)
            if la > best_lomo:
                s, _ = compute_map(y, oof_e)
                best_lomo = la
                best_skf = s
                best_w = (round(wl, 2), round(wc, 2), round(wx, 2), round(wd, 2))
                best_label = f"LGB={wl:.1f} CB={wc:.1f} XGB={wx:.1f} DRO={wd:.1f}"

print(f"  Best LOMO: {best_label} -> SKF={best_skf:.4f} LOMO={best_lomo:.4f}")

# Also find best SKF ensemble
best_skf2 = 0
best_w2 = best_label2 = None
for wl in np.arange(0, 1.05, 0.1):
    for wc in np.arange(0, 1.05 - wl, 0.1):
        for wx in np.arange(0, 1.05 - wl - wc, 0.1):
            wd = round(1.0 - wl - wc - wx, 2)
            if wd < -0.01:
                continue
            wd = max(wd, 0)
            oof_e = wl * oof_lgb + wc * oof_cb + wx * oof_xgb + wd * oof_dro
            s, _ = compute_map(y, oof_e)
            if s > best_skf2:
                best_skf2 = s
                lvals = []
                for held in sorted(set(months)):
                    mask = months == held
                    lm, _ = compute_map(y[mask], oof_e[mask])
                    lvals.append(lm)
                best_lomo2 = np.mean(lvals)
                best_w2 = (round(wl, 2), round(wc, 2), round(wx, 2), round(wd, 2))
                best_label2 = f"LGB={wl:.1f} CB={wc:.1f} XGB={wx:.1f} DRO={wd:.1f}"

print(f"  Best SKF:  {best_label2} -> SKF={best_skf2:.4f} LOMO={best_lomo2:.4f}")

# Build both ensembles
wl, wc, wx, wd = best_w
oof_best_lomo = wl * oof_lgb + wc * oof_cb + wx * oof_xgb + wd * oof_dro
test_best_lomo = wl * test_lgb + wc * test_cb + wx * test_xgb + wd * test_dro

wl2, wc2, wx2, wd2 = best_w2
oof_best_skf = wl2 * oof_lgb + wc2 * oof_cb + wx2 * oof_xgb + wd2 * oof_dro
test_best_skf = wl2 * test_lgb + wc2 * test_cb + wx2 * test_xgb + wd2 * test_dro

print("\n  Best LOMO ensemble per-class:")
eval_oof(oof_best_lomo, "BEST-LOMO")
print("\n  Best SKF ensemble per-class:")
eval_oof(oof_best_skf, "BEST-SKF")

# === SUMMARY ===
print(f"\n{'='*70}")
print(f"  PHASE 3 SUMMARY - ALL 316 features")
print(f"{'='*70}")
print(f"  {'Model':25s} {'SKF':>7s} {'LOMO':>7s}")
print(f"  {'LGB DART':25s} {skf_lgb:7.4f} {lomo_lgb:7.4f}")
print(f"  {'CB Balanced':25s} {skf_cb:7.4f} {lomo_cb:7.4f}")
print(f"  {'CB + Group DRO':25s} {skf_dro:7.4f} {lomo_dro:7.4f}")
print(f"  {'XGBoost':25s} {skf_xgb:7.4f} {lomo_xgb:7.4f}")
print(f"  {'Ensemble (best LOMO)':25s} {best_skf:7.4f} {best_lomo:7.4f}  {best_label}")
print(f"  {'Ensemble (best SKF)':25s} {best_skf2:7.4f} {best_lomo2:7.4f}  {best_label2}")

print(f"\n  Reference (prior best):")
print(f"  {'E170 (76f, LGB+XGB+CB)':25s} {'0.7013':>7s} {'0.5141':>7s}")
print(f"  {'E79  (36f, LGB+XGB+CB)':25s} {'0.7736':>7s} {'~0.36':>7s}")

# Save everything
np.save(ROOT / "oof_e175_lgb.npy", oof_lgb)
np.save(ROOT / "oof_e175_cb.npy", oof_cb)
np.save(ROOT / "oof_e175_dro.npy", oof_dro)
np.save(ROOT / "oof_e175_xgb.npy", oof_xgb)
np.save(ROOT / "oof_e175_best.npy", oof_best_lomo)
np.save(ROOT / "test_e175_lgb.npy", test_lgb)
np.save(ROOT / "test_e175_cb.npy", test_cb)
np.save(ROOT / "test_e175_dro.npy", test_dro)
np.save(ROOT / "test_e175_xgb.npy", test_xgb)
np.save(ROOT / "test_e175_best.npy", test_best_lomo)
np.save(ROOT / "test_e175_best_skf.npy", test_best_skf)
print(f"\n  All predictions saved.")
print(f"{'='*70}")
