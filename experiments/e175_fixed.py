"""E175 Fixed — All audit issues resolved, clean re-run.

Fixes applied:
  C-1: No early_stopping for DART (incompatible, was silently ignored)
  C-2: _safe_load row-count validation (in e174_all_data.py)
  C-3: Flock predictor now active (added to cached features)
  W-5: LOMO ensemble sweep uses test-month-weighted average
"""

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
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent

# Test month weights from src/validate.py (what LB actually measures)
TEST_MONTH_WEIGHTS = {9: 0.244, 10: 0.429, 2: 0.094, 5: 0.162, 12: 0.071}
# Train month -> test month proxy mapping
FOLD_TO_PROXY_WEIGHT = {1: 0.165, 4: 0.162, 9: 0.244, 10: 0.429}  # Jan->Feb+Dec, Apr->May

# ── Load data ──
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load features (now includes predicted_flock_size)
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_v3 = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
shared = sorted(set(train_v3.columns) & set(test_v3.columns))
const = {c for c in shared if train_v3[c].std() < 1e-10 or test_v3[c].std() < 1e-10}
feature_cols = sorted(set(shared) - const)

X_train = np.nan_to_num(train_v3[feature_cols].values.astype(np.float32))
X_test = np.nan_to_num(test_v3[feature_cols].values.astype(np.float32))


def eff_weights(y_arr, beta=0.999):
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-6)
    w = w / w.sum() * N_CLASSES
    return w[y_arr]


def weighted_lomo(oof):
    """Compute test-distribution-weighted LOMO (W-5 fix)."""
    total_w = 0.0
    weighted_sum = 0.0
    per_month = {}
    for held in sorted(set(months)):
        mask = months == held
        if mask.sum() < 5:
            continue
        lm, _ = compute_map(y[mask], oof[mask])
        per_month[held] = lm
        w = FOLD_TO_PROXY_WEIGHT.get(held, 0.1)
        weighted_sum += w * lm
        total_w += w
    wlomo = weighted_sum / total_w if total_w > 0 else 0.0
    return wlomo, per_month


def eval_oof(oof, label):
    skf, pc = compute_map(y, oof)
    wlomo, lomo_d = weighted_lomo(oof)
    ulomo = np.mean(list(lomo_d.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_d.items()))
    print(f"  [{label}] SKF={skf:.4f}  wLOMO={wlomo:.4f}  uLOMO={ulomo:.4f}  ({month_str})")
    return skf, wlomo, ulomo, pc


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print(f"  E175 FIXED — {len(feature_cols)} features (incl. flock predictor)")
print(f"  Fixes: DART no early_stop, row validation, flock active, weighted LOMO")
print("=" * 70)

# ═══ A: LGB DART (FIX C-1: no early_stopping) ═══
print("\n--- A: LGB DART (no early_stopping) ---")
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
    # FIX C-1: No early_stopping callback for DART
    m.fit(X_train[tidx], y[tidx], sample_weight=w_tr,
          eval_set=[(X_train[vidx], y[vidx])], eval_sample_weight=[w_va])
    oof_lgb[vidx] = m.predict_proba(X_train[vidx])
    test_lgb += m.predict_proba(X_test) / 5
    fm, _ = compute_map(y[vidx], oof_lgb[vidx])
    print(f"    Fold {fold+1}: {fm:.4f}", flush=True)
print(f"  [{time.time()-t:.0f}s]")
eval_oof(oof_lgb, "LGB DART")

# ═══ B: CatBoost + Group DRO ═══
print("\n--- B: CatBoost + Group DRO ---")
t = time.time()
oof_dro = np.zeros((len(y), N_CLASSES))
test_dro = np.zeros((len(test_df), N_CLASSES))
for fold, (tidx, vidx) in enumerate(sgkf.split(X_train, y, groups)):
    m_tr = months[tidx]
    # Round 1: find worst month
    m1 = cb.CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        depth=6, l2_leaf_reg=5.0, learning_rate=0.05, iterations=300,
        rsm=0.6, early_stopping_rounds=30,
        random_seed=42 + fold, verbose=0, task_type="CPU",
    )
    m1.fit(X_train[tidx], y[tidx], eval_set=cb.Pool(X_train[vidx], y[vidx]))
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

    # Round 2: train with DRO weights
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
    fm, _ = compute_map(y[vidx], oof_dro[vidx])
    print(f"    Fold {fold+1}: {fm:.4f}", flush=True)
print(f"  [{time.time()-t:.0f}s]")
eval_oof(oof_dro, "CB DRO")

# ═══ C: XGBoost ═══
print("\n--- C: XGBoost ---")
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
eval_oof(oof_xgb, "XGBoost")

# ═══ ENSEMBLE SWEEP (FIX W-5: test-weighted LOMO) ═══
print("\n--- ENSEMBLE SWEEP (test-weighted LOMO) ---")
best_wlomo = 0
best_config = None

for wl in np.arange(0, 1.05, 0.1):
    for wd in np.arange(0, 1.05 - wl, 0.1):
        for wx in np.arange(0, 1.05 - wl - wd, 0.1):
            rem = round(1.0 - wl - wd - wx, 2)
            if rem < -0.01:
                continue
            # rem is unused weight (effectively dropped)
            oof_e = wl * oof_lgb + wd * oof_dro + wx * oof_xgb
            total_w = wl + wd + wx
            if total_w < 0.01:
                continue
            oof_e = oof_e / total_w  # renormalize
            wlomo, _ = weighted_lomo(oof_e)
            if wlomo > best_wlomo:
                skf, _ = compute_map(y, oof_e)
                best_wlomo = wlomo
                best_skf = skf
                best_config = (round(wl/total_w, 2), round(wd/total_w, 2), round(wx/total_w, 2))

wl, wd, wx = best_config
print(f"  Best: LGB={wl} DRO={wd} XGB={wx} -> SKF={best_skf:.4f} wLOMO={best_wlomo:.4f}")

oof_best = wl * oof_lgb + wd * oof_dro + wx * oof_xgb
test_best = wl * test_lgb + wd * test_dro + wx * test_xgb
eval_oof(oof_best, "BEST ENSEMBLE")

# ═══ PP on test ═══
print("\n--- POST-PROCESSING ---")

def apply_pp(preds, gamma=0.10, tau_prior=0.15, tau_nb=0.25):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=tau_prior)
    sp = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
    mz = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
    xz = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
    ct = {"speed": sp, "alt_mid": 0.5*(mz+xz), "alt_range": xz-mz}
    sl, lps, mu, sig = build_nb_params(train_df, y, ct)
    sp_t = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
    mz_t = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
    xz_t = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
    ct_t = {"speed": sp_t, "alt_mid": 0.5*(mz_t+xz_t), "alt_range": xz_t-mz_t}
    w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
    ll = compute_log_p_u_given_c(test_df, sl, lps, ct_t, w, None, mu, sig)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
    return apply_nb_poe(out, ll, gamma=gamma, gate=gate)

test_best_pp = apply_pp(test_best)
test_lgb_pp = apply_pp(test_lgb)

# ═══ SAVE SUBMISSIONS ═══
print("\n--- SAVING SUBMISSIONS ---")
save_submission(test_lgb, "e175fix_lgb_raw", cv_map=round(compute_map(y, oof_lgb)[0], 4))
save_submission(test_lgb_pp, "e175fix_lgb_pp", cv_map=round(compute_map(y, oof_lgb)[0], 4))
save_submission(test_best, "e175fix_blend_raw", cv_map=round(best_skf, 4))
save_submission(test_best_pp, "e175fix_blend_pp", cv_map=round(best_skf, 4))
save_submission(test_dro, "e175fix_dro_raw", cv_map=round(compute_map(y, oof_dro)[0], 4))

# Also save OOF/test npy
np.save(ROOT / "oof_e175_lgb.npy", oof_lgb)
np.save(ROOT / "oof_e175_dro.npy", oof_dro)
np.save(ROOT / "oof_e175_xgb.npy", oof_xgb)
np.save(ROOT / "oof_e175_best.npy", oof_best)
np.save(ROOT / "test_e175_lgb.npy", test_lgb)
np.save(ROOT / "test_e175_dro.npy", test_dro)
np.save(ROOT / "test_e175_xgb.npy", test_xgb)
np.save(ROOT / "test_e175_best.npy", test_best)

# ═══ SUMMARY ═══
print(f"\n{'='*70}")
print(f"  E175 FIXED — FINAL RESULTS")
print(f"{'='*70}")
print(f"  Features: {len(feature_cols)} (incl. predicted_flock_size)")
print(f"  Fixes: DART no early_stop, row validation, flock active, weighted LOMO")
print(f"")
print(f"  {'Model':25s} {'SKF':>7s} {'wLOMO':>7s}")

for label, oof in [("LGB DART", oof_lgb), ("CB DRO", oof_dro), ("XGBoost", oof_xgb), ("Best Ensemble", oof_best)]:
    s, _ = compute_map(y, oof)
    wl_val, _ = weighted_lomo(oof)
    print(f"  {label:25s} {s:7.4f} {wl_val:7.4f}")

print(f"")
print(f"  Ensemble weights: LGB={best_config[0]} DRO={best_config[1]} XGB={best_config[2]}")
print(f"")
print(f"  Reference: E170 SKF=0.7013 wLOMO~0.51, E79 LB=0.59")
print(f"  Best to submit: e175fix_blend_raw or e175fix_blend_pp")
print(f"{'='*70}")
