"""E175 diversity ensemble: 3 feature-subset models + PCA compression."""

import sys
import warnings
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.data import load_train, load_test, CLASSES
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load features
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_v3 = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
shared = sorted(set(train_v3.columns) & set(test_v3.columns))
const = {c for c in shared if train_v3[c].std() < 1e-10 or test_v3[c].std() < 1e-10}
all_cols = sorted(set(shared) - const)

# Categorize features
v2_ext = [c for c in all_cols if not (c.startswith("lsig_") or "_c22_" in c or c.startswith("phys_")
          or c in ["alt_curvature", "alt_r2", "speed_cv", "speed_ac1", "speed_trend",
                   "predicted_flock_size"] or c.startswith("rcs_ac_lag"))]
catch22 = [c for c in all_cols if "_c22_" in c]
logsig = [c for c in all_cols if c.startswith("lsig_")]
physics = [c for c in all_cols if c.startswith("phys_")]
new_traj = [c for c in all_cols if c in ["alt_curvature", "alt_r2", "speed_cv", "speed_ac1",
            "speed_trend", "predicted_flock_size"] or c.startswith("rcs_ac_lag")]

print(f"Feature groups: v2_ext={len(v2_ext)}, catch22={len(catch22)}, logsig={len(logsig)}, "
      f"physics={len(physics)}, new_traj={len(new_traj)}")


def eff_weights(y_arr, beta=0.999):
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-6)
    w = w / w.sum() * N_CLASSES
    return w[y_arr]


def weighted_lomo(oof):
    WEIGHTS = {1: 0.165, 4: 0.162, 9: 0.244, 10: 0.429}
    total_w, weighted_sum = 0.0, 0.0
    per_month = {}
    for held in sorted(set(months)):
        mask = months == held
        if mask.sum() < 5:
            continue
        lm, _ = compute_map(y[mask], oof[mask])
        per_month[held] = lm
        w = WEIGHTS.get(held, 0.1)
        weighted_sum += w * lm
        total_w += w
    return weighted_sum / total_w if total_w > 0 else 0.0, per_month


def eval_model(oof, label):
    skf, pc = compute_map(y, oof)
    wlomo, lomo_d = weighted_lomo(oof)
    ulomo = np.mean(list(lomo_d.values()))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_d.items()))
    print(f"  [{label}] SKF={skf:.4f} wLOMO={wlomo:.4f} uLOMO={ulomo:.4f} ({month_str})")
    return skf, wlomo, pc


# ═══ Build PCA features ═══
print("\n--- PCA COMPRESSION ---")

# PCA on catch22 (88 -> N components)
X_c22_tr = train_v3[catch22].values.astype(np.float64)
X_c22_te = test_v3[catch22].values.astype(np.float64)
sc_c22 = StandardScaler()
X_c22_tr_s = sc_c22.fit_transform(X_c22_tr)
X_c22_te_s = sc_c22.transform(X_c22_te)

for n_comp in [5, 10, 15, 20]:
    pca = PCA(n_components=n_comp)
    pca.fit(X_c22_tr_s)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  catch22 PCA {n_comp} components: {100*var_explained:.1f}% variance")

# PCA on logsig (90 -> N components)
X_ls_tr = train_v3[logsig].values.astype(np.float64)
X_ls_te = test_v3[logsig].values.astype(np.float64)
sc_ls = StandardScaler()
X_ls_tr_s = sc_ls.fit_transform(X_ls_tr)
X_ls_te_s = sc_ls.transform(X_ls_te)

for n_comp in [5, 10, 15, 20]:
    pca = PCA(n_components=n_comp)
    pca.fit(X_ls_tr_s)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  logsig PCA {n_comp} components: {100*var_explained:.1f}% variance")

# Use 15 components each (capture ~80%+ variance)
N_PCA = 15
pca_c22 = PCA(n_components=N_PCA)
c22_tr_pca = pca_c22.fit_transform(X_c22_tr_s)
c22_te_pca = pca_c22.transform(X_c22_te_s)
print(f"\n  Using: catch22 PCA {N_PCA} ({100*pca_c22.explained_variance_ratio_.sum():.1f}%)")

pca_ls = PCA(n_components=N_PCA)
ls_tr_pca = pca_ls.fit_transform(X_ls_tr_s)
ls_te_pca = pca_ls.transform(X_ls_te_s)
print(f"  Using: logsig PCA {N_PCA} ({100*pca_ls.explained_variance_ratio_.sum():.1f}%)")


# ═══ Build 3 feature sets ═══
def make_X(feat_cols, c22_pca=None, ls_pca=None, split="train"):
    src = train_v3 if split == "train" else test_v3
    X = src[feat_cols].values.astype(np.float32)
    parts = [X]
    if c22_pca is not None:
        parts.append(c22_pca.astype(np.float32))
    if ls_pca is not None:
        parts.append(ls_pca.astype(np.float32))
    return np.nan_to_num(np.hstack(parts))


# Model A: v2+ext only (121f)
Xa_tr = make_X(v2_ext, split="train")
Xa_te = make_X(v2_ext, split="test")

# Model B: v2+ext + PCA(catch22) + PCA(logsig) + physics + new_traj
Xb_tr = make_X(v2_ext + physics + new_traj, c22_tr_pca, ls_tr_pca, "train")
Xb_te = make_X(v2_ext + physics + new_traj, c22_te_pca, ls_te_pca, "test")

# Model C: All 316f (DART)
Xc_tr = np.nan_to_num(train_v3[all_cols].values.astype(np.float32))
Xc_te = np.nan_to_num(test_v3[all_cols].values.astype(np.float32))

print(f"\n  Model A: {Xa_tr.shape[1]}f (v2+ext)")
print(f"  Model B: {Xb_tr.shape[1]}f (v2+ext+PCA+physics+new)")
print(f"  Model C: {Xc_tr.shape[1]}f (all 316)")


# ═══ Train all 3 ═══
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def train_lgb_dart(X_train, X_test, label):
    oof = np.zeros((len(y), N_CLASSES))
    test_p = np.zeros((len(test_df), N_CLASSES))
    t = time.time()
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
              eval_set=[(X_train[vidx], y[vidx])], eval_sample_weight=[w_va])
        oof[vidx] = m.predict_proba(X_train[vidx])
        test_p += m.predict_proba(X_test) / 5
    print(f"  [{label}] trained in {time.time()-t:.0f}s")
    return oof, test_p


print("\n--- TRAINING 3 MODELS ---")
oof_a, test_a = train_lgb_dart(Xa_tr, Xa_te, "A: v2+ext")
eval_model(oof_a, "A: v2+ext")

oof_b, test_b = train_lgb_dart(Xb_tr, Xb_te, "B: v2+ext+PCA")
eval_model(oof_b, "B: v2+ext+PCA")

oof_c, test_c = train_lgb_dart(Xc_tr, Xc_te, "C: all 316")
eval_model(oof_c, "C: all 316")


# ═══ Ensemble sweep ═══
print("\n--- ENSEMBLE SWEEP ---")
best_wlomo = 0
best_config = None

for wa in np.arange(0, 1.05, 0.1):
    for wb in np.arange(0, 1.05 - wa, 0.1):
        wc = round(1.0 - wa - wb, 2)
        if wc < -0.01:
            continue
        wc = max(wc, 0)
        oof_e = wa * oof_a + wb * oof_b + wc * oof_c
        wlomo, _ = weighted_lomo(oof_e)
        if wlomo > best_wlomo:
            skf, _ = compute_map(y, oof_e)
            best_wlomo = wlomo
            best_skf = skf
            best_config = (round(wa, 2), round(wb, 2), round(wc, 2))

wa, wb, wc = best_config
print(f"  Best: A={wa} B={wb} C={wc} -> SKF={best_skf:.4f} wLOMO={best_wlomo:.4f}")

oof_best = wa * oof_a + wb * oof_b + wc * oof_c
test_best = wa * test_a + wb * test_b + wc * test_c
eval_model(oof_best, "BEST DIVERSITY ENSEMBLE")

# Also try with CB DRO from Phase 3 if available
try:
    oof_dro = np.load(ROOT / "oof_e175_dro.npy")
    test_dro = np.load(ROOT / "test_e175_dro.npy")
    print("\n--- + CB DRO (4-model sweep) ---")
    best4_wlomo = 0
    best4_config = None
    for wa in np.arange(0, 1.05, 0.1):
        for wb in np.arange(0, 1.05 - wa, 0.1):
            for wc in np.arange(0, 1.05 - wa - wb, 0.1):
                wd = round(1.0 - wa - wb - wc, 2)
                if wd < -0.01:
                    continue
                wd = max(wd, 0)
                oof_e = wa * oof_a + wb * oof_b + wc * oof_c + wd * oof_dro
                wlomo, _ = weighted_lomo(oof_e)
                if wlomo > best4_wlomo:
                    skf, _ = compute_map(y, oof_e)
                    best4_wlomo = wlomo
                    best4_skf = skf
                    best4_config = (round(wa, 2), round(wb, 2), round(wc, 2), round(wd, 2))

    wa4, wb4, wc4, wd4 = best4_config
    print(f"  Best: A={wa4} B={wb4} C={wc4} DRO={wd4} -> SKF={best4_skf:.4f} wLOMO={best4_wlomo:.4f}")
    oof_best4 = wa4 * oof_a + wb4 * oof_b + wc4 * oof_c + wd4 * oof_dro
    test_best4 = wa4 * test_a + wb4 * test_b + wc4 * test_c + wd4 * test_dro
    eval_model(oof_best4, "BEST 4-MODEL ENSEMBLE")
except FileNotFoundError:
    print("  (CB DRO not available)")
    oof_best4 = oof_best
    test_best4 = test_best
    best4_skf = best_skf
    best4_wlomo = best_wlomo

# ═══ Save submissions ═══
print("\n--- SUBMISSIONS ---")
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES as NC,
    build_gbif_priors, apply_gated_ratio_priors, top2_margin,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

def apply_pp(preds, gamma=0.10, tau_nb=0.25):
    counts = np.bincount(y, minlength=NC).astype(float)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)
    out, _ = apply_gated_ratio_priors(preds, test_months, p_train, priors, BASE_ALPHA, tau=0.15)
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

save_submission(test_a, "e175div_v2ext_raw", cv_map=round(compute_map(y, oof_a)[0], 4))
save_submission(test_best, "e175div_3model_raw", cv_map=round(best_skf, 4))
save_submission(apply_pp(test_best), "e175div_3model_pp", cv_map=round(best_skf, 4))
save_submission(test_best4, "e175div_4model_raw", cv_map=round(best4_skf, 4))
save_submission(apply_pp(test_best4), "e175div_4model_pp", cv_map=round(best4_skf, 4))

# Save OOFs
np.save(ROOT / "oof_e175_best.npy", oof_best4)
np.save(ROOT / "test_e175_best.npy", test_best4)

print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")
