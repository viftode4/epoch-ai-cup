"""Compare old vs new architecture, isolating each change."""

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
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load v3 features (316 usable)
train_v3 = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_v3 = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
shared = sorted(set(train_v3.columns) & set(test_v3.columns))
const_cols = {c for c in shared if train_v3[c].std() < 1e-10 or test_v3[c].std() < 1e-10}
all_316 = sorted(set(shared) - const_cols)

# E79's 36 features
e79_feats = [l.strip() for l in (ROOT / "data" / "best_features.txt").read_text().splitlines() if l.strip()]
e79_available = [f for f in e79_feats if f in all_316]

# New-only features (log-sig, catch22, physics, new traj, flock)
new_only = [c for c in all_316 if c.startswith("lsig_") or "_c22_" in c or c.startswith("phys_")
            or c in ["alt_curvature", "alt_r2", "speed_cv", "speed_ac1", "speed_trend", "predicted_flock_size"]
            or c.startswith("rcs_ac_lag")]
e79_plus_new = sorted(set(e79_available + new_only))

print(f"Feature sets: E79={len(e79_available)}, E79+new={len(e79_plus_new)}, ALL={len(all_316)}")


def eff_weights(y_arr, beta=0.999):
    counts = np.bincount(y_arr, minlength=N_CLASSES).astype(float)
    eff = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-6)
    w = w / w.sum() * N_CLASSES
    return w[y_arr]


def run_config(feature_cols, boosting, use_ensemble, label):
    X = np.nan_to_num(train_v3[feature_cols].values.astype(np.float32))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros((len(y), N_CLASSES))
    oof_xgb = np.zeros((len(y), N_CLASSES))
    oof_cb = np.zeros((len(y), N_CLASSES))

    is_dart = boosting == "dart"

    for fold, (tidx, vidx) in enumerate(splitter.split(X, y, groups)):
        w_tr = eff_weights(y[tidx])
        w_va = eff_weights(y[vidx])

        # LGB
        lgb_p = dict(
            objective="multiclass", num_class=N_CLASSES,
            boosting_type=boosting,
            n_estimators=2000,
            learning_rate=0.03 if is_dart else 0.05,
            num_leaves=31 if is_dart else 63,
            min_child_samples=20 if is_dart else 10,
            colsample_bytree=0.6 if is_dart else 0.8,
            subsample=0.7 if is_dart else 0.8,
            is_unbalance=False, verbosity=-1, random_state=42, n_jobs=-1,
        )
        if is_dart:
            lgb_p["drop_rate"] = 0.15

        m = lgb.LGBMClassifier(**lgb_p)
        m.fit(X[tidx], y[tidx], sample_weight=w_tr,
              eval_set=[(X[vidx], y[vidx])], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[vidx] = m.predict_proba(X[vidx])

        if use_ensemble:
            mx = xgb.XGBClassifier(
                objective="multi:softprob", num_class=N_CLASSES, n_estimators=2000,
                learning_rate=0.05, max_depth=7, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                verbosity=0, random_state=42, n_jobs=-1, tree_method="hist",
                early_stopping_rounds=100,
            )
            mx.fit(X[tidx], y[tidx], sample_weight=w_tr,
                   eval_set=[(X[vidx], y[vidx])], sample_weight_eval_set=[w_va], verbose=False)
            oof_xgb[vidx] = mx.predict_proba(X[vidx])

            mc = cb.CatBoostClassifier(
                iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                random_seed=42, verbose=0, auto_class_weights=None,
                early_stopping_rounds=100, task_type="CPU",
            )
            mc.fit(X[tidx], y[tidx], sample_weight=w_tr,
                   eval_set=cb.Pool(X[vidx], y[vidx], weight=w_va))
            oof_cb[vidx] = mc.predict_proba(X[vidx])

    if use_ensemble:
        oof = 0.15 * oof_lgb + 0.05 * oof_xgb + 0.80 * oof_cb
    else:
        oof = oof_lgb

    skf, pc = compute_map(y, oof)
    lomo = {}
    for held in sorted(set(months)):
        mask = months == held
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof[mask])
            lomo[held] = lm
    return skf, np.mean(list(lomo.values())), lomo, pc


# ═══════════════════════════════════════════════════════════════
configs = [
    # (features, boosting, ensemble, label)
    # Baseline: reproduce E79/E173 architecture
    (e79_available, "gbdt", True,  "A. E79 repro (36f gbdt LGB+XGB+CB)"),
    # Change 1: DART only
    (e79_available, "dart", False, "B. E79 + DART (36f LGB-only)"),
    # Change 2: DART + ensemble
    (e79_available, "dart", True,  "C. E79 + DART + ensemble (36f)"),
    # Change 3: new features only (no ext dilution)
    (e79_plus_new,  "dart", False, "D. E79+sigs+c22+phys DART LGB"),
    # Change 4: all features
    (all_316,       "dart", False, "E. ALL 316f DART LGB-only"),
    # Change 5: all features + ensemble
    (all_316,       "dart", True,  "F. ALL 316f DART LGB+XGB+CB"),
]

print("=" * 95)
print("  COMPREHENSIVE ARCHITECTURE COMPARISON")
print("  Research refs: FINAL_ARCHITECTURE.md Decisions 1,3,4,6,7")
print("=" * 95)

results = {}
for feats, boost, ens, label in configs:
    t = time.time()
    ens_str = "LGB+XGB+CB" if ens else "LGB only"
    print(f"\n  {label} ({len(feats)}f, {ens_str})...", flush=True)
    skf, lomo, lomo_d, pc = run_config(feats, boost, ens, label)
    elapsed = time.time() - t
    results[label] = (skf, lomo, lomo_d, pc, len(feats))
    month_str = " ".join(f"{m}={v:.3f}" for m, v in sorted(lomo_d.items()))
    print(f"    SKF={skf:.4f}  LOMO={lomo:.4f}  ({month_str})  [{elapsed:.0f}s]")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*95}")
print(f"  RESULTS TABLE")
print(f"{'='*95}")
print(f"  {'Config':45s} {'N':>4s} {'SKF':>7s} {'LOMO':>7s} {'Jan':>6s} {'Apr':>6s} {'Sep':>6s} {'Oct':>6s}")
for label, (skf, lomo, lomo_d, _, nf) in results.items():
    vals = [lomo_d.get(m, 0) for m in [1, 4, 9, 10]]
    print(f"  {label:45s} {nf:4d} {skf:7.4f} {lomo:7.4f} {vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:6.3f} {vals[3]:6.3f}")

labels = list(results.keys())
base_skf, base_lomo = results[labels[0]][:2]
print(f"\n  Deltas vs A (E79 baseline):")
for label, (skf, lomo, _, _, _) in results.items():
    print(f"    {label:45s}  SKF {skf-base_skf:+7.4f}  LOMO {lomo-base_lomo:+7.4f}")

# Per-class: A vs F
print(f"\n  Per-class: A (E79 repro) vs F (ALL+DART+ensemble):")
pc_a = results[labels[0]][3]
pc_f = results[labels[-1]][3]
print(f"  {'Class':15s} {'A':>8s} {'F':>8s} {'Delta':>8s}")
for cls in CLASSES:
    a = pc_a.get(cls, 0)
    f = pc_f.get(cls, 0)
    d = f - a
    marker = " ***" if abs(d) > 0.02 else ""
    print(f"  {cls:15s} {a:8.4f} {f:8.4f} {d:+8.4f}{marker}")

# Attribution
print(f"\n  CHANGE ATTRIBUTION (cumulative):")
print(f"  {'Change':45s} {'LOMO delta':>12s} {'Source':>30s}")
lomo_a = results[labels[0]][1]
lomo_b = results[labels[1]][1]
lomo_c = results[labels[2]][1]
lomo_d_val = results[labels[3]][1]
lomo_e = results[labels[4]][1]
lomo_f = results[labels[-1]][1]
print(f"  {'DART boosting (gbdt->dart)':45s} {lomo_b-lomo_a:+12.4f} {'Decision 6, Agent 6':>30s}")
print(f"  {'+ ensemble (LGB->LGB+XGB+CB)':45s} {lomo_c-lomo_b:+12.4f} {'Decision 7':>30s}")
print(f"  {'+ new feats (sigs+c22+phys)':45s} {lomo_d_val-lomo_c:+12.4f} {'Decisions 1-2':>30s}")
print(f"  {'+ all external features':45s} {lomo_e-lomo_d_val:+12.4f} {'Decision 2':>30s}")
print(f"  {'+ ensemble on all features':45s} {lomo_f-lomo_e:+12.4f} {'Decision 7':>30s}")
print(f"{'='*95}")
