"""E68: Enhanced Biological Time + Shape + RCS Texture Features

Builds on E42 (best LOMO: 0.3799) by adding features that were either
missing or only partially covered:

1. Biological Time (non-leaky solar-derived):
   - hours_from_solar_noon: |hours_since_sunrise - daylight_hours/2|
   - is_thermal_window: solar_elevation > 25 (BoP soaring thermals)
   - is_dawn_dusk: solar -6 to 15° (migration activity window)

2. Trajectory Shape (circling / soaring detection):
   - turn_dir_consistency: |net_turning| / total_turning → 1=circling, 0=erratic
   - max_sustained_turn_frac: longest same-direction run / track length
   - turn_reversal_rate: reversals/sec → high=Songbirds, low=BoP/Gulls
   - path_loop_fraction: 1 - net_disp/total_path → high=circling

3. RCS Texture (flapping vs. gliding quality):
   - rcs_dominant_ac_lag: normalized lag of peak autocorrelation (wingbeat proxy)
   - rcs_flap_regularity: CV of flap bout durations (low=metronomic Pigeons/Waders)
   - rcs_glide_flap_var_ratio: RCS var contrast between modes
   - rcs_burst_fraction: bounding flight detection (Songbirds)

Evaluation: LOMO (honest) + binary specialists on minority classes.
Baseline: E42 LOMO = 0.3799.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.features import build_features, ALL_TEMPORAL
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)

print("=" * 60, flush=True)
print("E68: ENHANCED BIO TIME + SHAPE + RCS TEXTURE", flush=True)
print("=" * 60, flush=True)

# ── Data + labels ──────────────────────────────────────────────────
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# ── Build features ─────────────────────────────────────────────────
print("\nBuilding features...", flush=True)
feat_sets = [
    "core", "rcs_fft", "tabular", "targeted",
    "flight_mode", "weakclass", "enhanced_bio_shape",
]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)

# Remove leaky temporal features
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep]
test_feats = test_feats[keep]

# ── Weather (E38 config) ────────────────────────────────────────────
train_weather = pd.read_csv(ROOT / "data" / "train_weather.csv")
test_weather = pd.read_csv(ROOT / "data" / "test_weather.csv")
for col in train_weather.columns:
    train_feats[f"wx_{col}"] = train_weather[col].values
    test_feats[f"wx_{col}"] = test_weather[col].values

# ── Solar (E38 config) ──────────────────────────────────────────────
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
for col in train_solar.columns:
    train_feats[f"sol_{col}"] = train_solar[col].values
    test_feats[f"sol_{col}"] = test_solar[col].values

# ── Derived solar biological-time features ─────────────────────────
for feats, solar in [(train_feats, train_solar), (test_feats, test_solar)]:
    # Hours from solar noon: 0 at noon, max ~= daylight_hours/2 at dawn/dusk
    feats["hours_from_solar_noon"] = np.abs(
        solar["hours_since_sunrise"].values - solar["daylight_hours"].values / 2
    )
    # BoP soar when sun is high (thermals form above ~25°)
    feats["is_thermal_window"] = (solar["solar_elevation"].values > 25).astype(float)
    # Dawn/dusk migration window (-6° to 15° elevation)
    elev = solar["solar_elevation"].values
    feats["is_dawn_dusk"] = ((elev > -6) & (elev < 15)).astype(float)
    # Afternoon thermal window (solar elevation > 15 and past noon)
    feats["is_afternoon_thermal"] = (
        (solar["solar_elevation"].values > 15) &
        (solar["hours_since_sunrise"].values > solar["daylight_hours"].values / 2)
    ).astype(float)

# ── GBIF seasonal priors ────────────────────────────────────────────
gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls == "Clutter":
            si[i] = 1.0
        else:
            class_counts = gbif[cls].values
            class_mean = class_counts.mean()
            si[i] = row[cls] / class_mean if class_mean > 0 else 1.0
    gbif_si[m] = si

for i, cls in enumerate(CLASSES):
    col = f"gbif_si_{cls.lower().replace(' ', '_')}"
    train_feats[col] = [gbif_si[m][i] for m in train_months]
    test_feats[col] = [gbif_si[m][i] for m in test_months]

gbif_priors_df = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
month_entropy = {}
for _, row in gbif_priors_df.iterrows():
    m = int(row["month"])
    probs = np.maximum(np.array([row[cls] for cls in CLASSES]), 1e-10)
    month_entropy[m] = -np.sum(probs * np.log(probs))
train_feats["month_gbif_diversity"] = [month_entropy[m] for m in train_months]
test_feats["month_gbif_diversity"] = [month_entropy[m] for m in test_months]

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"  Total features: {len(fn)}", flush=True)

# ── Class weights (effective number) ───────────────────────────────
BETA = 0.999
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

# ── LOMO multiclass baseline ────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("PART 1: LOMO MULTICLASS (CB)", flush=True)
print("=" * 60, flush=True)

oof_multi = np.zeros((len(y), N_CLASSES))
test_multi_preds = []

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, verbose=0, early_stopping_rounds=80, task_type="CPU"
    )
    cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), verbose=0,
           sample_weight=sample_weights[tr_idx])
    oof_multi[va_idx] = cb.predict_proba(X[va_idx])
    test_multi_preds.append(cb.predict_proba(X_test))

    m_val, _ = compute_map(y[va_idx], oof_multi[va_idx])
    print(f"  Month {m}: mAP={m_val:.4f} (n={len(va_idx)})", flush=True)

test_multi = np.mean(test_multi_preds, axis=0)
multi_map, multi_per = compute_map(y, oof_multi)
print(f"\n  Multiclass LOMO mAP: {multi_map:.4f}  (E42 base was 0.3612)", flush=True)
print(f"\n  Per-class LOMO APs:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {multi_per.get(cls, 0):.4f}", flush=True)

# ── LGB + XGB for ensemble ──────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("PART 2: FULL ENSEMBLE (LGB + XGB + CB) LOMO", flush=True)
print("=" * 60, flush=True)

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 47, "max_depth": 7, "min_child_samples": 8,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1,
}

import xgboost as xgb

XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "n_estimators": 1000, "seed": 42, "verbosity": 0, "n_jobs": -1,
}

oof_lgb = np.zeros((len(y), N_CLASSES))
oof_xgb = np.zeros((len(y), N_CLASSES))
test_lgb_preds, test_xgb_preds = [], []

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]

    # LGB
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx])
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain)
    lgb_mdl = lgb.train(LGB_PARAMS, dtrain, 1500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = lgb_mdl.predict(X[va_idx]).reshape(-1, N_CLASSES)
    test_lgb_preds.append(lgb_mdl.predict(X_test).reshape(-1, N_CLASSES))

    # XGB
    xgb_mdl = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=80)
    xgb_mdl.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                verbose=False, sample_weight=sample_weights[tr_idx])
    oof_xgb[va_idx] = xgb_mdl.predict_proba(X[va_idx])
    test_xgb_preds.append(xgb_mdl.predict_proba(X_test))

test_lgb = np.mean(test_lgb_preds, axis=0)
test_xgb = np.mean(test_xgb_preds, axis=0)

# Ensemble weights (E38/E42 config)
W_LGB, W_XGB, W_CB = 0.33, 0.33, 0.34
oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_multi
test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_multi
ens_map, ens_per = compute_map(y, oof_ens)
print(f"\n  Ensemble LOMO mAP: {ens_map:.4f}  (E42 had ~0.3612)", flush=True)
print(f"\n  Per-class:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {ens_per.get(cls, 0):.4f}", flush=True)

# ── Binary specialists (E42 pattern) ───────────────────────────────
print("\n" + "=" * 60, flush=True)
print("PART 3: BINARY SPECIALISTS (LOMO)", flush=True)
print("=" * 60, flush=True)

minority_classes = ["Cormorants", "Waders", "Pigeons", "Birds of Prey", "Ducks"]
specialist_oof = {}
specialist_test = {}

for cls in minority_classes:
    cls_idx = list(CLASSES).index(cls)
    y_bin = (y == cls_idx).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    print(f"\n--- {cls} (pos={n_pos}, neg={n_neg}) ---", flush=True)

    oof_cb_bin = np.zeros(len(y))
    test_cb_bin = np.zeros(len(X_test))
    oof_lgb_bin = np.zeros(len(y))
    test_lgb_bin = np.zeros(len(X_test))

    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]
        y_tr_bin = y_bin[tr_idx]
        y_va_bin = y_bin[va_idx]
        n_pos_tr = y_tr_bin.sum()

        if n_pos_tr < 3:
            oof_cb_bin[va_idx] = n_pos_tr / max(len(y_tr_bin), 1)
            oof_lgb_bin[va_idx] = n_pos_tr / max(len(y_tr_bin), 1)
            test_cb_bin += n_pos_tr / max(len(y_tr_bin), 1) / len(unique_months)
            test_lgb_bin += n_pos_tr / max(len(y_tr_bin), 1) / len(unique_months)
            continue

        # CB binary
        cb_bin = CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=4, l2_leaf_reg=5,
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=42, verbose=0,
            early_stopping_rounds=80, task_type="CPU",
        )
        cb_bin.fit(X[tr_idx], y_tr_bin, eval_set=(X[va_idx], y_va_bin), verbose=0)
        oof_cb_bin[va_idx] = cb_bin.predict_proba(X[va_idx])[:, 1]
        test_cb_bin += cb_bin.predict_proba(X_test)[:, 1] / len(unique_months)

        # LGB binary
        lgb_params_bin = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
            "min_child_samples": 5, "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "is_unbalance": True, "verbose": -1, "seed": 42,
        }
        dtrain_b = lgb.Dataset(X[tr_idx], label=y_tr_bin)
        dval_b = lgb.Dataset(X[va_idx], label=y_va_bin, reference=dtrain_b)
        lgb_mdl_b = lgb.train(lgb_params_bin, dtrain_b, 1500, valid_sets=[dval_b],
                               callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_lgb_bin[va_idx] = lgb_mdl_b.predict(X[va_idx])
        test_lgb_bin += lgb_mdl_b.predict(X_test) / len(unique_months)

    # Compare CB vs LGB vs multiclass
    ap_cb = average_precision_score(y_bin, oof_cb_bin)
    ap_lgb = average_precision_score(y_bin, oof_lgb_bin)
    ap_ens_bin = average_precision_score(y_bin, 0.5 * oof_cb_bin + 0.5 * oof_lgb_bin)
    ap_multi = average_precision_score(y_bin, oof_ens[:, cls_idx])
    print(f"  LOMO AP: Multi={ap_multi:.4f}  CB={ap_cb:.4f}  LGB={ap_lgb:.4f}  Ens={ap_ens_bin:.4f}", flush=True)

    # Pick best
    best_ap = max(ap_cb, ap_lgb, ap_ens_bin, ap_multi)
    if best_ap == ap_ens_bin:
        specialist_oof[cls] = 0.5 * oof_cb_bin + 0.5 * oof_lgb_bin
        specialist_test[cls] = 0.5 * test_cb_bin + 0.5 * test_lgb_bin
    elif best_ap == ap_cb:
        specialist_oof[cls] = oof_cb_bin
        specialist_test[cls] = test_cb_bin
    elif best_ap == ap_lgb:
        specialist_oof[cls] = oof_lgb_bin
        specialist_test[cls] = test_lgb_bin
    else:
        specialist_oof[cls] = None
        specialist_test[cls] = None

# ── Alpha sweep: blend specialists with ensemble ────────────────────
print("\n" + "=" * 60, flush=True)
print("PART 4: SPECIALIST BLEND SWEEP", flush=True)
print("=" * 60, flush=True)

best_alpha = 0.4
best_hybrid_map = -1.0
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    oof_hybrid = oof_ens.copy()
    for cls in minority_classes:
        if specialist_oof[cls] is not None:
            ci = list(CLASSES).index(cls)
            oof_hybrid[:, ci] = ((1 - alpha) * oof_ens[:, ci]
                                 + alpha * specialist_oof[cls])
    oof_hybrid = oof_hybrid / oof_hybrid.sum(axis=1, keepdims=True)
    hm, _ = compute_map(y, oof_hybrid)
    print(f"  alpha={alpha:.1f}: LOMO mAP={hm:.4f}", flush=True)
    if hm > best_hybrid_map:
        best_hybrid_map = hm
        best_alpha = alpha

print(f"\n  Best alpha={best_alpha:.1f}, LOMO mAP={best_hybrid_map:.4f}", flush=True)

# ── Final predictions ───────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("PART 5: FINAL OOF + TEST PREDICTIONS", flush=True)
print("=" * 60, flush=True)

oof_final = oof_ens.copy()
test_final = test_ens.copy()
for cls in minority_classes:
    if specialist_oof[cls] is not None:
        ci = list(CLASSES).index(cls)
        oof_final[:, ci] = ((1 - best_alpha) * oof_ens[:, ci]
                            + best_alpha * specialist_oof[cls])
        test_final[:, ci] = ((1 - best_alpha) * test_ens[:, ci]
                             + best_alpha * specialist_test[cls])

oof_final = oof_final / oof_final.sum(axis=1, keepdims=True)
test_final = test_final / test_final.sum(axis=1, keepdims=True)
final_map, final_per = compute_map(y, oof_final)

print(f"\n  Final LOMO mAP: {final_map:.4f}  (E42 best was 0.3799)", flush=True)
print(f"\n  Per-class LOMO (final vs E42):", flush=True)
e42_per = {
    "Gulls": 0.859, "Songbirds": 0.451, "Pigeons": 0.261, "Waders": 0.067,
    "Birds of Prey": 0.328, "Clutter": 0.567, "Geese": 0.434, "Ducks": 0.400,
    "Cormorants": 0.064,
}
for cls in CLASSES:
    ap_new = final_per.get(cls, 0)
    ap_old = e42_per.get(cls, 0)
    delta = ap_new - ap_old
    print(f"    {cls:<18s}: {ap_new:.4f} (E42={ap_old:.4f}, {delta:+.4f})", flush=True)

# ── Save ────────────────────────────────────────────────────────────
np.save(ROOT / "oof_e68.npy", oof_final)
np.save(ROOT / "test_e68.npy", test_final)
save_submission(test_final, "e68_enhanced_features", cv_map=final_map)

# Also apply the winning E54 unseen-month prior adjustment on top
print("\n  Applying E54 winter_tilt priors to E68 test predictions...", flush=True)
gbif_priors_full = pd.read_csv(ROOT / "data" / "gbif_monthly_priors.csv")
gbif_prior_map = {}
for _, row in gbif_priors_full.iterrows():
    m = int(row["month"])
    gbif_prior_map[m] = np.array([row[cls] for cls in CLASSES])

# E54 winter_tilt alphas: {m2: 0.22, m5: 0.12, m12: 0.24}
unseen_alphas = {2: 0.22, 5: 0.12, 12: 0.24}
test_final_adj = test_final.copy()
test_months_arr = test_ts.dt.month.values
for idx in range(len(test_final_adj)):
    m = test_months_arr[idx]
    if m in unseen_alphas:
        alpha_adj = unseen_alphas[m]
        prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
        prior = prior / prior.sum()
        probs = test_final_adj[idx]
        adjusted = (1 - alpha_adj) * probs + alpha_adj * prior
        test_final_adj[idx] = adjusted / adjusted.sum()

save_submission(test_final_adj, "e68_enhanced_winterpriors", cv_map=final_map)

print(f"\n  Saved: e68_enhanced_features (LOMO {final_map:.4f})", flush=True)
print(f"  Saved: e68_enhanced_winterpriors (with E54 unseen-month prior adjustment)", flush=True)
print("\nDone!", flush=True)
