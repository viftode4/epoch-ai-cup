"""E71: Fixed wing morphology + per-group GBIF alpha optimisation.

Two improvements over E70 (LOMO 0.3507):

1. Wing morphology fix: E70 fell back to 0.70m / 0.07m² for all classes
   because BirdWingData_tidy_ver2.1.csv was absent.  src/external_priors.py
   now uses hardcoded per-class literature values, recovering the differenti-
   ating wing_loading / aspect_ratio features that made E50 beat E42.

2. Per-group GBIF alpha:  instead of one alpha per month (E54 plateau), we
   tune two parameters per unseen month:
     - alpha_rare   (Waders, Ducks, Cormorants, Geese)
     - alpha_common (Gulls, Songbirds, Birds of Prey, Pigeons, Clutter)
   Month-analog proxy for unseen months:
     Feb → Jan LOMO OOF,  May → Apr LOMO OOF,  Dec → Oct LOMO OOF.
   Grid: alpha_rare ∈ {0.1, 0.2, 0.3, 0.4, 0.5},
         alpha_common ∈ {0.0, 0.05, 0.10, 0.15, 0.20}  → 25 × 3 = 75 evals.

Ensemble: CB=0.50, LGB=0.30, XGB=0.20 (E70 best weights; skip Optuna).
Feature pipeline: same E48-C as E70.

Submissions:
  e71_base           — no prior correction
  e71_winter_tilt    — E54 uniform alphas (direct LB comparison)
  e71_pergroup       — per-group optimised alphas
  e71_pergroup_stronger — slight upward variant (+0.05 on rare, +0.02 common)
"""

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.external_priors import build_external_class_priors
from src.features import ALL_TEMPORAL, add_external_prior_features, build_features
from src.metrics import compute_map
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
BETA = 0.999

# ── Class index helpers ────────────────────────────────────────────
CLASS_IDX = {cls: i for i, cls in enumerate(CLASSES)}

RARE_CLASSES = ["Waders", "Ducks", "Cormorants", "Geese"]
COMMON_CLASSES = ["Gulls", "Songbirds", "Birds of Prey", "Pigeons", "Clutter"]
RARE_IDX = [CLASS_IDX[c] for c in RARE_CLASSES]
COMMON_IDX = [CLASS_IDX[c] for c in COMMON_CLASSES]

# Unseen test months → best LOMO proxy month for alpha tuning
MONTH_ANALOG = {2: 1, 5: 4, 12: 10}

# ── Model hyperparameters (E70 best weights, E48 defaults for XGB/CB) ──
W_LGB, W_XGB, W_CB = 0.30, 0.20, 0.50

LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
    "verbose": -1, "seed": 42, "n_jobs": -1, "device": "cpu",
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES,
    "eval_metric": "mlogloss", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.3, "reg_lambda": 1.5,
    "seed": 42, "nthread": -1, "verbosity": 0,
    "device": "cpu", "tree_method": "hist",
}
CB_PARAMS = {
    "iterations": 1400, "learning_rate": 0.05,
    "depth": 6, "l2_leaf_reg": 3,
    "loss_function": "MultiClass", "eval_metric": "MultiClass",
    "random_seed": 42, "verbose": 0,
    "early_stopping_rounds": 80, "task_type": "CPU",
}

print("=" * 65, flush=True)
print("E71: FIXED WING MORPHOLOGY + PER-GROUP GBIF ALPHA", flush=True)
print("=" * 65, flush=True)

# ── Data + labels ──────────────────────────────────────────────────
train_df = load_train()
test_df = load_test()
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
counts = np.bincount(y, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w /= class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])
train_months = train_ts.dt.month.values
test_months = test_ts.dt.month.values
unique_months = sorted(np.unique(train_months))

# ── Build E48-C features ───────────────────────────────────────────
print("\nBuilding E38 base features...", flush=True)
feat_sets = ["core", "rcs_fft", "tabular", "targeted", "flight_mode", "weakclass"]
train_feats = build_features(train_df, feature_sets=feat_sets)
test_feats = build_features(test_df, feature_sets=feat_sets)
keep = [c for c in train_feats.columns if c not in ALL_TEMPORAL]
train_feats = train_feats[keep].copy()
test_feats = test_feats[keep].copy()

print("Adding weather + solar + GBIF...", flush=True)
for split, feats in [("train", train_feats), ("test", test_feats)]:
    months = train_months if split == "train" else test_months
    weather = pd.read_csv(ROOT / "data" / f"{split}_weather.csv")
    solar = pd.read_csv(ROOT / "data" / f"{split}_solar.csv")
    for col in weather.columns:
        feats[f"wx_{col}"] = weather[col].values
    for col in solar.columns:
        feats[f"sol_{col}"] = solar[col].values

gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
gbif_si = {}
for _, row in gbif.iterrows():
    m = int(row["month"])
    si = np.ones(N_CLASSES)
    for i, cls in enumerate(CLASSES):
        if cls != "Clutter":
            cmean = gbif[cls].values.mean()
            si[i] = row[cls] / cmean if cmean > 0 else 1.0
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

print("Adding E48-C flight priors (fixed wing morphology)...", flush=True)
priors = build_external_class_priors(ROOT)
train_feats = add_external_prior_features(
    train_feats, priors, include_morph=False, include_flight=True
)
test_feats = add_external_prior_features(
    test_feats, priors, include_morph=False, include_flight=True
)

# Log wing morphology to verify fix
print("\n  Wing morphology priors (verifying fix):", flush=True)
for cls in CLASSES:
    p = priors[cls]
    print(f"    {cls:<18s}: span={p['wingspan_m']:.2f}m  "
          f"area={p['wing_area_m2']:.3f}m²  "
          f"WL={p['wing_loading']:.1f}  AR={p['aspect_ratio']:.1f}  "
          f"n_bw={p['n_birdwing']}", flush=True)

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0)

X = train_feats.values.astype(np.float32)
X_test = test_feats.values.astype(np.float32)
fn = list(train_feats.columns)
print(f"\n  Total features: {len(fn)}", flush=True)

# ── GBIF prior map ─────────────────────────────────────────────────
gbif_prior_map = {}
for _, row in gbif_priors_df.iterrows():
    m_val = int(row["month"])
    prior = np.array([row[cls] for cls in CLASSES], dtype=float)
    gbif_prior_map[m_val] = prior / prior.sum()

# ── LOMO training ──────────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("LOMO TRAINING — CB=0.50 / LGB=0.30 / XGB=0.20", flush=True)
print("=" * 65, flush=True)

oof_lgb = np.zeros((len(y), N_CLASSES))
oof_xgb = np.zeros((len(y), N_CLASSES))
oof_cb = np.zeros((len(y), N_CLASSES))
test_lgb_preds, test_xgb_preds, test_cb_preds = [], [], []

for m in unique_months:
    va_idx = np.where(train_months == m)[0]
    tr_idx = np.where(train_months != m)[0]
    print(f"\n  Month {m} (train={len(tr_idx)}, val={len(va_idx)})", flush=True)

    # LGB
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx],
                         weight=sample_weights[tr_idx], feature_name=fn)
    dval = lgb.Dataset(X[va_idx], label=y[va_idx],
                       feature_name=fn, reference=dtrain)
    lgb_mdl = lgb.train(LGB_PARAMS, dtrain, 1500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(80, verbose=False),
                                   lgb.log_evaluation(0)])
    oof_lgb[va_idx] = lgb_mdl.predict(X[va_idx]).reshape(-1, N_CLASSES)
    test_lgb_preds.append(lgb_mdl.predict(X_test).reshape(-1, N_CLASSES))

    # XGB
    dtr_xgb = xgb.DMatrix(X[tr_idx], label=y[tr_idx],
                           weight=sample_weights[tr_idx], feature_names=fn)
    dva_xgb = xgb.DMatrix(X[va_idx], label=y[va_idx], feature_names=fn)
    xgb_mdl = xgb.train(XGB_PARAMS, dtr_xgb, 1400,
                         evals=[(dva_xgb, "val")],
                         early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[va_idx] = xgb_mdl.predict(dva_xgb)
    test_xgb_preds.append(xgb_mdl.predict(
        xgb.DMatrix(X_test, feature_names=fn)))

    # CB
    cb_mdl = CatBoostClassifier(**CB_PARAMS)
    cb_mdl.fit(X[tr_idx], y[tr_idx],
               eval_set=(X[va_idx], y[va_idx]),
               sample_weight=sample_weights[tr_idx], verbose=0)
    oof_cb[va_idx] = cb_mdl.predict_proba(X[va_idx])
    test_cb_preds.append(cb_mdl.predict_proba(X_test))

    fold_ens = (W_LGB * oof_lgb[va_idx]
                + W_XGB * oof_xgb[va_idx]
                + W_CB * oof_cb[va_idx])
    m_map, _ = compute_map(y[va_idx], fold_ens)
    print(f"  Ensemble fold mAP: {m_map:.4f}", flush=True)

test_lgb = np.mean(test_lgb_preds, axis=0)
test_xgb = np.mean(test_xgb_preds, axis=0)
test_cb = np.mean(test_cb_preds, axis=0)

oof_ens = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb
test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb

lomo_map, lomo_per = compute_map(y, oof_ens)
lgb_lomo, _ = compute_map(y, oof_lgb)
xgb_lomo, _ = compute_map(y, oof_xgb)
cb_lomo, _ = compute_map(y, oof_cb)

print(f"\n{'='*65}", flush=True)
print("LOMO RESULTS", flush=True)
print(f"{'='*65}", flush=True)
print(f"  LGB:       {lgb_lomo:.4f}", flush=True)
print(f"  XGB:       {xgb_lomo:.4f}", flush=True)
print(f"  CB:        {cb_lomo:.4f}", flush=True)
print(f"  Ensemble:  {lomo_map:.4f}  (E70 was 0.3507, E50 was 0.3625)", flush=True)
print(f"\n  Per-class LOMO:", flush=True)
for cls in CLASSES:
    print(f"    {cls:<18s}: {lomo_per.get(cls, 0):.4f}", flush=True)

np.save(ROOT / "oof_e71.npy", oof_ens)
np.save(ROOT / "test_e71.npy", test_ens)
save_submission(test_ens, "e71_base", cv_map=lomo_map)

# ── Per-group alpha optimisation ───────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("PER-GROUP ALPHA OPTIMISATION (month-analog proxy)", flush=True)
print(f"{'='*65}", flush=True)
print(f"  Rare classes  ({', '.join(RARE_CLASSES)})", flush=True)
print(f"  Common classes ({', '.join(COMMON_CLASSES)})", flush=True)
print(f"  Month analogs: {MONTH_ANALOG}", flush=True)


def apply_pergroup_priors(preds, months, month_alphas):
    """Apply per-group GBIF priors.

    month_alphas: dict[month → (alpha_rare, alpha_common)]
    """
    out = preds.copy()
    for i in range(len(out)):
        m = months[i]
        if m not in month_alphas:
            continue
        alpha_rare, alpha_common = month_alphas[m]
        prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
        adj = out[i].copy()
        # Rare classes
        for j in RARE_IDX:
            adj[j] = (1 - alpha_rare) * out[i, j] + alpha_rare * prior[j]
        # Common classes
        for j in COMMON_IDX:
            adj[j] = (1 - alpha_common) * out[i, j] + alpha_common * prior[j]
        adj = np.clip(adj, 1e-10, None)
        out[i] = adj / adj.sum()
    return out


# Grid search per unseen month
alpha_rare_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
alpha_common_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

best_alphas = {}  # month → (alpha_rare, alpha_common)
for unseen_month, proxy_month in MONTH_ANALOG.items():
    # Check proxy month is in training data
    if proxy_month not in unique_months:
        # Fall back to nearest available month
        available = [m for m in unique_months if m != unseen_month]
        proxy_month = min(available, key=lambda m: abs(m - unseen_month))
        print(f"  Month {unseen_month}: proxy {MONTH_ANALOG[unseen_month]} not found, "
              f"using {proxy_month}", flush=True)

    proxy_idx = np.where(train_months == proxy_month)[0]
    y_proxy = y[proxy_idx]
    oof_proxy = oof_ens[proxy_idx]

    best_m_map, best_ar, best_ac = -1.0, 0.2, 0.0
    for ar in alpha_rare_grid:
        for ac in alpha_common_grid:
            prior = gbif_prior_map.get(proxy_month,
                                       np.ones(N_CLASSES) / N_CLASSES)
            adj = oof_proxy.copy()
            for j in RARE_IDX:
                adj[:, j] = (1 - ar) * oof_proxy[:, j] + ar * prior[j]
            for j in COMMON_IDX:
                adj[:, j] = (1 - ac) * oof_proxy[:, j] + ac * prior[j]
            adj = np.clip(adj, 1e-10, None)
            adj /= adj.sum(axis=1, keepdims=True)
            m_map, _ = compute_map(y_proxy, adj)
            if m_map > best_m_map:
                best_m_map = m_map
                best_ar, best_ac = ar, ac

    best_alphas[unseen_month] = (best_ar, best_ac)
    print(f"  Month {unseen_month} (proxy={proxy_month}): "
          f"alpha_rare={best_ar:.2f}, alpha_common={best_ac:.2f} "
          f"→ proxy mAP={best_m_map:.4f}", flush=True)

print(f"\n  Optimised per-group alphas: {best_alphas}", flush=True)

# ── Apply E54 uniform winter_tilt (direct LB comparison) ──────────
print(f"\n{'='*65}", flush=True)
print("VARIANT A: E54 UNIFORM WINTER-TILT (LB BASELINE COMPARISON)", flush=True)
print(f"{'='*65}", flush=True)


def apply_uniform_priors(preds, months, alphas):
    out = preds.copy()
    n_adj = 0
    for i in range(len(out)):
        m = months[i]
        if m in alphas:
            prior = gbif_prior_map.get(m, np.ones(N_CLASSES) / N_CLASSES)
            adj = (1 - alphas[m]) * out[i] + alphas[m] * prior
            out[i] = adj / adj.sum()
            n_adj += 1
    print(f"  Adjusted {n_adj} rows for months {sorted(alphas.keys())}", flush=True)
    return out


alphas_e54 = {2: 0.22, 5: 0.12, 12: 0.24}
test_A = apply_uniform_priors(test_ens, test_months, alphas_e54)
save_submission(test_A, "e71_winter_tilt_e54alphas", cv_map=lomo_map)

# ── Apply per-group optimised alphas ──────────────────────────────
print(f"\n{'='*65}", flush=True)
print("VARIANT B: PER-GROUP OPTIMISED ALPHAS", flush=True)
print(f"{'='*65}", flush=True)

test_B = apply_pergroup_priors(test_ens, test_months, best_alphas)
rare_str = "_".join(f"m{m}r{int(v[0]*100)}" for m, v in sorted(best_alphas.items()))
save_submission(test_B, f"e71_pergroup_{rare_str}", cv_map=lomo_map)

# Evaluate on proxy months for logging
oof_B_all = apply_pergroup_priors(oof_ens, train_months,
                                   {pm: best_alphas[um]
                                    for um, pm in MONTH_ANALOG.items()
                                    if um in best_alphas})
lomo_B, lomo_B_per = compute_map(y, oof_B_all)
print(f"  LOMO after per-group correction (proxy): {lomo_B:.4f}", flush=True)

# ── Variant C: slightly stronger alphas ───────────────────────────
print(f"\n{'='*65}", flush=True)
print("VARIANT C: PER-GROUP STRONGER (+0.05 rare, +0.02 common)", flush=True)
print(f"{'='*65}", flush=True)

stronger_alphas = {
    m: (min(v[0] + 0.05, 0.6), min(v[1] + 0.02, 0.25))
    for m, v in best_alphas.items()
}
print(f"  Stronger alphas: {stronger_alphas}", flush=True)
test_C = apply_pergroup_priors(test_ens, test_months, stronger_alphas)
save_submission(test_C, "e71_pergroup_stronger", cv_map=lomo_map)

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}", flush=True)
print("E71 SUMMARY", flush=True)
print(f"{'='*65}", flush=True)
print(f"  LOMO base:            {lomo_map:.4f}  (E70 was 0.3507)", flush=True)
print(f"  Wing morphology fix:  per-class literature values", flush=True)
print(f"  Per-group alphas:     {best_alphas}", flush=True)
print(f"  Proxy LOMO after pg:  {lomo_B:.4f}", flush=True)
print(f"\n  Submission order:", flush=True)
print(f"    1. e71_winter_tilt_e54alphas  (same as E54, better model)", flush=True)
print(f"    2. e71_pergroup_*             (new per-group correction)", flush=True)
print(f"    3. e71_pergroup_stronger      (if B > baseline)", flush=True)
print(f"\nDone.", flush=True)
