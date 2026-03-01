"""E70: BoP + Songbirds specialists using E68 features (no global feature dilution).

Hypothesis
----------
E68 added 8 enhanced bio-shape / RCS-texture features + 4 solar-derived features.
Adding them to the full multiclass ensemble hurt (feature dilution), but the E68
log suggests *Birds of Prey* and *Songbirds* improved individually.

So we train binary specialists for:
  - Birds of Prey
  - Songbirds
using ONLY the E68 feature block, and then inject these specialists into the
current best base predictions (E50 -> E54/E67 family).

Mechanism
---------
Let p be the base multiclass probabilities and s_c(x) a binary specialist for class c.
We update only columns c in {BoP, Songbirds}:

  p'_c = (1-λ_c) p_c + λ_c s_c
  p' = renorm_rows(p')

Then apply the proven unseen-month GBIF ratio tilt with uncertainty gating (E67):
  months {2,5,12}, alphas {2:0.22, 5:0.12, 12:0.24}, gate margin < tau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train  # noqa: E402
from src.features import ALL_TEMPORAL, build_features  # noqa: E402
from src.metrics import compute_map  # noqa: E402
from src.submission import save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BASE_ALPHA = {2: 0.22, 5: 0.12, 12: 0.24}
GATE_TAU = 0.15  # E67: tau=0.15 achieved LB 0.56

# Specialist injection is intentionally targeted at *unseen* months only (2/5/12),
# since the motivation is the train→test month-shift, not improving within-train months.
UNSEEN_MONTHS = (2, 5, 12)
INJECT_MARGIN_TAU = 0.25  # only touch relatively uncertain examples
LAM_BOP = 0.35
LAM_SONG = 0.25
DELTA_BOP = 0.10
DELTA_SONG = 0.08
MIN_SPEC_BOP = 0.15
MIN_SPEC_SONG = 0.20


def renorm_rows(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def top2_margin(pred: np.ndarray) -> np.ndarray:
    order = np.argsort(-pred, axis=1)
    p1 = pred[np.arange(pred.shape[0]), order[:, 0]]
    p2 = pred[np.arange(pred.shape[0]), order[:, 1]]
    return p1 - p2


def build_gbif_priors(p_train: np.ndarray) -> dict[int, np.ndarray]:
    """Build month priors from GBIF seasonal indices (as used in E38/E58/E67)."""
    gbif = pd.read_csv(ROOT / "data" / "gbif_monthly_counts.csv")
    si = {}
    for _, row in gbif.iterrows():
        month = int(row["month"])
        vals = np.ones(len(CLASSES))
        for i, cls in enumerate(CLASSES):
            if cls == "Clutter":
                vals[i] = 1.0
            else:
                class_mean = gbif[cls].values.mean()
                vals[i] = row[cls] / class_mean if class_mean > 0 else 1.0
        si[month] = vals

    priors = {}
    for month in range(1, 13):
        raw = np.maximum(p_train * si[month], 1e-8)
        priors[month] = raw / raw.sum()
    return priors


def apply_gated_ratio_priors(
    preds: np.ndarray,
    months: np.ndarray,
    p_train: np.ndarray,
    priors: dict[int, np.ndarray],
    alpha_map: dict[int, float],
    tau: float,
) -> tuple[np.ndarray, int]:
    """E67-style: apply ratio tilt only for uncertain examples (margin < tau)."""
    out = preds.copy()
    order = np.argsort(-out, axis=1)
    p1 = out[np.arange(out.shape[0]), order[:, 0]]
    p2 = out[np.arange(out.shape[0]), order[:, 1]]
    margin = p1 - p2

    changed = 0
    for month, alpha in alpha_map.items():
        mask_m = months == month
        if mask_m.sum() == 0 or alpha == 0:
            continue
        gate = mask_m & (margin < tau)
        if gate.sum() == 0:
            continue
        ratio = (priors[month] / np.maximum(p_train, 1e-12)) ** alpha
        out[gate] = out[gate] * ratio
        out[gate] = out[gate] / np.clip(out[gate].sum(axis=1, keepdims=True), 1e-12, None)
        changed += int(gate.sum())
    return renorm_rows(out), changed


def add_derived_solar_features(df_feat: pd.DataFrame, solar_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the 4 solar-derived 'biological time' features from E68."""
    hours_since = solar_df["hours_since_sunrise"].values.astype(float)
    daylight = solar_df["daylight_hours"].values.astype(float)
    elev = solar_df["solar_elevation"].values.astype(float)

    df_feat = df_feat.copy()
    df_feat["hours_from_solar_noon"] = np.abs(hours_since - daylight / 2.0)
    df_feat["is_thermal_window"] = (elev > 25).astype(float)
    df_feat["is_dawn_dusk"] = ((elev > -6) & (elev < 15)).astype(float)
    df_feat["is_afternoon_thermal"] = ((elev > 15) & (hours_since > daylight / 2.0)).astype(float)
    return df_feat


def train_lomo_binary_specialist(
    X: np.ndarray,
    y_bin: np.ndarray,
    train_months: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train binary LGB specialist with LOMO CV. Returns (oof, test_mean)."""
    unique_months = sorted(np.unique(train_months))
    oof = np.zeros(len(y_bin), dtype=float)
    test_acc = np.zeros(len(X_test), dtype=float)

    print(f"\nTraining specialist: {name}", flush=True)
    for m in unique_months:
        va_idx = np.where(train_months == m)[0]
        tr_idx = np.where(train_months != m)[0]

        y_tr = y_bin[tr_idx]
        y_va = y_bin[va_idx]

        n_pos = int(y_tr.sum())
        n_neg = int((1 - y_tr).sum())
        if n_pos < 5:
            # Too few positives: fallback to prevalence.
            fallback = n_pos / max(len(y_tr), 1)
            oof[va_idx] = fallback
            test_acc += fallback / len(unique_months)
            print(f"  Month {m}: pos={n_pos} fallback={fallback:.4f}", flush=True)
            continue

        spw = n_neg / max(n_pos, 1)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.8,
            "scale_pos_weight": spw,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        dtrain = lgb.Dataset(X[tr_idx], label=y_tr, feature_name=feature_names)
        dval = lgb.Dataset(X[va_idx], label=y_va, feature_name=feature_names, reference=dtrain)
        mdl = lgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        oof[va_idx] = mdl.predict(X[va_idx])
        test_acc += mdl.predict(X_test) / len(unique_months)

        ap = average_precision_score(y_va, oof[va_idx])
        print(f"  Month {m}: pos={n_pos} AP={ap:.4f} (n_va={len(va_idx)})", flush=True)

    ap_all = average_precision_score(y_bin, oof)
    print(f"  Overall AP ({name}): {ap_all:.4f}", flush=True)
    return oof, test_acc


def inject_specialists(
    preds: np.ndarray,
    spec_map: dict[int, tuple[np.ndarray, float]],
) -> np.ndarray:
    """Inject specialists into selected columns with per-class lambdas."""
    out = preds.copy()
    for cls_idx, (spec_pred, lam) in spec_map.items():
        out[:, cls_idx] = (1.0 - lam) * out[:, cls_idx] + lam * spec_pred
    return renorm_rows(out)


print("=" * 70, flush=True)
print("E70: BOP+SONGBIRDS SPECIALISTS (E68 FEATURES ONLY)".center(70), flush=True)
print("=" * 70, flush=True)

train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

# Base predictions (best pipeline backbone).
oof_base = renorm_rows(np.load(ROOT / "oof_e50.npy").astype(float))
test_base = renorm_rows(np.load(ROOT / "test_e50.npy").astype(float))
base_map, base_per = compute_map(y, oof_base)
print(f"\nBase (oof_e50) LOMO mAP: {base_map:.4f}", flush=True)

# Train priors for ratio tilt.
counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
p_train = counts / counts.sum()
priors = build_gbif_priors(p_train)

# Build ONLY enhanced bio-shape features (8 cols).
print("\nBuilding E68 enhanced bio-shape features...", flush=True)
feat_sets = ["enhanced_bio_shape"]
X_train_df = build_features(train_df, feature_sets=feat_sets)
X_test_df = build_features(test_df, feature_sets=feat_sets)

# Remove temporal leaks (defensive; these features shouldn't include them, but keep consistent).
keep_cols = [c for c in X_train_df.columns if c not in ALL_TEMPORAL]
X_train_df = X_train_df[keep_cols]
X_test_df = X_test_df[keep_cols]

# Add 4 derived solar features from E68.
train_solar = pd.read_csv(ROOT / "data" / "train_solar.csv")
test_solar = pd.read_csv(ROOT / "data" / "test_solar.csv")
X_train_df = add_derived_solar_features(X_train_df, train_solar)
X_test_df = add_derived_solar_features(X_test_df, test_solar)

specialist_cols = list(X_train_df.columns)
print(f"  Specialist feature count: {len(specialist_cols)}", flush=True)
print(f"  Specialist features: {specialist_cols}", flush=True)

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

# Train two specialists with LOMO CV.
bop_idx = CLASSES.index("Birds of Prey")
song_idx = CLASSES.index("Songbirds")

y_bop = (y == bop_idx).astype(int)
y_song = (y == song_idx).astype(int)

oof_bop, test_bop = train_lomo_binary_specialist(
    X_train, y_bop, train_months, X_test, specialist_cols, name="Birds of Prey"
)
oof_song, test_song = train_lomo_binary_specialist(
    X_train, y_song, train_months, X_test, specialist_cols, name="Songbirds"
)

# Inject specialists ONLY on unseen-month test rows with a conservative gate.
print("\nGenerating test submission (unseen-month specialist injection -> gated priors)...", flush=True)
margin_base = top2_margin(test_base)
unseen_mask = np.isin(test_months, UNSEEN_MONTHS)
uncertain = margin_base < INJECT_MARGIN_TAU

# Gate the injection using the *feature-derived* conditions that motivated E68:
# - BoP: thermals (high solar elevation) → soaring / circling pattern
# - Songbirds: dawn/dusk → bounding flight activity
is_thermal = X_test_df["is_thermal_window"].values > 0.5
is_dawn_dusk = X_test_df["is_dawn_dusk"].values > 0.5

gate_bop = (
    unseen_mask
    & uncertain
    & is_thermal
    & (test_bop >= (test_base[:, bop_idx] + DELTA_BOP))
    & (test_bop >= MIN_SPEC_BOP)
)
gate_song = (
    unseen_mask
    & uncertain
    & is_dawn_dusk
    & (test_song >= (test_base[:, song_idx] + DELTA_SONG))
    & (test_song >= MIN_SPEC_SONG)
)

print(
    f"  Inject gates: unseen={int(unseen_mask.sum())} uncertain={int(uncertain.sum())} "
    f"bop_gate={int(gate_bop.sum())} song_gate={int(gate_song.sum())}",
    flush=True,
)
for m in UNSEEN_MONTHS:
    mm = test_months == m
    print(
        f"    month={m}: n={int(mm.sum())} bop_gate={int((gate_bop & mm).sum())} "
        f"song_gate={int((gate_song & mm).sum())}",
        flush=True,
    )

test_spec = test_base.copy()
test_spec[gate_bop, bop_idx] = (1.0 - LAM_BOP) * test_spec[gate_bop, bop_idx] + LAM_BOP * test_bop[gate_bop]
test_spec[gate_song, song_idx] = (1.0 - LAM_SONG) * test_spec[gate_song, song_idx] + LAM_SONG * test_song[gate_song]
test_spec = renorm_rows(test_spec)

test_final, n_changed = apply_gated_ratio_priors(
    test_spec, test_months, p_train, priors, BASE_ALPHA, tau=GATE_TAU
)

print(f"  Applied gated priors: tau={GATE_TAU:.2f}, changed_rows={n_changed}", flush=True)
save_submission(
    test_final,
    (
        f"e70_unseeninj_bop{LAM_BOP:.2f}_song{LAM_SONG:.2f}"
        f"_marg{INJECT_MARGIN_TAU:.2f}_priors_tau{GATE_TAU:.2f}"
    ),
    cv_map=base_map,
)

print("\nDone.", flush=True)

