"""E108: Wind-Compensated Intrinsic Base Model.

Mathematical Motivation:
1. E107 failed because calibrating decoupled rankers is highly brittle under covariate shift.
2. The 0.59 plateau is bottlenecked by the shared months (Sep/Oct).
3. In autumn, Gulls/Waders/Pigeons are confused because raw `airspeed` (ground speed)
   is blurred by wind variance. A Gull with a tailwind looks like a Wader.
4. E100 proved wind-compensated kinematics (v_air = v_ground - v_wind) provide orthogonal signal.
5. By injecting `airspeed_wind`, `headwind`, and `crosswind` DIRECTLY into the base tree ensemble,
   the model can learn true flight effort and disentangle the classes in the shared months,
   without suffering from the double-counting of post-processing.

Pipeline:
1. Build E79 pruned base features (36 features).
2. Extract ground velocity from trajectory, load wind_u/wind_v from weather.csv.
3. Compute pure physics features: airspeed_wind, drift_angle, headwind, crosswind.
4. Train the E79 tuned tree ensemble (LGB + XGB + CB).
5. Apply E96 (heading+ac1) post-processing on unseen months.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train, parse_ewkb_4d, parse_trajectory_time
from src.features import ALL_TEMPORAL, build_features
from src.metrics import compute_map, print_results
from src.submission import save_submission
from experiments.e96_nbalt_heading_ac1 import (
    BASE_ALPHA, TAU_PRIOR, TAU_NB, GAMMA,
    build_gbif_priors, apply_gated_ratio_priors, extract_heading_ac1,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe, renorm_rows, top2_margin
)

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
SEED = 42
UNSEEN_MONTHS = (2, 5, 12)

# E79 Pruned Features (36 features)
PRUNED_FEATURES = [
    'airspeed', 'radar_bird_size', 'min_z', 'max_z', 'duration', 'track_length',
    'z_mean', 'z_std', 'rcs_mean', 'rcs_std', 'rcs_max', 'rcs_min',
    'speed_mean', 'speed_std', 'speed_max', 'speed_min', 'accel_mean', 'accel_std',
    'accel_max', 'accel_min', 'turn_mean', 'turn_std', 'turn_max', 'turn_min',
    'sinuosity', 'rcs_range', 'rcs_q25', 'rcs_q75', 'rcs_iqr', 'rcs_skew', 'rcs_kurtosis',
    'rcs_trend', 'rcs_p2p', 'rcs_smoothness', 'rcs_ac1', 'rcs_ac2'
]

def extract_ground_velocity(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    vg_e = np.full(n, np.nan)
    vg_n = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            times = parse_trajectory_time(row["trajectory_time"])
            if len(pts) < 3 or len(times) < 3:
                continue
            lons = np.array([p[0] for p in pts], dtype=float)
            lats = np.array([p[1] for p in pts], dtype=float)

            dt = np.diff(times).astype(float)
            dt = np.maximum(dt, 1e-3)
            dur = float(times[-1] - times[0])
            if not np.isfinite(dur) or dur <= 1e-3:
                continue

            mean_lat = float(np.mean(lats))
            lon_scale = 111000.0 * float(np.cos(np.deg2rad(mean_lat)))
            dx = np.diff(lons) * lon_scale
            dy = np.diff(lats) * 111000.0

            vx = dx / dt
            vy = dy / dt
            vg_e[i] = float(np.mean(vx))
            vg_n[i] = float(np.mean(vy))
            ok[i] = True
        except Exception:
            continue
    return vg_e, vg_n, ok

def build_wind_features(df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    vg_e, vg_n, ok = extract_ground_velocity(df)
    w_u = pd.to_numeric(wx_df["wind_u"], errors="coerce").values.astype(float)
    w_v = pd.to_numeric(wx_df["wind_v"], errors="coerce").values.astype(float)
    
    vg_norm = np.sqrt(vg_e * vg_e + vg_n * vg_n) + 1e-12
    vg_hat_e = vg_e / vg_norm
    vg_hat_n = vg_n / vg_norm
    
    # Air velocity
    va_e = vg_e - w_u
    va_n = vg_n - w_v
    air_w = np.sqrt(va_e * va_e + va_n * va_n)
    
    # Drift angle
    dot = vg_e * va_e + vg_n * va_n
    cross = vg_e * va_n - vg_n * va_e
    drift = np.abs(np.arctan2(cross, dot))
    
    # Headwind / Crosswind
    headwind = -(w_u * vg_hat_e + w_v * vg_hat_n)
    proj = (w_u * vg_hat_e + w_v * vg_hat_n)
    cw_e = w_u - proj * vg_hat_e
    cw_n = w_v - proj * vg_hat_n
    crosswind = np.sqrt(cw_e * cw_e + cw_n * cw_n)
    
    return pd.DataFrame({
        "airspeed_wind": np.where(ok, air_w, np.nan),
        "drift_angle": np.where(ok, drift, np.nan),
        "headwind": np.where(ok, headwind, np.nan),
        "crosswind": np.where(ok, crosswind, np.nan)
    })

def main() -> None:
    print("=" * 70, flush=True)
    print("E108 WIND-COMPENSATED INTRINSIC BASE MODEL".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()
    train_wx = pd.read_csv(ROOT / "data" / "train_weather.csv")
    test_wx = pd.read_csv(ROOT / "data" / "test_weather.csv")

    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.astype(int)
    train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    # 1. Build Base Features
    print("\nBuilding base features...", flush=True)
    feature_sets = ["core", "tabular", "rcs_fft"]
    Xtr_base = build_features(train_df, feature_sets=feature_sets)
    Xte_base = build_features(test_df, feature_sets=feature_sets)
    
    # Keep only E79 pruned features
    keep_cols = [c for c in PRUNED_FEATURES if c in Xtr_base.columns]
    Xtr_base = Xtr_base[keep_cols]
    Xte_base = Xte_base[keep_cols]

    # 2. Build Wind Features
    print("\nBuilding wind-compensated physics features...", flush=True)
    Xtr_wind = build_wind_features(train_df, train_wx)
    Xte_wind = build_wind_features(test_df, test_wx)
    
    Xtr_df = pd.concat([Xtr_base, Xtr_wind], axis=1)
    Xte_df = pd.concat([Xte_base, Xte_wind], axis=1)

    Xtr_df = Xtr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte_df = Xte_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xtr = Xtr_df.to_numpy(np.float32)
    Xte = Xte_df.to_numpy(np.float32)
    print(f"Final Feature Dim: {Xtr.shape[1]}", flush=True)

    # 3. Train Models (LOMO for OOF, Full for Test)
    print("\nTraining E79 Tuned Ensemble (LGB + XGB + CB)...", flush=True)
    unique_months = sorted(set(train_months))
    
    oof_lgb = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    oof_xgb = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    oof_cb = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    
    # Class weights for imbalanced learning
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w_class = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    sample_weights = w_class[y]

    for m in unique_months:
        tr_idx = np.where(train_months != m)[0]
        va_idx = np.where(train_months == m)[0]
        
        # LGB (Fixed min_data_in_leaf to prevent -inf gain warning loop)
        lgb = LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES,
            learning_rate=0.0159, num_leaves=31, max_depth=6,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            n_estimators=300, class_weight="balanced", random_state=SEED, device="cpu", n_jobs=-1,
            verbose=-1
        )
        lgb.fit(Xtr[tr_idx], y[tr_idx])
        oof_lgb[va_idx] = lgb.predict_proba(Xtr[va_idx])
        
        # XGB
        xgb = XGBClassifier(
            objective="multi:softprob", num_class=N_CLASSES,
            learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            n_estimators=300, random_state=SEED, device="cpu", n_jobs=-1
        )
        xgb.fit(Xtr[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
        oof_xgb[va_idx] = xgb.predict_proba(Xtr[va_idx])
        
        # CB
        cb = CatBoostClassifier(
            loss_function="MultiClass", iterations=400, learning_rate=0.05,
            depth=6, random_seed=SEED, task_type="CPU", thread_count=-1, verbose=0
        )
        cb.fit(Xtr[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
        oof_cb[va_idx] = cb.predict_proba(Xtr[va_idx])

    # Full train
    lgb = LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES,
        learning_rate=0.0159, num_leaves=31, max_depth=6,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        n_estimators=350, class_weight="balanced", random_state=SEED, device="cpu", n_jobs=-1,
        verbose=-1
    )
    lgb.fit(Xtr, y)
    test_lgb = lgb.predict_proba(Xte)
    
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        n_estimators=350, random_state=SEED, device="cpu", n_jobs=-1
    )
    xgb.fit(Xtr, y, sample_weight=sample_weights)
    test_xgb = xgb.predict_proba(Xte)
    
    cb = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, learning_rate=0.05,
        depth=6, random_seed=SEED, task_type="CPU", thread_count=-1, verbose=0
    )
    cb.fit(Xtr, y, sample_weight=sample_weights)
    test_cb = cb.predict_proba(Xte)

    # Ensemble weights (E79 optimal)
    W_LGB, W_XGB, W_CB = 0.50, 0.40, 0.10
    oof_ens = renorm_rows(W_LGB * oof_lgb + W_XGB * oof_xgb + W_CB * oof_cb)
    test_ens = renorm_rows(W_LGB * test_lgb + W_XGB * test_xgb + W_CB * test_cb)

    m, per = compute_map(y, oof_ens)
    print_results(m, per, "E108 Wind-Compensated Base (LOMO)")
    np.save("oof_e108.npy", oof_ens)
    np.save("test_e108.npy", test_ens)
    save_submission(test_ens, "e108_windbase_raw", cv_map=m)

    # 4. Apply E96 Post-Processing (Heading + AC1)
    print("\nApplying E96 Unseen-Month Post-Processing...", flush=True)
    p_train = counts / counts.sum()
    priors = build_gbif_priors(p_train)

    train_heading, train_ac1, train_ok = extract_heading_ac1(train_df)
    test_heading, test_ac1, test_ok = extract_heading_ac1(test_df)

    test_p0, changed = apply_gated_ratio_priors(
        test_ens, test_months, p_train, priors, BASE_ALPHA, tau=TAU_PRIOR
    )
    
    margin0 = top2_margin(test_p0)
    gate = np.isin(test_months, UNSEEN_MONTHS) & (margin0 < TAU_NB)
    
    size_levels, log_p_size, mu, sig = build_nb_params(
        train_df, y, train_heading, train_ac1, train_ok, use_heading=True, use_ac1=True
    )
    loglike_test = compute_log_p_u_given_c(
        test_df, size_levels, log_p_size, mu, sig, test_heading, test_ac1, test_ok, use_heading=True, use_ac1=True
    )
    
    P_final = apply_nb_poe(test_p0, loglike_test, gamma=GAMMA, gate=gate)
    
    name = f"e108_windbase_pp_heading_ac1_tau{TAU_NB:.2f}_g{GAMMA:.2f}_priortau{TAU_PRIOR:.2f}"
    save_submission(P_final, name, cv_map=None)
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
