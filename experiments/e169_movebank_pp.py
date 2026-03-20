"""E169b: Movebank GPS Speed Priors + Domain PP on E79 base.

E169 proved no feature set beats E79's 36 on LOMO. This script focuses
on the genuinely new contribution: Movebank GPS speed distributions
and domain-informed evidence channels for post-processing.

Tests: train priors vs Movebank vs Alerstam for speed channel,
plus cormorant wind model, grassland, and tidal evidence.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.features import haversine
from src.submission import save_submission
from src.postprocessing import (
    UNSEEN_MONTHS, BASE_ALPHA, N_CLASSES,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

ALERSTAM = {
    "Birds of Prey": (10.8, 2.4), "Cormorants": (14.4, 1.5),
    "Ducks": (15.6, 2.4), "Geese": (17.2, 2.5),
    "Gulls": (12.4, 2.2), "Pigeons": (15.2, 2.5),
    "Songbirds": (13.1, 2.2), "Waders": (14.9, 2.2),
}

CLASS_FILES = {
    "Gulls": ["movebank_deltatrack_gulls.csv", "movebank_lbbg_adult.csv",
              "movebank_lbbg_juvenile.csv", "movebank_lbbg_zeebrugge.csv",
              "movebank_medgull_antwerpen.csv"],
    "Waders": ["movebank_oystercatcher_ameland.csv", "movebank_oystercatcher_balgzand.csv",
               "movebank_oystercatcher_schier.csv", "movebank_oystercatcher_vlieland.csv",
               "movebank_oystercatcher_westerschelde.csv"],
    "Birds of Prey": ["movebank_marshharrier_groningen.csv", "movebank_marshharrier_waterland.csv",
                       "movebank_marshharrier_antwerpen.csv"],
    "Geese": ["movebank_goose_newyear.csv", "movebank_whitefronted_goose_family.csv",
              "movebank_whitefronted_goose_alterra.csv"],
}

print("=" * 70, flush=True)
print("E169b: MOVEBANK GPS PRIORS + DOMAIN PP".center(70), flush=True)
print("=" * 70, flush=True)

# ── Phase 1: Process Movebank GPS ────────────────────────────────────
print("\n--- Phase 1: Movebank GPS Processing ---", flush=True)

movebank_priors = {}
for cls, files in CLASS_FILES.items():
    all_speeds = []
    for fname in files:
        path = ROOT / "data" / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["timestamp", "location_lat", "location_long"])
            if len(df) > 200000:
                df = df.sample(200000, random_state=SEED)
            df = df.sort_values("timestamp").reset_index(drop=True)
            lats, lons = df["location_lat"].values, df["location_long"].values
            ts = pd.to_datetime(df["timestamp"])
            dt_sec = ts.diff().dt.total_seconds().values[1:]
            valid = (dt_sec > 0) & (dt_sec < 1800)
            if valid.sum() < 100:
                continue
            dists = np.array([haversine(lons[i], lats[i], lons[i+1], lats[i+1])
                              for i in range(len(lons)-1)])
            speeds = dists[valid] / dt_sec[valid]
            flight = speeds[(speeds > 2) & (speeds < 35)]
            all_speeds.extend(flight.tolist())
            print(f"  {fname}: {len(flight)} flight speeds", flush=True)
        except Exception as e:
            print(f"  ERROR {fname}: {e}", flush=True)

    if len(all_speeds) > 500:
        arr = np.array(all_speeds)
        movebank_priors[cls] = (float(np.mean(arr)), float(np.std(arr)))
        a = ALERSTAM.get(cls, ("N/A", "N/A"))
        print(f"  >> {cls}: Movebank mean={movebank_priors[cls][0]:.1f} sd={movebank_priors[cls][1]:.1f} "
              f"| Alerstam mean={a[0]} sd={a[1]}", flush=True)

for cls in CLASSES:
    if cls not in movebank_priors:
        movebank_priors[cls] = ALERSTAM.get(cls, (13.0, 3.0))
        print(f"  >> {cls}: using Alerstam {movebank_priors[cls]}", flush=True)

# ── Phase 2: Load data ──────────────────────────────────────────────
print("\n--- Phase 2: Load Data ---", flush=True)
train_df = load_train()
test_df = load_test()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); le.fit(CLASSES)
y = le.transform(train_df["bird_group"])
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()
test_base = renorm_rows(np.load(ROOT / "test_e79.npy").astype(float))

# Pre-load external columns
for split, df in [("train", train_df), ("test", test_df)]:
    for csv_name, cols in [("altitude_winds", ["wind_at_bird_alt"]),
                           ("tidal", ["hours_since_high_tide"]),
                           ("landuse", ["dist_to_grassland_m"])]:
        path = ROOT / "data" / f"{split}_{csv_name}.csv"
        if path.exists():
            ext = pd.read_csv(path)
            for col in cols:
                if col in ext.columns:
                    df[f"ext_{col}"] = pd.to_numeric(ext[col], errors="coerce").fillna(0).values

import src.validate as _val
_val._cache.clear()
_val._cache["train"] = (train_df, y, train_months)
from src.validate import eval_pp

# ── Phase 3: PP Experiments ─────────────────────────────────────────
print("\n--- Phase 3: PP Experiments ---", flush=True)

CH_MAP = {
    "cormorant_wind": ("ext_wind_at_bird_alt", 0.5, 0.3),
    "grassland": ("ext_dist_to_grassland_m", 0.3, 100.0),
    "tidal": ("ext_hours_since_high_tide", 0.4, 0.5),
}

def make_pp(speed_source="train", extra_channels=None, gamma=0.10, tau_nb=0.25):
    def pp_fn(preds, tdf, tm, trdf, yl):
        c = np.bincount(yl, minlength=N_CLASSES).astype(float)
        pt = c / c.sum()
        priors = build_gbif_priors(pt)
        out, _ = apply_gated_ratio_priors(preds, tm, pt, priors, BASE_ALPHA, tau=0.15)

        sp = pd.to_numeric(trdf["airspeed"], errors="coerce").values.astype(float)
        mz = pd.to_numeric(trdf["min_z"], errors="coerce").values.astype(float)
        xz = pd.to_numeric(trdf["max_z"], errors="coerce").values.astype(float)
        cont_tr = {"speed": sp, "alt_mid": 0.5*(mz+xz), "alt_range": xz-mz}
        w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
        ms = {}

        if extra_channels:
            for ch in extra_channels:
                if ch == "cormorant_wind" and "ext_wind_at_bird_alt" in trdf.columns:
                    wind = trdf["ext_wind_at_bird_alt"].values.astype(float)
                    cont_tr["cormorant_residual"] = np.abs(sp - (0.70 * wind + 14.4))
                    w["cormorant_residual"] = 0.5; ms["cormorant_residual"] = 0.3
                elif ch in CH_MAP:
                    col, wt, msig = CH_MAP[ch]
                    if col in trdf.columns:
                        cont_tr[ch] = trdf[col].values.astype(float)
                        w[ch] = wt; ms[ch] = msig

        sl, lps, mu, sig = build_nb_params(trdf, yl, cont_tr, min_sigma=ms)

        if speed_source == "movebank":
            for i, cls in enumerate(CLASSES):
                if cls in movebank_priors:
                    mu["speed"][i] = movebank_priors[cls][0]
                    sig["speed"][i] = max(movebank_priors[cls][1], 0.5)
        elif speed_source == "alerstam":
            for i, cls in enumerate(CLASSES):
                if cls in ALERSTAM:
                    mu["speed"][i], sig["speed"][i] = ALERSTAM[cls]

        sp_t = pd.to_numeric(tdf["airspeed"], errors="coerce").values.astype(float)
        mz_t = pd.to_numeric(tdf["min_z"], errors="coerce").values.astype(float)
        xz_t = pd.to_numeric(tdf["max_z"], errors="coerce").values.astype(float)
        cont_te = {"speed": sp_t, "alt_mid": 0.5*(mz_t+xz_t), "alt_range": xz_t-mz_t}
        if extra_channels:
            for ch in extra_channels:
                if ch == "cormorant_wind" and "ext_wind_at_bird_alt" in tdf.columns:
                    wind_t = tdf["ext_wind_at_bird_alt"].values.astype(float)
                    cont_te["cormorant_residual"] = np.abs(sp_t - (0.70*wind_t+14.4))
                elif ch in CH_MAP:
                    col = CH_MAP[ch][0]
                    if col in tdf.columns:
                        cont_te[ch] = tdf[col].values.astype(float)

        ll = compute_log_p_u_given_c(tdf, sl, lps, cont_te, w, None, mu, sig)
        gate = np.isin(tm, UNSEEN_MONTHS) & (top2_margin(out) < tau_nb)
        return apply_nb_poe(out, ll, gamma=gamma, gate=gate)
    return pp_fn

# Run experiments
configs = [
    ("train priors", "train", None),
    ("Movebank priors", "movebank", None),
    ("Alerstam priors", "alerstam", None),
    ("Movebank+cormorant", "movebank", ["cormorant_wind"]),
    ("Movebank+grassland", "movebank", ["grassland"]),
    ("Movebank+tidal", "movebank", ["tidal"]),
    ("Movebank+all_domain", "movebank", ["cormorant_wind", "grassland", "tidal"]),
    ("Alerstam+grassland", "alerstam", ["grassland"]),
]

results = {}
for label, src, chs in configs:
    r = eval_pp(make_pp(src, chs, gamma=0.10, tau_nb=0.25), verbose=False)
    cal = r.get("calibrated_lb")
    cal_s = f"{cal:.3f}" if cal else "N/A"
    results[label] = r
    print(f"  {label:25s}: IW-mAP={r['estimated_lb']:.4f}  cal_LB={cal_s}", flush=True)

best_label = max(results, key=lambda k: results[k]["estimated_lb"])
print(f"\n  BEST: {best_label} (IW-mAP={results[best_label]['estimated_lb']:.4f})", flush=True)

# Gamma sweep on best
best_cfg = [(l, s, c) for l, s, c in configs if l == best_label][0]
_, best_src, best_chs = best_cfg

print(f"\n--- Gamma/tau sweep on: {best_label} ---", flush=True)
best_iw, best_g, best_t = -1, 0.10, 0.25
for g in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for t in [0.20, 0.25, 0.30]:
        r = eval_pp(make_pp(best_src, best_chs, gamma=g, tau_nb=t), verbose=False)
        cal = r.get("calibrated_lb")
        cal_s = f"{cal:.3f}" if cal else "N/A"
        tag = " ***" if r["estimated_lb"] > best_iw else ""
        print(f"  g={g:.2f} t={t:.2f}: IW-mAP={r['estimated_lb']:.4f} cal_LB={cal_s}{tag}", flush=True)
        if r["estimated_lb"] > best_iw:
            best_iw, best_g, best_t = r["estimated_lb"], g, t

# Full report on best
print(f"\n--- Best config: g={best_g:.2f} t={best_t:.2f} ---", flush=True)
final_r = eval_pp(make_pp(best_src, best_chs, best_g, best_t), verbose=True)

# Generate submission
priors = build_gbif_priors(p_train)
test_pp, _ = apply_gated_ratio_priors(test_base.copy(), test_months, p_train, priors, BASE_ALPHA, tau=0.15)

sp_tr = pd.to_numeric(train_df["airspeed"], errors="coerce").values.astype(float)
mz_tr = pd.to_numeric(train_df["min_z"], errors="coerce").values.astype(float)
xz_tr = pd.to_numeric(train_df["max_z"], errors="coerce").values.astype(float)
cont_tr = {"speed": sp_tr, "alt_mid": 0.5*(mz_tr+xz_tr), "alt_range": xz_tr-mz_tr}
w = {"speed": 1.0, "alt_mid": 1.0, "alt_range": 0.5}
ms = {}
if best_chs:
    for ch in best_chs:
        if ch == "cormorant_wind" and "ext_wind_at_bird_alt" in train_df.columns:
            wind = train_df["ext_wind_at_bird_alt"].values.astype(float)
            cont_tr["cormorant_residual"] = np.abs(sp_tr - (0.70*wind+14.4))
            w["cormorant_residual"] = 0.5; ms["cormorant_residual"] = 0.3
        elif ch in CH_MAP:
            col, wt, msig = CH_MAP[ch]
            if col in train_df.columns:
                cont_tr[ch] = train_df[col].values.astype(float)
                w[ch] = wt; ms[ch] = msig

sl, lps, mu, sig = build_nb_params(train_df, y, cont_tr, min_sigma=ms)
if best_src == "movebank":
    for i, cls in enumerate(CLASSES):
        if cls in movebank_priors:
            mu["speed"][i] = movebank_priors[cls][0]
            sig["speed"][i] = max(movebank_priors[cls][1], 0.5)
elif best_src == "alerstam":
    for i, cls in enumerate(CLASSES):
        if cls in ALERSTAM:
            mu["speed"][i], sig["speed"][i] = ALERSTAM[cls]

sp_te = pd.to_numeric(test_df["airspeed"], errors="coerce").values.astype(float)
mz_te = pd.to_numeric(test_df["min_z"], errors="coerce").values.astype(float)
xz_te = pd.to_numeric(test_df["max_z"], errors="coerce").values.astype(float)
cont_te = {"speed": sp_te, "alt_mid": 0.5*(mz_te+xz_te), "alt_range": xz_te-mz_te}
if best_chs:
    for ch in best_chs:
        if ch == "cormorant_wind" and "ext_wind_at_bird_alt" in test_df.columns:
            wind_t = test_df["ext_wind_at_bird_alt"].values.astype(float)
            cont_te["cormorant_residual"] = np.abs(sp_te - (0.70*wind_t+14.4))
        elif ch in CH_MAP:
            col = CH_MAP[ch][0]
            if col in test_df.columns:
                cont_te[ch] = test_df[col].values.astype(float)

ll = compute_log_p_u_given_c(test_df, sl, lps, cont_te, w, None, mu, sig)
gate = np.isin(test_months, UNSEEN_MONTHS) & (top2_margin(test_pp) < best_t)
test_final = apply_nb_poe(test_pp, ll, gamma=best_g, gate=gate)

cal_lb = final_r.get("calibrated_lb") or final_r["estimated_lb"]
save_submission(test_final, f"e169b_movebank_g{best_g:.2f}_t{best_t:.2f}", cv_map=cal_lb)

print("\n" + "=" * 70, flush=True)
print("DONE".center(70), flush=True)
print("=" * 70, flush=True)
