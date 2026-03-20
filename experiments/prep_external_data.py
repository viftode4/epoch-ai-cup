"""Fetch and prepare ALL external data sources for bird radar classification.

Data sources (from research in project_external_data.md / project_new_features.md):
1. KNMI Pressure — station 280 (Eelde), hourly API (station 277 lacks P)
2. Insect activity index — hardcode from Hallmann et al. 2017
3. Tidal data — Rijkswaterstaat via ddlpy, Delfzijl station WATHTE/NAP
4. Moon/twilight — skyfield astronomical computations (Eemshaven lat/lon)
5. CAPE — Open-Meteo Historical Forecast API (free, no key)
6. Water proximity — OSM Overpass API (38K coastline points)
7. Visibility — KNMI station 280 (Eelde), VV/N/M/R variables
8. Turbine distance — OSM wind generators (135 turbines near Eemshaven)

Each section saves train/test CSVs aligned 1:1 with train.csv and test.csv rows.
Run: python experiments/prep_external_data.py

Dependencies: pip install rws-ddlpy skyfield scipy requests
(install ddlpy from git: pip install git+https://github.com/Deltares/ddlpy.git)
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_train, load_test, parse_ewkb_4d

# ── Constants ──────────────────────────────────────────────────────
RADAR_LAT = 53.4550
RADAR_LON = 6.7900

# ── Load train/test ───────────────────────────────────────────────
print("=" * 60)
print("EXTERNAL DATA PREPARATION")
print("=" * 60)

train_df = load_train()
test_df = load_test()
train_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"])
test_ts = pd.to_datetime(test_df["timestamp_start_radar_utc"])

print(f"Train: {len(train_df)} rows, {train_ts.min()} to {train_ts.max()}")
print(f"Test:  {len(test_df)} rows, {test_ts.min()} to {test_ts.max()}")


def get_track_centroids(df):
    lats, lons = [], []
    for hex_str in df["trajectory"]:
        pts = parse_ewkb_4d(hex_str)
        lons.append(np.mean([p[0] for p in pts]))
        lats.append(np.mean([p[1] for p in pts]))
    return np.array(lats), np.array(lons)


print("Extracting track centroids...")
train_lats, train_lons = get_track_centroids(train_df)
test_lats, test_lons = get_track_centroids(test_df)


def merge_hourly(timestamps, source_df, cols):
    """Match each timestamp to nearest hour in source (timezone-naive)."""
    results = []
    for t in timestamps.dt.floor("h"):
        idx = source_df.index.get_indexer([t], method="nearest")[0]
        results.append(source_df.iloc[idx][cols].to_dict())
    return pd.DataFrame(results)


def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    """Haversine distance from one point to array of points (meters)."""
    R = 6371000
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


results_summary = {}

# ======================================================================
# 1. KNMI PRESSURE (station 280 Eelde, ~50km from Eemshaven)
# ======================================================================
# Station 277 (Lauwersoog) has no pressure data.
# Eelde is the nearest full weather station.
# Target: migration wave detection — pressure drops trigger mass migration.
# ======================================================================
print("\n" + "=" * 60)
print("1. KNMI PRESSURE (Eelde stn 280)")
print("=" * 60)
try:
    url = "https://daggegevens.knmi.nl/klimatologie/uurgegevens"
    resp = requests.post(url, data={
        "stns": "280", "vars": "P",
        "start": "2023080101", "end": "2024053124", "fmt": "json",
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df_p = pd.DataFrame(data)
    df_p["datetime"] = (
        pd.to_datetime(df_p["date"]).dt.tz_localize(None)
        + pd.to_timedelta(df_p["hour"].astype(int) - 1, unit="h")
    )
    df_p["pressure_hpa"] = df_p["P"] / 10.0
    df_p = df_p.set_index("datetime").sort_index()
    df_p["pressure_hpa"] = df_p["pressure_hpa"].ffill().bfill()
    df_p["pressure_trend_3h"] = df_p["pressure_hpa"].diff(3).fillna(0)
    df_p["pressure_trend_12h"] = df_p["pressure_hpa"].diff(12).fillna(0)

    cols = ["pressure_hpa", "pressure_trend_3h", "pressure_trend_12h"]
    train_out = merge_hourly(train_ts, df_p, cols)
    test_out = merge_hourly(test_ts, df_p, cols)
    train_out.to_csv(ROOT / "data" / "train_pressure.csv", index=False)
    test_out.to_csv(ROOT / "data" / "test_pressure.csv", index=False)
    print(f"  OK: {len(df_p)} hourly records, pressure [{df_p['pressure_hpa'].min():.1f}, {df_p['pressure_hpa'].max():.1f}] hPa")
    results_summary["Pressure"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Pressure"] = f"FAILED: {e}"


# ======================================================================
# 2. INSECT ACTIVITY INDEX (Hallmann et al. 2017)
# ======================================================================
# Monthly insect biomass index for NW Europe. Clutter includes insects.
# Source: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0185809
# ======================================================================
print("\n" + "=" * 60)
print("2. INSECT ACTIVITY INDEX")
print("=" * 60)
try:
    INSECT_INDEX = {
        1: 0.00, 2: 0.01, 3: 0.05, 4: 0.15, 5: 0.60, 6: 0.85,
        7: 1.00, 8: 0.90, 9: 0.50, 10: 0.15, 11: 0.03, 12: 0.00,
    }
    for split, ts, name in [("train", train_ts, "train_insect.csv"), ("test", test_ts, "test_insect.csv")]:
        months = ts.dt.month
        out = pd.DataFrame({"insect_activity_index": months.map(INSECT_INDEX), "month": months.values})
        out.to_csv(ROOT / "data" / name, index=False)
    print("  OK")
    results_summary["Insect"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    results_summary["Insect"] = f"FAILED: {e}"


# ======================================================================
# 3. TIDAL DATA (Rijkswaterstaat, Delfzijl)
# ======================================================================
# Waders fly to roost ~3h before high tide (class_deep_research.md).
# Tidal phase is gravitational = perfectly month-invariant.
# ddlpy: pip install git+https://github.com/Deltares/ddlpy.git
# ======================================================================
print("\n" + "=" * 60)
print("3. TIDAL DATA (Delfzijl)")
print("=" * 60)
try:
    import ddlpy
    from scipy.signal import find_peaks

    locations = ddlpy.locations()
    delf = locations[
        (locations.index == "delfzijl") & (locations["Grootheid.Code"] == "WATHTE")
    ]
    loc = delf.iloc[0]  # Must be Series, not DataFrame
    print(f"  Station: Delfzijl, Lat={loc['Lat']}, Lon={loc['Lon']}")

    meas = ddlpy.measurements(loc, start_date=pd.Timestamp("2023-08-15"), end_date=pd.Timestamp("2024-05-15"))
    wl = pd.to_numeric(meas["Meetwaarde.Waarde_Numeriek"], errors="coerce").dropna()
    if wl.index.tz is not None:
        wl.index = wl.index.tz_convert("UTC").tz_localize(None)
    print(f"  Got {len(wl)} measurements, range [{wl.min():.0f}, {wl.max():.0f}] cm NAP")

    # Regular 10-min grid + smooth for peak detection
    wl_regular = wl.resample("10min").interpolate(method="linear")
    wl_smooth = wl_regular.rolling(window=13, center=True, min_periods=1).mean()

    # Detect high tides: min 10h apart, min 30cm prominence
    peak_idx, _ = find_peaks(wl_smooth.values, distance=60, prominence=30)
    high_tides = wl_smooth.index[peak_idx]
    periods_h = np.diff(high_tides).astype("timedelta64[m]").astype(float) / 60
    print(f"  High tides: {len(high_tides)}, mean period: {np.mean(periods_h):.2f}h")

    # Save raw water level
    wl_regular.to_csv(ROOT / "data" / "delfzijl_water_level.csv")

    TIDAL_PERIOD = 12.42  # M2 constituent

    def compute_tidal(timestamps):
        ht_arr = np.array(high_tides.values, dtype="datetime64[ns]")
        results = []
        for t in timestamps:
            diffs_h = (np.datetime64(t) - ht_arr) / np.timedelta64(1, "h")
            past = diffs_h[diffs_h >= 0]
            hours_since = float(past.min()) if len(past) > 0 else float(TIDAL_PERIOD + diffs_h.max())
            hours_since = hours_since % TIDAL_PERIOD
            phase = hours_since / TIDAL_PERIOD
            try:
                idx = wl_regular.index.get_indexer([np.datetime64(t)], method="nearest")[0]
                wl_val = float(wl_regular.iloc[idx])
            except Exception:
                wl_val = 0.0
            results.append({
                "hours_since_high_tide": hours_since,
                "tidal_phase": phase,
                "tide_rising": int(phase > 0.5),
                "water_level_cm": wl_val,
            })
        return pd.DataFrame(results)

    train_tidal = compute_tidal(train_ts)
    test_tidal = compute_tidal(test_ts)
    train_tidal.to_csv(ROOT / "data" / "train_tidal.csv", index=False)
    test_tidal.to_csv(ROOT / "data" / "test_tidal.csv", index=False)
    print(f"  OK: train {len(train_tidal)} rows, test {len(test_tidal)} rows")
    results_summary["Tidal"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Tidal"] = f"FAILED: {e}"


# ======================================================================
# 4. MOON & TWILIGHT (skyfield)
# ======================================================================
# Moon illumination for nocturnal migration detection.
# NOTE: All tracks in this dataset are daytime (sun_alt > 1°).
# Twilight features will be constant — keep moon_illumination + sun_altitude.
# ======================================================================
print("\n" + "=" * 60)
print("4. MOON & TWILIGHT (skyfield)")
print("=" * 60)
try:
    from skyfield.api import load as sf_load, wgs84
    from skyfield import almanac

    eph = sf_load("de421.bsp")
    ts_sf = sf_load.timescale()
    earth, moon_body, sun_body = eph["earth"], eph["moon"], eph["sun"]
    location = wgs84.latlon(RADAR_LAT, RADAR_LON)

    def compute_moon(timestamps):
        results = []
        for t in timestamps:
            t_utc = t.tz_localize("UTC") if t.tzinfo is None else t
            sf_t = ts_sf.from_datetime(t_utc)
            moon_illum = almanac.fraction_illuminated(eph, "moon", sf_t)
            observer = earth + location
            sun_alt = observer.at(sf_t).observe(sun_body).apparent().altaz()[0].degrees
            moon_alt = observer.at(sf_t).observe(moon_body).apparent().altaz()[0].degrees
            results.append({
                "moon_illumination": float(moon_illum),
                "moon_altitude_deg": float(moon_alt),
                "sun_altitude_deg": float(sun_alt),
                "is_day": int(sun_alt > 0),
                "is_civil_twilight": int(-6 < sun_alt <= 0),
                "is_nautical_twilight": int(-12 < sun_alt <= -6),
                "is_astronomical_night": int(sun_alt <= -18),
            })
            if len(results) % 500 == 0:
                print(f"    {len(results)}/{len(timestamps)}...")
        return pd.DataFrame(results)

    print("  Computing for train...")
    train_moon = compute_moon(train_ts)
    print("  Computing for test...")
    test_moon = compute_moon(test_ts)
    train_moon.to_csv(ROOT / "data" / "train_moon.csv", index=False)
    test_moon.to_csv(ROOT / "data" / "test_moon.csv", index=False)
    print(f"  OK (NOTE: all tracks daytime, twilight features constant)")
    results_summary["Moon"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Moon"] = f"FAILED: {e}"


# ======================================================================
# 5. CAPE (Open-Meteo)
# ======================================================================
# CAPE measures thermal instability. BoP depends on thermals for soaring.
# Needs month-normalization (higher in summer).
# ======================================================================
print("\n" + "=" * 60)
print("5. CAPE (Open-Meteo)")
print("=" * 60)
try:
    resp = requests.get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": RADAR_LAT, "longitude": RADAR_LON,
            "start_date": "2023-08-15", "end_date": "2024-05-15",
            "hourly": "cape,lifted_index,convective_inhibition",
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    cape_df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "cape_jkg": hourly.get("cape", [0] * len(hourly["time"])),
        "lifted_index": hourly.get("lifted_index", [0] * len(hourly["time"])),
        "cin": hourly.get("convective_inhibition", [0] * len(hourly["time"])),
    }).set_index("datetime").sort_index().fillna(0)

    # Month-normalize CAPE
    mm = cape_df.groupby(cape_df.index.month)["cape_jkg"]
    cape_df["cape_normalized"] = (cape_df["cape_jkg"] - cape_df.index.month.map(mm.mean())) / cape_df.index.month.map(mm.std().replace(0, 1))

    cols = ["cape_jkg", "cape_normalized", "lifted_index", "cin"]
    merge_hourly(train_ts, cape_df, cols).to_csv(ROOT / "data" / "train_cape.csv", index=False)
    merge_hourly(test_ts, cape_df, cols).to_csv(ROOT / "data" / "test_cape.csv", index=False)
    cape_df[cols].to_csv(ROOT / "data" / "cape_hourly.csv")
    print(f"  OK: {len(cape_df)} hourly CAPE values")
    results_summary["CAPE"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["CAPE"] = f"FAILED: {e}"


# ======================================================================
# 6. WATER PROXIMITY (OSM coastline)
# ======================================================================
# Ducks near water, Pigeons terrestrial. Static spatial.
# Uses OSM Overpass API for coastline + water body polygons.
# ======================================================================
print("\n" + "=" * 60)
print("6. WATER PROXIMITY (OSM)")
print("=" * 60)
try:
    overpass_query = """
    [out:json][timeout:30];
    (
      way["natural"="coastline"](53.35,6.60,53.55,6.95);
      way["natural"="water"](53.35,6.60,53.55,6.95);
      relation["natural"="water"](53.35,6.60,53.55,6.95);
    );
    out body;
    >;
    out skel qt;
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": overpass_query}, timeout=60)
    resp.raise_for_status()
    nodes = {}
    coast_lats, coast_lons = [], []
    for el in resp.json().get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
    for el in resp.json().get("elements", []):
        if el["type"] == "way" and "nodes" in el:
            for nid in el["nodes"]:
                if nid in nodes:
                    coast_lats.append(nodes[nid][0])
                    coast_lons.append(nodes[nid][1])
    coast_lats, coast_lons = np.array(coast_lats), np.array(coast_lons)
    print(f"  Got {len(coast_lats)} coastline points")

    def water_features(track_lats, track_lons):
        results = []
        for lat, lon in zip(track_lats, track_lons):
            d = haversine_vec(lat, lon, coast_lats, coast_lons)
            results.append({"dist_to_water_m": float(np.min(d)), "over_water": int(lat > 53.46 or np.min(d) < 100)})
        return pd.DataFrame(results)

    water_features(train_lats, train_lons).to_csv(ROOT / "data" / "train_water.csv", index=False)
    water_features(test_lats, test_lons).to_csv(ROOT / "data" / "test_water.csv", index=False)
    print("  OK")
    results_summary["Water"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Water"] = f"FAILED: {e}"


# ======================================================================
# 7. VISIBILITY (KNMI Eelde stn 280)
# ======================================================================
# Ducks fly in rain/poor visibility; Pigeons avoid it (class_deep_research).
# VV=visibility, N=cloud cover, M=fog, R=rain occurrence.
# ======================================================================
print("\n" + "=" * 60)
print("7. VISIBILITY (KNMI Eelde)")
print("=" * 60)
try:
    resp = requests.post(
        "https://daggegevens.knmi.nl/klimatologie/uurgegevens",
        data={"stns": "280", "vars": "VV:N:M:R", "start": "2023080101", "end": "2024053124", "fmt": "json"},
        timeout=60,
    )
    resp.raise_for_status()
    df_v = pd.DataFrame(resp.json())
    df_v["datetime"] = pd.to_datetime(df_v["date"]).dt.tz_localize(None) + pd.to_timedelta(df_v["hour"].astype(int) - 1, unit="h")
    df_v = df_v.set_index("datetime").sort_index()
    for col in ["VV", "N", "M", "R"]:
        df_v[col] = pd.to_numeric(df_v[col], errors="coerce")

    def vv_to_km(vv):
        if pd.isna(vv): return np.nan
        vv = int(vv)
        return vv * 0.1 if vv <= 49 else (vv - 50 + 5) if vv <= 79 else (vv - 80) * 5 + 30

    df_v["visibility_km"] = df_v["VV"].apply(vv_to_km)
    df_v["cloud_octants"] = df_v["N"].clip(0, 8)
    df_v["fog"] = df_v["M"].fillna(0).astype(int)
    df_v["rain_occurring"] = df_v["R"].fillna(0).astype(int)
    df_v = df_v.ffill().bfill()

    cols = ["visibility_km", "cloud_octants", "fog", "rain_occurring"]
    merge_hourly(train_ts, df_v, cols).to_csv(ROOT / "data" / "train_visibility.csv", index=False)
    merge_hourly(test_ts, df_v, cols).to_csv(ROOT / "data" / "test_visibility.csv", index=False)
    print("  OK")
    results_summary["Visibility"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Visibility"] = f"FAILED: {e}"


# ======================================================================
# 8. TURBINE DISTANCE (OSM wind generators)
# ======================================================================
# 135 wind turbines near Eemshaven from OSM.
# Spatial interaction with turbines may vary by species.
# ======================================================================
print("\n" + "=" * 60)
print("8. TURBINE DISTANCE (OSM)")
print("=" * 60)
try:
    import time
    time.sleep(2)  # Rate limit for Overpass API
    query = '[out:json][timeout:60];\nnode["generator:source"="wind"](53.35,6.6,53.55,6.95);\nout body;'
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=90)
    resp.raise_for_status()
    turbines = [
        {"lat": el["lat"], "lon": el["lon"], "name": el.get("tags", {}).get("name", ""),
         "operator": el.get("tags", {}).get("operator", "")}
        for el in resp.json().get("elements", []) if "lat" in el and "lon" in el
    ]
    turb_df = pd.DataFrame(turbines)
    turb_df.to_csv(ROOT / "data" / "eemshaven_turbines.csv", index=False)
    print(f"  Found {len(turb_df)} wind turbines")

    turb_lats, turb_lons = turb_df["lat"].values, turb_df["lon"].values

    def turbine_features(track_lats, track_lons):
        results = []
        for lat, lon in zip(track_lats, track_lons):
            d = haversine_vec(lat, lon, turb_lats, turb_lons)
            results.append({
                "dist_to_turbine_m": float(np.min(d)),
                "turbines_within_500m": int(np.sum(d < 500)),
                "turbines_within_1km": int(np.sum(d < 1000)),
                "turbines_within_2km": int(np.sum(d < 2000)),
            })
        return pd.DataFrame(results)

    turbine_features(train_lats, train_lons).to_csv(ROOT / "data" / "train_turbines.csv", index=False)
    turbine_features(test_lats, test_lons).to_csv(ROOT / "data" / "test_turbines.csv", index=False)
    print("  OK")
    results_summary["Turbines"] = "OK"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results_summary["Turbines"] = f"FAILED: {e}"


# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for source, status in results_summary.items():
    tag = "OK" if status == "OK" else "FAIL"
    print(f"  [{tag:4s}] {source}: {status}")

print(f"\nNew data files in {ROOT / 'data'}:")
for f in sorted((ROOT / "data").glob("*.csv")):
    if any(x in f.name for x in [
        "pressure", "insect", "tidal", "moon", "cape", "water",
        "visibility", "turbine", "delfzijl",
    ]):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
