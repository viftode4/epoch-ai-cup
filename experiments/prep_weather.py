"""Prepare KNMI weather data + solar position features for E38."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).resolve().parent.parent

# ======================================================================
# 1. Parse KNMI station 277 (Lauwersoog) hourly data
# ======================================================================
print("Parsing KNMI weather data...", flush=True)

with open(ROOT / "data" / "knmi_raw.txt", "r") as f:
    text = f.read()

lines = text.split("\n")
header_line = None
for i, line in enumerate(lines):
    if line.startswith("#") and i + 1 < len(lines) and not lines[i + 1].startswith("#"):
        header_line = line
        break

data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
csv_text = header_line.lstrip("#") + "\n".join(data_lines)
df = pd.read_csv(StringIO(csv_text), skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# KNMI HH: 1=00:00-01:00, 24=23:00-24:00 (UTC)
df["datetime"] = pd.to_datetime(df["YYYYMMDD"].astype(int).astype(str), format="%Y%m%d")
df["datetime"] = df["datetime"] + pd.to_timedelta((df["HH"].astype(int) - 1), unit="h")

# Convert units
df["wind_speed"] = df["FH"] / 10.0
df["wind_gust"] = df["FX"] / 10.0
df["wind_dir"] = df["DD"].replace(0, np.nan).replace(990, np.nan)
df["temp_c"] = df["T"] / 10.0
df["dewpoint_c"] = df["TD"] / 10.0
df["sunshine_hrs"] = df["SQ"].fillna(0) / 10.0
df["radiation"] = df["Q"].fillna(0)
df["precip_dur"] = df["DR"].fillna(0) / 10.0
df["precip_mm"] = df["RH"].clip(lower=0).fillna(0) / 10.0
df["humidity"] = df["U"]

wind_rad = np.radians(df["wind_dir"].fillna(0))
df["wind_u"] = -df["wind_speed"] * np.sin(wind_rad)
df["wind_v"] = -df["wind_speed"] * np.cos(wind_rad)

weather_cols = ["datetime", "wind_speed", "wind_gust", "wind_u", "wind_v",
                "temp_c", "dewpoint_c", "sunshine_hrs", "radiation",
                "precip_dur", "precip_mm", "humidity"]
weather = df[weather_cols].copy()
weather = weather.set_index("datetime").sort_index()
weather = weather.fillna(method="ffill").fillna(0)

print(f"  Weather: {len(weather)} hours, {weather.index.min()} to {weather.index.max()}", flush=True)

# Monthly averages
monthly = weather.groupby(weather.index.month).mean()
print("\n  Monthly averages:", flush=True)
print(f"  {'Month':>5s} {'Wind':>6s} {'Gust':>6s} {'Temp':>6s} {'Hum':>5s} {'Prec':>5s}", flush=True)
for m in monthly.index:
    r = monthly.loc[m]
    print(f"  {m:5d} {r['wind_speed']:>6.1f} {r['wind_gust']:>6.1f} {r['temp_c']:>6.1f} {r['humidity']:>5.0f} {r['precip_mm']:>5.2f}", flush=True)

weather.to_csv(ROOT / "data" / "knmi_hourly_277.csv")
print("\n  Saved: data/knmi_hourly_277.csv", flush=True)

# ======================================================================
# 2. Compute solar position features
# ======================================================================
print("\nComputing solar position features...", flush=True)
from astral import Observer
from astral.sun import sun, elevation
import datetime
import zoneinfo

EEMSHAVEN_LAT = 53.44
EEMSHAVEN_LON = 6.83
observer = Observer(latitude=EEMSHAVEN_LAT, longitude=EEMSHAVEN_LON, elevation=0)
tz = zoneinfo.ZoneInfo("Europe/Amsterdam")

sys.path.insert(0, str(ROOT))
from src.data import load_train, load_test

train_df = load_train()
test_df = load_test()

def compute_solar_features(df):
    """Compute solar features for each sample's timestamp."""
    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    results = []
    for t in ts:
        t_aware = t.tz_localize("UTC")
        t_local = t_aware.astimezone(tz)
        date = t_local.date()
        try:
            s = sun(observer, date=date, tzinfo=tz)
            sunrise = s["sunrise"]
            sunset = s["sunset"]
            daylight_sec = (sunset - sunrise).total_seconds()
            since_sunrise = (t_local - sunrise).total_seconds()
            until_sunset = (sunset - t_local).total_seconds()
            daylight_frac = since_sunrise / max(daylight_sec, 1)
            daylight_frac = np.clip(daylight_frac, -0.5, 1.5)
            sun_elev = elevation(observer, t_aware)
        except Exception:
            daylight_sec = 43200
            since_sunrise = 21600
            until_sunset = 21600
            daylight_frac = 0.5
            sun_elev = 30

        results.append({
            "solar_elevation": sun_elev,
            "daylight_hours": daylight_sec / 3600,
            "hours_since_sunrise": since_sunrise / 3600,
            "daylight_fraction": daylight_frac,
        })
    return pd.DataFrame(results)

print("  Train solar features...", flush=True)
train_solar = compute_solar_features(train_df)
print("  Test solar features...", flush=True)
test_solar = compute_solar_features(test_df)

print(f"\n  Solar feature ranges:", flush=True)
for col in train_solar.columns:
    print(f"    {col}: train [{train_solar[col].min():.2f}, {train_solar[col].max():.2f}], "
          f"test [{test_solar[col].min():.2f}, {test_solar[col].max():.2f}]", flush=True)

train_solar.to_csv(ROOT / "data" / "train_solar.csv", index=False)
test_solar.to_csv(ROOT / "data" / "test_solar.csv", index=False)
print("  Saved: data/train_solar.csv, data/test_solar.csv", flush=True)

# ======================================================================
# 3. Merge weather with train/test timestamps
# ======================================================================
print("\nMerging weather with samples...", flush=True)

def merge_weather(df, weather):
    """Match each sample to nearest weather hour."""
    ts = pd.to_datetime(df["timestamp_start_radar_utc"])
    # Round to nearest hour
    ts_hour = ts.dt.floor("h")
    results = []
    for t in ts_hour:
        # Find closest weather row
        idx = weather.index.get_indexer([t], method="nearest")[0]
        results.append(weather.iloc[idx].to_dict())
    return pd.DataFrame(results)

train_weather = merge_weather(train_df, weather)
test_weather = merge_weather(test_df, weather)

print(f"  Train weather: {train_weather.shape}", flush=True)
print(f"  Test weather: {test_weather.shape}", flush=True)
print(f"\n  Weather feature ranges:", flush=True)
for col in train_weather.columns:
    print(f"    {col}: train [{train_weather[col].min():.2f}, {train_weather[col].max():.2f}], "
          f"test [{test_weather[col].min():.2f}, {test_weather[col].max():.2f}]", flush=True)

train_weather.to_csv(ROOT / "data" / "train_weather.csv", index=False)
test_weather.to_csv(ROOT / "data" / "test_weather.csv", index=False)
print("  Saved: data/train_weather.csv, data/test_weather.csv", flush=True)

print("\nDone! Ready for E38.", flush=True)
