"""External ornithology priors built from public datasets.

This module consolidates class-level priors from:
- AVONET (mass, wing length, hand-wing index)
- BirdWingData (wingspan, wing area)
- Col de la Croix 1988 radar tracks (speed anchors for a few groups)
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, ROOT


_GEESE_GENERA = ("Anser", "Branta", "Cygnus")
_GULL_FAMILIES = {"Laridae", "Stercorariidae", "Alcidae"}
_WADER_FAMILIES = {
    "Scolopacidae",
    "Charadriidae",
    "Haematopodidae",
    "Recurvirostridae",
    "Burhinidae",
    "Rostratulidae",
    "Jacanidae",
}
_RAPTOR_ORDERS = {"Accipitriformes", "Falconiformes", "Strigiformes"}

# Priors from radar ornithology literature and previous internal ablations.
_SPEED_PRIOR_MS = {
    "Birds of Prey": 11.8,
    "Clutter": 2.0,
    "Cormorants": 16.0,
    "Ducks": 18.5,
    "Geese": 19.5,
    "Gulls": 13.5,
    "Pigeons": 16.5,
    "Songbirds": 11.5,
    "Waders": 17.0,
}
_WINGBEAT_PRIOR_HZ = {
    "Birds of Prey": 3.0,
    "Clutter": 0.5,
    "Cormorants": 3.8,
    "Ducks": 6.0,
    "Geese": 3.0,
    "Gulls": 4.2,
    "Pigeons": 6.8,
    "Songbirds": 11.5,
    "Waders": 8.5,
}


def _safe_mean(values: pd.Series, default: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().values
    if len(arr) == 0:
        return float(default)
    return float(np.mean(arr))


def _mass_to_size_bin(mass_g: float) -> float:
    """Map mass to radar_bird_size-style bin (0..3)."""
    if mass_g < 80:
        return 0.0
    if mass_g < 400:
        return 1.0
    if mass_g < 1200:
        return 2.0
    return 3.0


def _avonet_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    order = df["Order1"].fillna("")
    family = df["Family1"].fillna("")
    species = df["Species1"].fillna("")
    geese_mask = species.str.startswith(_GEESE_GENERA, na=False)

    masks = {
        "Birds of Prey": order.isin(_RAPTOR_ORDERS),
        "Clutter": pd.Series(False, index=df.index),
        "Cormorants": family.eq("Phalacrocoracidae"),
        "Ducks": order.eq("Anseriformes") & ~geese_mask,
        "Geese": geese_mask,
        "Gulls": family.isin(_GULL_FAMILIES),
        "Pigeons": order.eq("Columbiformes"),
        "Songbirds": order.eq("Passeriformes"),
        "Waders": family.isin(_WADER_FAMILIES),
    }

    # Fallback for sparse masks.
    if masks["Ducks"].sum() < 20:
        masks["Ducks"] = family.eq("Anatidae") & ~geese_mask
    if masks["Waders"].sum() < 20:
        masks["Waders"] = order.eq("Charadriiformes") & ~family.isin(_GULL_FAMILIES)
    if masks["Cormorants"].sum() < 8:
        masks["Cormorants"] = order.eq("Suliformes")

    return masks


def _birdwing_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    order = df["Order_IOC13.1"].fillna("")
    family = df["Family_IOC13.1"].fillna("")
    species = df["Species_IOC13.1"].fillna("")
    geese_mask = species.str.startswith(_GEESE_GENERA, na=False)

    masks = {
        "Birds of Prey": order.isin({x.upper() for x in _RAPTOR_ORDERS}),
        "Clutter": pd.Series(False, index=df.index),
        "Cormorants": family.eq("Phalacrocoracidae"),
        "Ducks": order.eq("ANSERIFORMES") & ~geese_mask,
        "Geese": geese_mask,
        "Gulls": family.isin(_GULL_FAMILIES),
        "Pigeons": order.eq("COLUMBIFORMES"),
        "Songbirds": order.eq("PASSERIFORMES"),
        "Waders": family.isin(_WADER_FAMILIES),
    }

    if masks["Ducks"].sum() < 20:
        masks["Ducks"] = family.eq("Anatidae") & ~geese_mask
    if masks["Waders"].sum() < 20:
        masks["Waders"] = order.eq("CHARADRIIFORMES") & ~family.isin(_GULL_FAMILIES)
    if masks["Cormorants"].sum() < 8:
        masks["Cormorants"] = order.eq("SULIFORMES")

    return masks


def _load_col_de_la_croix_speed_priors(path: Path) -> dict[str, float]:
    """Extract class-level speed anchors (m/s) from Col de la Croix data."""
    if not path.exists():
        return {}

    lines = path.read_text(encoding="latin1").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Code;Date;FNr;"):
            header_idx = i
            break
    if header_idx is None:
        return {}

    table_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(table_text), sep=";")
    df["FieldClass"] = pd.to_numeric(df["FieldClass"], errors="coerce")
    df["Va"] = pd.to_numeric(df["Va"], errors="coerce")
    df = df.dropna(subset=["FieldClass", "Va"]).copy()
    if df.empty:
        return {}

    field_to_group = {
        1: "Waders",
        2: "Waders",
        3: "Songbirds",
        4: "Songbirds",
        5: "Songbirds",
        6: "Birds of Prey",
    }
    df["group"] = df["FieldClass"].astype(int).map(field_to_group)
    df = df.dropna(subset=["group"]).copy()
    if df.empty:
        return {}

    df["speed_ms"] = df["Va"] / 100.0  # Va is in cm/s.
    out = {}
    for group in ["Birds of Prey", "Songbirds", "Waders"]:
        g = df.loc[df["group"] == group, "speed_ms"]
        if len(g) > 0:
            out[group] = float(g.median())
    return out


def build_external_class_priors(
    data_root: Path | None = None,
    col_speed_blend: float = 0.35,
) -> dict[str, dict[str, float]]:
    """Build class-level priors from AVONET + BirdWingData + Col de la Croix."""
    if data_root is None:
        data_root = ROOT

    avonet_path = data_root / "data" / "AVONET.xlsx"
    birdwing_path = (
        data_root
        / "data"
        / "other_datasets"
        / "figshare_23537892_birdwingdata"
        / "BirdWingData_tidy_ver2.1.csv"
    )
    col_path = data_root / "data" / "other_datasets" / "Col de la Croix 1988.csv"

    avonet = pd.read_excel(avonet_path, sheet_name="AVONET1_BirdLife")
    birdwing = pd.read_csv(birdwing_path)

    av_masks = _avonet_masks(avonet)
    bw_masks = _birdwing_masks(birdwing)
    col_speed = _load_col_de_la_croix_speed_priors(col_path)

    defaults = {
        "mass_g": _safe_mean(avonet["Mass"], 250.0),
        "wing_mm": _safe_mean(avonet["Wing.Length"], 170.0),
        "hwi": _safe_mean(avonet["Hand-Wing.Index"], 35.0),
        "wingspan_m": _safe_mean(birdwing["wingspan_m"], 0.7),
        "wing_area_m2": _safe_mean(birdwing["wing.area_m2"], 0.07),
    }

    priors = {}
    for cls in CLASSES:
        if cls == "Clutter":
            mass_g = 30.0
            wing_mm = 90.0
            hwi = 15.0
            span = 0.25
            area = 0.015
            n_av, n_bw = 0, 0
        else:
            av_m = av_masks[cls]
            bw_m = bw_masks[cls]
            av_cls = avonet.loc[av_m]
            bw_cls = birdwing.loc[bw_m]

            mass_g = _safe_mean(av_cls["Mass"], defaults["mass_g"])
            wing_mm = _safe_mean(av_cls["Wing.Length"], defaults["wing_mm"])
            hwi = _safe_mean(av_cls["Hand-Wing.Index"], defaults["hwi"])
            span = _safe_mean(bw_cls["wingspan_m"], defaults["wingspan_m"])
            area = _safe_mean(bw_cls["wing.area_m2"], defaults["wing_area_m2"])
            n_av = int(av_m.sum())
            n_bw = int(bw_m.sum())

        speed_ms = float(_SPEED_PRIOR_MS[cls])
        if cls in col_speed:
            speed_ms = (1.0 - col_speed_blend) * speed_ms + col_speed_blend * col_speed[cls]

        wingbeat_hz = float(_WINGBEAT_PRIOR_HZ[cls])
        wing_loading = ((mass_g / 1000.0) * 9.81) / max(area, 1e-4)
        aspect_ratio = (span * span) / max(area, 1e-4)
        size_bin = _mass_to_size_bin(mass_g)

        if cls == "Clutter":
            expected_rcs_db = -13.8
        else:
            expected_rcs_db = -30.0 + 10.0 * np.log10(max(mass_g, 5.0) / 100.0 + 0.01)

        priors[cls] = {
            "mass_g": float(mass_g),
            "wing_mm": float(wing_mm),
            "hwi": float(hwi),
            "wingspan_m": float(span),
            "wing_area_m2": float(area),
            "wing_loading": float(wing_loading),
            "aspect_ratio": float(aspect_ratio),
            "size_bin": float(size_bin),
            "speed_ms": float(speed_ms),
            "wingbeat_hz": float(wingbeat_hz),
            "expected_rcs_db": float(expected_rcs_db),
            "n_avonet": int(n_av),
            "n_birdwing": int(n_bw),
        }

    return priors


def priors_to_frame(priors: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert priors dict to a compact DataFrame for logging."""
    rows = []
    for cls in CLASSES:
        p = priors[cls]
        rows.append(
            {
                "class": cls,
                "mass_g": p["mass_g"],
                "wing_mm": p["wing_mm"],
                "hwi": p["hwi"],
                "span_m": p["wingspan_m"],
                "area_m2": p["wing_area_m2"],
                "wing_loading": p["wing_loading"],
                "speed_ms": p["speed_ms"],
                "wingbeat_hz": p["wingbeat_hz"],
                "expected_rcs_db": p["expected_rcs_db"],
                "n_avonet": p["n_avonet"],
                "n_birdwing": p["n_birdwing"],
            }
        )
    return pd.DataFrame(rows)
