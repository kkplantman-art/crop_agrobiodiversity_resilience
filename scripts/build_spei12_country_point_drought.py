"""Build a prototype country-year drought metric from SPEIbase point extraction.

This uses the SPEIbase v2.11 ncwebmapper chunked files directly:
- https://spei.csic.es/spei_database_2_11/nc/spei12-t.bin
- https://spei.csic.es/spei_database_2_11/nc/spei12-t.nc

For Phase 1D we extract the nearest 0.5-degree SPEI12 pixel at each country's
Natural Earth label point. This is a prototype drought proxy; final analyses
should use crop-area-weighted gridded climate metrics.
"""

from __future__ import annotations

import json
import math
import os
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw"
NE_PATH = RAW_DIR / "natural_earth_admin0_110m.geojson"
MAIN_SAMPLE_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_point_drought_metrics.csv"
POINTS_PATH = PROJECT_DIR / "processed" / "country_spei12_extraction_points.csv"

SPEI_BASE = "https://spei.csic.es/spei_database_2_11/nc"
SPEI_BIN_URL = f"{SPEI_BASE}/spei12-t.bin"
SPEI_NC_URL = f"{SPEI_BASE}/spei12-t.nc"

LON_MIN, LON_MAX, LON_NUM = -179.75, 179.75, 720
LAT_MIN, LAT_MAX, LAT_NUM = -89.75, 89.75, 360
CHUNK_DIR_RECORD_SIZE = 12  # uint64 offset + uint32 size
N_MONTHS = (2024 - 1901 + 1) * 12


def http_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "codex-agrobiodiversity-research/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def polygon_centroid(coords: list) -> tuple[float, float]:
    # Robust enough for Natural Earth fallback; label_x/label_y is preferred.
    points: list[tuple[float, float]] = []

    def walk(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)) and len(obj) >= 2:
            points.append((float(obj[0]), float(obj[1])))
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(coords)
    if not points:
        return (np.nan, np.nan)
    lon = float(np.mean([p[0] for p in points]))
    lat = float(np.mean([p[1] for p in points]))
    return lon, lat


def norm_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace("the ", "")
        .replace("republic of ", "")
        .replace("kingdom of ", "")
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .replace("'", "")
    )


def load_country_points() -> pd.DataFrame:
    with NE_PATH.open("r", encoding="utf-8") as handle:
        geo = json.load(handle)
    rows = []
    name_rows = []
    for feature in geo["features"]:
        props = feature["properties"]
        iso_n3 = str(props.get("ISO_N3", "")).strip()
        lon = props.get("LABEL_X")
        lat = props.get("LABEL_Y")
        if lon is None or lat is None:
            lon, lat = polygon_centroid(feature["geometry"]["coordinates"])
        base = {
            "natural_earth_name": props.get("NAME"),
            "natural_earth_admin": props.get("ADMIN"),
            "natural_earth_sovereignt": props.get("SOVEREIGNT"),
            "iso_a3": props.get("ISO_A3") if props.get("ISO_A3") != "-99" else props.get("ADM0_A3"),
            "lon": float(lon),
            "lat": float(lat),
        }
        if iso_n3 and iso_n3 != "-99":
            rows.append({"area_code_m49": "'" + iso_n3.zfill(3), **base})
        for key in ["NAME", "ADMIN", "SOVEREIGNT"]:
            value = props.get(key)
            if value:
                name_rows.append({"match_name": norm_name(value), **base})
    points = pd.DataFrame(rows).drop_duplicates("area_code_m49")
    names = pd.DataFrame(name_rows).drop_duplicates("match_name")
    sample_areas = pd.read_csv(MAIN_SAMPLE_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    points = sample_areas.merge(points, on="area_code_m49", how="left")
    missing_mask = points["lon"].isna() | points["lat"].isna()
    if missing_mask.any():
        fallback = points.loc[missing_mask, ["area_code_m49", "area"]].copy()
        fallback["match_name"] = fallback["area"].map(norm_name)
        fallback = fallback.merge(names, on="match_name", how="left")
        fill_cols = [
            "natural_earth_name",
            "natural_earth_admin",
            "natural_earth_sovereignt",
            "iso_a3",
            "lon",
            "lat",
        ]
        for col in fill_cols:
            points.loc[missing_mask, col] = fallback[col].to_numpy()
    return points


def locate_pixel(lon: float, lat: float) -> tuple[int, int, int]:
    lon_index = round((lon - LON_MIN) / ((LON_MAX - LON_MIN) / (LON_NUM - 1)))
    lat_index = round((lat - LAT_MIN) / ((LAT_MAX - LAT_MIN) / (LAT_NUM - 1)))
    if not (0 <= lon_index < LON_NUM and 0 <= lat_index < LAT_NUM):
        raise ValueError(f"point outside SPEI grid: lon={lon}, lat={lat}")
    chunk_index = lon_index + lat_index * LON_NUM
    return lon_index, lat_index, chunk_index


def extract_spei12_series(lon: float, lat: float) -> np.ndarray:
    _, _, chunk_index = locate_pixel(lon, lat)
    dir_start = chunk_index * CHUNK_DIR_RECORD_SIZE
    dir_bytes = http_range(SPEI_BIN_URL, dir_start, dir_start + CHUNK_DIR_RECORD_SIZE - 1)
    if len(dir_bytes) != CHUNK_DIR_RECORD_SIZE:
        raise RuntimeError(f"unexpected chunk directory byte length: {len(dir_bytes)}")
    offset, size = struct.unpack("<QI", dir_bytes)
    chunk = http_range(SPEI_NC_URL, int(offset), int(offset + size - 1))
    values = np.frombuffer(zlib.decompress(chunk), dtype="<f4")
    if values.size != N_MONTHS:
        raise RuntimeError(f"unexpected SPEI series length {values.size}, expected {N_MONTHS}")
    values = values.astype(float)
    values[np.isclose(values, 1.00000001504747e30)] = np.nan
    return values


def monthly_to_annual(area: str, area_code_m49: str, values: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("1901-01-01", periods=N_MONTHS, freq="MS")
    frame = pd.DataFrame({"date": dates, "spei12": values})
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    annual = (
        frame[frame["year"].between(1992, 2024)]
        .groupby("year", as_index=False)
        .agg(
            spei12_annual_mean=("spei12", "mean"),
            spei12_min_month=("spei12", "min"),
            spei12_december=("spei12", lambda s: s.iloc[-1] if len(s) else np.nan),
            n_spei_months=("spei12", lambda s: int(s.notna().sum())),
        )
    )
    annual["area_code_m49"] = area_code_m49
    annual["area"] = area
    annual["drought_spei12_mean_lt_minus1"] = annual["spei12_annual_mean"] < -1
    annual["drought_spei12_min_lt_minus1_5"] = annual["spei12_min_month"] < -1.5
    return annual[[
        "area_code_m49",
        "area",
        "year",
        "spei12_annual_mean",
        "spei12_min_month",
        "spei12_december",
        "n_spei_months",
        "drought_spei12_mean_lt_minus1",
        "drought_spei12_min_lt_minus1_5",
    ]]


def main() -> int:
    max_areas = os.environ.get("SPEI_MAX_AREAS")
    max_areas_int = int(max_areas) if max_areas else None

    points = load_country_points()
    missing = points[points["lon"].isna() | points["lat"].isna()].copy()
    points = points.dropna(subset=["lon", "lat"]).copy()
    if max_areas_int:
        points = points.head(max_areas_int)

    annual_rows = []
    point_rows = []
    for i, row in enumerate(points.itertuples(index=False), start=1):
        try:
            lon_index, lat_index, chunk_index = locate_pixel(row.lon, row.lat)
            values = extract_spei12_series(row.lon, row.lat)
            annual_rows.append(monthly_to_annual(row.area, row.area_code_m49, values))
            point_rows.append(
                {
                    "area_code_m49": row.area_code_m49,
                    "area": row.area,
                    "natural_earth_name": row.natural_earth_name,
                    "iso_a3": row.iso_a3,
                    "lon": row.lon,
                    "lat": row.lat,
                    "spei_lon_index": lon_index,
                    "spei_lat_index": lat_index,
                    "spei_chunk_index": chunk_index,
                }
            )
        except Exception as exc:
            point_rows.append(
                {
                    "area_code_m49": row.area_code_m49,
                    "area": row.area,
                    "natural_earth_name": getattr(row, "natural_earth_name", ""),
                    "iso_a3": getattr(row, "iso_a3", ""),
                    "lon": row.lon,
                    "lat": row.lat,
                    "error": str(exc),
                }
            )
        if i % 25 == 0:
            print(f"processed {i}/{len(points)} country points", flush=True)

    if not annual_rows:
        raise RuntimeError("No SPEI country point series extracted.")
    annual = pd.concat(annual_rows, ignore_index=True)
    point_df = pd.DataFrame(point_rows)
    if not missing.empty:
        missing = missing.assign(error="missing_natural_earth_point")
        point_df = pd.concat([point_df, missing], ignore_index=True, sort=False)

    annual.to_csv(OUT_PATH, index=False)
    point_df.to_csv(POINTS_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(annual):,} rows; {annual['area'].nunique():,} areas)")
    print(f"wrote {POINTS_PATH} ({len(point_df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
