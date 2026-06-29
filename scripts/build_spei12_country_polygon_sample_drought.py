"""Build country-year SPEI12 from multiple Natural Earth polygon sample pixels.

This improves on the Phase 1D country-point prototype by averaging SPEI12 over
several 0.5-degree pixels sampled inside each country's Natural Earth polygon.

It is still a prototype: pixels are evenly sampled within country polygons, not
weighted by crop area. The next stronger version should use SPAM/GAEZ crop-area
weights or harvested-area masks.
"""

from __future__ import annotations

import json
import math
import os
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw"
NE_PATH = RAW_DIR / "natural_earth_admin0_110m.geojson"
MAIN_SAMPLE_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_polygon_sample_drought_metrics.csv"
POINTS_PATH = PROJECT_DIR / "processed" / "country_spei12_polygon_sample_points.csv"
CACHE_DIR = RAW_DIR / "spei12_pixel_cache"
BATCH_DIR = PROJECT_DIR / "processed" / "spei_polygon_batches"

SPEI_BASE = "https://spei.csic.es/spei_database_2_11/nc"
SPEI_BIN_URL = f"{SPEI_BASE}/spei12-t.bin"
SPEI_NC_URL = f"{SPEI_BASE}/spei12-t.nc"

LON_MIN, LON_MAX, LON_NUM = -179.75, 179.75, 720
LAT_MIN, LAT_MAX, LAT_NUM = -89.75, 89.75, 360
DX = (LON_MAX - LON_MIN) / (LON_NUM - 1)
DY = (LAT_MAX - LAT_MIN) / (LAT_NUM - 1)
CHUNK_DIR_RECORD_SIZE = 12
N_MONTHS = (2024 - 1901 + 1) * 12

MANUAL_NAME_TO_M49 = {
    "france": "'250",
    "norway": "'578",
}


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


def locate_pixel(lon: float, lat: float) -> tuple[int, int, int, float, float]:
    lon_index = round((lon - LON_MIN) / DX)
    lat_index = round((lat - LAT_MIN) / DY)
    lon_index = max(0, min(LON_NUM - 1, lon_index))
    lat_index = max(0, min(LAT_NUM - 1, lat_index))
    chunk_index = lon_index + lat_index * LON_NUM
    pixel_lon = lon_index * DX + LON_MIN
    pixel_lat = lat_index * DY + LAT_MIN
    return lon_index, lat_index, chunk_index, pixel_lon, pixel_lat


def extract_pixel_series(chunk_index: int) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"spei12_chunk_{chunk_index}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    dir_start = chunk_index * CHUNK_DIR_RECORD_SIZE
    dir_bytes = http_range(SPEI_BIN_URL, dir_start, dir_start + CHUNK_DIR_RECORD_SIZE - 1)
    offset, size = struct.unpack("<QI", dir_bytes)
    chunk = http_range(SPEI_NC_URL, int(offset), int(offset + size - 1))
    values = np.frombuffer(zlib.decompress(chunk), dtype="<f4").astype(float)
    values[np.isclose(values, 1.00000001504747e30)] = np.nan
    if values.size != N_MONTHS:
        raise RuntimeError(f"unexpected series length {values.size}")
    np.save(cache_path, values)
    return values


def iter_rings(geometry: dict):
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        for ring in coords[:1]:
            yield ring
    elif geometry["type"] == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def feature_area_code(props: dict) -> str | None:
    iso_n3 = str(props.get("ISO_N3", "")).strip()
    if iso_n3 and iso_n3 != "-99":
        return "'" + iso_n3.zfill(3)
    for key in ["NAME", "ADMIN", "SOVEREIGNT"]:
        normalized = norm_name(props.get(key, ""))
        if normalized in MANUAL_NAME_TO_M49:
            return MANUAL_NAME_TO_M49[normalized]
    return None


def load_features_by_area() -> dict[str, dict]:
    with NE_PATH.open("r", encoding="utf-8") as handle:
        geo = json.load(handle)
    features: dict[str, dict] = {}
    for feature in geo["features"]:
        code = feature_area_code(feature["properties"])
        if code and code not in features:
            features[code] = feature
    return features


def sample_feature_pixels(feature: dict, target_points: int = 12) -> list[dict]:
    paths = []
    bounds = []
    for ring in iter_rings(feature["geometry"]):
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        paths.append(MplPath(arr[:, :2]))
        bounds.append((arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()))
    if not paths:
        return []

    min_lon = max(-180, min(b[0] for b in bounds))
    min_lat = max(-90, min(b[1] for b in bounds))
    max_lon = min(180, max(b[2] for b in bounds))
    max_lat = min(90, max(b[3] for b in bounds))

    # Adaptive grid: large countries get broad coverage, small countries get at
    # least a dense enough grid to find a land pixel.
    grid_n = max(5, min(35, int(math.sqrt(target_points * 20))))
    lons = np.linspace(min_lon, max_lon, grid_n)
    lats = np.linspace(min_lat, max_lat, grid_n)
    candidates = []
    for lon in lons:
        for lat in lats:
            if any(path.contains_point((lon, lat)) for path in paths):
                lon_i, lat_i, chunk_i, px_lon, px_lat = locate_pixel(lon, lat)
                candidates.append((lon_i, lat_i, chunk_i, px_lon, px_lat))

    # Fallback to label point if a small island is missed by the grid.
    if not candidates:
        lon = feature["properties"].get("LABEL_X")
        lat = feature["properties"].get("LABEL_Y")
        if lon is not None and lat is not None:
            candidates.append(locate_pixel(float(lon), float(lat)))

    # Deduplicate and select spread-out candidates.
    seen = set()
    unique = []
    for item in candidates:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    if len(unique) <= target_points:
        selected = unique
    else:
        idx = np.linspace(0, len(unique) - 1, target_points).round().astype(int)
        selected = [unique[i] for i in idx]

    return [
        {
            "spei_lon_index": lon_i,
            "spei_lat_index": lat_i,
            "spei_chunk_index": chunk_i,
            "spei_pixel_lon": px_lon,
            "spei_pixel_lat": px_lat,
        }
        for lon_i, lat_i, chunk_i, px_lon, px_lat in selected
    ]


def monthly_to_annual(area: str, area_code_m49: str, values: np.ndarray, n_pixels: int) -> pd.DataFrame:
    dates = pd.date_range("1901-01-01", periods=N_MONTHS, freq="MS")
    frame = pd.DataFrame({"date": dates, "spei12": values})
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    annual = (
        frame[frame["year"].between(1992, 2024)]
        .groupby("year", as_index=False)
        .agg(
            spei12_poly_mean_annual=("spei12", "mean"),
            spei12_poly_min_month=("spei12", "min"),
            spei12_poly_december=("spei12", lambda s: s.iloc[-1] if len(s) else np.nan),
            n_spei_months=("spei12", lambda s: int(s.notna().sum())),
        )
    )
    annual["area_code_m49"] = area_code_m49
    annual["area"] = area
    annual["n_spei_pixels"] = n_pixels
    annual["drought_poly_mean_lt_minus1"] = annual["spei12_poly_mean_annual"] < -1
    annual["drought_poly_min_lt_minus1_5"] = annual["spei12_poly_min_month"] < -1.5
    return annual[[
        "area_code_m49",
        "area",
        "year",
        "spei12_poly_mean_annual",
        "spei12_poly_min_month",
        "spei12_poly_december",
        "n_spei_months",
        "n_spei_pixels",
        "drought_poly_mean_lt_minus1",
        "drought_poly_min_lt_minus1_5",
    ]]


def main() -> int:
    max_areas = os.environ.get("SPEI_POLY_MAX_AREAS")
    max_areas_int = int(max_areas) if max_areas else None
    offset = int(os.environ.get("SPEI_POLY_AREA_OFFSET", "0"))
    limit_env = os.environ.get("SPEI_POLY_AREA_LIMIT")
    limit = int(limit_env) if limit_env else None
    batch_tag = os.environ.get("SPEI_POLY_BATCH_TAG")
    target_pixels = int(os.environ.get("SPEI_POLY_TARGET_PIXELS", "12"))

    sample_areas = pd.read_csv(MAIN_SAMPLE_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    if offset or limit:
        sample_areas = sample_areas.iloc[offset : offset + limit if limit else None].copy()
    if max_areas_int:
        sample_areas = sample_areas.head(max_areas_int)
    features = load_features_by_area()

    annual_rows = []
    point_rows = []
    for i, row in enumerate(sample_areas.itertuples(index=False), start=1):
        feature = features.get(row.area_code_m49)
        if feature is None:
            point_rows.append({"area_code_m49": row.area_code_m49, "area": row.area, "error": "missing_natural_earth_feature"})
            continue
        pixels = sample_feature_pixels(feature, target_points=target_pixels)
        series = []
        for pixel in pixels:
            try:
                vals = extract_pixel_series(int(pixel["spei_chunk_index"]))
                if np.isfinite(vals).sum() > 0:
                    series.append(vals)
                    point_rows.append({"area_code_m49": row.area_code_m49, "area": row.area, **pixel})
            except Exception as exc:
                point_rows.append({"area_code_m49": row.area_code_m49, "area": row.area, **pixel, "error": str(exc)})
        if series:
            stack = np.vstack(series)
            finite_counts = np.isfinite(stack).sum(axis=0)
            sum_values = np.nansum(stack, axis=0)
            mean_values = np.full_like(sum_values, np.nan, dtype=float)
            np.divide(sum_values, finite_counts, out=mean_values, where=finite_counts > 0)
            if np.isfinite(mean_values).sum() > 0:
                annual_rows.append(monthly_to_annual(row.area, row.area_code_m49, mean_values, len(series)))
            else:
                point_rows.append({"area_code_m49": row.area_code_m49, "area": row.area, "error": "all_nan_sampled_pixels"})
        else:
            point_rows.append({"area_code_m49": row.area_code_m49, "area": row.area, "error": "no_valid_spei_pixels"})
        if i % 25 == 0:
            print(f"processed {i}/{len(sample_areas)} countries", flush=True)

    if not annual_rows:
        raise RuntimeError("No polygon-sampled SPEI series extracted.")
    annual = pd.concat(annual_rows, ignore_index=True)
    points = pd.DataFrame(point_rows)
    if batch_tag:
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        out_path = BATCH_DIR / f"country_year_spei12_polygon_sample_drought_metrics_{batch_tag}.csv"
        points_path = BATCH_DIR / f"country_spei12_polygon_sample_points_{batch_tag}.csv"
    else:
        out_path = OUT_PATH
        points_path = POINTS_PATH
    annual.to_csv(out_path, index=False)
    points.to_csv(points_path, index=False)
    print(f"wrote {out_path} ({len(annual):,} rows; {annual['area'].nunique():,} areas)")
    print(f"wrote {points_path} ({len(points):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
