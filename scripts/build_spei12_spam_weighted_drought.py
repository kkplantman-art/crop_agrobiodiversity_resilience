"""Build country-year SPEI12 weighted by SPAM 2020 harvested area."""

from __future__ import annotations

import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw"
CACHE_DIR = RAW_DIR / "spei12_pixel_cache"
WEIGHTS_PATH = PROJECT_DIR / "processed" / "spam2020_H_TA_spei12_pixel_weights.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_spam_harvested_area_weighted_metrics.csv"
QC_PATH = PROJECT_DIR / "tables" / "spam2020_spei12_weighted_exposure_qc.csv"

SPEI_BASE = "https://spei.csic.es/spei_database_2_11/nc"
SPEI_BIN_URL = f"{SPEI_BASE}/spei12-t.bin"
SPEI_NC_URL = f"{SPEI_BASE}/spei12-t.nc"
CHUNK_DIR_RECORD_SIZE = 12
N_CHUNKS = 720 * 360
N_MONTHS = (2024 - 1901 + 1) * 12
START_MONTH_INDEX = (1992 - 1901) * 12
END_MONTH_INDEX = (2024 - 1901 + 1) * 12


def http_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "codex-agrobiodiversity-research/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def load_chunk_directory() -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "spei12_chunk_directory.bin"
    expected = N_CHUNKS * CHUNK_DIR_RECORD_SIZE
    if path.exists() and path.stat().st_size == expected:
        return path.read_bytes()
    data = http_range(SPEI_BIN_URL, 0, expected - 1)
    if len(data) != expected:
        raise RuntimeError(f"unexpected chunk directory size: {len(data)}")
    path.write_bytes(data)
    return data


def parse_chunk_records(directory: bytes, chunk_ids: list[int]) -> pd.DataFrame:
    rows = []
    for chunk_id in chunk_ids:
        start = int(chunk_id) * CHUNK_DIR_RECORD_SIZE
        offset, size = struct.unpack("<QI", directory[start : start + CHUNK_DIR_RECORD_SIZE])
        rows.append({"spei_chunk_index": int(chunk_id), "offset": int(offset), "size": int(size)})
    return pd.DataFrame(rows).sort_values("offset").reset_index(drop=True)


def build_ranges(records: pd.DataFrame, gap_threshold: int = 262_144) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    end = None
    for row in records.itertuples(index=False):
        s = int(row.offset)
        e = int(row.offset + row.size - 1)
        if start is None:
            start, end = s, e
        elif s <= end + 1 + gap_threshold:
            end = max(end, e)
        else:
            ranges.append((start, end))
            start, end = s, e
    if start is not None and end is not None:
        ranges.append((start, end))
    return ranges


def load_series_for_chunks(records: pd.DataFrame) -> dict[int, np.ndarray]:
    ranges = build_ranges(records)
    print(f"loading {len(records):,} SPEI chunks in {len(ranges):,} merged byte ranges", flush=True)
    series: dict[int, np.ndarray] = {}
    record_iter = records.itertuples(index=False)
    current = next(record_iter, None)
    for i, (range_start, range_end) in enumerate(ranges, start=1):
        blob = http_range(SPEI_NC_URL, range_start, range_end)
        while current is not None and int(current.offset + current.size - 1) <= range_end:
            offset = int(current.offset)
            size = int(current.size)
            raw = blob[offset - range_start : offset - range_start + size]
            values = np.frombuffer(zlib.decompress(raw), dtype="<f4").astype(float)
            if values.size != N_MONTHS:
                raise RuntimeError(
                    f"unexpected SPEI series length for chunk {current.spei_chunk_index}: "
                    f"{values.size}"
                )
            values[np.isclose(values, 1.00000001504747e30)] = np.nan
            series[int(current.spei_chunk_index)] = values[START_MONTH_INDEX:END_MONTH_INDEX]
            current = next(record_iter, None)
            if current is None or int(current.offset) > range_end:
                break
        if i % 10 == 0 or i == len(ranges):
            print(f"loaded range {i}/{len(ranges)}; chunks decoded={len(series):,}", flush=True)
    if len(series) != len(records):
        raise RuntimeError(f"decoded {len(series)} chunks, expected {len(records)}")
    return series


def weighted_country_monthly(weights: pd.DataFrame, series: dict[int, np.ndarray]) -> list[pd.DataFrame]:
    dates = pd.date_range("1992-01-01", periods=END_MONTH_INDEX - START_MONTH_INDEX, freq="MS")
    rows = []
    for area_code, group in weights.groupby("area_code_m49", sort=False):
        numerator = np.zeros(len(dates), dtype=float)
        denominator = np.zeros(len(dates), dtype=float)
        for row in group.itertuples(index=False):
            values = series[int(row.spei_chunk_index)]
            valid = np.isfinite(values)
            weight = float(row.weight_share)
            numerator[valid] += weight * values[valid]
            denominator[valid] += weight
        mean = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        frame = pd.DataFrame(
            {
                "area_code_m49": area_code,
                "area": group["area"].iloc[0],
                "date": dates,
                "spei12_spam_weighted": mean,
                "spam_weight_coverage": denominator,
                "n_spam_spei_pixels": group["spei_chunk_index"].nunique(),
                "spam_harvested_area_total": group["harvested_area"].sum(),
            }
        )
        rows.append(frame)
    return rows


def monthly_to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly["year"] = monthly["date"].dt.year
    annual = (
        monthly.groupby(["area_code_m49", "area", "year"], as_index=False)
        .agg(
            spei12_spam_weighted_mean_annual=("spei12_spam_weighted", "mean"),
            spei12_spam_weighted_min_month=("spei12_spam_weighted", "min"),
            spei12_spam_weighted_december=(
                "spei12_spam_weighted",
                lambda s: s.iloc[-1] if len(s) else np.nan,
            ),
            n_spei_months=("spei12_spam_weighted", lambda s: int(s.notna().sum())),
            mean_spam_weight_coverage=("spam_weight_coverage", "mean"),
            n_spam_spei_pixels=("n_spam_spei_pixels", "max"),
            spam_harvested_area_total=("spam_harvested_area_total", "max"),
        )
    )
    annual["drought_spam_mean_lt_minus1"] = annual["spei12_spam_weighted_mean_annual"] < -1
    annual["drought_spam_min_lt_minus1_5"] = annual["spei12_spam_weighted_min_month"] < -1.5
    return annual


def main() -> int:
    weights = pd.read_csv(WEIGHTS_PATH, dtype={"area_code_m49": str})
    weights = weights[weights["weight_share"] > 0].copy()
    chunk_ids = sorted(weights["spei_chunk_index"].astype(int).unique())
    directory = load_chunk_directory()
    records = parse_chunk_records(directory, chunk_ids)
    series = load_series_for_chunks(records)
    monthly = pd.concat(weighted_country_monthly(weights, series), ignore_index=True)
    annual = monthly_to_annual(monthly)
    annual.to_csv(OUT_PATH, index=False)

    qc = (
        annual.groupby(["area_code_m49", "area"], as_index=False)
        .agg(
            n_years=("year", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_spam_drought_years=("drought_spam_mean_lt_minus1", "sum"),
            n_spam_spei_pixels=("n_spam_spei_pixels", "max"),
            spam_harvested_area_total=("spam_harvested_area_total", "max"),
            mean_spam_weight_coverage=("mean_spam_weight_coverage", "mean"),
        )
        .sort_values("area")
    )
    qc.to_csv(QC_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(annual):,} rows; {annual['area'].nunique():,} areas)")
    print(f"wrote {QC_PATH} ({len(qc):,} areas)")
    print(qc.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
