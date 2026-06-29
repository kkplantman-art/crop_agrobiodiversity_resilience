"""Build crop-response asynchrony mechanism metrics.

For each area-year, this script measures whether crop-level production
anomalies move together or compensate for one another. A higher asynchrony
index indicates that aggregate production variance is lower than the
area-weighted variance of constituent crop anomalies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CROP_PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
MAIN_AREAS_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_crop_response_asynchrony_metrics.csv"
SUMMARY_PATH = PROJECT_DIR / "tables" / "crop_response_asynchrony_coverage_summary.csv"


def add_crop_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["area", "item"], sort=False):
        group = group.sort_values("year").copy()
        valid = group["log_crop_production"].notna()
        if valid.sum() >= 10:
            x = group.loc[valid, "year"].to_numpy(dtype=float)
            y = group.loc[valid, "log_crop_production"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, deg=1)
            group["crop_log_trend"] = intercept + slope * group["year"].to_numpy(dtype=float)
            group["crop_log_anomaly"] = group["log_crop_production"] - group["crop_log_trend"]
        else:
            group["crop_log_trend"] = np.nan
            group["crop_log_anomaly"] = np.nan
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def build_asynchrony(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (area_code, area, year), group in df.groupby(["area_code_m49", "area", "year"], sort=False):
        g = group.dropna(subset=["crop_log_anomaly", "area_share"]).copy()
        g = g[g["area_share"] > 0]
        if len(g) < 3:
            continue
        weights = g["area_share"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        crop_anom = g["crop_log_anomaly"].to_numpy(dtype=float)
        weighted_mean = float(np.sum(weights * crop_anom))
        weighted_abs = float(np.sum(weights * np.abs(crop_anom)))
        weighted_var = float(np.sum(weights * (crop_anom - weighted_mean) ** 2))
        synchrony_ratio = abs(weighted_mean) / weighted_abs if weighted_abs > 0 else np.nan
        rows.append(
            {
                "area_code_m49": area_code,
                "area": area,
                "year": year,
                "n_crops_asynchrony": len(g),
                "weighted_crop_anomaly_mean": weighted_mean,
                "weighted_crop_anomaly_abs_mean": weighted_abs,
                "weighted_crop_anomaly_variance": weighted_var,
                "crop_response_synchrony_ratio": synchrony_ratio,
                "crop_response_asynchrony_index": 1 - synchrony_ratio if np.isfinite(synchrony_ratio) else np.nan,
                "share_positive_crop_anomalies": float(np.sum(weights * (crop_anom > 0))),
                "share_negative_crop_anomalies": float(np.sum(weights * (crop_anom < 0))),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    panel = pd.read_csv(CROP_PANEL_PATH, dtype={"area_code_m49": str})
    main_areas = pd.read_csv(MAIN_AREAS_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    panel = panel.merge(main_areas, on=["area_code_m49", "area"], how="inner")
    panel = panel[panel["year"] >= 1992].copy()
    panel["production"] = pd.to_numeric(panel["production"], errors="coerce")
    panel["area_harvested"] = pd.to_numeric(panel["area_harvested"], errors="coerce")
    panel = panel[panel["production"].notna() & (panel["production"] > 0)].copy()

    totals = (
        panel.groupby(["area_code_m49", "area", "year"], as_index=False)["area_harvested"]
        .sum()
        .rename(columns={"area_harvested": "total_harvested_area"})
    )
    panel = panel.merge(totals, on=["area_code_m49", "area", "year"], how="left")
    panel["area_share"] = panel["area_harvested"] / panel["total_harvested_area"]
    panel["log_crop_production"] = np.log1p(panel["production"])

    panel = add_crop_anomalies(panel)
    asynchrony = build_asynchrony(panel)
    summary = (
        asynchrony.groupby("area", as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
            mean_n_crops_asynchrony=("n_crops_asynchrony", "mean"),
            mean_asynchrony=("crop_response_asynchrony_index", "mean"),
        )
        .sort_values(["n_years", "mean_n_crops_asynchrony"], ascending=False)
    )
    asynchrony.to_csv(OUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(asynchrony):,} rows; {asynchrony['area'].nunique():,} areas)")
    print(f"wrote {SUMMARY_PATH} ({len(summary):,} rows)")
    print(asynchrony.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
