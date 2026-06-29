"""Compare point-based and polygon-sampled SPEI12 exposure metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
POINT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_point_drought_metrics.csv"
POLY_PATH = PROJECT_DIR / "processed" / "country_year_spei12_polygon_sample_drought_metrics.csv"
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "phase1g_point_vs_polygon_spei_comparison.csv"


def main() -> int:
    point = pd.read_csv(POINT_PATH, dtype={"area_code_m49": str})
    poly = pd.read_csv(POLY_PATH, dtype={"area_code_m49": str})
    main = pd.read_csv(MAIN_PATH, dtype={"area_code_m49": str})[["area_code_m49", "area", "year"]].drop_duplicates()
    merged = main.merge(point, on=["area_code_m49", "area", "year"], how="left").merge(
        poly, on=["area_code_m49", "area", "year"], how="left", suffixes=("_point", "_poly")
    )
    merged["delta_mean"] = merged["spei12_poly_mean_annual"] - merged["spei12_annual_mean"]
    merged["delta_min"] = merged["spei12_poly_min_month"] - merged["spei12_min_month"]
    merged.to_csv(OUT_PATH, index=False)
    summary = merged[[
        "spei12_annual_mean",
        "spei12_poly_mean_annual",
        "spei12_min_month",
        "spei12_poly_min_month",
        "delta_mean",
        "delta_min",
    ]].corr(numeric_only=True)
    print(f"wrote {OUT_PATH} ({len(merged):,} rows)")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
