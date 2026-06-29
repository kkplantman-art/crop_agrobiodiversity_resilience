"""Merge Phase 1D main panel with SPEI12 point drought metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
DROUGHT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_point_drought_metrics.csv"
OUT_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
SUMMARY_PATH = PROJECT_DIR / "tables" / "phase1d_drought_merge_summary.csv"


def main() -> int:
    main = pd.read_csv(MAIN_PATH, dtype={"area_code_m49": str})
    drought = pd.read_csv(DROUGHT_PATH, dtype={"area_code_m49": str})
    merged = main.merge(drought, on=["area_code_m49", "area", "year"], how="left", validate="one_to_one")
    merged["has_spei12"] = merged["spei12_annual_mean"].notna()
    merged.to_csv(OUT_PATH, index=False)

    summary = (
        merged.groupby("area", as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
            n_spei_years=("has_spei12", "sum"),
            n_drought_mean_lt_minus1=("drought_spei12_mean_lt_minus1", "sum"),
            n_drought_min_lt_minus1_5=("drought_spei12_min_lt_minus1_5", "sum"),
        )
        .sort_values(["n_spei_years", "n_years"], ascending=False)
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(merged):,} rows; {merged['area'].nunique():,} areas)")
    print(f"wrote {SUMMARY_PATH} ({len(summary):,} areas)")
    print(merged[['area','year','shannon_crop_area','production_log_anomaly','spei12_annual_mean','drought_spei12_mean_lt_minus1']].dropna().head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
