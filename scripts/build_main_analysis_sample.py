"""Create a main analysis sample for post-1992 country-year models.

The FAOSTAT country-level panel contains historical entities that have valid
records but complicate fixed-effects and climate-merge analyses. This script
keeps all raw processed data intact and writes a separate main-sample panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "processed" / "phase1c_diversity_stability_model_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
EXCLUSION_PATH = PROJECT_DIR / "tables" / "main_analysis_sample_excluded_areas.csv"
SUMMARY_PATH = PROJECT_DIR / "tables" / "main_analysis_sample_coverage_summary.csv"

START_YEAR = 1992
MIN_YEARS = 25

HISTORICAL_ENTITIES = {
    "USSR",
    "Yugoslav SFR",
    "Czechoslovakia",
    "Ethiopia PDR",
    "Sudan (former)",
    "Serbia and Montenegro",
    "Belgium-Luxembourg",
}

NON_STANDARD_FAOSTAT_ENTITIES = {
    "China",  # FAOSTAT aggregate that overlaps China, mainland / SAR / Taiwan entries.
}


def main() -> int:
    df = pd.read_csv(INPUT_PATH, dtype={"area_code_m49": str})
    after_start = df[df["year"] >= START_YEAR].copy()

    coverage = (
        after_start.groupby(["area_code_m49", "area"], as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
            n_rolling10_years=("rolling10_anomaly_sd", lambda s: int(s.notna().sum())),
            mean_total_crop_production=("total_crop_production", "mean"),
        )
    )
    coverage["is_historical_entity"] = coverage["area"].isin(HISTORICAL_ENTITIES)
    coverage["is_non_standard_faostat_entity"] = coverage["area"].isin(NON_STANDARD_FAOSTAT_ENTITIES)
    coverage["exclude_reason"] = ""
    coverage.loc[coverage["is_historical_entity"], "exclude_reason"] = "historical_entity"
    coverage.loc[coverage["is_non_standard_faostat_entity"], "exclude_reason"] = "non_standard_faostat_entity"
    coverage.loc[coverage["n_years"] < MIN_YEARS, "exclude_reason"] = coverage["exclude_reason"].mask(
        coverage["exclude_reason"].eq(""),
        f"fewer_than_{MIN_YEARS}_years_since_{START_YEAR}",
    )
    excluded = coverage[coverage["exclude_reason"].ne("")].copy()
    keep_areas = coverage.loc[coverage["exclude_reason"].eq(""), ["area_code_m49", "area"]]

    main = after_start.merge(keep_areas, on=["area_code_m49", "area"], how="inner")
    main.to_csv(OUT_PATH, index=False)
    excluded.to_csv(EXCLUSION_PATH, index=False)
    coverage.to_csv(SUMMARY_PATH, index=False)

    print(f"wrote {OUT_PATH} ({len(main):,} rows; {main['area'].nunique():,} areas)")
    print(f"wrote {EXCLUSION_PATH} ({len(excluded):,} excluded areas)")
    print(f"wrote {SUMMARY_PATH} ({len(coverage):,} areas evaluated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
