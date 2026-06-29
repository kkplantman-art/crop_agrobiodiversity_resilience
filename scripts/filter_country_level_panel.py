"""Create a country/territory-level FAOSTAT crop panel.

The raw FAOSTAT production domain includes aggregate regions such as World,
Africa, Europe, Least Developed Countries and economic groupings. These are
useful for checks but should not enter country-year panel models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel.csv"
COUNTRY_PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
EXCLUDED_AREAS_PATH = PROJECT_DIR / "tables" / "excluded_faostat_aggregate_areas.csv"

AGGREGATE_AREA_NAMES = {
    "World",
    "Africa",
    "Americas",
    "Asia",
    "Europe",
    "Oceania",
    "Eastern Africa",
    "Middle Africa",
    "Northern Africa",
    "Southern Africa",
    "Western Africa",
    "Northern America",
    "Central America",
    "Caribbean",
    "South America",
    "Central Asia",
    "Eastern Asia",
    "Southern Asia",
    "South-eastern Asia",
    "Western Asia",
    "Eastern Europe",
    "Northern Europe",
    "Southern Europe",
    "Western Europe",
    "Australia and New Zealand",
    "Melanesia",
    "Micronesia",
    "Polynesia",
    "European Union (27)",
    "European Union (25)",
    "European Union (15)",
    "European Union (12)",
    "Least Developed Countries (LDCs)",
    "Land Locked Developing Countries (LLDCs)",
    "Low Income Food Deficit Countries (LIFDCs)",
    "Net Food Importing Developing Countries (NFIDCs)",
    "Small Island Developing States (SIDS)",
}


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, dtype={"area_code_m49": str})
    excluded = (
        panel.loc[panel["area"].isin(AGGREGATE_AREA_NAMES), ["area_code_m49", "area"]]
        .drop_duplicates()
        .sort_values("area")
        .reset_index(drop=True)
    )
    country = panel[~panel["area"].isin(AGGREGATE_AREA_NAMES)].copy()

    country.to_csv(COUNTRY_PANEL_PATH, index=False)
    excluded.to_csv(EXCLUDED_AREAS_PATH, index=False)

    print(f"wrote {COUNTRY_PANEL_PATH} ({len(country):,} rows; {country['area'].nunique():,} areas)")
    print(f"wrote {EXCLUDED_AREAS_PATH} ({len(excluded):,} aggregate areas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
