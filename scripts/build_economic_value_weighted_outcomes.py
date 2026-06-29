"""Build FAOSTAT constant-value crop production outcomes.

The outcome is country-year gross crop production value in constant
2014-2016 international dollars, restricted to the crop items retained in the
main FAOSTAT crop panel. It is intended as an economic-resilience sensitivity
outcome, not as a full welfare or market-access measure.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CROP_PANEL = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
VALUE_ZIP = PROJECT_DIR / "raw" / "faostat" / "Value_of_Production_E_All_Data_(Normalized).zip"
OUT_PANEL = PROJECT_DIR / "processed" / "country_year_economic_value_weighted_production_metrics.csv"
OUT_ITEM_AUDIT = PROJECT_DIR / "processed" / "crop_item_value_production_coverage_audit.csv"
OUT_SUMMARY = PROJECT_DIR / "tables" / "economic_value_weighted_outcome_coverage_summary.csv"

VALUE_ELEMENT = "Gross Production Value (constant 2014-2016 thousand I$)"


def detrend_country_year(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    trend_col = out_col.replace("_anomaly", "_trend")
    df[trend_col] = np.nan
    for _, idx in df.groupby("area").groups.items():
        sub = df.loc[idx, ["year", value_col]].dropna()
        if len(sub) < 10:
            continue
        x = sub["year"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * df.loc[idx, "year"].to_numpy(dtype=float) + intercept
        df.loc[idx, trend_col] = fit
        df.loc[idx, out_col] = df.loc[idx, value_col].to_numpy(dtype=float) - fit
    return df


def load_value_rows(valid_item_codes: set[str]) -> pd.DataFrame:
    usecols = ["Area Code (M49)", "Area", "Item Code (CPC)", "Item", "Element", "Year", "Unit", "Value", "Flag"]
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(VALUE_ZIP) as zf:
        data_file = [name for name in zf.namelist() if name.endswith(".csv") and "All_Data" in name][0]
        reader = pd.read_csv(
            zf.open(data_file),
            usecols=usecols,
            dtype={"Area Code (M49)": str, "Item Code (CPC)": str},
            chunksize=500_000,
        )
        for chunk in reader:
            sub = chunk[
                chunk["Element"].eq(VALUE_ELEMENT)
                & chunk["Item Code (CPC)"].isin(valid_item_codes)
            ].copy()
            if sub.empty:
                continue
            sub = sub.rename(
                columns={
                    "Area Code (M49)": "area_code_m49",
                    "Area": "value_area",
                    "Item Code (CPC)": "item_code_cpc",
                    "Item": "value_item",
                    "Year": "year",
                    "Value": "gross_production_value_constant_2014_2016_int_1000",
                    "Flag": "value_flag",
                }
            )
            rows.append(
                sub[
                    [
                        "area_code_m49",
                        "value_area",
                        "item_code_cpc",
                        "value_item",
                        "year",
                        "gross_production_value_constant_2014_2016_int_1000",
                        "value_flag",
                    ]
                ]
            )
    if not rows:
        return pd.DataFrame()
    value = pd.concat(rows, ignore_index=True)
    return value.drop_duplicates(["area_code_m49", "item_code_cpc", "year"])


def main() -> int:
    crop = pd.read_csv(
        CROP_PANEL,
        dtype={"area_code_m49": str, "item_code_cpc": str},
        usecols=["area_code_m49", "area", "item_code_cpc", "item", "year", "production", "crop_group"],
    )
    valid_item_codes = set(crop["item_code_cpc"].dropna().unique())
    value = load_value_rows(valid_item_codes)
    if value.empty:
        raise RuntimeError(f"No value rows found for {VALUE_ELEMENT}")

    item_audit = (
        crop.groupby(["item_code_cpc", "item", "crop_group"], as_index=False)
        .agg(total_production=("production", "sum"))
        .merge(
            value.groupby("item_code_cpc", as_index=False)
            .agg(
                n_value_rows=("gross_production_value_constant_2014_2016_int_1000", "count"),
                n_value_areas=("area_code_m49", "nunique"),
                first_value_year=("year", "min"),
                last_value_year=("year", "max"),
            ),
            on="item_code_cpc",
            how="left",
        )
    )
    item_audit["has_value_rows"] = item_audit["n_value_rows"].fillna(0).gt(0)
    item_audit.to_csv(OUT_ITEM_AUDIT, index=False)

    enriched = crop.merge(
        value[["area_code_m49", "item_code_cpc", "year", "gross_production_value_constant_2014_2016_int_1000"]],
        on=["area_code_m49", "item_code_cpc", "year"],
        how="left",
    )
    enriched["has_value_outcome"] = enriched["gross_production_value_constant_2014_2016_int_1000"].notna()
    enriched["economic_covered_crop_production"] = np.where(enriched["has_value_outcome"], enriched["production"], 0.0)

    out = (
        enriched.groupby(["area_code_m49", "area", "year"], as_index=False)
        .agg(
            total_crop_production=("production", "sum"),
            economic_covered_crop_production=("economic_covered_crop_production", "sum"),
            gross_production_value_constant_2014_2016_int_1000=("gross_production_value_constant_2014_2016_int_1000", "sum"),
            n_crop_items=("item", "nunique"),
            n_crop_items_with_value=("has_value_outcome", "sum"),
        )
        .sort_values(["area", "year"])
    )
    out["economic_value_coverage_production_share"] = (
        out["economic_covered_crop_production"] / out["total_crop_production"].replace(0, np.nan)
    )
    out["has_economic_value_outcome"] = out["gross_production_value_constant_2014_2016_int_1000"].gt(0)
    out["log_gross_production_value_constant_int"] = np.where(
        out["has_economic_value_outcome"],
        np.log1p(out["gross_production_value_constant_2014_2016_int_1000"]),
        np.nan,
    )
    out = detrend_country_year(
        out,
        "log_gross_production_value_constant_int",
        "economic_value_log_anomaly",
    )
    out.to_csv(OUT_PANEL, index=False)

    main_period = out[out["year"].between(1992, 2024)].copy()
    summary = pd.DataFrame(
        [
            {"metric": "crop_items_total", "value": float(item_audit["item_code_cpc"].nunique())},
            {"metric": "crop_items_with_value_rows", "value": float(item_audit["has_value_rows"].sum())},
            {
                "metric": "lifetime_production_share_with_value_rows",
                "value": float(
                    item_audit.loc[item_audit["has_value_rows"], "total_production"].sum()
                    / item_audit["total_production"].sum()
                ),
            },
            {"metric": "main_period_country_years", "value": float(len(main_period))},
            {
                "metric": "main_period_country_years_with_value_outcome",
                "value": float(main_period["has_economic_value_outcome"].sum()),
            },
            {
                "metric": "main_period_median_production_coverage_share",
                "value": float(main_period["economic_value_coverage_production_share"].median()),
            },
            {
                "metric": "main_period_country_years_coverage_ge_0_75",
                "value": float(main_period["economic_value_coverage_production_share"].ge(0.75).sum()),
            },
            {"metric": "main_period_areas", "value": float(main_period["area"].nunique())},
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {OUT_PANEL}")
    print(f"wrote {OUT_ITEM_AUDIT}")
    print(f"wrote {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
