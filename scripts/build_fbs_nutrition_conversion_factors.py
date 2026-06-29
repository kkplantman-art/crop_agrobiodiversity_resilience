from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW_ZIP = PROJECT / "raw" / "faostat" / "FoodBalanceSheets_E_All_Data_(Normalized).zip"
OUT_DIR = PROJECT / "processed"
TABLE_DIR = PROJECT / "tables"


def read_fbs_selected():
    elements = {
        "Food supply quantity (kg/capita/yr)",
        "Food supply (kcal/capita/day)",
        "Protein supply quantity (g/capita/day)",
        "Food supply (kcal)",
        "Protein supply quantity (t)",
    }
    cols = [
        "Area Code (M49)",
        "Area",
        "Item Code",
        "Item Code (FBS)",
        "Item",
        "Element",
        "Year",
        "Unit",
        "Value",
    ]
    frames = []
    with zipfile.ZipFile(RAW_ZIP) as z:
        csv_name = [n for n in z.namelist() if n.endswith("_All_Data_(Normalized).csv")][0]
        for chunk in pd.read_csv(
            z.open(csv_name),
            usecols=cols,
            encoding="latin1",
            chunksize=1_000_000,
            low_memory=False,
        ):
            sub = chunk[chunk["Element"].isin(elements)].copy()
            if not sub.empty:
                frames.append(sub)
    if not frames:
        raise RuntimeError("No selected FBS nutrition elements found.")
    return pd.concat(frames, ignore_index=True)


def build_conversion_factors(df: pd.DataFrame) -> pd.DataFrame:
    # Per-capita factors use kg food supply and kcal/protein per capita per day.
    wide = (
        df[df["Element"].isin(
            [
                "Food supply quantity (kg/capita/yr)",
                "Food supply (kcal/capita/day)",
                "Protein supply quantity (g/capita/day)",
            ]
        )]
        .pivot_table(
            index=["Area Code (M49)", "Area", "Item Code", "Item Code (FBS)", "Item", "Year"],
            columns="Element",
            values="Value",
            aggfunc="sum",
        )
        .reset_index()
    )
    food_kg = wide["Food supply quantity (kg/capita/yr)"].replace(0, np.nan)
    wide["kcal_per_kg_food_supply"] = wide["Food supply (kcal/capita/day)"] * 365.0 / food_kg
    wide["protein_g_per_kg_food_supply"] = wide["Protein supply quantity (g/capita/day)"] * 365.0 / food_kg

    valid = wide[
        (wide["kcal_per_kg_food_supply"].between(10, 9500))
        & (wide["protein_g_per_kg_food_supply"].between(0, 450))
    ].copy()

    factors = (
        valid.groupby(["Item Code", "Item Code (FBS)", "Item"], as_index=False)
        .agg(
            kcal_per_kg=("kcal_per_kg_food_supply", "median"),
            protein_g_per_kg=("protein_g_per_kg_food_supply", "median"),
            n_country_year=("kcal_per_kg_food_supply", "size"),
            n_areas=("Area", "nunique"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
        )
        .sort_values("Item")
    )
    return factors


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected = read_fbs_selected()
    selected.to_csv(OUT_DIR / "fbs_selected_nutrition_elements_long.csv", index=False)
    factors = build_conversion_factors(selected)
    factors.to_csv(OUT_DIR / "fbs_food_item_nutrition_conversion_factors.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "selected_fbs_rows",
                "value": len(selected),
            },
            {
                "metric": "unique_fbs_items_with_factors",
                "value": len(factors),
            },
            {
                "metric": "median_country_years_per_factor",
                "value": factors["n_country_year"].median(),
            },
        ]
    )
    summary.to_csv(TABLE_DIR / "nutrition_conversion_factor_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
