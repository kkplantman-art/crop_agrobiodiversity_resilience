from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CROP_PANEL = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
FBS_FACTORS = PROJECT_DIR / "processed" / "fbs_food_item_nutrition_conversion_factors.csv"
OUT_MAPPING = PROJECT_DIR / "processed" / "crop_item_to_fbs_nutrition_factor_mapping.csv"
OUT_PANEL = PROJECT_DIR / "processed" / "country_year_nutrition_weighted_production_metrics.csv"
OUT_SUMMARY = PROJECT_DIR / "tables" / "nutrition_weighted_outcome_coverage_summary.csv"


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("n.e.c.", "nec")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def map_crop_to_fbs(item: str, crop_group: str) -> tuple[str | None, str, str]:
    n = norm(item)

    # Explicit exclusions: produced biomass is not a food/nutrition outcome.
    if crop_group == "fibre_crops" or any(x in n for x in ["jute", "kenaf", "flax raw", "cotton unginned"]):
        return None, "exclude_non_food_fibre_crop", "high"

    exact = {
        "wheat": "Wheat and products",
        "rice": "Rice and products",
        "maize": "Maize and products",
        "maize corn": "Maize and products",
        "green corn": "Maize and products",
        "barley": "Barley and products",
        "sorghum": "Sorghum and products",
        "millet": "Millet and products",
        "oats": "Oats",
        "rye": "Rye and products",
        "potatoes": "Potatoes and products",
        "cassava fresh": "Cassava and products",
        "sweet potatoes": "Sweet potatoes",
        "yams": "Yams",
        "sugar cane": "Sugar cane",
        "sugar beet": "Sugar beet",
        "soya beans": "Soyabeans",
        "beans dry": "Beans",
        "peas dry": "Peas",
        "peas green": "Peas",
        "groundnuts excluding shelled": "Groundnuts",
        "sunflower seed": "Sunflower seed",
        "sesame seed": "Sesame seed",
        "coconuts in shell": "Coconuts - Incl Copra",
        "olives": "Olives (including preserved)",
        "bananas": "Bananas",
        "plantains and cooking bananas": "Plantains",
        "apples": "Apples and products",
        "grapes": "Grapes and products (excl wine)",
        "oranges": "Oranges, Mandarines",
        "pineapples": "Pineapples and products",
        "dates": "Dates",
        "tomatoes": "Tomatoes and products",
        "onions and shallots dry": "Onions",
        "onions and shallots green": "Onions",
        "coffee green": "Coffee and products",
        "tea leaves": "Tea (including mate)",
        "cocoa beans": "Cocoa Beans and products",
    }
    if n in exact:
        return exact[n], "exact_or_near_exact_name_rule", "high"

    if any(x in n for x in ["rape", "colza", "mustard seed"]):
        return "Rape and Mustardseed", "oilseed_specific_rule", "high"
    if "cottonseed" in n:
        return "Cottonseed", "oilseed_specific_rule", "medium"
    if "oil palm" in n or "palm fruit" in n:
        return "Oilcrops", "oilcrop_aggregate_rule_for_raw_oil_palm_fruit", "low"
    if any(x in n for x in ["linseed", "safflower", "castor", "melonseed", "karite", "shea", "tallowtree", "tung"]):
        return "Oilcrops, Other", "other_oilseed_aggregate_rule", "medium"
    if "other oil seeds" in n:
        return "Oilcrops, Other", "other_oilseed_aggregate_rule", "medium"

    if any(x in n for x in ["chick peas", "lentils", "cow peas", "pigeon peas", "lupins", "vetches", "broad beans", "horse beans", "other pulses"]):
        return "Pulses, Other and products", "pulse_aggregate_rule", "medium"
    if any(x in n for x in ["other beans", "string beans", "bambara beans", "locust beans"]):
        return "Beans", "bean_aggregate_rule", "medium"

    if any(x in n for x in ["buckwheat", "triticale", "mixed grain", "cereals nec", "fonio", "quinoa", "canary seed"]):
        return "Cereals, other", "other_cereal_aggregate_rule", "medium"

    if any(x in n for x in ["taro", "edible roots", "roots tubers", "chicory roots", "yautia"]):
        return "Roots, Other", "other_root_tuber_aggregate_rule", "medium"

    if any(x in n for x in ["lemons", "limes"]):
        return "Lemons, Limes and products", "citrus_specific_rule", "high"
    if any(x in n for x in ["tangerines", "mandarins", "clementines"]):
        return "Oranges, Mandarines", "citrus_aggregate_rule", "medium"
    if any(x in n for x in ["grapefruit", "pomelo"]):
        return "Grapefruit and products", "citrus_specific_rule", "high"
    if "other citrus" in n:
        return "Citrus, Other", "citrus_aggregate_rule", "medium"

    if any(x in n for x in ["almonds", "walnuts", "hazelnuts", "cashew nuts", "chestnuts", "other nuts", "pistachios", "areca nuts", "kola nuts"]):
        return "Treenuts", "treenut_aggregate_rule", "medium"

    if any(x in n for x in ["mango", "guava", "mangosteen", "papaya", "plums", "peaches", "nectarines", "apricots", "strawberries", "berries", "persimmons", "figs", "kiwi", "cherries", "avocados", "cashewapple", "other fruits", "tropical fruits", "pears", "currants", "quinces", "stone fruits", "kapok fruit", "other pome fruits"]):
        return "Fruits, other", "other_fruit_aggregate_rule", "medium"

    if any(x in n for x in ["watermelons", "melons", "cantaloupes", "cabbages", "cucumbers", "eggplants", "carrots", "turnips", "lettuce", "spinach", "garlic", "pumpkins", "squash", "gourds", "cauliflowers", "broccoli", "asparagus", "okra", "artichokes", "leeks", "other vegetables"]):
        return "Vegetables, other", "other_vegetable_aggregate_rule", "medium"
    if "chillies" in n or "peppers" in n:
        return "Pimento" if "green" in n else "Pepper", "pepper_specific_rule", "medium"

    if any(x in n for x in ["ginger", "anise", "coriander", "cumin", "caraway", "fennel", "juniper", "other stimulant spice", "spice", "cinnamon", "cloves", "nutmeg", "mace", "cardamoms", "peppermint", "spearmint"]):
        return "Spices, Other", "spice_aggregate_rule", "medium"
    if "pepper piper" in n:
        return "Pepper", "pepper_specific_rule", "medium"
    if "mate" in n or "maté" in item.lower():
        return "Tea (including mate)", "stimulant_specific_rule", "medium"

    if "other sugar crops" in n:
        return "Sugar Crops", "sugar_crop_aggregate_rule", "medium"

    if any(x in n for x in ["poppy seed", "hempseed"]):
        return "Oilcrops, Other", "other_oilseed_aggregate_rule", "medium"

    return None, "unmatched", "none"


def detrend_country_year(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    trend_col = out_col.replace("_anomaly", "_trend")
    df[trend_col] = np.nan
    for area, idx in df.groupby("area").groups.items():
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


def main() -> int:
    crop = pd.read_csv(CROP_PANEL, dtype={"area_code_m49": str, "item_code_cpc": str})
    factors = pd.read_csv(FBS_FACTORS)
    factors_by_item = (
        factors.groupby("Item", as_index=False)
        .agg(
            kcal_per_kg=("kcal_per_kg", "median"),
            protein_g_per_kg=("protein_g_per_kg", "median"),
            fbs_n_country_year=("n_country_year", "sum"),
            fbs_n_areas=("n_areas", "max"),
        )
    )

    items = crop.groupby(["item_code_cpc", "item", "crop_group"], as_index=False)["production"].sum()
    mapped_rows = []
    for row in items.itertuples(index=False):
        fbs_item, rule, confidence = map_crop_to_fbs(row.item, row.crop_group)
        mapped_rows.append(
            {
                "item_code_cpc": row.item_code_cpc,
                "item": row.item,
                "crop_group": row.crop_group,
                "total_production": row.production,
                "fbs_item": fbs_item,
                "mapping_rule": rule,
                "mapping_confidence": confidence,
            }
        )
    mapping = pd.DataFrame(mapped_rows)
    mapping = mapping.merge(factors_by_item, left_on="fbs_item", right_on="Item", how="left").drop(columns=["Item"])
    mapping["has_nutrition_factor"] = mapping["kcal_per_kg"].notna() & mapping["protein_g_per_kg"].notna()
    mapping.to_csv(OUT_MAPPING, index=False)

    enriched = crop.merge(
        mapping[
            [
                "item_code_cpc",
                "item",
                "fbs_item",
                "mapping_rule",
                "mapping_confidence",
                "kcal_per_kg",
                "protein_g_per_kg",
                "has_nutrition_factor",
            ]
        ],
        on=["item_code_cpc", "item"],
        how="left",
    )
    enriched["covered_production"] = np.where(enriched["has_nutrition_factor"], enriched["production"], 0.0)
    enriched["kcal_production"] = enriched["production"] * 1000.0 * enriched["kcal_per_kg"]
    enriched["protein_kg_production"] = enriched["production"] * enriched["protein_g_per_kg"]

    out = (
        enriched.groupby(["area_code_m49", "area", "year"], as_index=False)
        .agg(
            total_crop_production=("production", "sum"),
            nutrition_covered_crop_production=("covered_production", "sum"),
            nutrition_kcal_production=("kcal_production", "sum"),
            nutrition_protein_kg_production=("protein_kg_production", "sum"),
            n_crop_items=("item", "nunique"),
            n_crop_items_with_nutrition=("has_nutrition_factor", "sum"),
        )
        .sort_values(["area", "year"])
    )
    out["nutrition_coverage_production_share"] = (
        out["nutrition_covered_crop_production"] / out["total_crop_production"].replace(0, np.nan)
    )
    out["log_kcal_production"] = np.log1p(out["nutrition_kcal_production"])
    out["log_protein_production"] = np.log1p(out["nutrition_protein_kg_production"])
    out = detrend_country_year(out, "log_kcal_production", "nutrition_kcal_log_anomaly")
    out = detrend_country_year(out, "log_protein_production", "nutrition_protein_log_anomaly")
    out.to_csv(OUT_PANEL, index=False)

    main_years = out[out["year"].between(1992, 2024)].copy()
    item_total = mapping["total_production"].sum()
    item_covered = mapping.loc[mapping["has_nutrition_factor"], "total_production"].sum()
    summary = pd.DataFrame(
        [
            {"metric": "crop_items_total", "value": len(mapping)},
            {"metric": "crop_items_with_nutrition_factor", "value": int(mapping["has_nutrition_factor"].sum())},
            {"metric": "lifetime_production_share_mapped", "value": item_covered / item_total},
            {"metric": "main_period_country_years", "value": len(main_years)},
            {"metric": "main_period_median_production_coverage_share", "value": main_years["nutrition_coverage_production_share"].median()},
            {"metric": "main_period_country_years_coverage_ge_0_75", "value": int((main_years["nutrition_coverage_production_share"] >= 0.75).sum())},
            {"metric": "main_period_country_years_coverage_ge_0_90", "value": int((main_years["nutrition_coverage_production_share"] >= 0.90).sum())},
            {"metric": "main_period_areas", "value": main_years["area"].nunique()},
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {OUT_MAPPING}")
    print(f"wrote {OUT_PANEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
