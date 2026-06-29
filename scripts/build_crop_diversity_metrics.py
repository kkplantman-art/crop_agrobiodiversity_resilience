"""Build country-year crop diversity metrics from the FAOSTAT crop panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_crop_diversity_metrics.csv"


def shannon(proportions: pd.Series) -> float:
    p = proportions[(proportions > 0) & proportions.notna()]
    if p.empty:
        return np.nan
    return float(-(p * np.log(p)).sum())


def simpson(proportions: pd.Series) -> float:
    p = proportions[(proportions > 0) & proportions.notna()]
    if p.empty:
        return np.nan
    return float(1 - (p**2).sum())


def build_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["area_harvested"] = pd.to_numeric(df["area_harvested"], errors="coerce")
    df = df[df["area_harvested"].notna() & (df["area_harvested"] > 0)].copy()

    totals = (
        df.groupby(["area_code_m49", "area", "year"], as_index=False)["area_harvested"]
        .sum()
        .rename(columns={"area_harvested": "total_harvested_area"})
    )
    df = df.merge(totals, on=["area_code_m49", "area", "year"], how="left")
    df["area_share"] = df["area_harvested"] / df["total_harvested_area"]

    base = (
        df.groupby(["area_code_m49", "area", "year"])
        .agg(
            total_harvested_area=("area_harvested", "sum"),
            n_crop_items=("item", "nunique"),
            n_functional_groups=("crop_group", "nunique"),
            top_crop_share=("area_share", "max"),
            shannon_crop_area=("area_share", shannon),
            simpson_crop_area=("area_share", simpson),
        )
        .reset_index()
    )
    base["effective_number_crops"] = np.exp(base["shannon_crop_area"])

    group_share = (
        df.groupby(["area_code_m49", "area", "year", "crop_group"], as_index=False)["area_harvested"]
        .sum()
        .merge(totals, on=["area_code_m49", "area", "year"], how="left")
    )
    group_share["group_share"] = group_share["area_harvested"] / group_share["total_harvested_area"]
    wide = (
        group_share.pivot_table(
            index=["area_code_m49", "area", "year"],
            columns="crop_group",
            values="group_share",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename = {col: f"share_{col}" for col in wide.columns if col not in {"area_code_m49", "area", "year"}}
    wide = wide.rename(columns=rename)

    out = base.merge(wide, on=["area_code_m49", "area", "year"], how="left")
    return out.sort_values(["area", "year"]).reset_index(drop=True)


def main() -> int:
    panel = pd.read_csv(PANEL_PATH)
    metrics = build_metrics(panel)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(metrics):,} rows)")
    print(metrics.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
