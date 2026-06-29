"""Build preliminary agroecosystem-function stability metrics.

This Phase 1C prototype uses country-year total crop production from the
filtered FAOSTAT country-level panel. Nutritional conversion is intentionally
left for a later phase so the first stability diagnostic stays transparent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
OUTCOMES_PATH = PROJECT_DIR / "processed" / "country_year_function_stability_metrics.csv"
SUMMARY_PATH = PROJECT_DIR / "tables" / "function_stability_coverage_summary.csv"


def add_linear_trend_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for area, group in frame.groupby("area", sort=False):
        group = group.sort_values("year").copy()
        valid = group["log_total_production"].replace([np.inf, -np.inf], np.nan).notna()
        if valid.sum() >= 10:
            x = group.loc[valid, "year"].to_numpy(dtype=float)
            y = group.loc[valid, "log_total_production"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, deg=1)
            group["log_total_production_trend"] = intercept + slope * group["year"].to_numpy(dtype=float)
            group["production_log_anomaly"] = group["log_total_production"] - group["log_total_production_trend"]
            group["production_percent_anomaly"] = np.expm1(group["production_log_anomaly"]) * 100
        else:
            group["log_total_production_trend"] = np.nan
            group["production_log_anomaly"] = np.nan
            group["production_percent_anomaly"] = np.nan
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def add_rolling_stability(frame: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for area, group in frame.groupby("area", sort=False):
        group = group.sort_values("year").copy()
        rolling = group["production_log_anomaly"].rolling(window=window, min_periods=6)
        group[f"rolling{window}_anomaly_sd"] = rolling.std()
        group[f"rolling{window}_stability_inverse_sd"] = 1 / group[f"rolling{window}_anomaly_sd"].replace(0, np.nan)
        prod_roll = group["total_crop_production"].rolling(window=window, min_periods=6)
        group[f"rolling{window}_production_cv"] = prod_roll.std() / prod_roll.mean().replace(0, np.nan)
        group[f"rolling{window}_stability_inverse_cv"] = 1 / group[f"rolling{window}_production_cv"].replace(0, np.nan)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def build_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df["area_harvested"] = pd.to_numeric(df["area_harvested"], errors="coerce")
    df = df[df["production"].notna() & (df["production"] >= 0)].copy()

    outcomes = (
        df.groupby(["area_code_m49", "area", "year"], as_index=False)
        .agg(
            total_crop_production=("production", "sum"),
            n_crop_items_with_production=("item", "nunique"),
            total_harvested_area=("area_harvested", "sum"),
        )
        .sort_values(["area", "year"])
        .reset_index(drop=True)
    )
    outcomes["log_total_production"] = np.log1p(outcomes["total_crop_production"])
    outcomes = add_linear_trend_residuals(outcomes)
    outcomes = add_rolling_stability(outcomes, window=10)
    return outcomes


def build_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        outcomes.groupby("area", as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
            mean_total_crop_production=("total_crop_production", "mean"),
            mean_n_crop_items_with_production=("n_crop_items_with_production", "mean"),
            n_anomaly_years=("production_log_anomaly", lambda s: int(s.notna().sum())),
            n_rolling10_years=("rolling10_anomaly_sd", lambda s: int(s.notna().sum())),
        )
        .sort_values(["n_years", "mean_total_crop_production"], ascending=False)
    )
    summary["coverage_span_years"] = summary["last_year"] - summary["first_year"] + 1
    summary["year_coverage_fraction"] = summary["n_years"] / summary["coverage_span_years"].replace(0, np.nan)
    return summary


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, dtype={"area_code_m49": str})
    outcomes = build_outcomes(panel)
    summary = build_summary(outcomes)
    outcomes.to_csv(OUTCOMES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {OUTCOMES_PATH} ({len(outcomes):,} rows; {outcomes['area'].nunique():,} areas)")
    print(f"wrote {SUMMARY_PATH} ({len(summary):,} rows)")
    print(outcomes.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
