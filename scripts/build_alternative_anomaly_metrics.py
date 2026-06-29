"""Build alternative production anomaly metrics for sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FUNCTION_PATH = PROJECT_DIR / "processed" / "country_year_function_stability_metrics.csv"
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_alternative_anomaly_metrics.csv"


def add_alt_metrics(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("year").copy()
    logp = group["log_total_production"]
    group["production_log_growth"] = logp.diff()
    for window in [5, 10]:
        trend = logp.rolling(window=window, center=True, min_periods=max(3, window // 2 + 1)).mean()
        group[f"production_log_anomaly_roll{window}"] = logp - trend
        group[f"production_percent_anomaly_roll{window}"] = np.expm1(group[f"production_log_anomaly_roll{window}"]) * 100
    return group


def main() -> int:
    function = pd.read_csv(FUNCTION_PATH, dtype={"area_code_m49": str})
    main_areas = pd.read_csv(MAIN_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    function = function.merge(main_areas, on=["area_code_m49", "area"], how="inner")
    function = function[function["year"] >= 1992].copy()
    pieces = [add_alt_metrics(group) for _, group in function.groupby("area", sort=False)]
    out = pd.concat(pieces, ignore_index=True)
    keep_cols = [
        "area_code_m49",
        "area",
        "year",
        "production_log_anomaly",
        "production_percent_anomaly",
        "production_log_growth",
        "production_log_anomaly_roll5",
        "production_percent_anomaly_roll5",
        "production_log_anomaly_roll10",
        "production_percent_anomaly_roll10",
    ]
    out[keep_cols].to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows; {out['area'].nunique():,} areas)")
    print(out[keep_cols].head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
