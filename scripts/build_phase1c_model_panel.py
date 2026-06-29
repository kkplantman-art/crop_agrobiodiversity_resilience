"""Merge crop diversity and function-stability metrics for Phase 1C diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DIVERSITY_PATH = PROJECT_DIR / "processed" / "country_year_crop_diversity_metrics_country_level.csv"
FUNCTION_PATH = PROJECT_DIR / "processed" / "country_year_function_stability_metrics.csv"
OUT_PATH = PROJECT_DIR / "processed" / "phase1c_diversity_stability_model_panel.csv"


def main() -> int:
    diversity = pd.read_csv(DIVERSITY_PATH, dtype={"area_code_m49": str})
    function = pd.read_csv(FUNCTION_PATH, dtype={"area_code_m49": str})
    function = function.rename(columns={"total_harvested_area": "function_total_harvested_area"})
    merged = diversity.merge(
        function,
        on=["area_code_m49", "area", "year"],
        how="inner",
        validate="one_to_one",
    )
    merged.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(merged):,} rows; {merged['area'].nunique():,} areas)")
    print(merged.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
