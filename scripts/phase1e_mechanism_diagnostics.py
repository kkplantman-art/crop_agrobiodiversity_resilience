"""Run preliminary mechanism diagnostics for crop-response asynchrony."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
ASYNC_PATH = PROJECT_DIR / "processed" / "country_year_crop_response_asynchrony_metrics.csv"
MERGED_PATH = PROJECT_DIR / "processed" / "phase1e_mechanism_panel.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1e_mechanism_diagnostics.csv"


def corr_rows(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        work = df[[x, y]].dropna()
        rows.append(
            {
                "x": x,
                "y": y,
                "correlation": work[x].corr(work[y]) if len(work) > 2 else np.nan,
                "n": len(work),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    main = pd.read_csv(MAIN_PATH, dtype={"area_code_m49": str})
    async_df = pd.read_csv(ASYNC_PATH, dtype={"area_code_m49": str})
    merged = main.merge(async_df, on=["area_code_m49", "area", "year"], how="left", validate="one_to_one")
    merged.to_csv(MERGED_PATH, index=False)

    pairs = [
        ("shannon_crop_area", "crop_response_asynchrony_index"),
        ("effective_number_crops", "crop_response_asynchrony_index"),
        ("top_crop_share", "crop_response_asynchrony_index"),
        ("crop_response_asynchrony_index", "rolling10_anomaly_sd"),
        ("crop_response_asynchrony_index", "production_log_anomaly"),
        ("share_cereals", "crop_response_asynchrony_index"),
        ("share_legumes", "crop_response_asynchrony_index"),
    ]
    out = corr_rows(merged, pairs)

    drought = merged[merged["drought_spei12_mean_lt_minus1"] == True]
    drought_pairs = corr_rows(
        drought,
        [
            ("shannon_crop_area", "crop_response_asynchrony_index"),
            ("crop_response_asynchrony_index", "production_log_anomaly"),
        ],
    )
    drought_pairs["subset"] = "drought_years"
    out["subset"] = "all_years"
    out = pd.concat([out, drought_pairs], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {MERGED_PATH} ({len(merged):,} rows)")
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
