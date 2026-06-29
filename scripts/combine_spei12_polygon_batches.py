"""Combine SPEI12 polygon-sample batch outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_DIR / "processed" / "spei_polygon_batches"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_spei12_polygon_sample_drought_metrics.csv"
POINTS_PATH = PROJECT_DIR / "processed" / "country_spei12_polygon_sample_points.csv"


def main() -> int:
    annual_files = sorted(BATCH_DIR.glob("country_year_spei12_polygon_sample_drought_metrics_*.csv"))
    point_files = sorted(BATCH_DIR.glob("country_spei12_polygon_sample_points_*.csv"))
    if not annual_files:
        raise RuntimeError(f"No annual batch files found in {BATCH_DIR}")
    annual = pd.concat([pd.read_csv(path, dtype={"area_code_m49": str}) for path in annual_files], ignore_index=True)
    points = pd.concat([pd.read_csv(path, dtype={"area_code_m49": str}) for path in point_files], ignore_index=True) if point_files else pd.DataFrame()
    annual = annual.drop_duplicates(["area_code_m49", "area", "year"]).sort_values(["area", "year"])
    if not points.empty:
        point_cols = [col for col in ["area_code_m49", "area", "spei_chunk_index"] if col in points.columns]
        if point_cols:
            points = points.drop_duplicates(point_cols)
        points = points.sort_values(["area_code_m49", "area"])
    annual.to_csv(OUT_PATH, index=False)
    points.to_csv(POINTS_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(annual):,} rows; {annual['area'].nunique():,} areas)")
    print(f"wrote {POINTS_PATH} ({len(points):,} rows)")
    print("batch files:")
    for path in annual_files:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
