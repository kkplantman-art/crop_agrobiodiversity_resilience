"""Build country-level crop diversity metrics from the filtered FAOSTAT panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_crop_diversity_metrics import build_metrics


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "faostat_crop_country_year_panel_country_level.csv"
OUT_PATH = PROJECT_DIR / "processed" / "country_year_crop_diversity_metrics_country_level.csv"


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, dtype={"area_code_m49": str})
    metrics = build_metrics(panel)
    metrics.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(metrics):,} rows; {metrics['area'].nunique():,} areas)")
    print(metrics.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
