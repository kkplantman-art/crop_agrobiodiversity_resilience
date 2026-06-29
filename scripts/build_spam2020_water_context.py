"""Build country-level SPAM 2020 irrigated and rainfed harvested-area context.

SPAM harvested-area files provide total (H_TA), irrigated (H_TI) and rainfed
(H_TR) harvested area. This script collapses the three layers to countries and
matches them to the FAOSTAT/Natural Earth M49 country identifiers used in the
main analysis panel.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_spam2020_spei12_harvested_area_weights import load_fips_to_m49


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPAM_ZIP = PROJECT_DIR / "raw" / "spam2020" / "spam2020V2r2_global_harvested_area.csv.zip"
MAIN_SAMPLE_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_CONTEXT = PROJECT_DIR / "processed" / "country_spam2020_water_context.csv"
OUT_SUMMARY = PROJECT_DIR / "tables" / "spam2020_water_context_summary.csv"

SPAM_MEMBERS = {
    "total": "spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TA.csv",
    "irrigated": "spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TI.csv",
    "rainfed": "spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TR.csv",
}


def collapse_member(member: str, label: str) -> pd.DataFrame:
    with zipfile.ZipFile(SPAM_ZIP) as zf:
        with zf.open(member) as handle:
            header = pd.read_csv(handle, nrows=0)
        crop_cols = list(header.columns[9:])

    pieces = []
    with zipfile.ZipFile(SPAM_ZIP) as zf:
        with zf.open(member) as handle:
            reader = pd.read_csv(handle, chunksize=250_000)
            for i, chunk in enumerate(reader, start=1):
                harvested_area = chunk[crop_cols].sum(axis=1)
                keep = harvested_area > 0
                if not keep.any():
                    continue
                work = chunk.loc[keep, ["FIPS0", "ADM0_NAME"]].copy()
                work[f"spam2020_{label}_harvested_area"] = harvested_area.loc[keep].to_numpy(dtype=float)
                pieces.append(
                    work.groupby(["FIPS0", "ADM0_NAME"], as_index=False)[
                        f"spam2020_{label}_harvested_area"
                    ].sum()
                )
                if i % 15 == 0:
                    print(f"processed {label} chunk {i}", flush=True)
    if not pieces:
        return pd.DataFrame(columns=["FIPS0", "ADM0_NAME", f"spam2020_{label}_harvested_area"])
    return (
        pd.concat(pieces, ignore_index=True)
        .groupby(["FIPS0", "ADM0_NAME"], as_index=False)[f"spam2020_{label}_harvested_area"]
        .sum()
    )


def main() -> int:
    main_areas = pd.read_csv(MAIN_SAMPLE_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    fips_to_m49 = load_fips_to_m49()

    merged: pd.DataFrame | None = None
    for label, member in SPAM_MEMBERS.items():
        collapsed = collapse_member(member, label)
        if merged is None:
            merged = collapsed
        else:
            merged = merged.merge(collapsed, on=["FIPS0", "ADM0_NAME"], how="outer")

    if merged is None:
        raise RuntimeError("No SPAM water-context data were read.")

    merged["area_code_m49"] = merged["FIPS0"].map(fips_to_m49)
    merged = merged.merge(main_areas, on="area_code_m49", how="left")
    for col in [
        "spam2020_total_harvested_area",
        "spam2020_irrigated_harvested_area",
        "spam2020_rainfed_harvested_area",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    denom = merged["spam2020_total_harvested_area"].replace(0, np.nan)
    merged["spam2020_irrigated_share"] = merged["spam2020_irrigated_harvested_area"] / denom
    merged["spam2020_rainfed_share"] = merged["spam2020_rainfed_harvested_area"] / denom
    merged["spam2020_irrigated_plus_rainfed_share"] = (
        merged["spam2020_irrigated_harvested_area"] + merged["spam2020_rainfed_harvested_area"]
    ) / denom
    merged["in_main_sample"] = merged["area"].notna()
    merged = merged.sort_values(["in_main_sample", "area"], ascending=[False, True])

    matched = merged[merged["in_main_sample"]].copy()
    summary = pd.DataFrame(
        [
            {
                "n_spam_countries": len(merged),
                "n_main_sample_matched": matched["area"].nunique(),
                "median_irrigated_share": matched["spam2020_irrigated_share"].median(),
                "median_rainfed_share": matched["spam2020_rainfed_share"].median(),
                "p25_irrigated_share": matched["spam2020_irrigated_share"].quantile(0.25),
                "p75_irrigated_share": matched["spam2020_irrigated_share"].quantile(0.75),
                "mean_irrigated_plus_rainfed_share": matched[
                    "spam2020_irrigated_plus_rainfed_share"
                ].mean(),
            }
        ]
    )

    OUT_CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CONTEXT, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_CONTEXT} ({len(merged):,} rows)")
    print(f"wrote {OUT_SUMMARY}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
