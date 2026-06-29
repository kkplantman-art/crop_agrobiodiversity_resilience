"""Build SPAM 2020 harvested-area weights on the SPEI12 0.5-degree grid.

SPAM 2020 V2R2 harvested-area CSVs are 5 arc-minute crop pixels. For climate
exposure we first collapse all crop harvested area within each country to the
coarser SPEIbase 0.5-degree pixel containing each SPAM pixel.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw"
SPAM_ZIP = RAW_DIR / "spam2020" / "spam2020V2r2_global_harvested_area.csv.zip"
SPAM_MEMBER = "spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TA.csv"
NE_PATH = RAW_DIR / "natural_earth_admin0_110m.geojson"
MAIN_SAMPLE_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
OUT_WEIGHTS = PROJECT_DIR / "processed" / "spam2020_H_TA_spei12_pixel_weights.csv"
OUT_MATCH = PROJECT_DIR / "tables" / "spam2020_country_weight_match_summary.csv"

LON_MIN, LON_MAX, LON_NUM = -179.75, 179.75, 720
LAT_MIN, LAT_MAX, LAT_NUM = -89.75, 89.75, 360
DX = (LON_MAX - LON_MIN) / (LON_NUM - 1)
DY = (LAT_MAX - LAT_MIN) / (LAT_NUM - 1)


def locate_spei_pixel(lon: pd.Series, lat: pd.Series) -> pd.DataFrame:
    lon_index = np.rint((lon.to_numpy(dtype=float) - LON_MIN) / DX).astype(int)
    lat_index = np.rint((lat.to_numpy(dtype=float) - LAT_MIN) / DY).astype(int)
    lon_index = np.clip(lon_index, 0, LON_NUM - 1)
    lat_index = np.clip(lat_index, 0, LAT_NUM - 1)
    out = pd.DataFrame(
        {
            "spei_lon_index": lon_index,
            "spei_lat_index": lat_index,
        }
    )
    out["spei_chunk_index"] = out["spei_lon_index"] + out["spei_lat_index"] * LON_NUM
    out["spei_pixel_lon"] = out["spei_lon_index"] * DX + LON_MIN
    out["spei_pixel_lat"] = out["spei_lat_index"] * DY + LAT_MIN
    return out


def load_fips_to_m49() -> dict[str, str]:
    geo = json.loads(NE_PATH.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for feature in geo["features"]:
        props = feature["properties"]
        fips = str(props.get("FIPS_10", "")).strip()
        iso_n3 = str(props.get("ISO_N3", "")).strip()
        if fips and fips != "-99" and iso_n3 and iso_n3 != "-99":
            mapping[fips] = "'" + iso_n3.zfill(3)

    # SPAM uses a few FIPS-style country codes that need harmonization for
    # FAOSTAT/Natural Earth M49 matching.
    manual = {
        "CH": "'156",  # China
        "RS": "'643",  # Russian Federation
        "UP": "'804",  # Ukraine
        "TU": "'792",  # Turkey
        "NI": "'566",  # Nigeria
        "NG": "'562",  # Niger
        "BG": "'050",  # Bangladesh
        "BM": "'104",  # Myanmar
        "TZ": "'834",  # United Republic of Tanzania
        "IV": "'384",  # Cote d'Ivoire
        "VM": "'704",  # Viet Nam
        "CG": "'180",  # Democratic Republic of the Congo
        "IR": "'364",  # Iran (Islamic Republic of)
        "SP": "'724",  # Spain
        "UK": "'826",  # United Kingdom
        "KS": "'410",  # Republic of Korea
        "KN": "'408",  # Democratic People's Republic of Korea
        "LA": "'418",  # Lao People's Democratic Republic
        "SY": "'760",  # Syrian Arab Republic
        "EZ": "'203",  # Czechia
        "LO": "'703",  # Slovakia
        "RO": "'642",  # Romania
        "BU": "'100",  # Bulgaria
        "HR": "'191",  # Croatia
        "MJ": "'499",  # Montenegro
        "MK": "'807",  # North Macedonia
        "RI": "'688",  # Serbia
        "BK": "'070",  # Bosnia and Herzegovina
        "KV": "'383",  # Kosovo, not in FAOSTAT main sample
        "WA": "'516",  # Namibia
        "WZ": "'748",  # Eswatini
        "LT": "'426",  # Lesotho
        "AO": "'024",  # Angola
        "BY": "'112",  # Belarus
        "MD": "'498",  # Republic of Moldova
        "TX": "'795",  # Turkmenistan
        "UZ": "'860",  # Uzbekistan
        "KG": "'417",  # Kyrgyzstan
        "TI": "'762",  # Tajikistan
        "GG": "'268",  # Georgia
        "AM": "'051",  # Armenia
        "AJ": "'031",  # Azerbaijan
        "IZ": "'368",  # Iraq
        "YM": "'887",  # Yemen
        "JO": "'400",  # Jordan
        "LE": "'422",  # Lebanon
        "IS": "'376",  # Israel
        "WE": "'275",  # Palestine
        "CE": "'144",  # Sri Lanka
        "NP": "'524",  # Nepal
        "BT": "'064",  # Bhutan
        "CB": "'116",  # Cambodia
        "RP": "'608",  # Philippines
        "MY": "'458",  # Malaysia
        "BX": "'096",  # Brunei Darussalam
        "TT": "'626",  # Timor-Leste
        "PO": "'620",  # Portugal
        "GR": "'300",  # Greece
        "AL": "'008",  # Albania
        "SI": "'705",  # Slovenia
        "HU": "'348",  # Hungary
        "PL": "'616",  # Poland
        "LG": "'428",  # Latvia
        "LH": "'440",  # Lithuania
        "EN": "'233",  # Estonia
        "FR": "'250",  # France
        "NO": "'578",  # Norway
        "TG": "'768",  # Togo
        "TC": "'784",  # United Arab Emirates
        "MP": "'480",  # Mauritius
        "TP": "'678",  # Sao Tome and Principe
        "DO": "'212",  # Dominica
        "PQ": "'630",  # Puerto Rico
        "VC": "'670",  # Saint Vincent and the Grenadines
        "GJ": "'308",  # Grenada
        "BB": "'052",  # Barbados
        "ST": "'662",  # Saint Lucia
        "BA": "'048",  # Bahrain
        "AC": "'028",  # Antigua and Barbuda
        "SC": "'659",  # Saint Kitts and Nevis
        "MT": "'470",  # Malta
        "WE": "'275",  # Palestine
    }
    mapping.update(manual)
    return mapping


def main() -> int:
    main_areas = pd.read_csv(MAIN_SAMPLE_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area"]
    ].drop_duplicates()
    fips_to_m49 = load_fips_to_m49()

    with zipfile.ZipFile(SPAM_ZIP) as zf:
        with zf.open(SPAM_MEMBER) as handle:
            header = pd.read_csv(handle, nrows=0)
        crop_cols = list(header.columns[9:])

    grouped_parts = []
    country_parts = []
    with zipfile.ZipFile(SPAM_ZIP) as zf:
        with zf.open(SPAM_MEMBER) as handle:
            reader = pd.read_csv(handle, chunksize=200_000)
            for i, chunk in enumerate(reader, start=1):
                harvested_area = chunk[crop_cols].sum(axis=1)
                keep = harvested_area > 0
                if not keep.any():
                    continue
                work = chunk.loc[keep, ["FIPS0", "ADM0_NAME", "x", "y"]].copy()
                work["harvested_area"] = harvested_area.loc[keep].to_numpy(dtype=float)
                spei = locate_spei_pixel(work["x"], work["y"])
                work = pd.concat([work.reset_index(drop=True), spei], axis=1)
                grouped_parts.append(
                    work.groupby(
                        [
                            "FIPS0",
                            "ADM0_NAME",
                            "spei_lon_index",
                            "spei_lat_index",
                            "spei_chunk_index",
                            "spei_pixel_lon",
                            "spei_pixel_lat",
                        ],
                        as_index=False,
                    )["harvested_area"].sum()
                )
                country_parts.append(
                    work.groupby(["FIPS0", "ADM0_NAME"], as_index=False)["harvested_area"].sum()
                )
                if i % 10 == 0:
                    print(f"processed {i} SPAM chunks", flush=True)

    weights = (
        pd.concat(grouped_parts, ignore_index=True)
        .groupby(
            [
                "FIPS0",
                "ADM0_NAME",
                "spei_lon_index",
                "spei_lat_index",
                "spei_chunk_index",
                "spei_pixel_lon",
                "spei_pixel_lat",
            ],
            as_index=False,
        )["harvested_area"]
        .sum()
    )
    country = (
        pd.concat(country_parts, ignore_index=True)
        .groupby(["FIPS0", "ADM0_NAME"], as_index=False)["harvested_area"]
        .sum()
    )
    country["area_code_m49"] = country["FIPS0"].map(fips_to_m49)
    country = country.merge(main_areas, on="area_code_m49", how="left")
    country["in_main_sample"] = country["area"].notna()

    weights["area_code_m49"] = weights["FIPS0"].map(fips_to_m49)
    weights = weights.dropna(subset=["area_code_m49"]).merge(
        main_areas, on="area_code_m49", how="inner"
    )
    totals = weights.groupby("area_code_m49")["harvested_area"].transform("sum")
    weights["country_spam_harvested_area"] = totals
    weights["weight_share"] = weights["harvested_area"] / totals
    weights = weights[
        [
            "area_code_m49",
            "area",
            "FIPS0",
            "ADM0_NAME",
            "spei_lon_index",
            "spei_lat_index",
            "spei_chunk_index",
            "spei_pixel_lon",
            "spei_pixel_lat",
            "harvested_area",
            "country_spam_harvested_area",
            "weight_share",
        ]
    ].sort_values(["area", "spei_chunk_index"])

    country.to_csv(OUT_MATCH, index=False)
    weights.to_csv(OUT_WEIGHTS, index=False)
    print(
        f"wrote {OUT_WEIGHTS} ({len(weights):,} country-SPEI pixels; "
        f"{weights['area'].nunique():,} main-sample areas)"
    )
    print(f"wrote {OUT_MATCH} ({len(country):,} SPAM countries)")
    print(
        country.groupby("in_main_sample")["FIPS0"]
        .count()
        .rename("n_spam_countries")
        .to_string()
    )
    missing = country[country["area_code_m49"].isna() | ~country["in_main_sample"]].head(30)
    if not missing.empty:
        print("unmatched or outside main sample examples:")
        print(missing.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
