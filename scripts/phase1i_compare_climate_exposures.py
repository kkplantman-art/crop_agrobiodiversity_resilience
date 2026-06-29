"""Compare point, polygon and SPAM-weighted SPEI exposure diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_DIR / "processed"
TABLES = PROJECT_DIR / "tables"

POINT_PATH = PROCESSED / "country_year_spei12_point_drought_metrics.csv"
POLY_PATH = PROCESSED / "country_year_spei12_polygon_sample_drought_metrics.csv"
SPAM_PATH = PROCESSED / "country_year_spei12_spam_harvested_area_weighted_metrics.csv"
POINT_RESULTS = TABLES / "phase1e_fixed_effects_model_results.csv"
POLY_RESULTS = TABLES / "phase1g_polygon_spei_fixed_effects_results.csv"
SPAM_RESULTS = TABLES / "phase1i_spam_weighted_spei_fixed_effects_results.csv"
OUT_COMPARE = TABLES / "phase1i_climate_exposure_comparison.csv"
OUT_MODEL = TABLES / "phase1i_main_effects_by_climate_exposure.csv"


def exposure_comparison() -> pd.DataFrame:
    point = pd.read_csv(POINT_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area", "year", "spei12_annual_mean", "drought_spei12_mean_lt_minus1"]
    ].rename(
        columns={
            "spei12_annual_mean": "spei_point",
            "drought_spei12_mean_lt_minus1": "drought_point",
        }
    )
    poly = pd.read_csv(POLY_PATH, dtype={"area_code_m49": str})[
        ["area_code_m49", "area", "year", "spei12_poly_mean_annual", "drought_poly_mean_lt_minus1"]
    ].rename(
        columns={
            "spei12_poly_mean_annual": "spei_polygon",
            "drought_poly_mean_lt_minus1": "drought_polygon",
        }
    )
    spam = pd.read_csv(SPAM_PATH, dtype={"area_code_m49": str})[
        [
            "area_code_m49",
            "area",
            "year",
            "spei12_spam_weighted_mean_annual",
            "drought_spam_mean_lt_minus1",
            "n_spam_spei_pixels",
        ]
    ].rename(
        columns={
            "spei12_spam_weighted_mean_annual": "spei_spam_weighted",
            "drought_spam_mean_lt_minus1": "drought_spam_weighted",
        }
    )
    merged = point.merge(poly, on=["area_code_m49", "area", "year"], how="outer").merge(
        spam, on=["area_code_m49", "area", "year"], how="outer"
    )

    rows = []
    pairs = [
        ("point_vs_polygon", "spei_point", "spei_polygon", "drought_point", "drought_polygon"),
        (
            "point_vs_spam_weighted",
            "spei_point",
            "spei_spam_weighted",
            "drought_point",
            "drought_spam_weighted",
        ),
        (
            "polygon_vs_spam_weighted",
            "spei_polygon",
            "spei_spam_weighted",
            "drought_polygon",
            "drought_spam_weighted",
        ),
    ]
    for label, x, y, dx, dy in pairs:
        work = merged[[x, y, dx, dy, "area"]].dropna().copy()
        rows.append(
            {
                "comparison": label,
                "n_area_years": len(work),
                "n_areas": work["area"].nunique(),
                "spei_correlation": work[x].corr(work[y]),
                "mean_abs_spei_difference": (work[x] - work[y]).abs().mean(),
                "drought_agreement_share": (work[dx].astype(bool) == work[dy].astype(bool)).mean(),
                "drought_both_count": int((work[dx].astype(bool) & work[dy].astype(bool)).sum()),
                "drought_x_only_count": int((work[dx].astype(bool) & ~work[dy].astype(bool)).sum()),
                "drought_y_only_count": int((~work[dx].astype(bool) & work[dy].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_term(df: pd.DataFrame, outcome: str, spec: str, term: str) -> pd.Series:
    return df[df["outcome"].eq(outcome) & df["spec"].eq(spec) & df["term"].eq(term)].iloc[0]


def model_comparison() -> pd.DataFrame:
    point = pd.read_csv(POINT_RESULTS)
    poly = pd.read_csv(POLY_RESULTS)
    spam = pd.read_csv(SPAM_RESULTS)
    selections = [
        (
            "Point SPEI",
            point,
            "area_year_fe_shannon_drought",
            "shannon_z_x_drought",
            "Shannon diversity x drought",
        ),
        (
            "Point SPEI",
            point,
            "area_year_fe_dominance_drought",
            "dominance_z_x_drought",
            "Top-crop dominance x drought",
        ),
        (
            "Polygon SPEI",
            poly,
            "poly_spei_drought_shannon",
            "shannon_z_x_drought_poly",
            "Shannon diversity x drought",
        ),
        (
            "Polygon SPEI",
            poly,
            "poly_spei_drought_dominance",
            "dominance_z_x_drought_poly",
            "Top-crop dominance x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "spam_weighted_spei_drought_shannon",
            "shannon_z_x_drought_spam",
            "Shannon diversity x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "spam_weighted_spei_drought_dominance",
            "dominance_z_x_drought_spam",
            "Top-crop dominance x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "spam_weighted_spei_continuous_shannon",
            "shannon_z_x_spei12_spam",
            "Shannon diversity x continuous SPEI",
        ),
    ]
    rows = []
    for exposure, df, spec, term, label in selections:
        hit = select_term(df, "production_log_anomaly", spec, term)
        rows.append(
            {
                "climate_exposure": exposure,
                "term_label": label,
                "estimate": hit["estimate"],
                "cluster_se_area": hit["cluster_se_area"],
                "t_stat": hit["t_stat"],
                "n": int(hit["n"]),
                "n_areas": int(hit["n_areas"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    exposure = exposure_comparison()
    model = model_comparison()
    exposure.to_csv(OUT_COMPARE, index=False)
    model.to_csv(OUT_MODEL, index=False)
    print(f"wrote {OUT_COMPARE} ({len(exposure):,} rows)")
    print(exposure.to_string(index=False))
    print(f"wrote {OUT_MODEL} ({len(model):,} rows)")
    print(model.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
