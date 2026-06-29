"""Create compact result summaries for manuscript planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLES = PROJECT_DIR / "tables"
PROCESSED = PROJECT_DIR / "processed"


def fmt_estimate(estimate: float, se: float, t_stat: float) -> str:
    return f"{estimate:.3f} ({se:.3f}); t={t_stat:.2f}"


def summarize_main_effects() -> pd.DataFrame:
    point = pd.read_csv(TABLES / "phase1e_fixed_effects_model_results.csv")
    polygon = pd.read_csv(TABLES / "phase1g_polygon_spei_fixed_effects_results.csv")
    spam = pd.read_csv(TABLES / "phase1i_spam_weighted_spei_fixed_effects_results.csv")
    rows = []

    selections = [
        (
            "Point SPEI",
            point,
            "production_log_anomaly",
            "area_year_fe_shannon_drought",
            "shannon_z_x_drought",
            "Shannon diversity x drought",
        ),
        (
            "Point SPEI",
            point,
            "production_log_anomaly",
            "area_year_fe_dominance_drought",
            "dominance_z_x_drought",
            "Top-crop dominance x drought",
        ),
        (
            "Polygon SPEI",
            polygon,
            "production_log_anomaly",
            "poly_spei_drought_shannon",
            "shannon_z_x_drought_poly",
            "Shannon diversity x drought",
        ),
        (
            "Polygon SPEI",
            polygon,
            "production_log_anomaly",
            "poly_spei_drought_dominance",
            "dominance_z_x_drought_poly",
            "Top-crop dominance x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "production_log_anomaly",
            "spam_weighted_spei_drought_shannon",
            "shannon_z_x_drought_spam",
            "Shannon diversity x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "production_log_anomaly",
            "spam_weighted_spei_drought_dominance",
            "dominance_z_x_drought_spam",
            "Top-crop dominance x drought",
        ),
        (
            "SPAM harvested-area-weighted SPEI",
            spam,
            "production_log_anomaly",
            "spam_weighted_spei_continuous_shannon",
            "shannon_z_x_spei12_spam",
            "Shannon diversity x continuous SPEI",
        ),
    ]
    for exposure, df, outcome, spec, term, label in selections:
        hit = df[
            df["outcome"].eq(outcome) & df["spec"].eq(spec) & df["term"].eq(term)
        ].iloc[0]
        rows.append(
            {
                "evidence_block": "Drought buffering",
                "climate_exposure": exposure,
                "term_label": label,
                "outcome": outcome,
                "estimate": hit["estimate"],
                "cluster_se_area": hit["cluster_se_area"],
                "t_stat": hit["t_stat"],
                "n": int(hit["n"]),
                "n_areas": int(hit["n_areas"]),
                "formatted": fmt_estimate(hit["estimate"], hit["cluster_se_area"], hit["t_stat"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_robustness() -> pd.DataFrame:
    robust = pd.read_csv(TABLES / "phase1f_robustness_key_terms.csv")
    production = robust[robust["outcome"].eq("production_log_anomaly")].copy()
    production["term_family"] = production["term"].map(
        lambda x: "Shannon diversity" if str(x).startswith("shannon") else "Top-crop dominance"
    )
    summary = (
        production.groupby(["sample", "term_family"], as_index=False)
        .agg(
            n_specs=("term", "count"),
            n_expected_direction=("supports_buffering", "sum"),
            min_t=("t_stat", "min"),
            max_t=("t_stat", "max"),
            median_abs_t=("t_stat", lambda s: s.abs().median()),
        )
    )
    summary["share_expected_direction"] = summary["n_expected_direction"] / summary["n_specs"]
    return summary


def summarize_mechanism() -> pd.DataFrame:
    mech = pd.read_csv(TABLES / "phase1e_mechanism_diagnostics.csv")
    keep = mech[
        mech[["x", "y"]]
        .agg("::".join, axis=1)
        .isin(
            [
                "shannon_crop_area::crop_response_asynchrony_index",
                "top_crop_share::crop_response_asynchrony_index",
                "crop_response_asynchrony_index::rolling10_anomaly_sd",
                "shannon_crop_area::crop_response_asynchrony_index",
            ]
        )
    ].copy()
    keep["evidence_block"] = "Asynchrony mechanism"
    return keep[
        ["evidence_block", "subset", "x", "y", "correlation", "n"]
    ].sort_values(["subset", "x", "y"])


def summarize_coverage() -> pd.DataFrame:
    main = pd.read_csv(PROCESSED / "phase1d_main_analysis_sample_panel.csv")
    point = pd.read_csv(PROCESSED / "country_year_spei12_point_drought_metrics.csv")
    polygon = pd.read_csv(PROCESSED / "country_year_spei12_polygon_sample_drought_metrics.csv")
    spam = pd.read_csv(PROCESSED / "country_year_spei12_spam_harvested_area_weighted_metrics.csv")
    rows = [
        {
            "dataset": "Main FAOSTAT analysis sample",
            "rows": len(main),
            "n_areas": main["area"].nunique(),
            "year_min": int(main["year"].min()),
            "year_max": int(main["year"].max()),
        },
        {
            "dataset": "Point SPEI12 drought exposure",
            "rows": len(point),
            "n_areas": point["area"].nunique(),
            "year_min": int(point["year"].min()),
            "year_max": int(point["year"].max()),
        },
        {
            "dataset": "Polygon-sampled SPEI12 drought exposure",
            "rows": len(polygon),
            "n_areas": polygon["area"].nunique(),
            "year_min": int(polygon["year"].min()),
            "year_max": int(polygon["year"].max()),
        },
        {
            "dataset": "SPAM harvested-area-weighted SPEI12 drought exposure",
            "rows": len(spam),
            "n_areas": spam["area"].nunique(),
            "year_min": int(spam["year"].min()),
            "year_max": int(spam["year"].max()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    outputs = {
        "core_evidence_main_effects_summary.csv": summarize_main_effects(),
        "core_evidence_robustness_summary.csv": summarize_robustness(),
        "core_evidence_mechanism_summary.csv": summarize_mechanism(),
        "core_evidence_coverage_summary.csv": summarize_coverage(),
    }
    for name, df in outputs.items():
        path = TABLES / name
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df):,} rows)")
        print(df.to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
