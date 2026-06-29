"""Mechanism and planning diagnostics for Cell Reports Sustainability revision.

This phase strengthens two reviewer-facing elements:
1. drought-year mechanism diagnostics based on crop-response asynchrony and
   the harvested-area share of crops with negative anomalies;
2. a country-level crop-portfolio drought-risk diagnostic combining recent
   dominance, moderate drought frequency and observed drought loss contrast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1i_spam_weighted_spei_model_panel.csv"
ASYNC_PATH = PROJECT_DIR / "processed" / "country_year_crop_response_asynchrony_metrics.csv"
OUT_MECH_SUMMARY = PROJECT_DIR / "tables" / "phase1n_drought_mechanism_summary.csv"
OUT_MECH_CORR = PROJECT_DIR / "tables" / "phase1n_drought_mechanism_correlations.csv"
OUT_COUNTRY_DIAG = PROJECT_DIR / "tables" / "phase1n_country_portfolio_risk_diagnostic.csv"
OUT_COUNTRY_SUMMARY = PROJECT_DIR / "tables" / "phase1n_country_portfolio_risk_summary.csv"


def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def classify_diversity(df: pd.DataFrame) -> pd.Series:
    return pd.qcut(
        df["shannon_crop_area"].rank(method="first"),
        3,
        labels=["Low diversity", "Medium diversity", "High diversity"],
    )


def build_mechanism(panel: pd.DataFrame, async_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "area_code_m49",
        "area",
        "year",
        "shannon_crop_area",
        "top_crop_share",
        "production_log_anomaly",
        "spei12_spam_weighted_mean_annual",
        "has_spam_spei12",
    ]
    work = panel[cols].merge(async_df, on=["area_code_m49", "area", "year"], how="inner")
    work = work[work["has_spam_spei12"]].copy()
    work["moderate_drought"] = work["spei12_spam_weighted_mean_annual"] < -0.5
    work["drought_status"] = np.where(work["moderate_drought"], "Moderate drought", "Non-drought")
    work["diversity_tercile"] = classify_diversity(work)
    work["synchronized_negative_share"] = work["share_negative_crop_anomalies"]
    work["high_synchronized_negative_event"] = work["synchronized_negative_share"] > 0.60

    summary = (
        work.groupby(["drought_status", "diversity_tercile"], observed=True)
        .agg(
            n=("area", "size"),
            n_areas=("area", "nunique"),
            mean_asynchrony=("crop_response_asynchrony_index", "mean"),
            se_asynchrony=("crop_response_asynchrony_index", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            mean_synchronized_negative_share=("synchronized_negative_share", "mean"),
            se_synchronized_negative_share=(
                "synchronized_negative_share",
                lambda s: s.std(ddof=1) / np.sqrt(len(s)),
            ),
            mean_high_synchronized_negative_probability=("high_synchronized_negative_event", "mean"),
            se_high_synchronized_negative_probability=(
                "high_synchronized_negative_event",
                lambda s: s.std(ddof=1) / np.sqrt(len(s)),
            ),
            mean_production_log_anomaly=("production_log_anomaly", "mean"),
        )
        .reset_index()
    )
    for metric in ["asynchrony", "synchronized_negative_share", "high_synchronized_negative_probability"]:
        summary[f"ci_low_{metric}"] = summary[f"mean_{metric}"] - 1.96 * summary[f"se_{metric}"]
        summary[f"ci_high_{metric}"] = summary[f"mean_{metric}"] + 1.96 * summary[f"se_{metric}"]

    corr_rows = []
    for status, sub in work.groupby("drought_status"):
        for x, y in [
            ("shannon_crop_area", "crop_response_asynchrony_index"),
            ("top_crop_share", "crop_response_asynchrony_index"),
            ("shannon_crop_area", "synchronized_negative_share"),
            ("top_crop_share", "synchronized_negative_share"),
            ("synchronized_negative_share", "production_log_anomaly"),
        ]:
            g = sub[[x, y]].dropna()
            corr_rows.append(
                {
                    "drought_status": status,
                    "x": x,
                    "y": y,
                    "correlation": g[x].corr(g[y]) if len(g) >= 3 else np.nan,
                    "n": len(g),
                }
            )
    return summary, pd.DataFrame(corr_rows)


def build_country_diagnostic(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel[panel["has_spam_spei12"]].copy()
    work["moderate_drought"] = work["spei12_spam_weighted_mean_annual"] < -0.5
    recent = (
        work[work["year"].between(2015, 2024)]
        .groupby(["area_code_m49", "area"], as_index=False)
        .agg(
            recent_shannon=("shannon_crop_area", "mean"),
            recent_top_crop_share=("top_crop_share", "mean"),
            recent_harvested_area=("total_harvested_area", "mean"),
        )
    )
    drought = (
        work.groupby(["area_code_m49", "area"], as_index=False)
        .agg(
            drought_m05_frequency=("moderate_drought", "mean"),
            mean_production_anomaly=("production_log_anomaly", "mean"),
            drought_year_anomaly=("production_log_anomaly", lambda s: np.nan),
            n_years=("year", "nunique"),
            n_drought_years=("moderate_drought", "sum"),
        )
    )
    anomaly_rows = []
    for (code, area), sub in work.groupby(["area_code_m49", "area"], sort=False):
        drought_sub = sub[sub["moderate_drought"]]
        nondrought_sub = sub[~sub["moderate_drought"]]
        anomaly_rows.append(
            {
                "area_code_m49": code,
                "area": area,
                "mean_drought_anomaly": drought_sub["production_log_anomaly"].mean(),
                "mean_nondrought_anomaly": nondrought_sub["production_log_anomaly"].mean(),
                "drought_loss_contrast": (
                    nondrought_sub["production_log_anomaly"].mean()
                    - drought_sub["production_log_anomaly"].mean()
                    if len(drought_sub) >= 2 and len(nondrought_sub) >= 5
                    else np.nan
                ),
            }
        )
    drought = drought.drop(columns=["drought_year_anomaly"]).merge(
        pd.DataFrame(anomaly_rows), on=["area_code_m49", "area"], how="left"
    )
    out = recent.merge(drought, on=["area_code_m49", "area"], how="inner")
    out["dominance_risk_pct"] = pct_rank(out["recent_top_crop_share"])
    out["drought_frequency_pct"] = pct_rank(out["drought_m05_frequency"])
    out["drought_loss_pct"] = pct_rank(out["drought_loss_contrast"])
    out["portfolio_drought_risk_score"] = out[
        ["dominance_risk_pct", "drought_frequency_pct", "drought_loss_pct"]
    ].mean(axis=1)
    out["diagnostic_priority"] = pd.qcut(
        out["portfolio_drought_risk_score"].rank(method="first"),
        5,
        labels=["Very low", "Low", "Moderate", "High", "Very high"],
    )
    out = out.sort_values("portfolio_drought_risk_score", ascending=False)
    summary = (
        out.groupby("diagnostic_priority", observed=True)
        .agg(
            n_areas=("area", "nunique"),
            mean_recent_top_crop_share=("recent_top_crop_share", "mean"),
            mean_drought_frequency=("drought_m05_frequency", "mean"),
            mean_drought_loss_contrast=("drought_loss_contrast", "mean"),
        )
        .reset_index()
    )
    return out, summary


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, dtype={"area_code_m49": str})
    async_df = pd.read_csv(ASYNC_PATH, dtype={"area_code_m49": str})
    mech_summary, mech_corr = build_mechanism(panel, async_df)
    country_diag, country_summary = build_country_diagnostic(panel)

    OUT_MECH_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    mech_summary.to_csv(OUT_MECH_SUMMARY, index=False)
    mech_corr.to_csv(OUT_MECH_CORR, index=False)
    country_diag.to_csv(OUT_COUNTRY_DIAG, index=False)
    country_summary.to_csv(OUT_COUNTRY_SUMMARY, index=False)
    print(f"wrote {OUT_MECH_SUMMARY} ({len(mech_summary):,} rows)")
    print(f"wrote {OUT_MECH_CORR} ({len(mech_corr):,} rows)")
    print(f"wrote {OUT_COUNTRY_DIAG} ({len(country_diag):,} rows)")
    print(f"wrote {OUT_COUNTRY_SUMMARY}")
    print(country_diag.head(12)[["area", "recent_top_crop_share", "drought_m05_frequency", "drought_loss_contrast", "portfolio_drought_risk_score", "diagnostic_priority"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
