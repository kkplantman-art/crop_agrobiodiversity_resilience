"""Nutrition-weighted production outcome models for CRS strengthening.

This script merges FBS-derived kcal/protein production anomalies into the
SPAM-weighted SPEI model panel and estimates the same fixed-effects interaction
models used for mass-based production anomalies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_PANEL = PROJECT_DIR / "processed" / "phase1i_spam_weighted_spei_model_panel.csv"
NUTRITION_PANEL = PROJECT_DIR / "processed" / "country_year_nutrition_weighted_production_metrics.csv"
OUT_PANEL = PROJECT_DIR / "processed" / "phase1l_nutrition_weighted_model_panel.csv"
OUT_RESULTS = PROJECT_DIR / "tables" / "phase1l_nutrition_weighted_outcome_results.csv"
OUT_KEY = PROJECT_DIR / "tables" / "phase1l_nutrition_weighted_key_terms.csv"
OUT_SUMMARY = PROJECT_DIR / "tables" / "phase1l_nutrition_weighted_summary.csv"
OUT_SAMPLE_SUMMARY = PROJECT_DIR / "tables" / "phase1l_nutrition_weighted_sample_summary.csv"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def two_way_demean(df: pd.DataFrame, cols: list[str], entity: str, time: str) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        grand = df[col].mean()
        out[col] = df[col] - df.groupby(entity)[col].transform("mean") - df.groupby(time)[col].transform("mean") + grand
    return out


def ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in pd.unique(clusters):
        mask = clusters == cluster
        score = x[mask, :].T @ residuals[mask]
        meat += np.outer(score, score)
    n = x.shape[0]
    k = x.shape[1]
    g = len(pd.unique(clusters))
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    vcov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(vcov))
    return beta, se


def run_model(df: pd.DataFrame, outcome: str, regs: list[str], spec: str, sample: str) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regs].dropna().copy()
    if len(work) < 200 or work["area"].nunique() < 20:
        return pd.DataFrame()
    dm = two_way_demean(work, [outcome] + regs, "area", "year")
    beta, se = ols_cluster(
        dm[outcome].to_numpy(dtype=float),
        dm[regs].to_numpy(dtype=float),
        work["area"].to_numpy(),
    )
    return pd.DataFrame(
        [
            {
                "sample": sample,
                "outcome": outcome,
                "spec": spec,
                "term": term,
                "estimate": est,
                "cluster_se_area": stderr,
                "t_stat": est / stderr if stderr > 0 else np.nan,
                "n": len(work),
                "n_areas": work["area"].nunique(),
            }
            for term, est, stderr in zip(regs, beta, se)
        ]
    )


def prepare_panel() -> pd.DataFrame:
    base = pd.read_csv(BASE_PANEL, dtype={"area_code_m49": str})
    nut = pd.read_csv(NUTRITION_PANEL, dtype={"area_code_m49": str})
    keep = [
        "area_code_m49",
        "area",
        "year",
        "nutrition_covered_crop_production",
        "nutrition_kcal_production",
        "nutrition_protein_kg_production",
        "nutrition_coverage_production_share",
        "log_kcal_production",
        "log_protein_production",
        "nutrition_kcal_log_anomaly",
        "nutrition_protein_log_anomaly",
    ]
    df = base.merge(nut[keep], on=["area_code_m49", "area", "year"], how="left")
    df["has_nutrition_outcomes"] = df["nutrition_kcal_log_anomaly"].notna() & df["nutrition_protein_log_anomaly"].notna()
    df["drought_m05"] = (df["spei12_spam_weighted_mean_annual"] < -0.5).astype(float)
    df["drought_m10"] = (df["spei12_spam_weighted_mean_annual"] < -1.0).astype(float)
    df["shannon_z"] = zscore(df["shannon_crop_area"])
    df["dominance_z"] = zscore(df["top_crop_share"])
    df["spei_z"] = zscore(df["spei12_spam_weighted_mean_annual"])
    df["shannon_z_x_drought_m05"] = df["shannon_z"] * df["drought_m05"]
    df["dominance_z_x_drought_m05"] = df["dominance_z"] * df["drought_m05"]
    df["shannon_z_x_drought_m10"] = df["shannon_z"] * df["drought_m10"]
    df["dominance_z_x_drought_m10"] = df["dominance_z"] * df["drought_m10"]
    df["shannon_z_x_spei"] = df["shannon_z"] * df["spei_z"]
    return df


def main() -> int:
    df = prepare_panel()
    df.to_csv(OUT_PANEL, index=False)
    model_df = df[(df["has_spam_spei12"]) & (df["has_nutrition_outcomes"]) & (df["nutrition_coverage_production_share"] >= 0.75)].copy()

    rows: list[pd.DataFrame] = []
    for outcome in ["nutrition_kcal_log_anomaly", "nutrition_protein_log_anomaly"]:
        rows.append(run_model(model_df, outcome, ["shannon_z", "drought_m05", "shannon_z_x_drought_m05"], "nutrition_m05_shannon", "coverage_ge_0_75"))
        rows.append(run_model(model_df, outcome, ["dominance_z", "drought_m05", "dominance_z_x_drought_m05"], "nutrition_m05_dominance", "coverage_ge_0_75"))
        rows.append(run_model(model_df, outcome, ["shannon_z", "drought_m10", "shannon_z_x_drought_m10"], "nutrition_m10_shannon", "coverage_ge_0_75"))
        rows.append(run_model(model_df, outcome, ["dominance_z", "drought_m10", "dominance_z_x_drought_m10"], "nutrition_m10_dominance", "coverage_ge_0_75"))
        rows.append(run_model(model_df, outcome, ["shannon_z", "spei_z", "shannon_z_x_spei"], "nutrition_continuous_spei_shannon", "coverage_ge_0_75"))

    results = pd.concat([r for r in rows if not r.empty], ignore_index=True)
    results.to_csv(OUT_RESULTS, index=False)

    key = results[
        results["term"].str.contains("_x_drought", regex=False) | results["term"].str.endswith("_x_spei")
    ].copy()
    key["term_family"] = np.select(
        [key["term"].str.contains("dominance"), key["term"].str.contains("shannon")],
        ["Top-crop dominance", "Shannon diversity"],
        default="Other",
    )
    key["drought_or_gradient"] = np.select(
        [key["term"].str.contains("drought_m05"), key["term"].str.contains("drought_m10"), key["term"].str.contains("_x_spei")],
        ["SPEI12 < -0.5", "SPEI12 < -1.0", "continuous SPEI"],
        default="other",
    )
    key["supports_buffering"] = np.where(
        key["term"].str.contains("_x_spei"),
        key["estimate"] < 0,
        np.where(key["term"].str.contains("dominance"), key["estimate"] < 0, key["estimate"] > 0),
    )
    key.to_csv(OUT_KEY, index=False)

    summary = (
        key.groupby(["outcome", "term_family", "drought_or_gradient"], as_index=False)
        .agg(
            estimate=("estimate", "first"),
            cluster_se_area=("cluster_se_area", "first"),
            t_stat=("t_stat", "first"),
            n=("n", "first"),
            n_areas=("n_areas", "first"),
            supports_buffering=("supports_buffering", "first"),
        )
        .sort_values(["outcome", "term_family", "drought_or_gradient"])
    )
    sample_summary = pd.DataFrame(
        [
            {"metric": "rows_in_merged_panel", "value": len(df)},
            {"metric": "rows_with_spam_spei_nutrition_and_coverage_ge_0_75", "value": len(model_df)},
            {"metric": "areas_with_spam_spei_nutrition_and_coverage_ge_0_75", "value": model_df["area"].nunique()},
            {"metric": "median_nutrition_coverage_in_model_sample", "value": model_df["nutrition_coverage_production_share"].median()},
        ]
    )
    sample_summary.to_csv(OUT_SAMPLE_SUMMARY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(sample_summary.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"wrote {OUT_PANEL}")
    print(f"wrote {OUT_RESULTS}")
    print(f"wrote {OUT_KEY}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_SAMPLE_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
