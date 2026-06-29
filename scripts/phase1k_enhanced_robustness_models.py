"""Enhanced robustness models for CRS submission.

Adds three reviewer-facing checks:
1. country and year fixed effects plus country-specific linear trends;
2. pre-drought portfolio metrics using lagged and 3-year lagged moving-average diversity;
3. heterogeneity by baseline portfolio dominance and drought exposure frequency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1i_spam_weighted_spei_model_panel.csv"
WATER_CONTEXT_PATH = PROJECT_DIR / "processed" / "country_spam2020_water_context.csv"
OUT_RESULTS = PROJECT_DIR / "tables" / "phase1k_enhanced_robustness_results.csv"
OUT_KEY = PROJECT_DIR / "tables" / "phase1k_enhanced_robustness_key_terms.csv"
OUT_SUMMARY = PROJECT_DIR / "tables" / "phase1k_enhanced_robustness_summary.csv"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


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


def residualize_against_nuisance(work: pd.DataFrame, cols: list[str], include_country_trends: bool) -> pd.DataFrame:
    """Residualize columns against area FE, year FE and optionally area-specific linear trends."""
    area_dummies = pd.get_dummies(work["area"], prefix="area", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(work["year"], prefix="year", drop_first=True, dtype=float)
    blocks = [area_dummies, year_dummies]
    if include_country_trends:
        year_centered = work["year"].astype(float) - work["year"].mean()
        trend_cols = {}
        for area in sorted(work["area"].unique()):
            trend_cols[f"trend_{area}"] = (work["area"].eq(area).astype(float) * year_centered).to_numpy()
        blocks.append(pd.DataFrame(trend_cols, index=work.index))
    nuisance = pd.concat(blocks, axis=1).to_numpy(dtype=float)
    nuisance = np.column_stack([np.ones(len(work)), nuisance])
    out = pd.DataFrame(index=work.index)
    for col in cols:
        y = work[col].to_numpy(dtype=float)
        gamma, *_ = np.linalg.lstsq(nuisance, y, rcond=None)
        out[col] = y - nuisance @ gamma
    return out


def run_model(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    spec: str,
    sample: str,
    include_country_trends: bool = False,
) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regressors].dropna().copy()
    if len(work) < 200 or work["area"].nunique() < 20:
        return pd.DataFrame()
    resid = residualize_against_nuisance(work, [outcome] + regressors, include_country_trends)
    beta, se = ols_cluster(
        resid[outcome].to_numpy(dtype=float),
        resid[regressors].to_numpy(dtype=float),
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
                "country_trends": include_country_trends,
            }
            for term, est, stderr in zip(regressors, beta, se)
        ]
    )


def prepare_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH, dtype={"area_code_m49": str})
    df = df[df["has_spam_spei12"]].copy().sort_values(["area", "year"]).reset_index(drop=True)
    if WATER_CONTEXT_PATH.exists():
        water = pd.read_csv(WATER_CONTEXT_PATH, dtype={"area_code_m49": str})[
            [
                "area_code_m49",
                "spam2020_irrigated_share",
                "spam2020_rainfed_share",
                "spam2020_irrigated_plus_rainfed_share",
            ]
        ].drop_duplicates("area_code_m49")
        df = df.merge(water, on="area_code_m49", how="left")
    else:
        df["spam2020_irrigated_share"] = np.nan
        df["spam2020_rainfed_share"] = np.nan
        df["spam2020_irrigated_plus_rainfed_share"] = np.nan
    df["drought_m05"] = (df["spei12_spam_weighted_mean_annual"] < -0.5).astype(float)
    df["drought_m10"] = (df["spei12_spam_weighted_mean_annual"] < -1.0).astype(float)
    df["shannon_z"] = zscore(df["shannon_crop_area"])
    df["dominance_z"] = zscore(df["top_crop_share"])
    df["spei_z"] = zscore(df["spei12_spam_weighted_mean_annual"])
    df["log_harvested_area_z"] = zscore(np.log1p(df["total_harvested_area"]))
    df["cereal_share_z"] = zscore(df["share_cereals"])
    df["irrigated_share_z"] = zscore(df["spam2020_irrigated_share"])
    df["shannon_lag1"] = df.groupby("area")["shannon_crop_area"].shift(1)
    df["dominance_lag1"] = df.groupby("area")["top_crop_share"].shift(1)
    df["shannon_lag3ma"] = (
        df.groupby("area")["shannon_crop_area"]
        .shift(1)
        .rolling(3, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["dominance_lag3ma"] = (
        df.groupby("area")["top_crop_share"]
        .shift(1)
        .rolling(3, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    for col in ["shannon_lag1", "dominance_lag1", "shannon_lag3ma", "dominance_lag3ma"]:
        df[f"{col}_z"] = zscore(df[col])

    for metric in ["shannon_z", "dominance_z", "shannon_lag1_z", "dominance_lag1_z", "shannon_lag3ma_z", "dominance_lag3ma_z"]:
        for drought in ["drought_m05", "drought_m10"]:
            df[f"{metric}_x_{drought}"] = df[metric] * df[drought]
    df["shannon_z_x_spei"] = df["shannon_z"] * df["spei_z"]
    df["shannon_lag1_z_x_spei"] = df["shannon_lag1_z"] * df["spei_z"]
    df["shannon_lag3ma_z_x_spei"] = df["shannon_lag3ma_z"] * df["spei_z"]
    df["log_harvested_area_z_x_drought_m05"] = df["log_harvested_area_z"] * df["drought_m05"]
    df["cereal_share_z_x_drought_m05"] = df["cereal_share_z"] * df["drought_m05"]
    df["irrigated_share_z_x_drought_m05"] = df["irrigated_share_z"] * df["drought_m05"]

    baseline = df[df["year"].between(1992, 2001)].groupby("area").agg(
        baseline_dominance=("top_crop_share", "mean"),
        baseline_shannon=("shannon_crop_area", "mean"),
    )
    drought_freq = df.groupby("area")["drought_m05"].mean().rename("drought_m05_frequency")
    area_meta = baseline.join(drought_freq)
    area_meta["drought_m05_frequency_z"] = zscore(area_meta["drought_m05_frequency"])
    area_meta["baseline_dominance_group"] = pd.qcut(
        area_meta["baseline_dominance"], 2, labels=["low_baseline_dominance", "high_baseline_dominance"]
    )
    area_meta["drought_frequency_group"] = pd.qcut(
        area_meta["drought_m05_frequency"].rank(method="first"),
        2,
        labels=["lower_moderate_drought_frequency", "higher_moderate_drought_frequency"],
    )
    df = df.merge(area_meta.reset_index(), on="area", how="left")
    df["drought_m05_frequency_z_x_drought_m05"] = df["drought_m05_frequency_z"] * df["drought_m05"]
    valid_irrig = df.groupby("area")["spam2020_irrigated_share"].first().dropna()
    if valid_irrig.nunique() > 1:
        irrig_groups = pd.qcut(
            valid_irrig.rank(method="first"),
            2,
            labels=["lower_irrigated_share", "higher_irrigated_share"],
        ).rename("irrigated_share_group")
        df = df.merge(irrig_groups.reset_index(), on="area", how="left")
    else:
        df["irrigated_share_group"] = np.nan
    return df


def summarize_key_terms(key: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["enhancement", "outcome", "term_family", "drought_or_gradient"]
    return (
        key.groupby(group_cols, as_index=False)
        .agg(
            n_specs=("estimate", "size"),
            n_expected_direction=("supports_buffering", "sum"),
            share_expected_direction=("supports_buffering", "mean"),
            min_t=("t_stat", "min"),
            max_t=("t_stat", "max"),
            median_t=("t_stat", "median"),
            median_abs_t=("t_stat", lambda s: float(np.nanmedian(np.abs(s)))),
        )
        .sort_values(group_cols)
    )


def main() -> int:
    df = prepare_panel()
    rows: list[pd.DataFrame] = []
    outcome = "production_log_anomaly"

    # Enhancement 1: country-specific linear trends.
    trend_specs = [
        ("trend_shannon_m05", ["shannon_z", "drought_m05", "shannon_z_x_drought_m05"]),
        ("trend_shannon_m10", ["shannon_z", "drought_m10", "shannon_z_x_drought_m10"]),
        ("trend_dominance_m05", ["dominance_z", "drought_m05", "dominance_z_x_drought_m05"]),
        ("trend_dominance_m10", ["dominance_z", "drought_m10", "dominance_z_x_drought_m10"]),
        ("trend_shannon_continuous_spei", ["shannon_z", "spei_z", "shannon_z_x_spei"]),
    ]
    for spec, regs in trend_specs:
        rows.append(run_model(df, outcome, regs, spec, "all_spam_spei_areas", include_country_trends=True))

    # Enhancement 2: pre-drought portfolio metrics.
    predrought_specs = [
        ("lag1_shannon_m05", ["shannon_lag1_z", "drought_m05", "shannon_lag1_z_x_drought_m05"]),
        ("lag1_shannon_m10", ["shannon_lag1_z", "drought_m10", "shannon_lag1_z_x_drought_m10"]),
        ("lag1_dominance_m05", ["dominance_lag1_z", "drought_m05", "dominance_lag1_z_x_drought_m05"]),
        ("lag1_dominance_m10", ["dominance_lag1_z", "drought_m10", "dominance_lag1_z_x_drought_m10"]),
        ("lag3ma_shannon_m05", ["shannon_lag3ma_z", "drought_m05", "shannon_lag3ma_z_x_drought_m05"]),
        ("lag3ma_shannon_m10", ["shannon_lag3ma_z", "drought_m10", "shannon_lag3ma_z_x_drought_m10"]),
        ("lag3ma_dominance_m05", ["dominance_lag3ma_z", "drought_m05", "dominance_lag3ma_z_x_drought_m05"]),
        ("lag3ma_dominance_m10", ["dominance_lag3ma_z", "drought_m10", "dominance_lag3ma_z_x_drought_m10"]),
        ("lag1_shannon_continuous_spei", ["shannon_lag1_z", "spei_z", "shannon_lag1_z_x_spei"]),
        ("lag3ma_shannon_continuous_spei", ["shannon_lag3ma_z", "spei_z", "shannon_lag3ma_z_x_spei"]),
    ]
    for spec, regs in predrought_specs:
        rows.append(run_model(df, outcome, regs, spec, "all_spam_spei_areas", include_country_trends=False))

    # Enhancement 3: heterogeneity by country context.
    for group_col in ["baseline_dominance_group", "drought_frequency_group"]:
        for group_value, sample in df.groupby(group_col, observed=True):
            sample_name = f"{group_col}:{group_value}"
            rows.append(
                run_model(
                    sample,
                    outcome,
                    ["shannon_z", "drought_m05", "shannon_z_x_drought_m05"],
                    "heterogeneity_shannon_m05",
                    sample_name,
                )
            )
            rows.append(
                run_model(
                    sample,
                    outcome,
                    ["dominance_z", "drought_m05", "dominance_z_x_drought_m05"],
                    "heterogeneity_dominance_m05",
                    sample_name,
                )
            )
            rows.append(
                run_model(
                    sample,
                    outcome,
                    ["shannon_z", "spei_z", "shannon_z_x_spei"],
                    "heterogeneity_shannon_continuous_spei",
                    sample_name,
                )
            )

    # Enhancement 4: irrigated/rainfed water-context heterogeneity from SPAM H_TI/H_TR.
    if "irrigated_share_group" in df.columns:
        for group_value, sample in df.groupby("irrigated_share_group", observed=True):
            sample_name = f"irrigated_share_group:{group_value}"
            rows.append(
                run_model(
                    sample,
                    outcome,
                    ["shannon_z", "drought_m05", "shannon_z_x_drought_m05"],
                    "water_context_shannon_m05",
                    sample_name,
                )
            )
            rows.append(
                run_model(
                    sample,
                    outcome,
                    ["dominance_z", "drought_m05", "dominance_z_x_drought_m05"],
                    "water_context_dominance_m05",
                    sample_name,
                )
            )

    # Enhancement 5: conservative baseline comparison with structural drought interactions.
    baseline_controls = [
        "drought_m05",
        "log_harvested_area_z",
        "cereal_share_z",
        "log_harvested_area_z_x_drought_m05",
        "cereal_share_z_x_drought_m05",
        "irrigated_share_z_x_drought_m05",
        "drought_m05_frequency_z_x_drought_m05",
    ]
    rows.append(
        run_model(
            df,
            outcome,
            ["shannon_z", "shannon_z_x_drought_m05"] + baseline_controls,
            "baseline_comparison_shannon_m05",
            "all_spam_spei_areas",
        )
    )
    rows.append(
        run_model(
            df,
            outcome,
            ["dominance_z", "dominance_z_x_drought_m05"] + baseline_controls,
            "baseline_comparison_dominance_m05",
            "all_spam_spei_areas",
        )
    )

    results = pd.concat([r for r in rows if not r.empty], ignore_index=True)
    results["enhancement"] = np.select(
        [
            results["spec"].str.startswith("trend_"),
            results["spec"].str.startswith("lag"),
            results["spec"].str.startswith("heterogeneity_"),
            results["spec"].str.startswith("water_context_"),
            results["spec"].str.startswith("baseline_comparison_"),
        ],
        ["country_trends", "predrought_portfolio", "heterogeneity", "water_context", "baseline_comparison"],
        default="other",
    )
    results.to_csv(OUT_RESULTS, index=False)

    is_key = (
        results["term"].str.contains("_x_drought", regex=False)
        | results["term"].str.endswith("_x_spei")
    )
    key = results[is_key].copy()
    key["term_family"] = np.select(
        [
            key["term"].str.contains("dominance"),
            key["term"].str.contains("shannon"),
            key["term"].str.contains("irrigated"),
            key["term"].str.contains("cereal"),
            key["term"].str.contains("harvested_area"),
            key["term"].str.contains("drought_m05_frequency"),
        ],
        [
            "Top-crop dominance",
            "Shannon diversity",
            "Irrigated share",
            "Cereal share",
            "Harvested-area scale",
            "Drought frequency",
        ],
        default="Other",
    )
    key["drought_or_gradient"] = np.select(
        [
            key["term"].str.contains("drought_m05"),
            key["term"].str.contains("drought_m10"),
            key["term"].str.contains("_x_spei"),
        ],
        ["SPEI12 < -0.5", "SPEI12 < -1.0", "continuous SPEI"],
        default="other",
    )
    key["supports_buffering"] = np.nan
    shannon_mask = key["term"].str.contains("shannon")
    dominance_mask = key["term"].str.contains("dominance")
    spei_mask = key["term"].str.contains("_x_spei")
    key.loc[shannon_mask & spei_mask, "supports_buffering"] = key.loc[
        shannon_mask & spei_mask, "estimate"
    ].lt(0).astype(float)
    key.loc[shannon_mask & ~spei_mask, "supports_buffering"] = key.loc[
        shannon_mask & ~spei_mask, "estimate"
    ].gt(0).astype(float)
    key.loc[dominance_mask, "supports_buffering"] = key.loc[dominance_mask, "estimate"].lt(0).astype(float)
    key.to_csv(OUT_KEY, index=False)
    summarize_key_terms(key).to_csv(OUT_SUMMARY, index=False)

    print(f"wrote {OUT_RESULTS} ({len(results):,} rows)")
    print(f"wrote {OUT_KEY} ({len(key):,} key rows)")
    print(f"wrote {OUT_SUMMARY}")
    print(key[["enhancement", "sample", "spec", "term", "estimate", "cluster_se_area", "t_stat", "n", "n_areas", "supports_buffering"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
