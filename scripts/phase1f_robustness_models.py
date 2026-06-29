"""Phase 1F robustness diagnostics.

Robustness families:
- Current-year vs lagged crop diversity.
- Alternative SPEI12 drought thresholds.
- Excluding the largest crop producers.
- Excluding small-production countries.
- Alternative stability outcome definitions.

All models use area and year fixed effects via two-way demeaning and
area-clustered standard errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1f_robustness_model_results.csv"
SUMMARY_PATH = PROJECT_DIR / "tables" / "phase1f_robustness_key_terms.csv"

LARGE_PRODUCERS = {
    "China, mainland",
    "India",
    "United States of America",
    "Brazil",
}


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


def run_fe(df: pd.DataFrame, outcome: str, regressors: list[str], spec: str, sample: str) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regressors].dropna().copy()
    if work["area"].nunique() < 20 or len(work) < 200:
        return pd.DataFrame()
    dm = two_way_demean(work, [outcome] + regressors, "area", "year")
    y = dm[outcome].to_numpy(dtype=float)
    x = dm[regressors].to_numpy(dtype=float)
    beta, se = ols_cluster(y, x, work["area"].to_numpy())
    rows = []
    for term, estimate, stderr in zip(regressors, beta, se):
        rows.append(
            {
                "sample": sample,
                "outcome": outcome,
                "spec": spec,
                "term": term,
                "estimate": estimate,
                "cluster_se_area": stderr,
                "t_stat": estimate / stderr if stderr > 0 else np.nan,
                "n": len(work),
                "n_areas": work["area"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def prepare_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    df = df[df["has_spei12"]].copy()
    df = df.sort_values(["area", "year"])
    for col in ["shannon_crop_area", "effective_number_crops", "top_crop_share", "spei12_annual_mean"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    df["shannon_lag1"] = df.groupby("area")["shannon_crop_area"].shift(1)
    df["top_crop_share_lag1"] = df.groupby("area")["top_crop_share"].shift(1)
    for col in ["shannon_lag1", "top_crop_share_lag1"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    thresholds = {
        "drought_mean_lt_minus0_5": df["spei12_annual_mean"] < -0.5,
        "drought_mean_lt_minus1_0": df["spei12_annual_mean"] < -1.0,
        "drought_mean_lt_minus1_5": df["spei12_annual_mean"] < -1.5,
        "drought_min_lt_minus1_5": df["spei12_min_month"] < -1.5,
    }
    for name, values in thresholds.items():
        df[name] = values.astype(float)
        df[f"shannon_z_x_{name}"] = df["shannon_crop_area_z"] * df[name]
        df[f"dominance_z_x_{name}"] = df["top_crop_share_z"] * df[name]
        df[f"shannon_lag1_z_x_{name}"] = df["shannon_lag1_z"] * df[name]
    return df


def sample_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mean_prod = df.groupby("area")["total_crop_production"].mean()
    small_cutoff = mean_prod.quantile(0.10)
    return {
        "all_spei_areas": df,
        "exclude_top4_producers": df[~df["area"].isin(LARGE_PRODUCERS)].copy(),
        "exclude_smallest_10pct_producers": df[df["area"].map(mean_prod) > small_cutoff].copy(),
        "complete_1992_2024_spei": df.groupby("area").filter(lambda g: g["spei12_annual_mean"].notna().sum() == 33).copy(),
    }


def main() -> int:
    df = prepare_panel()
    rows = []
    outcomes = ["production_log_anomaly", "rolling10_anomaly_sd", "rolling10_production_cv"]
    drought_vars = [
        "drought_mean_lt_minus0_5",
        "drought_mean_lt_minus1_0",
        "drought_mean_lt_minus1_5",
        "drought_min_lt_minus1_5",
    ]
    for sample_name, sample in sample_frames(df).items():
        for outcome in outcomes:
            for drought in drought_vars:
                rows.append(
                    run_fe(
                        sample,
                        outcome,
                        ["shannon_crop_area_z", drought, f"shannon_z_x_{drought}"],
                        f"current_shannon_x_{drought}",
                        sample_name,
                    )
                )
                rows.append(
                    run_fe(
                        sample,
                        outcome,
                        ["top_crop_share_z", drought, f"dominance_z_x_{drought}"],
                        f"dominance_x_{drought}",
                        sample_name,
                    )
                )
            rows.append(
                run_fe(
                    sample,
                    outcome,
                    ["shannon_lag1_z", "drought_mean_lt_minus1_0", "shannon_lag1_z_x_drought_mean_lt_minus1_0"],
                    "lag1_shannon_x_drought_mean_lt_minus1_0",
                    sample_name,
                )
            )
    out = pd.concat([r for r in rows if not r.empty], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)

    key_mask = out["term"].str.contains("_x_drought", regex=False) | out["term"].str.contains("_x_drought_", regex=False)
    key = out[key_mask].copy()
    key["supports_buffering"] = np.where(
        key["term"].str.contains("shannon"),
        key["estimate"] > 0,
        np.where(key["term"].str.contains("dominance"), key["estimate"] < 0, np.nan),
    )
    key.to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(f"wrote {SUMMARY_PATH} ({len(key):,} key interaction rows)")
    print(key[["sample", "outcome", "spec", "term", "estimate", "cluster_se_area", "t_stat", "n", "n_areas", "supports_buffering"]].head(60).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
