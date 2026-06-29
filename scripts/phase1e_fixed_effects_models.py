"""Phase 1E fixed-effects diagnostics with clustered standard errors.

These models are still diagnostic, but they are closer to a paper-ready
specification than the pooled Phase 1D triage:

  outcome_it = beta1 diversity_it + beta2 drought_it
             + beta3 diversity_it x drought_it + area FE + year FE + error_it

Standard errors are clustered by area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1e_fixed_effects_model_results.csv"


def two_way_demean(df: pd.DataFrame, cols: list[str], entity: str, time: str) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        grand = df[col].mean()
        entity_mean = df.groupby(entity)[col].transform("mean")
        time_mean = df.groupby(time)[col].transform("mean")
        out[col] = df[col] - entity_mean - time_mean + grand
    return out


def ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in pd.unique(clusters):
        mask = clusters == cluster
        xg = x[mask, :]
        ug = residuals[mask]
        score = xg.T @ ug
        meat += np.outer(score, score)
    n = x.shape[0]
    k = x.shape[1]
    g = len(pd.unique(clusters))
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    vcov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(vcov))
    return beta, se


def run_model(df: pd.DataFrame, outcome: str, regressors: list[str], spec: str) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regressors].dropna().copy()
    dm = two_way_demean(work, [outcome] + regressors, entity="area", time="year")
    y = dm[outcome].to_numpy(dtype=float)
    x = dm[regressors].to_numpy(dtype=float)
    beta, se = ols_cluster(y, x, work["area"].to_numpy())
    rows = []
    for term, estimate, stderr in zip(regressors, beta, se):
        rows.append(
            {
                "outcome": outcome,
                "spec": spec,
                "term": term,
                "estimate": estimate,
                "cluster_se_area": stderr,
                "t_stat": estimate / stderr if stderr > 0 else np.nan,
                "n": len(work),
                "n_areas": work["area"].nunique(),
                "year_min": int(work["year"].min()),
                "year_max": int(work["year"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    df = pd.read_csv(PANEL_PATH)
    df = df[df["has_spei12"]].copy()
    df["drought"] = df["drought_spei12_mean_lt_minus1"].astype(float)
    for col in ["shannon_crop_area", "effective_number_crops", "top_crop_share", "spei12_annual_mean"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    df["shannon_z_x_drought"] = df["shannon_crop_area_z"] * df["drought"]
    df["dominance_z_x_drought"] = df["top_crop_share_z"] * df["drought"]
    df["shannon_z_x_spei12"] = df["shannon_crop_area_z"] * df["spei12_annual_mean_z"]
    df["dominance_z_x_spei12"] = df["top_crop_share_z"] * df["spei12_annual_mean_z"]

    results = []
    outcomes = ["production_log_anomaly", "rolling10_anomaly_sd", "rolling10_production_cv"]
    for outcome in outcomes:
        results.append(
            run_model(
                df,
                outcome,
                ["shannon_crop_area_z", "drought", "shannon_z_x_drought"],
                "area_year_fe_shannon_drought",
            )
        )
        results.append(
            run_model(
                df,
                outcome,
                ["top_crop_share_z", "drought", "dominance_z_x_drought"],
                "area_year_fe_dominance_drought",
            )
        )
        results.append(
            run_model(
                df,
                outcome,
                ["shannon_crop_area_z", "spei12_annual_mean_z", "shannon_z_x_spei12"],
                "area_year_fe_shannon_continuous_spei",
            )
        )

    out = pd.concat(results, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
