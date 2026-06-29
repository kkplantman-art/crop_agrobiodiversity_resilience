"""Fixed-effects diagnostics using SPAM harvested-area-weighted SPEI12."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_main_analysis_sample_panel.csv"
SPAM_SPEI_PATH = PROJECT_DIR / "processed" / "country_year_spei12_spam_harvested_area_weighted_metrics.csv"
OUT_PANEL = PROJECT_DIR / "processed" / "phase1i_spam_weighted_spei_model_panel.csv"
OUT_RESULTS = PROJECT_DIR / "tables" / "phase1i_spam_weighted_spei_fixed_effects_results.csv"


def two_way_demean(df: pd.DataFrame, cols: list[str], entity: str, time: str) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        grand = df[col].mean()
        out[col] = (
            df[col]
            - df.groupby(entity)[col].transform("mean")
            - df.groupby(time)[col].transform("mean")
            + grand
        )
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


def run(df: pd.DataFrame, outcome: str, regs: list[str], spec: str) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regs].dropna().copy()
    dm = two_way_demean(work, [outcome] + regs, "area", "year")
    y = dm[outcome].to_numpy(dtype=float)
    x = dm[regs].to_numpy(dtype=float)
    beta, se = ols_cluster(y, x, work["area"].to_numpy())
    return pd.DataFrame(
        [
            {
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


def main() -> int:
    main = pd.read_csv(MAIN_PATH, dtype={"area_code_m49": str})
    spam = pd.read_csv(SPAM_SPEI_PATH, dtype={"area_code_m49": str})
    df = main.merge(spam, on=["area_code_m49", "area", "year"], how="left")
    df["has_spam_spei12"] = df["spei12_spam_weighted_mean_annual"].notna()
    df.to_csv(OUT_PANEL, index=False)

    work = df[df["has_spam_spei12"]].copy()
    work["drought_spam"] = work["drought_spam_mean_lt_minus1"].astype(float)
    for col in ["shannon_crop_area", "top_crop_share", "spei12_spam_weighted_mean_annual"]:
        work[col + "_z"] = (work[col] - work[col].mean()) / work[col].std()
    work["shannon_z_x_drought_spam"] = work["shannon_crop_area_z"] * work["drought_spam"]
    work["dominance_z_x_drought_spam"] = work["top_crop_share_z"] * work["drought_spam"]
    work["shannon_z_x_spei12_spam"] = (
        work["shannon_crop_area_z"] * work["spei12_spam_weighted_mean_annual_z"]
    )

    rows = []
    for outcome in ["production_log_anomaly", "rolling10_anomaly_sd", "rolling10_production_cv"]:
        rows.append(
            run(
                work,
                outcome,
                ["shannon_crop_area_z", "drought_spam", "shannon_z_x_drought_spam"],
                "spam_weighted_spei_drought_shannon",
            )
        )
        rows.append(
            run(
                work,
                outcome,
                ["top_crop_share_z", "drought_spam", "dominance_z_x_drought_spam"],
                "spam_weighted_spei_drought_dominance",
            )
        )
        rows.append(
            run(
                work,
                outcome,
                [
                    "shannon_crop_area_z",
                    "spei12_spam_weighted_mean_annual_z",
                    "shannon_z_x_spei12_spam",
                ],
                "spam_weighted_spei_continuous_shannon",
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_RESULTS, index=False)
    print(f"wrote {OUT_PANEL} ({len(df):,} rows; {df['has_spam_spei12'].sum():,} with SPAM-SPEI)")
    print(f"wrote {OUT_RESULTS} ({len(out):,} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
