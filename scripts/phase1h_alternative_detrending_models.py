"""Run drought interaction models with alternative production anomaly metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
ALT_PATH = PROJECT_DIR / "processed" / "country_year_alternative_anomaly_metrics.csv"
POLY_PATH = PROJECT_DIR / "processed" / "country_year_spei12_polygon_sample_drought_metrics.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1h_alternative_detrending_results.csv"


def two_way_demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        out[col] = df[col] - df.groupby("area")[col].transform("mean") - df.groupby("year")[col].transform("mean") + df[col].mean()
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


def run(df: pd.DataFrame, outcome: str, drought: str, interaction: str, spec: str) -> pd.DataFrame:
    regs = ["shannon_crop_area_z", drought, interaction]
    work = df[["area", "year", outcome] + regs].dropna().copy()
    dm = two_way_demean(work, [outcome] + regs)
    beta, se = ols_cluster(dm[outcome].to_numpy(float), dm[regs].to_numpy(float), work["area"].to_numpy())
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
    alt = pd.read_csv(ALT_PATH, dtype={"area_code_m49": str})
    poly = pd.read_csv(POLY_PATH, dtype={"area_code_m49": str})
    df = main.drop(columns=[c for c in alt.columns if c in main.columns and c not in {"area_code_m49", "area", "year"}])
    df = df.merge(alt, on=["area_code_m49", "area", "year"], how="left")
    df = df.merge(poly, on=["area_code_m49", "area", "year"], how="left")
    df["shannon_crop_area_z"] = (df["shannon_crop_area"] - df["shannon_crop_area"].mean()) / df["shannon_crop_area"].std()
    df["drought_point"] = (df["spei12_annual_mean"] < -1).astype(float)
    df["drought_poly"] = (df["spei12_poly_mean_annual"] < -1).astype(float)
    df["shannon_z_x_drought_point"] = df["shannon_crop_area_z"] * df["drought_point"]
    df["shannon_z_x_drought_poly"] = df["shannon_crop_area_z"] * df["drought_poly"]

    outcomes = [
        "production_log_anomaly",
        "production_log_growth",
        "production_log_anomaly_roll5",
        "production_log_anomaly_roll10",
    ]
    rows = []
    for outcome in outcomes:
        rows.append(run(df, outcome, "drought_point", "shannon_z_x_drought_point", "point_spei"))
        rows.append(run(df, outcome, "drought_poly", "shannon_z_x_drought_poly", "polygon_spei"))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(out[out["term"].str.contains("shannon_z_x")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
