"""Fixed-effects diagnostics using polygon-sampled SPEI12 exposure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
POLY_PATH = PROJECT_DIR / "processed" / "country_year_spei12_polygon_sample_drought_metrics.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1g_polygon_spei_fixed_effects_results.csv"


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


def run(df: pd.DataFrame, outcome: str, regs: list[str], spec: str) -> pd.DataFrame:
    work = df[["area", "year", outcome] + regs].dropna().copy()
    dm = two_way_demean(work, [outcome] + regs, "area", "year")
    y = dm[outcome].to_numpy(dtype=float)
    x = dm[regs].to_numpy(dtype=float)
    beta, se = ols_cluster(y, x, work["area"].to_numpy())
    rows = []
    for term, est, stderr in zip(regs, beta, se):
        rows.append(
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
        )
    return pd.DataFrame(rows)


def main() -> int:
    main = pd.read_csv(PANEL_PATH)
    poly = pd.read_csv(POLY_PATH, dtype={"area_code_m49": str})
    df = main.merge(poly, on=["area_code_m49", "area", "year"], how="left")
    df = df[df["spei12_poly_mean_annual"].notna()].copy()
    df["drought_poly"] = (df["spei12_poly_mean_annual"] < -1).astype(float)
    for col in ["shannon_crop_area", "top_crop_share", "spei12_poly_mean_annual"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    df["shannon_z_x_drought_poly"] = df["shannon_crop_area_z"] * df["drought_poly"]
    df["dominance_z_x_drought_poly"] = df["top_crop_share_z"] * df["drought_poly"]
    df["shannon_z_x_spei12_poly"] = df["shannon_crop_area_z"] * df["spei12_poly_mean_annual_z"]

    rows = []
    for outcome in ["production_log_anomaly", "rolling10_anomaly_sd"]:
        rows.append(run(df, outcome, ["shannon_crop_area_z", "drought_poly", "shannon_z_x_drought_poly"], "poly_spei_drought_shannon"))
        rows.append(run(df, outcome, ["top_crop_share_z", "drought_poly", "dominance_z_x_drought_poly"], "poly_spei_drought_dominance"))
        rows.append(run(df, outcome, ["shannon_crop_area_z", "spei12_poly_mean_annual_z", "shannon_z_x_spei12_poly"], "poly_spei_continuous_shannon"))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
