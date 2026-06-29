"""Simple Phase 1D diversity x drought diagnostics.

These are descriptive regressions for triage, not final causal models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PANEL_PATH = PROJECT_DIR / "processed" / "phase1d_diversity_stability_drought_panel.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1d_drought_interaction_diagnostics.csv"


def fit_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    dof = max(x.shape[0] - x.shape[1], 1)
    sigma2 = float((residuals @ residuals) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return beta, se


def demean_by_group(series: pd.Series, group: pd.Series) -> pd.Series:
    return series - series.groupby(group).transform("mean")


def main() -> int:
    df = pd.read_csv(PANEL_PATH)
    df = df[df["has_spei12"]].copy()
    df["drought"] = df["drought_spei12_mean_lt_minus1"].astype(float)
    df["shannon_z"] = (df["shannon_crop_area"] - df["shannon_crop_area"].mean()) / df["shannon_crop_area"].std()
    df["dominance_z"] = (df["top_crop_share"] - df["top_crop_share"].mean()) / df["top_crop_share"].std()
    df["spei12_z"] = (df["spei12_annual_mean"] - df["spei12_annual_mean"].mean()) / df["spei12_annual_mean"].std()
    df["shannon_z_x_drought"] = df["shannon_z"] * df["drought"]
    df["dominance_z_x_drought"] = df["dominance_z"] * df["drought"]
    df["shannon_z_x_spei12_z"] = df["shannon_z"] * df["spei12_z"]

    outcomes = ["production_log_anomaly", "rolling10_anomaly_sd"]
    specs = [
        ("pooled_drought_interaction", ["Intercept", "shannon_z", "drought", "shannon_z_x_drought"]),
        ("pooled_spei_interaction", ["Intercept", "shannon_z", "spei12_z", "shannon_z_x_spei12_z"]),
        ("pooled_dominance_drought", ["Intercept", "dominance_z", "drought", "dominance_z_x_drought"]),
    ]
    rows = []
    for outcome in outcomes:
        for spec_name, cols in specs:
            work = df.copy()
            work["Intercept"] = 1.0
            y = work[outcome].to_numpy(dtype=float)
            x = work[cols].to_numpy(dtype=float)
            beta, se = fit_ols(y, x)
            for term, coef, term_se in zip(cols, beta, se):
                rows.append(
                    {
                        "outcome": outcome,
                        "spec": spec_name,
                        "term": term,
                        "estimate": coef,
                        "std_error_naive": term_se,
                        "n": int(np.isfinite(y).sum()),
                        "n_areas": int(work.loc[np.isfinite(y), "area"].nunique()),
                    }
                )

    # Two-way demeaning prototype: area and year fixed-effect residualization.
    fe = df.copy()
    fe["y_fe"] = demean_by_group(fe["production_log_anomaly"], fe["area"])
    fe["y_fe"] = demean_by_group(fe["y_fe"], fe["year"])
    for col in ["shannon_z", "drought", "shannon_z_x_drought"]:
        fe[col + "_fe"] = demean_by_group(fe[col], fe["area"])
        fe[col + "_fe"] = demean_by_group(fe[col + "_fe"], fe["year"])
    fe_cols = ["shannon_z_fe", "drought_fe", "shannon_z_x_drought_fe"]
    y = fe["y_fe"].to_numpy(dtype=float)
    x = fe[fe_cols].to_numpy(dtype=float)
    beta, se = fit_ols(y, x)
    for term, coef, term_se in zip(fe_cols, beta, se):
        rows.append(
            {
                "outcome": "production_log_anomaly",
                "spec": "area_year_demeaned_drought_interaction",
                "term": term,
                "estimate": coef,
                "std_error_naive": term_se,
                "n": int(np.isfinite(y).sum()),
                "n_areas": int(fe.loc[np.isfinite(y), "area"].nunique()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out):,} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
