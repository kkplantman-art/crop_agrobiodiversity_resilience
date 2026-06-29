"""Summarize SPAM-weighted SPEI robustness key terms."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
KEY_PATH = PROJECT_DIR / "tables" / "phase1j_spam_weighted_spei_robustness_key_terms.csv"
OUT_PATH = PROJECT_DIR / "tables" / "phase1j_spam_weighted_spei_robustness_summary.csv"
OUT_PROD_PATH = PROJECT_DIR / "tables" / "phase1j_spam_weighted_spei_production_key_terms_compact.csv"


def family(term: str) -> str:
    if "dominance" in term:
        return "Top-crop dominance"
    if term == "shannon_z_x_spei12_spam":
        return "Shannon continuous SPEI"
    if "lag1" in term:
        return "Lagged Shannon diversity"
    return "Shannon diversity"


def drought_definition(spec: str) -> str:
    if "continuous_spam_spei" in spec:
        return "continuous SPEI"
    for marker in [
        "mean_lt_minus0_5",
        "mean_lt_minus1_0",
        "mean_lt_minus1_5",
        "min_lt_minus1_5",
    ]:
        if marker in spec:
            return marker
    return "other"


def main() -> int:
    key = pd.read_csv(KEY_PATH)
    key["term_family"] = key["term"].map(family)
    key["drought_definition"] = key["spec"].map(drought_definition)
    summary = (
        key.groupby(["outcome", "term_family", "drought_definition"], as_index=False)
        .agg(
            n_specs=("term", "count"),
            n_expected_direction=("supports_buffering", "sum"),
            share_expected_direction=("supports_buffering", "mean"),
            min_t=("t_stat", "min"),
            max_t=("t_stat", "max"),
            median_t=("t_stat", "median"),
            median_abs_t=("t_stat", lambda s: s.abs().median()),
        )
        .sort_values(["outcome", "term_family", "drought_definition"])
    )
    summary.to_csv(OUT_PATH, index=False)

    production = key[key["outcome"].eq("production_log_anomaly")].copy()
    production = production[
        [
            "sample",
            "spec",
            "term",
            "estimate",
            "cluster_se_area",
            "t_stat",
            "n",
            "n_areas",
            "supports_buffering",
        ]
    ].sort_values(["sample", "spec", "term"])
    production.to_csv(OUT_PROD_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(summary):,} rows)")
    print(summary.to_string(index=False))
    print(f"wrote {OUT_PROD_PATH} ({len(production):,} rows)")
    print(production.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
