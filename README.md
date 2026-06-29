# Crop Agrobiodiversity Resilience Code

This repository contains analysis scripts for the manuscript on crop portfolio
diversification and drought-year production resilience.

## Inputs

The scripts use public crop-production, climate-exposure and nutrition/value
data products described in the manuscript Data Availability statement. The
expected project-root folders are:

```text
raw/          downloaded or manually staged raw input files
processed/    derived analysis panels and intermediate metrics
tables/       model-result and diagnostic CSV outputs
source_data/  compact panel-level data used for manuscript source data
```

Large input and intermediate data files are not stored in this code repository.
When using an archived data release, place the released folders at the repository
root before running downstream model scripts.

## Outputs

The analysis scripts write derived CSV files to:


## Environment

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The code was written for Python 3.10+.

## Stepwise Analysis Workflow

Run scripts from the repository root. The workflow is staged because several
steps require large public datasets or archived intermediate files.

### 1. FAOSTAT crop-production panel and crop portfolio metrics

```bash
python scripts/download_and_prepare_faostat_crop_panel.py
python scripts/filter_country_level_panel.py
python scripts/build_crop_diversity_metrics.py
python scripts/build_crop_diversity_metrics_country_level.py
python scripts/build_function_stability_metrics.py
python scripts/build_phase1c_model_panel.py
```

Main outputs: crop-country-year panels, country-level crop portfolio metrics,
production-anomaly metrics and stability diagnostics in `processed/`.

### 2. Point-SPEI drought exposure and prototype fixed-effects analysis

```bash
python scripts/build_spei12_country_point_drought.py
python scripts/build_main_analysis_sample.py
python scripts/build_phase1d_drought_model_panel.py
python scripts/phase1d_drought_interaction_diagnostics.py
python scripts/phase1e_fixed_effects_models.py
python scripts/phase1f_robustness_models.py
```

Main outputs: point-SPEI country-year drought metrics, main analysis sample,
prototype drought-interaction diagnostics and robustness tables.

### 3. Crop-response asynchrony diagnostics

```bash
python scripts/build_crop_response_asynchrony_metrics.py
python scripts/phase1e_mechanism_diagnostics.py
```

Main outputs: crop-response asynchrony metrics and mechanism diagnostics.

### 4. Polygon and SPAM harvested-area-weighted drought exposure

```bash
python scripts/build_spei12_country_polygon_sample_drought.py
python scripts/combine_spei12_polygon_batches.py
python scripts/phase1g_polygon_spei_comparison.py
python scripts/phase1g_polygon_spei_fixed_effects.py
python scripts/fetch_spam2020_manifest.py
python scripts/download_and_inspect_spam2020_harvested_area.py
python scripts/build_spam2020_spei12_harvested_area_weights.py
python scripts/build_spei12_spam_weighted_drought.py
python scripts/phase1i_compare_climate_exposures.py
python scripts/phase1i_spam_weighted_spei_fixed_effects.py
```

Main outputs: polygon-sampled and harvested-area-weighted drought-exposure
metrics, exposure-comparison diagnostics and main SPAM-weighted model results.

### 5. Robustness and alternative outcomes

```bash
python scripts/build_alternative_anomaly_metrics.py
python scripts/phase1h_alternative_detrending_models.py
python scripts/phase1j_spam_weighted_spei_robustness.py
python scripts/phase1j_summarize_spam_robustness.py
python scripts/phase1j_spam_weighted_spei_alternative_detrending.py
python scripts/build_spam2020_water_context.py
python scripts/phase1k_enhanced_robustness_models.py
```

Main outputs: alternative anomaly analyses, SPAM-weighted robustness tables,
water-context diagnostics and enhanced robustness models.

### 6. Nutrition, economic-value and final evidence summaries

```bash
python scripts/build_fbs_nutrition_conversion_factors.py
python scripts/build_nutrition_weighted_outcomes.py
python scripts/phase1l_nutrition_weighted_outcome_models.py
python scripts/build_economic_value_weighted_outcomes.py
python scripts/phase1m_economic_value_weighted_outcome_models.py
python scripts/phase1n_mechanism_planning_diagnostics.py
python scripts/summarize_core_evidence.py
```

Main outputs: nutrition-weighted and economic-value-weighted outcome panels,
corresponding fixed-effects results, country-level mechanism diagnostics and
core evidence summary tables.

## Script Order

`SCRIPT_MANIFEST.csv` lists the scripts in the same staged order shown above.
