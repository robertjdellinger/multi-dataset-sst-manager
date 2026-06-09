# observational-sst-reconstruction

A reproducible modern observational sea-surface-temperature construction
workflow for five annual SST products, covering 1850 to present where each
source is available.

This repository is a sea-surface-temperature-specific adaptation of the
upstream `jjk-code-otter/climate-indicator-manager` / Climind architecture. The
remaining public workflow is intentionally narrow: it downloads or reuses the
five configured observational SST inputs, standardizes them to annual
1991-2020-baselined anomalies, writes BADC-style CSV outputs, validates those
outputs against the reference archive, and generates the annual comparison
figures.

## Authoritative workflow

The production contract is controlled only by:

- `scripts/data_management/build_sst_outputs.py --strict`
- `climind/metadata_files/temperature/sst/build_pipeline/`

The active strict metadata files are:

- `climind/metadata_files/temperature/sst/build_pipeline/cma_sst.json`
- `climind/metadata_files/temperature/sst/build_pipeline/cmems_sst.json`
- `climind/metadata_files/temperature/sst/build_pipeline/dcent_sst_i.json`
- `climind/metadata_files/temperature/sst/build_pipeline/ersst_v6.json`
- `climind/metadata_files/temperature/sst/build_pipeline/hadsst4.json`

No spatial NetCDF workflow, regional product, sensitivity product, dashboard
workflow, raw data cache, or generated DATADIR artifact is an input to the six
required annual CSV outputs.

## Required outputs

Strict mode produces exactly these BADC-style CSV files in `outputs/tables/`:

| Output | Rows | Years |
| --- | ---: | --- |
| `outputs/tables/sst_CMA_SST.csv` | 176 | 1850-2025 |
| `outputs/tables/sst_CMEMS_SST.csv` | 43 | 1982-2024 |
| `outputs/tables/sst_DCENT_SST_I.csv` | 176 | 1850-2025 |
| `outputs/tables/sst_ERSST_v6.csv` | 176 | 1850-2025 |
| `outputs/tables/sst_HadSST4.csv` | 176 | 1850-2025 |
| `outputs/tables/sst_summary.csv` | 176 | 1850-2025 |

The row counts use the repository BADC-CSV parser, which reads the data block
and stops before `end data`.

## Processing method

`build_sst_outputs.py --strict` keeps the Climind metadata-driven flow:

1. Load the five build-pipeline metadata collections with
   `DataCollection.from_file()`.
2. Fetch or reuse source files under
   `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/`.
3. Read each product with the reader named in its JSON metadata.
4. Convert monthly sources to annual values with
   `TimeSeriesMonthly.make_annual()` after monthly coverage checks.
5. Rebaseline every annual series to 1991-2020 with Climind time-series
   methods.
6. Select the required annual range.
7. Write dataset CSVs with `write_csv()`.
8. Write the merged summary with `write_dataset_summary_file_with_metadata()`.
9. Validate each data block against
   `data/raw/reference/Sea-surface_temperature_data_files.zip`.

Monthly sources are annualized by arithmetic mean. Annual sources are read as
annual time series and then explicitly rebaselined. Strict mode fails if a
dataset lacks a known source value type or climatology state, or if any required
CSV is missing.

## Input/output contract

| Dataset | Metadata | Reader | Managed raw source location | Source value type | Native baseline | Output years | Rows | Validation tolerance |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| CMA-SST | `cma_sst.json` | `reader_cma_gmst` | `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/CMA-SST/CMA-SST_Global_Month_Temp_1981_2010.csv` | anomaly | 1981-2010 | 1850-2025 | 176 | 0.20 degC |
| CMEMS-SST | `cmems_sst.json` | `reader_cmems_sst_area_average` | `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/CMEMS-SST/global_omi_tempsal_sst_area_averaged_anomalies_19820101-20241231_R19912020_P20250516.nc` | anomaly | 1991-2020 | 1982-2024 | 43 | 0.01 degC |
| DCENT-SST-I | `dcent_sst_i.json` | `reader_dcenti` | `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/DCENT-SST-I/DCENT_DCENT_I_OST_monthly_statistics.txt` | anomaly | 1991-2020 | 1850-2025 | 176 | 0.01 degC |
| ERSST-v6 | `ersst_v6.json` | `reader_noaaglobaltemp` | `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/ERSST-v6/aravg.mon.ocean.90S.90N.v6.0.0.202512.asc` | anomaly | 1991-2020 | 1850-2025 | 176 | 0.01 degC |
| HadSST4-SST | `hadsst4.json` | `reader_hadsst_ts` | `$DATADIR/ManagedData/SeaSurfaceTemperature/Data/HadSST4-SST/HadSST.4.2.0.0_monthly_GLOBE.csv` | anomaly | 1961-1990 | 1850-2025 | 176 | 0.01 degC |

All five products are written as annual SST anomalies relative to 1991-2020.
Monthly sources are annualized only after source-appropriate coverage checks.
The generated data blocks are validated against
`data/raw/reference/Sea-surface_temperature_data_files.zip`.

DCENT-I uses the monthly ocean-statistics source as the active processing
input. The annual ocean-statistics source is retained in metadata only as a
cross-check for uncertainty attachment and validation; it is not the selected
strict data-value source.

The canonical DCENT-I landing page recorded in metadata is:

```text
https://doi.org/10.7910/DVN/ROG38Q
```

## Validation

The reference validation compares the generated data blocks against:

```text
data/raw/reference/Sea-surface_temperature_data_files.zip
```

Default validation tolerance is `0.01 degC`. CMA uses `0.20 degC` because the
active CMA fallback can derive an ocean mean from CMA product 16 fields that do
not exactly reproduce the precomputed CMA-SST reference series. That wider
tolerance is explicit and limited to CMA; all other products use the default
tolerance.

QA files are written under `outputs/logs/qa/`:

- `sst_source_acquisition_log.csv`
- `sst_processing_log.csv`
- `sst_baseline_audit.csv`
- `sst_reference_validation.csv`
- `sst_workflow_summary.json`

## Figures

The annual reference figure and annual diagnostic suite are generated with:

```bash
DATADIR="$HOME/data/multi-dataset-sst-manager" \
python scripts/sea_surface_temperature/plot_global_sst_reference_figure.py
```

Tracked figure outputs are:

- `outputs/figures/global_sea_surface_temperature_1850_2025_reference_style.png`
- `outputs/figures/annual_diagnostics/sst_annual_baseline_sensitivity.png`
- `outputs/figures/annual_diagnostics/sst_annual_dataset_availability_and_validation_summary.png`
- `outputs/figures/annual_diagnostics/sst_annual_modern_overlap_1982_present.png`
- `outputs/figures/annual_diagnostics/sst_annual_pairwise_residual_heatmap.png`
- `outputs/figures/annual_diagnostics/sst_annual_period_trend_comparison.png`
- `outputs/figures/annual_diagnostics/sst_annual_preindustrial_sensitivity_1850_1900.png`
- `outputs/figures/annual_diagnostics/sst_annual_rolling_differences.png`
- `outputs/figures/annual_diagnostics/sst_annual_source_coverage.png`
- `outputs/figures/annual_diagnostics/sst_annual_source_spread.png`

Each figure uses annual SST anomalies relative to 1991-2020 unless the file name
or title explicitly states the 1850-1900 sensitivity baseline. Figure captions
and labels distinguish the four products available from 1850-2025 from the
satellite-era CMEMS product available from 1982-2024.

The compatibility wrapper is retained for callers that still use the older
script name:

```bash
DATADIR="$HOME/data/multi-dataset-sst-manager" \
python scripts/global_sst_reconstruction_reference_plot.py
```

## Run checklist

From the repository root:

```bash
DATADIR="$HOME/data/multi-dataset-sst-manager" \
CMDCAPI_PATH="$HOME/Downloads/CMDCapi.py" \
python scripts/data_management/build_sst_outputs.py --strict

DATADIR="$HOME/data/multi-dataset-sst-manager" \
python scripts/sea_surface_temperature/plot_global_sst_reference_figure.py

python -m pytest -q
git diff --check
git status --short
```

Before committing, confirm that credentials, `.env` files, CMDC SDK files, raw
CMA downloads, NetCDF source files, ZIP files, DATADIR caches, temporary build
folders, and large generated data products are not staged.
