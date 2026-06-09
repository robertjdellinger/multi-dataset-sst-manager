#!/usr/bin/env python
"""Prepare a CMA-GMST product 16 ocean-only gridded sensitivity dataset.

The local CMA CMDC API cache is a land-ocean merged CMA-GMST anomaly field, not
a verified standalone CMA-SST gridded product. This driver therefore writes a
separately named sensitivity NetCDF after masking land cells and rebaselining to
1991-2020. It does not add CMA to gridded_pipeline metadata, does not alter the
strict six-output CSV workflow, and does not create MEOW/PPOW regional products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from climind.fetchers.fetcher_cma_api import (  # noqa: E402
    OCEAN_MASK_SOURCE,
    _identify_cma_data_variable,
    _infer_lat_lon_names,
    _infer_year_month,
    _mask_missing_values,
    _ocean_cell_mask,
    _select_monthly_grid,
    validate_cma_gridded_netcdf_files,
)
from audit_cma_gridded_cache import find_cma_netcdf_files  # noqa: E402
from prepare_sst_gridded_inputs import standardize_grid  # noqa: E402


DATASET_NAME = "CMA-GMST-ocean-sensitivity"
PRODUCT_ROLE = "cma_gmst_product_16_ocean_only_sensitivity"
OUTPUT_VARIABLE = "sst_anomaly_C"
TARGET_BASELINE = (1991, 2020)
DEFAULT_YEAR_RANGE = (1850, 2025)
MIN_MONTHS_PER_YEAR = 12
MIN_BASELINE_YEARS = 30
INVENTORY_NAME = "cma_gmst_ocean_sensitivity_preparation_inventory.csv"


def managed_sst_root(datadir: Path | None = None) -> Path:
    """Return the DATADIR-managed SeaSurfaceTemperature root."""
    if datadir is None:
        data_dir = os.environ.get("DATADIR")
        if not data_dir:
            raise RuntimeError(
                "DATADIR is not set. Set DATADIR first, for example: "
                'export DATADIR="$HOME/data/multi-dataset-sst-manager"'
            )
        datadir = Path(data_dir)
    return Path(datadir) / "ManagedData" / "SeaSurfaceTemperature"


def processed_gridded_dir(datadir: Path | None = None) -> Path:
    """Return the processed gridded output directory."""
    return managed_sst_root(datadir) / "processed" / "gridded"


def qa_dir(datadir: Path | None = None) -> Path:
    """Return the SST QA directory."""
    return managed_sst_root(datadir) / "logs" / "qa"


def inventory_path(datadir: Path | None = None) -> Path:
    """Return the CMA-GMST sensitivity preparation inventory path."""
    return qa_dir(datadir) / INVENTORY_NAME


def sha256(path: Path) -> str:
    """Calculate SHA-256 for a source or output file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_year_range(paths: list[Path], start_year: int, end_year: int) -> list[Path]:
    """Select CMA monthly NetCDF files whose inferred year falls in range."""
    selected = []
    for path in paths:
        try:
            with xr.open_dataset(path, decode_times=True) as dataset:
                year, _month = _infer_year_month(path, dataset)
        except Exception:
            continue
        if start_year <= year <= end_year:
            selected.append(path)
    return sorted(selected)


def group_monthly_files(paths: list[Path], start_year: int, end_year: int) -> dict[int, dict[int, Path]]:
    """Group monthly CMA files by year and verify complete monthly coverage."""
    grouped: dict[int, dict[int, Path]] = defaultdict(dict)
    duplicates: list[str] = []
    for path in sorted(paths):
        with xr.open_dataset(path, decode_times=True) as dataset:
            year, month = _infer_year_month(path, dataset)
        if not (start_year <= year <= end_year):
            continue
        if month in grouped[year]:
            duplicates.append(f"{year}-{month:02d}")
        grouped[year][month] = path

    if duplicates:
        raise RuntimeError(f"duplicate monthly CMA grids: {', '.join(sorted(duplicates))}")

    missing = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if month not in grouped.get(year, {}):
                missing.append(f"{year}-{month:02d}")
    if missing:
        preview = ", ".join(missing[:24])
        suffix = "..." if len(missing) > 24 else ""
        raise RuntimeError(f"missing monthly CMA grids: {preview}{suffix}")

    return dict(grouped)


def read_monthly_ocean_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Read one CMA-GMST monthly grid, mask land cells, and standardize axes."""
    with xr.open_dataset(path, decode_times=True) as dataset:
        variable_name = _identify_cma_data_variable(dataset)
        variable = dataset[variable_name]
        lat_name, lon_name = _infer_lat_lon_names(dataset, variable)
        monthly_grid = _select_monthly_grid(variable, lat_name, lon_name)
        values = _mask_missing_values(monthly_grid.values, monthly_grid.attrs)
        latitudes = np.asarray(dataset[lat_name].values, dtype=float)
        longitudes = np.asarray(dataset[lon_name].values, dtype=float)
        ocean = _ocean_cell_mask(latitudes, longitudes)
        values = np.where(ocean, values, np.nan)
        values, latitudes, longitudes = standardize_grid(values, latitudes, longitudes)
        metadata = {
            "variable_name": variable_name,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "source_units": str(variable.attrs.get("units", "unspecified") or "unspecified"),
        }
    return values, latitudes, longitudes, metadata


def annualize_grouped_files(
    grouped: dict[int, dict[int, Path]],
    min_months: int = MIN_MONTHS_PER_YEAR,
) -> tuple[xr.DataArray, dict[str, object]]:
    """Annualize complete monthly CMA-GMST ocean-only grids."""
    annual_fields = []
    reference_latitudes: np.ndarray | None = None
    reference_longitudes: np.ndarray | None = None
    metadata: dict[str, object] = {}

    for year in sorted(grouped):
        monthly_values = []
        for month in range(1, 13):
            values, latitudes, longitudes, month_metadata = read_monthly_ocean_grid(grouped[year][month])
            metadata.update(month_metadata)
            if reference_latitudes is None:
                reference_latitudes = latitudes
                reference_longitudes = longitudes
            if not np.allclose(reference_latitudes, latitudes) or not np.allclose(reference_longitudes, longitudes):
                raise RuntimeError("CMA monthly grids use inconsistent latitude/longitude coordinates.")
            monthly_values.append(values)

        stack = np.stack(monthly_values, axis=0)
        finite_count = np.isfinite(stack).sum(axis=0)
        monthly_sum = np.nansum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
        annual = np.where(finite_count > 0, monthly_sum / np.maximum(finite_count, 1), np.nan)
        annual = np.where(finite_count >= min_months, annual, np.nan)
        annual_fields.append(annual)

    if reference_latitudes is None or reference_longitudes is None or not annual_fields:
        raise RuntimeError("No annual CMA-GMST sensitivity fields were produced.")

    years = np.array(sorted(grouped), dtype=int)
    data = xr.DataArray(
        np.stack(annual_fields, axis=0),
        dims=("year", "lat", "lon"),
        coords={"year": years, "lat": reference_latitudes, "lon": reference_longitudes},
        name=OUTPUT_VARIABLE,
    )
    return data, metadata


def rebaseline_anomaly_grid(
    annual: xr.DataArray,
    baseline_start: int,
    baseline_end: int,
    min_baseline_years: int,
) -> xr.DataArray:
    """Rebaseline an anomaly grid to the target climatology period."""
    baseline_period = annual.sel(year=slice(baseline_start, baseline_end))
    expected_years = baseline_end - baseline_start + 1
    if baseline_period.sizes.get("year", 0) != expected_years:
        raise RuntimeError(f"CMA-GMST sensitivity grid does not cover full {baseline_start}-{baseline_end} baseline.")
    baseline_counts = baseline_period.count("year")
    baseline = baseline_period.mean("year", skipna=True).where(baseline_counts >= min_baseline_years)
    return annual - baseline


def output_filename(annual: xr.DataArray, baseline_start: int, baseline_end: int) -> str:
    """Return the stable sensitivity output filename."""
    return (
        f"sst_{DATASET_NAME}_annual_gridded_"
        f"{int(annual['year'].values.min())}_{int(annual['year'].values.max())}_"
        f"baseline_{baseline_start}_{baseline_end}.nc"
    )


def write_inventory(path: Path, row: dict[str, object]) -> None:
    """Write a one-row preparation inventory."""
    fieldnames = [
        "dataset",
        "product_role",
        "source_root",
        "source_file_count",
        "variable_name",
        "lat_name",
        "lon_name",
        "source_value_type",
        "source_units",
        "year_start",
        "year_end",
        "n_year",
        "n_lat",
        "n_lon",
        "native_climatology_start",
        "native_climatology_end",
        "baseline_start",
        "baseline_end",
        "annualization_rule",
        "ocean_mask_source",
        "ocean_mask_applied",
        "primary_gridded_sst_eligible",
        "sensitivity_gridded_eligible",
        "regional_meow_ppow_eligible",
        "missing_data_fraction",
        "output_path",
        "output_checksum",
        "status",
        "created_at_utc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_preparation(
    *,
    datadir: Path | None = None,
    start_year: int = DEFAULT_YEAR_RANGE[0],
    end_year: int = DEFAULT_YEAR_RANGE[1],
    baseline_start: int = TARGET_BASELINE[0],
    baseline_end: int = TARGET_BASELINE[1],
    min_baseline_years: int = MIN_BASELINE_YEARS,
    strict: bool = False,
) -> dict[str, object]:
    """Prepare the CMA-GMST product 16 ocean-only sensitivity annual NetCDF."""
    datadir = Path(datadir) if datadir is not None else None
    source_files = find_cma_netcdf_files(datadir)
    source_files = select_year_range(source_files, start_year, end_year)
    if not source_files:
        result = {
            "dataset": DATASET_NAME,
            "status": "missing_cma_gmst_cache",
            "source_file_count": 0,
            "primary_gridded_sst_eligible": False,
            "sensitivity_gridded_eligible": False,
            "regional_meow_ppow_eligible": False,
        }
        if strict:
            raise RuntimeError("No local CMA-GMST monthly NetCDF cache files found.")
        return result

    validation = validate_cma_gridded_netcdf_files(source_files)
    if not validation.get("valid"):
        if strict:
            raise RuntimeError(f"CMA-GMST source cache validation failed: {validation.get('reason')}")
        return {
            "dataset": DATASET_NAME,
            "status": "source_validation_failed",
            "reason": validation.get("reason", ""),
            "source_file_count": len(source_files),
            "primary_gridded_sst_eligible": False,
            "sensitivity_gridded_eligible": False,
            "regional_meow_ppow_eligible": False,
        }

    grouped = group_monthly_files(source_files, start_year, end_year)
    annual, metadata = annualize_grouped_files(grouped)
    anomaly = rebaseline_anomaly_grid(annual, baseline_start, baseline_end, min_baseline_years)
    anomaly.name = OUTPUT_VARIABLE

    output_dir = processed_gridded_dir(datadir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(anomaly, baseline_start, baseline_end)
    dataset = xr.Dataset({OUTPUT_VARIABLE: anomaly.astype("float32")})
    dataset[OUTPUT_VARIABLE].attrs.update(
        {
            "long_name": "Annual CMA-GMST product 16 ocean-only surface-temperature anomaly sensitivity",
            "units": "degC",
            "baseline": f"{baseline_start}-{baseline_end}",
            "source_value_type": "anomaly",
            "processing": f"anomaly_rebaseline_1981_2010_to_{baseline_start}_{baseline_end}",
        }
    )
    dataset.attrs.update(
        {
            "dataset": DATASET_NAME,
            "product_role": PRODUCT_ROLE,
            "source_value_type": "anomaly",
            "native_climatology_start": 1981,
            "native_climatology_end": 2010,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "annualization_method": f"calendar_year_mean_min_{MIN_MONTHS_PER_YEAR}_months_per_cell",
            "ocean_mask_source": OCEAN_MASK_SOURCE,
            "ocean_mask_applied": "True",
            "primary_gridded_sst_eligible": "False",
            "sensitivity_gridded_eligible": "True",
            "regional_meow_ppow_eligible": "False",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    dataset.to_netcdf(output_path)

    row = {
        "dataset": DATASET_NAME,
        "product_role": PRODUCT_ROLE,
        "source_root": str(Path(source_files[0]).parent),
        "source_file_count": len(source_files),
        "variable_name": metadata.get("variable_name", validation.get("variable_name", "")),
        "lat_name": metadata.get("lat_name", validation.get("lat_name", "")),
        "lon_name": metadata.get("lon_name", validation.get("lon_name", "")),
        "source_value_type": "anomaly",
        "source_units": metadata.get("source_units", validation.get("units", "unspecified")),
        "year_start": int(anomaly["year"].values.min()),
        "year_end": int(anomaly["year"].values.max()),
        "n_year": int(anomaly.sizes["year"]),
        "n_lat": int(anomaly.sizes["lat"]),
        "n_lon": int(anomaly.sizes["lon"]),
        "native_climatology_start": 1981,
        "native_climatology_end": 2010,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "annualization_rule": f"calendar_year_mean_min_{MIN_MONTHS_PER_YEAR}_months_per_cell",
        "ocean_mask_source": OCEAN_MASK_SOURCE,
        "ocean_mask_applied": True,
        "primary_gridded_sst_eligible": False,
        "sensitivity_gridded_eligible": True,
        "regional_meow_ppow_eligible": False,
        "missing_data_fraction": float(anomaly.isnull().mean().values),
        "output_path": str(output_path),
        "output_checksum": sha256(output_path),
        "status": "processed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_inventory(inventory_path(datadir), row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", type=Path, default=None, help="DATADIR root; defaults to $DATADIR.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_YEAR_RANGE[0])
    parser.add_argument("--end-year", type=int, default=DEFAULT_YEAR_RANGE[1])
    parser.add_argument("--baseline-start", type=int, default=TARGET_BASELINE[0])
    parser.add_argument("--baseline-end", type=int, default=TARGET_BASELINE[1])
    parser.add_argument("--min-baseline-years", type=int, default=MIN_BASELINE_YEARS)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_preparation(
        datadir=args.datadir,
        start_year=args.start_year,
        end_year=args.end_year,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        min_baseline_years=args.min_baseline_years,
        strict=args.strict,
    )
    print(
        "prepare_cma_gmst_ocean_sensitivity: "
        f"status={result['status']}; "
        f"source_files={result['source_file_count']}; "
        f"output={result.get('output_path', '')}"
    )
    return 0 if result["status"] in {"processed", "missing_cma_gmst_cache"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
