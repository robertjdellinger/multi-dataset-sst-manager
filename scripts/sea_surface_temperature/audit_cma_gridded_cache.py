#!/usr/bin/env python
"""Audit locally cached CMA monthly gridded files for future SST sensitivity work.

This script is intentionally separate from the strict six-output annual SST
workflow and from the ERSST/HadSST gridded preparation workflow. The CMA files
currently obtained through CMA product 16 are land-ocean merged CMA-GMST anomaly
grids. They are valid source/cache material for the existing strict CMA fallback
aggregation, but they are not promoted to primary gridded SST metadata here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climind.fetchers.fetcher_cma_api import (  # noqa: E402
    discover_local_cma_sources,
    validate_cma_gridded_netcdf_files,
    validate_cma_monthly_source_csv,
)


CMA_MANAGED_SOURCE_NAME = "CMA-SST_Global_Month_Temp_1981_2010.csv"
INVENTORY_NAME = "cma_gridded_cache_inventory.csv"
SUMMARY_NAME = "cma_gridded_cache_summary.csv"
PRODUCT_ROLE = "cma_gmst_product_16_land_ocean_merged_anomaly_cache"
NOT_PRIMARY_REASON = (
    "CMA product 16 cache is a land-ocean merged CMA-GMST anomaly field, not a "
    "verified standalone CMA-SST gridded product. Use only for a documented "
    "CMA-GMST ocean-only sensitivity pass after ocean-mask handling is explicit."
)


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


def cma_data_dir(datadir: Path | None = None) -> Path:
    """Return the managed CMA-SST source/cache directory."""
    return managed_sst_root(datadir) / "Data" / "CMA-SST"


def cma_raw_dir(datadir: Path | None = None) -> Path:
    """Return the managed CMA raw CMDC API cache directory."""
    return cma_data_dir(datadir) / "cma_api_raw"


def qa_dir(datadir: Path | None = None) -> Path:
    """Return the DATADIR QA log directory used by the SST workflows."""
    return managed_sst_root(datadir) / "logs" / "qa"


def inventory_path(datadir: Path | None = None) -> Path:
    """Return the CMA gridded cache inventory path."""
    return qa_dir(datadir) / INVENTORY_NAME


def summary_path(datadir: Path | None = None) -> Path:
    """Return the CMA gridded cache summary path."""
    return qa_dir(datadir) / SUMMARY_NAME


def sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum for a source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_token(path: Path) -> str:
    """Return the YYYYMM token encoded in a CMA monthly filename."""
    match = re.search(r"(?<!\d)((?:18|19|20)\d{2})(0[1-9]|1[0-2])(?!\d)", path.name)
    if not match:
        return ""
    return "".join(match.groups())


def find_cma_netcdf_files(datadir: Path | None = None) -> list[Path]:
    """Find local CMA NetCDF cache files under DATADIR."""
    raw_dir = cma_raw_dir(datadir)
    if not raw_dir.exists():
        return []
    return sorted(
        path
        for path in raw_dir.rglob("*.nc")
        if path.is_file() and not path.name.startswith(".")
    )


def _find_name(names: list[str], candidates: set[str]) -> str:
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return ""


def _choose_variable(dataset: xr.Dataset, lat_name: str, lon_name: str) -> str:
    for name, variable in dataset.data_vars.items():
        if not np.issubdtype(variable.dtype, np.number):
            continue
        if lat_name in variable.dims and lon_name in variable.dims:
            return str(name)
    return ""


def _time_coordinate_source(dataset: xr.Dataset) -> str:
    names = {str(name).lower() for name in list(dataset.coords) + list(dataset.dims)}
    if "time" in names or "date" in names:
        return "coordinate"
    return "filename"


def inspect_cma_netcdf_file(path: Path) -> dict[str, object]:
    """Inspect one CMA monthly NetCDF cache file without changing it."""
    path = Path(path)
    with xr.open_dataset(path, decode_times=True) as dataset:
        names = [str(name) for name in list(dataset.coords) + list(dataset.dims)]
        lat_name = _find_name(names, {"lat", "latitude", "y"})
        lon_name = _find_name(names, {"lon", "longitude", "x"})
        variable_name = _choose_variable(dataset, lat_name, lon_name) if lat_name and lon_name else ""
        variable = dataset[variable_name] if variable_name else None
        units = str(variable.attrs.get("units", "")) if variable is not None else ""
        sizes = dict(dataset.sizes)
        token = month_token(path)
        year = int(token[:4]) if token else ""
        month = int(token[4:]) if token else ""
        return {
            "dataset": "CMA-SST",
            "product_role": PRODUCT_ROLE,
            "source_file": str(path),
            "filename": path.name,
            "year": year,
            "month": month,
            "variable_name": variable_name,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "n_lat": int(sizes.get(lat_name, 0)) if lat_name else 0,
            "n_lon": int(sizes.get(lon_name, 0)) if lon_name else 0,
            "time_coordinate_source": _time_coordinate_source(dataset),
            "units": units or "unspecified",
            "source_value_type": "anomaly" if variable_name == "anomaly" else "unknown",
            "primary_gridded_sst_eligible": False,
            "sensitivity_gridded_eligible": bool(variable_name and lat_name and lon_name),
            "regional_meow_ppow_eligible": False,
            "requires_ocean_mask": True,
            "reason": NOT_PRIMARY_REASON,
            "checksum": sha256(path),
        }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_audit(
    *,
    datadir: Path,
    project_root: Path,
    netcdf_files: list[Path],
    validation: dict[str, object],
    inventory_rows: list[dict[str, object]],
    start_year: int | None,
    end_year: int | None,
) -> dict[str, object]:
    """Build a one-row CMA cache audit summary."""
    cma_root = cma_data_dir(datadir)
    managed_csv = cma_root / CMA_MANAGED_SOURCE_NAME
    monthly_csv_validation = validate_cma_monthly_source_csv(managed_csv)
    annual_discovery = discover_local_cma_sources([project_root / "outputs" / "tables"])
    tokens = [month_token(path) for path in netcdf_files if month_token(path)]
    time_sources = sorted({str(row.get("time_coordinate_source", "")) for row in inventory_rows if row})
    status = "valid_cma_gmst_cache" if validation.get("valid") else "missing_cma_gridded_cache"
    if netcdf_files and not validation.get("valid"):
        status = "invalid_cma_gridded_cache"

    sensitivity_eligible = bool(validation.get("valid"))
    first_month = min(tokens) if tokens else ""
    last_month = max(tokens) if tokens else ""
    return {
        "dataset": "CMA-SST",
        "status": status,
        "product_role": PRODUCT_ROLE,
        "source_root": str(cma_root),
        "raw_cache_root": str(cma_raw_dir(datadir)),
        "netcdf_file_count": int(len(netcdf_files)),
        "first_month": first_month,
        "last_month": last_month,
        "audit_start_year": start_year or "",
        "audit_end_year": end_year or "",
        "variable_name": validation.get("variable_name", ""),
        "lat_name": validation.get("lat_name", ""),
        "lon_name": validation.get("lon_name", ""),
        "units": validation.get("units", "unspecified") or "unspecified",
        "source_value_type": validation.get("source_value_type", ""),
        "time_coordinate_source": ";".join(time_sources) if time_sources else "",
        "validation_valid": bool(validation.get("valid", False)),
        "primary_gridded_sst_eligible": False,
        "sensitivity_gridded_eligible": sensitivity_eligible,
        "regional_meow_ppow_eligible": False,
        "requires_ocean_mask": bool(netcdf_files),
        "requires_fractional_ocean_mask_for_final_sst_separation": bool(netcdf_files),
        "managed_monthly_csv_path": str(managed_csv) if managed_csv.exists() else "",
        "managed_monthly_csv_valid": bool(monthly_csv_validation.get("valid", False)),
        "managed_monthly_csv_gridded_eligible": bool(monthly_csv_validation.get("gridded_eligible", False)),
        "annual_csv_count": len(annual_discovery["annual_csv"]),
        "annual_csv_gridded_eligible": False,
        "reason": validation.get("reason", "") if not validation.get("valid") else NOT_PRIMARY_REASON,
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_audit(
    *,
    datadir: Path | None = None,
    project_root: Path | None = None,
    strict: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, object]:
    """Run the CMA gridded cache audit and write DATADIR QA outputs."""
    if datadir is None:
        data_dir = os.environ.get("DATADIR")
        if not data_dir:
            raise RuntimeError(
                "DATADIR is not set. Set DATADIR first, for example: "
                'export DATADIR="$HOME/data/multi-dataset-sst-manager"'
            )
        datadir = Path(data_dir)
    else:
        datadir = Path(datadir)
    project_root = Path(project_root) if project_root is not None else PROJECT_ROOT

    netcdf_files = find_cma_netcdf_files(datadir)
    validation = validate_cma_gridded_netcdf_files(netcdf_files)
    if strict and not validation.get("valid"):
        raise RuntimeError(f"CMA gridded cache audit failed: {validation.get('reason')}")

    inventory_rows = [inspect_cma_netcdf_file(path) for path in netcdf_files] if validation.get("valid") else []
    inventory_fields = [
        "dataset",
        "product_role",
        "source_file",
        "filename",
        "year",
        "month",
        "variable_name",
        "lat_name",
        "lon_name",
        "n_lat",
        "n_lon",
        "time_coordinate_source",
        "units",
        "source_value_type",
        "primary_gridded_sst_eligible",
        "sensitivity_gridded_eligible",
        "regional_meow_ppow_eligible",
        "requires_ocean_mask",
        "reason",
        "checksum",
    ]
    write_csv(inventory_path(datadir), inventory_rows, inventory_fields)

    summary = summarize_audit(
        datadir=datadir,
        project_root=project_root,
        netcdf_files=netcdf_files,
        validation=validation,
        inventory_rows=inventory_rows,
        start_year=start_year,
        end_year=end_year,
    )
    write_csv(
        summary_path(datadir),
        [summary],
        [
            "dataset",
            "status",
            "product_role",
            "source_root",
            "raw_cache_root",
            "netcdf_file_count",
            "first_month",
            "last_month",
            "audit_start_year",
            "audit_end_year",
            "variable_name",
            "lat_name",
            "lon_name",
            "units",
            "source_value_type",
            "time_coordinate_source",
            "validation_valid",
            "primary_gridded_sst_eligible",
            "sensitivity_gridded_eligible",
            "regional_meow_ppow_eligible",
            "requires_ocean_mask",
            "requires_fractional_ocean_mask_for_final_sst_separation",
            "managed_monthly_csv_path",
            "managed_monthly_csv_valid",
            "managed_monthly_csv_gridded_eligible",
            "annual_csv_count",
            "annual_csv_gridded_eligible",
            "reason",
            "audit_time_utc",
        ],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", type=Path, default=None, help="DATADIR root; defaults to $DATADIR.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Repository root for output-table checks.")
    parser.add_argument("--strict", action="store_true", help="Fail if a CMA NetCDF cache is absent or invalid.")
    parser.add_argument("--start-year", type=int, default=None, help="Optional audit period start year.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional audit period end year.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        datadir=args.datadir,
        project_root=args.project_root,
        strict=args.strict,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(
        "audit_cma_gridded_cache: "
        f"status={summary['status']}; "
        f"netcdf_files={summary['netcdf_file_count']}; "
        f"primary_gridded_sst_eligible={summary['primary_gridded_sst_eligible']}; "
        f"sensitivity_gridded_eligible={summary['sensitivity_gridded_eligible']}; "
        f"summary={summary_path(args.datadir)}"
    )


if __name__ == "__main__":
    main()
