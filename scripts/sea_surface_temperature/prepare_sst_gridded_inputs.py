#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/sea_surface_temperature/prepare_sst_gridded_inputs.py
#
#  OPTIONAL: standardise true latitude-longitude gridded SST fields prior to
#  MEOW/PPOW regional averaging. Mirrors the staged upstream gridded pathway
#  (regrid/standardise first), but for SST and marine regions rather than WMO
#  regions. This is add-only and is not required for the six global SST CSVs.
#
#  Standardisation covers: longitude convention (-180..180), ascending latitude
#  order, missing-value handling, and monthly->annual aggregation with a
#  coverage threshold. Heavy IO/regridding is intentionally left to the caller;
#  the reusable numeric logic lives in sst_regional_core.

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from climind.data_manager.processing import DataCollection  # noqa: E402
import sst_regional_core as core  # noqa: E402

TARGET_YEAR_RANGE = (1850, 2025)
TARGET_BASELINE = (1991, 2020)
MIN_MONTHS_PER_YEAR = 12
MIN_BASELINE_YEARS = 30
GRIDDED_METADATA_DIR = (
    PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "gridded_pipeline"
)
PREPARATION_INVENTORY_NAME = "sst_gridded_preparation_inventory.csv"


def managed_sst_root() -> Path:
    data_dir = os.environ.get("DATADIR")
    if not data_dir:
        raise RuntimeError(
            "DATADIR is not set. Set DATADIR first, for example: "
            'export DATADIR="$HOME/data/multi-dataset-sst-manager"'
        )
    return Path(data_dir) / "ManagedData" / "SeaSurfaceTemperature"


def gridded_inventory_path() -> Path:
    return managed_sst_root() / "logs" / "qa" / "sst_gridded_source_inventory.csv"


def gridded_preparation_inventory_path() -> Path:
    return managed_sst_root() / "logs" / "qa" / PREPARATION_INVENTORY_NAME


def processed_gridded_dir() -> Path:
    return managed_sst_root() / "processed" / "gridded"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def metadata_by_dataset() -> dict[str, dict[str, object]]:
    records = {}
    for metadata_file in sorted(GRIDDED_METADATA_DIR.glob("*.json")):
        collection = DataCollection.from_file(metadata_file)
        dataset = collection.datasets[0].metadata
        records[collection.global_attributes["name"]] = {
            "collection": collection,
            "metadata_file": metadata_file,
            "dataset_metadata": dataset,
        }
    return records


def metadata_get(metadata, key: str, default=None):
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return default


def to_minus180_180(longitudes: np.ndarray) -> np.ndarray:
    """Convert a longitude coordinate to the -180..180 convention."""
    lon = np.asarray(longitudes, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def standardize_grid(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray):
    """Return (values, latitudes, longitudes) with ascending lat and -180..180 lon.

    ``values`` is a (lat, lon) field. Longitude is converted to -180..180 and
    both axes are sorted ascending so masks and weights align consistently.
    """
    values = np.asarray(values, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    lon = to_minus180_180(longitudes)

    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    return values[np.ix_(lat_order, lon_order)], lat[lat_order], lon[lon_order]


def annualize_monthly_grid(monthly_grids: np.ndarray, min_months: int = 12) -> np.ndarray:
    """Annual-mean a (month, lat, lon) stack with a per-cell coverage threshold."""
    stack = np.asarray(monthly_grids, dtype=float)
    if stack.ndim != 3:
        raise ValueError("monthly_grids must be (month, lat, lon).")
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    summed = np.nansum(np.where(finite, stack, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        annual = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
    annual[counts < min_months] = np.nan
    return annual


def normalize_gridded_dataarray(
    data: xr.DataArray,
    lat_name: str,
    lon_name: str,
    time_name: str,
) -> xr.DataArray:
    """Normalize coordinate names, longitude convention, and latitude ordering."""
    selected = data.transpose(time_name, lat_name, lon_name)
    selected = selected.assign_coords({lon_name: to_minus180_180(selected[lon_name].values)})
    selected = selected.sortby(lon_name).sortby(lat_name)
    selected = selected.rename({time_name: "time", lat_name: "lat", lon_name: "lon"})
    return selected


def convert_units(data: xr.DataArray, units: str, source_value_type: str) -> xr.DataArray:
    """Convert source fields to degrees C or degC-equivalent anomalies."""
    units_normalized = str(units).strip().lower()
    if source_value_type == "actual" and units_normalized in {"k", "kelvin"}:
        return data - 273.15
    # Temperature anomalies expressed in K have the same increment as degC.
    if units_normalized in {"k", "kelvin", "degc", "degree_c", "degrees_celsius", "celsius", "c"}:
        return data
    if not units_normalized:
        raise RuntimeError("Gridded SST source units are missing.")
    raise RuntimeError(f"Unsupported gridded SST units: {units}")


def annualize_dataarray(data: xr.DataArray, min_months: int = MIN_MONTHS_PER_YEAR) -> xr.DataArray:
    """Annualize monthly fields with a per-cell minimum monthly coverage rule."""
    years = np.unique(data["time"].dt.year.values.astype(int))
    annual_fields = []
    for year in years:
        if year < TARGET_YEAR_RANGE[0] or year > TARGET_YEAR_RANGE[1]:
            continue
        monthly = data.where(data["time"].dt.year == year, drop=True)
        if monthly.sizes.get("time", 0) == 0:
            continue
        counts = monthly.count("time")
        annual = monthly.mean("time", skipna=True).where(counts >= min_months)
        annual_fields.append(annual.expand_dims(year=[int(year)]))
    if not annual_fields:
        raise RuntimeError("No annual gridded SST fields were produced.")
    return xr.concat(annual_fields, dim="year")


def rebaseline_annual_grid(
    annual: xr.DataArray,
    source_value_type: str,
    native_start: int,
    native_end: int,
    target_start: int = TARGET_BASELINE[0],
    target_end: int = TARGET_BASELINE[1],
    min_baseline_years: int = MIN_BASELINE_YEARS,
) -> tuple[xr.DataArray, str]:
    """Convert actual or native-anomaly annual grids to 1991-2020 anomalies."""
    baseline_period = annual.sel(year=slice(target_start, target_end))
    if baseline_period.sizes.get("year", 0) != target_end - target_start + 1:
        raise RuntimeError(f"Annual grid does not cover the full {target_start}-{target_end} baseline.")
    baseline_counts = baseline_period.count("year")
    baseline = baseline_period.mean("year", skipna=True).where(baseline_counts >= min_baseline_years)
    adjusted = annual - baseline
    if source_value_type == "actual":
        note = f"actual_to_anomaly_{target_start}_{target_end}"
    else:
        note = f"anomaly_rebaseline_{native_start}_{native_end}_to_{target_start}_{target_end}"
    return adjusted, note


def output_filename(dataset_name: str, annual: xr.DataArray) -> str:
    start_year = int(annual["year"].values.min())
    end_year = int(annual["year"].values.max())
    return (
        f"sst_{dataset_name}_annual_gridded_{start_year}_{end_year}_"
        f"baseline_{TARGET_BASELINE[0]}_{TARGET_BASELINE[1]}.nc"
    )


def prepare_one_dataset(row: dict[str, str], metadata_records: dict[str, dict[str, object]]) -> dict[str, object]:
    dataset_name = row["dataset"]
    if dataset_name not in metadata_records:
        raise RuntimeError(f"No gridded metadata found for inventory dataset {dataset_name}.")
    metadata_record = metadata_records[dataset_name]
    dataset_metadata = metadata_record["dataset_metadata"]
    source_file = Path(row["source_file"])
    variable_name = row["variable_name"]
    lat_name = row["lat_name"]
    lon_name = row["lon_name"]
    time_name = row["time_name"]
    source_value_type = str(row.get("source_value_type") or "").lower()
    if source_value_type not in {"actual", "anomaly"}:
        raise RuntimeError(f"{dataset_name} has unknown source value type: {source_value_type!r}.")

    with xr.open_dataset(source_file, decode_times=True) as dataset:
        source_data = dataset[variable_name]
        source_units = str(source_data.attrs.get("units", metadata_get(dataset_metadata, "units", "")))
        normalized = normalize_gridded_dataarray(source_data, lat_name, lon_name, time_name)
        converted = convert_units(normalized, source_units, source_value_type)
        annual = annualize_dataarray(converted)
        annual_anomaly, processing_note = rebaseline_annual_grid(
            annual,
            source_value_type=source_value_type,
            native_start=int(metadata_get(dataset_metadata, "climatology_start")),
            native_end=int(metadata_get(dataset_metadata, "climatology_end")),
        )

    output_dir = processed_gridded_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(dataset_name, annual_anomaly)
    output = xr.Dataset(
        {
            "sst_anomaly_C": annual_anomaly.astype("float32"),
        }
    )
    output["sst_anomaly_C"].attrs.update(
        {
            "long_name": "Annual sea-surface temperature anomaly",
            "units": "degC",
            "baseline": f"{TARGET_BASELINE[0]}-{TARGET_BASELINE[1]}",
            "source_value_type": source_value_type,
            "processing": processing_note,
        }
    )
    output.attrs.update(
        {
            "dataset": dataset_name,
            "source_file": str(source_file),
            "source_checksum": sha256(source_file),
            "metadata_file": str(metadata_record["metadata_file"]),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "annualization_method": f"calendar_year_mean_min_{MIN_MONTHS_PER_YEAR}_months_per_cell",
            "baseline_start": TARGET_BASELINE[0],
            "baseline_end": TARGET_BASELINE[1],
            "min_baseline_years_per_cell": MIN_BASELINE_YEARS,
        }
    )
    output.to_netcdf(output_path)

    anomaly = output["sst_anomaly_C"]
    return {
        "dataset": dataset_name,
        "source_file": str(source_file),
        "source_checksum": sha256(source_file),
        "variable_name": variable_name,
        "lat_name": lat_name,
        "lon_name": lon_name,
        "time_name": time_name,
        "source_value_type": source_value_type,
        "source_units": source_units,
        "baseline_start": TARGET_BASELINE[0],
        "baseline_end": TARGET_BASELINE[1],
        "native_climatology_start": metadata_get(dataset_metadata, "climatology_start"),
        "native_climatology_end": metadata_get(dataset_metadata, "climatology_end"),
        "annualization_rule": f"calendar_year_mean_min_{MIN_MONTHS_PER_YEAR}_months_per_cell",
        "year_start": int(anomaly["year"].values.min()),
        "year_end": int(anomaly["year"].values.max()),
        "n_year": int(anomaly.sizes["year"]),
        "n_lat": int(anomaly.sizes["lat"]),
        "n_lon": int(anomaly.sizes["lon"]),
        "missing_data_fraction": float(anomaly.isnull().mean().values),
        "output_path": str(output_path),
        "output_checksum": sha256(output_path),
        "processing_notes": processing_note,
        "status": "processed",
    }


def selected_inventory_rows(datasets: list[str] | None = None) -> list[dict[str, str]]:
    inventory = gridded_inventory_path()
    if not inventory.exists():
        raise RuntimeError(f"Missing gridded source inventory: {inventory}")
    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row for row in rows
        if str(row.get("eligible_for_regional_processing", "")).lower() == "true"
    ]
    if datasets:
        requested = {normalize_dataset_name(dataset) for dataset in datasets}
        eligible = [
            row for row in eligible
            if normalize_dataset_name(row["dataset"]) in requested
        ]
        found = {normalize_dataset_name(row["dataset"]) for row in eligible}
        missing = requested - found
        if missing:
            raise RuntimeError(f"Requested gridded dataset(s) not eligible or missing: {sorted(missing)}")
    return eligible


def run_preparation(datasets: list[str] | None = None) -> list[dict[str, object]]:
    rows = selected_inventory_rows(datasets)
    metadata_records = metadata_by_dataset()
    records = [prepare_one_dataset(row, metadata_records) for row in rows]
    inventory_path = gridded_preparation_inventory_path()
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        fieldnames = list(records[0].keys())
        with inventory_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    return records


def run_eligibility_only() -> dict[str, object]:
    inventory = gridded_inventory_path()
    if not inventory.exists():
        return {
            "inventory_path": str(inventory),
            "rows": 0,
            "eligible": 0,
            "status": "missing_inventory",
        }

    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row for row in rows
        if str(row.get("eligible_for_regional_processing", "")).lower() == "true"
    ]
    return {
        "inventory_path": str(inventory),
        "rows": len(rows),
        "eligible": len(eligible),
        "status": "ok",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligibility-only",
        action="store_true",
        help="Read gridded inventory and exit without writing processed NetCDF files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail in eligibility-only mode if the gridded inventory is missing.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional gridded dataset names to prepare. Defaults to all eligible inventory rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eligibility_only:
        result = run_eligibility_only()
        if args.strict and result["status"] != "ok":
            raise RuntimeError(f"Gridded inventory missing: {result['inventory_path']}")
        print(
            "prepare_sst_gridded_inputs eligibility-only: "
            f"{result['eligible']} eligible of {result['rows']} inventory rows "
            f"({result['inventory_path']})"
        )
        return 0

    records = run_preparation(args.datasets)
    print(
        "prepare_sst_gridded_inputs: wrote "
        f"{len(records)} processed annual gridded SST anomaly file(s); "
        f"inventory={gridded_preparation_inventory_path()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
