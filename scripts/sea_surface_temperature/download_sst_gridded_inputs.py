#!/usr/bin/env python
"""Download/stage true gridded SST inputs for optional regional processing.

This driver is deliberately separate from scripts/data_management/build_sst_outputs.py.
It only reads metadata from climind/metadata_files/temperature/sst/gridded_pipeline,
downloads or locates gridded source files under DATADIR, inspects the gridded
file structure, and writes inventory/provenance logs. It does not build the six
global reference CSV outputs, does not plot figures, and does not calculate
MEOW/PPOW regional summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climind.data_manager.processing import DataCollection  # noqa: E402


GRIDDED_METADATA_DIR = (
    PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "gridded_pipeline"
)
INVENTORY_NAME = "sst_gridded_source_inventory.csv"
DOWNLOAD_LOG_NAME = "sst_gridded_download_log.csv"
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
TIME_NAMES = ("time", "date", "t")


def managed_sst_root() -> Path:
    data_dir = os.environ.get("DATADIR")
    if not data_dir:
        raise RuntimeError(
            "DATADIR is not set. Set DATADIR first, for example: "
            'export DATADIR="$HOME/data/multi-dataset-sst-manager"'
        )
    return Path(data_dir) / "ManagedData" / "SeaSurfaceTemperature"


def gridded_data_dir() -> Path:
    return managed_sst_root() / "Data" / "gridded"


def gridded_qa_dir() -> Path:
    return managed_sst_root() / "logs" / "qa"


def metadata_files() -> list[Path]:
    return sorted(GRIDDED_METADATA_DIR.glob("*.json"))


def normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def metadata_get(metadata, key: str, default=None):
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return default


def selected_metadata_files(requested: Optional[Iterable[str]] = None) -> list[Path]:
    paths = metadata_files()
    if not requested:
        return paths

    requested_normalized = {normalize_dataset_name(name) for name in requested}
    selected: list[Path] = []
    for metadata_file in paths:
        collection = DataCollection.from_file(metadata_file)
        name = collection.global_attributes["name"]
        stem = metadata_file.stem
        candidates = {normalize_dataset_name(name), normalize_dataset_name(stem)}
        if requested_normalized & candidates:
            selected.append(metadata_file)

    missing = requested_normalized - {
        candidate
        for metadata_file in selected
        for candidate in (
            normalize_dataset_name(DataCollection.from_file(metadata_file).global_attributes["name"]),
            normalize_dataset_name(metadata_file.stem),
        )
    }
    if missing:
        available = ", ".join(DataCollection.from_file(path).global_attributes["name"] for path in paths)
        raise RuntimeError(f"Unknown gridded SST dataset selector(s): {sorted(missing)}. Available: {available}")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_name(names, candidates) -> Optional[str]:
    lowered = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _choose_gridded_variable(dataset: xr.Dataset, lat_name: Optional[str], lon_name: Optional[str]) -> Optional[str]:
    if lat_name is None or lon_name is None:
        return None
    for name, variable in dataset.data_vars.items():
        if not np.issubdtype(variable.dtype, np.number):
            continue
        dims = set(str(dim) for dim in variable.dims)
        if lat_name in dims and lon_name in dims:
            return str(name)
    return None


def _time_bounds(dataset: xr.Dataset, time_name: Optional[str]) -> tuple[str, str, int]:
    if time_name is None or time_name not in dataset:
        return "", "", 0
    values = dataset[time_name].values
    n_time = int(values.size)
    if n_time == 0:
        return "", "", 0
    first = str(values[0])
    last = str(values[-1])
    return first, last, n_time


def _dimension_length(dataset: xr.Dataset, name: Optional[str]) -> int:
    if name is None:
        return 0
    if name in dataset.dims:
        return int(dataset.sizes[name])
    if name in dataset.coords:
        return int(dataset.coords[name].size)
    return 0


def source_value_type(metadata) -> str:
    actual = metadata_get(metadata, "actual")
    if isinstance(actual, (bool, np.bool_)):
        return "actual" if actual else "anomaly"
    return "unknown"


def inspect_gridded_source(
    source_file: Path,
    metadata,
    dataset_name: str,
    collection_name: str,
    metadata_file: Path,
    download_time_utc: str,
) -> dict[str, object]:
    source_file = Path(source_file)
    base_record = {
        "dataset": dataset_name,
        "collection": collection_name,
        "metadata_file": str(metadata_file),
        "source_file": str(source_file),
        "file_format": source_file.suffix.lower().lstrip(".") or "unknown",
        "variable_name": "",
        "lat_name": "",
        "lon_name": "",
        "time_name": "",
        "n_time": 0,
        "n_lat": 0,
        "n_lon": 0,
        "first_time": "",
        "last_time": "",
        "nominal_space_resolution": metadata_get(metadata, "space_resolution", ""),
        "crs": "",
        "source_value_type": source_value_type(metadata),
        "is_area_averaged": True,
        "eligible_for_regional_processing": False,
        "reason": "",
        "checksum": sha256(source_file) if source_file.exists() else "",
        "download_time_utc": download_time_utc,
    }

    if not source_file.exists():
        base_record["reason"] = "source file not found"
        return base_record
    if source_file.suffix.lower() != ".nc":
        base_record["reason"] = "source file is not NetCDF"
        return base_record

    try:
        with xr.open_dataset(source_file) as dataset:
            names = list(dataset.coords) + list(dataset.dims)
            lat_name = _find_name(names, LAT_NAMES)
            lon_name = _find_name(names, LON_NAMES)
            time_name = _find_name(names, TIME_NAMES)
            variable_name = _choose_gridded_variable(dataset, lat_name, lon_name)
            first_time, last_time, n_time = _time_bounds(dataset, time_name)
            n_lat = _dimension_length(dataset, lat_name)
            n_lon = _dimension_length(dataset, lon_name)
            crs = "EPSG:4326" if lat_name and lon_name else ""
    except Exception as exc:
        base_record["reason"] = f"could not inspect NetCDF: {exc}"
        return base_record

    reasons = []
    if lat_name is None:
        reasons.append("latitude coordinate not found")
    if lon_name is None:
        reasons.append("longitude coordinate not found")
    if time_name is None:
        reasons.append("time coordinate not found")
    if variable_name is None:
        reasons.append("no numeric data variable with latitude and longitude dimensions")
    if metadata_get(metadata, "type") != "gridded":
        reasons.append(f"type={metadata_get(metadata, 'type')!r} (need 'gridded')")
    if metadata_get(metadata, "space_resolution") == 999:
        reasons.append("space_resolution=999 (time-series sentinel)")
    if n_lat <= 1 or n_lon <= 1:
        reasons.append("not a two-dimensional latitude-longitude field")

    is_area_averaged = bool(lat_name is None or lon_name is None or n_lat <= 1 or n_lon <= 1)
    eligible = not reasons and not is_area_averaged
    base_record.update(
        {
            "variable_name": variable_name or "",
            "lat_name": lat_name or "",
            "lon_name": lon_name or "",
            "time_name": time_name or "",
            "n_time": n_time,
            "n_lat": n_lat,
            "n_lon": n_lon,
            "first_time": first_time,
            "last_time": last_time,
            "crs": crs,
            "is_area_averaged": is_area_averaged,
            "eligible_for_regional_processing": bool(eligible),
            "reason": "; ".join(reasons),
        }
    )
    return base_record


def expected_source_files(collection: DataCollection) -> list[Path]:
    collection_dir = gridded_data_dir() / collection.global_attributes["name"]
    paths: list[Path] = []
    for dataset in collection.datasets:
        for filename in dataset.metadata["filename"]:
            paths.append(collection_dir / Path(filename).name)
    return paths


def download_or_stage(collection: DataCollection, metadata_file: Path) -> list[dict[str, object]]:
    base_dir = gridded_data_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    collection_dir = base_dir / collection.global_attributes["name"]
    collection_dir.mkdir(parents=True, exist_ok=True)
    before = {
        path: {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        for path in expected_source_files(collection)
    }
    collection.download(base_dir)

    download_time_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for dataset in collection.datasets:
        for url, filename in zip(dataset.metadata["url"], dataset.metadata["filename"]):
            source_file = collection_dir / Path(filename).name
            if not source_file.exists():
                status = "missing"
            elif not before.get(source_file, {}).get("exists", False):
                status = "downloaded"
            elif before[source_file]["bytes"] != source_file.stat().st_size:
                status = "refreshed"
            else:
                status = "present_local"
            rows.append(
                {
                    "dataset": collection.global_attributes["name"],
                    "collection": collection.global_attributes["name"],
                    "metadata_file": str(metadata_file),
                    "source_url": url,
                    "source_file": str(source_file),
                    "status": status,
                    "bytes": source_file.stat().st_size if source_file.exists() else 0,
                    "checksum": sha256(source_file) if source_file.exists() else "",
                    "download_time_utc": download_time_utc,
                }
            )
    return rows


def build_inventory(collection: DataCollection, metadata_file: Path, download_time_utc: str) -> list[dict[str, object]]:
    rows = []
    collection_dir = gridded_data_dir() / collection.global_attributes["name"]
    for dataset in collection.datasets:
        for filename in dataset.metadata["filename"]:
            source_file = collection_dir / Path(filename).name
            rows.append(
                inspect_gridded_source(
                    source_file=source_file,
                    metadata=dataset.metadata,
                    dataset_name=collection.global_attributes["name"],
                    collection_name=collection.global_attributes["name"],
                    metadata_file=metadata_file,
                    download_time_utc=download_time_utc,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_download(datasets: Optional[list[str]] = None, strict: bool = False) -> tuple[Path, Path]:
    selected = selected_metadata_files(datasets)
    if not selected:
        raise RuntimeError(f"No gridded SST metadata files found in {GRIDDED_METADATA_DIR}.")

    download_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    inspection_time_utc = datetime.now(timezone.utc).isoformat()
    for metadata_file in selected:
        collection = DataCollection.from_file(metadata_file)
        download_rows.extend(download_or_stage(collection, metadata_file))
        inventory_rows.extend(build_inventory(collection, metadata_file, inspection_time_utc))

    qa_dir = gridded_qa_dir()
    download_log = qa_dir / DOWNLOAD_LOG_NAME
    inventory_log = qa_dir / INVENTORY_NAME
    write_csv(
        download_log,
        download_rows,
        [
            "dataset", "collection", "metadata_file", "source_url", "source_file",
            "status", "bytes", "checksum", "download_time_utc",
        ],
    )
    write_csv(
        inventory_log,
        inventory_rows,
        [
            "dataset", "collection", "metadata_file", "source_file", "file_format",
            "variable_name", "lat_name", "lon_name", "time_name", "n_time", "n_lat",
            "n_lon", "first_time", "last_time", "nominal_space_resolution", "crs",
            "source_value_type", "is_area_averaged", "eligible_for_regional_processing",
            "reason", "checksum", "download_time_utc",
        ],
    )

    if strict:
        failed = [row for row in inventory_rows if not row["eligible_for_regional_processing"]]
        if failed:
            details = "; ".join(f"{row['dataset']}: {row['reason']}" for row in failed)
            raise RuntimeError(f"Strict gridded SST validation failed: {details}")
    return download_log, inventory_log


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Optional gridded collection names to download.")
    parser.add_argument("--strict", action="store_true", help="Fail if any selected gridded source is ineligible.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    download_log, inventory_log = run_download(args.datasets, args.strict)
    print(f"Wrote gridded SST download log to {download_log}")
    print(f"Wrote gridded SST source inventory to {inventory_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
