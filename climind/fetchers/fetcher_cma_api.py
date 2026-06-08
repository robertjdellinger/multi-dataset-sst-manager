#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  Fetcher for China Meteorological Administration CMDC API downloads.
#
#  This fetcher follows the upstream Climind fetcher interface:
#
#      fetch(url: str, outdir: Path, filename: str) -> None
#
#  The CMA CMDC API guidance page documents a Python 3 access program,
#  CMDCapi.zip, and the CMDCClient(...).retrieve(params) request pattern:
#
#      https://data.cma.cn/en/#/Visualization/cra-api
#
#  Direct SDK archive:
#
#      https://cdcv4staticfile.jiangsu-10.zos.ctyun.cn/space/cdcv4/pic/cmdcapi/CMDCapi.zip
#
#  Required local setup:
#
#  1. Download CMDCapi.zip from the CMA SDK Download Guidance page.
#  2. Extract CMDCapi.py so Python can import it.
#  3. Put CMA_USER_ID in climind/fetchers/.env, or export it in the shell.

from __future__ import annotations

import functools
import csv
import importlib
import importlib.util
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from climind.fetchers.fetcher_utils import time_tag_string


CMDCAPI_ZIP_URL = (
    "https://cdcv4staticfile.jiangsu-10.zos.ctyun.cn/space/cdcv4/pic/"
    "cmdcapi/CMDCapi.zip"
)
MONTH_COLUMNS = ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"]
VARIABLE_NAME_HINTS = ("anomaly", "sst", "temp", "temperature")


class CMASourceError(RuntimeError):
    """CMA source acquisition error with a machine-readable status."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _import_cmdcapi_from_file(path: Path):
    """Import CMDCapi.py from an explicit SDK file path."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ImportError(f"CMDCAPI_PATH does not point to a file: {path}")
    if path.name != "CMDCapi.py":
        raise ImportError(f"CMDCAPI_PATH must point to CMDCapi.py, got {path.name}.")

    spec = importlib.util.spec_from_file_location("CMDCapi", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["CMDCapi"] = module
    spec.loader.exec_module(module)
    return module


def resolve_cmdcapi_module():
    """
    Resolve the CMA CMDCapi module without committing the SDK.

    Search order:
    1. Existing Python import path.
    2. CMDCAPI_PATH, if set to a CMDCapi.py file.
    3. CMA_SDK_DIR, if set to a directory containing CMDCapi.py.
    """
    try:
        return importlib.import_module("CMDCapi")
    except ImportError as path_exc:
        import_errors = [path_exc]

    cmdcapi_path = os.getenv("CMDCAPI_PATH")
    if cmdcapi_path:
        try:
            return _import_cmdcapi_from_file(Path(cmdcapi_path))
        except ImportError as exc:
            import_errors.append(exc)

    sdk_dir = os.getenv("CMA_SDK_DIR")
    if sdk_dir:
        try:
            return _import_cmdcapi_from_file(Path(sdk_dir) / "CMDCapi.py")
        except ImportError as exc:
            import_errors.append(exc)

    local_sdk_file = Path(__file__).parent / "local_sdk" / "CMDCapi.py"
    if local_sdk_file.exists():
        try:
            return _import_cmdcapi_from_file(local_sdk_file)
        except ImportError as exc:
            import_errors.append(exc)

    detail = "; ".join(str(exc) for exc in import_errors if str(exc))
    raise ImportError(
        "CMA-SST source files are missing and CMDCapi.py is not importable. "
        "Install or expose the CMA CMDC API SDK using "
        "CMDCAPI_PATH=/path/to/CMDCapi.py or CMA_SDK_DIR=/path/to/sdk_directory."
        + (f" Import diagnostics: {detail}" if detail else "")
    )


def _load_cma_client():
    """Import the CMA CMDC SDK client only when the fetcher is used."""
    module = resolve_cmdcapi_module()
    try:
        return module.CMDCClient
    except AttributeError as exc:
        raise ImportError("CMDCapi.py does not expose CMDCClient.") from exc


def resolve_cma_credentials() -> dict[str, str]:
    """Resolve CMA credentials from the fetcher .env file or process env."""
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv()
    user_id = os.getenv("CMA_USER_ID")
    if not user_id:
        return {}
    return {"CMA_USER_ID": user_id}


def _year_range(start_year: int = 1850, end_year: int | None = 2025) -> Iterable[int]:
    """Return the inclusive year range to request from CMA."""
    if end_year is None:
        end_year = datetime.utcnow().year

    return range(start_year, end_year + 1)


def _safe_unzip_archives(download_dir: Path) -> None:
    """Unzip any ZIP archives created by the CMA SDK into the same directory."""
    for zip_path in download_dir.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(download_dir)
        except zipfile.BadZipFile:
            print(f"Warning: {zip_path} is not a valid ZIP file. Leaving it unchanged.")


def _find_candidate_files(download_dir: Path) -> list[Path]:
    """Find likely tabular CMA output files after SDK download."""
    suffixes = {".csv", ".txt", ".dat", ".xlsx", ".xls"}

    candidates = [
        path
        for path in download_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not path.name.startswith(".")
    ]

    return sorted(candidates)


def _find_cma_netcdf_files(download_dir: Path) -> list[Path]:
    """Find CMA monthly gridded NetCDF files returned by the CMDC API."""
    return sorted(
        path for path in download_dir.rglob("*.nc")
        if path.is_file() and not path.name.startswith(".")
    )


def _is_likely_cma_file(path: Path) -> bool:
    """Return True for filenames used by CMA/CMDC SST source artifacts."""
    name = path.name.lower()
    return any(
        token in name
        for token in ("cma", "cmst", "gmst", "mst", "surf_cli_glb", "china")
    )


def discover_local_cma_sources(search_roots: list[Path]) -> dict[str, list[Path]]:
    """Classify local CMA source/cache candidates under the supplied roots."""
    discovered: dict[str, list[Path]] = {
        "monthly_source_csv": [],
        "annual_csv": [],
        "netcdf": [],
    }

    for root in search_roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            path = Path(path)
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix == ".nc" and _is_likely_cma_file(path):
                discovered["netcdf"].append(path)
                continue
            if suffix != ".csv":
                continue
            if path.name == "CMA-SST_Global_Month_Temp_1981_2010.csv":
                validation = validate_cma_monthly_source_csv(path)
                if validation["valid"]:
                    discovered["monthly_source_csv"].append(path)
                continue
            validation = validate_cma_annual_csv(path)
            if validation["valid"]:
                discovered["annual_csv"].append(path)

    return {key: sorted(paths) for key, paths in discovered.items()}


def validate_cma_monthly_source_csv(path: Path) -> dict[str, object]:
    """Validate the monthly CMA source CSV expected by reader_cma_gmst."""
    path = Path(path)
    if not path.is_file():
        return {"valid": False, "reason": f"missing file: {path}"}

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return {"valid": False, "reason": f"could not read monthly CMA CSV: {exc}"}

    if "year" not in frame.columns:
        return {"valid": False, "reason": "monthly CMA CSV is missing the year column"}
    missing_months = [column for column in MONTH_COLUMNS if column not in frame.columns]
    if missing_months:
        return {
            "valid": False,
            "reason": f"monthly CMA CSV is missing columns: {', '.join(missing_months)}",
        }
    if frame.empty:
        return {"valid": False, "reason": "monthly CMA CSV has no rows"}

    years = pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int)
    if years.empty:
        return {"valid": False, "reason": "monthly CMA CSV has no numeric years"}

    return {
        "valid": True,
        "source_kind": "monthly_source_csv",
        "time_resolution": "monthly",
        "gridded_eligible": False,
        "year_start": int(years.min()),
        "year_end": int(years.max()),
        "rows": int(len(frame)),
        "source_value_type": "anomaly",
        "path": str(path),
    }


def validate_cma_annual_csv(path: Path) -> dict[str, object]:
    """Validate a derived annual/global CMA BADC-CSV without making it gridded."""
    path = Path(path)
    if not path.is_file():
        return {"valid": False, "reason": f"missing file: {path}", "gridded_eligible": False}

    try:
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        data_start = next(i for i, row in enumerate(rows) if row and row[0] == "data")
        data_end = next(i for i, row in enumerate(rows) if row and row[0] == "end data")
    except Exception as exc:
        return {
            "valid": False,
            "reason": f"not a valid CMA annual BADC CSV: {exc}",
            "gridded_eligible": False,
        }

    header = rows[data_start + 1]
    if "year" not in header or "data" not in header:
        return {
            "valid": False,
            "reason": "CMA annual BADC CSV must contain year and data columns",
            "gridded_eligible": False,
        }

    frame = pd.DataFrame(rows[data_start + 2:data_end], columns=header)
    years = pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int)
    if years.empty:
        return {
            "valid": False,
            "reason": "CMA annual BADC CSV contains no numeric years",
            "gridded_eligible": False,
        }

    return {
        "valid": True,
        "source_kind": "annual_badc_csv",
        "time_resolution": "annual",
        "gridded_eligible": False,
        "year_start": int(years.min()),
        "year_end": int(years.max()),
        "rows": int(len(frame)),
        "source_value_type": "anomaly",
        "path": str(path),
    }


def _infer_source_value_type(variable_name: str, attrs: dict, paths: list[Path]) -> str:
    """Infer actual/anomaly state only from explicit schema or CMA product cues."""
    text_parts = [variable_name]
    text_parts.extend(str(value) for value in attrs.values())
    text_parts.extend(path.name for path in paths)
    text = " ".join(text_parts).lower()
    if "anomaly" in text or "anom" in text or "product 16" in text or "mst" in text:
        return "anomaly"
    if "absolute" in text or "actual" in text:
        return "actual"
    return "unknown"


def validate_cma_gridded_netcdf_files(paths: list[Path]) -> dict[str, object]:
    """Validate local CMA NetCDF files as true gridded latitude-longitude sources."""
    paths = sorted(Path(path) for path in paths)
    if not paths:
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": "no CMA NetCDF files supplied",
        }

    records = []
    variable_names: set[str] = set()
    lat_names: set[str] = set()
    lon_names: set[str] = set()
    units: set[str] = set()
    attrs_for_state: dict[str, object] = {}
    try:
        for path in paths:
            with xr.open_dataset(path, decode_times=True) as dataset:
                variable_name = _identify_cma_data_variable(dataset)
                variable = dataset[variable_name]
                lat_name, lon_name = _infer_lat_lon_names(dataset, variable)
                if dataset[lat_name].size <= 1 or dataset[lon_name].size <= 1:
                    raise RuntimeError(
                        "CMA NetCDF latitude and longitude coordinates must each contain "
                        "more than one cell for gridded eligibility."
                    )
                monthly_grid = _select_monthly_grid(variable, lat_name, lon_name)
                year, month = _infer_year_month(path, dataset)
                variable_names.add(variable_name)
                lat_names.add(lat_name)
                lon_names.add(lon_name)
                if "units" in variable.attrs:
                    units.add(str(variable.attrs["units"]))
                attrs_for_state.update(variable.attrs)
                records.append(
                    {
                        "path": path,
                        "year": year,
                        "month": month,
                        "shape": tuple(int(size) for size in monthly_grid.shape),
                    }
                )
    except Exception as exc:
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": f"missing latitude and longitude gridded CMA schema: {exc}",
        }

    if len(variable_names) != 1:
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": f"inconsistent CMA NetCDF variables: {sorted(variable_names)}",
        }
    if len(lat_names) != 1 or len(lon_names) != 1:
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": "inconsistent CMA NetCDF latitude and longitude coordinate names",
        }

    data = pd.DataFrame(records)
    duplicates = data[data.duplicated(["year", "month"], keep=False)]
    if not duplicates.empty:
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": "duplicate CMA NetCDF year-month files found",
        }

    variable_name = next(iter(variable_names))
    source_value_type = _infer_source_value_type(variable_name, attrs_for_state, paths)
    if source_value_type == "unknown":
        return {
            "valid": False,
            "gridded_eligible": False,
            "reason": "CMA NetCDF source value type is not documented as actual or anomaly",
        }

    return {
        "valid": True,
        "gridded_eligible": True,
        "source_kind": "monthly_gridded_netcdf",
        "time_resolution": "monthly",
        "source_value_type": source_value_type,
        "variable_name": variable_name,
        "lat_name": next(iter(lat_names)),
        "lon_name": next(iter(lon_names)),
        "months": int(len(data)),
        "year_start": int(data["year"].min()),
        "year_end": int(data["year"].max()),
        "units": ";".join(sorted(units)) if units else "unspecified",
        "files": [str(path) for path in paths],
    }


def _parse_year_month_from_filename(path: Path) -> tuple[int, int]:
    """Parse a YYYYMM token from a CMA monthly NetCDF filename."""
    matches = re.findall(r"(?<!\d)((?:18|19|20)\d{2})(0[1-9]|1[0-2])(?!\d)", path.name)
    if not matches:
        raise RuntimeError(f"Could not infer year and month from CMA filename {path.name}.")

    year_text, month_text = matches[-1]
    return int(year_text), int(month_text)


def _infer_year_month(path: Path, dataset: xr.Dataset) -> tuple[int, int]:
    """Infer the monthly timestamp from a time coordinate or from the filename."""
    time_names = [
        name for name in list(dataset.coords) + list(dataset.dims)
        if name.lower() in {"time", "date"}
    ]

    for name in time_names:
        if name not in dataset:
            continue
        values = np.asarray(dataset[name].values).reshape(-1)
        if values.size == 1:
            timestamp = pd.to_datetime(values[0])
            return int(timestamp.year), int(timestamp.month)

    return _parse_year_month_from_filename(path)


def _infer_lat_lon_names(dataset: xr.Dataset, variable: xr.DataArray) -> tuple[str, str]:
    """Identify latitude and longitude coordinate names for a CMA grid."""
    lat_candidates = {"lat", "latitude", "y"}
    lon_candidates = {"lon", "longitude", "x"}

    lat_names = [
        name for name in list(dataset.coords) + list(variable.dims)
        if name.lower() in lat_candidates and name in dataset
    ]
    lon_names = [
        name for name in list(dataset.coords) + list(variable.dims)
        if name.lower() in lon_candidates and name in dataset
    ]

    if not lat_names or not lon_names:
        raise RuntimeError(
            "Could not identify latitude and longitude coordinates in CMA NetCDF schema. "
            f"Coordinates: {list(dataset.coords)}; variable dims: {variable.dims}."
        )

    return lat_names[0], lon_names[0]


def _identify_cma_data_variable(dataset: xr.Dataset) -> str:
    """Identify the SST/anomaly variable in a CMA monthly gridded NetCDF file."""
    candidates: list[str] = []
    for name, variable in dataset.data_vars.items():
        if not np.issubdtype(variable.dtype, np.number):
            continue
        if variable.ndim < 2:
            continue
        try:
            _infer_lat_lon_names(dataset, variable)
        except RuntimeError:
            continue
        candidates.append(name)

    if not candidates:
        raise RuntimeError(
            "Could not identify a numeric gridded SST/anomaly variable in CMA NetCDF schema. "
            f"Data variables: {list(dataset.data_vars)}."
        )

    hinted = [
        name for name in candidates
        if any(hint in name.lower() for hint in VARIABLE_NAME_HINTS)
    ]
    if len(hinted) == 1:
        return hinted[0]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "CMA NetCDF schema contains multiple possible gridded variables; "
        f"refusing to guess among {candidates}."
    )


def _select_monthly_grid(variable: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """Return a two-dimensional latitude-longitude grid from a monthly CMA variable."""
    for dim in list(variable.dims):
        if dim in {lat_name, lon_name}:
            continue
        if variable.sizes[dim] != 1:
            raise RuntimeError(
                f"CMA variable {variable.name} has non-singleton extra dimension {dim}; "
                "a verified multi-time parser is required before aggregation."
            )
        variable = variable.isel({dim: 0})

    if lat_name not in variable.dims or lon_name not in variable.dims:
        raise RuntimeError(
            f"CMA variable {variable.name} does not contain both latitude and longitude dimensions."
        )

    return variable.transpose(lat_name, lon_name)


def _mask_missing_values(values: np.ndarray, attrs: dict) -> np.ndarray:
    """Mask non-finite and explicitly declared missing values."""
    masked = np.asarray(values, dtype=float)
    missing_values = []
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            missing_values.extend(np.asarray(attrs[key]).reshape(-1).tolist())

    finite = np.isfinite(masked)
    for missing in missing_values:
        finite &= ~np.isclose(masked, float(missing), equal_nan=True)

    return np.where(finite, masked, np.nan)


def _cosine_latitude_weighted_mean(values: np.ndarray, latitudes: np.ndarray) -> float:
    """Compute a cosine-latitude weighted mean over finite grid cells."""
    if values.ndim != 2:
        raise RuntimeError(f"Expected a two-dimensional CMA grid, got shape {values.shape}.")

    latitudes = np.asarray(latitudes, dtype=float)
    if latitudes.ndim != 1:
        raise RuntimeError("CMA latitude coordinate must be one-dimensional for this aggregation.")
    if latitudes.size != values.shape[0]:
        raise RuntimeError(
            f"CMA latitude coordinate length {latitudes.size} does not match grid shape {values.shape}."
        )
    if np.nanmin(latitudes) < -90.0 or np.nanmax(latitudes) > 90.0:
        raise RuntimeError("CMA latitude coordinate contains values outside [-90, 90].")

    weights = np.cos(np.deg2rad(latitudes))[:, None]
    weights = np.broadcast_to(weights, values.shape)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not np.any(valid):
        raise RuntimeError("CMA grid contains no finite weighted cells.")

    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


# CMA product 16 (CMA-GMST) is a land-ocean MERGED surface-temperature anomaly
# field, so the raw 2-degree grid carries finite anomalies over land. Averaging
# that field directly would yield a land-contaminated (amplitude-inflated) series,
# not a sea-surface-temperature series. This ocean filtering is only for the
# CMA-GMST fallback path; a verified standalone CMA-SST product is already oceanic
# and should not be masked again. The validated strict core currently uses the
# same Natural Earth land definition that upstream climind.data_types.grid uses
# for its own land masking. A deliberate CMA-GMST separation sensitivity pass can
# replace this with an externally managed fractional land-ocean mask, such as the
# ORNL DAAC ISLSCP II ancillary mask, after those files are acquired and audited.
OCEAN_MASK_SOURCE = "regionmask.defined_regions.natural_earth_v5_0_0.land_110"


@functools.lru_cache(maxsize=8)
def _ocean_cell_mask_cached(
    lat_bytes: bytes, lon_bytes: bytes, n_lat: int, n_lon: int
) -> np.ndarray:
    """Return a boolean ocean mask (True over ocean) for a fixed lat/lon grid.

    The mask depends only on the grid geometry, which is identical for every CMA
    monthly file, so it is cached and computed once per run.
    """
    import regionmask

    latitudes = np.frombuffer(lat_bytes, dtype=float).reshape(n_lat)
    longitudes = np.frombuffer(lon_bytes, dtype=float).reshape(n_lon)
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    # drop=False keeps the (single) land region even when no grid cell falls on
    # land, so .sel(region=0) is always valid; such cells are simply all-ocean.
    land_mask = np.asarray(
        land.mask_3D(longitudes, latitudes, drop=False).sel(region=0).values, dtype=bool
    )
    return ~land_mask


def _ocean_cell_mask(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Boolean ocean mask aligned to a (latitude, longitude) CMA grid."""
    latitudes = np.ascontiguousarray(np.asarray(latitudes, dtype=float))
    longitudes = np.ascontiguousarray(np.asarray(longitudes, dtype=float))
    ocean = _ocean_cell_mask_cached(
        latitudes.tobytes(), longitudes.tobytes(), latitudes.size, longitudes.size
    )
    if ocean.shape != (latitudes.size, longitudes.size):
        raise RuntimeError(
            "CMA ocean mask shape "
            f"{ocean.shape} does not match grid ({latitudes.size}, {longitudes.size})."
        )
    return ocean


def _aggregate_cma_netcdf_file(path: Path) -> dict[str, object]:
    """Inspect and aggregate one CMA monthly NetCDF grid (ocean cells only)."""
    with xr.open_dataset(path) as dataset:
        variable_name = _identify_cma_data_variable(dataset)
        variable = dataset[variable_name]
        lat_name, lon_name = _infer_lat_lon_names(dataset, variable)
        year, month = _infer_year_month(path, dataset)
        monthly_grid = _select_monthly_grid(variable, lat_name, lon_name)
        values = _mask_missing_values(monthly_grid.values, monthly_grid.attrs)
        latitudes = np.asarray(dataset[lat_name].values, dtype=float)
        longitudes = np.asarray(dataset[lon_name].values, dtype=float)
        # Drop land cells from the merged field before area-weighting (see
        # OCEAN_MASK_SOURCE note above); _cosine_latitude_weighted_mean ignores NaN.
        ocean = _ocean_cell_mask(latitudes, longitudes)
        values = np.where(ocean, values, np.nan)
        mean_value = _cosine_latitude_weighted_mean(values, latitudes)

    return {
        "year": year,
        "month": month,
        "data": mean_value,
        "filename": path.name,
        "variable": variable_name,
        "latitude": lat_name,
        "longitude": lon_name,
        "ocean_masked": True,
        "ocean_mask_source": OCEAN_MASK_SOURCE,
    }


def write_aggregated_cma_netcdf_csv(
    download_dir: Path,
    out_path: Path,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, object]:
    """
    Aggregate CMA monthly gridded NetCDF files to the expected Climind CSV.

    The output format matches reader_cma_gmst.read_monthly_ts: one header row,
    year in the first column, and twelve monthly anomaly columns.
    """
    netcdf_files = _find_cma_netcdf_files(Path(download_dir))
    if not netcdf_files:
        raise FileNotFoundError(f"No CMA NetCDF files found in {download_dir}.")

    records = [_aggregate_cma_netcdf_file(path) for path in netcdf_files]
    data = pd.DataFrame(records)
    duplicates = data[data.duplicated(["year", "month"], keep=False)]
    if not duplicates.empty:
        duplicate_names = ", ".join(sorted(duplicates["filename"].tolist()))
        raise RuntimeError(f"Duplicate CMA monthly grids found for the same year-month: {duplicate_names}.")

    if start_year is None:
        start_year = int(data["year"].min())
    if end_year is None:
        end_year = int(data["year"].max())

    expected = {
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    }
    observed = set(zip(data["year"].astype(int), data["month"].astype(int)))
    missing = sorted(expected - observed)
    if missing:
        preview = ", ".join(f"{year}-{month:02d}" for year, month in missing[:24])
        suffix = "..." if len(missing) > 24 else ""
        raise RuntimeError(f"missing monthly CMA grids: {preview}{suffix}")

    data = data[(data["year"] >= start_year) & (data["year"] <= end_year)].copy()
    if data["data"].isna().any():
        raise RuntimeError("CMA monthly aggregation produced non-finite monthly means.")

    variable_names = sorted(data["variable"].unique())
    if len(variable_names) != 1:
        raise RuntimeError(f"CMA monthly files used inconsistent variables: {variable_names}.")

    pivot = data.pivot(index="year", columns="month", values="data").sort_index()
    pivot = pivot.loc[list(range(start_year, end_year + 1)), list(range(1, 13))]
    if pivot.isna().any().any():
        raise RuntimeError("CMA monthly aggregation produced an incomplete annual-month table.")

    output = pivot.reset_index()
    output.columns = ["year", *MONTH_COLUMNS]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, float_format="%.6f")

    return {
        "files": int(len(data)),
        "months": int(len(data)),
        "year_start": int(start_year),
        "year_end": int(end_year),
        "variable": variable_names[0],
        "latitude": str(data["latitude"].iloc[0]),
        "longitude": str(data["longitude"].iloc[0]),
        "ocean_masked": bool(data["ocean_masked"].all()) if "ocean_masked" in data else False,
        "ocean_mask_source": OCEAN_MASK_SOURCE,
        "min": float(data["data"].min()),
        "max": float(data["data"].max()),
    }


def _existing_year_months_from_netcdf(download_dir: Path) -> set[tuple[int, int]]:
    """Return year-month pairs already available as extracted CMA NetCDF files."""
    observed: set[tuple[int, int]] = set()
    for path in _find_cma_netcdf_files(download_dir):
        try:
            observed.add(_parse_year_month_from_filename(path))
        except RuntimeError:
            continue
    return observed


def _missing_cma_years(download_dir: Path, start_year: int, end_year: int) -> list[int]:
    """Return years with any missing monthly CMA NetCDF grid."""
    observed = _existing_year_months_from_netcdf(download_dir)
    missing_years = []
    for year in range(start_year, end_year + 1):
        if any((year, month) not in observed for month in range(1, 13)):
            missing_years.append(year)
    return missing_years


def _write_qc_sidecar(out_path: Path, qc: dict[str, object]) -> None:
    """Write a small local QC sidecar next to the generated CMA CSV."""
    sidecar = out_path.with_suffix(".qc.json")
    sidecar.write_text(pd.Series(qc).to_json(indent=2), encoding="utf-8")


def _copy_or_report_outputs(
    download_dir: Path,
    out_path: Path,
    start_year: int = 1850,
    end_year: int = 2025,
) -> None:
    """
    Copy the CMA API output into the Climind-managed filename only when the SDK
    returns one already usable table.

    If the CMA API returns monthly NetCDF grids, aggregate them only after the
    schema and year-month completeness checks pass.
    """
    candidates = _find_candidate_files(download_dir)

    if len(candidates) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidates[0], out_path)
        shutil.copyfile(out_path, out_path.parent / time_tag_string(out_path.name))
        return

    if len(candidates) == 0:
        qc = write_aggregated_cma_netcdf_csv(download_dir, out_path, start_year, end_year)
        _write_qc_sidecar(out_path, qc)
        shutil.copyfile(out_path, out_path.parent / time_tag_string(out_path.name))
        return

    candidate_list = "\n".join(str(path) for path in candidates)
    raise RuntimeError(
        "CMA API returned multiple candidate data files. This is expected if the "
        "API returns separate files by year or request. Leave the raw files in "
        "cma_api_raw and add a verified CMA-specific merge/parser step before "
        "writing the final Climind filename.\n\n"
        f"Candidate files:\n{candidate_list}"
    )


def ensure_cma_source_files(
    outdir: Path,
    filename: str,
    strict: bool = True,
    start_year: int = 1850,
    end_year: int = 2025,
) -> dict[str, object]:
    """
    Ensure the Climind-managed CMA source CSV exists.

    Resolution order:
    1. Reuse a validated managed monthly source CSV or complete local NetCDF cache.
    2. Import the CMDC API SDK.
    3. Use configured CMA credentials to download through CMDC API.
    4. Validate and aggregate the downloaded source files.
    """
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / filename
    raw_download_dir = outdir / "cma_api_raw"
    raw_download_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        validation = validate_cma_monthly_source_csv(out_path)
        if validation["valid"]:
            return {
                "status": "source_cached",
                "paths": [out_path],
                "validation": validation,
            }
        if strict:
            raise CMASourceError(
                "source_validation_failed",
                f"Existing CMA source file failed validation: {validation['reason']}",
            )

    cached_netcdf = _find_cma_netcdf_files(raw_download_dir)
    if cached_netcdf:
        validation = validate_cma_gridded_netcdf_files(cached_netcdf)
        if not validation["valid"]:
            if strict:
                raise CMASourceError(
                    "source_validation_failed",
                    f"Cached CMA NetCDF files failed validation: {validation['reason']}",
                )
        else:
            _safe_unzip_archives(raw_download_dir)
            missing_years = _missing_cma_years(raw_download_dir, start_year, end_year)
            if not missing_years:
                _copy_or_report_outputs(raw_download_dir, out_path, start_year, end_year)
                return {
                    "status": "source_cached",
                    "paths": cached_netcdf,
                    "validation": validation,
                }

    try:
        CMDCClient = _load_cma_client()
    except Exception as exc:
        message = (
            "CMA-SST source files are missing and CMDCapi.py is not importable. "
            "Install or expose the CMA CMDC API SDK using "
            "CMDCAPI_PATH=/path/to/CMDCapi.py or CMA_SDK_DIR=/path/to/sdk_directory. "
            "Existing processed outputs were preserved."
        )
        if strict:
            raise CMASourceError("source_missing_sdk", message) from exc
        return {"status": "source_missing_sdk", "paths": [], "reason": message}

    credentials = resolve_cma_credentials()
    if not credentials:
        message = (
            "CMA-SST source files are missing and CMA credentials are not configured. "
            "Set CMA_USER_ID in climind/fetchers/.env or the process environment. "
            "Existing processed outputs were preserved."
        )
        if strict:
            raise CMASourceError("source_missing_credentials", message)
        return {"status": "source_missing_credentials", "paths": [], "reason": message}

    client = CMDCClient(user_id=credentials["CMA_USER_ID"], output_dir=str(raw_download_dir))
    _safe_unzip_archives(raw_download_dir)
    years_to_request = _missing_cma_years(raw_download_dir, start_year, end_year)

    for year in years_to_request:
        params = {
            "isZip": "2",
            "day": "1",
            "month": "1,2,3,4,5,6,7,8,9,10,11,12",
            "year": str(year),
            "productId": "16",
            "source": "1",
        }

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"Requesting CMA-SST / CMA-GMST data for {year} "
                    f"(attempt {attempt}/{max_attempts})"
                )
                client.retrieve(params)
                break
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                print(
                    f"CMA request for {year} failed on attempt "
                    f"{attempt}/{max_attempts}: {exc}. Retrying."
                )
                time.sleep(5 * attempt)

    _safe_unzip_archives(raw_download_dir)
    try:
        _copy_or_report_outputs(raw_download_dir, out_path, start_year, end_year)
    except Exception as exc:
        if strict:
            raise CMASourceError(
                "source_validation_failed",
                f"CMA API download completed but source validation or aggregation failed: {exc}",
            ) from exc
        return {"status": "source_validation_failed", "paths": [], "reason": str(exc)}

    validation = validate_cma_monthly_source_csv(out_path)
    if not validation["valid"]:
        if strict:
            raise CMASourceError(
                "source_validation_failed",
                f"CMA managed source CSV failed validation after download: {validation['reason']}",
            )
        return {"status": "source_validation_failed", "paths": [out_path], "validation": validation}

    return {
        "status": "source_downloaded",
        "paths": [out_path],
        "validation": validation,
    }


def fetch(url: str, outdir: Path, filename: str) -> None:
    """
    Fetch CMA-SST / CMA-GMST data using the CMA CMDC API.

    The metadata URL is retained for source traceability. The actual download is
    performed through CMDCClient.retrieve using productId 16 and source 1.
    """
    result = ensure_cma_source_files(Path(outdir), filename, strict=True)
    out_path = Path(outdir) / filename
    print(f"CMA-SST data written to {out_path} ({result['status']})")
