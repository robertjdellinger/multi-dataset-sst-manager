from pathlib import Path
from typing import List

import xarray as xr

import climind.data_types.grid as gd
from climind.data_manager.metadata import CombinedMetadata
from climind.readers.generic_reader import get_last_modified_time


LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
TIME_NAMES = ("time", "date", "t")


def _find_name(names, candidates):
    lowered = {str(name).lower(): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _choose_gridded_variable(dataset: xr.Dataset, lat_name: str, lon_name: str) -> str:
    for name, variable in dataset.data_vars.items():
        dims = set(variable.dims)
        if lat_name in dims and lon_name in dims:
            return name
    raise RuntimeError("No gridded SST variable with latitude and longitude dimensions was found.")


def _open_standard_grid(filename: Path, metadata: CombinedMetadata) -> gd.GridMonthly:
    dataset = xr.open_dataset(filename)
    lat_name = _find_name(list(dataset.coords) + list(dataset.dims), LAT_NAMES)
    lon_name = _find_name(list(dataset.coords) + list(dataset.dims), LON_NAMES)
    time_name = _find_name(list(dataset.coords) + list(dataset.dims), TIME_NAMES)
    if lat_name is None or lon_name is None or time_name is None:
        dataset.close()
        raise RuntimeError("Gridded SST file must contain time, latitude, and longitude coordinates.")

    variable_name = _choose_gridded_variable(dataset, lat_name, lon_name)
    dataset = dataset[[variable_name]].rename({lat_name: "lat", lon_name: "lon", time_name: "time"})
    dataset = dataset.rename({variable_name: "tas_mean"})
    metadata.dataset["last_modified"] = [get_last_modified_time(filename)]
    metadata.creation_message()
    return gd.GridMonthly(dataset, metadata)


def read_ts(out_dir: Path, metadata: CombinedMetadata, **kwargs) -> gd.GridMonthly:
    return read_monthly_grid(out_dir / metadata["filename"][0], metadata, **kwargs)


def read_monthly_grid(filename: Path | List[Path], metadata: CombinedMetadata, **kwargs) -> gd.GridMonthly:
    if isinstance(filename, list):
        filename = filename[0]
    return _open_standard_grid(Path(filename), metadata)
