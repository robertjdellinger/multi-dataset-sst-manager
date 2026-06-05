#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  Reader for the BHM SST paleo reconstruction (Ossandon et al.) full-field RDS,
#  reduced to a global-mean annual SST anomaly time series.
#
#  This is an optional, derived reader. The BHM source is an R serialized
#  ``.rds`` full-field reconstruction (1854-2014, May-Apr average) published on
#  Zenodo (https://zenodo.org/records/13993705). BHM is NOT one of the six
#  required reference SST outputs; it is an additional paleo record.
#
#  Requirements that are intentionally NOT vendored into the repository:
#    1. ``pyreadr`` must be installed to read the R ``.rds`` file
#       (``pip install pyreadr``).
#    2. The ``.rds`` file must be downloaded manually (the metadata uses
#       ``fetcher_no_url``) and placed in the dataset-managed directory.
#
#  Until both are present this reader raises a clear, actionable error rather
#  than failing silently. The dataset must therefore never be added to the
#  strict global SST builder's required set.
#
#  Reader interface follows the upstream Climind convention: the dataset
#  metadata names this module in its ``reader`` field, and for a
#  ``timeseries``/``annual`` dataset the generic reader dispatches to
#  ``read_annual_ts(filename, metadata)``.

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

import climind.data_types.timeseries as ts
from climind.data_manager.metadata import CombinedMetadata


_LAT_NAMES = ("lat", "latitude", "y")
_LON_NAMES = ("lon", "longitude", "x")
_YEAR_NAMES = ("year", "yr", "time")
_SST_NAMES = ("sst", "anomaly", "value", "temperature", "tas")


def _load_rds(path: Path):
    """Load an R ``.rds`` file, importing ``pyreadr`` lazily."""
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Reading the BHM full-field reconstruction requires the optional "
            "'pyreadr' package, which is not installed. Install it locally with "
            "'pip install pyreadr'. BHM is an optional paleo record and is not "
            "part of the six required SST outputs."
        ) from exc

    if not Path(path).exists():
        raise FileNotFoundError(
            f"BHM source file not found: {path}. The BHM metadata uses "
            "fetcher_no_url, so download "
            "'1854-2014_MAYtoAPRavg_SST_FullField.rds' from "
            "https://zenodo.org/records/13993705 and place it in the dataset's "
            "managed directory before reading."
        )

    return pyreadr.read_r(str(path))


def _pick(columns, candidates) -> str | None:
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _global_mean_from_long_frame(frame) -> "tuple[list[int], list[float]]":
    """Area-weighted (cosine-latitude) annual global mean from a long table."""
    lat_col = _pick(frame.columns, _LAT_NAMES)
    lon_col = _pick(frame.columns, _LON_NAMES)
    year_col = _pick(frame.columns, _YEAR_NAMES)
    sst_col = _pick(frame.columns, _SST_NAMES)

    if not (lat_col and lon_col and year_col and sst_col):
        raise NotImplementedError(
            "Could not identify lat/lon/year/sst columns in the BHM full-field "
            f"table (columns seen: {list(frame.columns)}). Inspect the .rds "
            "structure and extend reader_bhm_fullfield_to_global_mean_ts to "
            "match it."
        )

    work = frame[[year_col, lat_col, lon_col, sst_col]].copy()
    work.columns = ["year", "lat", "lon", "sst"]
    work = work.dropna(subset=["sst"])
    work["weight"] = np.cos(np.deg2rad(work["lat"].astype(float)))

    years: list[int] = []
    values: list[float] = []
    for year, block in work.groupby("year"):
        weight_sum = float(block["weight"].sum())
        if weight_sum <= 0:
            continue
        weighted = float((block["sst"].astype(float) * block["weight"]).sum())
        years.append(int(year))
        values.append(weighted / weight_sum)

    order = np.argsort(years)
    years = [years[i] for i in order]
    values = [values[i] for i in order]
    return years, values


def read_annual_ts(filename: List[Path], metadata: CombinedMetadata) -> ts.TimeSeriesAnnual:
    """Read the BHM full-field reconstruction and reduce it to a global-mean
    annual SST anomaly time series.

    The reconstruction is reduced with cosine-latitude area weighting over the
    finite (ocean) cells of the field. Baseline shifting and year-range
    selection are left to the normal Climind processing methods, as for the
    other SST datasets.
    """
    result = _load_rds(Path(filename[0]))

    # pyreadr returns an ordered dict of R object name -> pandas DataFrame.
    frames = [obj for obj in result.values() if obj is not None]
    if not frames:
        raise NotImplementedError(
            "The BHM .rds did not yield any tabular object via pyreadr. The "
            "full field may be stored as an R array/list that pyreadr cannot "
            "convert directly; export it to a tidy (year, lat, lon, sst) table "
            "or a NetCDF first, then extend this reader."
        )

    years, values = _global_mean_from_long_frame(frames[0])
    if not years:
        raise RuntimeError("BHM reduction produced no annual values.")

    metadata.creation_message()
    return ts.TimeSeriesAnnual(years, values, metadata=metadata)
