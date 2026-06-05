#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#  Copyright (c) 2026 Robert J. Dellinger
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

from pathlib import Path
from typing import List

import xarray as xa

import climind.data_types.timeseries as ts
from climind.data_manager.metadata import CombinedMetadata
from climind.readers.generic_reader import read_ts


def read_monthly_ts(filename: List[Path], metadata: CombinedMetadata) -> ts.TimeSeriesMonthly:
    ds = xa.open_dataset(filename[0])
    if "sst_anomaly" not in ds:
        raise ValueError("CMEMS SST source file does not contain the required sst_anomaly variable.")

    years = ds.time.dt.year.data.astype(int).tolist()
    months = ds.time.dt.month.data.astype(int).tolist()
    data = ds["sst_anomaly"].values.astype(float).tolist()

    metadata.creation_message()
    return ts.TimeSeriesMonthly(years, months, data, metadata=metadata)


def read_annual_ts(filename: List[Path], metadata: CombinedMetadata) -> ts.TimeSeriesAnnual:
    return read_monthly_ts(filename, metadata).make_annual()
