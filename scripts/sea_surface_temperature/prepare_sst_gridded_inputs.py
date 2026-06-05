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

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sst_regional_core as core  # noqa: E402


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


def main() -> int:
    print(
        "prepare_sst_gridded_inputs: reusable standardisation helpers for the "
        "optional SST regional workflow.\n"
        "Provide true gridded SST NetCDF inputs and call standardize_grid / "
        "annualize_monthly_grid from your driver script, or from "
        "calculate_sst_meow_ppow_averages.py. No default gridded SST source is "
        "wired here because the six required outputs are time series."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
