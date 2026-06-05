#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/sea_surface_temperature/sst_regional_core.py
#
#  Reusable, testable core logic for the OPTIONAL SST regional workflow
#  (MEOW/PPOW marine biogeographic regions). This module deliberately contains
#  no heavy import-time side effects and no large-data IO so it can be exercised
#  by small synthetic tests. geopandas/regionmask/xarray are imported lazily
#  inside the functions that need them.
#
#  This is add-only scaffolding. The required six global SST CSV outputs do not
#  depend on this module. Regional outputs are produced only from TRUE
#  latitude-longitude gridded SST fields and only after the global workflow has
#  validated. WMO regions are intentionally NOT used here.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# Metadata sentinel used by Climind for non-gridded (time series) datasets.
TIMESERIES_SPACE_RESOLUTION = 999

_LAT_NAMES = {"lat", "latitude", "y"}
_LON_NAMES = {"lon", "longitude", "x"}
_TIME_NAMES = {"time", "date", "t"}


# ---------------------------------------------------------------------------
# Area weighting / averaging
# ---------------------------------------------------------------------------
def cosine_latitude_weights(latitudes: Sequence[float]) -> np.ndarray:
    """Return cosine-latitude weights for a 1-D latitude coordinate."""
    lat = np.asarray(latitudes, dtype=float)
    if lat.ndim != 1:
        raise ValueError("latitudes must be one-dimensional.")
    if np.nanmin(lat) < -90.0 - 1e-9 or np.nanmax(lat) > 90.0 + 1e-9:
        raise ValueError("latitudes must lie within [-90, 90].")
    return np.cos(np.deg2rad(lat))


def area_weighted_mean(
    values: np.ndarray,
    latitudes: Sequence[float],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Cosine-latitude area-weighted mean of a (lat, lon) field.

    Missing (non-finite) cells are excluded from both numerator and
    denominator. ``mask`` (boolean, True = include) optionally restricts the
    average to a region. Returns the mean plus coverage diagnostics.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (lat, lon); got shape {values.shape}.")
    lat = np.asarray(latitudes, dtype=float)
    if lat.size != values.shape[0]:
        raise ValueError(
            f"latitude length {lat.size} does not match values rows {values.shape[0]}."
        )

    weights = np.broadcast_to(
        cosine_latitude_weights(lat)[:, None], values.shape
    ).astype(float)

    region = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if region.shape != values.shape:
        raise ValueError("mask shape must match values shape.")

    cells_in_region = int(np.sum(region))
    valid = np.isfinite(values) & region & np.isfinite(weights) & (weights > 0)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            "mean": float("nan"),
            "n_cells_total": cells_in_region,
            "n_cells_valid": 0,
            "valid_cell_fraction": 0.0,
            "area_weight_sum": 0.0,
        }

    area_weight_sum = float(np.sum(weights[valid]))
    mean = float(np.sum(values[valid] * weights[valid]) / area_weight_sum)
    return {
        "mean": mean,
        "n_cells_total": cells_in_region,
        "n_cells_valid": n_valid,
        "valid_cell_fraction": n_valid / cells_in_region if cells_in_region else 0.0,
        "area_weight_sum": area_weight_sum,
    }


def annual_from_monthly(monthly_values: Sequence[float], min_months: int = 12) -> float:
    """Arithmetic annual mean of monthly values, requiring sufficient coverage.

    Returns NaN if fewer than ``min_months`` finite monthly values are present
    (equal weight per month, matching the upstream annualisation convention).
    """
    vals = np.asarray(monthly_values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < min_months:
        return float("nan")
    return float(np.mean(finite))


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def classify_regional_eligibility(metadata: Dict[str, object]) -> Dict[str, object]:
    """Classify whether a dataset is eligible for regional gridded processing.

    A dataset is eligible only if it is a true latitude-longitude grid:
    ``type == 'gridded'`` and ``space_resolution != 999``. Global-mean and
    area-averaged time series (the sentinel ``space_resolution == 999`` or
    ``type == 'timeseries'``) are rejected.
    """
    dataset_type = metadata.get("type")
    space_resolution = metadata.get("space_resolution")
    time_resolution = metadata.get("time_resolution")

    reasons: List[str] = []
    if dataset_type != "gridded":
        reasons.append(f"type={dataset_type!r} (need 'gridded')")
    if space_resolution == TIMESERIES_SPACE_RESOLUTION:
        reasons.append("space_resolution=999 (time-series sentinel)")

    eligible = dataset_type == "gridded" and space_resolution != TIMESERIES_SPACE_RESOLUTION
    return {
        "type": dataset_type,
        "time_resolution": time_resolution,
        "space_resolution": space_resolution,
        "eligible": bool(eligible),
        "exclusion_reason": "" if eligible else "; ".join(reasons),
    }


def inspect_netcdf_spatial(path: Path) -> Dict[str, bool]:
    """Report whether a NetCDF file exposes latitude, longitude, and time."""
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        names = {str(n).lower() for n in list(ds.coords) + list(ds.dims)}
    return {
        "has_lat": bool(names & _LAT_NAMES),
        "has_lon": bool(names & _LON_NAMES),
        "has_time": bool(names & _TIME_NAMES),
    }


def assert_regional_eligible(metadata: Dict[str, object]) -> None:
    """Raise if a dataset must not be regionalised (time series / global mean)."""
    verdict = classify_regional_eligibility(metadata)
    if not verdict["eligible"]:
        raise ValueError(
            "Refusing to compute regional averages from a non-gridded dataset: "
            + (verdict["exclusion_reason"] or "not a latitude-longitude grid")
        )


# ---------------------------------------------------------------------------
# Region masking (MEOW/PPOW polygons) and regional averaging
# ---------------------------------------------------------------------------
def build_region_mask_stack(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    regions_gdf,
):
    """Return a regionmask 3-D boolean mask (region, lat, lon) for polygons.

    ``regions_gdf`` is a geopandas GeoDataFrame in EPSG:4326. drop=False keeps
    all regions so the region axis aligns positionally with the GeoDataFrame
    rows (the region coordinate uses the GeoDataFrame index).
    """
    import regionmask

    return regionmask.mask_3D_geopandas(
        regions_gdf,
        np.asarray(longitudes, dtype=float),
        np.asarray(latitudes, dtype=float),
        drop=False,
        overlap=False,
    )


def regional_means_for_field(
    values: np.ndarray,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    regions_gdf,
    region_id_field: str,
    region_name_field: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Area-weighted regional means of a (lat, lon) field over MEOW/PPOW polygons."""
    assert_geodataframe_is_4326(regions_gdf)
    mask_stack = build_region_mask_stack(latitudes, longitudes, regions_gdf)

    rows: List[Dict[str, object]] = []
    for axis_index in range(mask_stack.region.size):
        cell_mask = np.asarray(mask_stack.isel(region=axis_index).values, dtype=bool)
        stats = area_weighted_mean(values, latitudes, mask=cell_mask)
        record = regions_gdf.iloc[axis_index]
        rows.append(
            {
                "region_id": record[region_id_field],
                "region_name": record[region_name_field] if region_name_field else "",
                "sst_anomaly_C": stats["mean"],
                "n_cells_total": stats["n_cells_total"],
                "n_cells_valid": stats["n_cells_valid"],
                "valid_cell_fraction": stats["valid_cell_fraction"],
                "area_weight_sum": stats["area_weight_sum"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Spatial-file validation (MEOW/PPOW)
# ---------------------------------------------------------------------------
def assert_geodataframe_is_4326(gdf) -> None:
    """Raise unless a GeoDataFrame has CRS EPSG:4326."""
    crs = getattr(gdf, "crs", None)
    if crs is None:
        raise ValueError("Spatial file has no CRS; expected EPSG:4326.")
    if int(getattr(crs, "to_epsg", lambda: -1)() or -1) != 4326:
        raise ValueError(f"Spatial file CRS is {crs}; reproject to EPSG:4326 first.")


def validate_spatial_file(
    path: Path,
    region_id_field: str,
    region_name_field: Optional[str] = None,
) -> Dict[str, object]:
    """Open a MEOW/PPOW vector file with geopandas and validate it for masking."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is not None and int(gdf.crs.to_epsg() or -1) != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geom_types = sorted(set(gdf.geom_type.dropna().unique()))
    polygonal = {"Polygon", "MultiPolygon"}
    is_polygonal = bool(set(geom_types) & polygonal)

    has_id = region_id_field in gdf.columns
    audit = {
        "source_file": str(path),
        "crs": str(gdf.crs),
        "geometry_count": int(len(gdf)),
        "geometry_types": ",".join(geom_types),
        "is_polygonal": is_polygonal,
        "region_id_field": region_id_field,
        "has_region_id_field": has_id,
        "region_name_field": region_name_field or "",
        "has_region_name_field": bool(region_name_field) and region_name_field in gdf.columns,
        "valid": bool(is_polygonal and has_id),
    }
    if not audit["valid"]:
        problems = []
        if not is_polygonal:
            problems.append(f"non-polygonal geometry ({geom_types})")
        if not has_id:
            problems.append(f"missing region id field {region_id_field!r}")
        audit["problem"] = "; ".join(problems)
    return audit
