"""Synthetic tests for the optional SST MEOW/PPOW regional core logic.

These tests use tiny in-memory grids and a two-polygon GeoDataFrame so they run
without downloading MEOW/PPOW shapefiles or large NetCDF files.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sea_surface_temperature"
    / "sst_regional_core.py"
)


def _load_core():
    spec = importlib.util.spec_from_file_location("sst_regional_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_core()


def test_cosine_latitude_weights():
    weights = core.cosine_latitude_weights([0.0, 60.0])
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)


def test_cosine_latitude_weights_rejects_out_of_range():
    with pytest.raises(ValueError):
        core.cosine_latitude_weights([0.0, 120.0])


def test_area_weighted_mean_uses_cosine_weights():
    values = np.array([[1.0, 1.0], [3.0, 3.0]])
    stats = core.area_weighted_mean(values, [0.0, 60.0])
    # (1*1 + 1*1 + 3*0.5 + 3*0.5) / (1 + 1 + 0.5 + 0.5) = 5/3
    assert stats["mean"] == pytest.approx(5.0 / 3.0)
    assert stats["n_cells_total"] == 4
    assert stats["n_cells_valid"] == 4
    assert stats["valid_cell_fraction"] == pytest.approx(1.0)


def test_area_weighted_mean_excludes_missing_cells():
    values = np.array([[1.0, 1.0], [3.0, np.nan]])
    stats = core.area_weighted_mean(values, [0.0, 60.0])
    # (1*1 + 1*1 + 3*0.5) / (1 + 1 + 0.5) = 3.5 / 2.5
    assert stats["mean"] == pytest.approx(3.5 / 2.5)
    assert stats["n_cells_valid"] == 3
    assert stats["valid_cell_fraction"] == pytest.approx(0.75)


def test_area_weighted_mean_respects_region_mask():
    values = np.array([[1.0, 1.0], [3.0, 3.0]])
    mask = np.array([[True, True], [False, False]])
    stats = core.area_weighted_mean(values, [0.0, 60.0], mask=mask)
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["n_cells_total"] == 2
    assert stats["n_cells_valid"] == 2


def test_annual_from_monthly_requires_coverage():
    full = list(range(1, 13))
    assert core.annual_from_monthly(full) == pytest.approx(6.5)

    eleven = [float(v) for v in range(1, 12)] + [np.nan]
    assert np.isnan(core.annual_from_monthly(eleven, min_months=12))
    assert core.annual_from_monthly(eleven, min_months=6) == pytest.approx(
        np.mean(range(1, 12))
    )


def test_eligibility_rejects_timeseries_and_accepts_gridded():
    timeseries = {"type": "timeseries", "time_resolution": "monthly", "space_resolution": 999}
    gridded = {"type": "gridded", "time_resolution": "monthly", "space_resolution": 2}

    assert core.classify_regional_eligibility(timeseries)["eligible"] is False
    assert core.classify_regional_eligibility(gridded)["eligible"] is True


def test_assert_regional_eligible_blocks_global_mean_files():
    # A space_resolution=999 / timeseries file must never be regionalised.
    with pytest.raises(ValueError):
        core.assert_regional_eligible(
            {"type": "timeseries", "time_resolution": "annual", "space_resolution": 999}
        )


def _two_box_geodataframe():
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    box = shapely_geometry.box
    west = box(-90.0, -90.0, 0.0, 90.0)
    east = box(0.0, -90.0, 90.0, 90.0)
    return gpd.GeoDataFrame(
        {"REG_ID": ["A", "B"], "REG_NAME": ["West", "East"], "geometry": [west, east]},
        crs="EPSG:4326",
    )


def test_regional_means_assign_cells_to_correct_region():
    pytest.importorskip("regionmask")
    gdf = _two_box_geodataframe()

    latitudes = [10.0, 20.0]
    longitudes = [-50.0, 50.0]  # west cell, east cell
    values = np.array([[2.0, 4.0], [6.0, 8.0]])  # (lat, lon)

    rows = core.regional_means_for_field(
        values, latitudes, longitudes, gdf, "REG_ID", "REG_NAME"
    )
    by_id = {row["region_id"]: row for row in rows}

    w = core.cosine_latitude_weights(latitudes)
    west_expected = (2.0 * w[0] + 6.0 * w[1]) / (w[0] + w[1])
    east_expected = (4.0 * w[0] + 8.0 * w[1]) / (w[0] + w[1])

    assert by_id["A"]["sst_anomaly_C"] == pytest.approx(west_expected)
    assert by_id["B"]["sst_anomaly_C"] == pytest.approx(east_expected)
    assert by_id["A"]["region_name"] == "West"


def test_validate_geodataframe_requires_4326():
    gpd = pytest.importorskip("geopandas")
    gdf = _two_box_geodataframe().to_crs(epsg=3857)
    with pytest.raises(ValueError):
        core.assert_geodataframe_is_4326(gdf)
