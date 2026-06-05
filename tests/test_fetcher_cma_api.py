from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climind.fetchers.fetcher_cma_api import (
    _copy_or_report_outputs,
    _ocean_cell_mask,
    write_aggregated_cma_netcdf_csv,
)


def _write_cma_grid(path: Path, values: np.ndarray) -> None:
    ds = xr.Dataset(
        {"anomaly": (("lat", "lon"), values)},
        coords={"lat": np.array([0.0, 24.0, 60.0]), "lon": np.array([-150.0, 12.0])},
    )
    ds.to_netcdf(path)


def _write_cma_grid_with_scalar_month(path: Path, values: np.ndarray, month: int) -> None:
    ds = xr.Dataset(
        {"anomaly": (("lat", "lon"), values)},
        coords={
            "lat": np.array([0.0, 24.0, 60.0]),
            "lon": np.array([-150.0, 12.0]),
            "month": month,
        },
    )
    ds.to_netcdf(path)


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed.*:RuntimeWarning")
def test_write_aggregated_cma_netcdf_csv_uses_cosine_latitude_weights(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "CMA-SST_Global_Month_Temp_1981_2010.csv"

    base_grid = np.array(
        [
            [1.0, 3.0],
            [5.0, np.nan],
            [7.0, 9.0],
        ]
    )

    latitudes = np.array([0.0, 24.0, 60.0])
    longitudes = np.array([-150.0, 12.0])
    lat_weights = np.cos(np.deg2rad(latitudes))[:, None]
    # CMA product 16 is a land-ocean merged field, so aggregation now averages
    # ocean cells only; the expected value must apply the same mask.
    ocean = _ocean_cell_mask(latitudes, longitudes)

    for month in range(1, 13):
        grid = base_grid + month
        _write_cma_grid(raw_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_1850{month:02d}.nc", grid)

    qc = write_aggregated_cma_netcdf_csv(raw_dir, out_path)
    output = pd.read_csv(out_path)

    expected = []
    for month in range(1, 13):
        grid = np.where(ocean, base_grid + month, np.nan)
        finite = np.isfinite(grid)
        expected.append(float(np.sum(grid[finite] * np.broadcast_to(lat_weights, grid.shape)[finite]) /
                              np.sum(np.broadcast_to(lat_weights, grid.shape)[finite])))

    assert qc["ocean_masked"] is True

    assert qc["months"] == 12
    assert qc["year_start"] == 1850
    assert qc["year_end"] == 1850
    assert list(output.columns) == [
        "year", "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    assert output["year"].tolist() == [1850]
    assert np.allclose(output.iloc[0, 1:].to_numpy(dtype=float), expected)


def test_write_aggregated_cma_netcdf_csv_rejects_missing_months(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "CMA-SST_Global_Month_Temp_1981_2010.csv"

    grid = np.ones((3, 2))
    _write_cma_grid(raw_dir / "SURF_CLI_GLB_MST_MON_GRID_2DEG_185001.nc", grid)
    _write_cma_grid(raw_dir / "SURF_CLI_GLB_MST_MON_GRID_2DEG_185003.nc", grid)

    with pytest.raises(RuntimeError, match="missing monthly CMA grids"):
        write_aggregated_cma_netcdf_csv(raw_dir, out_path)


def test_copy_or_report_outputs_aggregates_netcdf_files_when_no_table_exists(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "CMA-SST_Global_Month_Temp_1981_2010.csv"

    grid = np.ones((3, 2))
    for month in range(1, 13):
        _write_cma_grid(raw_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_1850{month:02d}.nc", grid * month)

    _copy_or_report_outputs(raw_dir, out_path, start_year=1850, end_year=1850)

    output = pd.read_csv(out_path)
    assert output.shape == (1, 13)
    assert output.loc[0, "year"] == 1850
    assert output.loc[0, "jan"] == pytest.approx(1.0)
    assert output.loc[0, "dec"] == pytest.approx(12.0)
    assert out_path.with_suffix(".qc.json").exists()


def test_ocean_mask_excludes_land_cells_from_aggregation(tmp_path):
    # A mid-Pacific cell is ocean; a central-Sahara cell is land. With the merged
    # land-ocean field set so every ocean cell holds the same value and land cells
    # hold a wildly different value, the ocean-masked mean must equal the ocean
    # value exactly, proving land cells are dropped before averaging.
    latitudes = np.array([0.0, 24.0])
    longitudes = np.array([-150.0, 12.0])
    ocean = _ocean_cell_mask(latitudes, longitudes)
    assert ocean.any() and not ocean.all(), "test grid must contain both ocean and land"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "CMA-SST_Global_Month_Temp_1981_2010.csv"

    grid = np.where(ocean, 1.0, 999.0)
    for month in range(1, 13):
        ds = xr.Dataset(
            {"anomaly": (("lat", "lon"), grid)},
            coords={"lat": latitudes, "lon": longitudes},
        )
        ds.to_netcdf(raw_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_2000{month:02d}.nc")

    write_aggregated_cma_netcdf_csv(raw_dir, out_path, start_year=2000, end_year=2000)
    output = pd.read_csv(out_path)
    assert output.loc[0, "jan"] == pytest.approx(1.0)
    assert output.loc[0, "dec"] == pytest.approx(1.0)


def test_write_aggregated_cma_netcdf_csv_parses_year_from_filename_when_only_month_coord_exists(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "CMA-SST_Global_Month_Temp_1981_2010.csv"

    grid = np.ones((3, 2))
    for month in range(1, 13):
        _write_cma_grid_with_scalar_month(
            raw_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_2025{month:02d}.nc",
            grid * month,
            month,
        )

    qc = write_aggregated_cma_netcdf_csv(raw_dir, out_path, start_year=2025, end_year=2025)
    output = pd.read_csv(out_path)

    assert qc["year_start"] == 2025
    assert qc["year_end"] == 2025
    assert output["year"].tolist() == [2025]
    assert output.loc[0, "jan"] == pytest.approx(1.0)
    assert output.loc[0, "dec"] == pytest.approx(12.0)
