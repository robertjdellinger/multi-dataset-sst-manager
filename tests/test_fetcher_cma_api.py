from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climind.fetchers.fetcher_cma_api import (
    CMASourceError,
    _copy_or_report_outputs,
    _ocean_cell_mask,
    discover_local_cma_sources,
    ensure_cma_source_files,
    resolve_cmdcapi_module,
    validate_cma_annual_csv,
    validate_cma_gridded_netcdf_files,
    write_aggregated_cma_netcdf_csv,
)


def _write_cma_grid(path: Path, values: np.ndarray) -> None:
    ds = xr.Dataset(
        {"anomaly": (("lat", "lon"), values)},
        coords={"lat": np.array([0.0, 24.0, 60.0]), "lon": np.array([-150.0, 12.0])},
    )
    ds.to_netcdf(path)


def _write_cma_badc_annual(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Conventions,G,BADC-CSV,1",
                "history,G,Rebaselined to 1991-2020",
                "data",
                "time,year,data",
                "18262,1850,-0.6404",
                "18627,1851,-0.5922",
                "end data",
            ]
        )
    )


def _write_cma_monthly_source(path: Path) -> None:
    data = {"year": [1850]}
    data.update({month: [float(index)] for index, month in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )})
    pd.DataFrame(data).to_csv(path, index=False)


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


def test_discover_local_cma_sources_classifies_annual_csv_and_netcdf(tmp_path):
    annual = tmp_path / "sst_CMA_SST.csv"
    _write_cma_badc_annual(annual)
    grid = tmp_path / "SURF_CLI_GLB_MST_MON_GRID_2DEG_185001.nc"
    _write_cma_grid(grid, np.ones((3, 2)))

    discovered = discover_local_cma_sources([tmp_path])

    assert annual in discovered["annual_csv"]
    assert grid in discovered["netcdf"]
    assert discovered["monthly_source_csv"] == []


def test_validate_cma_annual_csv_is_not_gridded_eligible(tmp_path):
    annual = tmp_path / "sst_CMA_SST.csv"
    _write_cma_badc_annual(annual)

    result = validate_cma_annual_csv(annual)

    assert result["valid"] is True
    assert result["source_kind"] == "annual_badc_csv"
    assert result["time_resolution"] == "annual"
    assert result["gridded_eligible"] is False
    assert result["year_start"] == 1850
    assert result["year_end"] == 1851


def test_validate_cma_gridded_netcdf_files_reports_schema_and_source_state(tmp_path):
    for month in range(1, 13):
        _write_cma_grid(tmp_path / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_1850{month:02d}.nc", np.ones((3, 2)))

    result = validate_cma_gridded_netcdf_files(sorted(tmp_path.glob("*.nc")))

    assert result["valid"] is True
    assert result["gridded_eligible"] is True
    assert result["source_kind"] == "monthly_gridded_netcdf"
    assert result["time_resolution"] == "monthly"
    assert result["source_value_type"] == "anomaly"
    assert result["variable_name"] == "anomaly"
    assert result["lat_name"] == "lat"
    assert result["lon_name"] == "lon"
    assert result["months"] == 12


def test_validate_cma_gridded_netcdf_files_rejects_global_mean_file(tmp_path):
    path = tmp_path / "CMA_global_mean.nc"
    ds = xr.Dataset({"anomaly": (("time",), np.ones(12))}, coords={"time": pd.date_range("1850-01-01", periods=12, freq="MS")})
    ds.to_netcdf(path)

    result = validate_cma_gridded_netcdf_files([path])

    assert result["valid"] is False
    assert result["gridded_eligible"] is False
    assert "latitude and longitude" in result["reason"]


def test_resolve_cmdcapi_module_uses_explicit_file_path(monkeypatch, tmp_path):
    sdk = tmp_path / "CMDCapi.py"
    sdk.write_text(
        "class CMDCClient:\n"
        "    def __init__(self, user_id, output_dir):\n"
        "        self.user_id = user_id\n"
        "        self.output_dir = output_dir\n"
        "    def retrieve(self, params):\n"
        "        return None\n"
    )
    sys.modules.pop("CMDCapi", None)
    monkeypatch.setenv("CMDCAPI_PATH", str(sdk))
    monkeypatch.delenv("CMA_SDK_DIR", raising=False)

    module = resolve_cmdcapi_module()

    assert hasattr(module, "CMDCClient")
    sys.modules.pop("CMDCapi", None)


def test_ensure_cma_source_files_reuses_valid_monthly_cache_without_api(monkeypatch, tmp_path):
    filename = "CMA-SST_Global_Month_Temp_1981_2010.csv"
    _write_cma_monthly_source(tmp_path / filename)

    def fail_api():
        raise AssertionError("CMDC API should not be called when source cache is valid")

    monkeypatch.setattr("climind.fetchers.fetcher_cma_api._load_cma_client", fail_api)

    result = ensure_cma_source_files(tmp_path, filename, strict=True, start_year=1850, end_year=1850)

    assert result["status"] == "source_cached"
    assert result["paths"] == [tmp_path / filename]


def test_ensure_cma_source_files_missing_cache_and_sdk_reports_source_missing_sdk(monkeypatch, tmp_path):
    filename = "CMA-SST_Global_Month_Temp_1981_2010.csv"

    def missing_api():
        raise ImportError("missing test SDK")

    monkeypatch.setattr("climind.fetchers.fetcher_cma_api._load_cma_client", missing_api)

    with pytest.raises(CMASourceError) as excinfo:
        ensure_cma_source_files(tmp_path, filename, strict=True, start_year=1850, end_year=1850)

    assert excinfo.value.status == "source_missing_sdk"
    assert "CMDCapi.py is not importable" in str(excinfo.value)
    assert "Existing processed outputs were preserved" in str(excinfo.value)


def test_ensure_cma_source_files_missing_credentials_reports_source_missing_credentials(monkeypatch, tmp_path):
    filename = "CMA-SST_Global_Month_Temp_1981_2010.csv"

    class DummyClient:
        def __init__(self, user_id, output_dir):
            self.user_id = user_id
            self.output_dir = output_dir

    monkeypatch.setattr("climind.fetchers.fetcher_cma_api._load_cma_client", lambda: DummyClient)
    monkeypatch.setattr("climind.fetchers.fetcher_cma_api.resolve_cma_credentials", lambda: {})

    with pytest.raises(CMASourceError) as excinfo:
        ensure_cma_source_files(tmp_path, filename, strict=True, start_year=1850, end_year=1850)

    assert excinfo.value.status == "source_missing_credentials"
    assert "CMA credentials are not configured" in str(excinfo.value)


def test_ensure_cma_source_files_downloads_with_mocked_cmdc_api(monkeypatch, tmp_path):
    filename = "CMA-SST_Global_Month_Temp_1981_2010.csv"

    class DummyClient:
        def __init__(self, user_id, output_dir):
            self.output_dir = Path(output_dir)

        def retrieve(self, params):
            year = int(params["year"])
            for month in range(1, 13):
                _write_cma_grid(
                    self.output_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_{year}{month:02d}.nc",
                    np.ones((3, 2)) * month,
                )

    monkeypatch.setattr("climind.fetchers.fetcher_cma_api._load_cma_client", lambda: DummyClient)
    monkeypatch.setattr(
        "climind.fetchers.fetcher_cma_api.resolve_cma_credentials",
        lambda: {"CMA_USER_ID": "test-user"},
    )

    result = ensure_cma_source_files(tmp_path, filename, strict=True, start_year=1850, end_year=1850)

    assert result["status"] == "source_downloaded"
    assert (tmp_path / filename).exists()
    output = pd.read_csv(tmp_path / filename)
    assert output.shape == (1, 13)
    assert output.loc[0, "year"] == 1850


def test_ensure_cma_source_files_retries_transient_cmdc_api_failure(monkeypatch, tmp_path):
    filename = "CMA-SST_Global_Month_Temp_1981_2010.csv"
    calls = {"count": 0}

    class RetryClient:
        def __init__(self, user_id, output_dir):
            self.output_dir = Path(output_dir)

        def retrieve(self, params):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("transient timeout")
            year = int(params["year"])
            for month in range(1, 13):
                _write_cma_grid(
                    self.output_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_{year}{month:02d}.nc",
                    np.ones((3, 2)) * month,
                )

    monkeypatch.setattr("climind.fetchers.fetcher_cma_api._load_cma_client", lambda: RetryClient)
    monkeypatch.setattr(
        "climind.fetchers.fetcher_cma_api.resolve_cma_credentials",
        lambda: {"CMA_USER_ID": "test-user"},
    )
    monkeypatch.setattr("climind.fetchers.fetcher_cma_api.time.sleep", lambda seconds: None)

    result = ensure_cma_source_files(tmp_path, filename, strict=True, start_year=1850, end_year=1850)

    assert calls["count"] == 2
    assert result["status"] == "source_downloaded"
    assert (tmp_path / filename).exists()
