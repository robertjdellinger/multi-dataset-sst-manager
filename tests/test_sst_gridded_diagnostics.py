from pathlib import Path
import importlib.util
import uuid

import numpy as np
import pytest
import xarray as xr


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sea_surface_temperature"
    / "plot_sst_gridded_diagnostics.py"
)


def _load_module(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    module_name = f"plot_sst_gridded_diagnostics_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _annual_grid(years=(1991, 1992, 1993), *, missing=False):
    year_values = np.asarray(years, dtype=int)
    lat = np.array([-1.0, 1.0])
    lon = np.array([-180.0, 0.0, 120.0])
    values = np.empty((len(year_values), len(lat), len(lon)), dtype=float)
    for index, year in enumerate(year_values):
        values[index, :, :] = (year - year_values[0]) + np.arange(len(lat))[:, None] * 0.1
    if missing:
        values[1, 0, 1] = np.nan
        values[:, 1, 2] = np.nan
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={"year": year_values, "lat": lat, "lon": lon},
        name="sst_anomaly_C",
        attrs={"units": "degC", "baseline": "1991-2020"},
    )


def test_discover_processed_gridded_files_finds_only_expected_datasets(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    gridded_dir = (
        tmp_path
        / "ManagedData"
        / "SeaSurfaceTemperature"
        / "processed"
        / "gridded"
    )
    gridded_dir.mkdir(parents=True)
    ersst = gridded_dir / "sst_ERSST-v6-gridded_annual_gridded_1850_2025_baseline_1991_2020.nc"
    hadsst = gridded_dir / "sst_HadSST4-gridded_annual_gridded_1850_2025_baseline_1991_2020.nc"
    ignored = gridded_dir / "sst_CMA-SST-gridded_annual_gridded_1850_2025_baseline_1991_2020.nc"
    ersst.touch()
    hadsst.touch()
    ignored.touch()

    discovered = module.discover_processed_gridded_files()

    assert discovered == {
        "ERSST-v6-gridded": ersst,
        "HadSST4-gridded": hadsst,
    }


def test_period_selection_rejects_missing_period(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    annual = _annual_grid(years=(1991, 1992, 1993))

    selected = module.select_period(annual, 1992, 1993)

    assert selected.sizes["year"] == 2
    assert selected["year"].values.tolist() == [1992, 1993]
    with pytest.raises(RuntimeError, match="does not overlap"):
        module.select_period(annual, 1980, 1981)


def test_period_mean_trend_and_valid_fraction_preserve_missingness(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    annual = _annual_grid(years=(1991, 1992, 1993), missing=True)

    period_mean = module.calculate_period_mean(annual, 1991, 1993)
    trend = module.calculate_linear_trend(annual, 1991, 1993, min_years=2)
    valid_fraction = module.calculate_valid_year_fraction(annual, 1991, 1993)

    assert period_mean.dims == ("lat", "lon")
    assert np.isnan(period_mean.sel(lat=1.0, lon=120.0))
    assert trend.attrs["units"] == "degC per decade"
    assert trend.sel(lat=-1.0, lon=-180.0).item() == pytest.approx(10.0)
    assert np.isnan(trend.sel(lat=1.0, lon=120.0))
    assert valid_fraction.sel(lat=-1.0, lon=0.0).item() == pytest.approx(2 / 3)
    assert valid_fraction.sel(lat=1.0, lon=120.0).item() == 0.0


def test_difference_field_interpolates_comparison_grid_to_reference(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    left = _annual_grid(years=(1991, 1992, 1993))
    right = _annual_grid(years=(1991, 1992, 1993)).isel(lat=[0], lon=[0, 1])
    right = right.assign_coords(lat=[-1.0], lon=[-180.0, 0.0]) - 0.25

    difference = module.calculate_difference_field(left, right, 1991, 1993)

    assert difference.dims == ("lat", "lon")
    assert difference.sel(lat=-1.0, lon=-180.0).item() == pytest.approx(0.25)
    assert np.isnan(difference.sel(lat=1.0, lon=120.0))


def test_run_diagnostics_reports_missing_inputs_without_outputs(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)

    result = module.run_diagnostics(strict=False)

    assert result["status"] == "missing_inputs"
    assert result["figures"] == []
    assert result["tables"] == []
    assert not (
        tmp_path
        / "ManagedData"
        / "SeaSurfaceTemperature"
        / "Figures"
        / "gridded_diagnostics"
    ).exists()
