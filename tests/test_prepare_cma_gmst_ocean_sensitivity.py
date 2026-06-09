import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sea_surface_temperature"
    / "prepare_cma_gmst_ocean_sensitivity.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_cma_gmst_ocean_sensitivity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def cma_raw_dir(datadir: Path) -> Path:
    return datadir / "ManagedData" / "SeaSurfaceTemperature" / "Data" / "CMA-SST" / "cma_api_raw"


def write_cma_grid(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        {"anomaly": (("lat", "lon"), values.astype(float))},
        coords={"lat": np.array([-1.0, 1.0]), "lon": np.array([0.0, 10.0])},
    )
    dataset.to_netcdf(path)


def write_cma_year(datadir: Path, year: int, base_value: float) -> None:
    for month in range(1, 13):
        values = np.full((2, 2), base_value + month / 100.0, dtype=float)
        values[1, 1] = 999.0
        write_cma_grid(
            cma_raw_dir(datadir) / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_{year}{month:02d}.nc",
            values,
        )


def test_prepare_cma_gmst_sensitivity_writes_ocean_masked_rebaselined_netcdf(monkeypatch, tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"
    write_cma_year(datadir, 1991, 0.0)
    write_cma_year(datadir, 1992, 1.0)
    monkeypatch.setattr(
        module,
        "_ocean_cell_mask",
        lambda latitudes, longitudes: np.array([[True, True], [True, False]]),
    )

    result = module.run_preparation(
        datadir=datadir,
        start_year=1991,
        end_year=1992,
        baseline_start=1991,
        baseline_end=1992,
        min_baseline_years=2,
        strict=True,
    )

    output_path = Path(result["output_path"])
    inventory_path = module.inventory_path(datadir)
    assert output_path.exists()
    assert inventory_path.exists()

    with xr.open_dataset(output_path, decode_times=True) as dataset:
        assert dataset.attrs["dataset"] == "CMA-GMST-ocean-sensitivity"
        assert dataset.attrs["product_role"] == "cma_gmst_product_16_ocean_only_sensitivity"
        assert dataset.attrs["primary_gridded_sst_eligible"] == "False"
        assert dataset.attrs["sensitivity_gridded_eligible"] == "True"
        assert dataset.attrs["ocean_mask_applied"] == "True"
        anomaly = dataset["sst_anomaly_C"]
        assert anomaly.sizes == {"year": 2, "lat": 2, "lon": 2}
        assert anomaly.sel(year=1991, lat=-1.0, lon=0.0).item() == pytest.approx(-0.5)
        assert anomaly.sel(year=1992, lat=-1.0, lon=0.0).item() == pytest.approx(0.5)
        assert np.isnan(anomaly.sel(year=1991, lat=1.0, lon=10.0).item())

    inventory = pd.read_csv(inventory_path)
    assert len(inventory) == 1
    row = inventory.iloc[0].to_dict()
    assert row["dataset"] == "CMA-GMST-ocean-sensitivity"
    assert row["source_file_count"] == 24
    assert row["year_start"] == 1991
    assert row["year_end"] == 1992
    assert row["status"] == "processed"
    assert row["primary_gridded_sst_eligible"] is False or row["primary_gridded_sst_eligible"] == "False"
    assert row["sensitivity_gridded_eligible"] is True or row["sensitivity_gridded_eligible"] == "True"


def test_prepare_cma_gmst_sensitivity_rejects_missing_month(tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"
    write_cma_year(datadir, 1991, 0.0)
    (cma_raw_dir(datadir) / "SURF_CLI_GLB_MST_MON_GRID_2DEG_199112.nc").unlink()

    with pytest.raises(RuntimeError, match="missing monthly CMA grids"):
        module.run_preparation(
            datadir=datadir,
            start_year=1991,
            end_year=1991,
            baseline_start=1991,
            baseline_end=1991,
            min_baseline_years=1,
            strict=True,
        )


def test_prepare_cma_gmst_sensitivity_missing_cache_is_nonblocking_without_strict(tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"

    result = module.run_preparation(datadir=datadir, strict=False)

    assert result["status"] == "missing_cma_gmst_cache"
    assert result["primary_gridded_sst_eligible"] is False
    assert result["sensitivity_gridded_eligible"] is False
