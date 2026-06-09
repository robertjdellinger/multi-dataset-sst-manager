import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sea_surface_temperature"
    / "audit_cma_gridded_cache.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("audit_cma_gridded_cache", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def cma_data_root(datadir: Path) -> Path:
    return datadir / "ManagedData" / "SeaSurfaceTemperature" / "Data" / "CMA-SST"


def write_cma_grid(path: Path, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        {"anomaly": (("lat", "lon"), np.full((3, 2), value, dtype=float))},
        coords={"lat": np.array([-60.0, 0.0, 60.0]), "lon": np.array([-150.0, 12.0])},
    )
    dataset.to_netcdf(path)


def write_cma_monthly_source_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"year": 1850}
    row.update(
        {
            month: index
            for index, month in enumerate(
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                start=1,
            )
        }
    )
    pd.DataFrame([row]).to_csv(path, index=False)


def write_cma_badc_annual(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        ),
        encoding="utf-8",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cma_cache_audit_reports_valid_gmst_sensitivity_cache(tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"
    raw_dir = cma_data_root(datadir) / "cma_api_raw"
    for month in range(1, 13):
        write_cma_grid(raw_dir / f"SURF_CLI_GLB_MST_MON_GRID_2DEG_1850{month:02d}.nc", value=float(month))
    write_cma_monthly_source_csv(cma_data_root(datadir) / "CMA-SST_Global_Month_Temp_1981_2010.csv")

    result = module.run_audit(datadir=datadir, project_root=tmp_path, strict=True, start_year=1850, end_year=1850)

    assert result["status"] == "valid_cma_gmst_cache"
    assert result["netcdf_file_count"] == 12
    assert result["first_month"] == "185001"
    assert result["last_month"] == "185012"
    assert result["primary_gridded_sst_eligible"] is False
    assert result["sensitivity_gridded_eligible"] is True
    assert result["requires_ocean_mask"] is True
    assert result["regional_meow_ppow_eligible"] is False
    assert result["time_coordinate_source"] == "filename"
    assert result["managed_monthly_csv_gridded_eligible"] is False

    inventory_path = module.inventory_path(datadir)
    summary_path = module.summary_path(datadir)
    assert inventory_path.exists()
    assert summary_path.exists()
    inventory_rows = read_csv_rows(inventory_path)
    summary_rows = read_csv_rows(summary_path)
    assert len(inventory_rows) == 12
    assert len(summary_rows) == 1
    assert inventory_rows[0]["variable_name"] == "anomaly"
    assert inventory_rows[0]["lat_name"] == "lat"
    assert inventory_rows[0]["lon_name"] == "lon"
    assert inventory_rows[0]["time_coordinate_source"] == "filename"
    assert summary_rows[0]["primary_gridded_sst_eligible"] == "False"


def test_cma_cache_audit_reports_missing_cache_without_creating_inventory(tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"

    result = module.run_audit(datadir=datadir, project_root=tmp_path, strict=False)

    assert result["status"] == "missing_cma_gridded_cache"
    assert result["netcdf_file_count"] == 0
    assert result["primary_gridded_sst_eligible"] is False
    assert result["sensitivity_gridded_eligible"] is False
    assert module.summary_path(datadir).exists()
    assert module.inventory_path(datadir).exists()
    assert read_csv_rows(module.inventory_path(datadir)) == []


def test_cma_cache_audit_keeps_annual_global_csv_gridded_ineligible(tmp_path):
    module = load_script()
    datadir = tmp_path / "datadir"
    project_root = tmp_path / "repo"
    write_cma_badc_annual(project_root / "outputs" / "tables" / "sst_CMA_SST.csv")

    result = module.run_audit(datadir=datadir, project_root=project_root, strict=False)

    assert result["annual_csv_count"] == 1
    assert result["annual_csv_gridded_eligible"] is False
    assert result["primary_gridded_sst_eligible"] is False
    assert result["sensitivity_gridded_eligible"] is False
