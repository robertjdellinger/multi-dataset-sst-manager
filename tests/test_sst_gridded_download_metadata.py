from pathlib import Path
import importlib.util
import json
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRIDDED_METADATA_DIR = (
    PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "gridded_pipeline"
)
STRICT_BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "data_management" / "build_sst_outputs.py"
GRIDDED_DOWNLOAD_SCRIPT = (
    PROJECT_ROOT / "scripts" / "sea_surface_temperature" / "download_sst_gridded_inputs.py"
)
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "sea_surface_temperature" / "prepare_sst_gridded_inputs.py"
REGIONAL_SCRIPT = (
    PROJECT_ROOT / "scripts" / "sea_surface_temperature" / "calculate_sst_meow_ppow_averages.py"
)


def _load_module(path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gridded_metadata_family_is_separate_from_strict_global_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    build = _load_module(STRICT_BUILD_SCRIPT, "build_sst_outputs")

    strict_names = [path.name for path in build.SST_METADATA_FILES]
    assert strict_names == [
        "cma_sst.json",
        "cmems_sst.json",
        "dcent_sst_i.json",
        "ersst_v6.json",
        "hadsst4.json",
    ]
    assert all("gridded_pipeline" not in str(path) for path in build.SST_METADATA_FILES)

    gridded_names = sorted(path.name for path in GRIDDED_METADATA_DIR.glob("*.json"))
    assert gridded_names == ["ersst_v6_gridded.json", "hadsst4_gridded.json"]


def test_gridded_metadata_contracts_do_not_reuse_global_collection_names():
    global_names = set()
    build_dir = PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "build_pipeline"
    for path in build_dir.glob("*.json"):
        global_names.add(json.loads(path.read_text())["name"])

    expected = {
        "ersst_v6_gridded.json": ("ERSST-v6-gridded", 2, "reader_ersst_v6_gridded", "fetcher_ersst_v6_gridded"),
        "hadsst4_gridded.json": ("HadSST4-gridded", 5, "reader_hadsst4_gridded", "fetcher_hadsst4_gridded"),
    }
    for filename, (name, resolution, reader, fetcher) in expected.items():
        metadata = json.loads((GRIDDED_METADATA_DIR / filename).read_text())
        dataset = metadata["datasets"][0]
        assert metadata["name"] == name
        assert metadata["name"] not in global_names
        assert dataset["type"] == "gridded"
        assert dataset["time_resolution"] == "monthly"
        assert dataset["space_resolution"] == resolution
        assert dataset["space_resolution"] != 999
        assert dataset["reader"] == reader
        assert dataset["fetcher"] == fetcher


def test_inventory_detects_true_gridded_netcdf(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    module = _load_module(GRIDDED_DOWNLOAD_SCRIPT, "download_sst_gridded_inputs")
    source = tmp_path / "sst.mnmean.nc"
    times = pd.date_range("1991-01-01", periods=3, freq="MS")
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.ones((3, 2, 4)))},
        coords={"time": times, "lat": [-1.0, 1.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )
    ds.to_netcdf(source)

    record = module.inspect_gridded_source(
        source_file=source,
        metadata={"actual": True, "space_resolution": 2, "type": "gridded"},
        dataset_name="ERSST-v6-gridded",
        collection_name="ERSST-v6-gridded",
        metadata_file=GRIDDED_METADATA_DIR / "ersst_v6_gridded.json",
        download_time_utc="2026-06-05T00:00:00Z",
    )

    assert record["variable_name"] == "sst"
    assert record["lat_name"] == "lat"
    assert record["lon_name"] == "lon"
    assert record["time_name"] == "time"
    assert record["n_time"] == 3
    assert record["n_lat"] == 2
    assert record["n_lon"] == 4
    assert record["source_value_type"] == "actual"
    assert record["is_area_averaged"] is False
    assert record["eligible_for_regional_processing"] is True
    assert record["reason"] == ""
    assert len(record["checksum"]) == 64


def test_inventory_rejects_area_averaged_netcdf(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    module = _load_module(GRIDDED_DOWNLOAD_SCRIPT, "download_sst_gridded_inputs")
    source = tmp_path / "area_average.nc"
    times = pd.date_range("1991-01-01", periods=2, freq="MS")
    ds = xr.Dataset({"sst": (("time",), np.ones(2))}, coords={"time": times})
    ds.to_netcdf(source)

    record = module.inspect_gridded_source(
        source_file=source,
        metadata={"actual": False, "space_resolution": 999, "type": "timeseries"},
        dataset_name="CMEMS-area-average",
        collection_name="CMEMS-area-average",
        metadata_file=Path("not-active.json"),
        download_time_utc="2026-06-05T00:00:00Z",
    )

    assert record["is_area_averaged"] is True
    assert record["eligible_for_regional_processing"] is False
    assert "latitude coordinate not found" in record["reason"]
    assert "longitude coordinate not found" in record["reason"]


def test_prepare_eligibility_only_reads_gridded_inventory_without_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    prepare = _load_module(PREPARE_SCRIPT, "prepare_sst_gridded_inputs")
    qa_dir = tmp_path / "ManagedData" / "SeaSurfaceTemperature" / "logs" / "qa"
    qa_dir.mkdir(parents=True)
    inventory = qa_dir / "sst_gridded_source_inventory.csv"
    inventory.write_text(
        "dataset,eligible_for_regional_processing,reason\n"
        "ERSST-v6-gridded,True,\n"
        "HadSST4-gridded,False,missing file\n"
    )

    result = prepare.run_eligibility_only()

    assert result["inventory_path"] == str(inventory)
    assert result["rows"] == 2
    assert result["eligible"] == 1
    assert not (tmp_path / "ManagedData" / "SeaSurfaceTemperature" / "processed" / "gridded").exists()


def test_prepare_writes_annual_gridded_anomaly_netcdf(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    prepare = _load_module(PREPARE_SCRIPT, "prepare_sst_gridded_inputs")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "synthetic_ersst.nc"
    times = pd.date_range("1991-01-01", periods=30 * 12, freq="MS")
    yearly_offsets = np.repeat(np.arange(30, dtype=float), 12)
    values = 10.0 + yearly_offsets[:, None, None] + np.zeros((30 * 12, 2, 3))
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), values, {"units": "degC"})},
        coords={"time": times, "lat": [1.0, -1.0], "lon": [0.0, 180.0, 270.0]},
    )
    ds.to_netcdf(source)

    qa_dir = tmp_path / "ManagedData" / "SeaSurfaceTemperature" / "logs" / "qa"
    qa_dir.mkdir(parents=True)
    inventory = qa_dir / "sst_gridded_source_inventory.csv"
    inventory.write_text(
        "dataset,source_file,variable_name,lat_name,lon_name,time_name,"
        "source_value_type,eligible_for_regional_processing,reason\n"
        f"ERSST-v6-gridded,{source},sst,lat,lon,time,actual,True,\n"
    )

    records = prepare.run_preparation(["ERSST-v6-gridded"])

    assert len(records) == 1
    assert records[0]["status"] == "processed"
    assert records[0]["source_value_type"] == "actual"
    output_path = Path(records[0]["output_path"])
    assert output_path.exists()
    with xr.open_dataset(output_path) as output:
        assert "sst_anomaly_C" in output
        assert output["sst_anomaly_C"].sizes["year"] == 30
        assert output["sst_anomaly_C"].sizes["lat"] == 2
        assert output["sst_anomaly_C"].sizes["lon"] == 3
        assert output["sst_anomaly_C"].attrs["units"] == "degC"
        assert float(output["sst_anomaly_C"].mean("year").mean().values) == pytest.approx(0.0)
        assert list(output["lon"].values) == [-180.0, -90.0, 0.0]
        assert list(output["lat"].values) == [-1.0, 1.0]
    prep_inventory = qa_dir / "sst_gridded_preparation_inventory.csv"
    assert prep_inventory.exists()
    prep = pd.read_csv(prep_inventory)
    assert prep.loc[0, "dataset"] == "ERSST-v6-gridded"
    assert prep.loc[0, "annualization_rule"] == "calendar_year_mean_min_12_months_per_cell"


def test_regional_eligibility_requires_download_inventory_for_gridded_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    regional = _load_module(REGIONAL_SCRIPT, "calculate_sst_meow_ppow_averages")

    rows_without_inventory = regional.build_eligibility_rows()
    by_dataset = {row["dataset"]: row for row in rows_without_inventory}

    assert by_dataset["CMEMS-SST"]["eligible"] is False
    assert by_dataset["CMEMS-SST"]["space_resolution"] == 999
    assert by_dataset["ERSST-v6-gridded"]["eligible"] is False
    assert (
        by_dataset["ERSST-v6-gridded"]["exclusion_reason"]
        == "gridded source has not been downloaded and inspected"
    )

    qa_dir = tmp_path / "ManagedData" / "SeaSurfaceTemperature" / "logs" / "qa"
    qa_dir.mkdir(parents=True)
    inventory = qa_dir / "sst_gridded_source_inventory.csv"
    inventory.write_text(
        "dataset,source_file,lat_name,lon_name,time_name,eligible_for_regional_processing,reason\n"
        "ERSST-v6-gridded,/tmp/sst.mnmean.nc,lat,lon,time,True,\n"
    )

    rows_with_inventory = regional.build_eligibility_rows()
    by_dataset = {row["dataset"]: row for row in rows_with_inventory}

    assert by_dataset["ERSST-v6-gridded"]["eligible"] is True
    assert by_dataset["ERSST-v6-gridded"]["has_lat"] is True
    assert by_dataset["ERSST-v6-gridded"]["has_lon"] is True
    assert by_dataset["ERSST-v6-gridded"]["has_time"] is True
    assert by_dataset["CMEMS-SST"]["eligible"] is False
