from pathlib import Path
import importlib.util
import uuid

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sea_surface_temperature"
    / "plot_global_sst_reference_figure.py"
)


def _load_plot_module(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    module_name = f"plot_global_sst_reference_figure_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wide_merged_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "year": [1850, 1851],
            "CMA_SST": [-0.42, -0.40],
            "CMEMS_SST": [0.11, 0.12],
            "DCENT_SST_I": [-0.30, -0.29],
            "ERSST_v6": [-0.35, -0.34],
            "HadSST4": [-0.38, -0.37],
        }
    ).to_csv(path, index=False)


def test_prepare_sst_plot_data_reads_verified_wide_merged_output(monkeypatch, tmp_path):
    module = _load_plot_module(monkeypatch, tmp_path)
    _write_wide_merged_csv(module.INPUT_CSV)

    plot_data = module.prepare_sst_plot_data(module.INPUT_CSV)

    assert list(plot_data.columns) == ["dataset", "year", "sst_anomaly"]
    assert len(plot_data) == 10
    assert plot_data["dataset"].astype(str).unique().tolist() == module.DATASET_LEVELS
    assert plot_data.loc[
        (plot_data["dataset"].astype(str) == "ERSSTv6") & (plot_data["year"] == 1851),
        "sst_anomaly",
    ].item() == -0.34


def test_prepare_sst_plot_data_reads_long_anomaly_alias(monkeypatch, tmp_path):
    module = _load_plot_module(monkeypatch, tmp_path)
    module.INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": [
                "CMA-SST",
                "CMEMS",
                "DCENT-I",
                "ERSSTv6",
                "HadSST4",
            ],
            "year": [1850, 1850, 1850, 1850, 1850],
            "anomaly_sst_C": [-0.42, 0.11, -0.30, -0.35, -0.38],
        }
    ).to_csv(module.INPUT_CSV, index=False)

    plot_data = module.prepare_sst_plot_data(module.INPUT_CSV)

    assert list(plot_data.columns) == ["dataset", "year", "sst_anomaly"]
    assert plot_data["sst_anomaly"].tolist() == [-0.42, 0.11, -0.30, -0.35, -0.38]


def test_main_writes_reference_style_png_to_managed_sst_figures(monkeypatch, tmp_path):
    module = _load_plot_module(monkeypatch, tmp_path)
    _write_wide_merged_csv(module.INPUT_CSV)

    module.main()

    assert module.OUTPUT_PNG.exists()
    assert module.OUTPUT_PNG.parent == (
        tmp_path / "ManagedData" / "SeaSurfaceTemperature" / "Figures"
    )
