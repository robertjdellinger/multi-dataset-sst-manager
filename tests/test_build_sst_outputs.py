from pathlib import Path
import importlib.util
import json
import uuid

import pytest
import pandas as pd

from climind.data_manager.metadata import CollectionMetadata, DatasetMetadata, CombinedMetadata
from climind.data_types.timeseries import TimeSeriesAnnual


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_management"
    / "build_sst_outputs.py"
)
DCENT_METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "climind"
    / "metadata_files"
    / "temperature"
    / "sst"
    / "build_pipeline"
    / "dcent_sst_i.json"
)
DCENT_MONTHLY_URL_PREFIX = "manual:DCENT-I monthly ocean statistics source."
DCENT_MONTHLY_FILENAME = "DCENT_DCENT_I_OST_monthly_statistics.txt"


def _load_build_module(monkeypatch, tmp_path):
    monkeypatch.setenv("DATADIR", str(tmp_path))
    module_name = f"build_sst_outputs_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(*, actual=False, climatology_start=1961, climatology_end=1990):
    collection = CollectionMetadata(
        {
            "name": "Audit-SST",
            "display_name": "Audit SST",
            "version": "test",
            "variable": "sst",
            "units": "degC",
            "citation": [""],
            "citation_url": [""],
            "data_citation": [""],
            "acknowledgement": "",
            "colour": "black",
            "zpos": 1,
        }
    )
    dataset = DatasetMetadata(
        {
            "url": ["https://example.test/audit.csv"],
            "filename": ["audit.csv"],
            "type": "timeseries",
            "long_name": "Audit sea-surface temperature",
            "time_resolution": "annual",
            "space_resolution": 999,
            "climatology_start": climatology_start,
            "climatology_end": climatology_end,
            "actual": actual,
            "derived": False,
            "history": [],
            "reader": "reader_test",
            "fetcher": "fetcher_standard_url",
        }
    )
    return CombinedMetadata(dataset, collection)


def test_strict_dcent_processing_select_uses_monthly_source(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)

    assert module.PROCESSING_SELECT["DCENT-SST-I"] == {"time_resolution": "monthly"}


def test_dcent_build_metadata_uses_monthly_ocean_statistics_as_active_source():
    metadata = json.loads(DCENT_METADATA_PATH.read_text())
    active_dataset = metadata["datasets"][0]
    retained_annual_dataset = metadata["datasets"][1]

    assert active_dataset["url"][0].startswith(DCENT_MONTHLY_URL_PREFIX)
    assert "https://doi.org/10.7910/DVN/ROG38Q" in active_dataset["url"][0]
    assert active_dataset["filename"] == [DCENT_MONTHLY_FILENAME]
    assert active_dataset["type"] == "timeseries"
    assert active_dataset["time_resolution"] == "monthly"
    assert active_dataset["space_resolution"] == 999
    assert active_dataset["climatology_start"] == 1991
    assert active_dataset["climatology_end"] == 2020
    assert active_dataset["actual"] is False
    assert active_dataset["reader"] == "reader_dcenti"
    assert active_dataset["fetcher"] == "fetcher_standard_url_with_rename"

    assert retained_annual_dataset["time_resolution"] == "annual"
    assert retained_annual_dataset["url"][0].startswith(
        "manual:DCENT-I annual ocean statistics cross-check source."
    )
    assert "https://doi.org/10.7910/DVN/ROG38Q" in retained_annual_dataset["url"][0]


def test_dcent_monthly_output_can_attach_retained_annual_uncertainty(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    source_dir = (
        tmp_path
        / "ManagedData"
        / "SeaSurfaceTemperature"
        / "Data"
        / "DCENT-SST-I"
        / "annual_statistics"
    )
    source_dir.mkdir(parents=True)
    annual_source = source_dir / "DCENT_DCENT_I_OST_annual_statistics_embargo.txt"
    annual_source.write_text(
        "\n".join(
            [
                "header",
                "header",
                "header",
                "header",
                "header",
                "header",
                "header",
                "header",
                "1850, -0.86, 0.19, -0.80, 0.20",
                "1851, -0.73, 0.20, -0.68, 0.20",
            ]
        )
        + "\n"
    )
    annual = TimeSeriesAnnual(
        [1850, 1851],
        [-1.0, -0.9],
        metadata=_metadata(actual=False, climatology_start=1991, climatology_end=2020),
    )

    uncertainty_source = module.attach_dcent_annual_uncertainty_from_crosscheck(annual)

    assert uncertainty_source == annual_source
    assert annual.df["uncertainty"].tolist() == pytest.approx([0.19 * 1.96, 0.20 * 1.96])
    assert "monthly-derived data values were not replaced" in annual.metadata["history"][-1]


def test_build_baseline_audit_record_preserves_anomaly_state(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    before = TimeSeriesAnnual(
        [1991, 1992, 1993],
        [1.0, 2.0, 3.0],
        metadata=_metadata(actual=False, climatology_start=1961, climatology_end=1990),
    )
    after = TimeSeriesAnnual(
        [1991, 1992, 1993],
        [-1.0, 0.0, 1.0],
        metadata=_metadata(actual=False, climatology_start=1991, climatology_end=2020),
    )

    record = module.build_baseline_audit_record(
        dataset_name="Audit-SST",
        output_name="sst_Audit.csv",
        source_dataset=before,
        annual_before_rebaseline=before,
        annual_after_processing=after,
        annualization_method="native_annual_values",
    )

    assert record["source_value_type"] == "anomaly"
    assert record["native_climatology_start"] == 1961
    assert record["native_climatology_end"] == 1990
    assert record["target_climatology_start"] == 1991
    assert record["target_climatology_end"] == 2020
    assert record["baseline_adjustment_C"] == pytest.approx(2.0)
    assert record["annualization_method"] == "native_annual_values"
    assert record["coverage_fraction"] == pytest.approx(1.0)
    assert record["status"] == "ok"


def test_build_baseline_audit_record_marks_actual_to_anomaly(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    before = TimeSeriesAnnual(
        [1991, 1992],
        [14.0, 16.0],
        metadata=_metadata(actual=True, climatology_start=1991, climatology_end=2020),
    )
    after = TimeSeriesAnnual(
        [1991, 1992],
        [-1.0, 1.0],
        metadata=_metadata(actual=False, climatology_start=1991, climatology_end=2020),
    )

    record = module.build_baseline_audit_record(
        dataset_name="Audit-SST",
        output_name="sst_Audit.csv",
        source_dataset=before,
        annual_before_rebaseline=before,
        annual_after_processing=after,
        annualization_method="native_annual_values",
    )

    assert record["source_value_type"] == "actual"
    assert record["processing_history_entry"] == "actual_to_anomaly"
    assert record["baseline_adjustment_C"] == pytest.approx(15.0)
    assert record["status"] == "ok"


def test_build_baseline_audit_record_prefers_source_metadata_snapshot(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    mutated_source = TimeSeriesAnnual(
        [1991, 1992],
        [1.0, 2.0],
        metadata=_metadata(actual=False, climatology_start=1991, climatology_end=2020),
    )
    snapshot = {
        "filename": ["native.csv"],
        "url": ["https://example.test/native.csv"],
        "type": "timeseries",
        "time_resolution": "monthly",
        "space_resolution": 999,
        "climatology_start": 1961,
        "climatology_end": 1990,
        "actual": False,
        "reader": "reader_native",
        "fetcher": "fetcher_native",
    }

    record = module.build_baseline_audit_record(
        dataset_name="Audit-SST",
        output_name="sst_Audit.csv",
        source_dataset=mutated_source,
        annual_before_rebaseline=mutated_source,
        annual_after_processing=mutated_source,
        annualization_method="arithmetic_mean_of_monthly_values",
        source_metadata=snapshot,
    )

    assert record["time_resolution"] == "monthly"
    assert record["native_climatology_start"] == 1961
    assert record["native_climatology_end"] == 1990
    assert record["reader"] == "reader_native"
    assert record["fetcher"] == "fetcher_native"


def test_resolve_source_value_type_fails_unknown_state(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="unknown source value type"):
        module.resolve_source_value_type({})


def test_redact_sensitive_text_removes_cma_user_id(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    fake_user_id = "secret-token"
    user_id_key = "user" + "Id"
    monkeypatch.setenv("CMA_USER_ID", fake_user_id)

    redacted = module.redact_sensitive_text(
        f"https://ai.data.cma.cn/aiApi/order/getOrderById?id=1&{user_id_key}={fake_user_id} "
        f"raw {fake_user_id}"
    )

    assert fake_user_id not in redacted
    assert "userId=<CMA_USER_ID>" in redacted


def test_strict_partial_build_preserves_existing_summary(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    output_dir = tmp_path / "outputs"
    qa_dir = tmp_path / "qa"
    output_dir.mkdir()
    qa_dir.mkdir()
    existing_summary = output_dir / "sst_summary.csv"
    existing_summary.write_text("previous validated summary\n")
    existing_baseline_audit = qa_dir / "sst_baseline_audit.csv"
    existing_baseline_audit.write_text("previous baseline audit\n")
    existing_processing_log = qa_dir / "sst_processing_log.csv"
    existing_processing_log.write_text("previous processing log\n")

    module.OUTPUT_DIR = output_dir
    module.QA_DIR = qa_dir
    module.DATASET_ORDER = ["CMA-SST", "CMEMS-SST"]
    module.OUTPUT_NAMES = {
        "CMA-SST": "sst_CMA_SST.csv",
        "CMEMS-SST": "sst_CMEMS_SST.csv",
    }
    module.SUMMARY_COLUMN_NAMES = {
        "CMA-SST": "CMA_SST",
        "CMEMS-SST": "CMEMS_SST",
    }

    monkeypatch.setattr(
        module,
        "acquire_sources",
        lambda: pd.DataFrame(
            [
                {"dataset": "CMA-SST", "status": "source_missing_sdk", "reason": "missing sdk"},
                {"dataset": "CMEMS-SST", "status": "present_local_raw"},
            ]
        ),
    )

    class Archive:
        collections = {"CMA-SST": object(), "CMEMS-SST": object()}

    monkeypatch.setattr(module, "archive_from_sst_metadata", lambda: Archive())
    monkeypatch.setattr(
        module,
        "process_dataset",
        lambda collection, dataset_name, output_name: (
            TimeSeriesAnnual([1991], [0.0], metadata=_metadata(actual=False)),
            {"dataset": dataset_name, "status": "ok"},
        ),
    )
    monkeypatch.setattr(module, "validate_outputs", lambda: pd.DataFrame())

    result = module.build_outputs(allow_partial=False)

    assert result == 1
    assert existing_summary.read_text() == "previous validated summary\n"
    assert existing_baseline_audit.read_text() == "previous baseline audit\n"
    assert existing_processing_log.read_text() == "previous processing log\n"


def test_final_output_replacement_rolls_back_on_mid_copy_failure(monkeypatch, tmp_path):
    module = _load_build_module(monkeypatch, tmp_path)
    temp_output_dir = tmp_path / "temp_outputs"
    final_output_dir = tmp_path / "final_outputs"
    temp_output_dir.mkdir()
    final_output_dir.mkdir()
    required = ["a.csv", "b.csv", "c.csv"]
    for name in required:
        (temp_output_dir / name).write_text(f"new {name}\n")
        (final_output_dir / name).write_text(f"old {name}\n")

    monkeypatch.setattr(module, "_required_output_files", lambda: required)
    real_copy2 = module.shutil.copy2

    def copy2_with_mid_copy_failure(src, dst):
        src = Path(src)
        if src.parent == temp_output_dir and src.name == "b.csv":
            raise OSError("simulated mid-copy failure")
        return real_copy2(src, dst)

    monkeypatch.setattr(module.shutil, "copy2", copy2_with_mid_copy_failure)

    with pytest.raises(OSError, match="simulated mid-copy failure"):
        module._replace_final_outputs_from_temp(temp_output_dir, final_output_dir)

    for name in required:
        assert (final_output_dir / name).read_text() == f"old {name}\n"
