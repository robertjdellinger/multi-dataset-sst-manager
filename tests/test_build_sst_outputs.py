from pathlib import Path
import importlib.util
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
    existing_summary = output_dir / "sst_summary.csv"
    existing_summary.write_text("previous validated summary\n")

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
