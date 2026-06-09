import json
import io
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SST_METADATA_DIR = REPO_ROOT / "climind" / "metadata_files" / "temperature" / "sst"
STRICT_METADATA_DIR = SST_METADATA_DIR / "build_pipeline"
BHM_SST_METADATA = SST_METADATA_DIR / "bhm.json"
GRIDDED_METADATA_DIR = SST_METADATA_DIR / "gridded_pipeline"
DCENT_BUILD_METADATA = STRICT_METADATA_DIR / "dcent_sst_i.json"

REQUIRED_STRICT_METADATA = {
    "cma_sst.json",
    "cmems_sst.json",
    "dcent_sst_i.json",
    "ersst_v6.json",
    "hadsst4.json",
}

REQUIRED_OUTPUT_ROWS = {
    "sst_CMA_SST.csv": (176, 1850, 2025),
    "sst_CMEMS_SST.csv": (43, 1982, 2024),
    "sst_DCENT_SST_I.csv": (176, 1850, 2025),
    "sst_ERSST_v6.csv": (176, 1850, 2025),
    "sst_HadSST4.csv": (176, 1850, 2025),
    "sst_summary.csv": (176, 1850, 2025),
}

REQUIRED_FIGURES = {
    "outputs/figures/global_sea_surface_temperature_1850_2025_reference_style.png",
    "outputs/figures/annual_diagnostics/sst_annual_baseline_sensitivity.png",
    "outputs/figures/annual_diagnostics/sst_annual_dataset_availability_and_validation_summary.png",
    "outputs/figures/annual_diagnostics/sst_annual_modern_overlap_1982_present.png",
    "outputs/figures/annual_diagnostics/sst_annual_pairwise_residual_heatmap.png",
    "outputs/figures/annual_diagnostics/sst_annual_period_trend_comparison.png",
    "outputs/figures/annual_diagnostics/sst_annual_preindustrial_sensitivity_1850_1900.png",
    "outputs/figures/annual_diagnostics/sst_annual_rolling_differences.png",
    "outputs/figures/annual_diagnostics/sst_annual_source_coverage.png",
    "outputs/figures/annual_diagnostics/sst_annual_source_spread.png",
}

REMOVED_SST_OPTIONAL_PATHS = [
    GRIDDED_METADATA_DIR,
    REPO_ROOT / "scripts" / "data_management" / "download_spatial_reference_data.sh",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "audit_cma_gridded_cache.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "calculate_sst_meow_ppow_averages.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "download_spatial_reference_data.sh",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "download_sst_gridded_inputs.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "make_meow_ppow_region_masks.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "plot_sst_gridded_diagnostics.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "prepare_cma_gmst_ocean_sensitivity.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "prepare_sst_gridded_inputs.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "sst_regional_core.py",
    REPO_ROOT / "scripts" / "sea_surface_temperature" / "write_observational_sst_final_report.py",
    REPO_ROOT / "climind" / "fetchers" / "fetcher_ersst_v6_gridded.py",
    REPO_ROOT / "climind" / "fetchers" / "fetcher_hadsst4_gridded.py",
    REPO_ROOT / "climind" / "readers" / "reader_ersst_v6_gridded.py",
    REPO_ROOT / "climind" / "readers" / "reader_hadsst4_gridded.py",
    REPO_ROOT / "outputs" / "figures" / "gridded_diagnostics",
    REPO_ROOT / "outputs" / "logs" / "qa" / "sst_gridded_regional_eligibility.csv",
]

FORBIDDEN_README_TERMS = [
    "bhm",
    "coralhydro2k",
    "paleoda",
    "paleo",
    "proxy",
    "isotope",
    "optional gridded",
    "gridded_pipeline",
    "meow",
    "ppow",
    "regional summaries",
    "ocean-only sensitivity",
]


def _read_badc_csv_data(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    marker = lines.index("data")
    end = lines.index("end data")
    data_lines = lines[marker + 1 : end]
    return pd.read_csv(io.StringIO("\n".join(data_lines)))


def test_readme_is_strict_modern_observational_sst_only():
    text = README.read_text().lower()
    assert "modern observational sea-surface-temperature" in text
    assert "build_pipeline/" in text
    assert "scripts/data_management/build_sst_outputs.py --strict" in text
    assert "sst_cma_sst.csv" in text
    assert "sst_summary.csv" in text

    found = [term for term in FORBIDDEN_README_TERMS if term in text]
    assert found == []


def test_only_required_strict_sst_metadata_family_remains():
    assert STRICT_METADATA_DIR.exists()
    assert {path.name for path in STRICT_METADATA_DIR.glob("*.json")} == REQUIRED_STRICT_METADATA
    assert not GRIDDED_METADATA_DIR.exists()
    assert not BHM_SST_METADATA.exists()


def test_removed_optional_sst_workflow_paths_do_not_exist():
    remaining = [path for path in REMOVED_SST_OPTIONAL_PATHS if path.exists()]
    assert remaining == []


def test_dcent_build_pipeline_metadata_uses_monthly_ocean_statistics_source():
    metadata = json.loads(DCENT_BUILD_METADATA.read_text())
    serialized = json.dumps(metadata)
    active_dataset = metadata["datasets"][0]

    assert "https://doi.org/10.7910/DVN/ROG38Q" in serialized
    assert "10.7910/DVN/ZY0WM8" not in serialized
    forbidden_private_terms = ["dropbox" + ".com", "rl" + "key=", "s" + "t="]
    assert [term for term in forbidden_private_terms if term in serialized] == []
    assert "DCENT_DCENT_I_OST_monthly_statistics.txt" in serialized
    assert "DCENT_DCENT_I_OST_annual_statistics_embargo.txt" in serialized
    assert "dataverse.harvard.edu/file.xhtml?fileId=13202723" not in serialized

    assert active_dataset["filename"] == ["DCENT_DCENT_I_OST_monthly_statistics.txt"]
    assert active_dataset["time_resolution"] == "monthly"
    assert active_dataset["space_resolution"] == 999
    assert active_dataset["reader"] == "reader_dcenti"


def test_required_output_row_counts_and_year_ranges():
    for filename, (expected_rows, expected_start, expected_end) in REQUIRED_OUTPUT_ROWS.items():
        data = _read_badc_csv_data(REPO_ROOT / "outputs" / "tables" / filename)
        years = pd.to_numeric(data["year"], errors="coerce")

        assert len(data) == expected_rows
        assert int(years.min()) == expected_start
        assert int(years.max()) == expected_end


def test_required_annual_figures_exist_and_no_gridded_figures_remain():
    for relative_path in REQUIRED_FIGURES:
        path = REPO_ROOT / relative_path
        assert path.exists()
        assert path.stat().st_size > 0

    figure_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "outputs" / "figures").rglob("*.png")
    }
    assert figure_paths == REQUIRED_FIGURES


def test_no_ds_store_files_remain_in_outputs():
    assert list((REPO_ROOT / "outputs").rglob(".DS_Store")) == []
