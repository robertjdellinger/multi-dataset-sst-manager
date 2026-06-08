from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = PROJECT_ROOT / "scripts" / "sea_surface_temperature" / "download_spatial_reference_data.sh"
SST_METADATA_DIR = PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "build_pipeline"


def test_sst_spatial_reference_helper_documents_ornl_manual_acquisition():
    text = HELPER.read_text()

    required_snippets = [
        "ORNL_DAAC_ISLSCP_II_Land_Water_Masks",
        "ISLSCP II Land and Water Masks with Ancillary Data",
        "https://doi.org/10.3334/ORNLDAAC/1200",
        "C2785331161-ORNL_CLOUD",
        "land_ocean_masks_xdeg.zip",
        "0_land_ocean_masks_xdeg_readme.txt",
        "combined_ancillary_xdeg.pdf",
        "1_land_water_masks_doc.pdf",
        "No ORNL DAAC files were downloaded by this script",
    ]
    for snippet in required_snippets:
        assert snippet in text

    for line in text.splitlines():
        command = line.strip()
        if command.startswith(("curl ", "wget ")):
            assert "ORNL" not in command
            assert "ornl" not in command
            assert "land_ocean_masks" not in command


def test_islscp_is_not_active_sst_build_pipeline_metadata():
    metadata_text = "\n".join(path.read_text() for path in sorted(SST_METADATA_DIR.glob("*.json")))

    assert "ISLSCP" not in metadata_text
    assert "ORNLDAAC/1200" not in metadata_text
