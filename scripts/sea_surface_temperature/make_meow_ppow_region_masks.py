#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/sea_surface_temperature/make_meow_ppow_region_masks.py
#
#  OPTIONAL: validate the UNEP-WCMC MEOW/PPOW marine region vector file and build
#  cell-region masks for a target SST grid. Mirrors the upstream staged design
#  (make_new_regions.py builds region shape files / masks separately) but uses
#  marine biogeographic regions instead of WMO regions, and never edits the
#  upstream WMO scripts.
#
#  Spatial inputs live under DATADIR and are NOT committed:
#      $DATADIR/Shape_Files/UNEP_WCMC_MEOW_PPOW/
#  Masks are written under DATADIR and are NOT committed:
#      $DATADIR/ManagedData/SeaSurfaceTemperature/region_masks/
#  A small spatial-validation audit is written to outputs/logs/qa.

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sst_regional_core as core  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_DIR = PROJECT_ROOT / "outputs" / "logs" / "qa"
SPATIAL_AUDIT_CSV = QA_DIR / "sst_region_mask_audit.csv"

# Common MEOW/PPOW identifier/name fields (UNEP-WCMC schema). Adjust if the
# downloaded vector uses different column names.
DEFAULT_ID_FIELDS = ("ECOREGION", "PROVINCE", "PROV_CODE", "ECO_CODE", "REALM")
DEFAULT_NAME_FIELDS = ("ECOREGION", "PROVINCE", "REALM")


def shape_files_dir() -> Path:
    datadir = os.getenv("DATADIR")
    if not datadir:
        raise RuntimeError('DATADIR is not set. export DATADIR="$HOME/data/multi-dataset-sst-manager"')
    return Path(datadir) / "Shape_Files" / "UNEP_WCMC_MEOW_PPOW"


def region_masks_dir() -> Path:
    datadir = os.getenv("DATADIR")
    if not datadir:
        raise RuntimeError('DATADIR is not set. export DATADIR="$HOME/data/multi-dataset-sst-manager"')
    out = Path(datadir) / "ManagedData" / "SeaSurfaceTemperature" / "region_masks"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def audit_spatial_file(path: Path) -> dict:
    """Validate a MEOW/PPOW vector file and return an audit record."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    id_field = _first_present(gdf.columns, DEFAULT_ID_FIELDS) or DEFAULT_ID_FIELDS[0]
    name_field = _first_present(gdf.columns, DEFAULT_NAME_FIELDS)
    return core.validate_spatial_file(path, id_field, name_field)


def write_audit(records: list[dict]) -> Path:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with open(SPATIAL_AUDIT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return SPATIAL_AUDIT_CSV


def main() -> int:
    directory = shape_files_dir()
    if not directory.exists():
        print(
            f"MEOW/PPOW spatial files not found under {directory}.\n"
            "Download UNEP-WCMC MEOW/PPOW (https://wcmc.io/WCMC_036), extract the "
            "shapefile/GeoPackage there, then re-run. Files are local-only and "
            "must not be committed."
        )
        return 1

    vectors = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in {".shp", ".gpkg", ".geojson"}
    )
    if not vectors:
        print(f"No .shp/.gpkg/.geojson found under {directory}.")
        return 1

    records = [audit_spatial_file(path) for path in vectors]
    out = write_audit(records)
    print(f"Wrote spatial audit for {len(records)} vector file(s) to {out}")
    for record in records:
        status = "OK" if record.get("valid") else f"INVALID ({record.get('problem')})"
        print(f"  {Path(record['source_file']).name}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
