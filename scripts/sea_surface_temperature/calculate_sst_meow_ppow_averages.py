#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/sea_surface_temperature/calculate_sst_meow_ppow_averages.py
#
#  OPTIONAL: compute MEOW/PPOW (and Global) regional SST averages from TRUE
#  latitude-longitude gridded SST fields. Mirrors the upstream staged design
#  (calculate_wmo_ra_averages.py) but uses marine biogeographic regions instead
#  of WMO regions and never edits the upstream WMO scripts.
#
#  Running this script always writes the eligibility report; it only writes
#  regional CSVs when an eligible gridded SST source and its region masks exist.
#  The five required SST datasets are time series (space_resolution=999) and are
#  therefore reported as ineligible, by design. CMEMS (area-averaged NetCDF) is
#  also ineligible. Outputs are written under DATADIR and are not committed.

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sst_regional_core as core  # noqa: E402

from climind.data_manager.processing import DataCollection  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = (
    PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "build_pipeline"
)
GRIDDED_PIPELINE_DIR = (
    PROJECT_ROOT / "climind" / "metadata_files" / "temperature" / "sst" / "gridded_pipeline"
)
QA_DIR = PROJECT_ROOT / "outputs" / "logs" / "qa"
ELIGIBILITY_CSV = QA_DIR / "sst_gridded_regional_eligibility.csv"


def candidate_metadata_files() -> list[Path]:
    return sorted(PIPELINE_DIR.glob("*.json")) + sorted(GRIDDED_PIPELINE_DIR.glob("*.json"))


def managed_gridded_inventory_path() -> Path:
    data_dir = os.environ.get("DATADIR")
    if not data_dir:
        return Path("")
    return (
        Path(data_dir)
        / "ManagedData"
        / "SeaSurfaceTemperature"
        / "logs"
        / "qa"
        / "sst_gridded_source_inventory.csv"
    )


def load_gridded_inventory() -> dict[str, dict]:
    inventory = managed_gridded_inventory_path()
    if not inventory.exists():
        return {}
    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["dataset"]: row for row in rows}


def _csv_bool(value) -> bool:
    return str(value).lower() == "true"


def _meta_get(meta, key, default=None):
    """Read a field from a CombinedMetadata (subscript-only) or a dict."""
    try:
        return meta[key]
    except (KeyError, TypeError):
        return default


def build_eligibility_rows() -> list[dict]:
    rows: list[dict] = []
    inventory = load_gridded_inventory()
    for metadata_file in candidate_metadata_files():
        collection = DataCollection.from_file(metadata_file)
        for dataset in collection.datasets:
            meta = dataset.metadata
            meta_dict = {
                "name": _meta_get(meta, "name"),
                "type": _meta_get(meta, "type"),
                "time_resolution": _meta_get(meta, "time_resolution"),
                "space_resolution": _meta_get(meta, "space_resolution"),
            }
            verdict = core.classify_regional_eligibility(meta_dict)
            filenames = _meta_get(meta, "filename") or [""]
            source_filename = filenames[0] if filenames else ""

            has_lat = has_lon = has_time = False
            if verdict["eligible"]:
                inventory_row = inventory.get(meta_dict["name"])
                if inventory_row:
                    has_lat = bool(inventory_row.get("lat_name"))
                    has_lon = bool(inventory_row.get("lon_name"))
                    has_time = bool(inventory_row.get("time_name"))
                    verdict["eligible"] = _csv_bool(
                        inventory_row.get("eligible_for_regional_processing")
                    )
                    verdict["exclusion_reason"] = "" if verdict["eligible"] else inventory_row.get("reason", "")
                    source_filename = inventory_row.get("source_file", source_filename)
                else:
                    verdict["eligible"] = False
                    verdict["exclusion_reason"] = "gridded source has not been downloaded and inspected"

            rows.append(
                {
                    "dataset": meta_dict["name"],
                    "source_filename": source_filename,
                    "type": verdict["type"],
                    "time_resolution": verdict["time_resolution"],
                    "space_resolution": verdict["space_resolution"],
                    "has_lat": has_lat,
                    "has_lon": has_lon,
                    "has_time": has_time,
                    "eligible": verdict["eligible"],
                    "exclusion_reason": verdict["exclusion_reason"],
                }
            )
    return rows


def write_eligibility_report() -> Path:
    rows = build_eligibility_rows()
    QA_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "source_filename", "type", "time_resolution", "space_resolution",
        "has_lat", "has_lon", "has_time", "eligible", "exclusion_reason",
    ]
    with open(ELIGIBILITY_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return ELIGIBILITY_CSV


def main() -> int:
    out = write_eligibility_report()
    rows = build_eligibility_rows()
    eligible = [r for r in rows if r["eligible"]]
    print(f"Wrote regional eligibility report ({len(rows)} datasets) to {out}")
    print(f"  eligible gridded SST datasets: {len(eligible)}")
    for r in rows:
        flag = "ELIGIBLE" if r["eligible"] else f"excluded ({r['exclusion_reason']})"
        print(f"  {r['dataset']}: {flag}")
    if not eligible:
        print(
            "No regional outputs written: no true latitude-longitude gridded SST "
            "dataset is currently wired into the pipeline. Add a gridded SST "
            "collection (type=gridded, real space_resolution) and provide "
            "MEOW/PPOW masks to produce regional summaries."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
