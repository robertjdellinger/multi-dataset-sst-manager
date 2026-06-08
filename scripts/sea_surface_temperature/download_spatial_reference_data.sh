#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATADIR:-}" ]]; then
  echo "DATADIR is not set. Set it before running this helper." >&2
  echo 'Example: export DATADIR="$HOME/data/multi-dataset-sst-manager"' >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download Natural Earth Admin 0 countries." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to expand Natural Earth Admin 0 countries." >&2
  exit 1
fi

natural_earth_dir="${DATADIR}/Natural_Earth"
meow_ppow_dir="${DATADIR}/Shape_Files/UNEP_WCMC_MEOW_PPOW"
longhurst_dir="${DATADIR}/Shape_Files/Longhurst_Provinces"
ornl_dir="${DATADIR}/ManagedData/SeaSurfaceTemperature/SpatialReference/ORNL_DAAC_ISLSCP_II_Land_Water_Masks"
natural_earth_zip="${natural_earth_dir}/ne_10m_admin_0_countries.zip"
natural_earth_url="https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
ornl_catalog_url="https://doi.org/10.3334/ORNLDAAC/1200"
ornl_earthdata_search_url="https://search.earthdata.nasa.gov/search/granules?p=C2785331161-ORNL_CLOUD"
ornl_land_ocean_zip="${ornl_dir}/land_ocean_masks_xdeg.zip"
ornl_land_ocean_readme="${ornl_dir}/0_land_ocean_masks_xdeg_readme.txt"
ornl_doc_pdf_primary="${ornl_dir}/combined_ancillary_xdeg.pdf"
ornl_doc_pdf_alternate="${ornl_dir}/1_land_water_masks_doc.pdf"

mkdir -p "${natural_earth_dir}" "${meow_ppow_dir}" "${longhurst_dir}" "${ornl_dir}"

if [[ ! -f "${natural_earth_zip}" ]]; then
  curl --fail --location --show-error --output "${natural_earth_zip}" "${natural_earth_url}"
else
  echo "Natural Earth zip already exists: ${natural_earth_zip}"
fi

unzip -o "${natural_earth_zip}" -d "${natural_earth_dir}"

file_status() {
  if [[ -f "$1" ]]; then
    echo "present"
  else
    echo "missing"
  fi
}

ornl_doc_status="missing"
if [[ -f "${ornl_doc_pdf_primary}" || -f "${ornl_doc_pdf_alternate}" ]]; then
  ornl_doc_status="present"
fi

ornl_mask_status="$(file_status "${ornl_land_ocean_zip}")"
ornl_readme_status="$(file_status "${ornl_land_ocean_readme}")"

cat <<EOF

Natural Earth Admin 0 countries are available under:
  ${natural_earth_dir}

Manual MEOW/PPOW step:
  1. Open https://wcmc.io/WCMC_036
  2. Review the UNEP-WCMC access and citation terms.
  3. Download the MEOW/PPOW spatial files.
  4. Place the extracted files under:
     ${meow_ppow_dir}

MEOW/PPOW are not downloaded by this script because the WCMC source requires
manual review of access terms and source package contents.

Optional Longhurst province step:
  Longhurst provinces can be stored under:
     ${longhurst_dir}
  Treat Longhurst as a separate optional biogeochemical layer, not as the main
  MEOW/PPOW marine regionalization framework.

Optional ORNL DAAC / ISLSCP II ancillary mask step:
  Dataset:
     ISLSCP II Land and Water Masks with Ancillary Data, Version 1
  DOI/catalog:
     ${ornl_catalog_url}
  Earthdata collection concept ID:
     C2785331161-ORNL_CLOUD
  Earthdata Search:
     ${ornl_earthdata_search_url}
  Target directory:
     ${ornl_dir}

  Use the Earthdata "Data Access" or ORNL "User Guide" links from the catalog.
  Do not hard-code direct granule URLs in this repository; Earthdata access may
  require authentication and direct file locations may change.

  For a deliberate CMA-GMST ocean-only fallback or sensitivity pass, manually
  download these resources into the target directory:
     land_ocean_masks_xdeg.zip
     0_land_ocean_masks_xdeg_readme.txt
     combined_ancillary_xdeg.pdf or 1_land_water_masks_doc.pdf

  Use the 0.25-degree land-ocean percentage fields in land_ocean_masks_xdeg.zip
  to derive fractional ocean weights on the 2-degree CMA-GMST grid. Do not use
  inland_water_masks_xdeg.zip as the primary land-ocean separation mask. Do not
  use land_water_masks_xdeg.zip or land_water_masks-99_xdeg.zip as the primary
  mask unless the land-ocean file is unavailable. Do not use outline files for
  weighting. Download lat_lon_grid_coords_xdeg.zip only if a parser cannot
  reconstruct grid-cell centers from the ESRI ASCII grid headers.

  Current ORNL file-presence status:
     land_ocean_masks_xdeg.zip: ${ornl_mask_status}
     0_land_ocean_masks_xdeg_readme.txt: ${ornl_readme_status}
     combined_ancillary_xdeg.pdf or 1_land_water_masks_doc.pdf: ${ornl_doc_status}

No ORNL DAAC files were downloaded by this script. If all required ORNL files
show as present above, this helper has only verified their local presence.
EOF
