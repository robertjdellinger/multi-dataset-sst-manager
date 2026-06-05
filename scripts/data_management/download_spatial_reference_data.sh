#!/usr/bin/env bash
set -euo pipefail

: "${DATADIR:?Set DATADIR first, for example: export DATADIR=\"$HOME/data/multi-dataset-sst-manager\"}"

mkdir -p "$DATADIR/Natural_Earth"
mkdir -p "$DATADIR/Biogeographic_Provinces"

echo "Downloading Natural Earth Admin 0 Countries..."
curl -L \
  "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip" \
  -o "$DATADIR/Natural_Earth/ne_10m_admin_0_countries.zip"

unzip -o "$DATADIR/Natural_Earth/ne_10m_admin_0_countries.zip" \
  -d "$DATADIR/Natural_Earth"

echo "Natural Earth Admin 0 Countries downloaded and extracted to:"
echo "$DATADIR/Natural_Earth"

echo "For UNEP-WCMC MEOW/PPOW marine biogeographic provinces, download from:"
echo "https://data-gis.unep-wcmc.org/portal/home/item.html?id=05f529264a2b45dfa8a6865ee4612051"
echo "or use the short link:"
echo "https://wcmc.io/WCMC_036"
echo "Place the extracted files in:"
echo "$DATADIR/Biogeographic_Provinces"

echo "Done."
