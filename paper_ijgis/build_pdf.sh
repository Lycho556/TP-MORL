#!/bin/sh
# Compile manuscript and title page offline with tectonic + the local file bundle.
cd "$(dirname "$0")"
TEC=${TECTONIC:-tectonic}
BUNDLE="$(cd ..; pwd)/.tex_bundle"
for f in IJGIS_manuscript_EN.tex title_page.tex; do
  "$TEC" --keep-logs -b "$BUNDLE" "$f" || exit 1
done
