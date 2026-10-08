#!/bin/sh
# Rebuild IJGIS_manuscript_EN.docx from the LaTeX source (run from paper_ijgis/).
set -e
TEC=${TECTONIC:-tectonic}; PANDOC=${PANDOC:-pandoc}; PY=${PYTHON:-python}
mkdir -p /tmp/tex_aux
"$TEC" -k -o /tmp/tex_aux -b "$(cd ..; pwd)/.tex_bundle" IJGIS_manuscript_EN.tex
$PY word/tex2docx_prep.py IJGIS_manuscript_EN.tex /tmp/tex_aux/IJGIS_manuscript_EN.aux word/ms_for_word.tex
$PANDOC word/ms_for_word.tex -f latex -s -t json -o word/ms.json
$PY word/place_refs.py word/ms.json word/ms2.json
$PANDOC word/ms2.json -f json -o word/IJGIS_manuscript_EN.docx --citeproc \
  --bibliography references.bib --reference-doc word/reference_ijgis.docx --resource-path=.
