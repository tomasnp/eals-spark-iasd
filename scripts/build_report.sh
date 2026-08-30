#!/usr/bin/env bash
# results/*.csv -> figures -> LaTeX tables and macros -> report.pdf
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
python3 src/plots.py
python3 report/tables.py
cd report
pdflatex -interaction=nonstopmode report.tex >/dev/null
bibtex report >/dev/null || true
pdflatex -interaction=nonstopmode report.tex >/dev/null
pdflatex -interaction=nonstopmode report.tex >/dev/null
cp report.pdf ../Rapport_Sinapi_Ernadote_eALS_Spark.pdf
echo "Rapport_Sinapi_Ernadote_eALS_Spark.pdf: $(pdfinfo report.pdf | awk '/Pages/{print $2}') pages"
grep -c '\[n/a\]' tables/macros.tex | xargs -I{} echo "{} macro(s) still without data"
