#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
  echo "No LaTeX engine found. Please install latexmk or pdflatex." >&2
  exit 1
fi

echo "Built $(pwd)/main.pdf"
