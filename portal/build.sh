#!/usr/bin/env bash
# Validate FIGURE-INDEX, Next.js static export, stage _site/. No LaTeX.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path
import jsonschema

root = Path(".")
schema = json.loads((root / "papers/FIGURE-INDEX.schema.json").read_text(encoding="utf-8"))
index = json.loads((root / "papers/FIGURE-INDEX.json").read_text(encoding="utf-8"))
jsonschema.validate(instance=index, schema=schema)
pdfs = sorted(p for p in (root / "papers").rglob("*.pdf") if p.is_file())
if pdfs:
    raise SystemExit(f"refuse: PDFs under papers/: {pdfs}")
print("INDEX valid; no papers/**/*.pdf")
PY

mkdir -p portal/public/data/figs
cp papers/FIGURE-INDEX.json portal/public/data/figures.json
cp papers/FIGURE-INDEX.json portal/public/data/FIGURE-INDEX.json
if [[ -d papers/figs/summaries ]]; then
  rm -rf portal/public/data/figs/summaries
  cp -a papers/figs/summaries portal/public/data/figs/summaries
fi
if [[ -d papers/figs/previews ]]; then
  rm -rf portal/public/data/figs/previews
  cp -a papers/figs/previews portal/public/data/figs/previews
fi

(
  cd portal
  if [[ -f package-lock.json ]]; then
    npm ci
  else
    npm install
  fi
  npm run build
)

rm -rf _site
mkdir -p _site
cp -a portal/out/. _site/
if [[ ! -f _site/.nojekyll ]]; then
  : > _site/.nojekyll
fi
if [[ -e _site/experiments || -e _site/.omc ]]; then
  echo "I4: experiments or .omc leaked into _site" >&2
  exit 1
fi
echo "built _site/ from Next.js export (basePath /calibration-traps)"
