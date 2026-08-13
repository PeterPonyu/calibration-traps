#!/usr/bin/env bash
# Copy+validate portal. Does not compile papers. Does not enable GitHub Pages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

import jsonschema

root = Path(sys.argv[1])
schema = json.loads((root / "schema" / "figure-index.schema.json").read_text(encoding="utf-8"))
index_path = root / "papers" / "FIGURE-INDEX.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
jsonschema.validate(instance=index, schema=schema)
pdfs = list((root / "papers").rglob("*.pdf"))
if pdfs:
    raise SystemExit("refusing papers/**/*.pdf: " + ", ".join(str(p) for p in pdfs))
print("FIGURE-INDEX validated")
PY

SITE="$ROOT/_site"
rm -rf "$SITE"
mkdir -p "$SITE/data/summaries"
cp -a "$ROOT/portal/." "$SITE/"
rm -f "$SITE/build.sh"
cp "$ROOT/papers/FIGURE-INDEX.json" "$SITE/data/figures.json"
if [[ -d "$ROOT/papers/figs/summaries" ]]; then
  cp -a "$ROOT/papers/figs/summaries/." "$SITE/data/summaries/"
fi
if [[ -d "$ROOT/papers/previews" ]]; then
  mkdir -p "$SITE/data/previews"
  cp -a "$ROOT/papers/previews/." "$SITE/data/previews/"
fi
# Hygiene: never ship experiments logs or lab state.
rm -rf "$SITE/experiments" "$SITE/.omc" "$SITE/.omx"
echo "wrote $SITE"
