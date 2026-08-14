from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.json"
SCHEMA_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.schema.json"
WAREHOUSE_TEX = REPO_ROOT / "papers" / "E2" / "main.tex"
PORTAL = REPO_ROOT / "portal"
CONCEPT_DOI = "10.5281/zenodo.21020386"
VERSION_DOI = "10.5281/zenodo.21020387"
GITHUB = "PeterPonyu/calibration-traps"
GITHUB_URL = "https://github.com/PeterPonyu/calibration-traps"
PIPELINE = "papers/figs/PIPELINE.md"

# Opt-in hook for maintainers: point CALTRAPS_LAB_TREE at the private
# cross-paper working tree to enable byte-level comparisons against the
# upstream schema / canonical manuscript. CI runners never set this, so those
# comparisons skip there; the committed suite only relies on in-repo
# invariants. No absolute machine paths are recorded in the repo.
LAB_TREE = os.environ.get("CALTRAPS_LAB_TREE", "").strip()


def lab_path(*parts: str) -> Path | None:
    if not LAB_TREE:
        return None
    candidate = Path(LAB_TREE).joinpath(*parts)
    return candidate if candidate.is_file() else None


def portal_blob() -> str:
    parts: list[str] = []
    if not PORTAL.is_dir():
        return ""
    skip = {"node_modules", ".next", "out"}
    for path in PORTAL.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.suffix.lower() in {".html", ".css", ".js", ".md", ".json", ".ts", ".tsx"}:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def workflow_path(name: str) -> Path:
    return REPO_ROOT / ".github" / "workflows" / name
