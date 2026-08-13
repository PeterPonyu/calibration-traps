from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.json"
SCHEMA_PATH = REPO_ROOT / "papers" / "FIGURE-INDEX.schema.json"
LAB_SCHEMA = Path("/home/zeyufu/Desktop/dl-research/.omx/plans/figure-index.schema.json")
CANONICAL_TEX = Path("/home/zeyufu/Desktop/dl-research/papers/E2/main.tex")
PORTAL = REPO_ROOT / "portal"
CONCEPT_DOI = "10.5281/zenodo.21020386"
VERSION_DOI = "10.5281/zenodo.21020387"
GITHUB = "PeterPonyu/calibration-traps"
GITHUB_URL = "https://github.com/PeterPonyu/calibration-traps"
PIPELINE = "papers/figs/PIPELINE.md"


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
