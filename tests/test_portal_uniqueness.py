"""E2 portal uniqueness subset (test spec U1–U6, P-E2). No full console UI required."""

from __future__ import annotations

import re
from pathlib import Path

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
OTHER_PAPER_CLASSES = (
    "instrument",
    "atlas",
    "notebook",
    "field-guide",
    "field_guide",
    "fieldguide",
)
ROOT_ABSOLUTE_ASSET_RE = re.compile(
    r"""(?:href|src)=["']/(?!/)[^"']+""",
    re.IGNORECASE,
)


def _portal_files(repo_root: Path) -> list[Path]:
    portal = repo_root / "portal"
    assert portal.is_dir(), "portal/ source tree is required"
    return [
        path
        for path in portal.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".css", ".js", ".svg"}
    ]


def test_portal_uses_three_pane_detector_console_landmark(repo_root: Path) -> None:
    html = (repo_root / "portal" / "index.html").read_text(encoding="utf-8")
    assert "console" in html.lower()
    assert "detector" in html.lower() or "console-panes" in html or 'class="console"' in html


def test_portal_declares_jetbrains_mono_and_newsreader(repo_root: Path) -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in _portal_files(repo_root))
    assert "JetBrains Mono" in blob
    assert "Newsreader" in blob


def test_portal_nav_includes_nomogram_bigbench_and_preflight(repo_root: Path) -> None:
    html = (repo_root / "portal" / "index.html").read_text(encoding="utf-8")
    for label in ("Nomogram", "BIG-Bench", "Preflight"):
        assert label in html, label


def test_portal_does_not_reuse_other_papers_layout_class_names(repo_root: Path) -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in _portal_files(repo_root))
    for cls in OTHER_PAPER_CLASSES:
        assert cls not in blob, cls


def test_portal_contains_no_emoji(repo_root: Path) -> None:
    for path in _portal_files(repo_root):
        text = path.read_text(encoding="utf-8")
        assert EMOJI_RE.search(text) is None, path


def test_portal_footer_lists_concept_doi_github_and_dual_license(repo_root: Path) -> None:
    html = (repo_root / "portal" / "index.html").read_text(encoding="utf-8")
    assert "10.5281/zenodo.21020386" in html
    assert "github.com/PeterPonyu/calibration-traps" in html
    assert "MIT" in html
    assert "CC BY 4.0" in html or "CC-BY" in html


def test_portal_does_not_host_live_journal_pdfs(repo_root: Path) -> None:
    forbidden = ("main.pdf", "manuscript.pdf")
    for root_name in ("portal", "_site"):
        root = repo_root / root_name
        if not root.exists():
            continue
        names = {path.name for path in root.rglob("*") if path.is_file()}
        for name in forbidden:
            assert name not in names, f"{root_name}/{name}"


def test_portal_consumes_figure_index_contract(repo_root: Path) -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in _portal_files(repo_root))
    assert "FIGURE-INDEX.json" in blob or "data/figures.json" in blob
    assert "Figure3.pdf" not in blob


def test_portal_assets_use_relative_urls(repo_root: Path) -> None:
    for path in _portal_files(repo_root):
        text = path.read_text(encoding="utf-8")
        assert ROOT_ABSOLUTE_ASSET_RE.search(text) is None, path


def test_warehouse_has_no_shared_portal_theme_package(repo_root: Path) -> None:
    assert not (repo_root / "portal-theme").exists()
    assert not (repo_root / "portal" / "tokens.css").exists()
