"""Next.js export proof: basePath, out layout, INDEX copy, no leak in _site."""

from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import REPO_ROOT

LEAK = re.compile(
    r"42/45|26/45|0\.00272|201 checkpoints|"
    r"24\s+CONFIRM|7\s+UNPOWERED|"
    r"SYMBOLIC-42|ALGO-BFS-17|"
    r"hindu_knowledge|suicide_risk",
    re.I,
)


MODULE_ROUTES = (
    "nomogram",
    "testbed",
    "drift",
    "bigbench",
    "preflight",
    "reproduce",
)


def test_export_index_uses_base_path() -> None:
    site = REPO_ROOT / "_site"
    index = site / "index.html"
    assert index.is_file(), "run portal/build.sh before this check"
    html = index.read_text(encoding="utf-8")
    assert "/calibration-traps/_next/" in html
    assert ".nojekyll" in {p.name for p in site.iterdir()} or (
        site / ".nojekyll"
    ).is_file()


def test_export_nested_module_routes() -> None:
    site = REPO_ROOT / "_site"
    out = REPO_ROOT / "portal" / "out"
    for slug in MODULE_ROUTES:
        site_page = site / slug / "index.html"
        out_page = out / slug / "index.html"
        assert out_page.is_file(), f"missing portal/out/{slug}/index.html — run portal/build.sh"
        assert site_page.is_file(), f"missing _site/{slug}/index.html — run portal/build.sh"
        html = site_page.read_text(encoding="utf-8")
        assert "/calibration-traps/_next/" in html
        assert f"/calibration-traps/{slug}/" in html or f'href="/calibration-traps/{slug}' in html


def test_export_copies_figure_index() -> None:
    figures = REPO_ROOT / "_site" / "data" / "figures.json"
    assert figures.is_file()
    payload = json.loads(figures.read_text(encoding="utf-8"))
    assert payload["paper_id"] == "E2"
    assert (REPO_ROOT / "_site" / "data" / "figs" / "summaries" / "E2_nomogram.json").is_file()


FORBIDDEN_CHROME = re.compile(
    r"\bpapers?\b|\bjournals?\b|\bmanuscripts?\b|\bsubmissions?\b|"
    r"\bJMLR\b|main\.tex|FIGURE-INDEX|PIPELINE|\bwarehouse\b|"
    r"\bdocuments?\b|\bdocumented\b",
    re.I,
)


def test_export_html_has_no_forbidden_chrome() -> None:
    site = REPO_ROOT / "_site"
    hits: list[str] = []
    pages = [site / "index.html"]
    pages.extend(site / slug / "index.html" for slug in MODULE_ROUTES)
    for path in pages:
        assert path.is_file(), f"missing {path} — run portal/build.sh"
        text = path.read_text(encoding="utf-8", errors="replace")
        match = FORBIDDEN_CHROME.search(text)
        if match:
            hits.append(f"{path.relative_to(site)}: {match.group(0)}")
    assert not hits, hits


def test_export_html_does_not_leak_findings() -> None:
    hits: list[str] = []
    site = REPO_ROOT / "_site"
    for path in site.rglob("*"):
        if path.suffix.lower() not in {".html", ".js", ".css", ".json"}:
            continue
        if "summaries" in path.parts or path.name in {"figures.json", "FIGURE-INDEX.json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = LEAK.search(text)
        if match:
            hits.append(f"{path.relative_to(site)}: {match.group(0)}")
    assert not hits, hits


def test_build_script_is_next_export() -> None:
    text = (REPO_ROOT / "portal" / "build.sh").read_text(encoding="utf-8")
    assert "npm run build" in text
    assert "latexmk" not in text
    assert "portal/out" in text
