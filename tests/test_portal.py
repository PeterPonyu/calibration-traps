"""Portal uniqueness, Next.js chrome, and no-paper-leak (P-E2, U-STUB, U1, U3–U7)."""

from __future__ import annotations

import re

from conftest import GITHUB_URL, PORTAL, portal_blob

MODULE_ROUTES = (
    "nomogram",
    "testbed",
    "drift",
    "bigbench",
    "preflight",
    "reproduce",
)

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
STUB = re.compile(
    r"CI stub|instrument stub|Two-probe contract stub|"
    r"waits on a user-approved reference\.png|\bstub\b",
    re.I,
)
LEAK = re.compile(
    r"42/45|26/45|0\.00272|201 checkpoints|"
    r"24\s+CONFIRM|7\s+UNPOWERED|"
    r"SYMBOLIC-42|ALGO-BFS-17|NL-ENTAIL|TAB-REASON|"
    r"CODE-EXEC-23|MATH-INDUCT-14|STRUCT-EDIT-39|"
    r"hindu_knowledge|suicide_risk|"
    r"\\tau\s*=\s*0\.[78]|τ=0\.[78]",
    re.I,
)


def _source_files() -> list[Path]:
    skip = {"node_modules", ".next", "out"}
    files: list[Path] = []
    for path in PORTAL.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.suffix.lower() in {".html", ".css", ".js", ".ts", ".tsx", ".md"}:
            files.append(path)
    return files


def test_next_config_export_and_base_path() -> None:
    cfg = (PORTAL / "next.config.ts").read_text(encoding="utf-8")
    assert "output: \"export\"" in cfg or "output: 'export'" in cfg
    assert "/calibration-traps" in cfg
    assert "basePath" in cfg


def test_next_font_jetbrains_and_newsreader() -> None:
    fonts = (PORTAL / "app" / "fonts.ts").read_text(encoding="utf-8")
    # C8: self-hosted woff2 via next/font/local — no build-time font fetch.
    assert "next/font/google" not in fonts
    assert "next/font/local" in fonts
    assert "--font-chrome" in fonts
    assert "--font-prose" in fonts
    for path in re.findall(r'\{ path: "(\./fonts/[^"]+\.woff2)"', fonts):
        assert (PORTAL / "app" / path).is_file(), f"missing committed font {path}"
    assert (PORTAL / "app" / "fonts" / "OFL-jetbrains-mono.txt").is_file()
    assert (PORTAL / "app" / "fonts" / "OFL-newsreader.txt").is_file()


def test_anti_stub() -> None:
    assert STUB.search(portal_blob()) is None


def test_three_pane_console_landmark() -> None:
    src = (PORTAL / "components" / "Console.tsx").read_text(encoding="utf-8")
    assert "data-layout=" in src and "three-pane" in src
    assert "pane-controls" in src
    assert "pane-scan" in src
    assert "pane-adjudication" in src


def test_type_pair_not_siblings() -> None:
    blob = portal_blob()
    assert re.search(r"jetbrains[-_ ]mono", blob, re.I)
    assert re.search(r"newsreader", blob, re.I)
    for forbidden in (
        "IBM Plex Sans",
        "IBM Plex Mono",
        "Source Serif 4",
        "Source Sans 3",
        "Literata",
        "STIX Two Text",
        "Fraunces",
        "Atkinson Hyperlegible",
    ):
        assert forbidden not in blob


def test_nav_modules() -> None:
    blob = (PORTAL / "components" / "Console.tsx").read_text(
        encoding="utf-8"
    ) + (PORTAL / "lib" / "modules.ts").read_text(encoding="utf-8")
    for label in ("Nomogram", "Testbed", "Drift", "BIG-Bench", "Preflight", "Reproduce"):
        assert label in blob
    assert "hrefForModule" in blob
    assert "usePathname" in blob


def test_app_router_module_pages_exist() -> None:
    for slug in MODULE_ROUTES:
        page = PORTAL / "app" / slug / "page.tsx"
        assert page.is_file(), f"missing App Router page for /{slug}/"
        text = page.read_text(encoding="utf-8")
        assert "Console" in text


def test_footer_doi_github_license() -> None:
    blob = portal_blob()
    assert "10.5281/zenodo.21020386" in blob
    assert GITHUB_URL.replace("https://", "") in blob
    assert "MIT" in blob
    assert re.search(r"CC\s*BY\s*4\.0", blob)


CHROME_FILES = (
    PORTAL / "components" / "Console.tsx",
    PORTAL / "app" / "layout.tsx",
    PORTAL / "lib" / "modules.ts",
    *(PORTAL / "app").rglob("page.tsx"),
)

FORBIDDEN_CHROME = re.compile(
    r"\bpapers?\b|\bjournals?\b|\bmanuscripts?\b|\bsubmissions?\b|"
    r"\bJMLR\b|main\.tex|FIGURE-INDEX|PIPELINE|\bwarehouse\b|"
    r"\bdocuments?\b|\bdocumented\b",
    re.I,
)


def _chrome_blob() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in CHROME_FILES)


def test_console_chrome_has_no_forbidden_words() -> None:
    hits: list[str] = []
    for path in CHROME_FILES:
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN_CHROME.finditer(text):
            hits.append(f"{path.relative_to(PORTAL)}: {match.group(0)}")
    assert not hits, hits


def test_console_does_not_cite_venue_pdfs() -> None:
    blob = _chrome_blob()
    assert "papers/submissions/E2-jmlr/" not in blob
    assert "Figure3.pdf" not in blob
    assert "main.pdf" not in blob


def test_nomogram_documented_not_live() -> None:
    src = (PORTAL / "components" / "Console.tsx").read_text(encoding="utf-8")
    assert re.search(r"\bM\b", src)
    assert re.search(r"\bq\b", src)
    assert re.search(r"\bK\b", src)
    lowered = src.lower()
    assert "not live" in lowered or "hold" in lowered


def test_no_paper_number_leak() -> None:
    hits: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        match = LEAK.search(text)
        if match:
            hits.append(f"{path.relative_to(PORTAL)}: {match.group(0)}")
    assert not hits, hits


def test_no_emoji() -> None:
    assert EMOJI.search(portal_blob()) is None


def test_no_journal_pdf_in_portal() -> None:
    pdfs = [
        p
        for p in PORTAL.rglob("*.pdf")
        if "node_modules" not in p.parts and ".next" not in p.parts
    ]
    assert not pdfs


def test_no_shared_theme_package() -> None:
    blob = portal_blob()
    assert "portal-theme" not in blob
    assert "tokens.css" not in blob


def test_no_sibling_layout_tokens() -> None:
    blob = portal_blob().lower()
    assert "isobar" not in blob
    assert "ruled notebook" not in blob
    assert "notebook-gutter" not in blob
    assert "field-guide" not in blob
    assert "instrument-chrome" not in blob


def test_source_has_no_user_site_assets() -> None:
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        assert '"/assets' not in text
        assert "'/assets" not in text
        assert "url(/assets" not in text
