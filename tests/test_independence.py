"""Freeze / independence / no live journal PDF (Z3, Z5) plus lab-tree guards."""

from __future__ import annotations

from conftest import PORTAL, REPO_ROOT, portal_blob


def test_portal_has_no_compiled_manuscript_pdf() -> None:
    names = {
        p.name
        for p in PORTAL.rglob("*.pdf")
        if "node_modules" not in p.parts and ".next" not in p.parts
    }
    assert "main.pdf" not in names
    assert "manuscript.pdf" not in names


def test_no_suite_chrome() -> None:
    blob = portal_blob().lower()
    assert "five-paper" not in blob
    assert "companion suite" not in blob
    assert "muon-norm-cap-grokking" not in blob
    assert "grokking-clock" not in blob
    assert "architecture-staircase" not in blob
    assert "free-repetition-band" not in blob


def test_pointer_tex_is_full_manuscript() -> None:
    tex = REPO_ROOT / "papers" / "E2" / "main.tex"
    assert tex.is_file()
    text = tex.read_text(encoding="utf-8")
    assert "scan-statistic" in text.lower() or "emergence" in text.lower()
    assert len(text.splitlines()) > 1000
