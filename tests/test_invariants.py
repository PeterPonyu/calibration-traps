"""Warehouse invariants (G4, Z3) that CI can enforce without enabling Pages."""

from __future__ import annotations

from pathlib import Path


def test_license_states_mit_code_and_cc_by_40_data(repo_root: Path) -> None:
    text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Creative Commons Attribution 4.0" in text or "CC BY 4.0" in text


def test_portal_does_not_present_the_five_papers_as_one_website(
    repo_root: Path,
) -> None:
    portal = repo_root / "portal"
    if not portal.exists():
        raise AssertionError("portal/ is required")
    blob = "\n".join(
        path.read_text(encoding="utf-8")
        for path in portal.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".css", ".js"}
    ).lower()
    forbidden = (
        "five-paper",
        "paper suite",
        "companion papers",
        "muon-norm-cap-grokking",
        "grokking-clock",
        "architecture-staircase",
        "free-repetition-band",
    )
    for needle in forbidden:
        assert needle not in blob, needle
