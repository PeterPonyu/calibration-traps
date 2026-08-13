"""Citation / license contract (C1–C4, G4)."""

from __future__ import annotations

import json

import yaml

from conftest import CONCEPT_DOI, REPO_ROOT, VERSION_DOI


def test_citation_cff_parses_and_uses_concept_doi() -> None:
    data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data.get("title"), str) and data["title"].strip()
    assert "\n" not in data["title"]
    assert data["doi"] == CONCEPT_DOI
    values = {
        item.get("value")
        for item in (data.get("identifiers") or [])
        if isinstance(item, dict)
    }
    assert VERSION_DOI in values


def test_citation_cff_is_not_five_paper_bundle() -> None:
    blob = json.dumps(
        yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    ).lower()
    assert "muon-norm-cap" not in blob
    assert "free-repetition-band" not in blob
    assert "grokking-clock" not in blob
    assert "architecture-staircase" not in blob


def test_dual_license_notice_preserved() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Creative Commons Attribution 4.0" in license_text or "CC BY 4.0" in license_text


def test_readme_names_github() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/PeterPonyu/calibration-traps" in text
    assert CONCEPT_DOI in text
