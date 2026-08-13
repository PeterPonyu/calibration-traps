"""Citation / DOI hygiene (test spec C1, C3, C4) for Paper E2."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_citation_cff_parses_as_yaml(repo_root: Path) -> None:
    payload = yaml.safe_load((repo_root / "CITATION.cff").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "title" in payload
    assert isinstance(payload["title"], str)


def test_citation_doi_is_manuscript_concept_not_version(repo_root: Path) -> None:
    payload = yaml.safe_load((repo_root / "CITATION.cff").read_text(encoding="utf-8"))
    assert payload["doi"] == "10.5281/zenodo.21020386"


def test_version_doi_lives_under_identifiers(repo_root: Path) -> None:
    payload = yaml.safe_load((repo_root / "CITATION.cff").read_text(encoding="utf-8"))
    identifiers = payload.get("identifiers") or []
    values = {item.get("value") for item in identifiers}
    assert "10.5281/zenodo.21020387" in values


def test_citation_cff_is_per_paper_not_five_paper_bundle(repo_root: Path) -> None:
    text = (repo_root / "CITATION.cff").read_text(encoding="utf-8")
    assert "muon-norm-cap-grokking" not in text
    assert "architecture-staircase" not in text
    assert "free-repetition-band" not in text
    assert "grokking-clock" not in text
