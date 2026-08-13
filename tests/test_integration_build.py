"""Local unpublished integration (test spec I1–I4, architect CR7)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_portal_build_script_exists(repo_root: Path) -> None:
    script = repo_root / "portal" / "build.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111, "portal/build.sh must be executable"


def test_build_script_does_not_invoke_latexmk(repo_root: Path) -> None:
    text = (repo_root / "portal" / "build.sh").read_text(encoding="utf-8")
    assert "latexmk" not in text
    assert "pdflatex" not in text
    assert "lualatex" not in text


def test_build_script_validates_index_then_copies_portal_to_site(
    repo_root: Path,
) -> None:
    script = repo_root / "portal" / "build.sh"
    subprocess.run(["bash", str(script)], cwd=repo_root, check=True)
    site = repo_root / "_site"
    assert (site / "index.html").is_file()
    assert (site / "data" / "figures.json").is_file()
    assert not (site / "experiments").exists()
    assert not (site / ".omc").exists()
    assert not list(site.rglob("*.pdf"))


def test_pages_workflow_file_declares_pages_write_and_github_pages_environment(
    repo_root: Path,
) -> None:
    workflow = repo_root / ".github" / "workflows" / "pages.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "pages: write" in text
    assert "github-pages" in text
    assert "workflow_dispatch" in text


def test_pages_workflow_path_filters_are_portal_and_index_only(repo_root: Path) -> None:
    text = (repo_root / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "portal/**" in text
    assert "FIGURE-INDEX.json" in text
    assert "summaries/**" in text
    assert "experiments/**" not in text.split("paths:")[1].split("jobs:")[0]


def test_pages_deploy_job_is_gated_so_disabled_pages_cannot_fail_required_ci(
    repo_root: Path,
) -> None:
    text = (repo_root / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "deploy-pages" in text
    gated = (
        "vars.PAGES_ENABLED" in text
        or "if: false" in text
        or "if: ${{ false }}" in text
    )
    assert gated, "deploy job must be gated until Pages is explicitly enabled"


def test_required_ci_workflow_runs_the_comprehensive_pytest_suite(
    repo_root: Path,
) -> None:
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "pytest" in text
    assert "portal/build.sh" in text
