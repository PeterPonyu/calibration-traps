"""Visual-pixel jobs stay optional and cannot fail required CI."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_visual_pixel_workflow_exists(repo_root: Path) -> None:
    path = repo_root / ".github" / "workflows" / "visual-pixel.yml"
    assert path.is_file()


def test_visual_pixel_workflow_does_not_run_on_push_or_pull_request(
    repo_root: Path,
) -> None:
    text = (repo_root / ".github" / "workflows" / "visual-pixel.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" in text
    # Raw YAML: reject push/PR triggers so this file cannot fail required CI.
    trigger_block = text.split("jobs:")[0]
    assert "push:" not in trigger_block
    assert "pull_request:" not in trigger_block


def test_visual_pixel_job_is_gated_until_reference_is_approved(
    repo_root: Path,
) -> None:
    text = (repo_root / ".github" / "workflows" / "visual-pixel.yml").read_text(
        encoding="utf-8"
    )
    gated = (
        "VISUAL_REFERENCE_APPROVED" in text
        or "if: false" in text
        or "if: ${{ false }}" in text
    )
    assert gated


def test_required_ci_workflow_does_not_invoke_visual_pixel_or_pages_deploy(
    repo_root: Path,
) -> None:
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "deploy-pages" not in text
    assert "pixelmatch" not in text.lower()
    assert "visual-ralph" not in text.lower() or "gated" in text.lower()
