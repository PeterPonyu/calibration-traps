"""Actions, build, and pack hygiene (I1–I5b)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from conftest import REPO_ROOT, workflow_path


def _load(name: str) -> dict:
    path = workflow_path(name)
    assert path.is_file(), f"missing {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_pages_workflow_permissions_and_environment() -> None:
    data = _load("pages.yml")
    assert data.get("permissions", {}).get("pages") == "write"
    env_names = []
    for job in (data.get("jobs") or {}).values():
        env = job.get("environment")
        if isinstance(env, str):
            env_names.append(env)
        elif isinstance(env, dict):
            env_names.append(env.get("name"))
    assert "github-pages" in env_names


def test_pages_triggers_main_only_with_path_filters() -> None:
    data = _load("pages.yml")
    on = data.get("on") or data.get(True)
    assert "workflow_dispatch" in on
    push = on.get("push") or {}
    assert push.get("branches") == ["main"]
    paths = push.get("paths") or []
    joined = " ".join(paths)
    assert "portal/**" in joined
    assert "papers/FIGURE-INDEX.json" in joined
    assert "papers/figs/summaries/**" in joined
    assert "papers/figs/previews/**" in joined
    assert ".github/workflows/pages.yml" in joined
    assert "ci/comprehensive" not in yaml.safe_dump(data)


def test_pages_and_ci_do_not_run_latex() -> None:
    for name in ("pages.yml", "ci.yml"):
        text = workflow_path(name).read_text(encoding="utf-8").lower()
        assert "latexmk" not in text
        assert "pdflatex" not in text
        assert "lualatex" not in text


def test_ci_runs_pytest() -> None:
    data = _load("ci.yml")
    blob = yaml.safe_dump(data)
    assert "pytest" in blob


def test_build_script_copy_validate() -> None:
    build = REPO_ROOT / "portal" / "build.sh"
    assert build.is_file()
    text = build.read_text(encoding="utf-8")
    assert "jsonschema" in text
    assert "npm run build" in text
    assert "latexmk" not in text
    assert "cp -a experiments" not in text


def test_gitattributes_export_ignore_website_trees() -> None:
    attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "portal/ export-ignore" in attrs
    assert "_site/ export-ignore" in attrs
    assert ".github/ export-ignore" in attrs


def test_pack_script_excludes_website_trees() -> None:
    script = (REPO_ROOT / "pack_zenodo_tarball.sh").read_text(encoding="utf-8")
    assert "portal" in script
    assert "_site" in script
    assert ".github" in script


def test_site_gitignore() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "_site/" in gitignore
