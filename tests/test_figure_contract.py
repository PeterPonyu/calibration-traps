"""Figure-pointer contract (test spec F1–F8) for Paper E2."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest
import yaml

SCHEMATIC_SUFFIXES = ("_landscape", "_scheme")
MANIFEST_SUMMARY_IDS = {
    "E2_nomogram",
    "E2_td_grid",
    "E2_case",
    "E2_budget_grid",
    "E2_curriculum",
    "E2_induction",
    "E2_specroute",
    "E2_supervised_fit",
    "E2_routemat",
    "E2_positive_rescue",
}


def _load_index(figure_index_path: Path) -> dict:
    return json.loads(figure_index_path.read_text(encoding="utf-8"))


def test_figure_index_file_exists(figure_index_path: Path) -> None:
    assert figure_index_path.is_file(), "papers/FIGURE-INDEX.json is the portal contract"


def test_figure_index_validates_against_schema(
    repo_root: Path, figure_index_path: Path
) -> None:
    schema_path = repo_root / "schema" / "figure-index.schema.json"
    assert schema_path.is_file(), "shared FIGURE-INDEX schema must be vendored in the warehouse"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=_load_index(figure_index_path), schema=schema)


def test_figure_index_identifies_paper_e2_on_calibration_traps(
    figure_index_path: Path,
) -> None:
    index = _load_index(figure_index_path)
    assert index["paper_id"] == "E2"
    assert index["github"] == "PeterPonyu/calibration-traps"
    assert index["zenodo_concept_doi"] == "10.5281/zenodo.21020386"


def test_figure_ids_match_manifest_or_documented_schematics(
    repo_root: Path, figure_index_path: Path
) -> None:
    manifest_path = repo_root / "papers" / "figs" / "figure_manifest.yaml"
    assert manifest_path.is_file()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    known = {
        figure["artifact"]
        for figure in manifest["papers"]["E2"]["figures"]
    }
    index = _load_index(figure_index_path)
    for figure in index["figures"]:
        fig_id = figure["id"]
        is_schematic = fig_id.endswith(SCHEMATIC_SUFFIXES)
        assert fig_id in known or is_schematic, fig_id
        if is_schematic and fig_id not in known:
            assert figure.get("generator"), f"{fig_id} schematic needs a generator pointer"


def test_summaries_exist_for_every_manifest_figure_that_declares_summary(
    repo_root: Path, figure_index_path: Path
) -> None:
    index = _load_index(figure_index_path)
    by_id = {figure["id"]: figure for figure in index["figures"]}
    for fig_id in MANIFEST_SUMMARY_IDS:
        assert fig_id in by_id, fig_id
        summary_rel = by_id[fig_id]["summary"]
        summary_path = repo_root / "papers" / summary_rel
        if not summary_path.is_file():
            summary_path = repo_root / summary_rel
        assert summary_path.is_file(), f"missing summary for {fig_id}: {summary_rel}"


def test_git_tracks_no_pdf_under_paper_trees(repo_root: Path) -> None:
    listed = subprocess.check_output(
        ["git", "ls-files", "paper/**/*.pdf", "papers/**/*.pdf"],
        cwd=repo_root,
        text=True,
    ).strip()
    assert listed == "", listed


def test_main_tex_does_not_include_preview_svgs(main_tex_path: Path) -> None:
    text = main_tex_path.read_text(encoding="utf-8")
    assert "previews/" not in text


def test_main_tex_inputs_unmodified_figpreamble_contract(main_tex_path: Path) -> None:
    text = main_tex_path.read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in text


def test_main_tex_uses_figtikz_and_not_venue_flat_names(main_tex_path: Path) -> None:
    text = main_tex_path.read_text(encoding="utf-8")
    assert r"\figtikz" in text
    assert not re.search(r"\\includegraphics\{Figure\d+\.pdf\}", text)
    assert not re.search(r"\\includegraphics\{E2_landscape\.pdf\}", text)


def test_gitignore_excludes_compiled_figure_tiers(repo_root: Path) -> None:
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "figs/tex/" in gitignore or "papers/figs/tex/" in gitignore
    assert "figs/vec/" in gitignore or "papers/figs/vec/" in gitignore


def test_figpreamble_keeps_relative_tikz_and_vec_paths(repo_root: Path) -> None:
    preamble = (repo_root / "papers" / "figs" / "figpreamble.tex").read_text(
        encoding="utf-8"
    )
    assert r"\graphicspath{{../figs/vec/}{../figs/}}" in preamble
    assert r"\input{../figs/tex/#2.tex}" in preamble
