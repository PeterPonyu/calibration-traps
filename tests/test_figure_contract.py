"""Figure-pointer contract (test-spec F1–F9, F6b) for calibration-traps."""

from __future__ import annotations

import json
import subprocess

import jsonschema

from conftest import CANONICAL_TEX, INDEX_PATH, LAB_SCHEMA, REPO_ROOT, SCHEMA_PATH

MANIFEST_IDS = {
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


def _index() -> dict:
    assert INDEX_PATH.is_file(), f"F1: missing {INDEX_PATH}"
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_figure_index_and_shared_schema_exist() -> None:
    assert INDEX_PATH.is_file()
    assert SCHEMA_PATH.is_file()
    assert LAB_SCHEMA.is_file()
    assert SCHEMA_PATH.read_bytes() == LAB_SCHEMA.read_bytes()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    paper_id = schema["properties"]["paper_id"]
    assert "const" not in paper_id
    assert set(paper_id["enum"]) == {"A", "B", "C", "E1", "E2"}


def test_figure_index_validates() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=_index(), schema=schema)


def test_index_identifies_e2_warehouse() -> None:
    data = _index()
    assert data["paper_id"] == "E2"
    assert data["github"] == "PeterPonyu/calibration-traps"
    assert data["zenodo_concept_doi"] == "10.5281/zenodo.21020386"
    assert data["pipeline"] == "figs/PIPELINE.md"


def test_index_ids_match_manifest() -> None:
    ids = {fig["id"] for fig in _index()["figures"]}
    assert ids == MANIFEST_IDS
    assert "E2_grid" not in ids


def test_index_path_grammar() -> None:
    data = _index()
    for fig in data["figures"]:
        for key in ("generator", "summary", "preview_svg", "tex_build", "vec_build"):
            value = fig.get(key)
            if value is None:
                continue
            assert str(value).startswith("figs/"), f"F9: {fig['id']} {key}={value}"


def test_no_pdfs_committed_under_papers() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "papers/**/*.pdf"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not tracked, f"F4: {tracked}"


def test_tex_does_not_include_previews() -> None:
    tex = (REPO_ROOT / "papers" / "E2" / "main.tex").read_text(encoding="utf-8")
    assert "previews/" not in tex


def test_canonical_includes_survive() -> None:
    tex = (REPO_ROOT / "papers" / "E2" / "main.tex").read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in tex
    assert r"\graphicspath{{./}}" not in tex
    assert "Figure3.pdf" not in tex
    assert "E2_grid.pdf" in tex
    assert r"\figtikz" in tex


def test_full_canonical_tex_not_stub() -> None:
    warehouse = (REPO_ROOT / "papers" / "E2" / "main.tex").read_text(encoding="utf-8")
    canonical = CANONICAL_TEX.read_text(encoding="utf-8")
    expected = canonical.replace("\\graphicspath{{./}}\n", "", 1).replace(
        "% figpreamble severed by build_submission_figs.py "
        "(figures are self-contained standalone PDFs under figures/)\n",
        "",
        1,
    )
    assert warehouse == expected
    assert warehouse.count("\n") > 1000


def test_summaries_exist_when_declared() -> None:
    missing = []
    for fig in _index()["figures"]:
        summary = fig.get("summary")
        if not summary:
            continue
        path = REPO_ROOT / "papers" / summary
        if not path.is_file():
            missing.append(f"{fig['id']}: {summary}")
        preview = fig.get("preview_svg")
        if preview:
            ppath = REPO_ROOT / "papers" / preview
            if not ppath.is_file():
                missing.append(f"{fig['id']}: {preview}")
    assert not missing, missing
    assert not (REPO_ROOT / "papers" / "figs" / "summaries" / "E2_grid.json").exists()


def test_gitignore_excludes_compiled_figure_tiers() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "papers/figs/tex/" in gitignore
    assert "papers/figs/vec/" in gitignore
