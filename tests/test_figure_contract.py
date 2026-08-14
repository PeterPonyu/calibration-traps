"""Figure-pointer contract (test-spec F1–F9, F6b) for calibration-traps."""

from __future__ import annotations

import json
import re
import subprocess

import jsonschema
import pytest

from conftest import INDEX_PATH, REPO_ROOT, SCHEMA_PATH, WAREHOUSE_TEX, lab_path

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

# Line count of papers/E2/main.tex measured from the canonical manuscript on
# 2026-08-14 (2207 upstream lines minus the two stripped submission scars).
# The warehouse copy must stay within 5% of this floor so a stub or truncated
# pointer file can never pass.
CANONICAL_WAREHOUSE_LINES = 2205
LINE_TOLERANCE = 0.05


def _index() -> dict:
    assert INDEX_PATH.is_file(), f"F1: missing {INDEX_PATH}"
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_figure_index_and_shared_schema_exist() -> None:
    assert INDEX_PATH.is_file()
    assert SCHEMA_PATH.is_file()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # Structural invariants of the shared five-paper schema (in-repo SSOT).
    assert schema.get("type") == "object"
    assert "paper_id" in schema.get("required", [])
    assert "figures" in schema.get("required", [])
    paper_id = schema["properties"]["paper_id"]
    assert "const" not in paper_id
    assert set(paper_id["enum"]) == {"A", "B", "C", "E1", "E2"}
    figures = schema["properties"]["figures"]
    assert figures.get("type") == "array"
    for key in ("id", "generator", "summary", "preview_svg", "tex_build", "vec_build"):
        assert key in figures["items"]["properties"], f"schema missing figures[].{key}"


def test_schema_matches_upstream_when_available() -> None:
    # Optional maintainer check: byte-compare against the upstream schema when
    # CALTRAPS_LAB_TREE points at the private working tree. Always skips on
    # CI runners; the in-repo invariants above are the portable contract.
    upstream = lab_path(".omx", "plans", "figure-index.schema.json")
    if upstream is None:
        pytest.skip("upstream lab schema not configured (CALTRAPS_LAB_TREE unset)")
    assert SCHEMA_PATH.read_bytes() == upstream.read_bytes()


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
    tex = WAREHOUSE_TEX.read_text(encoding="utf-8")
    assert "previews/" not in tex


def test_canonical_includes_survive() -> None:
    tex = WAREHOUSE_TEX.read_text(encoding="utf-8")
    assert r"\input{../figs/figpreamble.tex}" in tex
    assert r"\graphicspath{{./}}" not in tex
    assert "Figure3.pdf" not in tex
    assert "E2_grid.pdf" in tex
    assert r"\figtikz" in tex


def test_full_canonical_tex_not_stub() -> None:
    assert WAREHOUSE_TEX.is_file(), f"missing {WAREHOUSE_TEX}"
    warehouse = WAREHOUSE_TEX.read_text(encoding="utf-8")
    # Full-length floor: canonical manuscript length measured 2026-08-14,
    # 5% tolerance. A stub or truncated pointer trips this immediately.
    lines = warehouse.count("\n")
    lower = CANONICAL_WAREHOUSE_LINES * (1 - LINE_TOLERANCE)
    upper = CANONICAL_WAREHOUSE_LINES * (1 + LINE_TOLERANCE)
    assert lower <= lines <= upper, (
        f"main.tex line count {lines} outside [{lower:.0f}, {upper:.0f}]"
    )
    # Scar-strip invariants: submission scars removed, pointer include kept,
    # no venue-flat figure includes.
    assert r"\input{../figs/figpreamble.tex}" in warehouse
    assert r"\graphicspath{{./}}" not in warehouse
    assert not re.search(r"Figure\d+\.pdf", warehouse)
    assert "% figpreamble severed by build_submission_figs.py" not in warehouse


def test_warehouse_tex_matches_canonical_when_available() -> None:
    # Optional maintainer check: full comparison against the canonical
    # manuscript when CALTRAPS_LAB_TREE points at the private working tree.
    canonical = lab_path("papers", "E2", "main.tex")
    if canonical is None:
        pytest.skip("canonical lab manuscript not configured (CALTRAPS_LAB_TREE unset)")
    warehouse = WAREHOUSE_TEX.read_text(encoding="utf-8")
    expected = canonical.read_text(encoding="utf-8").replace(
        "\\graphicspath{{./}}\n", "", 1
    ).replace(
        "% figpreamble severed by build_submission_figs.py "
        "(figures are self-contained standalone PDFs under figures/)\n",
        "",
        1,
    )
    assert warehouse == expected


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
