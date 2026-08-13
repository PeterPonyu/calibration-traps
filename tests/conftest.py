from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def figure_index_path(repo_root: Path) -> Path:
    return repo_root / "papers" / "FIGURE-INDEX.json"


@pytest.fixture(scope="session")
def main_tex_path(repo_root: Path) -> Path:
    return repo_root / "papers" / "E2" / "main.tex"
