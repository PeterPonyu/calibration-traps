#!/usr/bin/env python3
"""Generate Paper E2 four-panel summaries from the figure manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "papers" / "figs" / "figure_manifest.yaml"
SUMMARY_DIR = ROOT / "papers" / "figs" / "summaries"


def build_summaries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    figures = manifest["papers"]["E2"]["figures"]
    if len(figures) != 10:
        raise ValueError("Paper E2 manifest must contain exactly ten figures")

    summaries: dict[str, dict[str, Any]] = {}
    for figure in figures:
        panels = figure["panels"]
        if len(panels) != 4 or [panel["id"] for panel in panels] != list("abcd"):
            raise ValueError(f"{figure['artifact']}: panels must be exactly a,b,c,d")
        for panel in panels:
            if not panel["question"].strip() or not panel["source"]:
                raise ValueError(f"{figure['artifact']}/{panel['id']}: missing metadata")
        summaries[figure["artifact"]] = {
            "artifact": figure["artifact"],
            "layout": figure["layout"],
            "panel_count": 4,
            "panels": [
                {
                    "id": panel["id"],
                    "question": panel["question"],
                    "sources": list(panel["source"]),
                }
                for panel in panels
            ],
            "exception_reason": figure["exception_reason"],
            "generated_from_manifest": "papers/figs/figure_manifest.yaml",
        }
    return summaries


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summaries = build_summaries(manifest)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for artifact, summary in summaries.items():
        target = SUMMARY_DIR / f"{artifact}.json"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=SUMMARY_DIR, delete=False
        ) as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)
    print(f"wrote {len(summaries)} Paper E2 summaries")


if __name__ == "__main__":
    main()
