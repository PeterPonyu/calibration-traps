#!/usr/bin/env python3
"""Rebuild figures-redteam/redteam_e2_stats.json from ALL rescue verdict files.

2026-07-10 correction: the original stats JSON counted only the final attack
round (remote_main 8 + ultra2 6 + wide2 3 = 17), silently omitting the two
2026-06-18 pilot arms (positive_control 6, ultra_easy 3). All 26 are distinct
trained runs (different step budgets/seeds), zero sustained-0.8 positives, so
the honest zero-success pool is n=26. Kinds are aggregated per family so the
figure keeps its three bars: main = n4T6 across both rounds (n=14), ultra =
n2T4 across both rounds (n=9), wide = n2T6 897k (n=3).
"""
import json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

SOURCES = {
    "main": [
        "e2_td_positive_control_20260619/remote_main/verdict_main.json",
        "e2_td_positive_control_20260618/positive_control_verdict.json",
    ],
    "ultra": [
        "e2_td_positive_control_20260619/verdict_ultra.json",
        "e2_td_positive_control_20260618/ultra_easy_positive_control_verdict.json",
    ],
    "wide": [
        "e2_td_positive_rescue_20260619/verdict_wide.json",
    ],
}


def load(rel):
    with open(os.path.join(R, rel)) as fh:
        return json.load(fh)


def agg(kind, rels):
    n = 0
    best = -1.0
    near = 0
    pos = 0
    files = []
    for rel in rels:
        d = load(rel)
        summ = d.get("summaries", [])
        n += len(summ)
        best = max(best, max(s["max_val_acc"] for s in summ))
        near += int(d.get("n_near_positive_0p7") or 0)
        pos += int(d.get("n_positive_cells") or 0)
        files.append(rel)
    return {
        "file": " + ".join(files),
        "n_cells": n,
        "n_positive_cells": pos,
        "n_near_positive_0p7": near,
        "best_max_val_acc": best,
        "status": "positive_control_not_yet_passed",
        "kind": kind,
    }


rows = {k: agg(k, v) for k, v in SOURCES.items()}
n_tot = sum(r["n_cells"] for r in rows.values())
assert sum(r["n_positive_cells"] for r in rows.values()) == 0
upper = 1.0 - 0.05 ** (1.0 / n_tot)

out = {
    "fullattack": [rows["main"], rows["ultra"]],
    "redteam": [rows["wide"]],
    "zero_success_upper_95_if_no_success": {
        "successes": 0,
        "n": n_tot,
        "upper95": upper,
    },
    "_provenance": "build_redteam_e2_stats.py 2026-07-10: pools BOTH attack rounds (26 cells); supersedes the final-round-only 17-cell version",
}

dest = os.path.join(R, "figures-redteam", "redteam_e2_stats.json")
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"n_tot={n_tot} upper95={upper:.4f} -> {dest}")
for k, r in rows.items():
    print(f"  {k}: n={r['n_cells']} best={r['best_max_val_acc']:.3f} near0.7={r['n_near_positive_0p7']}")
