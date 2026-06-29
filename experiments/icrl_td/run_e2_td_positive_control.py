#!/usr/bin/env python3
"""Run a tiny trainable in-context TD positive-control grid in an isolated namespace.

This does not alter the canonical 016 grid. It asks whether the same model/evaluator
can learn an easier random-MRP TD task when the state space/context are reduced.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "icrl_td"))
sys.path.append(str(ROOT / "experiments"))
import train_icrl as TI  # noqa: E402

OUT = ROOT / "experiments" / "results" / "ultragoal_20260618" / "e2_td_positive_control"
OUT.mkdir(parents=True, exist_ok=True)

cells = []
for opt in ["adamw", "muon"]:
    for seed in [0, 1, 2]:
        cells.append(TI.Config(
            optimizer=opt,
            seed=seed,
            n_states=4,
            T=6,
            d_model=64,
            n_heads=2,
            n_layers=2,
            mlp_ratio=2,
            batch=128,
            steps=2500,
            eval_every=50,
            eval_mrps=128,
            acc_thresh=0.8,
            lr=2e-3,
            muon_lr=0.02,
            weight_decay=0.0,
        ))

summaries = []
def annotate_summary(summary, history=None, elapsed=None):
    if elapsed is not None:
        summary["elapsed_wall_sec"] = elapsed
    if history is None:
        return summary
    summary["max_val_acc"] = max((r.get("val_acc", 0) for r in history), default=0)
    summary["sustain2_0p8"] = any(
        history[j].get("val_acc", 0) >= 0.8 and history[j + 1].get("val_acc", 0) >= 0.8
        for j in range(max(0, len(history) - 1))
    )
    return summary


def completed_summary(path):
    if not path.exists():
        return None
    history = []
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            last = rec
            if "_summary" not in rec:
                history.append(rec)
    if isinstance(last, dict) and "_summary" in last:
        return annotate_summary(last["_summary"], history)
    return None


for i, cfg in enumerate(cells, 1):
    path = OUT / f"positive_{cfg.name()}.jsonl"
    summary = completed_summary(path)
    if summary is not None:
        print(f"[{i}/{len(cells)}] skip done {path.name}", flush=True)
        summaries.append(summary)
        continue
    t0 = time.time()
    print(f"[{i}/{len(cells)}] run {path.name} device={cfg.device}", flush=True)
    summary, hist = TI.train(cfg, out_path=str(path), log=lambda *_: None)
    annotate_summary(summary, hist, time.time() - t0)
    summaries.append(summary)
    print(
        f"[{i}/{len(cells)}] {cfg.name()} final={summary['final_val_acc']} "
        f"max={summary['max_val_acc']} em={summary['emergence_step']}",
        flush=True,
    )

verdict = {
    "namespace": str(OUT.relative_to(ROOT)),
    "task": "E2 trainable in-context TD positive control",
    "n_cells": len(summaries),
    "criterion": (
        "positive if any cell has max_val_acc>=0.8 and sustain2_0p8=true "
        "on the easier n_states=4,T=6 task"
    ),
    "n_positive_cells": sum(
        1 for s in summaries if (s.get("max_val_acc") or 0) >= 0.8 and s.get("sustain2_0p8")
    ),
    "summaries": summaries,
}
verdict["status"] = "positive_control_pass" if verdict["n_positive_cells"] else "positive_control_not_yet_passed"
(OUT / "positive_control_verdict.json").write_text(json.dumps(verdict, indent=2, allow_nan=True) + "\n")
print(json.dumps(verdict, indent=2, allow_nan=True), flush=True)
