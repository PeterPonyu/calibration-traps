#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "icrl_td"))
sys.path.append(str(ROOT / "experiments"))
import train_icrl as TI  # noqa:E402

OUT = ROOT / "experiments/results/ultragoal_20260619_redteam/e2_td_positive_rescue"


def annotate(summary, hist=None, elapsed=None):
    if elapsed is not None:
        summary["elapsed_wall_sec"] = elapsed
    if hist is not None:
        summary["max_val_acc"] = max((r.get("val_acc", 0) for r in hist), default=0)
        summary["sustain2_0p8"] = any(
            hist[j].get("val_acc", 0) >= 0.8 and hist[j + 1].get("val_acc", 0) >= 0.8
            for j in range(max(0, len(hist) - 1))
        )
        summary["sustain2_0p7"] = any(
            hist[j].get("val_acc", 0) >= 0.7 and hist[j + 1].get("val_acc", 0) >= 0.7
            for j in range(max(0, len(hist) - 1))
        )
    return summary


def completed(p: Path):
    if not p.exists():
        return None
    hist = []
    last = None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        last = rec
        if "_summary" not in rec and "_meta" not in rec:
            hist.append(rec)
    if isinstance(last, dict) and "_summary" in last:
        return annotate(last["_summary"], hist)
    return None


def cells(kind: str):
    out = []
    if kind in ("main", "all"):
        for opt in ["adamw", "muon"]:
            for seed in range(4):
                out.append(
                    (
                        "easy4",
                        TI.Config(
                            optimizer=opt,
                            seed=seed,
                            n_states=4,
                            T=6,
                            d_model=96,
                            n_heads=2,
                            n_layers=2,
                            mlp_ratio=2,
                            batch=256,
                            steps=5000,
                            eval_every=50,
                            eval_mrps=256,
                            acc_thresh=0.8,
                            lr=3e-3,
                            muon_lr=0.03,
                            weight_decay=0.0,
                        ),
                    )
                )
    if kind in ("ultra", "all"):
        for seed in range(6):
            out.append(
                (
                    "ultra2",
                    TI.Config(
                        optimizer="adamw",
                        seed=seed,
                        n_states=2,
                        T=4,
                        d_model=128,
                        n_heads=2,
                        n_layers=2,
                        mlp_ratio=2,
                        batch=512,
                        steps=6000,
                        eval_every=50,
                        eval_mrps=512,
                        acc_thresh=0.8,
                        lr=3e-3,
                        weight_decay=0.0,
                    ),
                )
            )
    if kind in ("wide", "all"):
        for seed in range(3):
            out.append(
                (
                    "wide2",
                    TI.Config(
                        optimizer="adamw",
                        seed=seed,
                        n_states=2,
                        T=6,
                        d_model=192,
                        n_heads=4,
                        n_layers=3,
                        mlp_ratio=2,
                        batch=512,
                        steps=8000,
                        eval_every=100,
                        eval_mrps=512,
                        acc_thresh=0.8,
                        lr=2e-3,
                        weight_decay=0.0,
                    ),
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["main", "ultra", "wide", "all"], default="ultra")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    cs = cells(a.kind)
    for i, (tag, cfg) in enumerate(cs, 1):
        p = (
            OUT
            / f"{tag}_{cfg.name()}_S{cfg.n_states}_dm{cfg.d_model}_L{cfg.n_layers}.jsonl"
        )
        s = completed(p)
        if s is not None:
            print(f"[{i}/{len(cs)}] skip {p.name}", flush=True)
            summaries.append(s)
            continue
        if p.exists():
            p.unlink()
        print(f"[{i}/{len(cs)}] run {p.name} device={cfg.device}", flush=True)
        t = time.time()
        s, h = TI.train(cfg, out_path=str(p), log=lambda *x: None)
        annotate(s, h, time.time() - t)
        summaries.append(s)
        print(
            json.dumps(
                {
                    "tag": tag,
                    "seed": cfg.seed,
                    "final": s.get("final_val_acc"),
                    "max": s.get("max_val_acc"),
                    "sustain2_0p8": s.get("sustain2_0p8"),
                    "emergence": s.get("emergence_step"),
                }
            ),
            flush=True,
        )
    verdict = {
        "namespace": str(OUT.relative_to(ROOT)),
        "kind": a.kind,
        "criterion": "trainable positive if any cell sustains val_acc>=0.8 for two evals; >=0.7 is reported as weaker near-positive evidence",
        "n_cells": len(summaries),
        "n_positive_cells": sum(1 for s in summaries if s.get("sustain2_0p8")),
        "n_near_positive_0p7": sum(1 for s in summaries if s.get("sustain2_0p7")),
        "best_max_val_acc": max(
            (s.get("max_val_acc") or 0 for s in summaries), default=0
        ),
        "summaries": summaries,
    }
    verdict["status"] = (
        "positive_control_pass"
        if verdict["n_positive_cells"]
        else "positive_control_not_yet_passed"
    )
    (OUT / f"verdict_{a.kind}.json").write_text(
        json.dumps(verdict, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(verdict, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
