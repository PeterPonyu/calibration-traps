#!/usr/bin/env python3
"""E2-B4 verdict -- does the gamma=0.5 rung respond to budget once the state
count is held fixed?

Reads revision2026/gpu2026/e2b4/*.jsonl (written by
run_e2b4_state_matched.py). Every cell is n_states=6 with the budget grid's
M=201 checkpoints and N=128 de-noised evaluator, so the (gamma x budget)
contrast is the only thing that moves.

Readouts per (gamma, steps, T, optimizer) cell, over its seeds:
  cross_TAU      single-crossing detector -- any eval >= TAU
  sustainK_TAU   K consecutive evals >= TAU (K = 2, 3), the de-noised
                 criterion; strict, so unlike train_icrl's emergence_step a
                 lone final-eval crossing does not count
  onset          first step of the earliest sustained-2 window at TAU
  best/final     max and last val_acc per run
  modal_baseline constant-predictor floor for the cell's (n_states, gamma)
                 from modal_baseline.py -- the number an accuracy must clear
                 before any of it is in-context computation

TAU = 0.7 is the pre-registered threshold the budget grid logged; TAU = 0.8
is the gamma ladder's, reported as the secondary.

Writes e2b4_verdict.json beside the run outputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import modal_baseline as MB  # noqa: E402

ROOT = _THIS.parents[1]
DEFAULT_DIR = ROOT / "experiments" / "revision2026" / "gpu2026" / "e2b4"
TAUS = (0.7, 0.8)
SUSTAIN_KS = (2, 3)


# ---------------------------------------------------------------- detectors

def crossed(accs, tau):
    """Single-crossing detector: any checkpoint at or above tau."""
    return any(a >= tau for a in accs)


def sustained(accs, tau, k):
    """K consecutive checkpoints at or above tau."""
    if k < 1:
        raise ValueError("k must be >= 1")
    run = 0
    for a in accs:
        run = run + 1 if a >= tau else 0
        if run >= k:
            return True
    return False


def sustained_onset(steps, accs, tau, k):
    """Step at which the earliest sustained-K window opens, else None."""
    run = 0
    for i, a in enumerate(accs):
        run = run + 1 if a >= tau else 0
        if run >= k:
            return steps[i - k + 1]
    return None


def detector_self_test():
    """Planted traces: the sustained criterion must reject isolated spikes."""
    spike = [0.1, 0.9, 0.1, 0.1]
    assert crossed(spike, 0.7) and not sustained(spike, 0.7, 2)
    real = [0.1, 0.2, 0.75, 0.85, 0.9]
    assert sustained(real, 0.7, 3) and sustained(real, 0.8, 2)
    assert not sustained(real, 0.8, 3)
    assert sustained_onset([0, 10, 20, 30, 40], real, 0.7, 2) == 20
    assert sustained_onset([0, 10, 20, 30], spike, 0.7, 2) is None
    tail = [0.1, 0.1, 0.9]
    assert crossed(tail, 0.8) and not sustained(tail, 0.8, 2), \
        "a lone final-eval crossing must not count as sustained"
    assert sustained(tail, 0.8, 1)
    return True


# ---------------------------------------------------------------- loading

def load_runs(directory):
    """One row per completed cell jsonl (sentinel `_e2b4_meta` required)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(str(directory), "*.jsonl"))):
        meta, tail, steps, accs = None, None, [], []
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "_e2b4_meta" in rec:
                tail = rec["_e2b4_meta"]
            elif "_summary" in rec:
                pass
            elif "val_acc" in rec:
                steps.append(rec["step"])
                accs.append(rec["val_acc"])
        if meta is None or tail is None:
            continue
        rows.append({"run": os.path.basename(path)[:-6],
                     "gamma": tail["gamma"], "n_states": tail["n_states"],
                     "steps": meta["steps"], "T": meta["T"],
                     "optimizer": meta["optimizer"], "seed": meta["seed"],
                     "eval_steps": steps, "accs": accs})
    return rows


# ---------------------------------------------------------------- summary

def summarize(rows, n_mrps=400_000):
    baselines = {}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["gamma"], r["steps"], r["T"], r["optimizer"])].append(r)

    out = []
    for key in sorted(cells):
        gamma, steps, t, opt = key
        runs = cells[key]
        bkey = (runs[0]["n_states"], gamma)
        if bkey not in baselines:
            baselines[bkey] = MB.modal_baseline(bkey[0], gamma,
                                                n_mrps=n_mrps)[0]
        best = [max(r["accs"]) if r["accs"] else float("nan") for r in runs]
        final = [r["accs"][-1] if r["accs"] else float("nan") for r in runs]
        rep = {"gamma": gamma, "steps": steps, "T": t, "optimizer": opt,
               "n_states": bkey[0], "n_seeds": len(runs),
               "seeds": sorted(r["seed"] for r in runs),
               "n_checkpoints": sorted({len(r["accs"]) for r in runs}),
               "modal_baseline": round(baselines[bkey], 4),
               "median_best_acc": round(statistics.median(best), 4),
               "median_final_acc": round(statistics.median(final), 4),
               "best_acc": [round(b, 4) for b in best],
               "final_acc": [round(f, 4) for f in final]}
        for tau in TAUS:
            rep[f"cross_{tau}"] = sum(crossed(r["accs"], tau) for r in runs)
            for k in SUSTAIN_KS:
                rep[f"sustain{k}_{tau}"] = sum(
                    sustained(r["accs"], tau, k) for r in runs)
            rep[f"onsets_{tau}"] = [
                sustained_onset(r["eval_steps"], r["accs"], tau, 2)
                for r in runs]
        out.append(rep)
    return {"n_runs": len(rows), "n_cells": len(out), "taus": list(TAUS),
            "sustain_ks": list(SUSTAIN_KS), "cells": out}


def pooled_2x2(report):
    """Emergence counts collapsed onto the (gamma x budget) 2x2 the arm
    exists to complete -- pooled over horizon, optimizer and seed."""
    grid = defaultdict(lambda: defaultdict(int))
    for c in report["cells"]:
        key = (c["gamma"], c["steps"])
        grid[key]["n"] += c["n_seeds"]
        for tau in report["taus"]:
            grid[key][f"cross_{tau}"] += c[f"cross_{tau}"]
            for k in report["sustain_ks"]:
                grid[key][f"sustain{k}_{tau}"] += c[f"sustain{k}_{tau}"]
    return {f"gamma{g}_steps{s}": dict(v) for (g, s), v in sorted(grid.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--n-mrps", type=int, default=400_000,
                    help="samples for the modal-baseline marginals")
    args = ap.parse_args()

    detector_self_test()
    rows = load_runs(args.dir)
    if not rows:
        print(f"no completed E2-B4 runs under {args.dir}")
        return
    report = summarize(rows, n_mrps=args.n_mrps)
    report["pooled_2x2"] = pooled_2x2(report)

    hdr = (f"{'gamma':>6}{'steps':>8}{'T':>4}{'opt':>7}{'n':>4}"
           f"{'floor':>8}{'medBest':>9}{'x0.7':>6}{'s2_0.7':>8}"
           f"{'s3_0.7':>8}{'x0.8':>6}{'s2_0.8':>8}")
    print(hdr)
    for c in report["cells"]:
        print(f"{c['gamma']:>6}{c['steps']:>8}{c['T']:>4}{c['optimizer']:>7}"
              f"{c['n_seeds']:>4}{c['modal_baseline']:>8.3f}"
              f"{c['median_best_acc']:>9.3f}{c['cross_0.7']:>6}"
              f"{c['sustain2_0.7']:>8}{c['sustain3_0.7']:>8}"
              f"{c['cross_0.8']:>6}{c['sustain2_0.8']:>8}")
    print("\npooled 2x2 (gamma x budget), all horizons/optimizers/seeds:")
    for k, v in report["pooled_2x2"].items():
        print(f"  {k:<22} n={v['n']:>3}  sustain2_0.7={v['sustain2_0.7']:>3}"
              f"  sustain2_0.8={v['sustain2_0.8']:>3}"
              f"  cross_0.7={v['cross_0.7']:>3}")

    out = os.path.join(args.dir, "e2b4_verdict.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
