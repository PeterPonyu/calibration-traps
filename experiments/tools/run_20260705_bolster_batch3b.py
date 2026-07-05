#!/usr/bin/env python3
"""Batch 3b (reserve follow-up, 2026-07-05): M- and N-axis tests in the LIVE
false-positive regime. Batch-1/2 analysis showed gamma=0 pure noise has q~0
(mean acc ~0.29 vs threshold 0.8) so its M/N arms are negative controls only;
the drift-elevated regime gamma=0.9 (qbar~0.035) is where the formula makes
non-trivial predictions. Cells:
  A7: gamma=0.9, steps=20000 (M=401), s0-9  -> single/sus2 rates vs M
  A8: gamma=0.9, eval_mrps in {16,64}, s0-4 -> measured q(N) vs Binom(N,p) tail
"""
from __future__ import annotations
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path('/home/zeyufu/Desktop/dl-research/experiments')
PY = sys.executable
N_WORKERS = 6

sys.path.insert(0, str(EXP / 'tools'))
from run_20260705_bolster_batch2 import ICRL_CUSTOM  # noqa: E402


def cell_cmds():
    cmds = []
    for s in range(10):
        cmds.append((f'A7 g0.9 M401 s{s}', [PY, '-c', ICRL_CUSTOM.format(
            gamma=0.9, steps=20000, mrps=32, seed=s,
            fname=f'gamma0p9_M401_T40_s{s}.jsonl')]))
    for mrps in (16, 64):
        for s in range(5):
            cmds.append((f'A8 g0.9 N{mrps} s{s}', [PY, '-c', ICRL_CUSTOM.format(
                gamma=0.9, steps=10000, mrps=mrps, seed=s,
                fname=f'gamma0p9_N{mrps}_T40_s{s}.jsonl')]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par3b] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par3b] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par3b] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
