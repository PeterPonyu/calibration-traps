#!/usr/bin/env python3
"""Parallel version of run_20260704_bolster_e1e2.py — same cells, N workers.

All cells are independent (one jsonl per cell, resume-aware: completed logs are
skipped, partial logs deleted and rerun), so they are safe to run concurrently.
Models are 0.4M-10M params; a single run uses <2 GB VRAM and ~10% GPU, so
N_WORKERS=6 roughly 6x-es throughput on the 24 GB 5090.
"""
from __future__ import annotations
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/zeyufu/Desktop/dl-research')
EXP = ROOT / 'experiments'
PY = sys.executable
N_WORKERS = 6
FINE_OUT = EXP / 'results' / 'repeated_data_finegrid'


def cell_cmds():
    cmds = []
    # Stage A: E2 gamma=0.9 s5..s19 (one process per cell; module skip-logic
    # handles completed cells)
    for s in range(5, 20):
        code = ("import sys; sys.path.insert(0, 'icrl_td')\n"
                "import run_e2_gamma_ladder as G\n"
                f"G.run_cells([(0.9, {s})], 'gamma09ext')\n")
        cmds.append((f'A gamma0.9 s{s}', [PY, '-c', code]))
    # Stage B: E1 boundary cells (runner is itself resume-aware)
    for ent in ('low', 'med'):
        for s in range(5, 15):
            cmds.append((f'B small_{ent} s{s}',
                         [PY, 'repeated_data/run_e1_seed_audit_cell.py',
                          '--entropy', ent, '--unique-tokens', '2000000',
                          '--seed', str(s)]))
    # Stage C: E1 med/med fine grid
    for E, U in [(6, 3_333_333), (8, 2_500_000), (12, 1_666_667), (16, 1_250_000)]:
        for s in range(3):
            name = f"med_med_U{U/1e6:g}M_E{E}_s{s}".replace('.', 'p')
            out = FINE_OUT / f"{name}.jsonl"
            code = (
                "import json, sys; from pathlib import Path\n"
                "sys.path.insert(0, 'repeated_data')\n"
                f"out = Path({str(out)!r})\n"
                "if out.exists():\n"
                "    last = ''\n"
                "    for line in out.open():\n"
                "        if line.strip(): last = line\n"
                "    try:\n"
                "        if '_summary' in json.loads(last):\n"
                f"            print('skip {name} (complete)'); raise SystemExit(0)\n"
                "    except SystemExit: raise\n"
                "    except Exception: pass\n"
                "    out.unlink()\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "from train_repeat import Config, run\n"
                f"cfg = Config(capacity='med', entropy_level='med', generator='markov',\n"
                f"             unique_tokens={U}, n_epochs={E}, total_budget=20_000_000, seed={s})\n"
                "run(cfg, out_path=str(out))\n")
            cmds.append((f'C {name}', [PY, '-c', code]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
