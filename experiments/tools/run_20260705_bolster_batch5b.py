#!/usr/bin/env python3
"""Batch 5b (red-team-driven, E2 arms; 2026-07-05 late).

X1: gradual-emergence latency control (E2 red-team MAJOR 2): copy/induction
    family at lowered lr (3e-4, 1.5e-4 vs default 1e-3), 4x budget, 4x finer
    eval cadence (25 steps). If the transition ramps through tau over multiple
    checkpoints, sustained-K confirmation latency becomes measurable; if it
    stays abrupt even at 6.7x lower lr, transition sharpness is robust.
X3: gamma=0.9 seeds 30-59 (E2 red-team MAJOR 3): n=60 halves the CI width on
    the sustained-2 rate and powers the split-half out-of-sample calibration.
"""
from __future__ import annotations
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments'
PY = sys.executable
N_WORKERS = 6

sys.path.insert(0, str(EXP / 'tools'))
from run_20260705_bolster_batch2 import ICRL_CUSTOM  # noqa: E402

GRADUAL = """
import json, sys, time
sys.path.insert(0, 'curriculum_order')
from pathlib import Path
import run_curriculum as RC
from train_curriculum import Config, run
lr, seed = {lr}, {seed}
tag = 'lr{lrtag}'
OUT = Path('results/gradual_emergence')
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f'copy_iid_adamw_{{tag}}_s{{seed}}.jsonl'
if out.exists() and '_summary' in out.read_text().strip().splitlines()[-1]:
    print(f'skip {{out.name}}'); raise SystemExit(0)
cfg = Config(family='copy', ordering='iid', optimizer='adamw', seed=seed,
             steps_per_stage=RC.STEPS_PER_STAGE * 4, eval_every=25, lr=lr)
t0 = time.time()
s, _ = run(cfg, out_path=str(out))
print(f"done {{out.name}}: acc={{s['final_eval_acc']:.3f}} "
      f"emergence={{s['emergence_step']}} ({{(time.time()-t0)/60:.1f}} min)")
"""


def cell_cmds():
    cmds = []
    for lr, lrtag in ((3e-4, '3em4'), (1.5e-4, '1p5em4')):
        for s in range(5):
            cmds.append((f'X1 gradual {lrtag} s{s}',
                         [PY, '-c', GRADUAL.format(lr=lr, lrtag=lrtag, seed=s)]))
    for s in range(30, 60):
        cmds.append((f'X3 g0.9 s{s}', [PY, '-c', ICRL_CUSTOM.format(
            gamma=0.9, steps=10000, mrps=32, seed=s,
            fname=f'gamma0p9_T40_s{s}.jsonl')]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par5b] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par5b] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par5b] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
