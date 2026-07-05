#!/usr/bin/env python3
"""Batch 3 of the E1/E2 bolstering runs (2026-07-05).

B4: small/{low,med} at n=6,8 x s0-9 — locate the small-cell decay onsets that
    batch-1 showed sit AT the 0.05 threshold at n=10.
B5: med/med fine grid +s3,s4 (n=5 per point).
B7: large/high fine grid +s3,s4 (n=5 per point).
B6: code-bytes real-text bridge (CPython stdlib + site-packages corpus,
    code_bytes.pt) med/50M x s0-4 — domain-generality of R_free on real
    low-entropy text; matches the WikiText arm's n-list exactly.
A6: E2 M-axis third point: gamma=0 pure noise, steps=40000 (M=801) x s0-4.
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

sys.path.insert(0, str(EXP / 'tools'))
from run_20260705_bolster_batch2 import ICRL_CUSTOM  # noqa: E402

CODE_BRIDGE = """
import json, sys
sys.path.insert(0, 'repeated_data')
from pathlib import Path
import run_repeat_realtext as R
import torch
seed = {seed}
OUT = Path('results/repeated_data_realtext_code')
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f'med_b50M_s{{seed}}.json'
if out.exists():
    print(f'skip {{out.name}}'); raise SystemExit(0)
R.BYTES_PATH = str(Path('repeated_data/code_bytes.pt').resolve())
R.N_LIST = [1, 2, 4, 10, 20, 40]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
v = R.run_sweep(50_000_000, 'med', 256, 32, seed, dev)
out.write_text(json.dumps(v, indent=2, default=str) + '\\n')
print(f'done {{out.name}} R_free={{v.get("R_free")}}')
"""


def finegrid_cmd(cap, ent, E, U, s):
    name = f"{cap}_{ent}_U{U/1e6:g}M_E{E}_s{s}".replace('.', 'p')
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
        f"cfg = Config(capacity='{cap}', entropy_level='{ent}', generator='markov',\n"
        f"             unique_tokens={U}, n_epochs={E}, total_budget=20_000_000, seed={s})\n"
        "run(cfg, out_path=str(out))\n")
    return (f'{name}', [PY, '-c', code])


def cell_cmds():
    cmds = []
    # B4: small-cell onset location at n=6,8 (runner computes E from U)
    for ent in ('low', 'med'):
        for U in (3_333_333, 2_500_000):
            for s in range(10):
                cmds.append((f'B4 small_{ent} U{U} s{s}',
                             [PY, 'repeated_data/run_e1_seed_audit_cell.py',
                              '--entropy', ent, '--unique-tokens', str(U),
                              '--seed', str(s)]))
    # B5 / B7: fine-grid seed extension to n=5
    for cap, ent in (('med', 'med'), ('large', 'high')):
        for E, U in [(6, 3_333_333), (8, 2_500_000), (12, 1_666_667), (16, 1_250_000)]:
            for s in (3, 4):
                d, args = finegrid_cmd(cap, ent, E, U, s)
                cmds.append((f'B57 {d}', args))
    # B6: code-bytes bridge
    for s in range(5):
        cmds.append((f'B6 code med b50M s{s}', [PY, '-c', CODE_BRIDGE.format(seed=s)]))
    # A6: M=801 pure-noise cells
    for s in range(5):
        cmds.append((f'A6 g0 M801 s{s}', [PY, '-c', ICRL_CUSTOM.format(
            gamma=0.0, steps=40000, mrps=32, seed=s,
            fname=f'gamma0p0_M801_T40_s{s}.jsonl')]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par3] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par3] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par3] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
