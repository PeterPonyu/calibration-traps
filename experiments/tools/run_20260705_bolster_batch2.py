#!/usr/bin/env python3
"""Batch 2 of the E1/E2 bolstering runs (2026-07-05). See batch 1:
run_20260704_bolster_parallel.py. All cells independent + resume-aware.

E2: gamma in {0.5,0.7} x s0-9 (ladder curve); gamma=0 M=401 x s0-9 (M-axis of
the FP formula); gamma=0 eval_mrps in {16,64} x s0-4 (N-axis / q validation);
gamma=0.9 s20-29 (binomial power).
E1: small/{low,med} U=1M (n=20) s5-14 (dissociation cell + corner edge);
realtext med_b50M s2-4 (bridge 2->5 seeds); large/high fine grid
E in {6,8,12,16} x s0-2 (decay-shape universality).
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

# custom icrl_td cell: trains with overrides, writes run_cells-compatible
# _gamma_summary line under a DISTINCT filename (never clobbers ladder files)
ICRL_CUSTOM = """
import json, sys, time
sys.path.insert(0, 'icrl_td')
import run_e2_gamma_ladder as G
from pathlib import Path
gamma, steps, mrps, seed, fname = {gamma}, {steps}, {mrps}, {seed}, {fname!r}
out = G.OUT / fname
if out.exists():
    last = None
    for line in out.read_text().splitlines():
        if line.strip(): last = json.loads(line)
    if isinstance(last, dict) and '_gamma_summary' in last:
        print(f'skip {{out.name}} (complete)'); raise SystemExit(0)
    out.unlink()
G.set_gamma(gamma)
import train_icrl as TI
cfg = TI.Config(optimizer='adamw', seed=seed, n_states=G.N_STATES, T=G.T,
                steps=steps, eval_every=50, eval_mrps=mrps,
                acc_thresh=0.8, weight_decay=0.01, lr=1e-3)
t0 = time.time()
summary, history = TI.train(cfg, out_path=str(out))
accs = [r['val_acc'] for r in history]
rec = {{'_gamma_summary': {{
    'gamma': gamma, 'seed': seed, 'steps': steps, 'eval_mrps': mrps,
    'final_val_acc': summary['final_val_acc'],
    'max_val_acc': max(accs) if accs else None,
    'single_cross_0p8': bool(any(a >= 0.8 for a in accs)),
    'sustain2_0p8': bool(any(a >= 0.8 and b >= 0.8 for a, b in zip(accs, accs[1:]))),
    'emergence_step': summary['emergence_step'],
    'elapsed_min': round((time.time() - t0) / 60, 2)}}}}
with open(out, 'a') as fh:
    fh.write(json.dumps(rec) + '\\n')
print(f"[custom] {{out.name}}: max={{rec['_gamma_summary']['max_val_acc']:.3f}} "
      f"single={{rec['_gamma_summary']['single_cross_0p8']}} "
      f"sus2={{rec['_gamma_summary']['sustain2_0p8']}}")
"""


def cell_cmds():
    cmds = []
    # E2 ladder rungs gamma 0.5 / 0.7 (standard path + naming)
    for g in (0.5, 0.7):
        for s in range(10):
            code = ("import sys; sys.path.insert(0, 'icrl_td')\n"
                    "import run_e2_gamma_ladder as G\n"
                    f"G.run_cells([({g}, {s})], 'ladderfill')\n")
            cmds.append((f'A2 gamma{g} s{s}', [PY, '-c', code]))
    # E2 M-axis: gamma=0, steps=20000 -> M=401
    for s in range(10):
        cmds.append((f'A3 g0 M401 s{s}', [PY, '-c', ICRL_CUSTOM.format(
            gamma=0.0, steps=20000, mrps=32, seed=s,
            fname=f'gamma0p0_M401_T40_s{s}.jsonl')]))
    # E2 N-axis: gamma=0, eval_mrps 16/64
    for mrps in (16, 64):
        for s in range(5):
            cmds.append((f'A5 g0 N{mrps} s{s}', [PY, '-c', ICRL_CUSTOM.format(
                gamma=0.0, steps=10000, mrps=mrps, seed=s,
                fname=f'gamma0p0_N{mrps}_T40_s{s}.jsonl')]))
    # E2 gamma=0.9 s20-29
    for s in range(20, 30):
        code = ("import sys; sys.path.insert(0, 'icrl_td')\n"
                "import run_e2_gamma_ladder as G\n"
                f"G.run_cells([(0.9, {s})], 'gamma09ext2')\n")
        cmds.append((f'A4 gamma0.9 s{s}', [PY, '-c', code]))
    # E1 upper-edge / dissociation cells: U=1M (n=20), s5-14
    for ent in ('low', 'med'):
        for s in range(5, 15):
            cmds.append((f'B1 small_{ent} U1M s{s}',
                         [PY, 'repeated_data/run_e1_seed_audit_cell.py',
                          '--entropy', ent, '--unique-tokens', '1000000',
                          '--seed', str(s)]))
    # E1 realtext bridge s2-4 (runner is idempotent per seed-file? it writes
    # <cap>_b50M_s<seed>.json; skip if exists)
    for s in (2, 3, 4):
        out = EXP / 'results' / 'repeated_data_realtext' / f'med_b50M_s{s}.json'
        if out.exists():
            continue
        cmds.append((f'B2 realtext med b50M s{s}',
                     [PY, 'repeated_data/run_20260619_e1_realtext.py',
                      '--budget', '50000000', '--capacity', 'med',
                      '--seeds', str(s)]))
    # E1 large/high fine grid
    for E, U in [(6, 3_333_333), (8, 2_500_000), (12, 1_666_667), (16, 1_250_000)]:
        for s in range(3):
            name = f"large_high_U{U/1e6:g}M_E{E}_s{s}".replace('.', 'p')
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
                f"cfg = Config(capacity='large', entropy_level='high', generator='markov',\n"
                f"             unique_tokens={U}, n_epochs={E}, total_budget=20_000_000, seed={s})\n"
                "run(cfg, out_path=str(out))\n")
            cmds.append((f'B3 {name}', [PY, '-c', code]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par2] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par2] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par2] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
