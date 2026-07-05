#!/usr/bin/env python3
"""Batch 4 of the E1/E2 bolstering runs (2026-07-05 evening).

E1:
  B8: small/{low,med} at n=12,16 x s0-9 — completes the fine grid in the two
      boundary cells (incl. the one dissociating cell, small/low) so the
      decay-vs-memorization onset gap there is measured, not inferred.
  B9: fine grid in two MORE cells (small/high, large/low) at n in {6,8,12,16}
      x s0-2 — P4 shape universality beyond the two probed corners.
E2:
  A9 : gamma=0   steps=40000 (M=801) s5..s19 — late-emergence RATE at 4x budget.
  A10: gamma=0.9 steps=40000 (M=801) s0..s9  — third point of the live M-axis.
  A11: gamma=0.9 T in {10,20} s0..s9 — drift calibration across the horizon
       nuisance axis (per-T qbar must predict per-T crossing rates).
All cells resume-aware; distinct filenames never clobber existing logs.
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

ICRL_T = """
import json, sys, time
sys.path.insert(0, 'icrl_td')
import run_e2_gamma_ladder as G
from pathlib import Path
gamma, steps, T, seed, fname = {gamma}, {steps}, {T}, {seed}, {fname!r}
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
cfg = TI.Config(optimizer='adamw', seed=seed, n_states=G.N_STATES, T=T,
                steps=steps, eval_every=50, eval_mrps=32,
                acc_thresh=0.8, weight_decay=0.01, lr=1e-3)
t0 = time.time()
summary, history = TI.train(cfg, out_path=str(out))
accs = [r['val_acc'] for r in history]
rec = {{'_gamma_summary': {{
    'gamma': gamma, 'seed': seed, 'steps': steps, 'T': T,
    'final_val_acc': summary['final_val_acc'],
    'max_val_acc': max(accs) if accs else None,
    'single_cross_0p8': bool(any(a >= 0.8 for a in accs)),
    'sustain2_0p8': bool(any(a >= 0.8 and b >= 0.8 for a, b in zip(accs, accs[1:]))),
    'emergence_step': summary['emergence_step'],
    'elapsed_min': round((time.time() - t0) / 60, 2)}}}}
with open(out, 'a') as fh:
    fh.write(json.dumps(rec) + '\\n')
print(f"[T-axis] {{out.name}}: max={{rec['_gamma_summary']['max_val_acc']:.3f}} "
      f"single={{rec['_gamma_summary']['single_cross_0p8']}} "
      f"sus2={{rec['_gamma_summary']['sustain2_0p8']}}")
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
    return (name, [PY, '-c', code])


def cell_cmds():
    cmds = []
    # B8: boundary-cell fine-grid completion via the seed-audit runner (small cap)
    for ent in ('low', 'med'):
        for U in (1_666_667, 1_250_000):  # E=12, E=16
            for s in range(10):
                cmds.append((f'B8 small_{ent} U{U} s{s}',
                             [PY, 'repeated_data/run_e1_seed_audit_cell.py',
                              '--entropy', ent, '--unique-tokens', str(U),
                              '--seed', str(s)]))
    # B9: two more fine-grid cells
    for cap, ent in (('small', 'high'), ('large', 'low')):
        for E, U in [(6, 3_333_333), (8, 2_500_000), (12, 1_666_667), (16, 1_250_000)]:
            for s in range(3):
                d, args = finegrid_cmd(cap, ent, E, U, s)
                cmds.append((f'B9 {d}', args))
    # A9: gamma=0 late-emergence rate at 4x budget
    for s in range(5, 20):
        cmds.append((f'A9 g0 M801 s{s}', [PY, '-c', ICRL_T.format(
            gamma=0.0, steps=40000, T=40, seed=s,
            fname=f'gamma0p0_M801_T40_s{s}.jsonl')]))
    # A10: gamma=0.9 M=801 (live M-axis third point)
    for s in range(10):
        cmds.append((f'A10 g0.9 M801 s{s}', [PY, '-c', ICRL_T.format(
            gamma=0.9, steps=40000, T=40, seed=s,
            fname=f'gamma0p9_M801_T40_s{s}.jsonl')]))
    # A11: T-axis at gamma=0.9
    for T in (10, 20):
        for s in range(10):
            cmds.append((f'A11 g0.9 T{T} s{s}', [PY, '-c', ICRL_T.format(
                gamma=0.9, steps=10000, T=T, seed=s,
                fname=f'gamma0p9_Tax{T}_s{s}.jsonl')]))
    return cmds


def run_cell(item):
    desc, args = item
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    print(f"[par4] {desc}: rc={r.returncode} ({dt:.1f} min) {tail[0][:120]}", flush=True)
    if r.returncode != 0:
        print((r.stderr or '')[-800:], flush=True)
    return (desc, r.returncode)


if __name__ == '__main__':
    cmds = cell_cmds()
    print(f"[par4] {len(cmds)} cells, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(run_cell, cmds))
    bad = [d for d, rc in results if rc != 0]
    print(f"[par4] TOTAL {(time.time()-t0)/60:.1f} min; failures: {len(bad)} {bad}", flush=True)
