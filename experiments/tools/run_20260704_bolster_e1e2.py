#!/usr/bin/env python3
"""6-GPU-hour bolstering runs for the E1/E2 PeerJ submissions (2026-07-04).

Stage A: E2 gamma=0.9 seed extension s5..s19 (nomogram binomial validation).
Stage B: E1 boundary cells small/{low,med} U=2M (n=10) seeds s5..s14.
Stage C: E1 fine-grid decay shape med/med n in {6,8,12,16} x seeds s0..s2.

All stages reuse the existing runners/training code verbatim; every cell is
resume-aware (skips completed logs), so the script can be re-run safely.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path('/home/zeyufu/Desktop/dl-research')
EXP = ROOT / 'experiments'
PY = sys.executable

def sh(desc, args):
    t0 = time.time()
    r = subprocess.run(args, cwd=str(EXP))
    dt = (time.time() - t0) / 60
    print(f"[bolster] {desc}: rc={r.returncode} ({dt:.1f} min)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED at {desc}")

# ---------------- Stage A: E2 gamma ladder extension ----------------
def stage_a():
    code = (
        "import sys; sys.path.insert(0, 'icrl_td');\n"
        "import run_e2_gamma_ladder as G\n"
        "G.run_cells([(0.9, s) for s in range(5, 20)], 'gamma09ext')\n"
    )
    sh('stageA gamma0.9 s5..s19', [PY, '-c', code])

# ---------------- Stage B: E1 boundary cells ----------------
def stage_b():
    for ent in ('low', 'med'):
        for s in range(5, 15):
            sh(f'stageB small_{ent} U2M s{s}',
               [PY, 'repeated_data/run_e1_seed_audit_cell.py',
                '--entropy', ent, '--unique-tokens', '2000000', '--seed', str(s)])

# ---------------- Stage C: E1 med/med fine-grid ----------------
FINE_OUT = EXP / 'results' / 'repeated_data_finegrid'

def stage_c_cell(E, U, seed):
    name = f"med_med_U{U/1e6:g}M_E{E}_s{seed}".replace('.', 'p')
    out = FINE_OUT / f"{name}.jsonl"
    if out.exists():
        last = ''
        for line in out.open():
            if line.strip():
                last = line
        try:
            if '_summary' in json.loads(last):
                print(f"[bolster] skip {name} (complete)", flush=True)
                return
        except Exception:
            pass
        out.unlink()
    code = (
        "import sys; sys.path.insert(0, 'repeated_data')\n"
        "from train_repeat import Config, run\n"
        f"cfg = Config(capacity='med', entropy_level='med', generator='markov',\n"
        f"             unique_tokens={U}, n_epochs={E}, total_budget=20_000_000, seed={seed})\n"
        f"run(cfg, out_path={str(out)!r})\n"
    )
    FINE_OUT.mkdir(parents=True, exist_ok=True)
    sh(f'stageC {name}', [PY, '-c', code])

def stage_c():
    # E chosen so E * U == 20M budget exactly as in the main grid convention
    cells = [(6, 3_333_333), (8, 2_500_000), (12, 1_666_667), (16, 1_250_000)]
    for E, U in cells:
        for s in range(3):
            stage_c_cell(E, U, s)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if which in ('a', 'all'): stage_a()
    if which in ('b', 'all'): stage_b()
    if which in ('c', 'all'): stage_c()
    print(f"[bolster] TOTAL {(time.time()-t0)/60:.1f} min", flush=True)
