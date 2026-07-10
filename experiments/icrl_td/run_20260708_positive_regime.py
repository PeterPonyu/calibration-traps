#!/usr/bin/env python3
"""E2 positive-regime frontier probe -- budget x horizon x optimizer grid at
a mid-difficulty discount rung (gamma=0.5), read out with the paper's own
de-noised evaluator.

Rationale (paper E2, "Stress tests along the checkpoint and eval-size axes",
tmlr-E2/main.tex sec:mnaxes): at four times the canonical grid's budget
(steps=40000, M=801 checkpoints), the discount ladder's EASIEST rung
(gamma=0, pure in-context lookup) produces genuine late TD emergences in
4/20 seeds (onsets ~32000-38500, sustained 0.84-1.0 to the end). That result
pins the missing ingredient at the easiest rung to budget, not task
difficulty. This runner asks whether the same budget increase -- and a
further doubling -- unlocks genuine emergence at a HARDER, mid-difficulty
rung: gamma=0.5, strictly between the ladder's solved lookup floor (0.0) and
the canonical unsolved ceiling (0.9) -- across the full T in {10,20,40}
horizon axis and both optimizers used elsewhere in this direction. Uses the
paper's N=128 de-noised readout (run_e2_denoised.py / run_e2_denoised_cell.py
convention) so any emergence found here is held to the same evidentiary
standard as that rescue.

The gamma patch must land BEFORE train_icrl/probes are imported (probes
binds data.GAMMA into default args at import time; see
run_e2_gamma_ladder.py, whose set_gamma() this mirrors).

Grid: steps in {40000, 80000} x T in {10, 20, 40} x optimizer in
{adamw, muon} x seed in {0..4} = 60 cells. gamma is fixed at 0.5 for the
whole grid via the SAME mechanism as run_e2_gamma_ladder.py: patch
data.GAMMA (and data.V_MAX) before train_icrl/probes are imported --
train_icrl.Config is kwargs-validated (its __init__ asserts hasattr(self, k)
for every kwarg, so it accepts no unknown keys and gamma is not one of its
fields) and is never touched. Each cell's gamma is instead recorded by
appending one extra jsonl line (`_gamma_meta`) after TI.train() closes the
file, mirroring how run_e2_gamma_ladder.py appends its own `_gamma_summary`
line post-hoc. Everything else (n_states, d_model, batch, lr, weight_decay,
muon_lr, acc_thresh) is left at train_icrl.py's defaults, matching
run_e2_denoised.py's convention
of touching only the axes under study. muon needs no monkeypatch here:
train_icrl.Config(optimizer="muon") already wires the 001-standard hybrid
Muon/AdamW split natively (train_icrl.build_optimizer), so this direction
has its own muon precedent and induction_subspace's mechanism is not needed.

Modes:
  --smoke     : tiny CPU cell, wiring self-test at gamma=0.5, no writes
  --dry-run   : print the planned cell list (this shard) + count
  (default)   : run the formal grid; resume-safe on a per-cell jsonl
                _gamma_meta sentinel; writes results/icrl_td_positive_regime/
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS = ROOT / "experiments" / "icrl_td"
if str(THIS) in sys.path:
    sys.path.remove(str(THIS))
sys.path.insert(0, str(THIS))
sys.path.append(str(ROOT / "experiments"))

import data as D  # noqa: E402  (patch BEFORE importing train_icrl/probes)

GAMMA = 0.5  # ladder's mid rung: strictly between the lookup floor (0.0)
             # and the canonical unsolved ceiling (0.9); run_e2_gamma_ladder.py


def set_gamma(g: float):
    D.GAMMA = g
    D.V_MAX = 1.0 if g == 0.0 else 1.0 / (1.0 - g)
    # probes.py binds data.GAMMA at ITS import; re-bind defaults if loaded.
    if "probes" in sys.modules:
        import probes as PR
        PR.GAMMA = g
        PR.td_trace.__defaults__ = (0.2, g)
        PR.bellman_residual.__defaults__ = (g,)


set_gamma(GAMMA)

import train_icrl as TI  # noqa: E402
from runner_utils import add_shard_args, shard_cells, validate_shard_args  # noqa: E402

OUT = ROOT / "experiments" / "results" / "icrl_td_positive_regime"

STEPS_VALUES = (40000, 80000)
T_VALUES = (10, 20, 40)
OPTIMIZERS = ("adamw", "muon")
SEEDS = (0, 1, 2, 3, 4)
EVAL_MRPS = 128  # paper's N=128 de-noised rerun convention (run_e2_denoised.py)
EVAL_EVERY = {40000: 200, 80000: 400}  # ~200 checkpoints either budget


def make_cfg(steps, t, opt, seed):
    return TI.Config(optimizer=opt, T=t, seed=seed, steps=steps,
                     eval_every=EVAL_EVERY[steps], eval_mrps=EVAL_MRPS)


def make_cells():
    return [make_cfg(steps, t, opt, seed)
            for steps in STEPS_VALUES for t in T_VALUES
            for opt in OPTIMIZERS for seed in SEEDS]


def cell_path(cfg):
    return OUT / f"{cfg.name()}_g0p5_steps{cfg.steps}.jsonl"


def cell_done(path: Path):
    """Complete iff the trailing line is this script's own `_gamma_meta`
    sentinel, appended only after TI.train() has already written its
    `_summary` line and closed the file (see the main loop below)."""
    if not path.exists():
        return False
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            last = line
    if last is None:
        return False
    try:
        rec = json.loads(last)
    except json.JSONDecodeError:
        return False
    return isinstance(rec, dict) and "_gamma_meta" in rec


def run_smoke():
    """<60s CPU wiring check at gamma=0.5: probe self-test + tiny mini-train.
    Writes nothing."""
    import probes as PR
    assert PR.self_test(), "probe self-test failed"
    assert D.GAMMA == GAMMA, "gamma patch did not land before train_icrl import"
    cfg = TI.Config(n_states=6, T=8, d_model=32, n_heads=2, n_layers=2,
                    batch=16, steps=60, eval_every=30, eval_mrps=8,
                    device="cpu", optimizer="muon")
    summary, history = TI.train(cfg, out_path=None, log=lambda *a: None)
    assert len(history) >= 2, "eval pipeline produced no records"
    for rec in history:
        assert math.isfinite(rec["train_loss"])
        assert math.isfinite(rec["val_acc"])
    print(f"SMOKE PASS: gamma={D.GAMMA} (V_MAX={D.V_MAX}) probes self-test OK; "
          f"mini-train {cfg.steps} steps (muon hybrid, {summary['n_params']} "
          f"params) finite; zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    add_shard_args(ap)
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    cells = make_cells()
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)

    if args.dry_run:
        for cfg in cells:
            print(cell_path(cfg).stem)
        print(f"{len(cells)} cells (this shard)")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(cells, 1):
        path = cell_path(cfg)
        if cell_done(path):
            print(f"[{i}/{len(cells)}] skip {path.name}", flush=True)
            continue
        t0 = time.time()
        summary, _ = TI.train(cfg, out_path=str(path), log=lambda *a: None)
        # gamma lives outside Config (data.GAMMA), so record it post-hoc as
        # its own jsonl line -- same pattern as run_e2_gamma_ladder.py's
        # appended `_gamma_summary` line.
        with open(path, "a") as fh:
            fh.write(json.dumps({"_gamma_meta": {
                "gamma": GAMMA, "v_max": D.V_MAX, "cell": cfg.name(),
                "steps": cfg.steps}}) + "\n")
        print(f"[{i}/{len(cells)}] {path.name}: "
              f"emergence={summary['emergence_step']} "
              f"final={summary['final_val_acc']} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("[icrl_td_positive_regime] DONE")


if __name__ == "__main__":
    main()
