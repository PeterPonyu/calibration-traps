#!/usr/bin/env python3
"""E2-B4 -- state-matched (discount x budget) de-confound of the E2 budget grid.

THE CONFOUND THIS ARM EXISTS TO REMOVE.
E2 reads a "difficulty-by-budget interaction" (and concludes the budget lever
is exhausted at this scale) off two arms that differ in THREE ways, not one:

  gamma=0.0  results/icrl_td_gamma_ladder/       n_states=6   4-5/20 emerge
             steps=40000, eval_every=50  (M=801), eval_mrps=32,  thresh 0.8
  gamma=0.5  results/icrl_td_positive_regime/    n_states=10  0/60 emerge
             steps={40000,80000}, eval_every={200,400} (M=201),
             eval_mrps=128, thresh 0.7

The state count moved with the discount (verified in the `_meta` of both
directories: run_e2_gamma_ladder.py line 49 pins N_STATES=6 and passes it
explicitly through make_cfg, while the budget grid left train_icrl.Config's
module default n_states=10 untouched), and the READOUT moved too. So the
gamma=0.5 rung OF THE ACTUAL LADDER was never run at elevated budget: any
"harder rung does not respond to budget" reading currently has three
candidate causes and cannot separate them.

WHAT THIS RUNNER DOES.
Completes the 2x2 (gamma x budget) at FIXED state count and FIXED readout:
n_states=6 (the ladder's own), M=201 checkpoints and N=128 de-noised eval
(the budget grid's own), at both elevated budgets. gamma=0.0 is the positive
control -- the rung that produced emergences at standard budget -- carried
through the same readout, so the gamma=0.5 cells are compared against a cell
that is known to be able to fire under exactly this instrument.

  gamma in {0.0, 0.5} x steps in {40000, 80000} x T in {10, 20, 40}
    x optimizer in {adamw, muon} x seed in {200..204}   = 120 cells

gamma lives in data.GAMMA, not in train_icrl.Config (whose __init__ asserts
hasattr for every kwarg), so it is patched module-side before train_icrl and
probes are imported and re-patched per cell -- the mechanism
run_e2_gamma_ladder.py established and run_20260708_positive_regime.py
reused. Each cell records its own gamma in a trailing `_e2b4_meta` jsonl line
appended after TI.train() has written `_summary` and closed the file; that
line is also the completion sentinel, so a cell is resumed only when the
whole cell (train + annotation) finished.

The constant-predictor floor for each configuration comes from the existing
modal_baseline.py (large-sample bucket marginals); its unit tests run inside
--smoke rather than being duplicated here.

Modes:
  --smoke     : CPU, few minutes, writes only to a temp dir -- modal-baseline
                unit tests, gamma patch at both rungs, detector self-test,
                a tiny two-gamma end-to-end train + sentinel + resume-skip,
                and the analyzer read over what it just produced
  --dry-run   : full cell list, run count, GPU-hour estimate; no training
  (default)   : run the grid; resume-safe; writes revision2026/gpu2026/e2b4/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS = ROOT / "experiments" / "icrl_td"


def _early_gpu_pin(argv):
    """CUDA_VISIBLE_DEVICES must be set before torch is imported."""
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
        elif a.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]


_early_gpu_pin(sys.argv)

if str(THIS) in sys.path:
    sys.path.remove(str(THIS))
sys.path.insert(0, str(THIS))
sys.path.append(str(ROOT / "experiments"))

import data as D  # noqa: E402  (patch BEFORE importing train_icrl/probes)

N_STATES = 6           # the ladder's own state count (run_e2_gamma_ladder.py)
GAMMAS = (0.0, 0.5)
STEPS_VALUES = (40000, 80000)
T_VALUES = (10, 20, 40)
OPTIMIZERS = ("adamw", "muon")
SEEDS = (200, 201, 202, 203, 204)
EVAL_MRPS = 128                       # budget grid's N=128 de-noised readout
EVAL_EVERY = {40000: 200, 80000: 400}  # M=201 checkpoints at either budget
ACC_THRESH = 0.7                      # budget grid's threshold; 0.8 read post-hoc

# Measured wall-clock from the n_states=6 ladder's own `_gamma_summary`
# elapsed_min (results/icrl_td_gamma_ladder), min per 1000 steps by horizon.
# Eval cost is matched by construction: M=201 x N=128 = 25728 MRP-evals here
# vs M=801 x N=32 = 25632 there, so those timings transfer directly.
RATE_MIN_PER_KSTEP = {10: 0.19, 20: 0.25, 40: 0.38}


def set_gamma(g: float):
    D.GAMMA = g
    D.V_MAX = 1.0 if g == 0.0 else 1.0 / (1.0 - g)
    # probes.py binds data.GAMMA at ITS import; re-bind defaults if loaded.
    if "probes" in sys.modules:
        import probes as PR
        PR.GAMMA = g
        PR.td_trace.__defaults__ = (0.2, g)
        PR.bellman_residual.__defaults__ = (g,)


set_gamma(GAMMAS[0])

import train_icrl as TI  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args, shard_cells, validate_shard_args)

OUT = ROOT / "experiments" / "revision2026" / "gpu2026" / "e2b4"


def gtag(gamma):
    return "g" + str(gamma).replace(".", "p")


def cell_name(cell):
    return (f"{gtag(cell['gamma'])}_n{N_STATES}_{cell['optimizer']}"
            f"_T{cell['T']}_s{cell['seed']}_steps{cell['steps']}")


def make_cells(gammas=GAMMAS, steps_values=STEPS_VALUES, t_values=T_VALUES,
               optimizers=OPTIMIZERS, seeds=SEEDS):
    return [{"gamma": g, "steps": st, "T": t, "optimizer": o, "seed": s}
            for g in gammas for st in steps_values for t in t_values
            for o in optimizers for s in seeds]


def make_cfg(cell):
    return TI.Config(optimizer=cell["optimizer"], T=cell["T"],
                     seed=cell["seed"], steps=cell["steps"],
                     n_states=N_STATES, eval_every=EVAL_EVERY[cell["steps"]],
                     eval_mrps=EVAL_MRPS, acc_thresh=ACC_THRESH)


def cell_path(cell):
    return OUT / f"{cell_name(cell)}.jsonl"


def cell_done(path: Path):
    """Complete iff the trailing line is this arm's `_e2b4_meta` sentinel,
    which is appended only after TI.train() wrote `_summary` and closed the
    file -- so the sentinel implies a `_summary` and a finished annotation."""
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
    return isinstance(rec, dict) and "_e2b4_meta" in rec


def cell_minutes(cell):
    return cell["steps"] / 1000.0 * RATE_MIN_PER_KSTEP[cell["T"]]


def write_manifest(out_dir: Path):
    """Full planned grid (never shard-filtered), so concurrent shards write
    identical content."""
    cells = make_cells()
    manifest = {
        "arm": "E2-B4",
        "purpose": "state-matched gamma x budget de-confound of the E2 "
                   "budget grid (gamma and n_states moved together)",
        "n_states": N_STATES,
        "gammas": list(GAMMAS),
        "steps_values": list(STEPS_VALUES),
        "t_values": list(T_VALUES),
        "optimizers": list(OPTIMIZERS),
        "seeds": list(SEEDS),
        "eval_mrps": EVAL_MRPS,
        "eval_every": {str(k): v for k, v in EVAL_EVERY.items()},
        "checkpoints_per_run": {str(k): k // EVAL_EVERY[k] + 1
                                for k in STEPS_VALUES},
        "acc_thresh_logged": ACC_THRESH,
        "acc_thresh_reported": [0.7, 0.8],
        "seed_block_rationale":
            "200-204 is disjoint from every prior icrl_td arm (gamma ladder "
            "0-59, canonical grid and positive regime 0-4, ultragoal denoised "
            "5-7)",
        "n_runs": len(cells),
        "est_gpu_hours": round(sum(cell_minutes(c) for c in cells) / 60.0, 2),
        "runs": [{"name": cell_name(c), **c,
                  "eval_every": EVAL_EVERY[c["steps"]],
                  "eval_mrps": EVAL_MRPS, "n_states": N_STATES,
                  "est_minutes": round(cell_minutes(c), 1)} for c in cells],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def run_cell(cell, out_dir: Path):
    path = out_dir / f"{cell_name(cell)}.jsonl"
    set_gamma(cell["gamma"])
    cfg = make_cfg(cell)
    t0 = time.time()
    summary, history = TI.train(cfg, out_path=str(path), log=lambda *a: None)
    accs = [r["val_acc"] for r in history]
    with open(path, "a") as fh:
        fh.write(json.dumps({"_e2b4_meta": {
            "arm": "E2-B4", "gamma": cell["gamma"], "v_max": D.V_MAX,
            "n_states": N_STATES, "cell": cell_name(cell),
            "steps": cfg.steps, "eval_every": cfg.eval_every,
            "eval_mrps": cfg.eval_mrps, "n_evals": len(accs),
            "max_val_acc": max(accs) if accs else None,
            "final_val_acc": summary["final_val_acc"],
            "emergence_step": summary["emergence_step"],
            "elapsed_min": round((time.time() - t0) / 60, 2)}}) + "\n")
    return summary, accs


def _pin_cpu_threads():
    """Muon's Newton-Schulz runs in bfloat16 (grokking/muon.py, closed infra).
    On CPU that kernel is thread-thrash bound: measured 700 ms/call at 24
    threads vs 0.9 ms/call at 1 on the 32x96..128x32 matrices this smoke
    builds. Single-threading is a smoke-only concession -- GPU runs never
    reach here."""
    import torch
    torch.set_num_threads(1)


def run_smoke():
    """CPU end-to-end over every path, into a temp dir; nothing persisted."""
    import shutil
    import tempfile

    import modal_baseline as MB
    import probes as PR

    _pin_cpu_threads()
    exercised = []

    MB.run_tests(n_mrps=100_000)
    b0, _ = MB.modal_baseline(N_STATES, 0.0, n_mrps=100_000)
    b5, _ = MB.modal_baseline(N_STATES, 0.5, n_mrps=100_000)
    exercised.append(f"modal baseline unit tests + n={N_STATES} floors "
                     f"gamma=0 {b0:.4f} / gamma=0.5 {b5:.4f}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_e2b4 as AN
    AN.detector_self_test()
    exercised.append("detector self-test (single-crossing + sustained-K)")

    for g in GAMMAS:
        set_gamma(g)
        assert D.GAMMA == g and D.V_MAX == (1.0 if g == 0 else 1 / (1 - g))
        assert PR.GAMMA == g and PR.bellman_residual.__defaults__ == (g,)
        assert PR.self_test(), f"probe self-test failed at gamma={g}"
    exercised.append(f"gamma patch + probe self-test at {list(GAMMAS)}")

    tmp = Path(tempfile.mkdtemp(prefix="e2b4_smoke_"))
    try:
        man = write_manifest(tmp)
        assert man["n_runs"] == 120, man["n_runs"]
        assert (tmp / "MANIFEST.json").exists()
        exercised.append(f"MANIFEST.json ({man['n_runs']} runs, "
                         f"{man['est_gpu_hours']} GPU-h)")

        tiny = []
        for g in GAMMAS:
            cell = {"gamma": g, "steps": 60, "T": 8, "optimizer": "muon",
                    "seed": 200}
            set_gamma(g)
            cfg = TI.Config(optimizer="muon", T=8, seed=200, steps=60,
                            n_states=N_STATES, d_model=32, n_heads=2,
                            n_layers=2, batch=16, eval_every=30, eval_mrps=8,
                            acc_thresh=ACC_THRESH, device="cpu")
            path = tmp / f"{cell_name(cell)}.jsonl"
            summary, history = TI.train(cfg, out_path=str(path),
                                        log=lambda *a: None)
            accs = [r["val_acc"] for r in history]
            assert len(history) >= 2 and all(
                isinstance(a, float) for a in accs)
            with open(path, "a") as fh:
                fh.write(json.dumps({"_e2b4_meta": {
                    "arm": "E2-B4", "gamma": g, "v_max": D.V_MAX,
                    "n_states": N_STATES, "cell": cell_name(cell),
                    "steps": cfg.steps, "eval_every": cfg.eval_every,
                    "eval_mrps": cfg.eval_mrps, "n_evals": len(accs),
                    "max_val_acc": max(accs),
                    "final_val_acc": summary["final_val_acc"],
                    "emergence_step": summary["emergence_step"],
                    "elapsed_min": 0.0}}) + "\n")
            assert cell_done(path), "sentinel written but cell_done() False"
            tiny.append(path)
        exercised.append(f"two-gamma tiny train -> jsonl + sentinel "
                         f"({len(tiny)} cells)")

        partial = tmp / "partial.jsonl"
        partial.write_text(json.dumps({"_meta": {}}) + "\n"
                           + json.dumps({"_summary": {}}) + "\n")
        assert not cell_done(partial), \
            "a cell with _summary but no sentinel must NOT be skipped"
        assert cell_done(tiny[0]), "completed cell must be skipped on resume"
        exercised.append("resume gate (sentinel required, bare _summary "
                         "re-runs)")

        rows = AN.load_runs(tmp)
        assert len(rows) == len(tiny), f"analyzer saw {len(rows)} of {len(tiny)}"
        rep = AN.summarize(rows, n_mrps=20_000)
        assert rep["n_runs"] == len(tiny)
        for cellrep in rep["cells"]:
            for key in ("cross_0.7", "sustain2_0.7", "sustain3_0.8",
                        "modal_baseline", "median_best_acc"):
                assert key in cellrep, key
        exercised.append(f"analyze_e2b4 read-back ({rep['n_runs']} runs, "
                         f"{len(rep['cells'])} cells, all readouts present)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("SMOKE PASS: " + "; ".join(exercised) + "; zero writes outside tmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter on the cell name")
    ap.add_argument("--gpu", default=None,
                    help="CUDA device id (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--gammas", default=None,
                    help="comma list, subset of 0.0,0.5")
    ap.add_argument("--steps", default=None,
                    help="comma list, subset of 40000,80000")
    ap.add_argument("--horizons", default=None,
                    help="comma list, subset of 10,20,40")
    ap.add_argument("--optimizers", default=None,
                    help="comma list, subset of adamw,muon")
    add_shard_args(ap)
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    def pick(raw, default, cast):
        return default if raw is None else tuple(
            cast(x) for x in raw.split(",") if x.strip())

    cells = make_cells(
        gammas=pick(args.gammas, GAMMAS, float),
        steps_values=pick(args.steps, STEPS_VALUES, int),
        t_values=pick(args.horizons, T_VALUES, int),
        optimizers=pick(args.optimizers, OPTIMIZERS, str))
    if args.only:
        cells = [c for c in cells if args.only in cell_name(c)]
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)

    if args.dry_run:
        for c in cells:
            print(f"{cell_name(c):<44} eval_every={EVAL_EVERY[c['steps']]:>3} "
                  f"N={EVAL_MRPS} ~{cell_minutes(c):.1f} min")
        mins = sum(cell_minutes(c) for c in cells)
        print(f"\n{len(cells)} runs (this selection) | "
              f"{mins / 60:.1f} GPU-hours estimated | "
              f"n_states={N_STATES} fixed, M=201 checkpoints, N={EVAL_MRPS}")
        full = make_cells()
        print(f"full grid: {len(full)} runs, "
              f"{sum(cell_minutes(c) for c in full) / 60:.1f} GPU-hours; "
              f"40000-step half alone: "
              f"{sum(cell_minutes(c) for c in full if c['steps'] == 40000) / 60:.1f}"
              f" GPU-hours")
        return

    write_manifest(OUT)
    for i, cell in enumerate(cells, 1):
        path = cell_path(cell)
        if cell_done(path):
            print(f"[{i}/{len(cells)}] skip {path.name}", flush=True)
            continue
        summary, accs = run_cell(cell, OUT)
        print(f"[{i}/{len(cells)}] {path.name}: "
              f"best={max(accs) if accs else float('nan'):.3f} "
              f"final={summary['final_val_acc']:.3f} "
              f"emergence={summary['emergence_step']}", flush=True)
    print("[e2b4] DONE", flush=True)


if __name__ == "__main__":
    main()
