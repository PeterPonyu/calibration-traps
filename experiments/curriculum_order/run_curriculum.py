"""Direction 011 — curriculum-ordering grid runner (NOT executed yet).

Grid (120 cells):
    ordering  ∈ {iid, easy_to_hard, hard_first, structured}   (the curriculum arm)
    optimizer ∈ {adamw, sgdm}                                  (sgdm = weak probe)
    family    ∈ {walsh, modadd, copy}                          (3 task families)
    seed      ∈ 0..4
4 x 2 x 3 x 5 = 120 cells. Each cell trains a model under a CurriculumSchedule
(stage-based easy->hard dataset switching + the named within-stage ordering) and
evaluates ALWAYS on the FINAL/hard target. The headline contrast: does any
ordering arm let SGDM (the 004 failure case) reach the hard target that the
single-target run could not — i.e. does a DATA curriculum substitute for the
TARGET ladder 004 found load-bearing?

walsh is the PRIMARY family (Walsh pure-degree ladder, final = pure degree-3 —
the exact 004 weak-optimizer killer target). modadd is the mod-add Fourier
control; copy is the lookup/copy transfer family.

Output: ../../experiments/results/curriculum_order/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke   : delegate to train_curriculum smoke (no files, <60s) and exit 0.
--dry-run : print planned cells and exit 0 (launches NOTHING).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.append(_EXPERIMENTS_DIR)

from train_curriculum import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


ORDERINGS = ["iid", "easy_to_hard", "hard_first", "structured"]
OPTIMIZERS = ["adamw", "sgdm"]
FAMILIES = ["walsh", "modadd", "copy"]
SEEDS = list(range(5))            # 0-4
STEPS_PER_STAGE = 800
EVAL_EVERY = 50

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "curriculum_order")


def _build_cells(orderings=ORDERINGS, optimizers=OPTIMIZERS,
                 families=FAMILIES, seeds=SEEDS):
    return [
        (ordering, opt, family, seed)
        for ordering in orderings
        for opt in optimizers
        for family in families
        for seed in seeds
    ]


def already_done(path: str) -> bool:
    """True iff the jsonl exists and ends with a _summary line."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return False
        fh.seek(max(0, size - 4096))
        tail = fh.read().decode("utf-8", errors="replace")
    return '"_summary"' in tail


def main():
    ap = argparse.ArgumentParser(description="curriculum-ordering grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    # Recalibration / pilot knobs (defaults reproduce the main grid exactly).
    ap.add_argument("--steps-per-stage", type=int, default=STEPS_PER_STAGE,
                    help="Override per-stage step budget (walsh needs ~10x)")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                    help="Override eval cadence (use 10 to de-floor modadd)")
    ap.add_argument("--families", default=None,
                    help="Comma list subset of walsh,modadd,copy")
    ap.add_argument("--orderings", default=None,
                    help="Comma list subset of iid,easy_to_hard,hard_first,structured")
    ap.add_argument("--optimizers", default=None,
                    help="Comma list subset of adamw,sgdm")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Number of seeds (default 5)")
    ap.add_argument("--tag", default=None,
                    help="Write to results/curriculum_order_<tag>/ instead of "
                         "the main namespace (keeps pilots isolated)")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    orderings = args.orderings.split(",") if args.orderings else ORDERINGS
    optimizers = args.optimizers.split(",") if args.optimizers else OPTIMIZERS
    families = args.families.split(",") if args.families else FAMILIES
    seeds = list(range(args.seeds)) if args.seeds else SEEDS
    out = (OUT + "_" + args.tag) if args.tag else OUT

    all_cells = _build_cells(orderings, optimizers, families, seeds)
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        print(f"[curriculum_order] dry-run: {len(cells)} cells planned"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        for i, (ordering, opt, family, seed) in enumerate(cells):
            name = f"{family}_{ordering}_{opt}_s{seed}"
            print(f"  [{i+1:03d}/{len(cells)}] {name}  "
                  f"steps_per_stage={args.steps_per_stage}")
        sys.exit(0)

    # --- real training path (only when neither flag set) ---
    os.makedirs(out, exist_ok=True)
    print(f"[curriculum_order] {len(cells)} cells -> {out}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells))
          + f" | steps_per_stage={args.steps_per_stage} eval_every={args.eval_every}",
          flush=True)

    for i, (ordering, opt, family, seed) in enumerate(cells):
        name = f"{family}_{ordering}_{opt}_s{seed}"
        path = os.path.join(out, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(
            family=family,
            ordering=ordering,
            optimizer=opt,
            seed=seed,
            steps_per_stage=args.steps_per_stage,
            eval_every=args.eval_every,
        )
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"acc={s['final_eval_acc']:.3f} "
            f"emergence={s['emergence_step']} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[curriculum_order] DONE", flush=True)


if __name__ == "__main__":
    main()
