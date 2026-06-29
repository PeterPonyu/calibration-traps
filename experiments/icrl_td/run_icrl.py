"""Direction 016 grid driver.

  python run_icrl.py --smoke     # probes self-test + CPU mini-train, no writes
  python run_icrl.py --dry-run   # print the planned grid
  python run_icrl.py [--shard-index I --num-shards N]   # formal grid (executor)

Main grid (direction doc "Conditions"): optimizer {muon, adamw, sgdm} x
T {10, 20, 40} x seeds {0..4} = 45 cells. Resume-safe: cells whose jsonl
already ends in a _summary line are skipped.
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, "..")))

import train_icrl as TI  # noqa: E402
from runner_utils import add_shard_args, shard_cells, validate_shard_args  # noqa: E402

OPTIMIZERS = ("muon", "adamw", "sgdm")
T_VALUES = (10, 20, 40)
SEEDS = (0, 1, 2, 3, 4)


def make_cells():
    return [TI.Config(optimizer=o, T=t, seed=s)
            for o in OPTIMIZERS for t in T_VALUES for s in SEEDS]


def cell_done(path):
    if not os.path.exists(path):
        return False
    last = ""
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return "_summary" in json.loads(last)
    except json.JSONDecodeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        TI.run_smoke()
        return
    cells = make_cells()
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for c in cells:
            print(c.name())
        print(f"{len(cells)} cells (this shard)")
        return
    os.makedirs(TI.RESULTS_DIR, exist_ok=True)
    for i, cfg in enumerate(cells):
        path = os.path.join(TI.RESULTS_DIR, cfg.name() + ".jsonl")
        if cell_done(path):
            print(f"[{i + 1}/{len(cells)}] skip {cfg.name()}")
            continue
        summary, _ = TI.train(cfg, out_path=path)
        print(f"[{i + 1}/{len(cells)}] {cfg.name()}: "
              f"emergence={summary['emergence_step']} "
              f"acc={summary['final_val_acc']}")
    print("[icrl_td] DONE")


if __name__ == "__main__":
    main()
