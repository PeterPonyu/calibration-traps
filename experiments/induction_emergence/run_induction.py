"""Direction 007 — induction/ICL emergence grid runner (NOT executed yet).

Main grid (90 cells):
    optimizer ∈ {muon, adamw, sgdm}
    seq_len   ∈ {64, 128, 256}
    seed      ∈ 0..9                 (10 seeds for cross-seed variance)
3 x 3 x 10 = 90 cells. Each cell trains online (fresh-batch causal LM) on the
synthetic induction task and logs the ICL-score curve + per-head prefix-match;
the summary records the emergence step / abruptness. Comparing the emergence
step distribution and abruptness across optimizer families (with matched seeds)
is the core test of whether Muon's orthogonalized update changes the timing,
sharpness, and cross-seed variance of induction-head / ICL emergence.

lr-arm stub (flagged, ~30 cells):
    --lr-arm adds a learning-rate robustness sweep so a timing difference cannot
    be dismissed as a single-lr artifact:
        muon  : muon_lr ∈ {0.01, 0.02, 0.04}
        adamw : lr      ∈ {5e-4, 1e-3, 2e-3}
        sgdm  : muon_lr ∈ {0.01, 0.02, 0.04}
    at the fixed middle seq_len (128) over seeds 0..2 -> 3 opt x 3 lr x ... but
    only the matching lr knob varies per family => 3 x 3 x ~3 ≈ 27-30 cells.

Output: ../../experiments/results/induction_emergence/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke   : delegate to train_induction smoke (no files, <60s) and exit 0.
--dry-run : print planned cells and exit 0 (launches NOTHING).
--lr-arm  : include the lr-robustness arm in the (dry-run or real) plan.
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

from train_induction import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


OPTIMIZERS = ["muon", "adamw", "sgdm"]
SEQ_LENS = [64, 128, 256]
SEEDS = list(range(10))           # 0-9
STEPS = 15000
EVAL_EVERY = 100

# lr-arm: which lr knob each family sweeps, and over which values.
LR_ARM = {
    "muon":  ("muon_lr", [0.01, 0.02, 0.04]),
    "adamw": ("lr",      [5e-4, 1e-3, 2e-3]),
    "sgdm":  ("muon_lr", [0.01, 0.02, 0.04]),
}
LR_ARM_SEQ_LEN = 128
LR_ARM_SEEDS = list(range(3))     # 0-2

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "induction_emergence")


def _main_cells():
    """Main grid cells: (name, overrides dict)."""
    cells = []
    for opt in OPTIMIZERS:
        for L in SEQ_LENS:
            for seed in SEEDS:
                name = f"{opt}_L{L}_s{seed}"
                cells.append((name, dict(optimizer=opt, seq_len=L, seed=seed)))
    return cells


def _lr_arm_cells():
    """lr-robustness arm cells (flagged): (name, overrides dict)."""
    cells = []
    for opt in OPTIMIZERS:
        knob, values = LR_ARM[opt]
        for val in values:
            for seed in LR_ARM_SEEDS:
                tag = f"{knob}{val}".replace("0.", "p")
                name = f"lrarm_{opt}_{tag}_s{seed}"
                ov = dict(optimizer=opt, seq_len=LR_ARM_SEQ_LEN, seed=seed)
                ov[knob] = val
                cells.append((name, ov))
    return cells


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
    ap = argparse.ArgumentParser(description="induction/ICL emergence grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    ap.add_argument("--lr-arm", action="store_true",
                    help="Include the lr-robustness arm in the plan")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _main_cells()
    if args.lr_arm:
        all_cells = all_cells + _lr_arm_cells()
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        n_main = len(_main_cells())
        n_lr = len(_lr_arm_cells())
        print(f"[induction_emergence] dry-run: {len(cells)} cells planned "
              f"(main={n_main}" + (f" + lr_arm={n_lr}" if args.lr_arm else "")
              + ")"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        print(f"  main grid: {len(OPTIMIZERS)} opt x {len(SEQ_LENS)} seq_len "
              f"x {len(SEEDS)} seeds = {n_main}")
        if args.lr_arm:
            print(f"  lr arm   : {n_lr} cells "
                  f"(seq_len={LR_ARM_SEQ_LEN}, seeds={LR_ARM_SEEDS})")
        for i, (name, ov) in enumerate(cells):
            print(f"  [{i+1:02d}/{len(cells)}] {name}  steps={STEPS}  {ov}")
        sys.exit(0)

    # --- real training path (only when neither smoke nor dry-run set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[induction_emergence] {len(cells)} cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (name, ov) in enumerate(cells):
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(steps=STEPS, eval_every=EVAL_EVERY, **ov)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"icl={s['final_icl_score']:.3f} "
            f"emerge={s['emergence_step']} "
            f"prefix={s['final_prefix_match_max']:.3f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[induction_emergence] DONE", flush=True)


if __name__ == "__main__":
    main()
