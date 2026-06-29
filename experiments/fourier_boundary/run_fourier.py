"""Direction 006 — Fourier-boundary grid runner (NOT executed yet).

Factorial grid resolving the from-scratch integer-add (2406.03445) vs mod-p
(2301.05217) Fourier-circuit contradiction by decoupling four axes:

    modular      ∈ {True, False}                 (wrap-around vs plain integer add)
    tokenization ∈ {single, digit}               (single-token vs digit-wise)
    coverage     ∈ {0.2, 0.4, 0.6, 0.8}          (train coverage fraction)
    seed         ∈ {0, 1, 2}

2 × 2 × 4 × 3 = 48 cells. Optimizer is FIXED AdamW (scope discipline vs
direction 004). `target` and `p_or_max` are derived from `modular`:
    modular=True  -> target="mod", p_or_max=97   (clean-Fourier regime)
    modular=False -> target="sum", p_or_max=200  (low-frequency regime)
so each cell stays a single coherent task definition.

Output: ../../experiments/results/fourier_boundary/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke   : delegate to train_fourier smoke (no files, <60s) and exit 0.
--dry-run : print the 48 planned cells and exit 0 (launches NOTHING).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

from train_fourier import Config, run, run_smoke  # noqa: E402
from data import DEFAULT_MODULAR_P, DEFAULT_NONMODULAR_MAX  # noqa: E402


MODULAR = [True, False]
TOKENIZATION = ["single", "digit"]
COVERAGE = [0.2, 0.4, 0.6, 0.8]
SEEDS = [0, 1, 2]
STEPS = 20000
EVAL_EVERY = 100

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results", "fourier_boundary")


def _cell_config(modular: bool, tok: str, cov: float, seed: int) -> Config:
    """Derive a coherent task definition for one factorial cell."""
    if modular:
        target, p_or_max = "mod", DEFAULT_MODULAR_P
    else:
        target, p_or_max = "sum", DEFAULT_NONMODULAR_MAX
    return Config(
        p_or_max=p_or_max, modular=modular, tokenization=tok,
        coverage=cov, target=target, seed=seed,
        steps=STEPS, eval_every=EVAL_EVERY,
    )


def _cell_name(modular: bool, tok: str, cov: float, seed: int) -> str:
    return f"mod{int(modular)}_{tok}_cov{cov}_s{seed}"


def _build_cells():
    return [
        (modular, tok, cov, seed)
        for modular in MODULAR
        for tok in TOKENIZATION
        for cov in COVERAGE
        for seed in SEEDS
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
    ap = argparse.ArgumentParser(description="Direction 006 Fourier-boundary grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        sys.exit(0)

    cells = _build_cells()

    if args.dry_run:
        print(f"[fourier_boundary] dry-run: {len(cells)} cells planned")
        for i, (modular, tok, cov, seed) in enumerate(cells):
            name = _cell_name(modular, tok, cov, seed)
            target = "mod" if modular else "sum"
            pm = DEFAULT_MODULAR_P if modular else DEFAULT_NONMODULAR_MAX
            print(f"  [{i+1:02d}/{len(cells)}] {name}  "
                  f"target={target} p_or_max={pm} steps={STEPS}")
        sys.exit(0)

    # --- real training path (only when neither flag set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[fourier_boundary] {len(cells)} cells -> {OUT}", flush=True)

    for i, (modular, tok, cov, seed) in enumerate(cells):
        name = _cell_name(modular, tok, cov, seed)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = _cell_config(modular, tok, cov, seed)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"test={s['final_test_acc']:.3f} sparsity={s['final_sparsity']:.3f} "
            f"top_freqs={s['final_top_freqs']} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[fourier_boundary] DONE", flush=True)


if __name__ == "__main__":
    main()
