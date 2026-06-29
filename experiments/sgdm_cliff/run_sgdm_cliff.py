"""Direction 020 — SGDM trainability cliff: (width × lr) phase diagram + staircase shift.

Reuses degree_staircase end-to-end (train_staircase.Config/run, run_staircase.already_done).
SGDM = SGD(momentum) on hidden matrices with lr = cfg.muon_lr (the existing hybrid
split); so the SGDM "lr" axis IS muon_lr. AdamW/Muon control arm (P3) sweeps the
same width×profile.

PRIMARY (default): optimizer=sgdm × width{32,64,128,256,512} × lr{.003,.01,.03,.1,.3,1}
  × profile{pure(deg3), staircase} × 5 seeds = 300 cells. success = final_fit_corr >= TAU.
CONTROL (--control): {adamw,muon} × same width × profile × lr{.03,.3} × 3 seeds (P3).
Cliff-column seed densification (veto-corrector: success-rate is binomial, the
critical-width band needs 10-15 seeds) = a follow-up pass once the first sweep
locates the cliff (run with --dense-seeds N --widths w1,w2 after).

Output: ../../experiments/results/sgdm_cliff/<name>.jsonl (own slug). Resume-safe.
"""
from __future__ import annotations

import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_THIS)
_DS = os.path.join(_EXP, "degree_staircase")
for p in (_THIS, _EXP, _DS):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_staircase import Config, run  # noqa: E402
from run_staircase import already_done    # noqa: E402

WIDTHS = [32, 64, 128, 256, 512]
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PROFILES = [("pure3", dict(profile="pure", pure_degree=3)),
            ("staircase", dict(profile="staircase"))]
TAU = 0.6                       # success: final_fit_corr >= TAU
OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "sgdm_cliff")


def _name(opt, prof, width, lr, seed):
    return f"{opt}_{prof}_w{width}_lr{lr:g}_s{seed}".replace(".", "p")


def build_cells(control: bool, dense_seeds=0, dense_widths=None):
    cells = []
    if control:
        cwidths = dense_widths if dense_widths else WIDTHS
        for opt in ("adamw", "muon"):
            for pname, _ in PROFILES:
                for width in cwidths:
                    for lr in (0.03, 0.3):
                        for s in range(3):
                            cells.append((opt, pname, width, lr, s))
        return cells
    # primary SGDM grid
    widths = dense_widths if dense_widths else WIDTHS
    nseed = dense_seeds if dense_seeds else 5
    for pname, _ in PROFILES:
        for width in widths:
            for lr in LRS:
                for s in range(nseed):
                    cells.append(("sgdm", pname, width, lr, s))
    return cells


def _cfg(opt, pname, width, lr, seed):
    prof_kw = next(p for n, p in PROFILES if n == pname)
    kw = dict(d_model=width, optimizer=opt, seed=seed, **prof_kw)
    # lr axis maps to the optimizer's hidden-matrix lr: muon_lr for sgdm/muon,
    # cfg.lr for adamw.
    if opt == "adamw":
        kw["lr"] = lr
    else:
        kw["muon_lr"] = lr
    return Config(**kw)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--dense-seeds", type=int, default=0)
    ap.add_argument("--dense-widths", default=None)
    try:
        from runner_utils import add_shard_args, shard_cells, validate_shard_args
        add_shard_args(ap); _shard = True
    except Exception:
        _shard = False
    args = ap.parse_args()

    if args.smoke:
        c = _cfg("sgdm", "staircase", 64, 0.1, 0)
        c.steps = 200
        s, _ = run(c)
        print(f"SMOKE sgdm w64 staircase 200-step: fit={s['final_fit_corr']:.3f} OK")
        c2 = _cfg("sgdm", "pure3", 64, 0.1, 0); c2.steps = 200
        s2, _ = run(c2)
        print(f"SMOKE sgdm w64 pure3 200-step: fit={s2['final_fit_corr']:.3f} OK")
        print("RUN_SGDM_CLIFF SMOKE PASS")
        sys.exit(0)

    dw = [int(x) for x in args.dense_widths.split(",")] if args.dense_widths else None
    cells = build_cells(args.control, args.dense_seeds, dw)
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        print(f"[sgdm_cliff] {len(cells)} cells "
              f"({'control' if args.control else 'primary'})")
        for c in cells[:8]:
            print("  ", _name(*c))
        sys.exit(0)

    os.makedirs(OUT, exist_ok=True)
    print(f"[sgdm_cliff] {len(cells)} cells -> {OUT}", flush=True)
    for i, (opt, pname, width, lr, seed) in enumerate(cells):
        name = _name(opt, pname, width, lr, seed)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = _cfg(opt, pname, width, lr, seed)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        fit = s["final_fit_corr"]
        print(f"[{i+1}/{len(cells)}] {name}: fit={fit:.3f} "
              f"success={fit >= TAU} ({time.time()-t0:.0f}s)", flush=True)
    print("[sgdm_cliff] DONE", flush=True)


if __name__ == "__main__":
    main()
