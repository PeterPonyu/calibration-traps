#!/usr/bin/env python3
"""E2 positive-control rescue: a discount (gamma) difficulty ladder in the
SAME in-context-TD pipeline (model, tokenization, evaluator, detector).

Rationale: the canonical grid (gamma=0.9, 10 states) never learns, and all
prior easier-cell controls (fewer states, shorter T) also failed, so the
detector has no true transition to be validated against. At gamma=0 the value
function collapses to the immediate discretized reward, V(s)=r_disc(s), making
the task an in-context lookup of the queried state's reward token -- the same
competence class as the induction positive control, but inside THIS pipeline.
gamma in {0.0, 0.3, 0.9} spans solvable -> hard with no other change.

The gamma patch must land BEFORE train_icrl/probes are imported (probes binds
data.GAMMA into default args at import time).

Modes:
  --smoke  : coverage/oracle-ceiling audit + 1500-step gamma=0 learnability
             probe (writes nothing to results)
  --pilot  : 1 seed per gamma, full schedule
  --full   : 5 seeds x 3 gammas (skips completed pilot logs)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = ROOT / "experiments" / "icrl_td"
sys.path.insert(0, str(THIS))

import data as D  # noqa: E402  (patch BEFORE importing train_icrl/probes)


def set_gamma(g: float):
    D.GAMMA = g
    D.V_MAX = 1.0 if g == 0.0 else 1.0 / (1.0 - g)
    # probes.py binds data.GAMMA at ITS import; re-bind defaults if loaded.
    if "probes" in sys.modules:
        import probes as PR
        PR.GAMMA = g
        PR.td_trace.__defaults__ = (0.2, g)
        PR.bellman_residual.__defaults__ = (g,)


GAMMAS = [0.0, 0.3, 0.9]
N_STATES = 6
T = 40
STEPS = 10000
OUT = ROOT / "experiments" / "results" / "icrl_td_gamma_ladder"


def make_cfg(TI, gamma, seed, steps=STEPS):
    return TI.Config(optimizer="adamw", seed=seed, n_states=N_STATES, T=T,
                     steps=steps, eval_every=50, eval_mrps=32,
                     acc_thresh=0.8, weight_decay=0.01, lr=1e-3)


def name(gamma, seed):
    return f"gamma{str(gamma).replace('.', 'p')}_T{T}_s{seed}"


def coverage_audit(gamma, n=2000):
    """Oracle ceiling: accuracy of the analytic-value predictor restricted to
    in-context information (query state visited or not). T=40 chosen so the
    ceiling (~0.95) clears the 0.8 detection bar with margin."""
    set_gamma(gamma)
    rng = np.random.default_rng(0)
    hit_cov, orac = 0, 0
    for _ in range(n):
        mrp = D.make_mrp(rng, N_STATES)
        toks, tgt, meta = D.sample_sequence(rng, mrp, T)
        visited = meta["s_q"] in set(meta["states"].tolist())
        hit_cov += visited
        # oracle: exact target if visited; central bucket guess otherwise
        guess = tgt if visited else D.value_bucket(0.0)
        orac += (guess == tgt)
    return hit_cov / n, orac / n


def run_cells(cells, tag):
    set_gamma(cells[0][0])  # ensure module state initialized pre-import
    import train_icrl as TI  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    for gamma, seed in cells:
        out = OUT / f"{name(gamma, seed)}.jsonl"
        if out.exists():
            last = None
            for line in out.read_text().splitlines():
                if line.strip():
                    last = json.loads(line)
            if isinstance(last, dict) and "_summary" in last:
                print(f"skip {out.name} (complete)")
                continue
        set_gamma(gamma)
        cfg = make_cfg(TI, gamma, seed)
        t0 = time.time()
        summary, history = TI.train(cfg, out_path=str(out))
        # annotate gamma + detector-facing stats into a final line
        accs = [r["val_acc"] for r in history]
        rec = {"_gamma_summary": {
            "gamma": gamma, "seed": seed,
            "final_val_acc": summary["final_val_acc"],
            "max_val_acc": max(accs) if accs else None,
            "single_cross_0p8": bool(any(a >= 0.8 for a in accs)),
            "sustain2_0p8": bool(any(a >= 0.8 and b >= 0.8
                                     for a, b in zip(accs, accs[1:]))),
            "emergence_step": summary["emergence_step"],
            "elapsed_min": round((time.time() - t0) / 60, 2)}}
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{tag}] {out.name}: final={summary['final_val_acc']:.3f} "
              f"max={rec['_gamma_summary']['max_val_acc']:.3f} "
              f"sustain2={rec['_gamma_summary']['sustain2_0p8']} "
              f"({rec['_gamma_summary']['elapsed_min']} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        for g in GAMMAS:
            cov, orac = coverage_audit(g)
            print(f"gamma={g}: query-visited coverage={cov:.3f} "
                  f"oracle-ceiling accuracy={orac:.3f}")
        # learnability probe: gamma=0, short budget, no writes
        set_gamma(0.0)
        import train_icrl as TI
        cfg = make_cfg(TI, 0.0, seed=0, steps=1500)
        summary, history = TI.train(cfg, out_path=None)
        accs = [r["val_acc"] for r in history]
        print(f"gamma=0 smoke: acc t0={accs[0]:.3f} -> "
              f"mid={accs[len(accs)//2]:.3f} -> final={accs[-1]:.3f} "
              f"(max {max(accs):.3f})")
        print("SMOKE " + ("PASS: learning signal present"
                          if max(accs) > accs[0] + 0.15 else
                          "FAIL: no learning slope -- redesign needed"))
        return

    if args.pilot:
        run_cells([(g, 0) for g in GAMMAS], "pilot")
        return

    if args.full:
        run_cells([(g, s) for g in GAMMAS for s in range(5)], "full")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
