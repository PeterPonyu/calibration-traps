"""Quick muon_lr validation for the C-KILL-1 optaxis runners (red-team MAJOR-4).

Runs AdamW (reference) + Muon at a few lrs on the d128/L2 staircase (1 seed) and
picks the muon_lr whose final staircase fit is CLOSEST to AdamW's — so the optaxis
geometry comparison isn't an lr artifact. Writes the chosen lr to .muon_lr_chosen.
Fallback: 0.02 if the sweep fails.
"""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "degree_staircase"))
import train_staircase as TS  # noqa: E402

CANDS = (0.01, 0.02, 0.04)
STEPS = 4000


def fit(opt, mlr=None):
    kw = dict(profile="staircase", d_model=128, n_layers=2, steps=STEPS,
              eval_every=200, seed=0, optimizer=opt)
    if opt == "adamw":
        kw["lr"] = 1e-3
    else:
        kw["muon_lr"] = mlr
    s, _ = TS.run(TS.Config(**kw), out_path=None)
    return s["final_fit_corr"]


def main():
    chosen = 0.02
    try:
        ref = fit("adamw")
        cands = {m: fit("muon", m) for m in CANDS}
        chosen = min(cands, key=lambda m: abs(cands[m] - ref))
        print(f"adamw fit={ref:.3f}; muon fits={ {m: round(v,3) for m,v in cands.items()} }; "
              f"chosen muon_lr={chosen} (closest to adamw)")
    except Exception as e:
        print(f"muon_lr sweep failed ({e}); falling back to 0.02")
    with open(os.path.join(_THIS, ".muon_lr_chosen"), "w") as f:
        f.write(str(chosen))


if __name__ == "__main__":
    main()
