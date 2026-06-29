"""Direction 007 analysis — P1–P3 verdicts for Muon × induction-head emergence.

Reads results/induction_emergence/*.jsonl (3 opt × L{64,128,256} × 10 seeds).
Writes to results/figures-007/:
  induction_verdicts.json
  fig_emergence_scaling.png   emergence step vs L (log-log) + fitted exponents (P1/P3)
  fig_variance_signature.png  per-cell seed spread of emergence step (P2)
  fig_sharpness.png           transition slope & width by cell
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import sys as _sys
_sys.path.insert(0, "/home/zeyufu/Desktop/dl-research/experiments")
import figstyle
figstyle.apply()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
DIR = os.path.join(RES, "induction_emergence")
FIG = os.path.join(RES, "figures-007")
COLORS = {"adamw": figstyle.OPT["adamw"], "muon": figstyle.OPT["muon"],
          "sgdm": figstyle.OPT["sgdm"]}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}


def load_summaries():
    cells = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        last = None
        with open(p) as f:
            for line in f:
                last = line
        try:
            s = json.loads(last).get("_summary")
        except Exception:
            s = None
        if s:
            cells[(s["optimizer"], s["seq_len"])].append(s)
    return cells


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = load_summaries()
    print(f"cells: {[(k, len(v)) for k, v in sorted(cells.items())]}")

    table = {}
    print(f"{'cell':16s} {'emerge mean':>11s} {'std':>8s} {'min..max':>15s} "
          f"{'slope':>8s} {'width':>8s} {'icl':>6s} {'n_emerged':>9s}")
    for (opt, L), ss in sorted(cells.items()):
        em = [s["emergence_step"] for s in ss if s["emergence_step"] is not None]
        slopes = [s["emergence_max_slope"] for s in ss
                  if s.get("emergence_max_slope") is not None]
        widths = [s["emergence_transition_width"] for s in ss
                  if s.get("emergence_transition_width") is not None]
        icl = [s["final_icl_score"] for s in ss]
        row = {
            "n": len(ss), "n_emerged": len(em),
            "emergence_mean": float(np.mean(em)) if em else None,
            "emergence_std": float(np.std(em)) if em else None,
            "emergence_min": min(em) if em else None,
            "emergence_max": max(em) if em else None,
            "slope_mean": float(np.mean(slopes)) if slopes else None,
            "width_mean": float(np.mean(widths)) if widths else None,
            "icl_mean": float(np.mean(icl)),
        }
        table[f"{opt}_L{L}"] = row
        if em:
            print(f"{opt+'_L'+str(L):16s} {row['emergence_mean']:11.0f} "
                  f"{row['emergence_std']:8.1f} "
                  f"{str(row['emergence_min'])+'..'+str(row['emergence_max']):>15s} "
                  f"{row['slope_mean'] if row['slope_mean'] else float('nan'):8.4f} "
                  f"{row['width_mean'] if row['width_mean'] else float('nan'):8.0f} "
                  f"{row['icl_mean']:6.3f} {row['n_emerged']:>4d}/{row['n']}")

    # ---- P3: log-log exponent fit per optimizer ----
    exponents = {}
    for opt in ["adamw", "sgdm", "muon"]:
        xs, ys = [], []
        for (o, L), ss in cells.items():
            if o != opt:
                continue
            for s in ss:
                if s["emergence_step"] is not None:
                    xs.append(np.log(L))
                    ys.append(np.log(s["emergence_step"]))
        if len(set(xs)) >= 2:
            coef, cov = np.polyfit(xs, ys, 1, cov=True)
            exponents[opt] = {"exponent": float(coef[0]),
                              "stderr": float(np.sqrt(cov[0, 0])),
                              "n": len(xs)}
    table["_P3_exponents"] = exponents
    print("P3 exponents:", json.dumps(exponents, indent=None))

    with open(os.path.join(FIG, "induction_verdicts.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote induction_verdicts.json")

    # ---- scaling figure (single column, ~3.35in printed) ----
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN["col2_single"], 2.7))
    for opt in ["adamw", "sgdm", "muon"]:
        Ls, means, stds = [], [], []
        for (o, L), ss in sorted(cells.items()):
            if o != opt:
                continue
            em = [s["emergence_step"] for s in ss if s["emergence_step"] is not None]
            if em:
                Ls.append(L)
                means.append(np.mean(em))
                stds.append(np.std(em))
            for s in ss:  # seed scatter
                if s["emergence_step"] is not None:
                    ax.scatter(L * (1 + 0.02 * (["adamw", "sgdm", "muon"].index(opt) - 1)),
                               s["emergence_step"], color=COLORS[opt], s=14, alpha=0.45)
        if Ls:
            # legend = optimizer name only (the scaling exponent is internal and
            # not discussed in the manuscript text)
            ax.errorbar(Ls, means, yerr=stds, fmt="-o", color=COLORS[opt],
                        capsize=4, label=OPT_LABEL[opt])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context length L")
    ax.set_ylabel("ICL emergence step")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_emergence_scaling.png"))
    plt.close(fig)
    print("wrote fig_emergence_scaling.png")

    # ---- variance signature ----
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    xt, xl = [], []
    i = 0
    for L in [64, 128, 256]:
        for opt in ["adamw", "sgdm", "muon"]:
            ss = cells.get((opt, L), [])
            em = [s["emergence_step"] for s in ss if s["emergence_step"] is not None]
            if em:
                ax.scatter([i] * len(em), em, color=COLORS[opt], s=28, alpha=0.6)
                ax.bar(i, np.std(em), width=0.6, color=COLORS[opt], alpha=0.18)
            xt.append(i)
            xl.append(f"{opt}\nL{L}")
            i += 1
        i += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel("emergence step (dots) / seed std (bars)")
    ax.set_title("P2: cross-seed variance signature (10 seeds per cell)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_variance_signature.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_variance_signature.png")

    # ---- sharpness ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(3)
    for j, opt in enumerate(["adamw", "sgdm", "muon"]):
        sl, wd = [], []
        for L in [64, 128, 256]:
            r = table.get(f"{opt}_L{L}", {})
            sl.append(r.get("slope_mean") or 0)
            wd.append(r.get("width_mean") or 0)
        axes[0].bar(x + (j - 1) * 0.26, sl, 0.26, color=COLORS[opt], alpha=0.85,
                    label=opt)
        axes[1].bar(x + (j - 1) * 0.26, wd, 0.26, color=COLORS[opt], alpha=0.85,
                    label=opt)
    for ax, lab in [(axes[0], "max transition slope"),
                    (axes[1], "transition width (steps)")]:
        ax.set_xticks(x)
        ax.set_xticklabels(["L64", "L128", "L256"])
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Emergence sharpness by cell")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sharpness.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_sharpness.png")


if __name__ == "__main__":
    main()
