"""Detector validation on REAL emergence: the 90 induction-head runs.

E2's sustained-K criterion is calibrated against noise (the in-context-TD
grid, where nothing is learned). This script supplies the other half of the
validation -- behaviour on genuine transitions -- using the induction-head
emergence runs (3 optimizers x 3 context lengths x 10 seeds), where the
repeat-token accuracy really does rise from chance to ~1.

Per run: does the single-crossing detector fire? does sustained-K=2? how much
detection latency does sustaining cost? and on runs that never emerge, does
either detector fire (false positives on real, structured non-emergence)?

Pure CPU; reads results/induction_emergence/*.jsonl.
"""
import json
import glob
import os
from statistics import median

TAU = 0.8
EMERGED_FINAL = 0.9   # a run counts as truly emerged if final acc >= this
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "induction_emergence")


def load(f):
    steps, accs = [], []
    meta = {}
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if "_meta" in r:
            meta = r["_meta"]
            continue
        if "repeat_acc" in r:
            steps.append(r["step"])
            accs.append(r["repeat_acc"])
    return meta, steps, accs


def first_cross(steps, accs, tau=TAU):
    for s, a in zip(steps, accs):
        if a >= tau:
            return s
    return None


def first_sustain(steps, accs, k, tau=TAU):
    run = 0
    for i, a in enumerate(accs):
        run = run + 1 if a >= tau else 0
        if run >= k:
            return steps[i - k + 1]  # start of the qualifying run
    return None


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
        meta, steps, accs = load(f)
        if not accs:
            continue
        final = median(accs[-5:])
        rows.append({
            "name": os.path.basename(f).replace(".jsonl", ""),
            "opt": meta.get("optimizer", "?"),
            "final": final,
            "emerged": final >= EMERGED_FINAL,
            "cross": first_cross(steps, accs),
            "sus2": first_sustain(steps, accs, 2),
            "sus3": first_sustain(steps, accs, 3),
            "n_ckpt": len(accs),
        })

    emerged = [r for r in rows if r["emerged"]]
    non = [r for r in rows if not r["emerged"]]
    print(f"runs: {len(rows)} | emerged (final>= {EMERGED_FINAL}): "
          f"{len(emerged)} | non-emerged: {len(non)}")

    # TPR
    tp1 = sum(r["cross"] is not None for r in emerged)
    tp2 = sum(r["sus2"] is not None for r in emerged)
    tp3 = sum(r["sus3"] is not None for r in emerged)
    print(f"TPR single-crossing: {tp1}/{len(emerged)}")
    print(f"TPR sustained-2:     {tp2}/{len(emerged)}")
    print(f"TPR sustained-3:     {tp3}/{len(emerged)}")

    # latency cost of sustaining (among runs where both fire)
    lat2 = [r["sus2"] - r["cross"] for r in emerged
            if r["cross"] is not None and r["sus2"] is not None]
    lat3 = [r["sus3"] - r["cross"] for r in emerged
            if r["cross"] is not None and r["sus3"] is not None]
    if lat2:
        print(f"latency sustained-2 minus single: median {median(lat2)} steps "
              f"(max {max(lat2)}); zero-latency runs: "
              f"{sum(l == 0 for l in lat2)}/{len(lat2)}")
    if lat3:
        print(f"latency sustained-3 minus single: median {median(lat3)} steps "
              f"(max {max(lat3)})")

    # false fires on non-emerged runs
    fp1 = [r for r in non if r["cross"] is not None]
    fp2 = [r for r in non if r["sus2"] is not None]
    print(f"non-emerged runs firing single-crossing: {len(fp1)}/{len(non)}"
          + (f"  ({[r['name'] for r in fp1]})" if fp1 else ""))
    print(f"non-emerged runs firing sustained-2:     {len(fp2)}/{len(non)}"
          + (f"  ({[r['name'] for r in fp2]})" if fp2 else ""))

    # by optimizer
    for opt in ("adamw", "muon", "sgdm"):
        sub = [r for r in rows if r["opt"] == opt]
        if sub:
            em = sum(r["emerged"] for r in sub)
            print(f"  {opt}: {em}/{len(sub)} emerged; sustained-2 fired on "
                  f"{sum(r['sus2'] is not None for r in sub)}")

    out = os.path.join(RES, "..", "figures-deepcheck",
                       "detector_validation_induction.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
