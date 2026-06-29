# Findings 016 — As-run, the in-context-TD testbed does NOT show a clean emergence; the detector fires on noise (testbed recalibration needed before P1–P4 can be adjudicated)

**Direction:** `directions/016-icrl-td-emergence.md` (bank ICRL-TD1 7.7; the
repo's **first RL-family** direction — pure-simulation MRP value-learning streams,
in-context TD). **Status: testbed-calibration negative — the headline questions
(sharp emergence / precursor diagnostics / optimizer-moves-transition) are NOT
adjudicable from this grid and should not be reported as positive.**
**Data:** `results/icrl_td/` — 45 runs: {adamw, muon, sgdm} × horizon T∈{10,20,40}
× 5 seeds, n_states=10, acc_thresh=0.7.

## Headline (honest negative)

The model **does not learn in-context TD to high accuracy at this config**:
final validation accuracy is **chance-level almost everywhere** (median 0.562,
min 0.406, max 0.875; **only 1/45 cells exceed 0.8**), and the per-eval val_acc
trajectories oscillate (0.5↔0.72) rather than showing a sustained grokking-like
jump. The `emergence_step` detector — first crossing of acc_thresh=0.7 — therefore
fires on **transient noise crossings**, not on a genuine phase transition: cells
flagged "emerged" settle back to ~0.56 final accuracy. The apparent pattern
"emergence rate rises with T (T10: 0–1/5 → T40: 5/5) and emergence_step drops with
T (T40: 250–800 steps)" is an **artifact of the noise-crossing detector** — more
in-context examples at larger T make the eval estimate noisier / the spurious
0.7-crossing faster — NOT evidence of faster TD computation.

## Why the P-predictions are not adjudicable here

- **P1 (sharp emergence):** cannot be tested — there is no sustained high-accuracy
  state to emerge INTO. final_val_acc ≈ chance.
- **P2 (precursor diagnostics):** final_td_alignment is small and inconsistent
  (0.076–0.297, some negative, e.g. sgdm/T10 −0.311); no clean TD-alignment
  signal builds, so it cannot pre-predict an emergence that does not occur.
- **P4 (optimizer moves the transition):** the optimizer differences in
  "emergence rate/step" are differences in the **noise-crossing statistics**, not
  in a real transition point — uninterpretable until the task is solved.

## Diagnosis (parallel to 011's walsh-unlearnable finding)

Like 011's Walsh arm, this is a **testbed-calibration failure, not a scientific
result about TD**: the ICRL-TD task at n_states=10 with this tiny model / step
budget / eval protocol does not reach a solved regime, and the val_acc readout is
too noisy (small eval batches) to detect emergence even if it occurred. Candidate
fixes before re-running P1–P4:
1. **Verify solvability**: confirm the MRP value-learning task is in-principle
   learnable to high acc by SOME setting (larger model, more steps, smaller
   n_states / easier horizon) — establish a solved reference cell first.
2. **De-noise the readout**: many more eval MRP streams per eval (the val_acc
   bounce of ±0.2 indicates too-small eval batches); report value MSE / regret,
   not just a thresholded acc.
3. **Sustained-emergence criterion**: require acc ≥ thresh held for K consecutive
   evals (the 009/012 lesson: a single threshold crossing is noise; a sustained
   plateau is emergence) — kills the spurious-crossing detector.
4. Only then sweep optimizer × T for the real P1/P2/P4.

## Limitations

- One task scale (n_states=10), one model size, one eval protocol — all of which
  the diagnosis implicates.
- A solved-reference cell was not in the grid, so "unlearnable at this config"
  cannot yet be separated from "needs more steps/capacity" — fix #1 decides this.
- The pure-simulation MRP harness and probes themselves passed smoke (the code
  runs); the negative is about task/eval calibration, not a code bug.
- **Publication status (red-team 2026-06-14):** this is an UNCONTROLLED negative
  (no solved-reference cell → cannot separate "unlearnable at this config" from
  "under-trained"), so it is **not includable as a standalone result**; at most a
  2-sentence cautionary vignette ("threshold-crossing emergence detectors fire on
  eval noise; require sustained-K-eval criteria") inside a methodology note — which
  014 already teaches, so it is largely redundant. Do not give it its own paper
  section or figure.

## Figures / data

`results/icrl_td/` per-cell jsonl (emergence_step, final_val_acc[/_ood],
final_td_alignment, final_p4_tracking + per-eval trajectories). No figure pass —
the result is a calibration-negative; a figure would imply a signal that is absent.
