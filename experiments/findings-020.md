# Findings 020 — SGDM has a sharp (width × lr) trainability cliff on a Boolean degree task, and the degree-staircase target shifts it ~8× to lower width

**Direction:** `directions/020-sgdm-trainability-cliff.md` (bank C2 8.1; killer-sweep
PROCEED high-confidence, `.omc/research/novelty-c2-2026-0614-raw.json`).
**Motivated by our own SGDM-floor evidence** (004: SGDM fit 0.00 on pure-deg-3 vs
0.73–0.88 on the staircase ladder; 014: SGDM = config-unpredictable pure-noise
regime; 019: SGDM 0/5 on every group incl abelian Z60). The floor was established
as point-failures; 020 maps its (width × lr) PHASE STRUCTURE.
**Data:** `results/sgdm_cliff/` — PRIMARY 300 runs: sgdm × width{32,64,128,256,512}
× lr{0.003,0.01,0.03,0.1,0.3,1.0} (the SGD-side `muon_lr`) × profile{pure(deg-3),
staircase} × 5 seeds. Reuses `degree_staircase` trainer/data/probes UNCHANGED.
success = final_fit_corr ≥ 0.6. CONTROL arm (P3) 120 runs {adamw,muon} × same
width{32–256} × profile × lr{0.03,0.3} × 3 seeds — **CLOSED 2026-06-15: cliff is
SGDM-specific (Muon no cliff), see P3** (w512 dropped as least-informative for P3).

## Headline

**On a Boolean degree task, plain SGD+momentum exhibits a SHARP trainability cliff
in the (width × lr) plane — and the nested degree-STAIRCASE target shifts that
cliff ~8× toward lower width, "rescuing" trainability.** Pure-degree-3 is trainable
only in a small corner (large width × small lr); the staircase target makes nearly
the whole low-lr column trainable down to the smallest width. The cliff's upper
region (large lr) is a shared SGDM death zone for both targets — consistent with
the optimizer-family floor (014/019).

## P1 — Sharp (width × lr) cliff for SGDM: **CONFIRMED (a step, not a smooth ramp)**

Success-rate (fit_corr ≥ 0.6, n=5) per (width × lr):

**pure-degree-3:**

| w\lr | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| 32  | 0% | 0% | 0% | 0% | 0% | 0% |
| 64  | 20% | 0% | 0% | 0% | 0% | 40% |
| 128 | 0% | 0% | 0% | 60% | 40% | 0% |
| 256 | 80% | 100% | 100% | 0% | 0% | 0% |
| 512 | 100% | 100% | 80% | 20% | 0% | 0% |

At fixed small lr the success-rate-vs-width is a **step**, not a smooth ramp (e.g.
lr=0.01: 0,0,0,100,100 — a clean break between w128 and w256). The trainable set is
a bottom-left corner: **7/30 cells**. The mid-range cells (20–60%) are n=5 binomial
noise; the 0%/100% cells are the robust backbone. P1 is confirmed: the SGDM floor
is a sharp (width × lr) phase boundary, not a gradual decline.

## P2 — The staircase target shifts the cliff to lower width: **CONFIRMED (boundary shift, ~8×, not a uniform lift)**

**staircase (full degree ladder):**

| w\lr | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| 32  | 80% | 100% | 100% | 80% | 40% | 60% |
| 64  | 100% | 100% | 100% | 100% | 20% | 0% |
| 128 | 100% | 100% | 80% | 40% | 40% | 20% |
| 256 | 100% | 100% | 80% | 60% | 0% | 0% |
| 512 | 100% | 100% | 80% | 0% | 0% | 0% |

Trainable set **19/30 cells** (vs pure3's 7/30). The cliff EDGE (min width with
success-rate ≥ 60%) moves sharply:

| lr | pure3 min-width | staircase min-width |
|---|---|---|
| 0.003 | 256 | **32** (8× lower) |
| 0.01 | 256 | **32** (8×) |
| 0.03 | 256 | **32** (8×) |
| 0.1 | 128 | **32** (4×) |
| 0.3 | never | never |
| 1.0 | never | **32** |

This is a **boundary SHIFT, not a uniform additive lift**: at low lr the staircase
makes width-32 trainable where pure3 needs width-256 — an ~8× reduction in the
critical width. The preregistered P2 falsifier (staircase gives only a uniform lift
or no cliff) does NOT fire. The staircase "rescues" SGDM trainability by moving the
cliff, quantified as the edge-shift table above.

## P3 — AdamW / Muon control: is the cliff SGDM-specific? **CONFIRMED (Muon has no cliff)**

Control arm ({adamw,muon} × width{32–256} × profile × lr{0.03,0.3} × 3 seeds; w512
dropped as least-informative for P3). At the informative low-lr column (lr=0.03),
pure-deg-3 success across w{32,64,128,256}:
- **SGDM**: 0/5, 0/5, 0/5, 5/5 — the sharp width cliff (fails below w256).
- **Muon**: **3/3 at EVERY width** — no cliff; Muon trains pure-deg-3 even at w32.
- AdamW: 2/3, 1/3, 0/3, 0/3 — a different, non-cliff pattern, but lr=0.03 is ~30×
  AdamW's standard 1e-3 (the shared "lr" axis is the hidden-matrix lr), so AdamW is
  off its operating point here and uninformative for the cliff question.

**Verdict: the (width × lr) trainability cliff is SGDM-specific — NOT a shared
capacity floor.** Muon succeeds exactly where SGDM fails (w32–128 at lr=0.03): a
model of that width CAN learn pure-deg-3, so it is SGD-momentum specifically that
cannot, below the critical width. The cliff is an optimizer-family property of
plain SGD+momentum (confirming the 014/019 floor framing); the staircase rescue
(P2) then lowers SGDM's critical width ~8×. Caveats: control n=3; the clean
comparison is Muon-vs-SGDM (AdamW's lr=0.03 is off its operating point).

## Boundary discovery (unplanned)

**The large-lr region (0.3, 1.0) is a shared SGDM death zone for BOTH targets** —
even the staircase rescue cannot recover lr=0.3 at most widths. This is the
(width × lr) face of the optimizer-family floor that 014 (SGDM config-share zero)
and 019 (SGDM 0/5 on every group) measured as point-failures: the floor is not
uniform — it has a trainable corner (low lr, sufficient width) whose size the
target structure modulates.

## Limitations

- n=5 per cell; intermediate success-rates (20–80%) are binomial-noisy. The cliff
  columns warrant densification to 10–15 seeds (the veto-corrector) before any
  fine claim about the exact critical width; the 0%/100% backbone and the ~8×
  edge-shift are robust to this.
- success threshold τ=0.6 (=3/5 seeds at the edge); the qualitative phase structure
  is threshold-robust (the 0%/100% cells dominate).
- One trainer/testbed (degree_staircase, online fresh-batch MSE regression, L=16,
  d-sweep); the "lr" axis is the SGD-side hidden-matrix lr (the existing hybrid
  split), AdamW on the rest.
- P2's staircase-rescue MECHANISM is established on the adjacent step-count /
  sample-complexity axis (2301.13105 / 2306.16921, cited as MOTIVATION); 020 claims
  the EMPIRICAL (width × lr) boundary shift, which is the unoccupied contribution.
- P3 control arm CLOSED (2026-06-15): cliff is SGDM-specific (Muon no cliff at any
  width); control n=3, AdamW lr=0.03 off its operating point so Muon-vs-SGDM is the
  clean comparison.

## Figures / data

`results/sgdm_cliff/` (300 primary + 120 control). Figure pass (figures-020) and
the (width × lr) phase-diagram render are DEFERRED to the final unified packaging
pass (per the science-before-packaging directive).
