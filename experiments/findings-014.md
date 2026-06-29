# Findings 014 — Early trajectories carry no usable run-level information beyond configuration identity (clean negatives on all three prediction laws)

**Direction:** `directions/014-spec-route-predictor.md` (bank W1-SPEC-ROUTE 8.5;
ROUND-1 zero-GPU post-hoc; banned claim honored: this is NOT "first early
prediction of grokking" — increment-over-config accounting against 2306.13253 /
2602.16967 et al. is the contribution).
**Corpus:** 364 existing runs across 11 closed namespaces (TIER_A norm channels:
grid_main, lr_control(+sc3), wd_sweep, tf_sweep, task_mul, task_s5, fine_eval,
s5_rescue; TIER_B mech probes: mech, s5_mech), 252 route-labeled grok events.
Zero new training. Windows: W-mem (up to memorization) and W-200 (first 200
steps); features: trajectory-only summaries vs config-identity baseline;
models: numpy ridge / L2-logistic, group-aware CV (held-out config cell /
seed / task), label-permutation nulls.
**Output:** `results/spec_route/posthoc_round1.json`, `results/figures-014/`.

## Headline

At this corpus scale, **the early window is a fingerprint of the configuration,
not of the run**: outcome variance is config-saturated (Muon), unfittable by the
linear predictor (SGDM: R²<0 — see P4 caveat), or carries a signal share that
fails to convert into any preregistered prediction (AdamW). P1, P2, P3 all land on their preregistered clean-negative
branches; P4's predictability-structure expectation is confirmed. The pooled
route AUROC of 0.815 collapsing to 0.57/0.47 within-AdamW is a quantified
warning for the timing-prediction literature: **optimizer identity masquerades
as trajectory predictiveness** unless the optimizer axis is frozen.

## P1 — Seed-level signal increment for delay: **FALSIFIED**

Within high-seed-variance cells (LOO residual prediction, within-cell
permutation null): Spearman ρ = 0.077 (W-mem, n=74, p=0.015) and 0.014 (W-200,
n=109, p=0.005) — statistically detectable, practically nil — while the signal
model's MAE is *worse* than the predict-zero config baseline (0.525 vs 0.426
W-mem; 0.510 vs 0.330 W-200). Per the preregistered reading: at seed
granularity the outcome is decided after the early window or by unobserved
state; the "trajectory determines fate" implicit reading of the early-warning
literature does not hold at the granularity its applications would need.

## P2 — Route readable before generalization (within-AdamW): **FALSIFIED — preregistered kill control triggered**

| route task (held-out config cells) | W-mem | W-200 |
|---|---|---|
| pooled, all optimizers | 0.674 (p=0.005) | 0.815 (p=0.005) |
| **within-AdamW (the preregistered quantity, n=72)** | **0.466 (p=0.44)** | **0.57 (p=0.12)** |

The pooled classifier clears the 0.8 bar only because growth-route ≈ Muon:
restricted to AdamW grok events (where 001 R5's λ-driven route switch lives),
route discrimination is chance-level at W-mem — the direction doc's
preregistered kill control (a) fires verbatim. Within a fixed optimizer the route is **not distinguishable from chance** in the
first 200 steps (W-200 AUROC 0.57, p=0.12; W-mem 0.466, p=0.44 — at n=72 a weak
signal ~0.6 cannot be excluded, so this is power-bounded, NOT "route proven not
imprinted"); "更新几何即时印刻路线" is at most bounded to what config identity
already tells you.

## P3 — Cross-task transfer: **FALSIFIED**

Trajectory-only grok prediction under CV-task (train mod-add/mul → test S5):
AUROC 0.451 (W-mem, below chance) / 0.673 (W-200, p=0.080 n.s.), versus the
config-only baseline's 0.834/0.640. The trajectory signal is task-local
parameterization; no transferable prediction law.

## P4 — Predictability structure = optimizer signature: **CONFIRMED (descriptive)**

Out-of-fold variance decomposition of log-delay (W-mem/tierA):

| optimizer × task | config share | signal share | residual |
|---|---|---|---|
| muon × add (n=32) | **0.966** | 0.0003 | 0.034 |
| adamw × add (n=62) | 0.703 | **0.182** | 0.115 |
| sgdm × add (n=26) | 0.000 | 0.000 | **1.263 (R²<0: linear fit FAILED, not a regime)** |

Preregistered expectations hold: Muon is config-saturated (the information-
theoretic restatement of 007's zero-variance signature — nothing left for any
signal to add), AdamW carries a real but unconvertible signal share, and for SGDM **the linear
predictor simply fails** (residual_share 1.263 > 1 ⇒ R²<0, the out-of-fold model
predicts worse than the mean) — this is a *failed fit on n=26*, NOT a discovered
"config-unpredictable regime", so no positive conclusion is drawn and it is **not**
used as C2 evidence.

## Methodological note (the audit catch)

The first analysis pass pooled all optimizers in the route task; the
within-AdamW restriction was added before any verdict was issued, after a
source-level audit of split semantics. Also: permutation p-values attach to raw
model scores (vs shuffled labels), not to Δ(signal − config); all deltas are
reported as point estimates. STAGE-2 (confirmatory grid) is **not triggered**
per the preregistered gate (requires P2/P3 positive); pre-X1's bank status and
立项门槛 are unaffected (the cost-share clause is moot).

## Limitations

- Corpus is observational and config-imbalanced (e.g., S5 grok events are
  Muon-only — the route/task split inherits this).
- TIER_B (mech channels) underpowered throughout (n≤30; nan-heavy splits).
- Window features are norm-channel summaries; richer per-step features could
  in principle revive P1/P2 — bounded, not impossible-in-principle.
- Single model family (linear); deliberate (the science is the comparison
  against config baselines, not model capacity).

## Figures

`results/figures-014/fig_spec_route.png` (grok deltas | P2 kill panel |
P1 MAE | P4 decomposition); raw: `results/spec_route/posthoc_round1.json`.
