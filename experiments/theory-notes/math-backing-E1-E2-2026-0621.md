# Mathematical / Statistical Backing for Papers E1 and E2

**Date:** 2026-06-21
**Scope:** Which conclusions of E1 (4–10 epoch free-repetition band; no ~100× capacity shift) and E2 (small-scale emergence calibration traps) admit *derived* mathematical/statistical support vs. which only admit *rationalization*.
**Method:** Read `papers/E1/main.tex`, `papers/E2/main.tex`, `experiments/findings-{013,016,014,011}.md`, and the actual run code (`experiments/icrl_td/{train_icrl.py,run_icrl.py}`, `experiments/icrl_td/probes.py`, `.omx/ultragoal-archive-.../calibrate_e2_icrl_td.py`). Empirical noise statistics pulled directly from the 45 `results/icrl_td/*.jsonl` logs.

**Verdict in one line:** E2's single-crossing false-positive calculation is *tractable now and quantitatively matches the observed data* — it is the strongest math item in either paper. E1's claims are mostly empirical falsification with a *legitimate but weak* power argument and an *information-theoretic story that mostly rationalizes* rather than predicts the R_free ≈ const result.

---

## Part A — E2 (the math-amenable paper)

### A.1 Single-crossing vs sustained-K false-positive rate — FULL WORKED DERIVATION

**This is the headline result and it is fully tractable with E2's real parameters.**

#### A.1.1 The actual eval parameters (from code, not guessed)

From `experiments/icrl_td/train_icrl.py`:

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `N` = eval_mrps | held-out sequences per evaluation | **32** | `Config.eval_mrps = 32` (line 54) |
| `steps` | training steps | 10000 | line 52 |
| `eval_every` | eval cadence | 50 | line 53 |
| `M` | eval points per run = steps/eval_every + 1 | **201** | confirmed in logs (min/med/max = 201) |
| `τ` (acc_thresh) | detector threshold | **0.7** (also 0.8 in audit table) | line 55; `calibrate_e2_icrl_td.py` THRESHOLDS=[0.7,0.8] |
| `p` | baseline (chance) per-eval accuracy | **≈ 0.562–0.571** | findings-016 median 0.562; logs per-run mean 0.576 |
| runs | grid size | 45 = 3 opt × 3 T × 5 seed | `run_icrl.py` |
| `V_BUCKETS` | value classes (why chance ≠ 1/9) | 9, but value dist is concentrated → modal-bucket baseline ≈ 0.56 | `data.py:24` |

**Empirical noise check (pulled from the 45 logs):** within-run std of `val_acc` is median **0.089** (range 0.079–0.097); pooled mean 0.571, pooled std 0.104; fraction of all evals ≥0.7 is 0.099, ≥0.8 is 0.010.

#### A.1.2 The model

Each evaluation reports an accuracy that is the mean of `N=32` per-sequence correct/incorrect outcomes. Under the null "the model sits at baseline accuracy p with no real transition," each eval is

  acc_t = (1/N) Σ Bernoulli(p),  so  N·acc_t ~ Binomial(N, p).

The per-eval standard error is **σ = √(p(1−p)/N)**. Plugging p=0.562, N=32:

  σ = √(0.562·0.438/32) = √0.00769 = **0.0877.**

This is the load-bearing sanity check: the *predicted* binomial σ = 0.0877 matches the *measured* within-run σ = 0.089 to within 2%. The Bernoulli/Gaussian model of the readout is therefore not an assumption — it is empirically validated by the logs. (Theoretical σ at p=0.571 is 0.0875; still matched.)

#### A.1.3 Per-eval crossing probability

Single-eval probability of exceeding threshold τ (exact binomial; need k ≥ ⌈τN⌉ of N correct):

  q(τ) = P(Binom(N,p) ≥ ⌈τN⌉).

| τ | ⌈τN⌉ / N | z=(τ−p)/σ | q (exact Binomial, p=0.562) | q (Gaussian approx) |
|---|---|---|---|---|
| 0.7 | 23 / 32 | 1.57 | **0.0517** | 0.0578 |
| 0.8 | 26 / 32 | 2.71 | **0.00272** | 0.00333 |

(Empirical pooled fraction ≥0.7 = 0.099 and ≥0.8 = 0.010 — somewhat above the iid-binomial q because real trajectories have positive autocorrelation and a handful of genuinely-elevated cells; this *strengthens* the paper's point, since the noise floor is if anything fatter than the clean model.)

#### A.1.4 Single-crossing detector false-positive rate per run

A "fires at least once in the run" detector over M independent evals:

  P_single(τ) = 1 − (1 − q(τ))^M.

| τ | q | P_single per run (M=201) | Expected #cells flagged (×45) | **Observed** |
|---|---|---|---|---|
| 0.7 | 0.0517 | **1 − (1−0.0517)^201 ≈ 1.000** | 45.0 | (detector at 0.7 fires broadly; consistent) |
| 0.8 | 0.00272 | **1 − (1−0.00272)^201 = 0.421** | 18.9 | **26/45** (calibrate_e2 single_cross_0p8) |

The single-crossing detector is *almost guaranteed* to fire at τ=0.7 (P≈1) and fires in ~40–50% of pure-noise runs at τ=0.8. The model predicts **~19/45** at τ=0.8; the data show **26/45**. The under-prediction is expected and explainable (autocorrelation lengthens excursions; the pooled tail above is fatter than binomial; a few cells genuinely peak — max observed 0.875–0.938). The order of magnitude and the qualitative claim ("single crossing fires on noise") are exactly reproduced.

#### A.1.5 Sustained-K detector suppresses it (the fix)

Require τ to be held for K consecutive evals. The probability a run contains at least one K-length run of crossings, in M iid Bernoulli(q) evals, is computed by the exact run-length recurrence (Markov chain on consecutive-success count). Results:

| τ | K | P_sustain per run | Expected #cells (×45) | **Observed** |
|---|---|---|---|---|
| 0.7 | 2 | 0.400 | 18.0 | — |
| 0.7 | 5 | 6.9×10⁻⁵ | 0.003 | — |
| 0.8 | 2 | **0.00147** | **0.066** | **0/45** (sustain2_0p8) |
| 0.8 | 5 | 2.9×10⁻¹¹ | ~0 | **0/45** (sustain5_0p8) |

So the calculation predicts the paper's exact audit row: at τ=0.8, K=2 expected ≈ 0.07 cells ≈ **0**, K=5 ≈ 0 — and the data give **0/45 and 0/45**. The suppression factor from single→sustained-2 at τ=0.8 is P_single/P_sustain2 ≈ 0.421/0.00147 ≈ **285×**. This is a clean, closed-form explanation of why a single-crossing emergence detector manufactures phantom emergence and why a sustained criterion removes it.

**The key inequality (the takeaway the paper can state as a theorem-flavored claim):** for a noisy readout at baseline p with per-eval crossing prob q over M evals,

  P_single ≈ 1 − (1−q)^M  ≫  P_sustain-K ≈ (M)·q^K   (for small q, K≥2),

i.e. single-crossing risk grows *linearly in M* (number of eval points) while sustained-K risk is *suppressed by q^(K−1)*. More evaluations make a single-crossing detector worse; sustained-K is nearly M-independent. This also explains findings-016's "emergence rate rises with horizon T" artifact: larger T → noisier eval estimate → larger σ → larger q → more single-crossing false positives, with **zero** change in real computation.

**FEASIBILITY: Tractable now. Worth it: YES (highest value item).** **RISK: low.** Two honest caveats to state in-paper: (1) the iid-across-evals assumption slightly under-counts (autocorrelation), so present P_single as a *lower bound* on the noise-driven count and note the data exceed it; (2) p is the empirical baseline (≈0.56), not 1/9 — state this explicitly so a reader does not object that "chance = 0.11."

---

### A.2 AUROC collapse: pooled-vs-within as a Simpson / mixture statement

**Claim (Result 3):** pooled route AUROC = 0.815 but within-AdamW AUROC = 0.47–0.57; the early window fingerprints the *configuration* (optimizer), not the *run*.

**Formalization (tractable, short):** Let the early-window feature be X and the binary route label be Y. Suppose the population is a mixture over optimizer groups g with mixing weights π_g, and within each group X carries (near) zero information about Y: AUROC(X→Y | g) ≈ 0.5. If optimizer identity g is *itself* correlated with both X (different optimizers leave different early-trajectory signatures) and Y (different optimizers land in different routes), then the pooled AUROC

  AUROC_pooled = P(X_i > X_j | Y_i=1, Y_j=0)

is inflated purely by *between-group separation*: when a positive case is drawn predominantly from group A (high-X, route-1) and a negative from group B (low-X, route-0), X separates them with no within-group signal at all. This is the ranking-statistic form of Simpson's paradox / the classic "AUC is not collapsible." A two-group worked bound: if groups are perfectly X-separated and route-imbalanced across groups, pooled AUROC can approach max(π-weighted between-group concordance) ≫ 0.5 while every within-group AUROC = 0.5.

**Status:** This is a *correct and standard* formalization (AUC non-collapsibility is established; e.g. the mixture/Simpson framing). It is **Plausible-with-work** to make a tight, quoted bound that reproduces 0.815 from the observed group sizes and per-group route base-rates — that needs the 014 group/label tallies (`results/figures-014`), which were not opened here. **Worth it: medium** — it converts a verbal warning into a checkable identity, but the paper's empirical pooled/within split already carries the argument. **RISK: medium** — to *predict* 0.815 you must plug in real per-group base rates; if they don't reproduce 0.815 the quantitative claim weakens (though the qualitative non-collapsibility statement is safe regardless). Also honest caveat (already in the manuscript's Limitations): within-AdamW n=72 gives W-200 AUROC 0.57 with p≈0.12 — the within-group "chance" is itself *power-bounded*, so the formalization should say "consistent with chance," not "proven chance."

---

### A.3 Floor-censoring: coarse eval grid censors fast transitions

**Claim (discipline #2, Result 1 mod-add arm):** a coarse eval grid reads a fast transition as "instant"/"zero-variance."

**Formalization (tractable sampling argument):** If the true transition occurs at step T* with the metric crossing within a window of width w around T*, and evaluations occur every Δ steps, then the *measured* onset is quantized to the grid: measured_onset = Δ·⌈T*/Δ⌉. If the entire transition fits inside one inter-eval interval (w < Δ) and completes before the first post-T* eval, the first eval already shows the post-transition value → apparent delay = 0 ("floored") and apparent variance across seeds = 0 if all T* fall in [0, Δ). Concretely: P(transition invisible / floored) = P(T* + w ≤ first eval after 0) = P(T* < Δ − w). With induction emergence at ~1850 steps and Δ=100 (`induction_emergence` EVAL_EVERY=100) the induction arm is *not* floored (1850 ≫ 100), but a mod-add arm that groks before step Δ is fully censored. The quantization error on any measured onset is uniform on [0, Δ), giving a floor-induced bias of Δ/2 and a spurious zero-variance whenever the seed-spread of T* is < Δ.

**Status: Tractable now** as a quantization/sampling statement; **worth it: low–medium** (it is a standard measurement argument, useful as one or two sentences with the Δ values, not a centerpiece). **RISK: low.** This mainly *frames* an empirical control (run a fine-Δ arm) rather than proving anything deep — be honest that it justifies the control, it is not itself the evidence.

---

## Part B — E1 (mostly empirical; weaker math)

### B.1 An information-theoretic / capacity argument for R_free, and why no ~100× shift

**What the paper claims:** R_free ∈ [4,10] reproduces ~1000× below LLM scale and does NOT shift ~100× with the capacity/entropy ratio; damage onset = memorization onset.

#### B.1.1 What a capacity argument *can* legitimately say

A defensible (but only semi-quantitative) derivation: repetition is "free" while the model has not yet *memorized* the unique set, i.e. while the information it has fit is bounded by the unique-data entropy rather than by the per-token identity of repeated examples. Let U = unique tokens, H = per-token entropy (nats), so unique information content ≈ U·H. Let C = model capacity in nats (a fraction of parameter count, C ≈ κ·P with κ of order a few bits/param). Memorization onset — the point where the model starts fitting the *identity* of repeated tokens beyond their generative entropy — occurs when the optimizer has had enough gradient exposure to drive train loss below the entropy floor, which empirically tracks **epochs**, not raw capacity, once C ≳ U·H. Because the budget B = U·n is fixed, "free epochs" R_free is governed by *how many passes it takes to begin overfitting the unique set*, which is a property of the optimization dynamics + the U·H/C ratio — and crucially is **logarithmically weak** in C.

This is the crux of "why no 100× shift": if memorization-onset epoch scales like a slowly-varying (log or low-power) function of the capacity-to-entropy ratio C/(U·H), then a 4× change in capacity (the paper's actual span: 2.5M→10M params) and a >10× change in entropy move R_free by at most a small multiplicative factor — consistent with the observed 4→10 (2.5×), NOT the 100× that a *strong linear* "free-epochs ∝ C/(U·H)" hypothesis predicts. In other words, the artifact hypothesis is the strawman "R_free scales linearly/steeply with the ratio"; the data falsify *that specific functional form*.

#### B.1.2 Is there a derivation predicting R_free ≈ const (4–10)?

**Honest answer: NO clean first-principles derivation pins the constant at 4–10.** What is derivable is a *direction and a weak slope*: R_free should be (i) increasing in capacity headroom, (ii) decreasing in entropy/U, (iii) and only *logarithmically/weakly* so — which predicts "roughly constant, small drift," matching the data, but does not predict the number 4–10. The empirical anchors confirm the weak dependence: findings-013 shows the only modulation is mild and capacity-side (small models tolerate slightly more repetition, R_free 10 vs 4), exactly the sign B.1.1 predicts (smaller C → ... here the sign is *opposite* to naive C-headroom and instead matches "smaller model memorizes the unique set more slowly," a dynamics argument, not a capacity-headroom argument — note the tension below).

**Tension to be honest about:** findings-013's own explanation ("small-capacity models tolerate MORE repetition because they memorize slower") is a *dynamics/rate* argument, and it points the *opposite way* from a naive information-capacity-headroom argument (more capacity → more room before saturation → more free epochs). So the capacity story does not cleanly close: the data say smaller capacity → more free epochs in 2 cells. The only self-consistent reading is the rate/dynamics one (memorization onset is set by optimization passes over the unique set, weakly modulated by size), and that is *not* an information-theoretic capacity bound — it is a learning-dynamics claim that this analysis cannot derive from first principles. **Mark this as rationalization, not derivation.**

**FEASIBILITY: Plausible-with-work** for the *weak-dependence / no-strong-shift* framing (you can write down R_free = f(C/(U·H)) with df/d(ratio) small and show the artifact hypothesis assumed a steep f); **Hard/Open** for deriving the constant 4–10. **Worth it: medium** — the weak-slope framing genuinely strengthens P2 by naming the strawman functional form the data kill; do not oversell it as predicting the constant. **RISK: high if overstated** — a referee will correctly note (a) no derivation of 4–10, (b) the small-capacity-tolerates-more sign contradicts a naive capacity argument, (c) "entropy" is construction-controlled not measured. Keep it as "the data refute a steep R_free(ratio); they are consistent with a weak/log dependence," which is true and defensible.

### B.2 Decay-onset == memorization-onset coincidence — is it *necessary*?

**Claim (P3):** excess-loss onset coincides with canary-memorization onset (7/8 cells).

**Can it be argued mathematically that they MUST coincide?** Partially. Excess val loss = val_loss(repeated) − val_loss(fresh) rises precisely when the model starts fitting features of the *repeated* set that do not generalize — which is operationally what the copied-canary gap measures (train-only memorization). So there is a near-tautological coupling: generalization gap on repeated data and memorization of repeated-only content are two readouts of the *same* event (the model allocating capacity to instance identity rather than the generative distribution). A clean statement: under a fixed generator, val_loss(fresh) is invariant to n; therefore excess(n) is entirely driven by the train-distribution shift induced by repetition, whose *only* non-generalizing component is memorization of the finite repeated set. Hence onset(excess) ≥ onset(memorization is detectable) is expected, and exact coincidence on a coarse grid (both quantized to the same {4,10,20} bracket) is the generic outcome.

**Status:** This is a *legitimate "two-readouts-of-one-event" argument*, **Tractable-now** as a qualitative inequality, but it is **not a theorem** — there is a logically possible regime where excess loss rises from optimization interference (e.g. repeated-data double-descent) *without* canary memorization; the paper itself notes the non-monotone double-descent falsifier did not fire, which is the empirical guard. The 7/8 (not 8/8) and the coarse-grid quantization mean "coincide within one eval bracket," which weakens "must coincide" to "co-quantize." **Worth it: medium** (a one-paragraph mechanistic argument). **RISK: medium** — do not claim causality (the manuscript correctly already flags "onset coincidence is mechanistic evidence, not proof of causality"). Honest: the math says *coupling is expected*, not *forced*.

### B.3 Is E1's NULL a properly-powered falsification? (the grid-resolution argument)

**This is E1's most defensible statistical point and it is real.** The artifact hypothesis predicts a ~100× shift in R_free. The grid resolves R_free on the ladder n ∈ {1,2,4,10,20,40}. A ~100× shift from a base of ~4 would land R_free at ~400 epochs (off the grid entirely) or, in the other direction, at ~0.04 (below n=1). The grid spans 1→40 (40×). So:

- **The grid CAN detect any shift that moves R_free outside [4,10] up to the n=40 ceiling** — and a 100× shift would be *unmissable* (it would push the cliff past n=40 in every cell, i.e. "still free at 40 epochs," or pull it below n=2). Neither happened: all cells break by n=20–40.
- **Power statement:** the falsification is properly powered *against the specific alternative it names* (a ~100× multiplicative shift), because that alternative predicts an effect (≥10× change in the break-point) far larger than the grid's resolution floor (the grid distinguishes 4 vs 10 vs 20 vs 40, i.e. ~2.5× steps). The observed spread is 4–10 (2.5×) — within one-to-two grid steps — versus a predicted 100×. So this is a *large-effect-size* falsification, the kind that is robust to coarse resolution.

**Caveat that limits the power (be honest):** (1) the large-capacity U-ladder is now fully covered (findings-013, refreshed 2026-06-29), so the *capacity* axis spans the full ~4×; even so the 100×-shift falsification remains strongest along *entropy* (>10× span) and the capacity span (~4×) is the narrower axis. (2) R_free is a threshold (0.05 nats) on a coarse grid → the true R_free is only bracketed to (4,10) for most cells, and the real-text bridge is threshold-sensitive (2/3 seeds would move to 10 at a 0.1-nat threshold). (3) No formal hypothesis test / CI is attached to "R_free"; the falsification is effect-size-vs-resolution reasoning, not a p-value.

**FEASIBILITY: Tractable now** to write the resolution/effect-size argument formally (predicted shift ≥10× grid step vs observed ≤1 grid step). **Worth it: YES** — this is the cleanest way to answer "is the null real or just unfound": frame it as *the named alternative predicts an effect 1–2 orders of magnitude above the detection floor, and it is absent*. **RISK: medium** — must scope it to entropy (fully powered) vs capacity (under-powered due to coverage gap); a referee will catch the large-capacity gap if it is hidden. A small honest add: state the grid's detectable-shift range explicitly ("a true R_free anywhere in ~2–40 epochs would have been seen; only a shift entirely past 40 or below 2 could hide").

---

## Per-claim summary tables

### E1

| Claim | Math available? | Feasibility | Worth it | Risk | Honest note |
|---|---|---|---|---|---|
| R_free ≈ const (4–10), no 100× shift | Weak-dependence framing only; constant 4–10 NOT derivable | Plausible-with-work (framing); Hard (constant) | Medium | High if overstated | Capacity headroom argument's sign is contradicted by "small models tolerate more"; the real argument is learning-dynamics, = rationalization |
| Grid is properly-powered falsification | YES — effect-size vs resolution | Tractable now | **YES** | Medium | Powered on entropy (>10×, full), under-powered on capacity (coverage gap); no formal CI |
| decay-onset == memorization-onset | "Two readouts of one event" inequality | Tractable now (qualitative) | Medium | Medium | Coupling expected, not forced; coarse grid = "co-quantize," not "coincide"; no causality |
| No double-descent (monotone decay) | n/a (empirical) | — | — | — | Empirical falsifier, no derivation needed |
| Real-text bridge | n/a (empirical, threshold-sensitive) | — | — | — | Keep as bridge; threshold-sensitive at 0.1 nat |

### E2

| Claim | Math available? | Feasibility | Worth it | Risk | Honest note |
|---|---|---|---|---|---|
| Single-crossing fires on noise; sustained-K suppresses | **YES — full closed-form, matches data** | **Tractable now** | **YES (top item)** | Low | Binomial σ=0.088 matches measured 0.089; predicts ~19/45 single (obs 26), ~0/45 sustained (obs 0/45); 285× suppression |
| AUROC pooled≫within (Simpson/mixture) | Standard non-collapsibility statement; quoting 0.815 needs 014 tallies | Plausible-with-work | Medium | Medium | Qualitative form safe; quantitative bound needs group base-rates; within-AdamW is power-bounded (n=72) |
| Floor-censoring of fast transitions | Quantization/sampling argument | Tractable now | Low–Medium | Low | Justifies the fine-Δ control; not itself evidence |
| Order does not compress delay (Result 1) | n/a (Welch t already reported) | — | — | — | Already statistical (t≈2.25, p≈0.05, n=5); honest "borderline" |
| TD trainable-control upper bound 0.162 | Already a Clopper-Pearson-style 95% bound (0/17) | Tractable now (already done) | Low (done) | Low | 0/17 → 95% UB ≈ 0.162 is a correct rule-of-three-ish bound |

---

## Top recommendations (prioritized)

1. **[E2, highest impact, low effort] Add the single-crossing false-positive derivation (A.1) as a short formal subsection or appendix.** It is fully tractable, uses the paper's *actual* N=32 / M=201 / τ / p, and the binomial σ=0.088 matches the measured 0.089 — converting "the detector fires on noise" from assertion to derivation, and reproducing the 26/45 → 0/45 audit numbers within explainable error. State P_single ≈ 1−(1−q)^M vs P_sustain-K ≈ M·q^K as the key inequality and connect the "rate rises with T" artifact to σ growing with horizon. Caveat the iid assumption (present P_single as a lower bound; data exceed it).

2. **[E1, high impact, low effort] Formalize the grid-resolution power argument (B.3)** as "the named alternative (~100× shift) predicts an effect 1–2 orders of magnitude above the grid's detection floor (2.5× steps, 40× span); it is absent." Scope explicitly to entropy (fully powered) vs capacity (coverage-gap-limited). This is the honest answer to "is the null real."

3. **[E2, medium] Add the one-paragraph Simpson/non-collapsibility formalization (A.2)**; only attempt the quantitative 0.815 reproduction if the `results/figures-014` group base-rates are pulled and actually reproduce it — otherwise keep it qualitative.

4. **[E1, medium] Keep the capacity/decay-onset arguments as mechanistic framing, clearly labeled as rationalization, not derivation.** Do NOT claim a derivation of the constant 4–10, and reconcile the "small models tolerate more repetition" sign explicitly (it is a dynamics, not a capacity-headroom, effect).

## Trade-offs

| Option | Pros | Cons |
|---|---|---|
| Add A.1 derivation to E2 | Turns the paper's central diagnostic into a theorem-flavored, data-matched result; cheap; low risk | Must caveat iid/autocorrelation honestly or a referee bites |
| Add B.3 power argument to E1 | Directly answers "is the null real"; large-effect-size logic is robust to coarse grid | Exposes the large-capacity coverage gap if scoped honestly (better to disclose than hide) |
| Push the E1 capacity derivation hard | Would be a strong theoretical contribution if it worked | It does not work cleanly (sign tension, constant not derivable); high referee risk |

## References (file:line)

- `papers/E1/main.tex:18-44` — abstract / R_free band + no-100×-shift claim; `:166-174` P2 falsification; `:176-182` P3 onset coincidence; `:227-244` evidence ledger
- `papers/E2/main.tex:90-113` — Result 2 detector-fires-on-noise; `:141-160` calibration audit table (single 26/45, sustained 0/45); `:162-177` Result 3 AUROC collapse; `:198-200` floor-censoring discipline; `:238-247` 0/17 → 0.162 bound; `:250-252` within-AdamW power bound
- `experiments/icrl_td/train_icrl.py:54` — eval_mrps=32 (N); `:52-53` steps=10000, eval_every=50 (M=201); `:55` acc_thresh=0.7; `:118-128` evaluate() acc = fraction of N correct
- `experiments/icrl_td/probes.py:95-101` — emergence_step detector (note: code already has a weak 2-eval check; paper's "single crossing" = K=1 audit in calibrate script)
- `experiments/icrl_td/data.py:24` — V_BUCKETS=9 (why naive chance≠observed baseline 0.56)
- `.omx/ultragoal-archive-20260618-0830-complete/calibrate_e2_icrl_td.py:35-40,103-110` — first_k() K-sustained counts; THRESHOLDS=[0.7,0.8], KS=[1,2,3,5] → the 26/45, 0/45 numbers
- `experiments/results/icrl_td/*.jsonl` (45 files) — empirical: M=201, within-run σ(val_acc)=0.089 (matches binomial 0.088), pooled mean 0.571 std 0.104, P(≥0.7)=0.099, P(≥0.8)=0.010
- `experiments/findings-013.md:42-63` — P2 falsification + small-capacity-tolerates-more sign; `:65-75` P3 7/8; `:86-91` large-capacity coverage gap
- `experiments/findings-016.md:13-23` — median 0.562, 1/45>0.8, detector-fires-on-noise, "rate rises with T" artifact

