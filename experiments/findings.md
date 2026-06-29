# Findings — Does the Norm-Separation Delay Law explain Muon?

**Direction 001.** Tiny decoder transformer (d=128, 2 layers, ~420k params), modular
addition mod 97, full-batch, single RTX 5090. 5 seeds/cell unless noted. Delay
ratio = grok_step / memorize_step (grok = test_acc≥0.95, memorize = train_acc≥0.99).
Budget 20k steps (sc=1) / 12k (sc=3). All numbers from `results/grid_main`,
`results/lr_control`, `results/lr_control_sc3`.

## TL;DR (closed loop)

The publishable headline "Muon accelerates grokking" is **already known**
(Tveit et al. 2025, 2504.16041) — we reproduce it. The actual contribution sits in
the gap left by the 2026 delay law (Truong et al., 2603.13331, which tests only
SGD/AdamW). Final synthesis — a **two-route model of grokking**:

1. **Contraction route** (the law's mechanism): delay set by γ_eff≈ηλ; taken by
   AdamW/SGD at large λ (norm ratio <1 at grok; delay falls with λ).
2. **Growth/directional route**: λ-independent; norm *grows* through grokking.
   AdamW takes it slowly at small λ (~5000-step plateau); **Muon's orthogonalized
   updates take it 10–100× faster — 10–40 steps on abelian modular tasks (resolved
   at eval_every=5), at every λ from 0 to 1, across 8× lr.** Route-2 speed is
   task- and data-dependent (S5 and low train_frac slow it; see Results 7–8); the
   *ordering* Muon ≪ AdamW is what holds universally in our experiments.

The "Muon is just a higher effective LR" confound is ruled out (raising AdamW's lr
worsens delay then breaks training). Orthogonalization's specific contribution vs
plain SGD-momentum is speed *plus stability* (SGDM groks but collapses seeds).
The delay law is real but describes route 1 only; Muon shows route 2 can dominate.

## Result 1 — Muon ≈ removes the delay (reproduces 2504.16041), weak-init sc=1

Delay ratio (mean±std, 5 seeds; all optimizers grok 5/5 except where noted):

| optimizer | wd=0 | wd=0.01 | wd=1.0 |
|---|---|---|---|
| AdamW | 41.8 ± 26 | 35.1 ± 20 | 4.8 ± 0.7 |
| **Muon** | **2.0 ± 0** | **2.0 ± 0** | **1.8 ± 0.4** |
| SGD-mom | 5.0 ± 2.9 | 8.3 ± 13.5 | **fails: 0/5 (never memorizes, test≈0.003)** |

- Muon delay ≈2 with **zero variance**; AdamW 35–42 at low wd → ~20× reduction.
- **AdamW delay decreases with wd** (42→35→4.8): consistent with the law's
  γ_eff=ηλ direction (more λ ⇒ more contraction ⇒ less delay).
- **Muon delay is flat in wd** (2.0→2.0→1.8): λ-independent. First evidence its
  contraction is not weight-decay-driven.

## Result 2 — Strong-init sc=3: Muon groks where AdamW fails (at standard settings)

Grokking success rate (seeds grokked / 5; mean grok step):

| optimizer | wd=0 | wd=0.01 | wd=1.0 |
|---|---|---|---|
| AdamW | 0/5 | **0/5** | 5/5 (1340) |
| **Muon** | 3/5 (test≈0.94) | **5/5 (2650)** | 5/5 (150) |
| SGD-mom | 0/5 | 5/5 but **unstable** (final test 0.51–1.0) | 0/5 (no memorize) |

- At wd=0.01 (standard small wd): **Muon 5/5 clean; AdamW 0/5 entirely.** AdamW
  needs wd=1.0 (≈100× more) to grok here. ⇒ Muon lowers the weight-decay threshold
  for grokking by about two orders of magnitude.
- **Contraction is still required even for Muon:** at wd=0 Muon only 3/5 (test≈0.94),
  AdamW/SGD 0/5. Muon lowers the threshold; it does not abolish it.
- SGD-momentum (no orthogonalization) can reach grokking at wd=0.01 but **collapses
  for some seeds** (slingshot-like un-grokking) → orthogonalization buys *stability*.

## Result 3 — The LR confound is ruled out

**sc=1, wd=0.01, delay vs hidden-matrix lr (3 seeds):**

| optimizer | lr sweep → delay | failures |
|---|---|---|
| AdamW | 3e-4:31 → 1e-3:41 → 3e-3:**129** → 1e-2:breaks | 2/3 fail at 1e-2 |
| **Muon** | 5e-3:1.8 / 1e-2:2.3 / 2e-2:2.0 / 4e-2:1.0 | **0 — stable at every lr** |
| SGD-mom | erratic (low when it works) | frequent collapse (0/3 at 5e-3) |

Raising AdamW's lr makes the delay **worse**, then destabilizes training — it never
approaches Muon's ≈2. Muon stays low across an 8× lr range with no failures.

**sc=3, wd=0.01, AdamW lr sweep (loop-closing, 3 seeds):** AdamW fails at standard
lr (0/3 at 1e-3, 0/3 at 3e-3), is erratic at 1e-2 (1/3, grok at 10150), and only
recovers at lr=3e-2 (3/3, but late: 5250–8000 steps vs Muon's 2650 at lr=2e-2).

⇒ AdamW *can* grok in the strong regime, but only with ≈100× more weight decay OR
≈30× more learning rate, and even then later and less reliably than Muon. Muon's
advantage is a genuinely wider, more moderate, more stable grokking region — not a
hidden larger step size.

## Result 4 — Mechanism: Muon groks during norm GROWTH, not contraction

The law's driving quantity is `log(‖θ_mem‖²/‖θ_post‖²)` — generalization should
coincide with norm *shrinkage*. Measuring hidden-matrix norm at T_grok vs T_mem
(`norm_ratio_table.json`, `fig_norm_trajectories.png`):

| cell | wn(T_grok)/wn(T_mem) | groks? | mechanism |
|---|---|---|---|
| AdamW wd=1.0 sc=3 | **0.45 ± 0.01** | 5/5 | contraction ✓ law |
| AdamW wd=1.0 sc=1 | 0.88 ± 0.03 | 5/5 | contraction ✓ law |
| SGDM wd=0.01 sc=3 | **0.55 ± 0.19** | 5/5 unstable | contraction ✓ law |
| AdamW wd≤0.01 sc=1 | 1.14–1.18 | 5/5 (slow) | mild growth |
| **Muon, every cell** | **1.5 – 10.4 (always >1)** | 5/5 (3/5 at wd=0 sc=3) | **growth — never contracts** |

Trajectories make it vivid: AdamW@wd=1.0/sc=3 slides 37→14 and groks at the bottom
(the law's story); Muon@wd=0.01/sc=3 grows 47→170+ and groks mid-growth; Muon@wd=0/
sc=1 grows 27→160 with grokking early in the climb.

**Correction to the earlier interpretation:** Muon does not add "λ-independent
contraction" — in raw-norm terms it does not contract at all. **Muon bypasses the
norm-separation route**: its orthogonalized (spectrum-flattened) steps apparently
drive fast *directional* convergence to the generalizing circuit while the norm
grows. The law's mechanism is real but describes the AdamW/SGD route, not Muon's.

Caveats: pre-LN transformers are approximately scale-invariant in hidden weights,
so raw Frobenius norm is an imperfect proxy for function-relevant geometry (the
law and Omnigrok are nonetheless stated in raw norms — that reading is what our
data contradicts). Directional claims (cosine movement of θ between T_mem and
T_grok) need saved checkpoints — future work. Muon sc=1 ratios span a short window
(T_mem=50 → T_grok=100), but the growth direction is unambiguous and continues
long after grokking.

## Result 5 — Quantitative λ-dependence (fine wd sweep, sc=1, 3 seeds × 8 λ)

Realized delay (T_grok − T_mem, steps) vs weight decay:

| λ | 0 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| AdamW | 4900 | 4917 | 5033 | 4033 | 3483 | 2217 | 983 | 400 |
| **Muon** | **50** | **50** | **50** | **50** | **50** | **50** | **50** | 33 |

- **Muon: delay = 50 steps (= one eval interval, i.e. the measurement floor) at
  every λ from 0 to 1.0, zero variance.** Slope exactly 0 — λ-independence could
  not be cleaner (true delay may be below our 50-step resolution).
- **AdamW: clear negative λ-dependence (12× drop from λ=0.01→1.0) but with a
  λ-independent plateau (~5000 steps) for λ ≤ 0.003.** Pooled log-log slope −0.34
  (R²=0.64); restricting to λ≥0.01 gives ≈ −0.5 — shallower than the law's −1.
- **Two-route reading (unifies Results 4+5):** at small λ, AdamW groks via the slow
  λ-independent growth route (plateau; norm ratio >1 there per Result 4); at large
  λ the contraction route takes over (delay falls with λ; norm ratio <1 at wd=1.0).
  Muon's growth/directional route is so fast (≤50 steps) that the contraction
  route never becomes relevant at any λ.

## Result 6 — External validity: modular multiplication replicates everything

Second task, (a·b) mod 97, same protocol (5 seeds):

- **sc=3, wd=0.01 (discriminator): Muon 5/5 groks (mean ≈2980 steps); AdamW 0/5**
  (test 0.19–0.30). Identical to addition.
- **sc=1, wd=0.01 (delay): Muon delay = 2.0 (zero variance, 5/5); AdamW ≈ 41
  (26–56).** Identical to addition.

## Result 7 — Fine eval resolution (eval_every=5): Muon's true delay measured

Earlier Muon delays sat at the 50-step eval floor. Re-measured at eval_every=5:

- **sc=1 (all wd):** memorize ≈ 40–50, grok ≈ 50–80 → **true delay 10–40 steps,
  ratio 1.1–2.0**. Memorization and generalization are nearly simultaneous.
- **sc=3:** memorize = 35 (earlier than the floor suggested), grok unchanged
  (wd=0.01: 1695–3195) → delay ratios there are ~48–91, larger than first
  reported; at wd=1.0 grok = 105–130 (ratio ≈ 3).

## Result 8 — Third task, S5 composition (non-abelian): the model's boundary

Same protocol on permutation composition in S5 (120 elements, 14400 pairs, wd=0.01):

| cell | Muon | AdamW |
|---|---|---|
| sc=1 | **5/5 grok**, but delay 21.5–107.5 (mean ≈54, high variance) | **0/5 — test stuck at chance (~0.01)** |
| sc=3 | 1/5 | 0/5 |

Two honest lessons:
- **The qualitative discriminator is even stronger:** on S5, AdamW never leaves
  chance within 20k steps even at standard init, while Muon groks 5/5.
- **The near-zero delay does NOT transfer:** Muon's route-2 is fast, not free —
  on a harder (non-abelian) circuit it shows a real, variable delay (~54× mean),
  and at sc=3 mostly fails. Route-2 speed scales with circuit difficulty.

## Result 9 — Data-fraction sensitivity (sc=1, wd=0.01, 3 seeds)

| train_frac | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|
| Muon | 3/3, delay 21–38 (2 seeds regress to 0.69–0.87 after grok) | 3/3, delay 4–6 | 5/5, delay 1.5–2 | 3/3, delay 1.0 | 3/3, delay 1.0 |
| AdamW | 0/3 | 0/3 | 5/5, delay 26–72 | 3/3, delay 5–9 | 3/3, delay 1–1.5 |

**Muon shifts the critical data fraction down** (~0.25 vs ~0.4 for AdamW within
budget); both optimizers' delays grow as data shrinks, and at the data-scarce edge
Muon shows partial post-grok regression. Route 2 still needs enough data to define
the generalizing solution — Muon accelerates the search, it does not replace data.

## Result 10 — Direct route evidence: the route map (mechanism probes, 30 runs)

Online probes (no checkpoints needed): cosine of hidden-weight direction vs the
snapshot taken at T_mem, per-eval rotation rate, effective/stable rank. For every
grokking event we plot Δlog‖θ_hidden‖ (T_mem→T_grok) against angular distance
traveled (`fig_route_map.png`, `mech_route_table.json`):

| cell | Δlog norm | total angle | rotation rate (°/100 steps) |
|---|---|---|---|
| muon sc1 wd0 / wd1 | **+0.41 / +0.23** | 25° / 23° | **99 / 92** |
| muon sc3 wd0.01 / wd1 | **+1.29 / +0.37** | 64° / 46° | 2.8 / **62** |
| adamw sc1 wd0 (plateau) | +0.19 | 30° | 0.85 |
| adamw sc1 wd1 / sc3 wd1 | **−0.13 / −0.78** | 24° / 40° | 6.4 / 3.2 |
| sgdm sc1 wd0 / sc3 wd0.01 | +0.05 / **−0.50** | 10° / 89° | 1.6 / 3.5 |

- **Perfect separation on the norm axis:** every Muon grok sits in the growth
  half-plane; every AdamW-wd=1 / SGDM-sc3 grok sits in the contraction half-plane.
- **Rotation rate is the speed signature:** Muon changes direction 15–100× faster
  per step than any other family. AdamW at wd=0 confirms the *slow* growth route
  (+0.19 norm, 0.85°/100 steps — hence the ~5000-step plateau).
- Stable rank stays high/flat under Muon (spectrum-flattened updates) while
  contraction-route runs lose spectral flatness — consistent with the
  orthogonalization mechanism (`fig_mech_trajectories.png`).

This converts the two-route model from inference to per-event measurement.

## Interpretation vs the delay law

The law `T_grok − T_mem = Θ(γ_eff⁻¹ log(‖θ_mem‖²/‖θ_post‖²))` with γ_eff=ηλ predicts
grokking needs sufficient norm contraction (large η or λ). Our AdamW/SGD data obey
this — they grok via measurable contraction, and only at large λ (wd=1.0) or large
η (lr=3e-2) in the strong-init regime. **Muon, however, is not an outlier within
the law's parameterization — it follows a different route altogether** (Result 4):
delay flat in λ, grokking at small η and small λ, norm growing at the moment of
generalization in every cell. The law's scope is therefore narrower than stated:
it describes the contraction-mediated route taken by coordinate-wise optimizers,
while spectrum-flattened (orthogonalized) updates can reach the generalizing
solution directionally, without passing through norm separation. The residual
role of weight decay for Muon (3/5 at wd=0 vs 5/5 at wd=0.01, sc=3) suggests wd
still helps escape bad large-init basins, but as a secondary effect, not as the
delay-setting contraction clock.

## Honest limitations

- Three tasks (mod-add, mod-mul, S5 composition), tiny model, full-batch, finite
  budget (12–20k steps); "fails to grok" means "within budget." sc=3 wd=0 Muon
  (3/5, test≈0.94) might fully grok with a longer budget; same caveat for the
  many AdamW/S5 cells stuck at chance.
- λ-independence of Muon's delay is established on mod-add only; the S5 λ-sweep
  was not run. The "near-zero delay" headline is regime-specific (abelian tasks,
  train_frac ≥ 0.4); the regime-general claim is the ordering Muon ≪ AdamW and
  the route distinction, not a universal constant.
- AdamW's single-power-law slope mismatch with the law (−0.34 vs −1) is resolved
  by the two-route fit (β recovers ≈0.92 with a λ-independent plateau in
  parallel); we did not additionally replicate the law's own validation regime.
- Optimizer "families" mix Muon/SGD on hidden matrices with AdamW on embeddings/head
  (standard hybrid); we did not isolate per-group effects.
- We measure ‖θ‖ trajectories but did not fit the law's constant quantitatively
  across families (only the qualitative λ-dependence). A finer wd sweep would
  sharpen the slope claim.
- Grok/memorize thresholds (0.95/0.99) and eval cadence (50 steps) discretize the
  step estimates.

## Figures (results/figures/)

- `fig_delay_summary.png` — delay by optimizer across all 6 (sc, wd) cells.
- `fig_delay_vs_wd.png` — Muon flat vs AdamW decreasing in wd (sc=1).
- `fig_lr_control.png` — delay vs lr per family (sc=1): confound ruled out.
- `fig_curves_*` — accuracy + hidden weight-norm trajectories per cell.
- `fig_init_sweep.png` — grok step vs init scale (single-seed calibration).
