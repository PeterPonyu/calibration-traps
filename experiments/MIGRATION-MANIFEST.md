# Tool-workspace → permanent migration manifest (2026-06-21)

Goal: ensure every artifact the 5 papers depend on lives under `papers/` + `experiments/`,
with **zero dependency on the ephemeral tool workspaces `.omx/` and `.omc/`**.
Verified: all 5 papers rebuild clean (0 undefined, 0 Float-too-large) and all 47 figures
re-render with `grep -c .omx papers/*/main.tex papers/figs/*.R == 0`.

## Figure source scripts (were in `.omx`, now canonical in `papers/figs/`)
| script | figures | input now reads from |
|---|---|---|
| `papers/figs/make_deepcheck_figures_r.R` | A/B/C/E1/E2_deepcheck_* (5) | `experiments/results/figures-deepcheck/*.json` |
| `papers/figs/make_redteam_stats_figures_r.R` | C_redteam_scale_boundary | `experiments/results/figures-redteam/redteam_c_scale_stats.json` |
| `papers/figs/make_final_polish_figures_r.R` | A_plasticity_p5_bridge, E2_specroute | `experiments/results/muon_plasticity_p5_bridge/`, `spec_route/` (already permanent) |
| `papers/figs/make_{A,E1,E2}_figs_r.R` (redteam panels) | A_redteam_lmc_extension, E1_redteam_realtext_threshold, E2_redteam_positive_rescue | `experiments/results/figures-redteam/redteam_{a_lmc,e1_manifest,e2}*.json` |

SVG companions now write to `papers/figs/evidence_r/` (was `.omx`).

## Figure-input data (copied `.omx` → `experiments/results/`)
- `experiments/results/figures-deepcheck/` — 5 deepcheck summary JSONs
- `experiments/results/figures-redteam/` — redteam stats JSONs (incl a_lmc/e1_manifest/e2/c_scale)

## Experiment runner/analyzer scripts (29, were `.omx`-only → `experiments/<dir>/`)
icrl_td/ (E2 TD calibrate/denoised/positive-control, 9), muon_plasticity/ (A plasticity P5 + perm-mnist ext, 3),
lmc_instability/ (A LMC minimal/redteam/deepcheck, 3), arch_staircase/ (C scale + lr-sensitivity, 4),
repeated_data/ (E1 realtext + seed-audit, 4), eos_tiny/ (B EoS neighbors/bridge/dense/replicate, 5),
tools/ (final_verify, redteam_audit), legacy_matplotlib/ (make_redteam_figures.py — SUPERSEDED by R pipeline, kept for audit trail).
All 29 pass `python3 -m py_compile`. Full per-file table: see ultragoal ledger / this session.

## Theory + QA tooling
- `experiments/theory-notes/math-backing-{A-B,C,E1-E2}-2026-0621.md` — worked derivations + closed-loop number
  verification for: B bf16-underflow proposition (λ*=2^-8/η≈3.91), E2 single-crossing false-positive,
  A growth-floor, C interaction-order efficiency, E1 grid-power. (Derivations themselves are already IN the papers.)
- `experiments/tools/` — figure_scale_gate.py, paper_integrity_gate.py, paper_visual_data_audit.py, render_contacts.py
  (re-runnable pre-submission QA gates).

## Intentionally left as process scratch (NOT load-bearing; safe to leave or archive)
`.omc/ultragoal/` + `.omx/ultragoal*/` ledgers/briefs/state, `.omx/logs`, `.omx/runtime`, `.omx/research`,
`.omx/reports`, `.omx/visual`, `.omx/plans/findings-ideas-*.md` (early brainstorm idea docs, NOT the real
experiments/findings-*.md), `.omc/progress.txt`, `.omc/sessions`, stale `.omx` R-script duplicates
(canonical copies now in papers/figs/), `.omx/audits` (canonical copies now in experiments/tools/).

## Note
The `.omx` originals of migrated scripts/data were left in place (not deleted) as a safety net;
they are now stale duplicates. They may be removed once submission is final.
