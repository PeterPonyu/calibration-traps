# E2 figure generators (pointers)

Compiled `figs/tex/` and `figs/vec/` are gitignored build products. Do not
commit venue-flat `E2_*.pdf` next to `papers/E2/main.tex`.

| Artifact | Generator | Notes |
|---|---|---|
| E2_scheme | hand-authored TikZ (see lab `papers/figs/tex/E2_scheme.tex`) | schematic |
| E2_landscape | `make_landscape_r.R` | schematic |
| E2_nomogram | `make_E2_nomogram_r.R` | summary JSON tracked |
| E2_td_grid | `make_E2_new_figs_r.R` | |
| E2_case | `make_E2_figs_r.R` | |
| E2_budget_grid | `make_E2_new_figs_r.R` | |
| E2_curriculum | `make_E2_figs_r.R` | |
| E2_induction | `make_E2_new_figs_r.R` | |
| E2_specroute | `make_final_polish_figures_r.R` | |
| E2_supervised_fit | `make_evidence_figures_r.R` | |
| E2_routemat | `make_E2_new_figs_r.R` | |
| E2_positive_rescue | `make_E2_figs_r.R` | |
| E2_walsh | `make_E2_new_figs_r.R` | appendix control |
| E2_grid | `make_E2_figs_r.R` | heatmap → `figs/vec/` |
