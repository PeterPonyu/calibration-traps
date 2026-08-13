#!/usr/bin/env Rscript
# make_E2_nomogram_r.R — K-selection nomogram for Paper E2 (analytic; no run data).
#   E2_nomogram [body]: required sustained-run length K = (ln M - ln alpha) / ln(1/q)
#   as a function of the per-checkpoint exceedance probability q, with contours for
#   false-positive budgets alpha in {0.05, 0.01, 0.001} at M in {50, 200, 1000}
#   checkpoints; the in-context-TD operating point (q = 0.00272, M = 201) is starred.
# Style matches make_E2_new_figs_r.R (TeX Gyre Termes, ragg 300dpi, 6.5in column).
# Run:  cd papers/figs && Rscript make_E2_nomogram_r.R

ver <- paste(R.version$major, sub('\\..*', '', R.version$minor), sep = '.')
userlib <- file.path(Sys.getenv('HOME'), 'R', 'x86_64-pc-linux-gnu-library', ver)
.libPaths(c(userlib, .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(ragg)
})

find_root <- function() {
  cand <- normalizePath(getwd())
  for (i in 1:6) {
    if (dir.exists(file.path(cand, 'experiments', 'results')) &&
        dir.exists(file.path(cand, 'papers'))) return(cand)
    cand <- dirname(cand)
  }
  stop('could not locate repo root (need experiments/results + papers)')
}
root    <- find_root()
fig_dir <- file.path(root, 'papers', 'figs')
source(file.path(fig_dir, 'fig_pipeline.R'))
source(file.path(fig_dir, 'E2_panel_contract.R'))

theme_paper <- function(base = 9) {
  theme_minimal(base_size = base, base_family = 'TeX Gyre Termes') +
    theme(
      text             = element_text(family = 'TeX Gyre Termes'),
      plot.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.97), margin = margin(b = 3)),
      axis.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.92)),
      axis.text        = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.84), colour = '#1a1a1a'),
      legend.position  = 'top',
      legend.text      = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.84)),
      legend.title     = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.86)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.24, colour = '#E2E2E2')
    )
}

save_png <- function(p, name, w = 6.5, h = 4.2) {
  png_path <- file.path(fig_dir, paste0(name, '.png'))
  ragg::agg_png(png_path, width = w, height = h, units = 'in', res = 300, scaling = 1, background = 'white')
  print(p); dev.off()
  emit_vector(p, name, w, h)
  info <- file.info(png_path)
  cat(sprintf('  saved %-40s  (%s bytes)\n', paste0(name, '.png'), formatC(info$size, big.mark = ',')))
  invisible(png_path)
}

# ---- analytic surface: K(q; alpha, M) = (ln M - ln alpha) / ln(1/q) ----------
qs     <- 10^seq(log10(1e-4), log10(0.5), length.out = 400)
alphas <- c(0.05, 0.01, 0.001)
Ms     <- c(50, 200, 1000)

grid <- expand.grid(q = qs, alpha = alphas, M = Ms) |>
  mutate(K = (log(M) - log(alpha)) / log(1 / q),
         alpha_lab = factor(sprintf('α = %s', alpha),
                            levels = sprintf('α = %s', alphas)),
         M_lab = factor(sprintf('M = %d', M), levels = sprintf('M = %d', Ms)))

# operating point: in-context-TD testbed (q = 0.00272, M = 201, alpha = 0.05)
op_q <- 0.00272
op_K <- (log(201) - log(0.05)) / log(1 / op_q)   # = 1.40 -> K = 2

pal <- c('#0072B2', '#D55E00', '#4a235a')

p <- ggplot(grid, aes(q, K, colour = alpha_lab, linetype = M_lab)) +
  geom_hline(yintercept = 2, colour = '#9aa0a6', linewidth = 0.35, linetype = 'dotted') +
  geom_line(linewidth = 0.55) +
  annotate('point', x = op_q, y = op_K, shape = 8, size = 2.6, stroke = 0.9,
           colour = '#1a1a1a') +
  annotate('text', x = op_q * 1.5, y = op_K - 0.18, hjust = 0, vjust = 1,
           family = 'TeX Gyre Termes', size = 2.8, colour = '#1a1a1a',
           label = 'operating point\nq = 0.00272, M = 201\nK >= 2') +
  scale_x_log10(breaks = c(1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                labels = c('1e-4', '1e-3', '0.01', '0.1', '0.5')) +
  scale_y_log10(breaks = c(1, 2, 3, 5, 10, 20), limits = c(0.9, 22)) +
  scale_colour_manual(values = pal, name = NULL) +
  scale_linetype_manual(values = c('solid', '42', '13'), name = NULL) +
  guides(colour = guide_legend(order = 1), linetype = guide_legend(order = 2)) +
  labs(x = 'per-checkpoint exceedance probability q',
       y = 'required sustained-run length K',
       title = '(a) K-selection nomogram') +
  theme_paper()

# (b) observed vs null-predicted false fires on three corpora: the instrument
#     predicts the false-positive count BEFORE any correction is applied
library(jsonlite)
calib <- fromJSON(file.path(root, 'experiments', 'results', 'icrl_td_ultragoal_calib',
                            'icrl_td_calibration_verdict.json'), simplifyVector = FALSE)
py <- fromJSON(file.path(root, 'experiments', 'revision2026', 'se2-pythia',
                         'se2_analysis_v1_m12.json'), simplifyVector = FALSE)
bb <- fromJSON(file.path(root, 'experiments', 'results', 'bigbench_audit',
                         'audit_table.json'), simplifyVector = FALSE)
cmp <- data.frame(
  corpus = factor(c('TD (45)', 'Pythia (18)', 'BIG-\nBench (89)'),
                  levels = c('TD (45)', 'Pythia (18)', 'BIG-\nBench (89)')),
  observed = c(calib$aggregate$single_cross_0p7,
               py$summary$chance_level$mc_only$single_fires_observed,
               bb$summary$headline$n_tasks_flagged_apparent_emergence),
  predicted = c(NA_real_,
                py$summary$chance_level$mc_only$single_fires_predicted_binomial_null,
                bb$summary$headline$E_false_emergents_predicted_by_chance))
cmp_long <- rbind(
  cmp %>% mutate(qty = 'observed', val = observed) %>% select(corpus, qty, val),
  cmp %>% filter(!is.na(predicted)) %>%
    mutate(qty = 'null-predicted', val = predicted) %>% select(corpus, qty, val))
p_b <- ggplot(cmp_long, aes(corpus, val, fill = qty)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.62, colour = '#1f2937',
           linewidth = 0.15) +
  geom_text(aes(label = sprintf('%.1f', val), group = qty),
            position = position_dodge(width = 0.75), vjust = -0.4, size = 2.5,
            colour = '#1a1a1a', show.legend = FALSE) +
  scale_fill_manual(values = c(observed = '#0072B2', `null-predicted` = '#999999'),
                    name = NULL) +
  labs(x = NULL, y = 'single-crossing false fires',
       title = '(b) Observed vs predicted (3 corpora)') +
  theme_paper() +
  theme(  axis.text.x = element_text(size = rel(0.74), angle = 0, hjust = 0.5, lineheight = 0.85),
        legend.position = 'top')

# (c) the pooled-q cautionary example: routing through the single pooled q
#     under-prescribes K (K = 3 gives 14/45 = 6.2x over budget); the per-run
#     mixture prescribes K = 5 and restores calibration (1/45)
agg <- calib$aggregate
cau <- data.frame(
  route = factor(c('pooled q -> K = 3', 'per-run mixture -> K = 5'),
                 levels = c('pooled q -> K = 3', 'per-run mixture -> K = 5')),
  fp = c(agg$sustain3_0p7, agg$sustain5_0p7),
  budget = 45 * 0.05)
txt_budget <- data.frame(route = factor('per-run mixture -> K = 5', levels = levels(cau$route)),
                         fp = 45 * 0.05,
                         lab = 'budget (0.05 x 45 = 2.25)')
txt_over <- data.frame(route = factor('pooled q -> K = 3', levels = levels(cau$route)),
                       fp = agg$sustain3_0p7 + 1.6,
                       lab = sprintf('%.1fx over budget', (agg$sustain3_0p7 / 45) / 0.05))
p_c <- ggplot(cau, aes(route, fp)) +
  geom_hline(yintercept = 45 * 0.05, linetype = 'dashed', colour = '#b91c1c',
             linewidth = 0.6) +
  geom_col(fill = '#0072B2', width = 0.5) +
  geom_text(aes(label = sprintf('%d/45', fp)), vjust = -0.4, size = 3,
            colour = '#1a1a1a') +
  geom_text(data = txt_budget, aes(label = lab), vjust = -0.5, size = 2.6,
            colour = 'black') +
  geom_text(data = txt_over, aes(label = lab), vjust = -0.4, size = 2.7,
            colour = '#b91c1c') +
  coord_cartesian(ylim = c(0, agg$sustain3_0p7 * 1.25)) +
  labs(x = NULL, y = 'false-positive runs',
       title = '(c) Pooled-q cautionary example') +
  theme_paper() +
  theme(axis.text.x = element_text(size = rel(0.74)))

# (d) power/latency tradeoff: suppression factor vs added confirmation latency
#     (on the genuine-emergence controls, sustained-K removes every false fire
#     while keeping 8/9 genuine detections at zero added latency)
Ks <- 1:7
trad <- data.frame(K = Ks,
                   suppression = 0.00272^(-(Ks - 1)),
                   latency = Ks - 1)
trad_long <- rbind(
  trad %>% transmute(K, qty = 'suppression factor (x)', val = suppression),
  trad %>% transmute(K, qty = 'added latency (checkpoints)', val = latency))
p_d <- ggplot(trad_long, aes(K, val, colour = qty, group = qty)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  scale_y_log10() +
  scale_colour_manual(values = c('suppression factor (x)' = '#0072B2',
                                 'added latency (checkpoints)' = '#D55E00'),
                      name = NULL) +
  annotate('point', x = 2, y = 286, shape = 8, size = 2.6, stroke = 0.9,
           colour = '#1a1a1a') +
  annotate('text', x = 2.25, y = 286, label = '286x at the\noperating point',
           hjust = 0, size = 2.6, colour = 'black', lineheight = 0.9) +
  labs(x = 'sustained-run length K', y = 'value (log scale)',
       title = '(d) Suppression vs latency tradeoff') +
  theme_paper() +
  # Moving the legend to the right keeps it clear of the title on the tikz
  # tier, where the title sits inside the plot box and the top-legend block
  # drops into the title row on narrow panels.
  theme(legend.position = 'right')

library(patchwork)
save_png(compose_E2_four_panel(list(p, p_b, p_c, p_d), 'E2_nomogram', 7.2, 5.4, root), 'E2_nomogram', w = 7.2, h = 5.4)
cat(sprintf('operating point K = %.3f (ceil -> 2)\n', op_K))
