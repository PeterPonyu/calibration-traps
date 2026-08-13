#!/usr/bin/env Rscript
# make_E2_new_figs_r.R — NEW standalone figures for Paper E2 from already-measured data.
# All plot-only (no GPU). Figures:
#   E2_td_grid    [body]    : 9-panel (3 opt x 3 horizon) val-acc traces + folded
#                             precursor-metric null row (Bellman residual)  [folds E2-5]
#   E2_induction  [body]    : induction-circuit emergence positive control vs context length
#   E2_walsh      [appendix]: unlearnable-target control (accuracy vs training budget)
#   E2_routemat   [appendix]: route-AUROC window x tier matrix (pooled -> within-AdamW collapse)
# Style matches make_E2_figs_r.R (TeX Gyre Termes, ragg 300dpi, 6.5in column).
# Run:  cd papers/figs && Rscript make_E2_new_figs_r.R

# ----- library setup ----------------------------------------------------------
ver <- paste(R.version$major, sub('\\..*', '', R.version$minor), sep = '.')
userlib <- file.path(Sys.getenv('HOME'), 'R', 'x86_64-pc-linux-gnu-library', ver)
.libPaths(c(userlib, .libPaths()))

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
  library(ragg)
})

# ----- robust root detection (works from any CWD) -----------------------------
find_root <- function() {
  # try CWD chain first
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
res_dir <- file.path(root, 'experiments', 'results')
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# vector-figure emitter (tikz/PDF) for the LuaLaTeX pipeline
source(file.path(fig_dir, 'fig_pipeline.R'))
source(file.path(fig_dir, 'E2_panel_contract.R'))

# ----- shared theme -----------------------------------------------------------
theme_paper <- function(base = 9) {
  theme_minimal(base_size = base, base_family = 'TeX Gyre Termes') +
    theme(
      text             = element_text(family = 'TeX Gyre Termes'),
      plot.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.97), margin = margin(b = 3)),
      plot.subtitle    = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.86), colour = '#1a1a1a', margin = margin(b = 5)),
      axis.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.92)),
      axis.text        = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.84), colour = '#1a1a1a'),
      legend.title     = element_blank(),
      legend.position  = 'top',
      legend.text      = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.84)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.24, colour = '#E2E2E2'),
      strip.text       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.86), colour = '#1a202c'),
      plot.margin      = margin(5.5, 7, 5.5, 7),
      plot.tag         = element_text(face = 'bold', size = 10),
      plot.tag.position = 'topleft'
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

# read per-step records (lines with a "step" key) from a jsonl run file
read_steps <- function(path) {
  lines <- readLines(path, warn = FALSE)
  rows <- lapply(lines, function(l) {
    d <- tryCatch(fromJSON(l, simplifyVector = FALSE), error = function(e) NULL)
    if (!is.null(d) && !is.null(d$step)) d else NULL
  })
  Filter(Negate(is.null), rows)
}
read_summary <- function(path) {
  lines <- readLines(path, warn = FALSE)
  d <- fromJSON(lines[length(lines)], simplifyVector = FALSE)
  d[['_summary']]
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x
OPT_LABELS <- c(adamw = 'AdamW', muon = 'Muon', sgdm = 'SGD+momentum')
OPT_LEVELS <- c('adamw', 'muon', 'sgdm')
OPT_PAL    <- c('AdamW' = '#0072B2', 'Muon' = '#D55E00', 'SGD+momentum' = '#999999')

# ============================================================
# 1. E2_td_grid : 9-panel TD calibration grid + Bellman-residual null row (folds E2-5)
# ============================================================
make_td_grid <- function() {
  cat('--- E2_td_grid ---\n')
  opts <- OPT_LEVELS; Ts <- c(10, 20, 40)
  traces <- list(); precursor <- list()
  med_finals <- c(); single08 <- 0; sust2 <- 0; finals_all <- c()
  for (opt in opts) for (T in Ts) {
    fs <- sort(Sys.glob(file.path(res_dir, 'icrl_td', sprintf('%s_T%d_s*.jsonl', opt, T))))
    cell_finals <- c(); cell_single <- 0
    for (f in fs) {
      r <- read_steps(f)
      step <- vapply(r, function(x) as.numeric(x$step), numeric(1))
      va   <- vapply(r, function(x) as.numeric(x$val_acc), numeric(1))
      br   <- vapply(r, function(x) as.numeric(x$bellman_residual), numeric(1))
      seed <- sub('.*_s(\\d+)\\.jsonl$', '\\1', basename(f))
      traces[[length(traces) + 1]] <- tibble(opt = opt, T = T, seed = seed, step = step, val_acc = va)
      precursor[[length(precursor) + 1]] <- tibble(opt = opt, seed = seed, step = step, bellman = br)
      cell_finals <- c(cell_finals, va[length(va)]); finals_all <- c(finals_all, va[length(va)])
      if (max(va) >= 0.8) { cell_single <- cell_single + 1; single08 <- single08 + 1 }
      if (any(va[-length(va)] >= 0.8 & va[-1] >= 0.8)) sust2 <- sust2 + 1
    }
    med_finals <- c(med_finals, median(cell_finals))
  }
  trc <- bind_rows(traces)
  prc <- bind_rows(precursor)
  chance <- median(finals_all)
  cat(sprintf('  TD n=%d median-final=%.4f  final>=0.8: %d/%d  single>=0.8: %d/%d  sustained-2>=0.8: %d/%d\n',
              length(finals_all), chance, sum(finals_all >= 0.8), length(finals_all),
              single08, length(finals_all), sust2, length(finals_all)))

  trc <- trc %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = unname(OPT_LABELS)),
           # Short strip labels: "horizon T = 10" clips in narrow columns;
           # "T = 10/20/40" is unambiguous and fits without truncation.
           T_lab   = factor(paste0('T = ', T), levels = paste0('T = ', Ts)))

  # 9-panel small multiple of per-seed traces; the three reference lines get a
  # proper legend (so the subtitle stays short instead of describing each line).
  ref_df <- data.frame(
    y   = c(0.8, 0.7, chance),
    ref = factor(c('0.8 detector', '0.7 detector', 'chance'),
                 levels = c('0.8 detector', '0.7 detector', 'chance'))
  )
  p_grid <- ggplot(trc, aes(step, val_acc)) +
    geom_hline(data = ref_df, aes(yintercept = y, colour = ref, linetype = ref),
               linewidth = 0.4) +
    geom_line(aes(group = seed), colour = '#0072B2', linewidth = 0.30, alpha = 0.7) +
    facet_grid(opt_lab ~ T_lab) +
    scale_colour_manual(values = c('0.8 detector' = '#b91c1c', '0.7 detector' = '#047857',
                                   'chance' = '#6b7280'), name = NULL) +
    scale_linetype_manual(values = c('0.8 detector' = 'dashed', '0.7 detector' = '33',
                                     'chance' = 'dotted'), name = NULL) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = 'k'),
                       breaks = c(0, 5000, 10000)) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0.1, 0.95), breaks = c(0.2, 0.4, 0.6, 0.8)) +
    labs(title = 'Validation accuracy across horizons and optimizers',
         subtitle = 'five seeds per cell',
         x = 'training step', y = 'validation accuracy') +
    theme_paper(9) +
    theme(legend.position = 'top', plot.subtitle = element_text(size = rel(0.80)))

  # precursor null row: median Bellman residual over training, per optimizer
  prc_s <- prc %>% group_by(opt, step) %>%
    summarise(med = median(bellman), .groups = 'drop') %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = unname(OPT_LABELS)))
  br_init  <- median(prc$bellman[prc$step == min(prc$step)])
  br_final <- median(prc$bellman[prc$step == max(prc$step)])
  cat(sprintf('  Bellman residual median: init=%.3f final=%.3f (never reaches 0 => no value learning)\n',
              br_init, br_final))

  p_prec <- ggplot(prc_s, aes(step, med, colour = opt_lab)) +
    geom_hline(yintercept = 0, linetype = 'dotted', colour = '#6b7280', linewidth = 0.4) +
    geom_line(linewidth = 0.6) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = 'k'),
                       breaks = c(0, 5000, 10000)) +
    scale_y_continuous(limits = c(0, NA)) +
    scale_colour_manual(values = OPT_PAL) +
    labs(title = 'Bellman residual vs training step',
         subtitle = 'median over five seeds per optimizer',
         x = 'training step', y = 'Bellman residual') +
    theme_paper(9) +
    theme(plot.subtitle = element_text(size = rel(0.80)))

  # (c) single-crossing fire map at tau = 0.7: which cells the detector fires
  #     on (42/45 runs; the false-positive rate the theorem predicts)
  calib <- jsonlite::fromJSON(file.path(res_dir, 'icrl_td_ultragoal_calib',
                                        'icrl_td_calibration_verdict.json'),
                              simplifyVector = FALSE)
  fire_df <- bind_rows(lapply(calib$runs, function(r) {
    tibble(opt = r$optimizer, T = r$T, fires = !is.null(r$first_ge_0.7_K1))
  })) %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = unname(OPT_LABELS)),
           # Match the shortened strip labels used in p_grid above.
           T_lab = factor(paste0('T = ', T), levels = paste0('T = ', Ts)))
  fire_cell <- fire_df %>% group_by(opt_lab, T_lab) %>%
    summarise(n_fire = sum(fires), n = n(), .groups = 'drop') %>%
    mutate(rate = n_fire / n,
           label = ifelse(n_fire < n, paste0(n_fire, '/', n), ''),
           label_y = ifelse(rate > 0.15, rate - 0.04, 0.035),
           label_colour = ifelse(rate > 0.15, 'white', '#1a1a1a'))
  p_fire <- ggplot(fire_cell, aes(T_lab, rate, fill = opt_lab)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65, colour = '#1f2937',
             linewidth = 0.15) +
    geom_text(aes(y = label_y, label = label, group = opt_lab,
                  colour = label_colour),
              position = position_dodge(width = 0.75),
              vjust = 1.1, size = 2.2, show.legend = FALSE) +
    scale_colour_identity() +
    scale_fill_manual(values = OPT_PAL, name = NULL, guide = 'none') +
    coord_cartesian(ylim = c(0, 1.15)) +
    labs(title = '(b) Single-crossing fire rate (tau = 0.7)',
         x = 'horizon', y = 'fire rate (5 seeds)') +
    theme_paper(9) +
    theme(legend.position = 'top')

  # (d) sustained-K ladder: observed false-positive counts collapse with K
  #     (tau = 0.7: 42/24/14/1; tau = 0.8: 26/0/0/0). The per-run mixture of
  #     Section sec:r2 predicts 1.5 at K = 5 against the observed 1.
  agg <- calib$aggregate
  lad <- bind_rows(
    tibble(K = factor(c('K=1', 'K=2', 'K=3', 'K=5'), levels = c('K=1', 'K=2', 'K=3', 'K=5')),
           series = 'tau = 0.7',
           fp = c(agg$single_cross_0p7, agg$sustain2_0p7, agg$sustain3_0p7, agg$sustain5_0p7)),
    tibble(K = factor(c('K=1', 'K=2', 'K=3', 'K=5'), levels = c('K=1', 'K=2', 'K=3', 'K=5')),
           series = 'tau = 0.8',
           fp = c(agg$single_cross_0p8, agg$sustain2_0p8, agg$sustain3_0p8, agg$sustain5_0p8)))
  p_lad <- ggplot(lad, aes(K, fp, fill = series)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, colour = '#1f2937',
             linewidth = 0.15) +
    geom_text(aes(label = fp, group = series), position = position_dodge(width = 0.7),
              vjust = -0.4, size = 2.8, colour = '#1a1a1a', show.legend = FALSE) +
    annotate('point', x = 4.34, y = 1.5, shape = 8, size = 3, stroke = 1, colour = '#1a1a1a') +
    annotate('text', x = 4.40, y = 1.5, label = 'predicted 1.5', hjust = 0, vjust = 0.5,
             size = 2.2, colour = '#1a1a1a') +
    # Place the annotation as a subtitle under the panel title, where it has
    # full panel width and never collides with bars. Use annotate with x at
    # the panel mid and y above the tallest bar (y = 50) so the text sits in
    # clear white space near the top.
    annotate('text', x = 2.5, y = 51,
             label = 'mixture predicts 1.5 at K = 5',
             size = 2.3, colour = 'black', hjust = 0.5, lineheight = 0.9) +
    coord_cartesian(ylim = c(0, 56), clip = 'off') +
    scale_fill_manual(values = c('tau = 0.7' = '#047857', 'tau = 0.8' = '#b91c1c'),
                      name = NULL) +
    labs(title = '(d) Sustained-K ladder (observed FP of 45 runs)',
         x = NULL, y = 'false-positive runs') +
    theme_paper(9) +
    theme(legend.position = 'top',
          plot.margin = margin(5.5, 7, 5.5, 7))

  comp <- compose_E2_four_panel(list(p_grid, p_fire, p_prec, p_lad),
                                'E2_td_grid', 7.2, 7.4, root)
  save_png(comp, 'E2_td_grid', w = 7.2, h = 7.4)
}

# ============================================================
# 2. E2_induction : emergence positive control vs context length
# ============================================================
make_induction <- function() {
  cat('--- E2_induction ---\n')
  opts <- OPT_LEVELS; Ls <- c(64, 128, 256)
  rows <- list()
  for (opt in opts) for (L in Ls) {
    fs <- sort(Sys.glob(file.path(res_dir, 'induction_emergence', sprintf('%s_L%d_s*.jsonl', opt, L))))
    for (f in fs) {
      s <- read_summary(f)
      es  <- if (!is.null(s$emergence_step)) as.numeric(s$emergence_step) else NA_real_
      icl <- as.numeric(s$final_icl_score)
      rows[[length(rows) + 1]] <- tibble(opt = opt, L = L, emergence_step = es, icl = icl)
    }
  }
  df <- bind_rows(rows)
  cat('  per-cell median emergence_step / median ICL:\n')
  for (opt in opts) for (L in Ls) {
    sub <- df %>% filter(opt == !!opt, L == !!L)
    cat(sprintf('    %-5s L%-3d  med_step=%3.0f  med_icl=%.3f  (n=%d)\n',
                opt, L, median(sub$emergence_step, na.rm = TRUE), median(sub$icl), nrow(sub)))
  }

  df <- df %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = unname(OPT_LABELS)))
  summ <- df %>% group_by(opt_lab, L) %>%
    summarise(med = median(emergence_step, na.rm = TRUE),
              lo = quantile(emergence_step, 0.25, na.rm = TRUE),
              hi = quantile(emergence_step, 0.75, na.rm = TRUE),
              med_icl = median(icl), .groups = 'drop')

  # main: emergence step vs L (small horizontal dodge so optimizers don't overplot)
  Lpos <- c('64' = 1, '128' = 2, '256' = 3)
  df    <- df %>% mutate(xpos = Lpos[as.character(L)] +
                           (as.integer(opt_lab) - 2) * 0.10)
  summ  <- summ %>% mutate(xpos = Lpos[as.character(L)] +
                            (as.integer(opt_lab) - 2) * 0.10)
  p_main <- ggplot(summ, aes(xpos, med, colour = opt_lab)) +
    geom_jitter(data = df, aes(xpos, emergence_step, colour = opt_lab),
                width = 0.05, height = 8, size = 1.1, alpha = 0.30, show.legend = FALSE) +
    geom_line(aes(group = opt_lab), linewidth = 0.7) +
    geom_pointrange(aes(ymin = lo, ymax = hi), size = 0.35, linewidth = 0.5) +
    scale_x_continuous(breaks = 1:3, labels = c('64', '128', '256')) +
    scale_y_continuous(limits = c(0, NA)) +
    scale_colour_manual(values = OPT_PAL) +
    labs(title = 'Emergence step vs context length',
         subtitle = 'faint dots = per seed; line + range = median ± IQR',
         x = 'context length', y = 'emergence step') +
    theme_paper(9)

  # companion: final ICL score (all cells high => genuine learning)
  p_icl <- ggplot(df, aes(opt_lab, icl, colour = opt_lab)) +
    geom_hline(yintercept = 0.5, linetype = 'dotted', colour = '#6b7280', linewidth = 0.4) +
    geom_jitter(width = 0.12, height = 0, size = 1.0, alpha = 0.35, show.legend = FALSE) +
    stat_summary(fun = median, geom = 'point', size = 2.6, shape = 18, show.legend = FALSE) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
    scale_colour_manual(values = OPT_PAL) +
    labs(title = 'Final ICL score by optimizer',
         subtitle = '◆ = median; faint dots = per run',
         x = NULL, y = 'ICL score') +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(0.74)))

  # (c) sustained-K latency audit: per run, the sustained-K=2 crossing is at
  #     most one checkpoint later than the single crossing (zero added latency)
  lat_rows <- list()
  for (opt in opts) for (L in Ls) {
    fs <- sort(Sys.glob(file.path(res_dir, 'induction_emergence', sprintf('%s_L%d_s*.jsonl', opt, L))))
    for (f in fs) {
      r <- read_steps(f)
      steps <- vapply(r, function(x) as.numeric(x$step), numeric(1))
      icl   <- vapply(r, function(x) as.numeric(x$icl_score %||% NA_real_), numeric(1))
      hi    <- !is.na(icl) & icl >= 0.5
      if (!any(hi)) next
      single <- steps[which(hi)[1]]
      both   <- which(hi[-1] & hi[-length(hi)])
      sust   <- if (length(both)) steps[both[1] + 1] else NA_real_
      lat_rows[[length(lat_rows) + 1]] <- tibble(
        opt = opt, L = L, single = single, sustained = sust,
        delay = ifelse(is.na(sust), NA_real_, sust - single))
    }
  }
  lat <- bind_rows(lat_rows) %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = unname(OPT_LABELS)))
  lat_summ <- lat %>% group_by(opt_lab) %>%
    summarise(zero = mean(delay == 0, na.rm = TRUE),
              med_delay = median(delay, na.rm = TRUE), n = n(), .groups = 'drop')
  p_lat <- ggplot(lat, aes(opt_lab, delay, colour = opt_lab)) +
    stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a',
                 na.rm = TRUE) +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 83),
               size = 1.6, alpha = 0.75, show.legend = FALSE) +
    scale_colour_manual(values = OPT_PAL) +
    labs(title = '(c) Sustained-K latency (single -> K=2)',
         x = NULL, y = 'added checkpoints (0 = none)') +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(0.7), angle = 25, hjust = 1))

  # (d) per-seed onset steps by context length (the raw distributions)
  p_onset <- ggplot(df, aes(factor(L), emergence_step, colour = opt_lab)) +
    stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
                 position = position_dodge(width = 0.6)) +
    geom_point(position = position_jitterdodge(jitter.width = 0.05, dodge.width = 0.6,
                                               seed = 83),
               size = 1.2, alpha = 0.6) +
    scale_colour_manual(values = OPT_PAL) +
    labs(title = '(d) Per-seed onset steps',
         x = 'context length', y = 'emergence step') +
    theme_paper(9) +
    theme(legend.position = 'top')

  comp <- compose_E2_four_panel(list(p_main, p_icl, p_lat, p_onset),
                                'E2_induction', 7.2, 5.4, root)
  save_png(comp, 'E2_induction', w = 7.2, h = 5.4)
}

# ============================================================
# 3. E2_walsh : unlearnable-target control (accuracy vs training budget)
# ============================================================
make_walsh <- function() {
  cat('--- E2_walsh ---\n')
  # Use only the comparable single-target (pure degree-3) ladder so the budget axis
  # compares like with like; the staircase-target variants use a DIFFERENT target and
  # are excluded to avoid conflating two targets on one axis.
  dirs <- c(
    'curriculum_order'        ,  # total budget 2400
    'curriculum_order_pilot8k',  # total budget 24000
    'curriculum_order_pilot16k'  # total budget 48000
  )
  collect <- function(dir, fam) {
    fs <- Sys.glob(file.path(res_dir, dir, sprintf('%s_iid_adamw_s*.jsonl', fam)))
    out <- list()
    for (f in fs) {
      s <- read_summary(f)
      stages <- as.numeric(s$stage_boundaries[[length(s$stage_boundaries)]])
      out[[length(out) + 1]] <- tibble(
        family = fam, budget = stages,
        acc = as.numeric(s$final_eval_acc))
    }
    bind_rows(out)
  }
  walsh <- bind_rows(lapply(dirs, collect, fam = 'walsh'))
  cat('  walsh (single-target) accuracy vs budget:\n')
  ws <- walsh %>% group_by(budget) %>% summarise(med = median(acc), n = n(), .groups = 'drop') %>% arrange(budget)
  for (i in seq_len(nrow(ws))) cat(sprintf('    budget=%6.0f  med_acc=%.4f  (n=%d)\n', ws$budget[i], ws$med[i], ws$n[i]))
  cat(sprintf('  budget span: %.0fx\n', max(walsh$budget) / min(walsh$budget)))

  # copy learnable reference (iid adamw) -- plot the ACTUAL per-seed copy runs as
  # a data series (not just a reference line) so the contrast is empirical: the
  # learnable target solves while the non-Fourier target stays at the floor.
  copy_df <- bind_rows(lapply(dirs, function(d) {
    fs <- Sys.glob(file.path(res_dir, d, 'copy_iid_adamw_s*.jsonl'))
    if (length(fs) == 0) return(NULL)
    bind_rows(lapply(fs, function(f) {
      s <- read_summary(f)
      stages <- as.numeric(s$stage_boundaries[[length(s$stage_boundaries)]])
      tibble(budget = stages, acc = as.numeric(s$final_eval_acc))
    }))
  }))
  copyacc <- median(copy_df$acc)
  cat(sprintf('  copy reference (learnable) median final acc=%.4f (n=%d)\n', copyacc, nrow(copy_df)))

  summ      <- walsh %>% group_by(budget) %>% summarise(med = median(acc), .groups = 'drop')
  floor_med <- median(walsh$acc)
  mid_budget <- sort(unique(walsh$budget))[2]

  p <- ggplot(walsh, aes(budget, acc)) +
    geom_hline(yintercept = copyacc, linetype = 'dashed', colour = '#0072B2', linewidth = 0.55) +
    geom_jitter(data = copy_df, aes(budget, acc), width = 0.03, height = 0,
                colour = '#0072B2', size = 1.9, shape = 17, alpha = 0.75) +
    geom_line(data = summ, aes(budget, med), colour = '#4a235a', linewidth = 0.5, linetype = '22') +
    geom_jitter(width = 0.03, height = 0, colour = '#7d3c98', size = 1.8, alpha = 0.6) +
    geom_point(data = summ, aes(budget, med), colour = '#4a235a', size = 2.8, shape = 18) +
    annotate('text', x = mid_budget, y = floor_med + 0.12,
             label = sprintf('non-Fourier target stays at floor (~%.2f)', floor_med),
             hjust = 0.5, size = 2.7, colour = 'black', family = 'TeX Gyre Termes') +
    # label the ceiling series in-panel so the wide empty middle band reads as
    # the finding (ceiling vs floor), not as a blank plot
    annotate('text', x = mid_budget, y = copyacc - 0.09,
             label = sprintf('learnable copy target solves (~%.2f)', copyacc),
             hjust = 0.5, size = 2.7, colour = '#0072B2', family = 'TeX Gyre Termes') +
    scale_x_log10(breaks = sort(unique(walsh$budget)), labels = comma_format()) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05),
                       breaks = seq(0, 1, 0.2)) +
    labs(title = '(a) Final accuracy vs training budget: learnable vs non-Fourier',
         subtitle = 'triangles = learnable copy runs; circles/diamond = non-Fourier per-seed/median (AdamW)',
         x = 'total training budget (steps, log scale)', y = 'final accuracy') +
    theme_paper(9) +
    theme(plot.subtitle = element_text(size = rel(0.82)))

  # (b) floor is ordering-robust: the non-Fourier target stays at the floor
  #     under every data-ordering policy, not just iid
  walsh_ord <- bind_rows(lapply(c('curriculum_order'), function(d) {
    ords <- c('iid', 'easy_to_hard', 'hard_first', 'structured')
    bind_rows(lapply(ords, function(o) {
      fs <- Sys.glob(file.path(res_dir, d, sprintf('walsh_%s_adamw_s*.jsonl', o)))
      bind_rows(lapply(fs, function(f) {
        s <- read_summary(f)
        tibble(ordering = o, acc = as.numeric(s$final_eval_acc))
      }))
    }))
  })) %>%
    mutate(ordering = factor(ordering,
                             levels = c('iid', 'easy_to_hard', 'hard_first', 'structured'),
                             labels = c('iid', 'easy→hard', 'hard first', 'structured')))
  p_b <- ggplot(walsh_ord, aes(ordering, acc)) +
    geom_hline(yintercept = floor_med, linetype = '22', colour = '#6b7280', linewidth = 0.5) +
    stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 87),
               size = 1.8, colour = '#7d3c98', alpha = 0.75) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05)) +
    labs(title = '(b) Floor is ordering-robust',
         x = NULL, y = 'final accuracy') +
    theme_paper(9)

  # (c) budget-response slope per target (medians): the learnable target
  #     improves with budget, the non-Fourier target does not
  both <- bind_rows(
    walsh %>% mutate(target = 'non-Fourier (walsh)'),
    copy_df %>% mutate(target = 'learnable (copy)'))
  both_summ <- both %>% group_by(target, budget) %>%
    summarise(med = median(acc), .groups = 'drop')
  p_c <- ggplot(both_summ, aes(budget, med, colour = target)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.4) +
    scale_colour_manual(values = c('learnable (copy)' = '#0072B2',
                                   'non-Fourier (walsh)' = '#7d3c98'), name = NULL) +
    scale_x_log10(breaks = sort(unique(both$budget)), labels = comma_format()) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05)) +
    labs(title = '(c) Budget-response per target (medians)',
         x = 'total training budget (steps, log)', y = 'median final accuracy') +
    theme_paper(9) +
    theme(legend.position = 'top')

  # (d) learnable-minus-floor gap by budget: the discriminative margin of the
  #     discipline-3 control
  gap <- both_summ %>% tidyr::pivot_wider(names_from = target, values_from = med) %>%
    mutate(gap = `learnable (copy)` - `non-Fourier (walsh)`)
  p_d <- ggplot(gap, aes(budget, gap)) +
    geom_line(colour = '#1a1a1a', linewidth = 0.8) +
    geom_point(colour = '#1a1a1a', size = 2.4) +
    scale_x_log10(breaks = gap$budget, labels = comma_format()) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05)) +
    labs(title = '(d) Learnable-vs-floor margin by budget',
         x = 'total training budget (steps, log)', y = 'accuracy margin') +
    theme_paper(9)

  save_png((p | p_b | p_c | p_d) + plot_layout(nrow = 2), 'E2_walsh', w = 7.2, h = 5.4)
}

# ============================================================
# 4. E2_routemat : route-AUROC window x tier matrix (pooled -> within-AdamW)
# ============================================================
make_routemat <- function() {
  cat('--- E2_routemat ---\n')
  txt <- paste(readLines(file.path(res_dir, 'spec_route', 'posthoc_round1.json'), warn = FALSE), collapse = '\n')
  txt <- gsub('\\bNaN\\b', 'null', txt)
  spec <- fromJSON(txt, simplifyVector = FALSE)

  cells <- c('W-200/tierA', 'W-200/tierB', 'W-mem/tierA', 'W-mem/tierB')
  rows <- list()
  for (c in cells) {
    pooled <- spec[[c]][['route/config']][['auroc']]
    within <- spec[[c]][['route/config_adamw']][['auroc']]
    win <- ifelse(grepl('W-200', c), '200-step window', 'mem-onset window')
    tier <- ifelse(grepl('tierA', c), 'tier A', 'tier B')
    rows[[length(rows) + 1]] <- tibble(window = win, tier = tier,
                                       cohort = 'pooled (all optimizers)',
                                       auroc = if (is.null(pooled)) NA_real_ else as.numeric(pooled))
    rows[[length(rows) + 1]] <- tibble(window = win, tier = tier,
                                       cohort = 'within AdamW',
                                       auroc = if (is.null(within)) NA_real_ else as.numeric(within))
  }
  df <- bind_rows(rows) %>%
    mutate(window = factor(window, levels = c('200-step window', 'mem-onset window')),
           tier   = factor(tier, levels = c('tier A', 'tier B')),
           cohort = factor(cohort, levels = c('pooled (all optimizers)', 'within AdamW')))
  cat('  AUROC matrix:\n'); print(as.data.frame(df))

  # Tier A reproduces the four bars already in the main-text spec-route figure
  # (pooled vs within-AdamW collapse); drop it here to avoid duplication and keep
  # only the tier-B breakdown, which the main figure does not show.
  # Tier B has NO estimable within-AdamW cell (label imbalance), so a two-facet
  # pooled/within layout was half "n/a" whitespace — plot the two pooled bars in
  # ONE panel and state the missing cohort as an in-panel note.
  df <- df %>% filter(tier == 'tier B', cohort == 'pooled (all optimizers)')

  p <- ggplot(df, aes(window, auroc)) +
    geom_col(width = 0.45, fill = '#0072B2', colour = '#1f2937', linewidth = 0.2) +
    geom_hline(yintercept = 0.5, linetype = 'dotted', colour = '#4b5563', linewidth = 0.45) +
    geom_text(aes(label = sprintf('%.2f', auroc)), vjust = -0.5, size = 3.1) +
    annotate('text', x = 0.55, y = 0.90, hjust = 0, vjust = 1, size = 2.5, colour = '#374151',
             label = 'bars: pooled across optimizers;\nwithin-AdamW not estimable in tier B\n(insufficient label balance)') +
    annotate('text', x = 0.55, y = 0.53, hjust = 0, size = 2.7, colour = '#4b5563',
             label = 'chance') +
    scale_x_discrete(labels = c('200-step window' = '200-step',
                                'mem-onset window' = 'mem-onset')) +
    scale_y_continuous(limits = c(0, 1.0), breaks = c(0, 0.5, 0.8)) +
    labs(title = '(a) Route-prediction AUROC (tier B), pooled',
         x = NULL, y = 'route-prediction AUROC') +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(1.0)),
          legend.position = 'none')

  # (b) tier-A reference: pooled vs within-AdamW collapse (from the same json;
  #     the main-text spec-route figure's four bars, restated for completeness)
  ta <- bind_rows(rows) %>%
    mutate(window = factor(window, levels = c('200-step window', 'mem-onset window')),
           tier   = factor(tier, levels = c('tier A', 'tier B')),
           cohort = factor(cohort, levels = c('pooled (all optimizers)', 'within AdamW'))) %>%
    filter(tier == 'tier A')
  p_b <- ggplot(ta, aes(window, auroc, fill = cohort)) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55,
             colour = '#1f2937', linewidth = 0.2) +
    geom_hline(yintercept = 0.5, linetype = 'dotted', colour = '#4b5563', linewidth = 0.45) +
    geom_text(aes(label = sprintf('%.2f', auroc), group = cohort),
              position = position_dodge(width = 0.65), vjust = -0.4, size = 2.8,
              show.legend = FALSE) +
    scale_x_discrete(labels = c('200-step window' = '200-step',
                                'mem-onset window' = 'mem-onset')) +
    scale_fill_manual(values = c('pooled (all optimizers)' = '#0072B2',
                                 'within AdamW' = '#D62728'), name = NULL,
                      labels = c('pooled (all optimizers)' = 'pooled across optimizers',
                                 'within AdamW' = 'within AdamW')) +
    coord_cartesian(ylim = c(0.4, 0.95)) +
    labs(title = '(b) Tier A: pooled vs within-AdamW collapse',
         x = NULL, y = 'route-prediction AUROC') +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(0.82)), legend.position = 'top')

  # (c) per-window delta (within AdamW minus pooled, tier A): the collapse
  #     quantified as the optimizer-identity component
  dl <- ta %>% tidyr::pivot_wider(names_from = cohort, values_from = auroc) %>%
    mutate(delta = `pooled (all optimizers)` - `within AdamW`)
  p_c <- ggplot(dl, aes(window, delta)) +
    geom_hline(yintercept = 0, linetype = 'dotted', colour = '#4b5563', linewidth = 0.45) +
    geom_col(width = 0.45, fill = '#7F7F7F', colour = '#1f2937', linewidth = 0.2) +
    geom_text(aes(label = sprintf('%+.2f', delta)), vjust = -0.5, size = 3.0) +
    scale_x_discrete(labels = c('200-step window' = '200-step',
                                'mem-onset window' = 'mem-onset')) +
    labs(title = '(c) Optimizer-identity component (tier A)',
         x = NULL, y = 'AUROC delta (pooled - within)') +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.20))) +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(0.82)))

  # (d) variance decomposition (tier A, per optimizer): config share is the
  #     dominant component in every optimizer — the fingerprint, not the run
  p4_src <- spec[['W-mem/tierA']][['p4_decomposition']]
  p4 <- bind_rows(lapply(names(p4_src), function(k) {
    x <- p4_src[[k]]
    tibble(optimizer = sub(',.*', '', gsub("[()' ]", '', k)),
           config = x$config_share, signal = x$signal_share, residual = x$residual_share)
  })) %>% pivot_longer(c(config, signal, residual), names_to = 'component',
                       values_to = 'share') %>%
    mutate(component = factor(component, levels = c('config', 'signal', 'residual')))
  p_d <- ggplot(p4, aes(optimizer, share, fill = component)) +
    geom_col(width = 0.6, colour = 'white', linewidth = 0.2) +
    scale_fill_manual(values = c(config = '#009E73', signal = '#0072B2',
                                 residual = '#7F7F7F'), name = NULL) +
    labs(title = '(d) Variance decomposition (mem-onset window)',
         x = NULL, y = 'variance share') +
    theme_paper(9) +
    theme(legend.position = 'top')

  save_png(compose_E2_four_panel(list(p, p_b, p_c, p_d), 'E2_routemat', 7.2, 5.4, root), 'E2_routemat', w = 7.2, h = 5.4)
}

# ============================================================
cat('\n=== make_E2_new_figs_r.R ===\n')
# ============================================================
# 5. E2_budget_grid : gamma=0.5 sixty-run budget grid (4x/8x), clean negative
#    Data: experiments/results/icrl_td_positive_regime/*.jsonl (60 runs).
#    Every plotted value computed here from the raw jsonls: per-run max val_acc
#    over all ~201 checkpoints (points) and per-cell median FINAL val_acc
#    (crossbars). Files end with a _gamma_meta line, so the summary is located
#    by scanning for the _summary record, not by reading the last line.
# ============================================================
make_budget_grid <- function() {
  cat('--- E2_budget_grid ---\n')
  src <- file.path(res_dir, 'icrl_td_positive_regime')
  files <- list.files(src, pattern = '\\.jsonl$', full.names = TRUE)
  stopifnot(length(files) == 60)

  rows <- lapply(files, function(f) {
    m <- regmatches(basename(f),
                    regexec('^(adamw|muon)_T(\\d+)_s(\\d+)_g0p5_steps(\\d+)\\.jsonl$', basename(f)))[[1]]
    stopifnot(length(m) == 5)
    maxacc <- -Inf; finacc <- NA_real_
    for (l in readLines(f, warn = FALSE)) {
      d <- tryCatch(fromJSON(l, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(d)) next
      if (!is.null(d$val_acc)) maxacc <- max(maxacc, d$val_acc)
      if (!is.null(d[['_summary']])) finacc <- d[['_summary']]$final_val_acc
    }
    tibble(opt = m[2], T = as.integer(m[3]), seed = as.integer(m[4]),
           steps = as.integer(m[5]), max_acc = maxacc, final_acc = finacc)
  })
  df <- bind_rows(rows) %>%
    mutate(opt_lab = factor(OPT_LABELS[opt], levels = OPT_LABELS[c('adamw','muon')]),
           budget  = factor(ifelse(steps == 40000, '4x budget (40k steps)', '8x budget (80k steps)'),
                            levels = c('4x budget (40k steps)', '8x budget (80k steps)')),
           T_f     = factor(T, levels = c(10, 20, 40)))
  stopifnot(all(is.finite(df$max_acc)), !any(is.na(df$final_acc)))
  med <- df %>% group_by(budget, T_f, opt_lab) %>%
    summarise(med_final = median(final_acc), .groups = 'drop')
  cat(sprintf('  runs=%d  max(max_acc)=%.3f  final range %.3f-%.3f  crossings>=0.7: %d\n',
              nrow(df), max(df$max_acc), min(df$final_acc), max(df$final_acc),
              sum(df$max_acc >= 0.7)))

  pd <- position_dodge(width = 0.55)
  p <- ggplot(df, aes(T_f, max_acc, colour = opt_lab, group = opt_lab)) +
    geom_hline(yintercept = 0.8, linetype = 'dashed', colour = '#b91c1c', linewidth = 0.5) +
    geom_hline(yintercept = 0.7, linetype = '33', colour = '#047857', linewidth = 0.45) +
    geom_point(position = pd, size = 1.5, alpha = 0.55, shape = 16) +
    geom_crossbar(data = med, aes(T_f, med_final, ymin = med_final, ymax = med_final,
                                  colour = opt_lab, group = opt_lab),
                  position = pd, width = 0.42, linewidth = 0.55, inherit.aes = FALSE) +
    facet_wrap(~ budget, nrow = 1) +
    annotate('text', x = 0.62, y = 0.845, label = 'detection bar (0.8)',
             size = 2.5, colour = 'black', hjust = 0) +
    scale_colour_manual(values = OPT_PAL) +
    scale_y_continuous(limits = c(0, 0.9), breaks = seq(0, 0.8, 0.2)) +
    labs(title = '(a) gamma = 0.5 budget grid (max acc points, final crossbars)',
         x = 'in-context horizon T',
         y = 'validation accuracy') +
    theme_paper(9)

  # (b) per-horizon marginal: max accuracy by horizon (pooling budget/optimizer)
  p_b <- ggplot(df, aes(T_f, max_acc)) +
    stat_summary(fun = median, geom = 'crossbar', width = 0.45, colour = '#1a1a1a') +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 89),
               size = 1.4, alpha = 0.5, colour = '#555555') +
    geom_hline(yintercept = 0.7, linetype = '33', colour = '#047857', linewidth = 0.45) +
    scale_y_continuous(limits = c(0, 0.9), breaks = seq(0, 0.8, 0.2)) +
    labs(title = '(b) Marginal by horizon (all 60 runs)',
         x = 'in-context horizon T', y = 'max validation accuracy') +
    theme_paper(9)

  # (c) the gamma ladder: crossing rate (max acc >= 0.7) by discount factor —
  #     "emergence" appears only on the drifted/no-discount rungs
  lad_dir <- file.path(res_dir, 'icrl_td_gamma_ladder')
  gam_rows <- bind_rows(lapply(c('gamma0p0_T40', 'gamma0p3_T40', 'gamma0p5_T40',
                                 'gamma0p7_T40'), function(g) {
    fs <- Sys.glob(file.path(lad_dir, sprintf('%s_s*.jsonl', g)))
    bind_rows(lapply(fs, function(f) {
      mx <- -Inf
      for (l in readLines(f, warn = FALSE)) {
        d <- tryCatch(fromJSON(l, simplifyVector = FALSE), error = function(e) NULL)
        if (!is.null(d) && !is.null(d$val_acc)) mx <- max(mx, d$val_acc)
      }
      tibble(gamma = g, cross = mx >= 0.7)
    }))
  }))
  gam_summ <- gam_rows %>% group_by(gamma) %>%
    summarise(rate = mean(cross), n = n(), .groups = 'drop') %>%
    mutate(glab = factor(sub('gamma0p(.*)_T40', '\\1', gamma),
                         levels = c('0p0', '0p3', '0p5', '0p7')))
  gam_summ <- bind_rows(gam_summ,
    tibble(gamma = 'gamma0p5_grid', rate = sum(df$max_acc >= 0.7) / nrow(df),
           n = nrow(df), glab = factor('0p5-grid', levels = c('0p0', '0p3', '0p5', '0p7', '0p5-grid'))))
  gam_summ$glab <- factor(gam_summ$glab,
                          levels = c('0p0', '0p3', '0p5', '0p7', '0p5-grid'))
  p_c <- ggplot(gam_summ, aes(glab, rate)) +
    geom_col(fill = '#5b21b6', width = 0.6, colour = '#1f2937', linewidth = 0.15) +
    geom_text(aes(label = sprintf('%d/%d', round(rate * n), n)), vjust = -0.4,
              size = 2.7, colour = '#1a1a1a') +
    coord_cartesian(ylim = c(0, 1.05)) +
    labs(title = '(c) Crossing rate by discount gamma',
         x = 'discount factor (0p5-grid = this arm)', y = 'rate of max acc >= 0.7') +
    theme_paper(9)

  # (d) budget-lever exhaustion: best-of-60 per budget, still below the bar
  exh <- df %>% group_by(budget) %>%
    summarise(best = max(max_acc), n = n(), .groups = 'drop')
  p_d <- ggplot(exh, aes(budget, best)) +
    geom_hline(yintercept = 0.7, linetype = '33', colour = '#047857', linewidth = 0.45) +
    geom_col(fill = '#5b21b6', width = 0.5, colour = '#1f2937', linewidth = 0.15) +
    geom_text(aes(label = sprintf('%.3f\n(0/%d >= 0.7)', best, n)), vjust = -0.4,
              size = 2.7, colour = '#1a1a1a', lineheight = 0.9) +
    scale_y_continuous(limits = c(0, 0.9), breaks = seq(0, 0.8, 0.2)) +
    labs(title = '(d) Best-of-grid per budget (lever exhausted)',
         x = NULL, y = 'best max accuracy') +
    theme_paper(9) +
    theme(axis.text.x = element_text(size = rel(0.74)))

  save_png(compose_E2_four_panel(list(p, p_b, p_c, p_d), 'E2_budget_grid', 7.2, 5.4, root), 'E2_budget_grid', w = 7.2, h = 5.4)
}

cat(sprintf('root = %s\n', root))
make_td_grid()
make_induction()
make_walsh()
make_routemat()
make_budget_grid()

cat('\n=== verification ===\n')
for (nm in c('E2_td_grid', 'E2_induction', 'E2_walsh', 'E2_routemat', 'E2_budget_grid')) {
  p <- file.path(fig_dir, paste0(nm, '.png'))
  info <- file.info(p)
  cat(sprintf('  %-20s  %s\n', nm,
              ifelse(is.na(info$size), 'MISSING', paste0(formatC(unname(info$size), big.mark = ','), ' bytes'))))
}
cat('=== done ===\n')
