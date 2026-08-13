#!/usr/bin/env Rscript
# make_E2_figs_r.R — professional R/ggplot2 replacements for Paper E2's 4 matplotlib figures.
# Figures: E2_case, E2_curriculum, E2_grid, E2_positive_rescue
# Style: matches make_evidence_figures_r.R + make_final_polish_figures_r.R
# Output: papers/figs/<name>.png  (ragg 300dpi) + papers/figs/evidence_r/<name>.svg

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
  library(svglite)
})

root    <- normalizePath(file.path(getwd()))
fig_dir <- file.path(root, 'papers', 'figs')
evd_dir <- file.path(fig_dir, 'evidence_r')
res_dir <- file.path(root, 'experiments', 'results')
dir.create(evd_dir, recursive = TRUE, showWarnings = FALSE)

# vector-figure emitter (tikz/PDF) for the LuaLaTeX pipeline
source(file.path(fig_dir, 'fig_pipeline.R'))
source(file.path(fig_dir, 'E2_panel_contract.R'))

# ----- shared theme (shared R style) -----------------------------------
# Typography standard: only panel tags (a)(b)(c) bold; all other text plain.
# Target sizes at 6.5in linewidth: tag 11 bold, title 10-11 plain,
# axis.title 9, axis.text 8.5, legend 8.5, annot 8-8.5.
theme_paper <- function(base = 11) {
  theme_minimal(base_size = base, base_family = 'TeX Gyre Termes') +
    theme(
      text             = element_text(family = 'TeX Gyre Termes'),
      plot.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.97), margin = margin(b = 3)),
      plot.subtitle    = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.86), colour = '#1a1a1a', margin = margin(b = 5)),
      axis.title       = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.88)),
      axis.text        = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.80), colour = '#1a1a1a'),
      legend.title     = element_blank(),
      legend.position  = 'top',
      legend.text      = element_text(family = 'TeX Gyre Termes', face = 'plain', size = rel(0.80)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.24, colour = '#E2E2E2'),
      strip.text       = element_text(family = 'TeX Gyre Termes', face = 'plain', colour = '#1a202c'),
      plot.margin      = margin(5.5, 7, 5.5, 7),
      plot.tag         = element_text(face = 'bold', size = 11),
      plot.tag.position = c(0.02, 0.98)
    )
}

# save PNG (300dpi ragg) + SVG (svglite)
save_both <- function(p, name, w = 6.5, h = 4.2) {
  png_path <- file.path(fig_dir, paste0(name, '.png'))
  svg_path <- file.path(evd_dir, paste0(name, '.svg'))
  ragg::agg_png(png_path, width = w, height = h, units = 'in', res = 300, scaling = 1, background = 'white')
  print(p); dev.off()
  svglite::svglite(svg_path, width = w, height = h, bg = 'white')
  print(p); dev.off()
  emit_vector(p, name, w, h)
  info <- file.info(png_path)
  cat(sprintf('  saved %-45s  (%s bytes)\n', png_path, formatC(info$size, big.mark = ',')))
  invisible(list(png = png_path, svg = svg_path, bytes = unname(info$size)))
}

# ============================================================
# 1. E2_case: single false-positive trace (adamw_T40_s3)
# ============================================================
make_e2_case <- function() {
  cat('--- E2_case ---\n')
  path <- file.path(res_dir, 'icrl_td', 'adamw_T40_s3.jsonl')
  lines <- readLines(path, warn = FALSE)

  rows <- lapply(lines, function(l) {
    d <- fromJSON(l, simplifyVector = FALSE)
    if (!is.null(d$step)) {
      tibble(step = as.integer(d$step),
             val_acc = as.numeric(d$val_acc))
    } else NULL
  })
  df <- bind_rows(Filter(Negate(is.null), rows))

  peak_row  <- df[which.max(df$val_acc), ]
  final_row <- df[nrow(df), ]

  # ---- detector logic (Derivation 1, threshold tau = 0.8) ----------------
  thr   <- 0.8
  chance <- 0.5625
  # single-crossing detector: any evaluation at or above threshold fires it.
  cross  <- df[df$val_acc >= thr, ]
  # sustained-K=2 detector: requires two CONSECUTIVE evaluations >= threshold.
  hi      <- df$val_acc >= thr
  sust_k2 <- which(hi[-1] & hi[-length(hi)])   # index i where eval i and i-1 both >= thr
  cat(sprintf('  peak: step=%d acc=%.4f\n', peak_row$step, peak_row$val_acc))
  cat(sprintf('  final: step=%d acc=%.4f\n', final_row$step, final_row$val_acc))
  cat(sprintf('  single-crossings >= %.2f : %d (first step %d)\n', thr, nrow(cross), min(cross$step)))
  cat(sprintf('  sustained-K=2 crossings  : %d\n', length(sust_k2)))

  p <- ggplot(df, aes(step, val_acc)) +
    # chance band: +/- one binomial SE (0.089) around the empirical chance level
    annotate('rect', xmin = -Inf, xmax = Inf, ymin = chance - 0.089, ymax = chance + 0.089,
             fill = '#9ca3af', alpha = 0.16) +
    geom_hline(yintercept = chance, linetype = 'dotted', colour = '#4b5563', linewidth = 0.45) +
    geom_hline(yintercept = thr,    linetype = 'dashed', colour = '#b91c1c', linewidth = 0.6) +
    geom_line(colour = '#0072B2', linewidth = 0.7) +
    # every single-crossing spike (open red circles) -- the events the single detector fires on
    geom_point(data = cross, colour = '#b91c1c', size = 2.4, shape = 1, stroke = 0.9) +
    # the step-150 peak, the first crossing
    geom_point(data = peak_row, colour = '#D55E00', size = 3.4, shape = 18) +
    annotate('text', x = peak_row$step + 350, y = peak_row$val_acc + 0.018,
             label = sprintf('peak %.3f (step %d)', peak_row$val_acc, peak_row$step),
             hjust = 0, size = 2.9, colour = 'black') +
    annotate('text', x = 3000, y = thr + 0.018,
             label = 'detector threshold 0.8', size = 2.9, colour = 'black', hjust = 0) +
    annotate('text', x = 3000, y = chance - 0.115,
             label = 'chance 0.5625 (+/- 1 SE)', size = 2.9, colour = 'black', hjust = 0) +
    annotate('label', x = final_row$step, y = final_row$val_acc - 0.045,
             label = sprintf('final %.4f', final_row$val_acc), size = 2.9, colour = '#1a1a1a',
             hjust = 1, fill = 'white', label.size = 0,
             label.padding = unit(0.07, 'lines')) +
    scale_x_continuous(labels = comma_format(), breaks = seq(0, 10000, 2000),
                       expand = expansion(mult = c(0.02, 0.06))) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0.3, 0.95), breaks = seq(0.3, 0.9, 0.1)) +
    labs(
      title = '(a) Validation accuracy vs training step (detector thresholds)',
      x     = 'training step',
      y     = 'validation accuracy'
    ) +
    theme_paper(10)

  # (b) zoom on the crossing cluster: the false positive is a transient spike
  zwin <- df %>% filter(step >= max(0, peak_row$step - 1500),
                        step <= peak_row$step + 2500)
  pz <- ggplot(zwin, aes(step, val_acc)) +
    annotate('rect', xmin = -Inf, xmax = Inf, ymin = chance - 0.089, ymax = chance + 0.089,
             fill = '#9ca3af', alpha = 0.16) +
    geom_hline(yintercept = chance, linetype = 'dotted', colour = '#4b5563', linewidth = 0.45) +
    geom_hline(yintercept = thr, linetype = 'dashed', colour = '#b91c1c', linewidth = 0.6) +
    geom_line(colour = '#0072B2', linewidth = 0.8) +
    geom_point(data = zwin[zwin$val_acc >= thr, ], colour = '#b91c1c', size = 2.4,
               shape = 1, stroke = 0.9) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(title = '(b) Zoom: the firing is a transient',
         x = 'training step', y = 'validation accuracy') +
    theme_paper(10)

  # (c) run-length distribution vs the geometric null: observed longest run at
  #     the bar vs what a memoryless checkpoint process produces at the same q
  q_hat <- mean(df$val_acc >= thr)
  runs  <- rle(df$val_acc >= thr)$lengths[rle(df$val_acc >= thr)$values]
  longest <- if (length(runs)) max(runs) else 0L
  sim <- replicate(20000, {
    x <- rle(runif(nrow(df)) < q_hat)$lengths
    v <- rle(runif(nrow(df)) < q_hat)$values
    if (any(v)) max(x[v]) else 0L
  })
  rl_df <- data.frame(x = factor(c('observed', 'geometric null (sim)'),
                                 levels = c('observed', 'geometric null (sim)')),
                      y = c(longest, median(sim)))
  pc <- ggplot(rl_df, aes(x, y)) +
    geom_col(fill = '#0072B2', width = 0.5) +
    geom_text(aes(label = sprintf('%d', round(y))), vjust = -0.4, size = 3,
              colour = '#1a1a1a') +
    labs(title = sprintf('(c) Longest run at the bar (q = %.4f)', q_hat),
         x = NULL, y = 'Consecutive checkpoints >= threshold') +
    theme_paper(10)

  # (d) sustained-K rejection: every single-crossing spike fails K >= 2, so the
  #     sustained criterion never fires on this trajectory
  n_hi <- length(hi)
  k2_pass <- any(hi[2:n_hi] & hi[1:(n_hi - 1)])
  k3_pass <- any(hi[3:n_hi] & hi[2:(n_hi - 1)] & hi[1:(n_hi - 2)])
  kd <- data.frame(K = factor(c('K = 1', 'K = 2', 'K = 3'), levels = c('K = 1', 'K = 2', 'K = 3')),
                   fires = c(nrow(cross) > 0, k2_pass, k3_pass))
  pd <- ggplot(kd, aes(K, as.integer(fires))) +
    geom_col(fill = c('#b91c1c', '#3a8a3a', '#3a8a3a'), width = 0.5) +
    geom_text(aes(label = ifelse(fires, 'fires', 'silent')), vjust = -0.4, size = 3,
              colour = '#1a1a1a') +
    coord_cartesian(ylim = c(0, 1.25)) +
    labs(title = '(d) Sustained-K rejects the false positive',
         x = NULL, y = 'Detector fires?') +
    theme_paper(10)

  save_both(compose_E2_four_panel(list(p, pz, pc, pd), 'E2_case', 7.2, 5.4, root),
            'E2_case', w = 7.2, h = 5.4)
}

# ============================================================
# 2. E2_curriculum: data ordering does NOT compress delay
# ============================================================
make_e2_curriculum <- function() {
  cat('--- E2_curriculum ---\n')
  files <- Sys.glob(file.path(res_dir, 'curriculum_order', '*.jsonl'))

  parse_file <- function(f) {
    name <- basename(f)
    pat <- '^(copy|walsh|modadd)_(iid|easy_to_hard|hard_first|structured)_(adamw|sgdm)_s(\\d+)\\.jsonl$'
    if (!grepl(pat, name)) return(NULL)
    fam   <- sub(pat, '\\1', name)
    order <- sub(pat, '\\2', name)
    opt   <- sub(pat, '\\3', name)
    seed  <- as.integer(sub(pat, '\\4', name))
    lines  <- readLines(f, warn = FALSE)
    last_d <- fromJSON(lines[length(lines)], simplifyVector = FALSE)
    s      <- last_d[['_summary']]
    estep  <- if (!is.null(s) && !is.null(s$emergence_step)) as.numeric(s$emergence_step) else NA_real_
    tibble(family = fam, ordering = order, optimizer = opt, seed = seed, emergence_step = estep)
  }

  raw <- bind_rows(lapply(files, parse_file)) %>% filter(!is.na(emergence_step))

  df <- raw %>%
    filter(family == 'copy') %>%
    mutate(
      ordering  = factor(ordering, levels = c('iid', 'easy_to_hard', 'hard_first', 'structured'),
                         labels = c('iid', 'easy→hard', 'hard first', 'structured')),
      optimizer = factor(optimizer, levels = c('adamw', 'sgdm'))
    )

  summ <- df %>%
    group_by(ordering, optimizer) %>%
    summarise(mean = mean(emergence_step), sd = sd(emergence_step),
              .groups = 'drop')

  base_adamw_iid <- mean(df$emergence_step[df$optimizer == 'adamw' & df$ordering == 'iid'])
  cat(sprintf('  iid-AdamW baseline: %.0f\n', base_adamw_iid))

  pal2 <- c(adamw = '#0072B2', sgdm = '#999999')
  pd   <- position_dodge(width = 0.55)

  p <- ggplot(summ, aes(ordering, mean, colour = optimizer)) +
    geom_hline(yintercept = base_adamw_iid, linetype = 'dotted',
               colour = '#0072B2', linewidth = 1.0) +
    geom_point(data = df, aes(ordering, emergence_step, colour = optimizer),
               position = position_dodge(width = 0.55),
               size = 1.4, alpha = 0.35, shape = 16) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                  position = pd, width = 0.18, linewidth = 0.55) +
    geom_point(position = pd, size = 2.6, shape = 18) +
    annotate('text', x = 2.5, y = 2060,
             label = sprintf('iid-AdamW baseline (%d)', round(base_adamw_iid)),
             hjust = 0.5, size = 2.8, colour = 'black') +
    scale_fill_manual(values = pal2, labels = c(adamw = 'AdamW', sgdm = 'SGDM'))  +
    scale_colour_manual(values = pal2, labels = c(adamw = 'AdamW', sgdm = 'SGDM')) +
    # Data span ~1700-2000; the earlier zero-based axis left ~85% of the
    # panel empty. The zoomed range keeps the null message honest via the
    # dotted baseline + sd error bars.
    coord_cartesian(ylim = c(1550, 2100)) +
    scale_y_continuous(breaks = seq(1600, 2000, 100)) +
    labs(
      title = '(a) Induction emergence step by data ordering and optimizer',
      x     = 'data ordering',
      y     = 'emergence step'
    ) +
    theme_paper(10) +
    theme(legend.position = 'top')

  # (b) per-seed strips by ordering (the distributions behind the mean +/- sd)
  p_b <- ggplot(df, aes(ordering, emergence_step, colour = optimizer)) +
    stat_summary(fun = median, geom = 'crossbar', width = 0.5, colour = '#1a1a1a',
                 position = position_dodge(width = 0.55)) +
    geom_point(position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.55,
                                               seed = 81),
               size = 1.6, alpha = 0.7) +
    scale_colour_manual(values = pal2, labels = c(adamw = 'AdamW', sgdm = 'SGDM')) +
    coord_cartesian(ylim = c(1550, 2100)) +
    labs(title = '(b) Per-seed emergence steps',
         x = 'data ordering', y = 'emergence step') +
    theme_paper(10) +
    theme(legend.position = 'none')

  # (c) iid-fastest effect: per-optimizer delta vs its own iid baseline
  deltas <- df %>%
    group_by(optimizer, ordering) %>%
    summarise(delta = mean(emergence_step) -
                mean(df$emergence_step[df$optimizer == optimizer[1] & df$ordering == 'iid']),
              .groups = 'drop') %>%
    filter(ordering != 'iid')
  p_c <- ggplot(deltas, aes(ordering, delta, fill = optimizer)) +
    geom_hline(yintercept = 0, linetype = 'dotted', colour = '#4b5563', linewidth = 0.6) +
    geom_col(position = position_dodge(width = 0.6), width = 0.52) +
    geom_text(aes(label = sprintf('%+.0f', delta), group = optimizer),
              position = position_dodge(width = 0.6), vjust = -0.4, size = 2.7,
              colour = '#1a1a1a', show.legend = FALSE) +
    scale_fill_manual(values = pal2, labels = c(adamw = 'AdamW', sgdm = 'SGDM')) +
    labs(title = '(c) Delta vs iid baseline (no compression)',
         x = 'data ordering', y = 'delta emergence step') +
    theme_paper(10) +
    theme(legend.position = 'top')

  # (d) all three task families: the no-compression verdict is family-robust
  fam_summ <- raw %>%
    group_by(family, ordering, optimizer) %>%
    summarise(mean = mean(emergence_step), .groups = 'drop') %>%
    mutate(family = factor(family, levels = c('copy', 'walsh', 'modadd')),
           ordering = factor(ordering,
                             levels = c('iid', 'easy_to_hard', 'hard_first', 'structured'),
                             labels = c('iid', 'easy→hard', 'hard first', 'structured')))
  p_d <- ggplot(fam_summ, aes(ordering, mean, colour = optimizer, group = optimizer)) +
    geom_line(linewidth = 0.8, position = position_dodge(width = 0.3)) +
    geom_point(size = 2.2, position = position_dodge(width = 0.3)) +
    facet_wrap(~ family, nrow = 1) +
    scale_colour_manual(values = pal2, labels = c(adamw = 'AdamW', sgdm = 'SGDM')) +
    labs(title = '(d) Family-robustness (120-run grid)',
         x = 'data ordering', y = 'emergence step') +
    theme_paper(10) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1, size = 7.5),
          legend.position = 'top')

  save_both(compose_E2_four_panel(list(p, p_b, p_c, p_d), 'E2_curriculum', 7.2, 5.6, root),
            'E2_curriculum', w = 7.2, h = 5.6)
}

# ============================================================
# 3. E2_grid: run-grids behind the three negatives
# ============================================================
make_e2_grid <- function() {
  cat('--- E2_grid ---\n')

  # ---- panel A: curriculum ordering (log scale) ----
  parse_curric <- function(f) {
    name <- basename(f)
    pat <- '^(copy|walsh|modadd)_(iid|easy_to_hard|hard_first|structured)_(adamw|sgdm)_s(\\d+)\\.jsonl$'
    if (!grepl(pat, name)) return(NULL)
    fam   <- sub(pat, '\\1', name)
    order <- sub(pat, '\\2', name)
    lines <- readLines(f, warn = FALSE)
    last_d <- fromJSON(lines[length(lines)], simplifyVector = FALSE)
    s <- last_d[['_summary']]
    estep <- if (!is.null(s) && !is.null(s$emergence_step)) as.numeric(s$emergence_step) else NA_real_
    tibble(family = fam, ordering = order, emergence_step = estep)
  }
  curric_raw <- bind_rows(lapply(Sys.glob(file.path(res_dir, 'curriculum_order', '*.jsonl')), parse_curric)) %>%
    filter(!is.na(emergence_step))

  ORDER_LEVELS  <- c('iid', 'easy_to_hard', 'hard_first', 'structured')
  ORDER_LABELS  <- c('iid', 'easy→hard', 'hard first', 'structured')

  pA_df <- curric_raw %>%
    filter(family %in% c('copy', 'modadd')) %>%
    mutate(
      ordering = factor(ordering, levels = ORDER_LEVELS, labels = ORDER_LABELS),
      family   = factor(family, levels = c('copy', 'modadd'))
    ) %>%
    group_by(family, ordering) %>%
    summarise(med = median(emergence_step), .groups = 'drop')

  scat_A <- curric_raw %>%
    filter(family %in% c('copy', 'modadd')) %>%
    mutate(ordering = factor(ordering, levels = ORDER_LEVELS, labels = ORDER_LABELS),
           family   = factor(family, levels = c('copy', 'modadd')))

  copy_iid_med <- median(curric_raw$emergence_step[curric_raw$family == 'copy' & curric_raw$ordering == 'iid'])

  # Task-family colours: E2 supplementary (non-optimizer) scheme — violet
  # '#5b21b6' / pink '#CC79A7' — deliberately off the optimizer palette
  # (AdamW blue, Muon orange, SGDM gray) so "copy" is not confused with
  # AdamW elsewhere. Shared with E2_supervised_fit and
  # E2_positive_rescue.
  fam_pal <- c(copy = '#CC79A7', modadd = '#5b21b6')

  pA <- ggplot(pA_df, aes(ordering, med, fill = family)) +
    geom_col(position = position_dodge(0.6), width = 0.52, alpha = 0.88) +
    geom_jitter(data = scat_A, aes(ordering, emergence_step, colour = family),
                position = position_dodge(0.6), size = 1.5, alpha = 0.4, show.legend = FALSE) +
    geom_hline(yintercept = copy_iid_med, linetype = 'dotted', colour = '#CC79A7', linewidth = 0.9) +
    annotate('text', x = 0.6, y = 3600,
             label = 'copy iid baseline', hjust = 0, size = 2.9, colour = 'black') +
    annotate('text', x = 4.45, y = 330,
             label = 'walsh: no emergence\nwithin budget',
             size = 2.6, colour = 'black', hjust = 1, lineheight = 0.95) +
    scale_fill_manual(values = fam_pal) +
    scale_colour_manual(values = fam_pal) +
    # LESSONS 3: limits= inside the scale DROPS the geom_col rows entirely
    # (bar base 0 -> -Inf on log10); zoom with coord_cartesian instead so the
    # median bars actually render (they were silently censored before).
    scale_y_log10(breaks = c(50, 100, 200, 500, 1000, 2000, 4000),
                  labels = comma_format()) +
    coord_cartesian(ylim = c(30, 4500)) +
    labs(
      title = 'Emergence step by ordering',
      x     = 'data ordering',
      y     = 'emergence step (log)'
    ) +
    theme_paper(11) +
    theme(legend.position = 'top', axis.text.x = element_text(size = 9.5))

  # ---- panel B: in-context TD heatmap ----
  parse_td <- function(f) {
    name <- basename(f)
    pat <- '^(adamw|muon|sgdm)_T(\\d+)_s(\\d+)\\.jsonl$'
    if (!grepl(pat, name)) return(NULL)
    opt <- sub(pat, '\\1', name)
    T   <- as.integer(sub(pat, '\\2', name))
    lines <- readLines(f, warn = FALSE)
    last_d <- fromJSON(lines[length(lines)], simplifyVector = FALSE)
    s <- last_d[['_summary']]
    acc <- if (!is.null(s) && !is.null(s$final_val_acc)) as.numeric(s$final_val_acc) else NA_real_
    tibble(optimizer = opt, T = T, val_acc = acc)
  }
  td_raw <- bind_rows(lapply(Sys.glob(file.path(res_dir, 'icrl_td', '*.jsonl')), parse_td)) %>%
    filter(!is.na(val_acc))

  # Overlay = single-crossing ≥0.8 (first_ge_0.8_K1), matching Table 3 (26/45).
  # Final val_acc>0.8 is a different event (1/45) and must not be used here.
  calib08 <- jsonlite::fromJSON(file.path(res_dir, 'icrl_td_ultragoal_calib',
                                         'icrl_td_calibration_verdict.json'),
                               simplifyVector = FALSE)
  fire08 <- bind_rows(lapply(calib08$runs, function(r) {
    tibble(optimizer = r$optimizer, T = as.integer(r$T),
           hit = !is.null(r[['first_ge_0.8_K1']]))
  })) %>%
    group_by(optimizer, T) %>%
    summarise(nhit = sum(hit), n = n(), .groups = 'drop')

  cat(sprintf('  icrl_td: n=%d, median=%.4f, single-cross ≥0.8: %d/%d (final>0.8: %d/%d)\n',
              nrow(td_raw), median(td_raw$val_acc),
              sum(fire08$nhit), sum(fire08$n),
              sum(td_raw$val_acc > 0.8), nrow(td_raw)))

  OPTS <- c('adamw', 'muon', 'sgdm')
  TS   <- c(10, 20, 40)

  pB_df <- td_raw %>%
    group_by(optimizer, T) %>%
    summarise(
      med_acc = median(val_acc),
      .groups = 'drop'
    ) %>%
    left_join(fire08, by = c('optimizer', 'T')) %>%
    mutate(
      optimizer = factor(optimizer, levels = OPTS, labels = c('AdamW', 'Muon', 'SGDM')),
      T         = factor(T, levels = TS, labels = paste0('T=', TS)),
      label     = sprintf('%.2f\n%d/%d', med_acc, nhit, n)
    )
  stopifnot(sum(pB_df$nhit) == 26L, sum(pB_df$n) == 45L)

  pB <- ggplot(pB_df, aes(T, optimizer, fill = med_acc)) +
    geom_tile(colour = 'white', linewidth = 1.0) +
    geom_text(aes(label = label), size = 3.0, lineheight = 0.90) +
    scale_fill_gradient2(low = '#f7fbff', mid = '#9ecae1', high = '#08519c',
                         midpoint = 0.575, limits = c(0.5, 0.65),
                         name = 'median\nval acc',
                         labels = percent_format(accuracy = 1)) +
    labs(
      title = 'In-context TD accuracy by horizon',
      x     = 'context horizon T',
      y     = 'optimizer'
    ) +
    theme_paper(11) +
    theme(legend.position = 'right',
          panel.grid.major = element_blank(),
          axis.text = element_text(size = 10))

  # ---- panel C: early-window AUROC ----
  spec_txt <- paste(readLines(file.path(res_dir, 'spec_route', 'posthoc_round1.json'), warn = FALSE),
                    collapse = '\n')
  spec_txt <- gsub('\\bNaN\\b', 'null', spec_txt)
  spec     <- fromJSON(spec_txt, simplifyVector = FALSE)

  pC_df <- bind_rows(
    tibble(window = '200-step\nwindow',   cohort = 'pooled (all opts)',
           auroc  = spec[['W-200/tierA']][['route/config']][['auroc']]),
    tibble(window = '200-step\nwindow',   cohort = 'within AdamW',
           auroc  = spec[['W-200/tierA']][['route/config_adamw']][['auroc']]),
    tibble(window = 'mem-onset\nwindow',  cohort = 'pooled (all opts)',
           auroc  = spec[['W-mem/tierA']][['route/config']][['auroc']]),
    tibble(window = 'mem-onset\nwindow',  cohort = 'within AdamW',
           auroc  = spec[['W-mem/tierA']][['route/config_adamw']][['auroc']])
  ) %>%
    mutate(cohort = factor(cohort, levels = c('pooled (all opts)', 'within AdamW')),
           # lift bar labels clear of the dotted chance line at 0.5: bars whose
           # value sits near/below chance get their label floated above the line.
           labely = ifelse(auroc >= 0.55, auroc + 0.03, 0.56))

  cat('  014 AUROC:\n')
  print(as.data.frame(pC_df))

  pC <- ggplot(pC_df, aes(window, auroc, fill = cohort)) +
    geom_col(position = position_dodge(0.65), width = 0.55,
             colour = '#1f2937', linewidth = 0.2) +
    geom_hline(yintercept = 0.5, linetype = 'dotted', colour = '#4b5563', linewidth = 0.55) +
    geom_text(aes(y = labely, label = sprintf('%.2f', auroc)),
              position = position_dodge(0.65), vjust = 0, size = 3.5) +
    # far-left margin, below the dotted line — clear of all four bars
    annotate('text', x = 0.45, y = 0.465, label = 'chance', size = 3.0,
             colour = 'black', hjust = 0) +
    scale_fill_manual(values = c('pooled (all opts)' = '#0072B2', 'within AdamW' = '#D62728')) +
    scale_y_continuous(breaks = seq(0.4, 0.9, 0.1),
                       labels = number_format(accuracy = 0.1)) +
    coord_cartesian(ylim = c(0.4, 0.93)) +
    labs(
      title = 'Route-prediction AUROC by window',
      x     = 'early window',
      y     = 'route-prediction AUROC'
    ) +
    theme_paper(11) +
    theme(legend.position = 'top',
          axis.text.x = element_text(size = 10))

  # ---- compose with patchwork tag system ----
  comp <- (pA / pB / pC) +
    plot_annotation(
      tag_levels = 'a',
      tag_prefix = '(',
      tag_suffix = ')'
    ) &
    theme_paper(11) &
    theme(
      plot.tag          = element_text(face = 'bold', size = 10),
      plot.tag.position = 'topleft'
    )

  save_both(comp, 'E2_grid', w = 6.5, h = 7.0)
}

# ============================================================
# 4. E2_positive_rescue
# ============================================================
make_e2_positive_rescue <- function() {
  cat('--- E2_positive_rescue ---\n')

  st_path <- file.path(root, 'experiments','results','figures-redteam', 'redteam_e2_stats.json')
  st <- fromJSON(st_path, simplifyVector = FALSE)

  `%||%` <- function(x, y) if (is.null(x)) y else x

  rows_all <- c(st$fullattack, st$redteam)
  df <- bind_rows(lapply(rows_all, function(x) {
    tibble(
      kind    = x$kind,
      n_cells = as.integer(x$n_cells),
      near    = as.integer(x$n_near_positive_0p7 %||% 0),
      best    = as.numeric(x$best_max_val_acc)
    )
  })) %>%
    mutate(kind = factor(kind, levels = c('main', 'ultra', 'wide')))

  upper <- as.numeric(st$zero_success_upper_95_if_no_success$upper95)
  n_tot <- as.integer(st$zero_success_upper_95_if_no_success$n)

  cat(sprintf('  rescue families: %s\n', paste(levels(df$kind), collapse = ', ')))
  cat(sprintf('  best per family: %s\n', paste(sprintf('%.3f', df$best), collapse = ', ')))
  cat(sprintf('  upper95=%.4f  n=%d\n', upper, n_tot))

  # E2 supplementary (non-optimizer) scheme: violet / light violet (tier shade)
  # / pink — keeps rescue families off the optimizer palette (blue/orange/gray)
  kind_pal <- c(main = '#5b21b6', ultra = '#a78bfa', wide = '#CC79A7')

  p1 <- ggplot(df, aes(kind, best, fill = kind)) +
    geom_col(width = 0.52, colour = '#1f2937', linewidth = 0.25, show.legend = FALSE) +
    geom_hline(yintercept = 0.8,  linetype = 'dashed', colour = '#b91c1c', linewidth = 0.6) +
    geom_hline(yintercept = 0.7,  linetype = '33',     colour = '#047857', linewidth = 0.55) +
    geom_text(aes(label = paste0('n=', n_cells)),
              vjust = 1.3, size = 3.0, colour = 'white') +
    annotate('text', x = 0.55, y = 0.818, label = 'positive (0.8)',
             size = 2.6, colour = 'black', hjust = 0) +
    annotate('text', x = 0.55, y = 0.682, label = 'near-positive (0.7)',
             size = 2.6, colour = 'black', hjust = 0) +
    scale_fill_manual(values = kind_pal) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, 0.9)) +
    labs(
      title = 'Best max validation accuracy',
      x     = 'rescue family',
      y     = 'best max validation accuracy'
    ) +
    theme_paper(11)

  bound_df <- tibble(
    label = sprintf('0/%d successes', n_tot),
    upper = upper
  )
  p2 <- ggplot(bound_df, aes(label, upper)) +
    geom_col(width = 0.42, fill = '#5b21b6', colour = '#1f2937', linewidth = 0.25) +  # supplementary violet (avoid Muon orange)
    geom_text(aes(label = sprintf('%.3f', upper)), vjust = -0.45, size = 3.2) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 0.22)) +
    labs(title = 'Zero-success upper bound',
         x = NULL, y = '95% upper bound') +
    theme_paper(10)

  # (c) near-positive (0.7) counts per family: the wide family comes closest,
  #     but near-positive is not a positive control
  p3 <- ggplot(df, aes(kind, near, fill = kind)) +
    geom_col(width = 0.52, colour = '#1f2937', linewidth = 0.25, show.legend = FALSE) +
    geom_text(aes(label = paste0(near, '/', n_cells)), vjust = -0.4, size = 3.0,
              colour = 'black') +
    coord_cartesian(ylim = c(0, 3.6)) +
    scale_fill_manual(values = kind_pal) +
    labs(title = 'Near-positive cells (acc >= 0.7)',
         x = 'rescue family', y = 'cells near positive') +
    theme_paper(10)

  # (d) pool provenance: cells pooled across both attack rounds, zero successes.
  # Reader-facing round labels (the underlying campaigns are internal codenames).
  pool_df <- df %>%
    mutate(round = ifelse(kind %in% c('main', 'ultra'), 'round 1', 'round 2'))
  p4 <- ggplot(pool_df, aes(round, n_cells, fill = kind)) +
    geom_col(width = 0.55, colour = '#1f2937', linewidth = 0.2) +
    geom_text(aes(label = n_cells, group = kind),
              position = position_stack(vjust = 0.5), size = 2.9, colour = 'white') +
    annotate('text', x = 1.5, y = n_tot * 1.03,
             label = sprintf('0/%d successes\n(upper 95%% bound %.3f)', n_tot, upper),
             size = 2.6, colour = 'black', hjust = 0.5, lineheight = 0.95) +
    scale_fill_manual(values = kind_pal, name = NULL) +
    coord_cartesian(ylim = c(0, n_tot * 1.15)) +
    labs(title = 'Pooled rescue pool (both attack rounds)',
         x = 'attack round', y = 'cells') +
    theme_paper(10) +
    theme(legend.position = 'top')

  comp <- compose_E2_four_panel(list(p1, p2, p3, p4), 'E2_positive_rescue', 7.2, 5.4, root)
  save_both(comp, 'E2_positive_rescue', w = 7.2, h = 5.4)
}

# ============================================================
# Run all four
# ============================================================
cat('\n=== make_E2_figs_r.R ===\n')
make_e2_case()
make_e2_curriculum()
make_e2_grid()
make_e2_positive_rescue()

cat('\n=== verification ===\n')
targets <- c('E2_case', 'E2_curriculum', 'E2_grid', 'E2_positive_rescue')
for (nm in targets) {
  p <- file.path(fig_dir, paste0(nm, '.png'))
  s <- file.path(evd_dir, paste0(nm, '.svg'))
  info <- file.info(p)
  cat(sprintf('  %-40s  PNG: %s bytes  SVG: %s\n',
              nm,
              ifelse(is.na(info$size), 'MISSING', formatC(unname(info$size), big.mark = ',')),
              ifelse(file.exists(s), 'OK', 'MISSING')))
}
cat('=== done ===\n')
