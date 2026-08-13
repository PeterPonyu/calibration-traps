suppressPackageStartupMessages({library(jsonlite); library(ggplot2); library(dplyr); library(tidyr); library(scales); library(patchwork); library(svglite); library(ragg)})
figdir <- 'papers/figs'; evdir <- 'experiments/results/figures-deepcheck'; svgdir <- 'papers/figs/evidence_r'; dir.create(figdir, showWarnings=FALSE, recursive=TRUE); dir.create(svgdir, showWarnings=FALSE, recursive=TRUE)
theme_paper <- function(base=9) theme_minimal(base_size=base, base_family='TeX Gyre Termes') + theme(
  panel.grid.minor=element_blank(),
  plot.title=element_text(face='plain', size=base+1),
  plot.subtitle=element_blank(),
  strip.text=element_text(face='plain', size=base),
  legend.position='bottom', legend.text=element_text(size=base-1), legend.title=element_text(size=base-1),
  axis.text=element_text(size=base-1), axis.title=element_text(size=base),
  plot.tag=element_text(face='bold', size=11), plot.tag.position=c(0.02, 0.98)
)
source(file.path(figdir, 'fig_pipeline.R'))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(figdir, 'E1_panel_contract.R'))
source(file.path(figdir, 'E2_panel_contract.R'))
save_both <- function(p, name, w=5.0, h=3.5){ pngp <- file.path(figdir, paste0(name,'.png')); svgp <- file.path(svgdir, paste0(name,'.svg')); ragg::agg_png(pngp, width=w, height=h, units='in', res=300, scaling=1); print(p); dev.off(); svglite::svglite(svgp, width=w, height=h); print(p); dev.off(); emit_vector(p, name, w, h); cat('saved', pngp, svgp, '\n') }

# A: LMC short controls -- plot the actual per-seed loss barrier heights against
#    the no-merge (k*) threshold of 0.1, so "0 of 5 crossings" is shown with the
#    real barrier numbers (all ~5.3, far above the threshold). The evidence
#    summary only stores k_star=null, so barriers are re-aggregated from each
#    seed's raw per-spawn jsonl (the file paths the summary records).
A <- fromJSON(file.path(evdir,'a_lmc_deepcheck_summary.json'), simplifyVector=FALSE)
LMC_THRESH <- 0.1   # barrier < 0.1 == basins merge (k* hit); matches train_fork barrier_threshold
read_seed_barriers <- function(rec) {
  fp <- rec$file
  if (is.null(fp) || !file.exists(fp)) return(NULL)
  lines <- readLines(fp, warn=FALSE)
  lines <- lines[nchar(trimws(lines)) > 0]
  recs <- lapply(lines, function(l) tryCatch(fromJSON(l, simplifyVector=TRUE), error=function(e) NULL))
  bars <- do.call(rbind, lapply(recs, function(d) {
    if (is.null(d) || is.null(d$barrier) || is.null(d$spawn_step)) return(NULL)
    data.frame(spawn_step=as.numeric(d$spawn_step), barrier=as.numeric(d$barrier))
  }))
  if (is.null(bars) || nrow(bars)==0) return(NULL)
  bars$name <- ifelse(is.null(rec$name), basename(fp), rec$name)
  bars
}
a_df <- bind_rows(lapply(A$combined_muon_short_records, read_seed_barriers)) %>%
  mutate(label_clean=gsub('^muon_s', 's', name))
seed_order <- unique(a_df$label_clean[order(as.numeric(gsub('s','',a_df$label_clean)))])
a_df$label_clean <- factor(a_df$label_clean, levels=seed_order)
n_seeds <- length(seed_order)
ymax    <- max(a_df$barrier) * 1.12
n_meas  <- nrow(a_df)
n_below <- sum(a_df$barrier < LMC_THRESH)
med_bar <- median(a_df$barrier)
gap_x   <- med_bar / LMC_THRESH
# Encode the spawn-step dimension (each seed is measured at 0/50/100), quantify
# the barrier-to-threshold gap, and report the crossing count so the figure says
# more than "all points look alike" -- the sameness IS the robust no-merge result.
pA <- ggplot(a_df, aes(label_clean, barrier)) +
  geom_hline(yintercept=LMC_THRESH, linetype='dashed', colour='#b2182b', linewidth=.6) +
  annotate('text', x=seed_order[1], y=LMC_THRESH, label='no-merge / k* threshold (0.1)',
           hjust=0, vjust=-0.6, size=2.8, colour='black') +
  annotate('text', x=n_seeds+0.4, y=ymax, hjust=1, vjust=1, size=2.7, colour='black',
           label=sprintf('median barrier %.2f; %d/%d below 0.1', med_bar, n_below, n_meas)) +
  geom_point(aes(colour=label_clean, shape=factor(spawn_step)), size=2.6, alpha=.85,
             position=position_jitter(width=.12, height=0, seed=1)) +
  scale_shape_manual(values=c(16,17,15), name='spawn step') +
  scale_y_continuous(limits=c(0, ymax)) +
  scale_colour_brewer(palette='Dark2', guide='none') +
  labs(title='Inter-child LMC barrier vs no-merge threshold',
       x='fork seed', y='inter-child loss barrier') +
  theme_paper(9) +
  theme(legend.position='top')
save_both(pA, 'A_lmc_n5', 5.0, 3.2)

# B: EoS dense boundary (4-panel)
B <- fromJSON(file.path(evdir,'b_eos_dense_summary.json'))
b_df <- as.data.frame(B$rows) %>% mutate(loss=toupper(loss), label=paste0(nonfinite,'/',n), finite_rate=finite/n)
pB1 <- ggplot(b_df, aes(factor(lr), loss, fill=nonfinite/n)) +
  geom_tile(color='white', linewidth=.8) +
  geom_text(aes(label=label), size=3.2) +
  scale_fill_gradient(low='#e8f5e9', high='#b2182b', limits=c(0,1), labels=percent, name='nonfinite rate') +
  labs(title='(a) EoS boundary is loss-dependent', x='learning rate', y='loss') +
  theme_paper(9)
# (b) per-lr marginal finite rate
b_marg <- b_df %>% group_by(lr, loss) %>% summarise(rate=mean(finite_rate), .groups='drop')
pB2 <- ggplot(b_marg, aes(factor(lr), rate, fill=loss)) +
  geom_col(position=position_dodge(.75), width=.68, colour=NA) +
  scale_fill_manual(values=c(CE='#2166ac', MSE='#d73027'), name=NULL) +
  coord_cartesian(ylim=c(0,1.05)) +
  labs(title='(b) Marginal finite rate by lr', x='learning rate', y='finite rate') +
  theme_paper(9) + theme(legend.position='top')
# (c) sharpness level where the readout is finite (mean ratio vs the 2/eta clamp)
pB3 <- ggplot(b_df, aes(factor(lr), mean_ratio_finite, colour=loss, group=loss)) +
  geom_hline(yintercept=1, linetype='dotted', colour='#888') +
  geom_line(linewidth=.9) + geom_point(size=2.4) +
  scale_colour_manual(values=c(CE='#2166ac', MSE='#d73027'), name=NULL) +
  labs(title='(c) Ratio where finite (vs 2/eta clamp)', x='learning rate', y='mean ratio (finite runs)') +
  theme_paper(9) + theme(legend.position='top')
# (d) task honesty: mean test accuracy stays near chance in every cell
pB4 <- ggplot(b_df, aes(factor(lr), mean_test_acc, colour=loss, group=loss)) +
  geom_hline(yintercept=.5, linetype='dashed', colour='#888') +
  geom_line(linewidth=.9) + geom_point(size=2.4) +
  scale_colour_manual(values=c(CE='#2166ac', MSE='#d73027'), name=NULL) +
  coord_cartesian(ylim=c(0,1)) +
  labs(title='(d) Task stays unsolved (acc ~ 0.5)', x='learning rate', y='mean test accuracy') +
  theme_paper(9) + theme(legend.position='none')
pB <- (pB1 | pB2 | pB3 | pB4) + plot_layout(nrow=2)
save_both(pB, 'B_eos_dense', 7.2, 5.2)

# C: Scale seed recovery
C <- fromJSON(file.path(evdir,'c_scale_deepcheck_summary.json'), simplifyVector=FALSE)
c_vals <- bind_rows(lapply(C$rows, function(r) bind_rows(lapply(r$values, function(v) data.frame(optimizer=r$optimizer, layers=r$layers, seed=v$seed, deg4=v$deg4, fit=v$fit))))) %>% mutate(layer_label=paste0(layers,' layer'), strong=deg4>0.3)
pC <- ggplot(c_vals, aes(factor(seed), deg4, color=optimizer, shape=strong)) +
  geom_hline(yintercept=.3, linetype='dashed', color='grey45') +
  geom_hline(yintercept=0, color='grey75') +
  geom_point(size=2.8, stroke=0.9, position=position_dodge(width=.35)) +
  facet_wrap(~layer_label, nrow=1) +
  scale_color_manual(values=c(adamw='#0072B2', muon='#D55E00'),
                     labels=c(adamw='AdamW', muon='Muon')) +
  scale_shape_manual(values=c('TRUE'=17,'FALSE'=16),
                     labels=c('FALSE'='no', 'TRUE'='yes'), name='deg-4 corr > 0.3') +
  labs(title='Scale: sparse deg-4 recovery by seed', x='seed', y='degree-4 correlation') +
  theme_paper(9)
save_both(pC, 'C_scale_seeds', 5.0, 3.2)

# E1: Real-text capacity bridge (4-panel)
E1 <- fromJSON(file.path(evdir,'e1_capacity_bridge_summary.json'), simplifyVector=FALSE)
e1_df <- bind_rows(lapply(E1$records, function(r) bind_rows(lapply(r$rows, function(v) data.frame(capacity=r$capacity, seed=r$seed, n=v$n, excess=v$excess, val_loss=v$val_loss))))) %>% mutate(seed=factor(seed), n=factor(n, levels=c(1,2,4,10,20)))
e1_first_facet <- levels(factor(e1_df$capacity))[1]
e1_thresh_lab <- data.frame(capacity=e1_first_facet,
                            n=factor(1, levels=c(1,2,4,10,20)),
                            excess=c(.05, .10), lab=c('0.05', '0.10'))
pE1a <- ggplot(e1_df, aes(n, excess, group=seed, color=seed)) +
  geom_hline(yintercept=.05, linetype='dashed', color='grey40') +
  geom_hline(yintercept=.1, linetype='dotted', color='grey40') +
  geom_text(data=e1_thresh_lab, aes(x=n, y=excess, label=lab), color='black', size=2.7,
            hjust=0, vjust=-0.4, inherit.aes=FALSE) +
  geom_line(linewidth=0.75) +
  geom_point(size=2.2) +
  facet_wrap(~capacity, nrow=1) +
  scale_y_continuous(labels=number_format(accuracy=.01)) +
  labs(title='(a) Excess loss vs repeat count (large vs small capacity)', x='repeat count n', y='excess validation loss vs n=1') +
  theme_paper(9)
# (b) R_free by threshold (0.05 vs 0.10) per seed per capacity
e1_rf <- e1_df %>% group_by(capacity, seed) %>%
  summarise(R05 = max(as.integer(as.character(n))[excess < .05]),
            R10 = max(as.integer(as.character(n))[excess < .10]), .groups='drop') %>%
  pivot_longer(c(R05, R10), names_to='thr', values_to='R_free') %>%
  mutate(thr = recode(thr, R05='0.05 nats', R10='0.10 nats'))
pE1b <- ggplot(e1_rf, aes(seed, R_free, fill=thr)) +
  geom_col(position=position_dodge(.72), width=.62, colour='white', linewidth=.2) +
  facet_wrap(~capacity, nrow=1) +
  scale_fill_manual(values=c('0.05 nats'='#2563eb', '0.10 nats'='#1e3a8a'), name=NULL) +
  scale_y_continuous(breaks=c(0,4,10,20), limits=c(0,22)) +
  labs(title='(b) R_free by threshold', x=NULL, y='R_free (epochs)') +
  theme_paper(9) + theme(legend.position='top')
# (c) per-seed excess at the two decision rungs (n=4 and n=10)
e1_dec <- e1_df %>% filter(n %in% c(4,10))
pE1c <- ggplot(e1_dec, aes(n, excess, colour=seed, group=seed)) +
  geom_hline(yintercept=.05, linetype='dashed', color='grey40') +
  geom_line(linewidth=.8) + geom_point(size=2.4) +
  facet_wrap(~capacity, nrow=1) +
  scale_y_continuous(labels=number_format(accuracy=.01)) +
  labs(title='(c) Decision rungs n=4 / n=10', x='repeat count n', y='excess (nats)') +
  theme_paper(9) + theme(legend.position='none')
# (d) cost at n=20 (the consistently-costly end) per seed
 e1_20 <- e1_df %>% filter(n == 20)
pE1d <- ggplot(e1_20, aes(seed, excess, colour=seed)) +
  geom_point(size=2.6, show.legend=FALSE) +
  stat_summary(fun=median, geom='crossbar', width=.45, colour='#1a1a1a') +
  facet_wrap(~capacity, nrow=1) +
  labs(title='(d) Cost at n=20', x='seed', y='excess (nats)') +
  theme_paper(9) + theme(legend.position='none')
pE1 <- compose_E1_four_panel(list(pE1a, pE1b, pE1c, pE1d), 'E1_capacity_bridge', 7.2, 5.2, root = getwd())
save_both(pE1, 'E1_capacity_bridge', 7.2, 5.2)

# E2: Supervised fit sanity
E2 <- fromJSON(file.path(evdir,'e2_supervised_fit_summary.json'), simplifyVector=FALSE)
e2_df <- bind_rows(lapply(E2$records, function(r) data.frame(seed=r$seed, support=as.factor(r$support), support_acc=r$max_support_acc, fresh_acc=r$max_fresh_val_acc))) %>% pivot_longer(c(support_acc,fresh_acc), names_to='metric', values_to='acc') %>% mutate(metric=recode(metric, support_acc='finite support', fresh_acc='fresh MRP'))
make_e2_panel <- function(sup_val, panel_title, show_thresh_lab=FALSE) {
  d <- filter(e2_df, as.character(support)==as.character(sup_val))
  p <- ggplot(d, aes(factor(seed), acc, fill=metric)) +
    geom_col(position=position_dodge(width=.75), width=.68) +
    geom_hline(yintercept=.8, linetype='dashed', color='grey35')
  if (show_thresh_lab) p <- p +
    annotate('text', x=length(unique(d$seed))+0.45, y=.8,
             label='fit threshold 0.8',
             hjust=1, vjust=-0.4, size=2.7, colour='black')
  p +
    # E2 supplementary (non-optimizer) scheme: violet '#5b21b6' / pink '#CC79A7'
    # (shared with E2_grid panel a and E2_positive_rescue; deliberately
    # off the optimizer palette AdamW blue / Muon orange / SGDM gray)
    scale_fill_manual(values=c('finite support'='#5b21b6','fresh MRP'='#CC79A7'),
                      name='metric',
                      labels=c('finite support'='finite support', 'fresh MRP'='fresh MRP')) +
    scale_y_continuous(labels=percent, limits=c(0,1)) +
    labs(title=panel_title, x='seed', y='maximum validation accuracy') +
    theme_paper(9)
}
sup_vals <- sort(unique(as.numeric(as.character(e2_df$support))))
pE2a <- make_e2_panel(as.factor(sup_vals[1]), paste0('(a) Support = ', sup_vals[1]), show_thresh_lab=TRUE)
pE2b <- make_e2_panel(as.factor(sup_vals[2]), paste0('(b) Support = ', sup_vals[2]))
# (c) support-fit vs fresh-MRP gap per seed: fitting labelled support is not
#     TD emergence (the margin is the finding)
e2_wide <- e2_df %>% pivot_wider(names_from = metric, values_from = acc)
pE2c <- ggplot(e2_wide, aes(seed, `finite support` - `fresh MRP`, fill = `finite support` - `fresh MRP` > 0)) +
  geom_col(width = 0.6, colour = '#1f2937', linewidth = 0.15, show.legend = FALSE) +
  scale_fill_manual(values = c('TRUE' = '#5b21b6', 'FALSE' = '#CC79A7')) +
  facet_wrap(~ support, nrow = 1) +
  scale_y_continuous(labels = percent) +
  labs(title = '(c) Support-fit margin over fresh MRP', x = 'seed', y = 'accuracy margin') +
  theme_paper(9)
# (d) fit-threshold tally: support-fit >= 0.8 in most seeds, >= 0.9 in none —
#     a calibration boundary, not a solved reference
e2_tally <- e2_df %>% filter(metric == 'finite support') %>%
  group_by(support) %>%
  summarise(n = n(), ge80 = mean(acc >= 0.8), ge90 = mean(acc >= 0.9), .groups = 'drop')
e2_tally_l <- e2_tally %>% pivot_longer(c(ge80, ge90), names_to = 'thr', values_to = 'frac') %>%
  mutate(thr = recode(thr, ge80 = '>= 0.8', ge90 = '>= 0.9'))
pE2d <- ggplot(e2_tally_l, aes(thr, frac, fill = thr)) +
  geom_col(width = 0.55, colour = '#1f2937', linewidth = 0.15, show.legend = FALSE) +
  geom_text(aes(label = percent(frac, accuracy = 1)), vjust = -0.4, size = 2.8) +
  facet_wrap(~ support, nrow = 1) +
  scale_fill_manual(values = c('>= 0.8' = '#5b21b6', '>= 0.9' = '#CC79A7')) +
  scale_y_continuous(labels = percent, limits = c(0, 1),
                     expand = expansion(mult = c(0.05, 0.12))) +
  labs(title = '(d) Support-fit threshold tally', x = NULL, y = 'fraction of seeds') +
  theme_paper(9)
pE2 <- compose_E2_four_panel(list(pE2a, pE2b, pE2c, pE2d),
                             'E2_supervised_fit', 7.2, 5.4, root = getwd())
save_both(pE2, 'E2_supervised_fit', 7.2, 5.4)
