suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(dplyr); library(tidyr); library(patchwork); library(scales); library(ragg); library(svglite)
})
fig_dir <- "papers/figs"
out_dir <- "papers/figs/evidence_r"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
source(file.path(fig_dir, "fig_pipeline.R"))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(fig_dir, "E2_panel_contract.R"))
`%||%` <- function(a,b) if (!is.null(a)) a else b
pal <- c(adamw="#0072B2", muon="#D55E00", sgdm="#999999")
route_pal <- c(config="#009E73", signal="#0072B2", residual="#7F7F7F", pooled="#0072B2", `AdamW-only`="#D62728")
theme_paper <- function(base=9){
  theme_minimal(base_size=base, base_family="TeX Gyre Termes") +
    theme(
      plot.title=element_text(face="plain", size=rel(1.05), margin=margin(b=3)),
      plot.subtitle=element_blank(),
      axis.title=element_text(size=rel(.9)), axis.text=element_text(size=rel(.82), colour="#222222"),
      legend.title=element_blank(), legend.position="top", legend.text=element_text(size=rel(.82)),
      panel.grid.minor=element_blank(), panel.grid.major=element_line(linewidth=.24, colour="#E2E2E2"),
      plot.margin=margin(5.5,7,5.5,7),
      plot.tag=element_text(face="bold", size=11), plot.tag.position=c(0.02, 0.98)
    )
}
save_both <- function(plot, name, w, h, dpi=300){
  png_path <- file.path(fig_dir, name)
  svg_path <- file.path(out_dir, sub("\\.png$", ".svg", name))
  ragg::agg_png(png_path, width=w, height=h, units="in", res=dpi, background="white")
  print(plot); dev.off()
  svglite::svglite(svg_path, width=w, height=h, bg="white")
  print(plot); dev.off()
  emit_vector(plot, sub("\\.png$", "", name), w, h)  # vector tier (name carries .png here)
  info <- file.info(png_path)
  list(png=png_path, svg=svg_path, width_in=w, height_in=h, dpi=dpi, bytes=unname(info$size))
}

# --- A: P5 bridge from raw JSONL ---
read_jsonl_run <- function(path){
  lines <- readLines(path, warn=FALSE)
  meta <- fromJSON(lines[[1]])$`_meta`
  rows <- lapply(lines[-1], fromJSON, simplifyVector=FALSE)
  bind_rows(lapply(rows, function(x){
    tibble(
      optimizer=meta$optimizer, start=ifelse(isTRUE(meta$warm_start), "warm", "cold"), seed=meta$seed,
      task=x$task,
      steps=ifelse(is.null(x$steps_to_threshold), NA_real_, as.numeric(x$steps_to_threshold)),
      censored=is.null(x$steps_to_threshold),
      final_acc=as.numeric(x$final_acc),
      feat_eff_rank=as.numeric(x$probes$feat_eff_rank %||% NA_real_),
      dead_frac=as.numeric(x$probes$dead_frac %||% NA_real_)
    )
  }))
}
a_files <- list.files("experiments/results/muon_plasticity_p5_bridge", pattern="jsonl$", full.names=TRUE)
a_raw <- bind_rows(lapply(a_files, read_jsonl_run)) %>% mutate(optimizer=factor(optimizer, levels=c("adamw","muon","sgdm")), start=factor(start, levels=c("warm","cold")))
a_line <- a_raw %>% mutate(steps_plot=ifelse(is.na(steps), 120, steps)) %>% group_by(start, optimizer, task) %>% summarise(med=median(steps_plot), lo=quantile(steps_plot,.25), hi=quantile(steps_plot,.75), censor=sum(censored), .groups="drop")
a_summ <- jsonlite::fromJSON("experiments/results/muon_plasticity_p5_bridge/p5_bridge_verdict.json")$summary
a_diag <- bind_rows(lapply(names(a_summ), function(k){
  x <- a_summ[[k]]; parts <- strsplit(k,"_")[[1]]
  tibble(optimizer=parts[1], start=parts[2], fit=x$median_n_tasks_fit, mean_fit=x$median_mean_fit, rank=x$median_final_feat_eff_rank, dead=100*x$median_final_dead_frac)
})) %>% mutate(optimizer=factor(optimizer, levels=c("adamw","muon","sgdm")), start=factor(start, levels=c("warm","cold")))
make_a_panel <- function(start_val, panel_title) {
  d <- filter(a_line, start==start_val)
  ggplot(d, aes(task, med, colour=optimizer, fill=optimizer)) +
    geom_ribbon(aes(ymin=lo, ymax=hi), alpha=.12, colour=NA) +
    geom_line(linewidth=.75) + geom_point(size=1.25, stroke=0) +
    geom_point(data=filter(d, censor>0), aes(y=120, shape="censored"), colour="black", size=2.0, stroke=.8,
               show.legend=c(colour=FALSE, fill=FALSE, shape=TRUE)) +
    scale_colour_manual(values=pal, labels=c(adamw="AdamW", muon="Muon", sgdm="SGDM")) +
    scale_fill_manual(values=pal, labels=c(adamw="AdamW", muon="Muon", sgdm="SGDM")) +
    scale_shape_manual(values=c(censored=4), labels=c(censored="censored: <80% train acc in 120 steps"), name=NULL) +
    scale_x_continuous(breaks=c(0,5,10,15,20,25,29)) + scale_y_continuous(limits=c(0,125), breaks=seq(0,120,30)) +
    labs(x="task index", y="steps to 80% train accuracy", title=panel_title) +
    theme_paper(9) + theme(legend.position="top")
}
pa_warm <- make_a_panel("warm", "(a) Warm-start continual")
# Cold panel carries no censored points, so its guide differs from the warm
# panel's (no shape key) and patchwork cannot merge them — that produced two
# side-by-side legends that overflowed and clipped at both canvas edges.
# Suppress the cold legend and collect only the warm panel's complete legend.
pa_cold <- make_a_panel("cold", "(b) Cold-start control") + theme(legend.position='none')

# (c) warm-minus-cold speed ratio per task: the bridge effect as a paired
#     quantity (median over seeds of steps_cold / steps_warm, uncensored tasks)
a_pair <- a_raw %>% filter(!censored) %>%
  select(optimizer, seed, task, start, steps) %>%
  pivot_wider(names_from = start, values_from = steps) %>%
  filter(!is.na(warm), !is.na(cold)) %>%
  group_by(optimizer, task) %>%
  summarise(speedup = median(cold / warm), .groups = "drop")
pc_bridge <- ggplot(a_pair, aes(task, speedup, colour = optimizer)) +
  geom_hline(yintercept = 1, linetype = "dotted", colour = "#7F7F7F", linewidth = .5) +
  geom_line(linewidth = .75) + geom_point(size = 1.4, stroke = 0) +
  scale_colour_manual(values = pal, labels = c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")) +
  scale_x_continuous(breaks = c(0,5,10,15,20,25,29)) +
  labs(x = "task index", y = "cold / warm steps",
       title = "(c) Bridge effect: warm-start speedup per task") +
  theme_paper(9) + theme(legend.position = "none")

# (d) endpoint plasticity by optimizer x start: final feature rank (points) and
#     dead fraction (labels) from the verdict summary
pd_bridge <- ggplot(a_diag, aes(optimizer, rank, fill = start)) +
  geom_col(position = position_dodge(.72), width = .62, colour = "white", linewidth = .25) +
  geom_text(aes(label = sprintf("%.0f%% dead", dead), group = start),
            position = position_dodge(.72), vjust = -0.35, size = 2.4) +
  scale_fill_manual(values = c(warm = "#E69F00", cold = "#56B4E9")) +
  scale_x_discrete(labels = c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")) +
  labs(x = NULL, y = "final feature effective rank",
       title = "(d) Endpoint plasticity (warm vs cold)") +
  theme_paper(9) + theme(legend.position = "top")

a_fig <- (pa_warm | pa_cold) / (pc_bridge | pd_bridge) +
  plot_annotation(
    theme=theme(plot.tag=element_text(face='bold', size=11), plot.tag.position=c(0.02, 0.98),
                legend.position='top', legend.box='horizontal')
  ) +
  plot_layout(guides='collect')

# --- E2: spec-route from JSON summary ---
e_txt <- readLines("experiments/results/spec_route/posthoc_round1.json", warn=FALSE)
e_txt <- gsub("NaN", "null", e_txt, fixed=TRUE)
e <- jsonlite::fromJSON(paste(e_txt, collapse="\n"), simplifyVector=FALSE)
cell_get <- function(cell, key) e[[cell]][[key]]
delta <- bind_rows(lapply(c("W-mem/tierA","W-200/tierA"), function(cell){
  bind_rows(lapply(c("config","seed","task"), function(key){
    x <- cell_get(cell, paste0("grok/",key)); tibble(cell=cell, predictor=key, delta=x$delta, perm_p=x$perm_p)
  }))
})) %>% mutate(cell=factor(cell, levels=c("W-mem/tierA","W-200/tierA")), predictor=factor(predictor, levels=c("config","seed","task")))
route <- bind_rows(
  tibble(cell="W-mem", cohort="pooled (all optimizers)", auroc=cell_get("W-mem/tierA","route/config")$auroc, p=cell_get("W-mem/tierA","route/config")$perm_p),
  tibble(cell="W-mem", cohort="within AdamW", auroc=cell_get("W-mem/tierA","route/config_adamw")$auroc, p=cell_get("W-mem/tierA","route/config_adamw")$perm_p),
  tibble(cell="W-200", cohort="pooled (all optimizers)", auroc=cell_get("W-200/tierA","route/config")$auroc, p=cell_get("W-200/tierA","route/config")$perm_p),
  tibble(cell="W-200", cohort="within AdamW", auroc=cell_get("W-200/tierA","route/config_adamw")$auroc, p=cell_get("W-200/tierA","route/config_adamw")$perm_p)
) %>% mutate(cell=factor(cell, levels=c("W-mem","W-200")), cohort=factor(cohort, levels=c("pooled (all optimizers)","within AdamW")))
p1_data <- bind_rows(
  tibble(cell="W-mem", model="signal", mae=cell_get("W-mem/tierA","p1_seed_increment")$mae_signal, p=cell_get("W-mem/tierA","p1_seed_increment")$perm_p, n=cell_get("W-mem/tierA","p1_seed_increment")$n),
  tibble(cell="W-mem", model="config baseline", mae=cell_get("W-mem/tierA","p1_seed_increment")$mae_config, p=cell_get("W-mem/tierA","p1_seed_increment")$perm_p, n=cell_get("W-mem/tierA","p1_seed_increment")$n),
  tibble(cell="W-200", model="signal", mae=cell_get("W-200/tierA","p1_seed_increment")$mae_signal, p=cell_get("W-200/tierA","p1_seed_increment")$perm_p, n=cell_get("W-200/tierA","p1_seed_increment")$n),
  tibble(cell="W-200", model="config baseline", mae=cell_get("W-200/tierA","p1_seed_increment")$mae_config, p=cell_get("W-200/tierA","p1_seed_increment")$perm_p, n=cell_get("W-200/tierA","p1_seed_increment")$n)
) %>% mutate(model=factor(model, levels=c("signal","config baseline")))
p4_src <- cell_get("W-mem/tierA","p4_decomposition")
p4 <- bind_rows(lapply(names(p4_src), function(k){
  x <- p4_src[[k]]
  opt <- gsub("[()' ]", "", k)
  tibble(optimizer=sub(",.*", "", opt), config=x$config_share, signal=x$signal_share, residual=x$residual_share)
})) %>% pivot_longer(c(config,signal,residual), names_to="component", values_to="share") %>% mutate(optimizer=factor(optimizer, levels=c("adamw","muon","sgdm")), component=factor(component, levels=c("config","signal","residual")))
p_delta <- ggplot(delta, aes(predictor, delta, fill=cell)) +
  geom_hline(yintercept=0, linewidth=.25) +
  geom_col(position=position_dodge(.7), width=.62, colour="white", linewidth=.25) +
  geom_text(aes(label=sprintf("%+.2f", delta)), position=position_dodge(.7), vjust=ifelse(delta$delta>=0,-.35,1.25), size=2.5) +
  scale_fill_manual(values=c("W-mem/tierA"="#56B4E9","W-200/tierA"="#E69F00"),
                    labels=c("W-mem/tierA"="memory-onset / tier A",
                             "W-200/tierA"="200-step / tier A"),
                    name="window") +
  labs(x=NULL, y="Δ AUROC (signal − config-only)", title="Δ AUROC") +
  coord_cartesian(ylim=c(-.48,.095), clip="off") + theme_paper(9)
p_route <- ggplot(route, aes(x=cell, y=auroc, colour=cohort, group=cohort)) +
  geom_hline(yintercept=.5, linetype="dotted", linewidth=.35) +
  geom_hline(yintercept=.8, linetype="dashed", colour="#228B22", linewidth=.35) +
  geom_point(position=position_dodge(width=.45), size=3.0) +
  geom_text(aes(label=sprintf("%.2f", auroc)), position=position_dodge(width=.45),
            vjust=-.8, size=2.8, colour="#111111") +
  scale_x_discrete(labels=c("W-mem"="mem-onset", "W-200"="200-step")) +
  scale_colour_manual(values=c("pooled (all optimizers)"="#0072B2", `within AdamW`="#D62728"),
                      labels=c("pooled (all optimizers)"="pooled across optimizers",
                               "within AdamW"="within AdamW"), name="cohort") +
  scale_y_continuous(breaks=c(0,.2,.4,.5,.6,.8)) +
  coord_cartesian(ylim=c(0,.95), clip="off") +
  labs(x=NULL, y="route AUROC", title="Route AUROC") +
  theme_paper(9) + theme(
    # The composite owns the collected guide; keep only the semantic cohort
    # guide here and use one short display label per window.
    axis.text.x=element_text(size=7.5, angle=0, hjust=.5, lineheight=.9),
    legend.position="top")
p_mae <- ggplot(p1_data, aes(cell, mae, fill=model)) +
  geom_col(position=position_dodge(.72), width=.62, colour="white", linewidth=.25) +
  geom_text(aes(label=sprintf("%.2f", mae)), position=position_dodge(.72), vjust=-.35, size=2.8) +
  scale_fill_manual(values=c(signal="#0072B2", `config baseline`="#009E73")) +
  labs(x=NULL, y="LOO MAE (log delay residual)", title="LOO MAE") +
  coord_cartesian(ylim=c(0,.60), clip="off") + theme_paper(9)
p_decomp <- ggplot(p4, aes(optimizer, share, fill=component)) +
  geom_col(width=.68, colour="white", linewidth=.2) +
  geom_text(data=p4 %>% group_by(optimizer) %>% summarise(total=sum(share), .groups="drop"),
            aes(x=optimizer, y=total, label=sprintf("%.2f", total)), inherit.aes=FALSE, vjust=-.25, size=2.6) +
  scale_fill_manual(values=c(config="#009E73", signal="#0072B2", residual="#7F7F7F")) +
  scale_y_continuous(expand=expansion(mult=c(0,.12))) +
  labs(x=NULL, y="variance share", title="Variance share") +
  theme_paper(9)
e_fig <- compose_E2_four_panel(list(p_delta, p_route, p_mae, p_decomp),
                               'E2_specroute', 6.5, 5.0, root = getwd())

res <- list(
  A=save_both(a_fig, "A_plasticity_p5_bridge.png", 7.2, 5.4, 300),
  E2=save_both(e_fig, "E2_specroute.png", 6.5, 5.0, 300),
  source="R/ggplot2+ragg+svglite from raw JSONL and posthoc JSON summaries"
)
write_json(res, file.path(out_dir,"final-polish-r-summary.json"), pretty=TRUE, auto_unbox=TRUE)
