# Paper E1 adapter for manifest-derived four-panel summaries.

# Layout components live beside this file. Resolve the directory
# deterministically: renderer-defined fig_dir, else the directory of the
# file currently being sourced, else getwd()/papers/figs.
.ofile_dir <- tryCatch({
  f <- sys.frame(1)$ofile
  if (is.null(f)) NULL else dirname(normalizePath(f))
}, error = function(e) NULL)
.layout_candidates <- c(
  if (exists("fig_dir")) fig_dir,
  .ofile_dir,
  file.path(getwd(), "papers", "figs"),
  getwd()
)
.layout_dir <- NULL
for (cand in .layout_candidates) {
  if (!is.null(cand) && file.exists(file.path(cand, "paper_theme_safe.R"))) {
    .layout_dir <- cand
    break
  }
}
if (is.null(.layout_dir)) {
  stop("E1_panel_contract.R cannot locate paper_theme_safe.R beside it",
       call. = FALSE)
}
if (!exists("paper_theme_safe")) {
  source(file.path(.layout_dir, 'paper_theme_safe.R'), local = FALSE)
}
if (!exists("load_layout_config")) {
  source(file.path(.layout_dir, 'layout_utils.R'), local = FALSE)
}

load_E1_panel_summary <- function(artifact, root = getwd()) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Paper E1 panel summaries require the jsonlite package", call. = FALSE)
  }
  path <- file.path(root, "papers", "figs", "summaries", paste0(artifact, ".json"))
  if (!file.exists(path)) {
    stop(sprintf("missing Paper E1 panel summary: %s", path), call. = FALSE)
  }
  summary <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(summary$artifact, artifact)) {
    stop(sprintf("Paper E1 summary artifact mismatch: expected %s", artifact),
         call. = FALSE)
  }
  if (!identical(as.integer(summary$panel_count), 4L) ||
      !is.list(summary$panels) || length(summary$panels) != 4L) {
    stop("Paper E1 summary must contain exactly four panels", call. = FALSE)
  }
  ids <- vapply(summary$panels, function(panel) as.character(panel$id), character(1))
  if (!identical(ids, letters[1:4])) {
    stop("Paper E1 summary panel IDs must be a, b, c, d", call. = FALSE)
  }
  for (i in seq_along(summary$panels)) {
    panel <- summary$panels[[i]]
    question <- as.character(panel$question)
    sources <- as.character(unlist(panel$sources, use.names = FALSE))
    if (length(question) != 1L || anyNA(question) || !nzchar(trimws(question)) ||
        !length(sources) || anyNA(sources) || any(!nzchar(trimws(sources)))) {
      stop(sprintf("Paper E1 summary panel %s lacks question/source metadata", ids[[i]]),
           call. = FALSE)
    }
  }
  summary
}

compose_E1_four_panel <- function(panels, artifact, width, height, root = getwd()) {
  summary <- load_E1_panel_summary(artifact, root)
  if (!is.list(panels) || length(panels) != 4L) {
    stop("compose_E1_four_panel requires exactly four plots", call. = FALSE)
  }

  config <- load_layout_config(artifact, root, prefix = 'E1')
  wrap_width <- config$title_wrap_width

  enriched <- lapply(seq_along(panels), function(i) {
    panel <- panels[[i]]

    # Fold the (a)..(d) tag INTO the title: left-aligned markdown, wrapped to
    # the panel's physical text budget. patchwork's own tag mechanism centres
    # titles over the panel and lets long titles spill into the neighbouring
    # column, colliding with the neighbour's tag — that mechanism is disabled
    # below for exactly this reason.
    title <- panel$labels$title
    if (inherits(panel, "ggplot") && is.character(title) && length(title) == 1L) {
      clean <- sub("^\\([a-d]\\)\\s*", "", title)
      clean <- sub("^\\*\\*\\([a-d]\\)\\*\\*\\s*", "", clean)  # idempotent re-fold
      # ggtext markdown collapses single \n to a space; <br> is the hard break
      wrapped <- paste(strwrap(clean, width = wrap_width), collapse = '<br>')
      panel <- panel + ggplot2::labs(
        title = sprintf("**(%s)** %s", letters[[i]], wrapped))
    }

    panel <- panel + paper_theme_safe(
      plot_margin = config$plot_margin,
      panel_spacing = config$panel_spacing
    )

    metadata <- summary$panels[[i]]
    with_panel_metadata(
      panel,
      as.character(metadata$question),
      as.character(unlist(metadata$sources, use.names = FALSE))
    )
  })

  composite <- compose_four_panel(
    enriched,
    artifact,
    width,
    height,
    layout = as.character(summary$layout),
    exception_reason = summary$exception_reason,
    legend_layout = as.character(config$legend_layout %||% "outer"),
    legend_height = as.numeric(config$legend_height %||% 0.16),
    legend_direction = as.character(config$legend_direction %||% "horizontal"),
    add_panel_tags = FALSE
  )


  # Collected guides follow the COMPOSITE-level legend.position, not the
  # panel's — panel-level legend.position='bottom' is ignored by
  # plot_layout(guides='collect'). The canonical override is `& theme()`,
  # applied here when a figure asks for it via config.
  if (!is.null(config$legend_position)) {
    composite <- composite & ggplot2::theme(legend.position = config$legend_position)
  }
  composite
}
