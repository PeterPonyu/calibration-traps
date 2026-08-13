# Shared four-panel composition and metadata helpers.

if (!exists("%||%", mode = "function", inherits = TRUE)) {
  `%||%` <- function(x, y) {
    if (is.null(x) || length(x) == 0L) y else x
  }
}

.fp_panel_value <- function(panel, key) {
  value <- attr(panel, key, exact = TRUE)
  if (is.null(value)) character(0) else as.character(value)
}

with_panel_metadata <- function(panel, question, source) {
  question <- trimws(as.character(question))
  source <- trimws(as.character(source))
  if (length(question) != 1L || anyNA(question) || !nzchar(question)) {
    stop("panel question must be one non-empty string", call. = FALSE)
  }
  if (!length(source) || anyNA(source) || any(!nzchar(source))) {
    stop("panel source must contain non-empty strings", call. = FALSE)
  }
  attr(panel, "question") <- question
  attr(panel, "source") <- source
  panel
}

.fp_normalize_panels <- function(panels) {
  if (!is.list(panels) || length(panels) != 4L) {
    stop("compose_four_panel requires exactly four panels", call. = FALSE)
  }
  lapply(seq_along(panels), function(i) {
    question <- .fp_panel_value(panels[[i]], "question")
    source <- .fp_panel_value(panels[[i]], "source")
    if (length(question) != 1L || anyNA(question) || !nzchar(trimws(question))) {
      stop(sprintf("panel %s has no non-empty question metadata", letters[[i]]),
           call. = FALSE)
    }
    if (!length(source) || anyNA(source) || any(!nzchar(trimws(source)))) {
      stop(sprintf("panel %s has no non-empty source metadata", letters[[i]]),
           call. = FALSE)
    }
    list(id = letters[[i]], question = question, source = unname(source))
  })
}

compose_four_panel <- function(panels, name, width, height,
                               layout = "balanced-2x2",
                               exception_reason = NULL,
                               legend_layout = "outer",
                               legend_height = 0.16,
                               legend_direction = "horizontal",
                               # When FALSE, skip patchwork tags (caller folds bold
                               # **(a)** into panel titles — avoids (a) (a) doubles).
                               add_panel_tags = TRUE,
                               # Optional column-width ratios. When top/bottom differ,
                               # rows are composed separately so spines need not align
                               # (e.g. long y-labels on (b) must not pad (d)).
                               widths = NULL,
                               top_widths = NULL,
                               bottom_widths = NULL,
                               # 1-based panel indices released from patchwork axis-space
                               # alignment (typically panel 4 / (d)).
                               free_panels = NULL,
                               free_type = "space",
                               free_side = "l") {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("compose_four_panel requires the patchwork package", call. = FALSE)
  }
  name <- trimws(as.character(name))
  if (length(name) != 1L || anyNA(name) || !nzchar(name)) {
    stop("figure name must be one non-empty string", call. = FALSE)
  }
  if (!is.numeric(width) || length(width) != 1L || !is.finite(width) || width <= 0 ||
      !is.numeric(height) || length(height) != 1L || !is.finite(height) || height <= 0) {
    stop("figure width and height must be positive finite numbers", call. = FALSE)
  }
  allowed_layouts <- c("balanced-2x2", "evidence-ladder")
  if (length(layout) != 1L || !layout %in% allowed_layouts) {
    stop(sprintf("unsupported four-panel layout: %s", layout), call. = FALSE)
  }
  allowed_legend_layouts <- c("outer", "row-gap")
  legend_layout <- trimws(as.character(legend_layout))
  if (length(legend_layout) != 1L || anyNA(legend_layout) || !legend_layout %in% allowed_legend_layouts) {
    stop(sprintf("unsupported four-panel legend layout: %s", legend_layout), call. = FALSE)
  }
  if (!is.numeric(legend_height) || length(legend_height) != 1L ||
      !is.finite(legend_height) || legend_height <= 0) {
    stop("legend_height must be a positive finite number", call. = FALSE)
  }
  allowed_legend_directions <- c("vertical", "horizontal")
  legend_direction <- trimws(as.character(legend_direction))
  if (length(legend_direction) != 1L || anyNA(legend_direction) ||
      !legend_direction %in% allowed_legend_directions) {
    stop(sprintf("unsupported four-panel legend direction: %s", legend_direction),
         call. = FALSE)
  }
  if (!identical(layout, "balanced-2x2") && !identical(legend_layout, "outer")) {
    stop("non-default four-panel layouts only support outer legends", call. = FALSE)
  }
  if (identical(layout, "balanced-2x2") && !is.null(exception_reason)) {
    stop("balanced-2x2 figures cannot have an exception reason", call. = FALSE)
  }
  if (!identical(layout, "balanced-2x2")) {
    exception_reason <- trimws(as.character(exception_reason))
    if (length(exception_reason) != 1L || anyNA(exception_reason) || !nzchar(exception_reason)) {
      stop("non-default layouts require a non-empty exception reason", call. = FALSE)
    }
  }

  metadata_panels <- .fp_normalize_panels(panels)

  # Optional width overrides (balanced-2x2 only). Prefer nested top/bot rows with
  # independent widths so (b)'s long y-labels do not pad (d). Avoid free() on a
  # nested child plot — patchwork 1.3 errors / re-aligns; split widths suffice.
  .fp_as_widths2 <- function(x, label) {
    if (is.null(x)) return(NULL)
    # yaml::read_yaml yields lists for [1, 1.08]; coerce to numeric vector.
    x <- as.numeric(unlist(x, use.names = FALSE))
    if (length(x) != 2L || any(!is.finite(x)) || any(x <= 0)) {
      stop(sprintf("%s must be length-2 positive numerics", label), call. = FALSE)
    }
    x
  }
  tw <- .fp_as_widths2(top_widths %||% widths, "top_widths/widths")
  bw <- .fp_as_widths2(bottom_widths %||% widths, "bottom_widths/widths")
  use_split_widths <- (!is.null(tw) || !is.null(bw)) &&
    identical(layout, "balanced-2x2")
  if (use_split_widths) {
    tw <- tw %||% c(1, 1)
    bw <- bw %||% c(1, 1)
  }

  # free_panels only with shared-column wrap_plots (not nested split widths).
  panels_use <- panels
  if (!is.null(free_panels)) {
    if (use_split_widths) {
      warning("free_panels ignored when top_widths/bottom_widths are set; ",
              "nested row widths already release (b)/(d) spine alignment",
              call. = FALSE)
    } else {
      free_panels <- as.integer(unlist(free_panels, use.names = FALSE))
      if (any(!is.finite(free_panels)) || any(free_panels < 1L) ||
          any(free_panels > 4L) || anyDuplicated(free_panels)) {
        stop("free_panels must be distinct integers in 1..4", call. = FALSE)
      }
      for (i in free_panels) {
        panels_use[[i]] <- patchwork::free(
          panels_use[[i]], type = free_type, side = free_side
        )
      }
    }
  }

  if (identical(layout, "balanced-2x2") && identical(legend_layout, "row-gap")) {
    if (use_split_widths) {
      top <- (panels[[1]] | panels[[2]]) +
        patchwork::plot_layout(widths = tw)
      bot <- (panels[[3]] | panels[[4]]) +
        patchwork::plot_layout(widths = bw)
      composite <- top / patchwork::guide_area() / bot
    } else {
      composite <- (panels_use[[1]] | panels_use[[2]]) /
        patchwork::guide_area() /
        (panels_use[[3]] | panels_use[[4]])
    }
    heights <- grid::unit.c(
      grid::unit((1 - legend_height) / 2, "null"),
      grid::unit(legend_height, "null"),
      grid::unit((1 - legend_height) / 2, "null")
    )
    composite <- composite & ggplot2::theme(
      legend.position = "bottom",
      legend.box = "vertical",
      legend.direction = legend_direction,
      legend.box.just = "center",
      legend.spacing.y = grid::unit(1, "pt"),
      legend.margin = ggplot2::margin(0, 0, 0, 0)
    )
  } else if (identical(layout, "balanced-2x2")) {
    if (use_split_widths) {
      # Independent row widths: do not force (b)/(d) spine alignment.
      top <- (panels[[1]] | panels[[2]]) +
        patchwork::plot_layout(widths = tw)
      bot <- (panels[[3]] | panels[[4]]) +
        patchwork::plot_layout(widths = bw)
      composite <- top / bot
    } else if (!identical(panels_use, panels)) {
      composite <- patchwork::wrap_plots(panels_use, nrow = 2, ncol = 2)
    } else {
      composite <- patchwork::wrap_plots(panels, nrow = 2, ncol = 2)
    }
    heights <- grid::unit(c(1, 1), "null")
  } else {
    if (!is.null(free_panels) || use_split_widths) {
      stop("free_panels/widths overrides require balanced-2x2 layout",
           call. = FALSE)
    }
    composite <- patchwork::wrap_plots(panels, design = "AAB\nCCD")
    heights <- grid::unit(c(1, 1), "null")
  }
  composite <- composite +
    patchwork::plot_layout(guides = "collect", heights = heights)
  if (isTRUE(add_panel_tags)) {
    composite <- composite + patchwork::plot_annotation(tag_levels = "a")
  }
  attr(composite, "figure_panel_metadata") <- list(
    artifact = name,
    layout = layout,
    panel_count = 4L,
    panels = metadata_panels,
    exception_reason = exception_reason,
    width = as.numeric(width),
    height = as.numeric(height)
  )
  composite
}

write_panel_sidecar <- function(name, panels, layout, exception_reason, out_dir) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("write_panel_sidecar requires the jsonlite package", call. = FALSE)
  }
  if (!dir.exists(out_dir)) {
    stop(sprintf("sidecar output directory does not exist: %s", out_dir),
         call. = FALSE)
  }
  if (!is.list(panels) || length(panels) != 4L) {
    stop("panel sidecar requires exactly four panel records", call. = FALSE)
  }
  for (i in seq_along(panels)) {
    panel <- panels[[i]]
    question <- as.character(panel$question)
    source <- as.character(panel$source)
    if (length(question) != 1L || anyNA(question) || !nzchar(trimws(question)) ||
        !length(source) || anyNA(source) || any(!nzchar(trimws(source)))) {
      stop("panel sidecar records require a non-empty question and source",
           call. = FALSE)
    }
  }
  ids <- vapply(panels, function(panel) as.character(panel$id), character(1))
  if (!identical(ids, letters[1:4])) {
    stop("panel sidecar IDs must be a, b, c, d in order", call. = FALSE)
  }
  ext <- if (grepl("^tex", basename(normalizePath(out_dir)))) ".tex" else ".pdf"
  destination <- file.path(out_dir, paste0(name, ext, ".panels.json"))
  payload <- list(
    artifact = name,
    layout = layout,
    panel_count = 4L,
    panels = lapply(panels, function(panel) list(
      id = panel$id,
      question = panel$question,
      source = unname(as.list(panel$source))
    )),
    exception_reason = exception_reason
  )
  temporary <- tempfile(pattern = paste0(".", name, ".panels-"), tmpdir = out_dir)
  jsonlite::write_json(payload, temporary, auto_unbox = TRUE, pretty = TRUE,
                       null = "null")
  if (!file.rename(temporary, destination)) {
    unlink(temporary)
    stop(sprintf("failed to install panel sidecar: %s", destination), call. = FALSE)
  }
  invisible(destination)
}
