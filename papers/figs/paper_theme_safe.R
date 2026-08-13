# papers/figs/paper_theme_safe.R
# Text-width-aware theme for four-panel figures.
#
# The dominant overlap failure is NOT at figure edges but between adjacent
# grid columns: ggplot centres panel titles over the panel, and any title
# wider than the panel spills into the neighbouring column, colliding with
# the neighbour's title and its (a)/(b) tag. plot.margin cannot fix that.
# This theme therefore left-aligns titles (rendered through ggtext so the
# panel tag can be folded into the title itself) and keeps margins moderate.

paper_theme_safe <- function(base_size = 9,
                             plot_margin = c(8, 10, 8, 10),
                             panel_spacing = 16,
                             title_size = 9,
                             title_lineheight = 1.12) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("paper_theme_safe requires ggplot2", call. = FALSE)
  }
  if (!requireNamespace("ggtext", quietly = TRUE)) {
    stop("paper_theme_safe requires ggtext for folded panel tags", call. = FALSE)
  }

  # CRITICAL: this is an INCREMENTAL theme only. A complete theme
  # (theme_minimal/theme_bw/...) resets every theme setting to its own
  # defaults and would wipe the renderer's per-panel customisations
  # (legend.position='none', rotated axis ticks, strip sizes) wholesale.
  # Renderers apply their own complete theme before compose; we only nudge
  # the four things compose itself owns.
  ggplot2::theme(
    panel.spacing = grid::unit(panel_spacing, "pt"),

    # Left-align markdown titles; zero left title margin so panel tags sit
    # slightly lefter. Bold comes from **(a)** markdown folds, not whole title.
    plot.title = ggtext::element_markdown(hjust = 0, size = title_size,
                                          lineheight = title_lineheight,
                                          colour = '#111827',
                                          margin = ggplot2::margin(t = 0, r = 0, b = 3, l = 0)),
    plot.subtitle = ggplot2::element_blank(),
    plot.margin = ggplot2::margin(t = plot_margin[1], r = plot_margin[2],
                                  b = plot_margin[3], l = plot_margin[4]),

    axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 5)),
    axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 6))
  )
}
