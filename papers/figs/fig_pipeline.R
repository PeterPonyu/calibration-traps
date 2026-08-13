#!/usr/bin/env Rscript
# fig_pipeline.R — shared vector-figure emitter for the LuaLaTeX paper pipeline.
#
# Strategy (hybrid, tiered by figure type):
#   * plot-tier  (lines/scatter/bars/panels) -> tikzDevice .tex  in figs/tex/
#       => axis text & math are typeset by LaTeX in the paper's newtxmath font,
#          so figure typography == body typography (highest grade).
#   * heatmap-tier (geom_raster / geom_tile)  -> cairo_pdf .pdf  in figs/vec/
#       => crisp vector data layer, no multi-MB tikz path explosion. Text uses
#          TeX Gyre Termes, the same Times-compatible family as the LuaLaTeX body.
#
# Tier is auto-detected from the built ggplot's layers, so the existing make_*.R
# plot code is untouched — they only need to call emit_vector(p, name, w, h).
#
# Outputs are consumed by main.tex via:
#   plot-tier   : \figtikz{<name>}   (== \input{../figs/tex/<name>.tex})
#   heatmap-tier: \includegraphics[...]{<name>.pdf}   (graphicspath ../figs/vec/)
#
# Idempotent: source() once; sets options + dirs on first load.

suppressPackageStartupMessages({
  library(ggplot2)
  library(tikzDevice)
})

# ── global font default for text geoms ────────────────────────────────────────
# theme(base_family=) only styles axis/title/legend text; geom_text()/annotate()
# /geom_label() do NOT inherit it and fall back to the graphics-device default,
# which can differ from the body font. Keep every Cairo-rendered text layer on
# the same Times-compatible family used by the LuaLaTeX body.
.fp_font_family <- "TeX Gyre Termes"
update_geom_defaults("text",  list(family = .fp_font_family))
update_geom_defaults("label", list(family = .fp_font_family))
if (requireNamespace("ggrepel", quietly = TRUE)) {
  try(update_geom_defaults("text_repel",  list(family = .fp_font_family)), silent = TRUE)
  try(update_geom_defaults("label_repel", list(family = .fp_font_family)), silent = TRUE)
}

# ── locate figs/ robustly (run from papers/figs or repo root) ─────────────────
.fp_figdir <- local({
  wd <- getwd()
  if (basename(wd) == "figs") normalizePath(wd)
  else if (dir.exists(file.path(wd, "papers", "figs")))
    normalizePath(file.path(wd, "papers", "figs"))
  else normalizePath(wd)
})
.fp_texdir <- file.path(.fp_figdir, "tex")   # tikz .tex outputs (\input-ed)
.fp_vecdir <- file.path(.fp_figdir, "vec")   # cairo_pdf heatmap outputs
dir.create(.fp_texdir, showWarnings = FALSE, recursive = TRUE)
dir.create(.fp_vecdir, showWarnings = FALSE, recursive = TRUE)
source(file.path(.fp_figdir, "figure_panel_helpers.R"))

# ── PeerJ re-review opt-in: title-stripped variants in sibling dirs ───────────
# PeerJ requires images with no baked-in titles/legends at re-review. Setting
# FIG_STRIP_TITLES (any non-empty value) redirects output to tex-peerj/vec-peerj
# and strips plot titles/subtitles in emit_vector() below. With the env var
# unset, .fp_texdir/.fp_vecdir keep their values from above and this file's
# behavior is byte-identical to before this block existed.
if (nzchar(Sys.getenv("FIG_STRIP_TITLES", unset = ""))) {
  .fp_texdir <- file.path(.fp_figdir, "tex-peerj")
  .fp_vecdir <- file.path(.fp_figdir, "vec-peerj")
  dir.create(.fp_texdir, showWarnings = FALSE, recursive = TRUE)
  dir.create(.fp_vecdir, showWarnings = FALSE, recursive = TRUE)
}

# ── Unicode + special-char map (applied by tikzDevice's sanitize at draw AND ──
#    width-calc time, so glyphs like ∞/λ are measured and typeset correctly).
#    These OVERRIDE tikzDevice's defaults, so we must re-list the standard TeX
#    specials too. Verified: the make_*.R labels contain NO intentional $...$
#    LaTeX math, so escaping every special as literal text is safe; Unicode
#    glyphs are routed to newtxmath symbols.
.fp_san <- c(
  # standard TeX specials -> literal
  "%","$","}","{","^","_","#","&","~",
  # Unicode glyphs -> math
  "λ","η","ε","Δ","∆","×","·","±","∞","→","≥","≤","≈","≫","≪","§",
  "—","–","−","μ","σ","ρ","θ","α","β","γ","π","φ","ω","τ"
)
.fp_rep <- c(
  "\\%","\\$","\\}","\\{","\\^{}","\\_","\\#","\\&","\\~{}",
  "$\\lambda$","$\\eta$","$\\epsilon$","$\\Delta$","$\\Delta$","$\\times$","$\\cdot$","$\\pm$",
  "$\\infty$","$\\rightarrow$","$\\geq$","$\\leq$","$\\approx$","$\\gg$","$\\ll$","\\S{}",
  "---","--","-","$\\mu$","$\\sigma$","$\\rho$","$\\theta$","$\\alpha$","$\\beta$",
  "$\\gamma$","$\\pi$","$\\phi$","$\\omega$","$\\tau$"
)
stopifnot(length(.fp_san) == length(.fp_rep))

# ── tikzDevice: make width-metric fonts == paper body fonts (newtxmath) ───────
# Use a per-process string-metrics dictionary.  tikzDevice locks this file while
# measuring text; sharing it across concurrent renderers can make otherwise-good
# TikZ output silently fall back to PDF.  build_vectors.sh sets FP_METRICS_DICT;
# direct interactive runs get an isolated temp cache instead of a shared one.
.fp_metrics_dict <- local({
  d <- Sys.getenv("FP_METRICS_DICT", unset = "")
  if (nzchar(d)) return(d)
  file.path(tempdir(), sprintf("tikzmetrics_%s", Sys.getpid()))
})
options(
  tikzDefaultEngine = "pdftex",
  tikzMetricsDictionary = .fp_metrics_dict,
  tikzSanitizeCharacters = .fp_san,
  tikzReplacementCharacters = .fp_rep,
  tikzLatexPackages = c(getOption("tikzLatexPackages"),
                        "\\usepackage{newtxtext,newtxmath}\n"),
  tikzMetricPackages = c(getOption("tikzMetricPackages"),
                         "\\usepackage{newtxtext,newtxmath}\n")
)

# ── tier detection: any raster/tile layer => heatmap tier ─────────────────────
.fp_patchwork_nodes <- function(p) {
  # A patchwork node stores its final child on the node itself and earlier
  # children in patches$plots, so scan the node as well as every descendant.
  subs <- if (inherits(p, "patchwork")) {
    tryCatch(p$patches$plots, error = function(e) NULL)
  } else NULL
  descendants <- if (length(subs)) {
    unlist(lapply(subs, .fp_patchwork_nodes), recursive = FALSE)
  } else list()
  c(list(p), descendants)
}

.fp_is_heatmap <- function(p) {
  # Scan every node, including arbitrarily nested patchwork composites.
  for (pl in .fp_patchwork_nodes(p)) {
    geoms <- tryCatch(
      vapply(pl$layers, function(l) class(l$geom)[1], character(1)),
      error = function(e) character(0))
    if (any(geoms %in% c("GeomRaster", "GeomTile"))) return(TRUE)
  }
  FALSE
}

# ── plotmath detection: tikzDevice's plotmath translator SILENTLY DROPS Greek
#    and subscript symbols (e.g. expression(lambda[max]) -> the symbol is absent
#    from the .tex, no error). cairo_pdf renders plotmath correctly via R's native
#    engine (Nimbus Roman ≈ Times), so any figure with an expression()/parse label
#    is routed to the vector-PDF tier rather than producing a broken TikZ figure.
.fp_has_plotmath <- function(p) {
  for (pl in .fp_patchwork_nodes(p)) {
    labs <- tryCatch(pl$labels, error = function(e) NULL)
    if (length(labs)) {
      hit <- vapply(labs, function(l) is.language(l) || is.expression(l), logical(1))
      if (any(hit)) return(TRUE)
    }
    # geom_text/geom_label with parse=TRUE renders aes(label) as plotmath
    parsed <- tryCatch(
      vapply(pl$layers,
             function(l) isTRUE(l$geom_params$parse) || isTRUE(l$aes_params$parse),
             logical(1)),
      error = function(e) logical(0))
    if (any(parsed)) return(TRUE)
  }
  FALSE
}

# ── LaTeX-safe escaping for plain (non-math) strings in tikz output ───────────
# Only used by callers that opt in; raw labels with intentional math pass through.
fp_tex_escape <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([&%$#_{}])", "\\\\\\1", x)
  x <- gsub("~", "\\\\textasciitilde{}", x)
  x <- gsub("\\^", "\\\\textasciicircum{}", x)
  x
}

.fp_fallback_log <- Sys.getenv("FP_FALLBACK_LOG", unset = "")
.fp_record_fallback <- function(name, reason, outdir) {
  msg <- sprintf("%s\t%s\t%s", name, outdir, reason)
  message(sprintf("  [tikz->pdf fallback] %s: %s", name, reason))
  if (nzchar(.fp_fallback_log)) {
    dir.create(dirname(.fp_fallback_log), showWarnings = FALSE, recursive = TRUE)
    cat(msg, "\n", file = .fp_fallback_log, append = TRUE)
  }
}

# ── Cairo vector device ───────────────────────────────────────────────────────
# Cairo uses R's font resolver rather than LaTeX. Set the family explicitly so
# plotmath/text layers do not silently switch to Nimbus or a sans fallback.
.fp_emit_pdf <- function(p, name, w, h, tag = "vec ") {
  out <- file.path(.fp_vecdir, paste0(name, ".pdf"))
  p <- if (inherits(p, "patchwork")) {
    p & ggplot2::theme(text = ggplot2::element_text(family = .fp_font_family))
  } else {
    p + ggplot2::theme(text = ggplot2::element_text(family = .fp_font_family))
  }
  grDevices::cairo_pdf(out, width = w, height = h, family = .fp_font_family)
  print(p); invisible(grDevices::dev.off())
  cat(sprintf("[%s] %s  (%.1fx%.1f in, %.1f KB, family=%s)\n",
              tag, basename(out), w, h, file.size(out) / 1024, .fp_font_family))
  invisible(out)
}

# ── PeerJ opt-in: strip baked-in titles/subtitles, keep panel tags ───────────
# A title/subtitle beginning with a panel-tag pattern "(a) ..." keeps just the
# "(a)" (captions reference these); everything else is NULLed. Patchwork's own
# "(a)"/"(b)" tags come from plot_annotation(tag_levels=), a layer separate
# from labs(title=), so they survive this untouched either way. Only called
# when FIG_STRIP_TITLES is set (see emit_vector below); otherwise dead code.
#
# Broadcasting via `&` (rather than walking p$patches$plots by hand) is what
# correctly reaches every leaf plot in an arbitrarily NESTED patchwork layout
# (e.g. (p1|p2)/(p3|p4)) — a one-level traversal of $patches$plots misses
# panels buried in a nested composite. A custom ggplot_add() method lets each
# leaf plot's title be inspected and stripped individually as `&` visits it.
.fp_keep_tag <- function(x) {
  if (is.null(x)) return(NULL)
  # plotmath titles (built with expression()) can never match the tag regex
  # below (it needs a plain character string) — strip them like any other.
  if (is.language(x) || is.expression(x)) return(NULL)
  # Plain "(a) title" and markdown "**(a)** title" from compose_C_four_panel.
  # Keep markdown bold so PeerJ strip titles stay **(a)** under element_markdown.
  m <- regmatches(x, regexpr("^\\*\\*\\(([A-Za-z0-9]+)\\)\\*\\*\\s*", x))
  if (length(m) && nzchar(m)) {
    tag <- sub("^\\*\\*\\(([A-Za-z0-9]+)\\)\\*\\*.*$", "**(\\1)**", m)
    return(tag)
  }
  m <- regmatches(x, regexpr("^\\(([A-Za-z0-9]+)\\)\\s*", x))
  if (length(m) && nzchar(m)) {
    raw <- sub("\\s*$", "", m)
    # promote plain (a) → bold markdown for strip-only tag titles
    return(sub("^\\(([A-Za-z0-9]+)\\)$", "**(\\1)**", raw))
  }
  NULL
}
.fp_title_stripper <- structure(list(), class = "fp_title_stripper")
ggplot_add.fp_title_stripper <- function(object, plot, object_name) {
  plot + ggplot2::labs(title = .fp_keep_tag(plot$labels$title),
                        subtitle = .fp_keep_tag(plot$labels$subtitle))
}
.fp_strip_titles <- function(p) {
  if (inherits(p, "patchwork")) p & .fp_title_stripper
  else p + ggplot2::labs(title = .fp_keep_tag(p$labels$title),
                          subtitle = .fp_keep_tag(p$labels$subtitle))
}

# Write panel metadata beside the emitted artifact and remove stale metadata.
.fp_sync_panel_sidecar <- function(p, name, out_dir, stale_artifact) {
  stale_sidecar <- paste0(stale_artifact, ".panels.json")
  if (file.exists(stale_sidecar)) unlink(stale_sidecar)
  current_ext <- if (grepl("^tex", basename(normalizePath(out_dir)))) ".tex" else ".pdf"
  current_sidecar <- file.path(out_dir, paste0(name, current_ext, ".panels.json"))
  metadata <- attr(p, "figure_panel_metadata", exact = TRUE)
  if (is.null(metadata)) {
    if (file.exists(current_sidecar)) unlink(current_sidecar)
    return(invisible(NULL))
  }
  if (!identical(metadata$artifact, name)) {
    stop(sprintf("panel metadata artifact %s does not match emitted name %s",
                 metadata$artifact, name), call. = FALSE)
  }
  write_panel_sidecar(name, metadata$panels, metadata$layout,
                      metadata$exception_reason, out_dir)
}

# ── tikz tier: de-markdown panel titles ───────────────────────────────────────
# ggtext/gridtext lays out an element_markdown title by measuring each WORD and
# emitting it as its own absolutely-positioned \node. Word gaps are sized from a
# strwidth(" ") query, and under tikzDevice that query goes to LaTeX, where a
# standalone space measures ~0 — so the words abut and visibly collide
# ("De-confoundedgroupladder"). cairo_pdf is unaffected (it draws text natively),
# which is why heatmap-tier figures never showed this.
#
# For the tikz tier we therefore render the title as ONE plain element_text node
# and let LaTeX do the spacing. The markdown tag "**(a)**" becomes "(a)" and the
# gridtext hard break "<br>" becomes a real newline. We cannot substitute LaTeX
# markup (\textbf) here: tikz() runs with sanitize = TRUE, which would escape the
# backslash and print the macro literally. Full titles stay plain for spacing;
# bare PeerJ-stripped tags ("(a)" only) are re-bolded via element_text(face=
# "bold") in .fp_text_titles_elem so tikzDevice emits \bfseries (a).
.fp_demarkdown_title <- function(x) {
  if (is.null(x) || is.language(x) || is.expression(x)) return(x)
  if (!is.character(x) || length(x) != 1L) return(x)
  x <- gsub("<br\\s*/?>", "\n", x)
  x <- gsub("\\*\\*(\\([a-zA-Z0-9]+\\))\\*\\*", "\\1", x)  # **(a)** -> (a)
  x <- gsub("\\*\\*([^*]+)\\*\\*", "\\1", x)               # any other bold run
  x <- gsub("(?<!\\*)\\*([^*]+)\\*(?!\\*)", "\\1", x, perl = TRUE)  # italics
  x
}

# Swap an element_markdown title for an element_text one, carrying over the
# geometry/colour so the layout the theme negotiated is preserved.
.fp_text_titles_elem <- function(plot) {
  el <- tryCatch(plot$theme$plot.title, error = function(e) NULL)
  if (!inherits(el, "element_markdown")) return(plot)
  # Assign into the theme in place rather than adding a theme() layer:
  # ggplot2 merges same-name elements by class, and element_text cannot be
  # merged over an element_markdown ("Can't merge the `plot.title` theme
  # element"). Direct replacement sidesteps the merge entirely.
  #
  # PeerJ strip leaves titles as bare "(a)" / "(b)" only. tikzDevice cannot
  # carry markdown bold through sanitize=TRUE, but face="bold" emits
  # \bfseries — matching C_case / C_subspace plot.tag tags. Full titles
  # ("(a) Speedup ...") stay plain so multi-word spacing stays correct.
  title <- plot$labels$title
  tag_only <- is.character(title) && length(title) == 1L &&
    !is.na(title) && grepl("^\\([A-Za-z0-9]+\\)$", title)
  face <- if (tag_only) {
    "bold"
  } else if (is.null(el$face)) {
    NULL
  } else {
    el$face
  }
  plot$theme$plot.title <- ggplot2::element_text(
    hjust      = if (is.null(el$hjust)) 0 else el$hjust,
    size       = el$size,
    colour     = el$colour,
    lineheight = el$lineheight,
    margin     = el$margin,
    face       = face)
  plot
}

.fp_title_detexer <- structure(list(), class = "fp_title_detexer")
ggplot_add.fp_title_detexer <- function(object, plot, object_name) {
  plot <- plot + ggplot2::labs(
    title    = .fp_demarkdown_title(plot$labels$title),
    subtitle = .fp_demarkdown_title(plot$labels$subtitle))
  .fp_text_titles_elem(plot)
}

# Broadcast with `&` so every leaf of an arbitrarily nested patchwork is visited
# (walking $patches$plots one level deep misses panels inside sub-composites).
.fp_demarkdown_for_tikz <- function(p) {
  if (inherits(p, "patchwork")) p & .fp_title_detexer
  else ggplot_add.fp_title_detexer(NULL, p, NULL)
}

emit_vector <- function(p, name, w = 6.5, h = 4.0) {
  panel_metadata <- attr(p, "figure_panel_metadata", exact = TRUE)
  strip_titles <- nzchar(Sys.getenv("FIG_STRIP_TITLES", unset = ""))
  # Pin the -peerj variant to whichever tier the CANONICAL figure already
  # uses, when known. Tier can otherwise flip after stripping: e.g. a title
  # containing a glyph tikzDevice can't measure forces the canonical build
  # onto the vector-PDF fallback tier, but removing that title lets the
  # SAME plot compile as TikZ post-strip. main.tex picks \figtikz vs
  # \includegraphics per figure based on the canonical tier, so an unpinned
  # flip would silently strand the -peerj variant somewhere main.tex never
  # looks. Pinning is skipped if the canonical output doesn't exist yet (or
  # is ambiguous) and falls back to ordinary auto-detection.
  pinned_tier <- NULL
  if (strip_titles) {
    canon_tex_exists <- file.exists(file.path(.fp_figdir, "tex", paste0(name, ".tex")))
    canon_pdf_exists <- file.exists(file.path(.fp_figdir, "vec", paste0(name, ".pdf")))
    if (canon_tex_exists && !canon_pdf_exists) pinned_tier <- "tikz"
    else if (canon_pdf_exists && !canon_tex_exists) pinned_tier <- "vec"
  }
  is_heatmap_tier <- if (!is.null(pinned_tier)) identical(pinned_tier, "vec")
                     else .fp_is_heatmap(p) || .fp_has_plotmath(p)
  if (strip_titles) {
    p <- .fp_strip_titles(p)
    if (!is.null(panel_metadata)) attr(p, "figure_panel_metadata") <- panel_metadata
  }
  # drop any stale output from the other tier so classification stays unambiguous
  stale_tex <- file.path(.fp_texdir, paste0(name, ".tex"))
  stale_pdf <- file.path(.fp_vecdir, paste0(name, ".pdf"))
  if (is_heatmap_tier) {
    if (file.exists(stale_tex)) unlink(stale_tex)
    .fp_emit_pdf(p, name, w, h)
    .fp_sync_panel_sidecar(p, name, .fp_vecdir, stale_tex)
    return(invisible(NULL))
  }
  # plot tier: TikZ (LaTeX-typeset fonts). Some complex plotmath expressions
  # defeat tikzDevice's translator — auto-fall back to a vector PDF so the
  # pipeline is resilient and never hard-fails on a single figure. This still
  # runs even when pinned_tier == "tikz" (stripping only ever removes text,
  # so it should keep compiling, but the safety net costs nothing to keep).
  out <- file.path(.fp_texdir, paste0(name, ".tex"))
  # ggtext titles must become plain text BEFORE the tikz device sees them
  # (see .fp_demarkdown_title for why). Metadata is re-attached because the
  # patchwork `&` broadcast returns a rebuilt object without our attribute.
  p_tikz <- .fp_demarkdown_for_tikz(p)
  if (!is.null(panel_metadata)) {
    attr(p_tikz, "figure_panel_metadata") <- panel_metadata
  }
  ok <- tryCatch({
    tikzDevice::tikz(out, width = w, height = h, standAlone = FALSE,
                     sanitize = TRUE, verbose = FALSE)
    print(p_tikz); invisible(grDevices::dev.off())
    TRUE
  }, error = function(e) {
    if (!is.null(grDevices::dev.list())) invisible(grDevices::dev.off())
    if (file.exists(out)) unlink(out)
    .fp_record_fallback(name, conditionMessage(e), .fp_vecdir)
    FALSE
  })
  if (ok) {
    if (file.exists(stale_pdf)) unlink(stale_pdf)
    .fp_sync_panel_sidecar(p, name, .fp_texdir, stale_pdf)
    cat(sprintf("[tikz] %s  (%.1fx%.1f in, %.1f KB)\n",
                basename(out), w, h, file.size(out) / 1024))
  } else {
    .fp_emit_pdf(p, name, w, h, tag = "vec*")  # vec* = tikz fallback
    .fp_sync_panel_sidecar(p, name, .fp_vecdir, stale_tex)
  }
  invisible(NULL)
}
