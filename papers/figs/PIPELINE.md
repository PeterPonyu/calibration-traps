# Figure pipeline — R → TikZ/vector-PDF → LuaLaTeX

Highest-grade, governable figure typography for the paper bundle. Figure text is
typeset by LaTeX in the paper's own font, so axis labels and inline math match the
body exactly (no raster, no font mismatch).

## Architecture (hybrid, tiered by figure type)

| Tier | Detected by | R device | Output | LaTeX include |
|------|-------------|----------|--------|---------------|
| **plot** (lines/scatter/bars/panels) | default | `tikzDevice` | `figs/tex/<name>.tex` | `\figtikz{<name>}` |
| **heatmap** (dense raster/matrix) | any `geom_raster`/`geom_tile` layer | `cairo_pdf` | `figs/vec/<name>.pdf` | `\includegraphics{<name>.pdf}` |

Why hybrid: TikZ gives LaTeX-typeset fonts (the real upgrade) but renders every
heatmap cell as a separate PGF path → multi-MB `.tex`, slow/fragile. Heatmaps have
almost no text to font-match, so they go to compact vector PDF instead.

**Plotmath routing (important).** tikzDevice's `plotmath` translator in this toolchain
**silently drops** Greek/subscript symbols — `expression(lambda[max])` produces a
`.tex` with the symbol simply *absent* (no error). So `fig_pipeline.R` proactively
detects any figure whose labels use `expression()`/`parse=TRUE` and routes it to the
vector-PDF tier, where R's native plotmath renders correctly in Nimbus Roman (a URW
Times clone ≈ body newtxmath). Net effect: figures with plain/literal-Unicode labels
get full LaTeX-typeset TikZ; figures with plotmath get correct vector PDF. A separate
`tryCatch` also catches *hard* tikz failures and falls back the same way
(`[tikz->pdf fallback]`).

## Files

- `fig_pipeline.R` — shared emitter. `source()` it after a script defines `figdir`,
  then call `emit_vector(p, name, w, h)` from each save helper. It auto-detects the
  tier, maps Unicode glyphs (λ ∞ → ≥ …) and TeX specials to newtxmath, caches string
  metrics (`.tikzmetrics*`), and cleans the other tier's stale output.
- `figpreamble.tex` — `\input` it in each `main.tex` after `newtxtext,newtxmath`.
  Loads `tikz`, sets `\graphicspath` (adds `figs/vec/`), defines
  `\figtikz[<width>]{<name>}` (scales the TikZ to the column width the PNG used).
- `reconcile_includes.py` — idempotently rewrites each `main.tex` figure include to its
  CURRENT tier (`\figtikz{}` for tex, `\includegraphics{*.pdf}` for vec, `*.png` for
  PNG-only), preserving column width. Run after any tier change.
- `build_vectors.sh` (in `papers/`) — regenerate all figures, reconcile includes, compile.

## Usage

```bash
cd papers
./build_vectors.sh            # all figures + LuaLaTeX compile, all papers
./build_vectors.sh figs       # figures only
./build_vectors.sh A          # one paper
```

Per-paper concurrent renders must use a distinct metrics dictionary to avoid
corruption: `FP_METRICS_DICT=papers/figs/.tikzmetrics_<P> Rscript …` (build_vectors.sh
does this automatically).

## Engine

LuaLaTeX (not pdflatex): native UTF-8, larger memory for TikZ, `fontspec`-ready.
`latexmk -lualatex` in each paper dir; figures cache via `tikzExternalize`-style
metric dictionaries so rebuilds are fast.

## Adding a new figure

Nothing special — author the ggplot as usual and call the script's existing
`save_*` helper. `emit_vector` runs inside it and routes automatically. PNG/SVG are
still emitted alongside for previews; the paper consumes the vector tier.

## Not vectorized (intentional)

Matplotlib schematics (e.g. `*_landscape`, conceptual `*_bridge` diagrams) have no R
source and remain PNG; they are diagrams, not data plots, so font-matching is moot.
