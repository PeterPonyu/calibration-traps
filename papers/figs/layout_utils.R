# papers/figs/layout_utils.R
# Configuration loading and diagnostic utilities for layout system

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

load_layout_config <- function(artifact, root = getwd(), prefix = "") {
  # Build config file path
  config_filename <- if (nzchar(prefix)) {
    sprintf("%s_layout_config.yaml", prefix)
  } else {
    "C_layout_config.yaml"
  }
  config_path <- file.path(root, "papers", "figs", config_filename)

  # Hardcoded defaults (used when file missing or yaml unavailable)
  defaults <- list(
    plot_margin = c(8, 10, 8, 10),
    panel_spacing = 16,
    title_wrap_width = 38L
  )

  # Check if yaml package available
  if (!requireNamespace("yaml", quietly = TRUE)) {
    warning("yaml package not installed, using default layout config", call. = FALSE)
    return(defaults)
  }

  # Check if config file exists
  if (!file.exists(config_path)) {
    # Return defaults silently (config file is optional)
    return(defaults)
  }

  # Load YAML
  all_config <- tryCatch(
    yaml::read_yaml(config_path),
    error = function(e) {
      warning(sprintf("Failed to parse %s: %s. Using defaults.",
                      config_filename, e$message),
              call. = FALSE)
      return(NULL)
    }
  )

  if (is.null(all_config)) {
    return(defaults)
  }

  # Look for artifact-specific config, fallback to _default
  config <- if (artifact %in% names(all_config)) {
    all_config[[artifact]]
  } else if ("_default" %in% names(all_config)) {
    all_config[["_default"]]
  } else {
    list()
  }

  # Fill missing fields from defaults
  for (key in names(defaults)) {
    if (is.null(config[[key]])) {
      config[[key]] <- defaults[[key]]
    }
  }

  config
}

diagnose_layout <- function(composite_plot, artifact) {
  cat(sprintf("\n=== Layout Diagnosis: %s ===\n", artifact))

  # Extract metadata if available
  meta <- attr(composite_plot, "figure_panel_metadata")
  if (!is.null(meta)) {
    cat(sprintf("Figure size: %.1f x %.1f inches\n", meta$width, meta$height))
    cat(sprintf("Layout: %s\n", meta$layout))
  }

  # Debug output controlled by environment variable
  if (Sys.getenv("LAYOUT_DEBUG") == "1") {
    cat("\nDEBUG: Set LAYOUT_DEBUG=0 to hide detailed output\n")
  }

  cat("\nRecommendation: Render and visually inspect for overlaps.\n")
  invisible(NULL)
}
