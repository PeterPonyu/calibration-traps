# Testbed-calibration traps in small-scale emergence science: two pre-registered negatives

Open **code and result logs** for this study. This is the reproducibility archive
behind the manuscript's reported results; figure-rendering and manuscript sources
are intentionally not included.

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.
- `experiments/findings-*.md` — method + headline results for each sub-experiment.
- `experiments/theory-notes/` — worked derivations.

## Reproducing
The committed per-run logs are the recorded outputs. To re-run from scratch
(GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded; seed
lists appear in the findings files and result-log filenames. Dependencies:
Python 3.11+, PyTorch, numpy. All inputs are synthetic and fully specified in the
code, except large standard datasets (MNIST / WikiText) which are not bundled.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0.

## Citing
See `CITATION.cff`. Archival DOI minted on deposit (Zenodo).
