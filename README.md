# Testbed-calibration traps in small-scale emergence — code & data

Reproducibility archive: **experiment code and per-run result logs only**.
Manuscript and write-up/derivation documents are intentionally **not** included.

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.

## Reproducing
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded
(seed lists appear in result-log filenames). Dependencies: Python 3.11+, PyTorch,
numpy. All inputs are synthetic and fully specified in the code, except large
standard datasets (MNIST / WikiText) which are not bundled.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.
