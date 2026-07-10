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

## Budget-grid and rescue-inventory additions (v1.5, 2026-07)
- `experiments/icrl_td/run_20260708_positive_regime.py` + `experiments/results/icrl_td_positive_regime/`
  (60 runs): the gamma=0.5 budget grid (4x/8x canonical budget; 0/60 emergences).
- Rescue-control raw data now archived: `experiments/results/e2_td_positive_control_20260618/`,
  `e2_td_positive_control_20260619/`, `e2_td_positive_rescue_20260619/` (26 trained cells).
- `experiments/icrl_td/build_redteam_e2_stats.py` rebuilds
  `experiments/results/figures-redteam/redteam_e2_stats.json` (26-cell pool, upper95=0.109)
  from those verdicts.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.
