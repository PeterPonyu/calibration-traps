# Testbed-calibration traps in small-scale emergence — code, data, pointer tex

Public warehouse for **calibration-traps**: experiment code, per-run logs, pointer
manuscript `papers/E2/main.tex`, figure generators, and a preflight console under
`portal/`. Compiled journal PDFs are not hosted here.

- GitHub: https://github.com/PeterPonyu/calibration-traps
- Zenodo concept: https://doi.org/10.5281/zenodo.21020386
- Figure contract: `papers/FIGURE-INDEX.json`
- Rebuild figures: `papers/figs/PIPELINE.md`

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.
- `papers/E2/main.tex` — full canonical pointer manuscript (`\input{../figs/figpreamble.tex}`).
- `papers/figs/` — generators + JSON summaries (compiled `tex/` and `vec/` are gitignored).
- `portal/` — Next.js phosphor preflight console (`output: 'export'`, `basePath: /calibration-traps`). HOLD door only; findings stay in the warehouse.

## Reproducing
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded
(seed lists appear in result-log filenames). Dependencies: Python 3.11+, PyTorch,
numpy. All inputs are synthetic and fully specified in the code, except large
standard datasets (MNIST / WikiText) which are not bundled.

Local portal preview (Next.js static export, no LaTeX):

```
python -m pip install -r requirements-ci.txt
bash portal/build.sh
mkdir -p /tmp/ct-pages/calibration-traps
cp -a _site/. /tmp/ct-pages/calibration-traps/
python -m http.server -d /tmp/ct-pages 8000
# open http://127.0.0.1:8000/calibration-traps/
```

## License
Code: MIT (`LICENSE`). Result logs and figures: CC BY 4.0. See `CITATION.cff`.
`portal/` and `_site/` are omitted from Zenodo `git archive` packs.
