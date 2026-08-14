# Calibration Traps

**Live door:** https://peterponyu.github.io/calibration-traps/

Open the preflight console first. Modules: Nomogram, Testbed, Drift, BIG-Bench, Preflight, Reproduce-as-rebuild.

- GitHub: https://github.com/PeterPonyu/calibration-traps
- Zenodo: https://doi.org/10.5281/zenodo.21020386

## What's here
- `experiments/` — runners and per-run logs
- `portal/` — phosphor preflight console (static export)

## Reproduce
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded.
Dependencies: Python 3.11+, PyTorch, numpy. Inputs are synthetic and specified
in the runners, except large standard datasets (MNIST / WikiText) which are not bundled.

Local console preview:

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
