# Direction 006 — Fourier-feature emergence boundary

Resolves the apparent contradiction between **2406.03445** (from-scratch
*standard integer* addition → only low-frequency structure, no clean Fourier
circuits) and the **2301.05217 lineage** (from-scratch *mod-p* addition → clean
per-frequency Fourier circuits) by **factorially decoupling** the four axes the
two papers confound:

| axis | knob | values |
|---|---|---|
| (a) modular wrap-around vs plain integer add | `modular` | True / False |
| (b) tokenization | `tokenization` | single-token / digit-wise |
| (c) train coverage fraction | `coverage` | 0.2 / 0.4 / 0.6 / 0.8 |
| (d) target frequency content | `target` | sum (integer) / mod (wrap) |

Optimizer is **FIXED AdamW** — no optimizer sweep (scope discipline vs direction
004, which owns the muon/adamw/sgdm comparison).

**Measurement** (`probes.py`): (i) row-wise **FFT** of the token-embedding and
unembedding matrices over the integer-token axis → per-frequency power spectrum +
**spectral sparsity index** (fraction of power in the top-k frequencies);
(ii) **per-frequency logit attribution** (project logits onto the Fourier basis of
the answer space); (iii) **frequency-band ablation** (FFT → mask → iFFT a band in
the embedding, on a *copy*, and measure the accuracy drop).

Default cells: a modular `p≈97` cell (clean-Fourier regime) and a non-modular
`max≈200` integer-addition cell (low-frequency regime).

## Smoke check (no files written, <60 s)
```
python train_fourier.py --smoke      # SMOKE DATASET/PARAM/LOSS/OPTIMIZER/FFT lines
python probes.py                     # FFT probe self-test (PASS on a pure-cosine embedding)
```

## Dry run (prints the 48 planned cells, launches nothing)
```
python run_fourier.py --dry-run
```

## Real grid (run when ready — do NOT launch yet)
```
python run_fourier.py
# {modular: yes/no} × {tokenization: single/digit} × {coverage: 0.2/0.4/0.6/0.8} × 3 seeds = 48 cells
```
Results land in `experiments/results/fourier_boundary/` (one `<name>.jsonl` per
cell, resume-aware). Per-eval logs: train/test acc/loss + spectral sparsity +
top-frequency list.

## Import discipline
Reuses `experiments/grokking/model.py` (`GrokTransformer`) **by import only** —
the grokking dir is never modified. Per the repo convention (root `README.md`):
this dir is inserted at the **front** of `sys.path` so the local `data.py` /
`probes.py` shadow grokking's same-named `data.py`; the grokking dir is
**appended** to the back (only `model` is pulled from it, which does not collide).

## Reference
See `directions/006-fourier-emergence-boundary.md` for the full research
write-up (purpose, hypotheses, falsification criteria, supplements).
