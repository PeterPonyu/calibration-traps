# Direction 007 — Muon × induction-head / ICL emergence

Tests whether Muon's orthogonalized (spectrum-flattening) update changes the
**timing, abruptness, and cross-seed variance** of induction-head /
in-context-learning (ICL) emergence in tiny transformers, vs AdamW and SGDM. A
synthetic induction task (repeated-segment sequences with distinct per-period
tokens) supplies fresh online batches; emergence is read off the **ICL-score
curve** (repeat-position accuracy minus first-occurrence accuracy) as a phase
transition, and an **attention prefix-match probe** on the final-layer heads
confirms the induction circuit (attention mass from position *t* onto the token
after the earlier occurrence of `x[t]`).

Reuses the grokking infra by import (the grokking files are NOT modified):
- `model.py` — `SeqTransformer` subclasses grokking `GrokTransformer` (2 layers,
  4 heads, d=128) and overrides `forward` to emit **full-sequence** next-token
  logits (grokking returns last-position only). Adds `forward_with_attn` for the
  attention probe. Muon split re-exported from `grokking/muon.py` (name-based,
  applies unchanged).
- Optimizer hybrid: Muon on 2-D block matrices, AdamW on embeddings / unembed /
  norms. SGDM = same momentum+lr as Muon **without** Newton-Schulz (isolates the
  orthogonalization).

Online fresh-batch causal-LM training — no fixed train set, no grokking
memorization phase; eval uses a fixed held-out stream so the ICL-score curve is
comparable across steps and runs.

## Smoke check (no files written, <60 s)
```
python data.py                          # data oracle self-test (PASS)
python probes.py                        # probe self-test (PASS on oracle circuit)
python train_induction.py --smoke       # labeled smoke lines (incl. ICL probe)
python run_induction.py --smoke         # delegates to the trainer smoke
```

The `--smoke` contract prints exactly:
```
SMOKE DATASET SHAPE: ...
SMOKE PARAM COUNT: <n>
SMOKE FORWARD LOSS: <float>
SMOKE OPTIMIZER STEP: OK
SMOKE ICL PROBE: icl_score=<f> prefix_match=<f>
```
(<=1 step, no jsonl, exit 0; untrained ICL/prefix-match values are near 0.)

## Data oracle self-test (what `python data.py` certifies)
The induction oracle (copy the token after the most recent earlier occurrence of
the current token) achieves **~100%** next-token accuracy on REPEAT positions and
**~chance** on FIRST occurrences — proving repeat positions are genuinely
induction-predictable and first positions are not.

## Dry run (prints planned cells, launches nothing)
```
python run_induction.py --dry-run           # 90 cells
python run_induction.py --dry-run --lr-arm  # 90 + 27 lr-robustness cells
```

## Real grid (run when ready — do NOT launch yet)
```
python run_induction.py             # muon/adamw/sgdm × L∈{64,128,256} × seeds 0-9 = 90 cells
python run_induction.py --lr-arm    # + lr-robustness arm (~27 cells)
```
Results land in `experiments/results/induction_emergence/`. Resume-aware: a cell
whose `.jsonl` already ends with a `_summary` line is skipped.

## Reference
See `directions/007-muon-induction-emergence.md` for the full research write-up.
