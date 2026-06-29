# curriculum_order — Direction 011: can a data curriculum substitute for the target ladder?

**Direction doc:** `directions/011-ordering-emergence-delay.md`
**Builds on:** `experiments/findings-004-degree-staircase.md`

## Purpose

Direction 004 found the nested **TARGET** ladder is load-bearing for weak
optimizers: SGD-momentum cannot learn a pure degree-3 Walsh target (fit ≈ 0.00,
5/5 seeds) but on the nested staircase ladder (one monomial per degree 1–4) it
learns all four degrees (fit 0.73–0.88). The low-degree components bootstrap the
high-degree one.

**011 asks the dual question:** can a **DATA-PRESENTATION** curriculum substitute
for that target structure? Instead of giving the weak optimizer an easier
*target* (the ladder), give it an easier *schedule* — train first on degree-1
task data, then degree-2, …, finally the pure/hard degree-3 target — and order
examples *within* each stage. Evaluation is **ALWAYS on the final/hard target**.
The headline contrast: does any ordering arm let SGDM reach the hard target the
single-target run could not?

Two orthogonal curriculum axes:

1. **Stage schedule** (easy→hard across stages): a `CurriculumSchedule` is an
   ordered list of `(dataset_spec, n_steps)` stages; the trainer switches the
   dataset at each boundary; the last stage is the hard FINAL target.
2. **Within-stage ordering** ∈ `{iid, easy_to_hard, hard_first, structured}`:
   given each example's difficulty score, present the batch in that order.
   `iid` is the control; `easy_to_hard` is the within-set curriculum;
   `hard_first` the anti-curriculum; `structured` a fixed band-interleave that
   covers the difficulty range in every presented chunk.

### Difficulty scoring (per family, in `data_router.py`)

- **walsh** (Walsh pure-degree, PRIMARY): multi-degree targets → degree-weighted
  count of positively-active monomials; pure task → target-sign margin proxy.
- **modadd** (mod-add Fourier control): operand distance `|a − b|` (diagonal /
  near-identity pairs easiest).
- **copy** (lookup/copy transfer): segment period `P` / first-occurrence share
  (shorter repeat distance & more in-context support = easier).

## Files

| file | role |
|---|---|
| `curriculum.py` | **the new load-bearing module**: `CurriculumSchedule`, `Stage`, the four ordering policies (`order_indices`), and the per-family stage-ladder builders. |
| `data_router.py` | importlib adapters loading the three families from `degree_staircase/data.py`, `grokking/data.py`, `induction_emergence/data.py` behind a uniform `(X, Y, meta, difficulty_scores)` interface. |
| `train_curriculum.py` | training loop honoring a `CurriculumSchedule` (stage switching + ordered presentation), optimizer ∈ `{adamw, sgdm}`, per-eval jsonl on the FINAL target. |
| `run_curriculum.py` | the 120-cell grid (NOT executed), resume-aware, `--dry-run`. |

### Reuse / import discipline

All reused modules are **UNMODIFIED**. Because three sibling dirs each ship a
file named `data.py` (and two ship `model.py`), every external module is loaded
**by file path** via `importlib.util.spec_from_file_location` under a unique
`curr011_*` name (the pattern `induction_emergence/model.py:45-51` uses).
Modules are registered in `sys.modules` **before** `exec_module` — required so
the reused `@dataclass` specs resolve their annotations under Python 3.13.

- `walsh` / `modadd` → `GrokTransformer` (predict at the EQ position).
- `copy` → `SeqTransformer` (full-sequence next-token).
- Optimizer hybrid + Walsh probe reused from `grokking/muon.py` and
  `degree_staircase/probes.py`. Model param count: **397,440** (walsh; matches
  the 004 scaffold).

`muon` is deliberately **not** an arm — 011 is about the data curriculum, not
the optimizer geometry (scope discipline). `sgdm` is the weak-optimizer probe.

## Smoke

```bash
/home/zeyufu/miniconda3/envs/dl/bin/python train_curriculum.py --smoke
# or via the runner:
/home/zeyufu/miniconda3/envs/dl/bin/python run_curriculum.py --smoke
```

Prints the labeled smoke lines, runs ≤1 step, writes **no files**, exits 0
(< 60 s). Self-tests:

```bash
python curriculum.py     # CURRICULUM SELF-TEST: PASS (stage boundaries + orderings)
python data_router.py    # DATA-ROUTER SELF-TEST: PASS (all 3 families load)
```

## Grid (NOT executed)

```bash
python run_curriculum.py --dry-run        # prints 120 planned cells, runs nothing
```

120 cells = `ordering {iid, easy_to_hard, hard_first, structured}` ×
`optimizer {adamw, sgdm}` × `family {walsh, modadd, copy}` × `seed 0..4`.

Output namespace: `experiments/results/curriculum_order/<family>_<ordering>_<opt>_s<seed>.jsonl`.
Resume-aware (skips any cell whose jsonl already ends with a `_summary` line);
`--num-shards` / `--shard-id` split the deterministic cell list across machines.

Per-eval jsonl records: stage index, train/eval loss + acc on the FINAL target,
per-degree Walsh correlations (walsh family), and the emergence step.
