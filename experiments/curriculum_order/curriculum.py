"""Direction 011 — the curriculum machine (the NEW load-bearing module).

Direction 004 found the nested TARGET ladder is load-bearing for weak optimizers
(SGDM: pure degree-3 fit 0.00, but on the staircase ladder fit 0.73-0.88). 011
asks the dual question: can a DATA-PRESENTATION curriculum substitute for that
target structure? I.e. instead of giving SGDM an easier *target* (the ladder),
give it an easier *schedule* — train first on degree-1 task data, then degree-2,
..., finally the pure/hard target — and order examples within each stage from
easy to hard. Eval is ALWAYS on the FINAL/hard target.

This module supplies the two orthogonal curriculum axes:

  1. STAGE schedule (the "easy->hard across stages" axis).
     A ``CurriculumSchedule`` is an ordered list of ``Stage`` objects, each a
     ``(dataset_spec, n_steps)`` pair. The trainer walks the stages in order,
     switching the dataset at each boundary; the final stage's ``dataset_spec``
     is the hard/FINAL target the run is evaluated against throughout.

  2. WITHIN-STAGE ordering (the "{iid, easy_to_hard, hard_first, structured}"
     axis). Given a batch's per-example ``difficulty_scores`` (from
     data_router.py), an ``OrderingPolicy`` returns a deterministic permutation
     of example indices:
        iid          : seeded shuffle (difficulty ignored) — the control arm.
        easy_to_hard : ascending difficulty (a within-set easy->hard curriculum).
        hard_first   : descending difficulty (the anti-curriculum probe).
        structured   : a deterministic interleave that walks difficulty bands in
                       a fixed striped order (low band, next band, ...), so each
                       presented chunk spans the difficulty range in a stable,
                       structure-preserving pattern rather than a monotone sort.
     Ties are broken by the seeded shuffle so every policy is a *total* order and
     fully deterministic given the seed.

Difficulty scoring lives in data_router.py (per family). This module is
family-agnostic: it consumes the score vector and produces orderings, and it
builds the stage ladders that the three families share.

Everything is deterministic given ``seed``. No torch model, no files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# The four within-stage ordering arms (the 011 ordering factor).
ORDERINGS = ("iid", "easy_to_hard", "hard_first", "structured")


# ---------------------------------------------------------------------------
# Stage schedule
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    """One curriculum stage: a dataset spec + how many steps to train on it.

    ``dataset_spec`` is a plain dict of keyword overrides forwarded to a
    data_router family loader (e.g. {"profile": "pure", "pure_degree": 1} for the
    Walsh family, or {"period": 8} for the copy family). ``n_steps`` is the number
    of optimizer steps the trainer spends in this stage. ``name`` is a short label
    used in logging / boundary assertions.
    """
    dataset_spec: dict[str, Any]
    n_steps: int
    name: str = ""

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError(f"stage {self.name!r}: n_steps must be > 0")


@dataclass
class CurriculumSchedule:
    """An ordered list of stages + the within-stage ordering policy name.

    The LAST stage's ``dataset_spec`` is the FINAL/hard target the whole run is
    evaluated against (eval never changes across stages). ``ordering`` is one of
    ``ORDERINGS`` and governs example presentation WITHIN every stage.
    """
    stages: list[Stage]
    ordering: str = "iid"

    def __post_init__(self):
        if not self.stages:
            raise ValueError("schedule needs at least one stage")
        if self.ordering not in ORDERINGS:
            raise ValueError(f"ordering {self.ordering!r} not in {ORDERINGS}")

    @property
    def total_steps(self) -> int:
        return sum(s.n_steps for s in self.stages)

    @property
    def final_spec(self) -> dict[str, Any]:
        """The hard/FINAL target spec (last stage) — what eval always uses."""
        return self.stages[-1].dataset_spec

    def stage_boundaries(self) -> list[int]:
        """Cumulative step index at which each stage ENDS (1-based step counts).

        boundaries[i] = sum of n_steps over stages[0..i]. So stage i owns the
        half-open step interval [boundaries[i-1], boundaries[i]).
        """
        out, acc = [], 0
        for s in self.stages:
            acc += s.n_steps
            out.append(acc)
        return out

    def stage_at_step(self, step: int) -> int:
        """Index of the stage that owns ``step`` (0-based; clamps past the end)."""
        for i, b in enumerate(self.stage_boundaries()):
            if step < b:
                return i
        return len(self.stages) - 1


# ---------------------------------------------------------------------------
# Within-stage ordering policies
# ---------------------------------------------------------------------------
def _seeded_perm(n: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF)
    return torch.randperm(n, generator=g).to(device)


def order_indices(difficulty: torch.Tensor, ordering: str, seed: int) -> torch.Tensor:
    """Return a deterministic permutation of [0..n) under the named policy.

    difficulty : [n] float (higher = harder), the data_router score vector.
    ordering   : one of ORDERINGS.
    seed       : determinism + tie-break seed.

    The returned LongTensor is a total order (every policy resolves ties via the
    same seeded shuffle), so re-running with the same seed reproduces it exactly.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering {ordering!r} not in {ORDERINGS}")
    n = int(difficulty.shape[0])
    device = difficulty.device
    shuffle = _seeded_perm(n, seed, device)        # tie-break + iid base

    if ordering == "iid":
        return shuffle

    # Stable sort on difficulty, with the seeded shuffle as the tie-break order:
    # apply the shuffle first, then a STABLE sort, so equal-difficulty items keep
    # their shuffled (seed-determined) relative order.
    shuffled_diff = difficulty[shuffle]
    if ordering == "easy_to_hard":
        sub = torch.argsort(shuffled_diff, stable=True)              # ascending
        return shuffle[sub]
    if ordering == "hard_first":
        sub = torch.argsort(shuffled_diff, descending=True, stable=True)
        return shuffle[sub]
    # structured: sort into ascending difficulty bands, then stripe across bands
    # so each contiguous chunk spans the difficulty range in a fixed pattern.
    asc = shuffle[torch.argsort(shuffled_diff, stable=True)]         # easy->hard
    return _stripe(asc, difficulty[asc])


def _stripe(order: torch.Tensor, sorted_diff: torch.Tensor) -> torch.Tensor:
    """Deterministic band-interleave of an already easy->hard ordering.

    Split the sorted indices into ``n_bands`` contiguous difficulty bands, then
    read one index from each band in turn (round-robin) until exhausted. This
    preserves coverage of the whole difficulty range in every presented chunk
    while remaining a fixed, deterministic function of the sorted order — the
    "structured" arm, distinct from both monotone sorts and the iid shuffle.
    """
    n = int(order.shape[0])
    if n <= 2:
        return order
    n_unique = int(torch.unique(sorted_diff).numel())
    n_bands = max(2, min(n_unique, int(round(n ** 0.5))))
    # CONTIGUOUS difficulty bands: split the easy->hard order into n_bands blocks,
    # each a distinct difficulty range (block 0 = easiest range, last = hardest).
    bands = list(torch.chunk(order, n_bands))
    # round-robin pull one element per band, in band order, until all consumed —
    # so each presented chunk of size n_bands spans easy..hard difficulty once.
    out: list[torch.Tensor] = []
    pos = [0] * n_bands
    remaining = n
    while remaining > 0:
        for bi in range(n_bands):
            if pos[bi] < bands[bi].shape[0]:
                out.append(bands[bi][pos[bi]].reshape(1))
                pos[bi] += 1
                remaining -= 1
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Stage-ladder builders (the "easy->hard target proxy across stages" axis)
# ---------------------------------------------------------------------------
def walsh_degree_ladder(final_degree: int = 3, steps_per_stage: int = 800,
                        L: int = 16, D: int = 4,
                        final_target: str = "staircase",
                        n_stages: int = 3) -> list[Stage]:
    """Walsh curriculum stages. The EVAL target is the run's last stage.

    final_target (2026-06-13 REDESIGN, after the v1 staircase-final pilot still
    failed at fit 0.13 — the pure-deg ramp burned the budget / scrambled weights
    on the unlearnable pure-deg-3 stage before the short staircase stage):

      - "staircase" (default): follow the MODADD pattern — present the SAME
        learnable degree-STAIRCASE target (sum chi_k, the 004/017-proven target,
        017 fit 0.91 trained DIRECTLY) across `n_stages` identical-target stages.
        The curriculum lever is the WITHIN-stage example ordering (iid baseline =
        017's direct-staircase setup), NOT a target ramp. This is the only setup
        that both LEARNS and keeps the eval target fixed so ordering can act.
      - "ramp": the pure deg-1..deg-`final_degree` ramp THEN a staircase terminal
        (the v1 design; kept — it embeds an unlearnable pure-deg-3 stage, useful
        as a "curriculum that fights the target" negative arm).
      - "pure": ORIGINAL terminal = pure degree-`final_degree` (single high-degree
        Walsh character). DEPRECATED/unlearnable (fit ~0.00; 011's own note "SGDM
        pure deg-3 fit 0.00 vs staircase 0.73-0.88"). Negative record only.
    """
    Dsc = max(D, final_degree)
    if final_target == "staircase":
        return [Stage(dataset_spec=dict(profile="staircase", L=L, D=Dsc),
                      n_steps=steps_per_stage, name=f"staircase_stage{i+1}")
                for i in range(n_stages)]
    if final_target in ("ramp", "pure"):
        stages = [Stage(dataset_spec=dict(profile="pure", pure_degree=k, L=L, D=D),
                        n_steps=steps_per_stage, name=f"pure_deg{k}")
                  for k in range(1, final_degree + 1)]
        if final_target == "ramp":
            stages.append(Stage(
                dataset_spec=dict(profile="staircase", L=L, D=Dsc),
                n_steps=steps_per_stage, name="staircase_final"))
        return stages
    raise ValueError(f"final_target must be 'staircase'|'ramp'|'pure', got {final_target!r}")


def modadd_distance_ladder(p: int = 23, steps_per_stage: int = 800,
                           n_stages: int = 3) -> list[Stage]:
    """Easy->hard mod-add stages by operand-distance band (proxy difficulty).

    Each stage uses the same modular task; the curriculum lever is the within-
    stage ordering (operand distance |a-b|). All stages carry the SAME final
    target (the full table), so the FINAL eval target is unchanged — only the
    presentation order is staged. n_stages identical-target stages give the
    ordering policy room to act across a longer horizon.
    """
    return [
        Stage(dataset_spec=dict(p=p, op="add"), n_steps=steps_per_stage,
              name=f"modadd_stage{i+1}")
        for i in range(n_stages)
    ]


def copy_period_ladder(final_period: int = 8, steps_per_stage: int = 800,
                       vocab_size: int = 32, seq_len: int = 32,
                       n_stages: int = 3) -> list[Stage]:
    """Easy->hard copy stages: short repeat distance -> long repeat distance.

    Period (segment length) = the induction look-back distance. Short period =
    nearby copy (easy); the final stage uses ``final_period`` (the hard, long-
    range copy that the run is evaluated on). Earlier stages shrink the period so
    the induction circuit forms on an easier copy first.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if n_stages == 1:
        periods = [final_period]
    else:
        lo = max(2, final_period // n_stages)
        step = (final_period - lo) / (n_stages - 1)
        periods = [int(round(lo + step * i)) for i in range(n_stages)]
        periods[-1] = final_period
    return [
        Stage(dataset_spec=dict(period=P, vocab_size=vocab_size, seq_len=seq_len),
              n_steps=steps_per_stage, name=f"copy_period{P}")
        for P in periods
    ]


LADDERS = {
    "walsh": walsh_degree_ladder,
    "modadd": modadd_distance_ladder,
    "copy": copy_period_ladder,
}


def build_schedule(family: str, ordering: str = "iid",
                   steps_per_stage: int = 800, **ladder_kw) -> CurriculumSchedule:
    """Construct a family's easy->hard schedule under a given ordering arm."""
    if family not in LADDERS:
        raise ValueError(f"unknown family {family!r}; have {sorted(LADDERS)}")
    stages = LADDERS[family](steps_per_stage=steps_per_stage, **ladder_kw)
    return CurriculumSchedule(stages=stages, ordering=ordering)


# ---------------------------------------------------------------------------
# Self-test: stage boundaries materialize as declared; every ordering policy is
# deterministic, a true permutation, and actually sorted by the difficulty key
# (for the monotone arms). Run with `python curriculum.py`.
# ---------------------------------------------------------------------------
def _self_test() -> int:
    ok = True

    # ---- 1. stage boundaries (test "pure" mode = 3 ramp stages) ----
    sched = build_schedule("walsh", ordering="easy_to_hard",
                            steps_per_stage=100, final_degree=3,
                            final_target="pure")
    boundaries = sched.stage_boundaries()
    expected = [100, 200, 300]
    print(f"SELF-TEST stages (pure): n={len(sched.stages)} boundaries={boundaries} "
          f"total={sched.total_steps}")
    if boundaries != expected:
        ok = False
        print(f"  FAIL: boundaries {boundaries} != {expected}")
    if sched.total_steps != 300:
        ok = False
        print(f"  FAIL: total_steps {sched.total_steps} != 300")
    # stage_at_step lands in the right interval
    checks = {0: 0, 99: 0, 100: 1, 199: 1, 200: 2, 299: 2, 5000: 2}
    for step, want in checks.items():
        got = sched.stage_at_step(step)
        if got != want:
            ok = False
            print(f"  FAIL: stage_at_step({step}) = {got} != {want}")
    if sched.final_spec.get("pure_degree") != 3:
        ok = False
        print(f"  FAIL: pure-mode final_spec {sched.final_spec} not pure deg-3")
    # default REDESIGN mode: SAME staircase target across n_stages (modadd pattern)
    sched_sc = build_schedule("walsh", ordering="easy_to_hard",
                              steps_per_stage=100, final_degree=3, n_stages=3)
    all_sc = all(s.dataset_spec.get("profile") == "staircase" for s in sched_sc.stages)
    if len(sched_sc.stages) != 3 or not all_sc:
        ok = False
        print(f"  FAIL: staircase-mode stages not 3x identical staircase "
              f"(n={len(sched_sc.stages)}, all_staircase={all_sc})")
    else:
        print(f"SELF-TEST stages (staircase default): n={len(sched_sc.stages)} "
              f"all-staircase={all_sc} final={sched_sc.final_spec}")

    # ---- 2. orderings: deterministic, permutation, actually sorted ----
    g = torch.Generator().manual_seed(7)
    difficulty = torch.rand(257, generator=g)  # non-trivial, has unique values
    arange = torch.arange(257)

    for ordering in ORDERINGS:
        idx_a = order_indices(difficulty, ordering, seed=3)
        idx_b = order_indices(difficulty, ordering, seed=3)
        # determinism
        if not torch.equal(idx_a, idx_b):
            ok = False
            print(f"  FAIL: {ordering} not deterministic for fixed seed")
        # true permutation
        if not torch.equal(torch.sort(idx_a).values, arange):
            ok = False
            print(f"  FAIL: {ordering} is not a permutation of [0..n)")
        # different seed -> different order for iid/structured tie-break
        idx_c = order_indices(difficulty, ordering, seed=99)
        sorted_diff = difficulty[idx_a]
        if ordering == "easy_to_hard":
            mono = bool((sorted_diff[1:] >= sorted_diff[:-1]).all())
            print(f"  {ordering:13s}: ascending-sorted={mono}")
            if not mono:
                ok = False
                print(f"  FAIL: {ordering} not ascending by difficulty")
        elif ordering == "hard_first":
            mono = bool((sorted_diff[1:] <= sorted_diff[:-1]).all())
            print(f"  {ordering:13s}: descending-sorted={mono}")
            if not mono:
                ok = False
                print(f"  FAIL: {ordering} not descending by difficulty")
        elif ordering == "iid":
            # iid must depend on seed (a shuffle), NOT equal the identity / sort
            seed_sensitive = not torch.equal(idx_a, idx_c)
            print(f"  {ordering:13s}: seed-sensitive shuffle={seed_sensitive}")
            if not seed_sensitive:
                ok = False
                print("  FAIL: iid ordering is not seed-sensitive")
        else:  # structured
            # not equal to either monotone sort, but full coverage (permutation
            # already checked); confirm it is neither pure-ascending nor identity
            asc = order_indices(difficulty, "easy_to_hard", seed=3)
            differs = not torch.equal(idx_a, asc)
            print(f"  {ordering:13s}: differs-from-monotone={differs}")
            if not differs:
                ok = False
                print("  FAIL: structured ordering collapsed to a monotone sort")

    # ---- 3. all three family ladders build ----
    for fam in LADDERS:
        s = build_schedule(fam, ordering="structured", steps_per_stage=50)
        if s.total_steps != 50 * len(s.stages):
            ok = False
            print(f"  FAIL: {fam} ladder total_steps mismatch")
        print(f"  ladder {fam:7s}: stages={[st.name for st in s.stages]}")

    print("CURRICULUM SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
