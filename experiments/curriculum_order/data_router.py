"""Direction 011 — uniform data-family router (importlib adapters, READ-ONLY reuse).

Direction 011 asks whether a DATA-PRESENTATION curriculum (easy->hard staging +
within-stage ordering) can substitute for the nested TARGET ladder that 004 found
to be load-bearing for weak optimizers. To compare a single curriculum machine
across three task families, we need ONE uniform interface over three EXISTING data
generators that live in three sibling experiment dirs — without modifying any of
them. The catch: all three dirs ship a file literally named ``data.py``, so a plain
``from data import ...`` would collide. We therefore load each external module by
its file path via ``importlib.util.spec_from_file_location`` (the exact pattern
``induction_emergence/model.py:45-51`` uses to pull ``grokking/model.py``), giving
each a unique module name so the three ``data.py`` files coexist.

Python 3.13 note
----------------
A module loaded via ``spec_from_file_location`` that defines an ``@dataclass`` must
be registered in ``sys.modules`` under its spec name BEFORE ``exec_module`` runs;
the stdlib dataclass machinery looks the owning module up by name to resolve
annotations (``KW_ONLY`` etc.). All three reused ``data.py`` files use
``@dataclass`` specs, so ``_load_module`` registers first, then execs.

Uniform interface
-----------------
``FAMILIES`` maps a family key -> a loader callable. Each loader returns a
``FamilyBatch``:
    X                : LongTensor [N, T]      token ids (model input)
    Y                : Tensor     [N] or [N, T]  targets (family-specific dtype/shape)
    meta             : dict       family descriptors (vocab_size, seq_len, the
                       loaded spec object, optional masks, the FINAL-target sets
                       for the Walsh family, etc.)
    difficulty_scores: FloatTensor [N]        per-example difficulty (higher = harder),
                       the orderable key consumed by curriculum.py within-stage
                       ordering policies. Defined per family:
                         * walsh   : multi-degree -> max target-component degree present;
                                     pure task -> -|target margin| (small |g| = harder).
                         * modadd  : |a - b| (operand distance) as an arithmetic-difficulty
                                     proxy (far operands = less "near-identity" structure).
                         * copy    : period P (segment repetition distance) -> shorter
                                     repeat distance is easier, longer is harder.

The three families:
    walsh   -> degree_staircase/data.py  (Walsh staircase / pure / mixed; PRIMARY)
    modadd  -> grokking/data.py          (modular-addition Fourier control)
    copy    -> induction_emergence/data.py (copy / induction transfer)

Nothing here trains or writes files. Deterministic given (spec, seed).
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# --- locate the three sibling experiment dirs (paths only; no sys.path edits
#     for these, we load by file path to dodge the data.py name collision) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)
_DEGREE_DIR = os.path.join(_EXPERIMENTS_DIR, "degree_staircase")
_GROKKING_DIR = os.path.join(_EXPERIMENTS_DIR, "grokking")
_INDUCTION_DIR = os.path.join(_EXPERIMENTS_DIR, "induction_emergence")


def _load_module(unique_name: str, path: str):
    """Load an external module by file path under a unique name.

    Registers in sys.modules BEFORE exec (py3.13 @dataclass requirement) and
    caches so repeated family loads reuse the same module object.
    """
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = _ilu.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {unique_name} from {path}")
    module = _ilu.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


# Module-name prefixes are namespaced ("curr011_*") so they never clash with the
# local `data`/`model` names or with the reused dirs' own internal imports.
def _ds_data():
    return _load_module("curr011_ds_data",
                        os.path.join(_DEGREE_DIR, "data.py"))


def _gk_data():
    return _load_module("curr011_gk_data",
                        os.path.join(_GROKKING_DIR, "data.py"))


def _ind_data():
    return _load_module("curr011_ind_data",
                        os.path.join(_INDUCTION_DIR, "data.py"))


@dataclass
class FamilyBatch:
    """Uniform (X, Y, meta, difficulty_scores) bundle across all task families."""
    X: torch.Tensor
    Y: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)
    difficulty_scores: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def scores(self) -> torch.Tensor:
        """Non-optional difficulty vector (every family loader populates it)."""
        if self.difficulty_scores is None:
            raise ValueError("FamilyBatch has no difficulty_scores")
        return self.difficulty_scores


# ---------------------------------------------------------------------------
# Walsh family (degree_staircase/data.py) — PRIMARY
# ---------------------------------------------------------------------------
def load_walsh(n: int, seed: int, *, L: int = 16, D: int = 4,
               profile: str = "staircase", pure_degree: int = 3,
               device: str = "cpu") -> FamilyBatch:
    """One Walsh-family batch with a per-example difficulty score.

    profile in {"staircase","mixed","pure"} (forwarded to StaircaseSpec).
    Difficulty:
      * multi-degree targets (staircase/mixed): per-example MAX degree whose
        monomial chi_{S_k}(x) == +1 is "active and positively contributing";
        we use the degree of the highest-degree present component as the example's
        difficulty (a +1 high-degree monomial is the hard part of the target).
        Concretely score = sum_k k * 1[chi_{S_k}(x) > 0] / (#stages) — a smooth,
        deterministic proxy whose ordering puts low-degree-dominated examples first.
      * pure task: harder = SMALLER target margin |g(x)| (here g in {-1,+1}, so
        we fall back to a deterministic sign-stable proxy: examples whose target
        is -1 are ranked before +1 so "hard_first" vs "easy_to_hard" still split
        the set; the |margin| handle is exposed for richer multi-weight targets).
    """
    m = _ds_data()
    spec = m.StaircaseSpec(L=L, D=D, profile=profile,
                           pure_degree=pure_degree, seed=seed)
    X, y, signs = m.sample_batch(spec, n, seed=seed, device=device)
    sets = m.monomial_sets(spec)

    # per-example component activations chi_{S_k}(x) in {-1,+1}
    degs = sorted(sets.keys())
    n_stages = max(1, len(degs))
    score = torch.zeros(X.shape[0], device=device)
    for k, S in sets.items():
        chi = torch.ones(X.shape[0], device=device)
        for i in S:
            chi = chi * signs[:, i]
        # a +1 degree-k component is the "present hard part" -> weight by degree
        score = score + (k * (chi > 0).float())
    if profile == "pure":
        # single-degree: rank by target sign as a deterministic margin proxy
        # (y in {-1,+1}); harder := y < 0 sorted first under ascending key.
        score = (y > 0).float()
    else:
        score = score / float(n_stages)

    meta = {
        "family": "walsh",
        "vocab_size": spec.vocab_size,
        "seq_len": spec.seq_len,
        "spec": spec,
        "sets": sets,
        "signs": signs,
        "profile": profile,
        "target_kind": "regression",  # scalar g(x) via logit[1]-logit[0]
    }
    return FamilyBatch(X=X, Y=y, meta=meta, difficulty_scores=score)


# ---------------------------------------------------------------------------
# Modular-addition family (grokking/data.py) — Fourier control
# ---------------------------------------------------------------------------
def load_modadd(n: int, seed: int, *, p: int = 23, op: str = "add",
                device: str = "cpu") -> FamilyBatch:
    """A modular-arithmetic batch (subset of the full p*p table) + difficulty.

    The full table is p*p pairs [a, b, EQ] -> (a op b) mod p. We take a seeded
    permutation subset of size min(n, p*p) so curriculum stages can present
    different difficulty bands. Difficulty := |a - b| (operand distance): pairs
    with a == b (and small |a-b|) sit near the "identity/diagonal" structure the
    network grabs first; far-apart operands are the harder generalization band.
    """
    m = _gk_data()
    X_full, Y_full = m.make_modular_dataset(p=p, op=op, device=device)
    total = X_full.shape[0]
    take = min(n, total)
    g = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF)
    idx = torch.randperm(total, generator=g)[:take]
    X = X_full[idx]
    Y = Y_full[idx]
    a = X[:, 0].float()
    b = X[:, 1].float()
    score = (a - b).abs()  # operand distance; 0 on the diagonal (easiest)

    meta = {
        "family": "modadd",
        "vocab_size": p + 1,
        "seq_len": 3,
        "p": p,
        "op": op,
        "target_kind": "classification",  # predict (a op b) mod p
    }
    return FamilyBatch(X=X, Y=Y, meta=meta, difficulty_scores=score)


# ---------------------------------------------------------------------------
# Copy / induction family (induction_emergence/data.py) — transfer
# ---------------------------------------------------------------------------
def load_copy(n: int, seed: int, *, vocab_size: int = 32, seq_len: int = 32,
              period: int = 0, device: str = "cpu") -> FamilyBatch:
    """A copy/induction batch + difficulty.

    Returns full-sequence causal-LM targets. Difficulty := the segment period P
    (repetition distance): a SHORTER repeat distance means the induction copy is
    an easier, closer look-back; a longer period spreads the predictor further
    back in context. Per batch P is fixed by the spec, so the difficulty score is
    a constant vector here (the orderable handle lives at the STAGE level, where
    curriculum.py can stage increasing `period`); we still expose it per example
    so the uniform interface holds and `structured` ordering is well-defined.
    """
    m = _ind_data()
    spec = m.InductionSpec(vocab_size=vocab_size, seq_len=seq_len, period=period)
    X, Y, repeat_mask, target_mask = m.sample_batch(spec, n, seed=seed,
                                                    device=device)
    # difficulty proxy: per-example fraction of FIRST (non-repeat, hard) target
    # positions — a higher first-occurrence share = less in-context support = harder.
    valid = target_mask.float()
    first = (target_mask & (~repeat_mask)).float()
    denom = valid.sum(dim=1).clamp(min=1.0)
    score = first.sum(dim=1) / denom

    meta = {
        "family": "copy",
        "vocab_size": spec.vocab_size,
        "seq_len": spec.seq_len,
        "spec": spec,
        "period": spec.P,
        "repeat_mask": repeat_mask,
        "target_mask": target_mask,
        "target_kind": "seq_lm",  # next-token over all positions
    }
    return FamilyBatch(X=X, Y=Y, meta=meta, difficulty_scores=score)


# Family registry: key -> loader callable with a uniform (n, seed, **kw) signature.
FAMILIES: dict[str, Callable[..., FamilyBatch]] = {
    "walsh": load_walsh,
    "modadd": load_modadd,
    "copy": load_copy,
}


def load_family(family: str, n: int, seed: int, **kw) -> FamilyBatch:
    """Dispatch to a family loader by key (raises on unknown family)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; have {sorted(FAMILIES)}")
    return FAMILIES[family](n, seed, **kw)


# ---------------------------------------------------------------------------
# Self-test: every family loads via importlib and yields the uniform bundle.
# Run with `python data_router.py`.
# ---------------------------------------------------------------------------
def _self_test() -> int:
    ok = True
    print("DATA-ROUTER SELF-TEST: loading all three families via importlib")
    cases: list[tuple[str, int, int, dict[str, Any]]] = [
        ("walsh", 64, 0, dict(L=16, D=4, profile="staircase")),
        ("modadd", 64, 0, dict(p=23)),
        ("copy", 8, 0, dict(vocab_size=32, seq_len=32)),
    ]
    for fam, n, seed, kw in cases:
        b = load_family(fam, n, seed, **kw)
        d = b.scores()
        has_scores = d.shape[0] == b.X.shape[0]
        print(f"  {fam:7s}: X={tuple(b.X.shape)} Y={tuple(b.Y.shape)} "
              f"vocab={b.meta['vocab_size']} seq_len={b.meta['seq_len']} "
              f"difficulty[{tuple(d.shape)}] ok={has_scores}")
        if b.X.ndim != 2:
            ok = False
            print(f"  FAIL: {fam} X is not 2-D")
        if not has_scores:
            ok = False
            print(f"  FAIL: {fam} difficulty scores missing / wrong length")
        if not torch.is_floating_point(d):
            ok = False
            print(f"  FAIL: {fam} difficulty scores must be float")
    print("DATA-ROUTER SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
