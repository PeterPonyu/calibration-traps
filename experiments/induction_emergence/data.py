"""Direction 007 — synthetic induction / in-context-learning (ICL) task.

We study how the induction-head circuit (the canonical mechanism behind
in-context copying) *emerges* over training. The data generator must therefore
make the induction structure (i) present, so a head that implements
"attend to the token after the previous occurrence of the current token, then
copy it" gets the answer right, and (ii) cleanly measurable, so we can split
each sequence's next-token predictions into "first occurrence" (genuinely
unpredictable from context) vs "repeat occurrence" (induction-predictable). The
gap between those two accuracies is the ICL score.

Construction (repeated-segment induction, online / fresh every batch)
--------------------------------------------------------------------
A sequence of length L over a vocabulary of size V is built by sampling a random
*segment* of length P and tiling it to fill the sequence (the last tile is
truncated to land exactly on L). Concretely, with segment s[0..P-1],

    x[t] = s[t mod P]          for t = 0 .. L-1.

The segment tokens are drawn DISTINCT (a random length-P subset of the vocab,
requires P <= V). Distinctness matters: it guarantees that within one period a
token value is unique, so the induction rule "find the most recent earlier
position with the same token value, copy the token after it" lands on the
*periodic* predecessor and is therefore exactly correct on repeat positions. If
segment tokens could collide within a period, an accidental in-period match
would mislead the most-recent-match oracle. With the default V=64, P=L//4 this
constraint holds comfortably.

Next-token targets are the causal-LM shift y[t] = x[t+1] (the last position has
no target and is masked out of every metric).

Why this yields a clean induction signal. Consider predicting y[t] = x[t+1].
If position t is NOT the first time the token value x[t] has appeared (i.e. the
same token value occurred at some earlier position t' < t), then because the
sequence is periodic with period P, the token that followed that earlier
occurrence (x[t'+1]) equals the token that follows now (x[t+1]). So an oracle
that finds the most recent earlier position t' with x[t'] == x[t] and copies
x[t'+1] predicts y[t] perfectly. That is *exactly* the induction-head rule
("prev-token match -> copy the next token"). At FIRST occurrences there is no
earlier match, so the next token is unpredictable from context (chance ~ 1/V).

Position roles (for the ICL split)
----------------------------------
For each predicted position t (0 <= t < L-1) we label it:
  - "repeat"  : token value x[t] occurred at some earlier position t' < t
                (induction-predictable);
  - "first"   : x[t] is the first occurrence of that value (unpredictable).
Position L-1 has no target and is excluded.

Determinism. The whole batch is a pure function of an integer `seed` (via a
torch.Generator), so the online stream `batch_seed(base_seed, step)` is exactly
reproducible while still presenting a fresh batch every step.

Self-test (`python data.py`)
----------------------------
Builds one batch and checks the induction *oracle* (copy the token after the
most recent earlier occurrence of the current token):
  * achieves ~100% next-token accuracy on REPEAT positions, and
  * achieves ~chance (~1/V) accuracy on FIRST positions.
This certifies that the repeat positions are genuinely induction-predictable
and the first positions are genuinely not.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InductionSpec:
    """Fully describes the synthetic induction task (deterministic given seed)."""
    vocab_size: int = 64       # V — number of distinct token values
    seq_len: int = 128         # L — sequence length
    period: int = 0            # P — repeated-segment length; 0 => auto (L // 4)

    @property
    def P(self) -> int:
        """Effective segment length (auto = seq_len // 4, clamped to >= 2)."""
        if self.period and self.period > 0:
            return min(self.period, self.seq_len)
        return max(2, self.seq_len // 4)


def batch_seed(base_seed: int, step: int) -> int:
    """Deterministic per-step seed for the online (fresh-batch) stream."""
    # Large odd multiplier -> well-separated, reproducible per-step streams.
    return (base_seed * 1_000_003 + step * 100_003 + 1) & 0x7FFF_FFFF


def sample_batch(spec: InductionSpec, n: int, seed: int, device: str = "cpu"):
    """Sample n induction sequences and their causal-LM targets + position roles.

    Returns (X, Y, repeat_mask, target_mask):
      X            : LongTensor [n, L]      input token ids
      Y            : LongTensor [n, L]      next-token targets (Y[:, :-1] valid;
                                            last column duplicated, but masked out)
      repeat_mask  : BoolTensor [n, L]      True where position t is a REPEAT
                                            occurrence (induction-predictable)
      target_mask  : BoolTensor [n, L]      True where a target exists (all but
                                            the last column)
    Deterministic given (spec, seed). Built in memory; fresh-batch friendly.
    """
    g = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF)
    L, V, P = spec.seq_len, spec.vocab_size, spec.P
    assert P <= V, (
        f"period P={P} must be <= vocab_size V={V} for distinct segment tokens "
        f"(induction-oracle exactness). Raise vocab_size or lower period/seq_len."
    )

    # Random DISTINCT segment (per row), then tile to length L. Distinct tokens
    # within a period make the most-recent-match induction rule exact on repeats.
    rand = torch.rand(n, V, generator=g)                      # [n, V] per-row keys
    seg = rand.argsort(dim=1)[:, :P]                          # [n, P] distinct ids
    idx = torch.arange(L) % P                                  # [L]
    X = seg[:, idx]                                            # [n, L] periodic

    # Causal-LM next-token targets: Y[t] = X[t+1]; last column has no target.
    Y = torch.empty_like(X)
    Y[:, :-1] = X[:, 1:]
    Y[:, -1] = X[:, -1]                                        # dummy (masked out)

    target_mask = torch.ones(n, L, dtype=torch.bool)
    target_mask[:, -1] = False                                # last position: no target

    repeat_mask = _repeat_occurrence_mask(X)                  # [n, L] bool
    repeat_mask = repeat_mask & target_mask                   # only where a target exists

    return (X.to(device), Y.to(device),
            repeat_mask.to(device), target_mask.to(device))


def _repeat_occurrence_mask(X: torch.Tensor) -> torch.Tensor:
    """True at [b, t] iff token value X[b, t] appeared at some earlier t' < t."""
    n, L = X.shape
    # eq[b, t, s] = (X[b, t] == X[b, s]); earlier-match exists for s < t.
    eq = X[:, :, None] == X[:, None, :]                       # [n, L, L]
    strictly_earlier = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    return (eq & strictly_earlier[None]).any(dim=2)          # [n, L]


def prev_occurrence_index(X: torch.Tensor) -> torch.Tensor:
    """For each position t, the most recent earlier t' < t with X[t']==X[t].

    Returns LongTensor [n, L] of indices; entries with no earlier match are -1.
    Used by the induction oracle (self-test) and the attention prefix-match probe.
    """
    n, L = X.shape
    eq = X[:, :, None] == X[:, None, :]                       # [n, L, L]
    strictly_earlier = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    match = eq & strictly_earlier[None]                       # [n, L, L]
    pos = torch.arange(L)[None, None, :].expand(n, L, L)      # candidate t' values
    # most recent earlier match = max valid t'; -1 where none.
    masked_pos = torch.where(match, pos, torch.full_like(pos, -1))
    return masked_pos.max(dim=2).values                      # [n, L]


def induction_oracle_predictions(X: torch.Tensor) -> torch.Tensor:
    """Oracle next-token prediction by the induction rule.

    pred[t] = X[ prev_occurrence_index(t) + 1 ] when an earlier match exists,
    else -1 (declines to predict; counts as wrong at first occurrences).
    Returns LongTensor [n, L].
    """
    n, L = X.shape
    prev = prev_occurrence_index(X)                          # [n, L], -1 if none
    has = prev >= 0
    src = (prev + 1).clamp(max=L - 1)                        # index of token AFTER match
    pred = torch.gather(X, 1, src.clamp(min=0))             # [n, L]
    return torch.where(has, pred, torch.full_like(pred, -1))


# ---------------------------------------------------------------------------
# Self-test: the induction oracle must score ~100% on repeat positions and
# ~chance (1/V) on first positions. Run with `python data.py`.
# ---------------------------------------------------------------------------
def _self_test() -> int:
    torch.manual_seed(0)
    spec = InductionSpec(vocab_size=64, seq_len=128, period=0)
    X, Y, repeat_mask, target_mask = sample_batch(spec, n=256, seed=0)

    pred = induction_oracle_predictions(X)                   # [n, L]
    correct = (pred == Y)                                    # [n, L]

    first_mask = target_mask & (~repeat_mask)                # predictable target, first occ

    rep_n = int(repeat_mask.sum())
    first_n = int(first_mask.sum())
    rep_acc = float((correct & repeat_mask).sum()) / max(1, rep_n)
    first_acc = float((correct & first_mask).sum()) / max(1, first_n)
    chance = 1.0 / spec.vocab_size

    print("SELF-TEST induction oracle (copy token after most-recent earlier match):")
    print(f"  vocab_size V = {spec.vocab_size}, seq_len L = {spec.seq_len}, "
          f"period P = {spec.P}")
    print(f"  repeat positions: n = {rep_n:6d}, oracle acc = {rep_acc:.4f} "
          f"(expect ~1.0000)")
    print(f"  first  positions: n = {first_n:6d}, oracle acc = {first_acc:.4f} "
          f"(expect ~chance {chance:.4f})")

    ok = True
    if rep_acc < 0.999:
        ok = False
        print(f"  FAIL: repeat-position oracle acc {rep_acc:.4f} < 0.999")
    if first_acc > chance + 0.05:
        ok = False
        print(f"  FAIL: first-position oracle acc {first_acc:.4f} "
              f">> chance {chance:.4f} (positions are not unpredictable)")

    # Shape / mask sanity.
    n, L = X.shape
    if Y.shape != (n, L) or repeat_mask.shape != (n, L):
        ok = False
        print("  FAIL: shape mismatch among X / Y / masks")
    if bool(target_mask[:, -1].any()):
        ok = False
        print("  FAIL: last position should have no target")

    print("DATA SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
