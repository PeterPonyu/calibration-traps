"""Direction 006 — factorized arithmetic dataset for the Fourier-feature
emergence boundary.

The study resolves the 2406.03445 (from-scratch standard *integer* addition →
only low-frequency / no clean Fourier circuits) vs 2301.05217-lineage
(from-scratch *mod-p* addition → clean per-frequency Fourier circuits)
contradiction by factorially decoupling FOUR axes that the two papers confound:

  (a) modular wrap-around vs plain integer addition      -> `modular`
  (b) single-token  vs digit-wise tokenization           -> `tokenization`
  (c) train coverage fraction                            -> `coverage`
  (d) target frequency content (sum vs mod)              -> `target`

Sequence format (Nanda-style, EQ-terminated, predict at final position):

  single-token:  [a, b, EQ]                       (a, b, label are integer tokens)
  digit-wise  :  [a_hi.. a_lo, b_hi.. b_lo, EQ]   (base-`digit_base` digits)

For digit-wise encoding the integer tokens 0..base-1 are the *digits*; operands
are zero-padded to a fixed width so seq_len is constant within a dataset. The
label is ALWAYS a single integer token in the answer vocabulary (the FFT/logit
probes operate on this contiguous integer-token axis) — this keeps the
unembedding rows a clean function of the integer answer, which is exactly the
axis the embedding/unembedding FFT probe scans.

Vocabulary layout (single contiguous integer-token block first, so the probe can
slice rows [0:n_int] without gaps):

  single-token:  ints 0..(V_int-1), then EQ = V_int.        vocab = V_int + 1
  digit-wise  :  digits 0..(base-1)  used in the INPUT,
                 answer ints 0..(V_ans-1) used in the LABEL,
                 EQ token last.  We embed BOTH in one table; the integer-token
                 axis the probe scans is 0..(V_ans-1) (a superset of the digit
                 range when base <= V_ans, which holds for our defaults).

`make_arith_dataset(...)` is deterministic per `seed` (seeded permutation split),
returns (Xtr, Ytr), (Xte, Yte), meta. `meta` exposes vocab_size / seq_len /
n_int (size of the integer-token axis for the probes) / answer_size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class DatasetMeta:
    vocab_size: int
    seq_len: int
    n_int: int          # size of the contiguous integer-token axis the FFT probe scans
    answer_size: int    # number of distinct labels (= n_int; the answer space)
    eq_token: int
    modular: bool
    tokenization: str
    coverage: float
    target: str
    p_or_max: int
    digit_base: int
    digit_width: int
    n_total: int
    n_train: int
    n_test: int
    extra: dict = field(default_factory=dict)


def _digits(values: torch.Tensor, base: int, width: int) -> torch.Tensor:
    """Most-significant-first base-`base` digit expansion of a 1-D LongTensor.

    Returns [N, width] with values[i] = sum_j out[i, j] * base**(width-1-j).
    """
    out = torch.empty(values.shape[0], width, dtype=torch.long)
    v = values.clone()
    for j in range(width - 1, -1, -1):
        out[:, j] = v % base
        v = v // base
    return out


def _answer_size(p_or_max: int, modular: bool, target: str) -> int:
    """Number of distinct labels in the answer space."""
    if target == "mod" or modular:
        return p_or_max                       # labels in [0, p)
    # plain integer sum of two operands in [0, max): max value is 2*(max-1).
    return 2 * (p_or_max - 1) + 1             # labels in [0, 2*max-2]


def make_arith_dataset(
    p_or_max: int,
    modular: bool,
    tokenization: Literal["single", "digit"],
    coverage: float,
    target: Literal["sum", "mod"],
    seed: int,
    digit_base: int = 10,
    device: str = "cpu",
):
    """Build a factorized arithmetic dataset.

    Parameters
    ----------
    p_or_max   : modulus p (modular) OR exclusive operand bound max (non-modular).
                 Operands a, b are drawn from [0, p_or_max).
    modular    : if True, label uses wrap-around (a+b) % p_or_max; else plain a+b.
    tokenization: 'single' (one token per integer) or 'digit' (base-`digit_base`).
    coverage   : fraction of all a*b operand pairs placed in the train split.
    target     : 'sum' (label = a+b, integer) or 'mod' (label = (a+b) % p_or_max).
                 NOTE target='mod' forces a modular label regardless of `modular`;
                 `modular` governs whether the operand *value* semantics wrap. The
                 factorial design treats (modular, target) as two distinct knobs so
                 the "label space wraps" vs "task is defined mod p" confound is
                 broken — see the direction doc.
    seed       : controls the deterministic train/test split permutation.
    digit_base : base for digit-wise tokenization (default 10).

    Returns (Xtr, Ytr), (Xte, Yte), meta:DatasetMeta.
    """
    if tokenization not in ("single", "digit"):
        raise ValueError(f"unknown tokenization {tokenization!r}")
    if target not in ("sum", "mod"):
        raise ValueError(f"unknown target {target!r}")
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must be in (0,1), got {coverage}")

    P = int(p_or_max)
    a = torch.arange(P).repeat_interleave(P)   # [P*P]
    b = torch.arange(P).repeat(P)              # [P*P]

    if target == "mod" or modular:
        y = (a + b) % P
    else:
        y = a + b                              # plain integer sum

    answer_size = _answer_size(P, modular, target)
    # integer-token axis the probe scans = the answer space (contiguous 0..answer_size-1)
    n_int = answer_size

    if tokenization == "single":
        eq_token = n_int                       # ints occupy 0..n_int-1
        vocab_size = n_int + 1
        digit_width = 1
        eq = torch.full_like(a, eq_token)
        X = torch.stack([a, b, eq], dim=1)     # [P*P, 3]
    else:  # digit-wise
        # fixed width covering the largest operand value (P-1) in base digit_base
        width = max(1, len(_int_to_digits(P - 1, digit_base)))
        digit_width = width
        a_d = _digits(a, digit_base, width)    # [P*P, width]
        b_d = _digits(b, digit_base, width)
        # one shared table holds both input digits (0..base-1) and the answer-int
        # axis (0..n_int-1); EQ is the final token. n_int >= base for our defaults.
        eq_token = max(n_int, digit_base)
        vocab_size = eq_token + 1
        eq = torch.full((a.shape[0], 1), eq_token, dtype=torch.long)
        X = torch.cat([a_d, b_d, eq], dim=1)   # [P*P, 2*width + 1]

    seq_len = X.shape[1]

    # deterministic seeded split into train (coverage) / test (1 - coverage)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(round(coverage * n))
    n_train = max(1, min(n - 1, n_train))      # keep both splits non-empty
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    X = X.to(device)
    y = y.to(device)
    Xtr, Ytr = X[tr_idx], y[tr_idx]
    Xte, Yte = X[te_idx], y[te_idx]

    meta = DatasetMeta(
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_int=n_int,
        answer_size=answer_size,
        eq_token=eq_token,
        modular=modular,
        tokenization=tokenization,
        coverage=coverage,
        target=target,
        p_or_max=P,
        digit_base=digit_base,
        digit_width=digit_width,
        n_total=n,
        n_train=int(n_train),
        n_test=int(n - n_train),
    )
    return (Xtr, Ytr), (Xte, Yte), meta


def _int_to_digits(value: int, base: int) -> list[int]:
    """MSB-first digit list of a non-negative int (at least one digit)."""
    if value == 0:
        return [0]
    out = []
    v = value
    while v > 0:
        out.append(v % base)
        v //= base
    return out[::-1]


# ---------------------------------------------------------------------------
# Default cells (referenced by the trainer / runner and the direction doc).
#   - a modular cell at p ≈ 97 (the 2301.05217 clean-Fourier regime)
#   - a non-modular integer cell at max ≈ 200 (the 2406.03445 regime)
# ---------------------------------------------------------------------------
DEFAULT_MODULAR_P = 97
DEFAULT_NONMODULAR_MAX = 200


def default_modular(coverage: float = 0.4, tokenization: str = "single",
                    seed: int = 0, device: str = "cpu"):
    """p≈97 modular-addition cell (clean-Fourier regime)."""
    return make_arith_dataset(
        p_or_max=DEFAULT_MODULAR_P, modular=True, tokenization=tokenization,
        coverage=coverage, target="mod", seed=seed, device=device)


def default_nonmodular(coverage: float = 0.4, tokenization: str = "single",
                       seed: int = 0, device: str = "cpu"):
    """max≈200 plain integer-addition cell (low-frequency regime)."""
    return make_arith_dataset(
        p_or_max=DEFAULT_NONMODULAR_MAX, modular=False, tokenization=tokenization,
        coverage=coverage, target="sum", seed=seed, device=device)
