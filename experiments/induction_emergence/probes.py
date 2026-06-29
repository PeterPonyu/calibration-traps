"""Direction 007 — induction / ICL probes.

Three read-only (@no_grad) probes plus an emergence-detection utility:

(i)  icl_score(model, batch)
        repeat-position next-token accuracy MINUS first-occurrence accuracy.
        The headline ICL emergence signal: ~0 before the induction circuit
        forms (the model cannot copy), rising toward ~1 once it does. The
        first-occurrence accuracy is the "unpredictable" baseline (≈ chance),
        so the gap isolates the induction-attributable performance.

(ii) prefix_match_score(model, batch)
        attention-pattern probe on the layer-2 (final block) heads: for each
        repeat position t, how much attention mass the head puts on
        (earlier occurrence of x[t]) + 1 — i.e. the token the induction head
        must read to copy. This is the mechanistic SIGNATURE of an induction
        head (prev-token match -> attend to the following token). Reported per
        head and as the max over heads.

(iii) emergence detection: given an ICL-score curve over training steps, find
        the emergence step (first crossing of a threshold) and the abruptness
        (max discrete derivative d(score)/d(log step), plus a transition-window
        width). Distinguishes a sharp phase transition from a gradual ramp —
        the core cross-optimizer comparison (does Muon change timing / sharpness
        / cross-seed variance of emergence?).

All probes consume a `batch` = (X, Y, repeat_mask, target_mask) as produced by
data.sample_batch. Run `python probes.py` for a self-test against the induction
oracle (a hand-built perfect circuit) confirming the probes read ~1.0.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def _split_masks(repeat_mask, target_mask):
    """(repeat positions, first-occurrence positions) as boolean masks."""
    first_mask = target_mask & (~repeat_mask)
    return repeat_mask, first_mask


@torch.no_grad()
def position_accuracies(model, batch):
    """Per-role next-token accuracy + loss on one batch.

    Returns dict with repeat_acc, first_acc, repeat_loss, first_loss, loss
    (loss = mean cross-entropy over all valid target positions).
    """
    X, Y, repeat_mask, target_mask = batch
    logits = model(X)                                   # [B, T, V]
    V = logits.shape[-1]
    pred = logits.argmax(dim=-1)                         # [B, T]
    correct = (pred == Y)

    rep_mask, first_mask = _split_masks(repeat_mask, target_mask)
    rep_n = int(rep_mask.sum())
    first_n = int(first_mask.sum())
    rep_acc = float((correct & rep_mask).sum()) / max(1, rep_n)
    first_acc = float((correct & first_mask).sum()) / max(1, first_n)

    # token-level cross-entropy, split by role
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), Y.reshape(-1), reduction="none"
    ).reshape(Y.shape)                                  # [B, T]
    tgt_n = int(target_mask.sum())
    loss = float((ce * target_mask).sum()) / max(1, tgt_n)
    rep_loss = float((ce * rep_mask).sum()) / max(1, rep_n)
    first_loss = float((ce * first_mask).sum()) / max(1, first_n)

    return {
        "loss": loss,
        "repeat_acc": rep_acc, "first_acc": first_acc,
        "repeat_loss": rep_loss, "first_loss": first_loss,
        "repeat_n": rep_n, "first_n": first_n,
    }


@torch.no_grad()
def icl_score(model, batch) -> float:
    """ICL score = repeat-position acc - first-occurrence acc (on this batch)."""
    pa = position_accuracies(model, batch)
    return pa["repeat_acc"] - pa["first_acc"]


@torch.no_grad()
def prefix_match_score(model, batch, layer: int = -1):
    """Attention prefix-match probe on a transformer layer (default: last/L2).

    For each REPEAT position t, the induction head should place attention mass on
    the position (prev_occurrence_index(t) + 1): the token immediately AFTER the
    earlier occurrence of x[t] — the source the head copies. We average that
    attention mass over repeat positions, per head.

    Requires `model.forward_with_attn(X) -> (logits, attn_list)` where attn_list[l]
    is [B, n_heads, T, T] (provided by SeqTransformer). Returns dict:
        per_head : list[float]  mean prefix-match mass for each head (layer L)
        max_head : float        the strongest head's score (the induction head)
        mean_head: float        mean over heads
    Values ~1/T for an untrained model (diffuse attention); ~1 for a clean
    induction head.
    """
    from data import prev_occurrence_index

    X, _, repeat_mask, _ = batch
    logits, attn_list = model.forward_with_attn(X)
    attn = attn_list[layer]                             # [B, n_heads, T, T]
    B, H, T, _ = attn.shape

    prev = prev_occurrence_index(X.cpu()).to(X.device)  # [B, T], -1 if none
    src = (prev + 1).clamp(0, T - 1)                    # source position to attend to
    valid = repeat_mask & (prev >= 0)                   # [B, T] repeat positions w/ match

    # gather attention mass attn[b, h, t, src[b,t]] for each head
    src_idx = src[:, None, :, None].expand(B, H, T, 1)  # [B, H, T, 1]
    mass = torch.gather(attn, dim=3, index=src_idx).squeeze(-1)  # [B, H, T]

    valid_h = valid[:, None, :].expand(B, H, T)         # broadcast mask to heads
    denom = valid_h.sum(dim=(0, 2)).clamp(min=1)        # [H]
    per_head = (mass * valid_h).sum(dim=(0, 2)) / denom  # [H]
    per_head_list = [float(x) for x in per_head]
    return {
        "per_head": per_head_list,
        "max_head": float(per_head.max()),
        "mean_head": float(per_head.mean()),
    }


# ---------------------------------------------------------------------------
# Emergence detection over an ICL-score curve.
# ---------------------------------------------------------------------------
def detect_emergence(steps, scores, threshold: float = 0.5):
    """Locate the emergence of in-context learning from an ICL-score curve.

    Args:
        steps      : list/seq of training steps (monotically increasing, >=0).
        scores     : list/seq of ICL scores at those steps (same length).
        threshold  : ICL-score level whose first up-crossing marks "emergence".

    Returns dict:
        emergence_step : first step at which score >= threshold (None if never).
        max_slope      : max discrete d(score)/d(log10 step) over the curve
                         (abruptness; large => sharp phase transition).
        max_slope_step : the step at which that max slope occurs.
        transition_width: number of LOG10-step units spanning the rise from
                         10% to 90% of the curve's final score (None if not
                         both crossed); smaller => more abrupt.
        final_score    : last score in the curve.
    A sharp transition has large max_slope and small transition_width; a gradual
    ramp has small max_slope and large transition_width.
    """
    n = len(steps)
    assert n == len(scores), "steps and scores must be the same length"
    if n == 0:
        return {"emergence_step": None, "max_slope": 0.0, "max_slope_step": None,
                "transition_width": None, "final_score": None}

    steps = [float(s) for s in steps]
    scores = [float(s) for s in scores]

    # emergence step: first up-crossing of threshold.
    emergence_step = None
    for s, v in zip(steps, scores):
        if v >= threshold:
            emergence_step = s
            break

    # abruptness: max discrete derivative wrt log10(step) (step>0 to define log).
    max_slope = 0.0
    max_slope_step = None
    for i in range(1, n):
        s0, s1 = steps[i - 1], steps[i]
        if s0 <= 0 or s1 <= 0:
            continue
        dlog = math.log10(s1) - math.log10(s0)
        if dlog <= 0:
            continue
        slope = (scores[i] - scores[i - 1]) / dlog
        if slope > max_slope:
            max_slope = slope
            max_slope_step = s1

    # transition width: log10-step span from first 10% crossing to first 90%
    # crossing of the FINAL score (only meaningful if final score is positive).
    final_score = scores[-1]
    transition_width = None
    if final_score > 1e-6:
        lo_level, hi_level = 0.1 * final_score, 0.9 * final_score
        lo_step = hi_step = None
        for s, v in zip(steps, scores):
            if lo_step is None and v >= lo_level and s > 0:
                lo_step = s
            if hi_step is None and v >= hi_level and s > 0:
                hi_step = s
                break
        if lo_step is not None and hi_step is not None and hi_step >= lo_step:
            transition_width = math.log10(hi_step) - math.log10(max(lo_step, 1.0))

    return {
        "emergence_step": emergence_step,
        "max_slope": max_slope,
        "max_slope_step": max_slope_step,
        "transition_width": transition_width,
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# Self-test: a hand-built perfect induction "model" must drive icl_score -> 1,
# the prefix-match probe -> 1 on its induction head, and emergence detection
# must fire on a synthetic sharp curve. Run with `python probes.py`.
# ---------------------------------------------------------------------------
class _OracleModel:
    """A perfect induction circuit faked at the logit/attention level.

    forward            : one-hot logits at the oracle next-token prediction.
    forward_with_attn  : attention that puts ALL mass on (prev_occurrence+1)
                         in head 0 of every layer (a perfect induction head).
    """

    def __init__(self, vocab_size, n_heads=4, n_layers=2):
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        from data import induction_oracle_predictions
        pred = induction_oracle_predictions(X)              # [B, T], -1 where none
        B, T = X.shape
        logits = torch.full((B, T, self.vocab_size), -10.0)
        safe = pred.clamp(min=0)
        logits.scatter_(2, safe[..., None], 10.0)
        # where oracle declines (-1), leave near-uniform (no confident copy).
        return logits

    def forward_with_attn(self, X):
        from data import prev_occurrence_index
        B, T = X.shape
        prev = prev_occurrence_index(X)
        src = (prev + 1).clamp(0, T - 1)
        attn = torch.zeros(B, self.n_heads, T, T)
        # head 0 = perfect induction head; others uniform-causal-ish (left at 0).
        for b in range(B):
            for t in range(T):
                if prev[b, t] >= 0:
                    attn[b, 0, t, src[b, t]] = 1.0
        logits = self.forward(X)
        return logits, [attn for _ in range(self.n_layers)]


def _self_test() -> int:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import InductionSpec, sample_batch

    torch.manual_seed(0)
    spec = InductionSpec(vocab_size=64, seq_len=128, period=0)
    batch = sample_batch(spec, n=64, seed=0)
    model = _OracleModel(spec.vocab_size, n_heads=4, n_layers=2)

    ok = True

    pa = position_accuracies(model, batch)
    s = icl_score(model, batch)
    print("SELF-TEST oracle ICL probe:")
    print(f"  repeat_acc = {pa['repeat_acc']:.4f} (expect ~1.0)")
    print(f"  first_acc  = {pa['first_acc']:.4f} (expect ~0, oracle declines)")
    print(f"  icl_score  = {s:.4f} (expect ~1.0)")
    if pa["repeat_acc"] < 0.999:
        ok = False; print("  FAIL: oracle repeat_acc not ~1.0")
    if s < 0.95:
        ok = False; print("  FAIL: oracle icl_score not ~1.0")

    pm = prefix_match_score(model, batch, layer=-1)
    print("SELF-TEST oracle prefix-match probe (last layer):")
    print(f"  per_head  = {[round(x,3) for x in pm['per_head']]}")
    print(f"  max_head  = {pm['max_head']:.4f} (expect ~1.0 on induction head)")
    if pm["max_head"] < 0.999:
        ok = False; print("  FAIL: induction head prefix-match not ~1.0")

    # emergence detection on a synthetic sharp sigmoid-like curve.
    steps = [10, 30, 100, 300, 1000, 3000, 10000]
    scores = [0.00, 0.01, 0.02, 0.05, 0.85, 0.95, 0.96]   # sharp jump near 1000
    em = detect_emergence(steps, scores, threshold=0.5)
    print("SELF-TEST emergence detection (synthetic sharp curve):")
    print(f"  emergence_step   = {em['emergence_step']} (expect 1000)")
    print(f"  max_slope        = {em['max_slope']:.3f} at step {em['max_slope_step']}")
    print(f"  transition_width = {em['transition_width']}")
    if em["emergence_step"] != 1000:
        ok = False; print("  FAIL: emergence step should be 1000")
    if em["max_slope"] <= 0:
        ok = False; print("  FAIL: max_slope should be positive")

    # a flat (no-emergence) curve must report emergence_step None.
    em0 = detect_emergence(steps, [0.0] * len(steps), threshold=0.5)
    if em0["emergence_step"] is not None:
        ok = False; print("  FAIL: flat curve should have no emergence step")

    print("PROBE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
