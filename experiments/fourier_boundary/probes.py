"""Direction 006 — Fourier-feature probes.

Three measurements that operationalize "did clean per-frequency Fourier circuits
emerge?" on the contiguous integer-token axis of the embedding/unembedding:

  (i)  embedding_fft(model, n_int)
         Row-wise real FFT of the token-embedding and unembedding matrices over
         the integer-token axis x = 0..n_int-1. Returns the per-frequency power
         spectrum (averaged over the embedding dimension) and a spectral SPARSITY
         INDEX = fraction of spectral power carried by the top-k frequencies. A
         clean Fourier solution concentrates power in a few frequencies
         (sparsity -> 1); a low-frequency / diffuse solution does not.

  (ii) freq_logit_attribution(model, X, n_int, answer_size)
         Project the model's logits over the answer space onto the Fourier basis
         of that answer space, per example, and return the mean power per answer
         frequency. Tells which output frequencies the logits actually use.

  (iii) band_ablation(model, band, n_int, eval_fn)
         Zero a frequency band in the token-embedding via FFT -> mask -> iFFT on a
         DEEP COPY of the model (never mutates the live model), then measure the
         accuracy drop reported by `eval_fn(ablated_model)`. A causal test of
         whether a band carries task-relevant structure.

Run `python probes.py` for a self-test: a synthetic embedding whose integer-token
rows are pure cos(2*pi*k*x/n_int) must rank frequency k as the top spectral peak
with sparsity ≈ 1.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class FFTResult:
    power: torch.Tensor          # [n_freq] mean per-frequency power (tok emb)
    power_unembed: torch.Tensor  # [n_freq] mean per-frequency power (unembed)
    top_freqs: list              # top-k frequency indices (token embedding)
    sparsity: float              # fraction of power in top-k freqs (token embedding)
    sparsity_unembed: float      # same for unembedding
    n_int: int
    top_k: int


def _row_fft_power(rows: torch.Tensor) -> torch.Tensor:
    """Mean (over columns) per-frequency power of a real row-wise FFT.

    rows: [n_int, d]  (each column is a signal over the integer-token axis).
    Returns [n_freq] with n_freq = n_int // 2 + 1, the DC term zeroed out so the
    sparsity index measures *oscillatory* concentration, not the constant offset.
    """
    rows = rows.detach().float()
    rows = rows - rows.mean(dim=0, keepdim=True)        # remove per-column DC
    spec = torch.fft.rfft(rows, dim=0)                  # [n_freq, d]
    power = (spec.real ** 2 + spec.imag ** 2).mean(dim=1)  # [n_freq]
    power[0] = 0.0                                      # drop residual DC bin
    return power


def _sparsity(power: torch.Tensor, top_k: int) -> tuple[float, list]:
    total = float(power.sum())
    if total <= 0.0:
        return 0.0, []
    vals, idx = torch.topk(power, min(top_k, power.shape[0]))
    frac = float(vals.sum()) / total
    return frac, [int(i) for i in idx.tolist()]


@torch.no_grad()
def embedding_fft(model, n_int: int, top_k: int = 5) -> FFTResult:
    """Row-wise FFT power spectrum + sparsity index of token-emb and unembed.

    Only the contiguous integer-token rows 0..n_int-1 are scanned (the answer/
    integer axis); EQ and any digit-only tokens beyond n_int are excluded.
    """
    tok = model.tok_emb.weight[:n_int]            # [n_int, d]
    une = model.unembed.weight[:n_int]            # [n_int, d] (out_features axis)

    p_tok = _row_fft_power(tok)
    p_une = _row_fft_power(une)
    spars_tok, top_tok = _sparsity(p_tok, top_k)
    spars_une, _ = _sparsity(p_une, top_k)

    return FFTResult(
        power=p_tok, power_unembed=p_une,
        top_freqs=top_tok, sparsity=spars_tok, sparsity_unembed=spars_une,
        n_int=n_int, top_k=top_k,
    )


@torch.no_grad()
def freq_logit_attribution(model, X: torch.Tensor, answer_size: int) -> torch.Tensor:
    """Mean power of the logits projected onto the Fourier basis of the answer space.

    Returns [n_freq] (n_freq = answer_size // 2 + 1), DC bin zeroed. High power at
    frequency k means the model's output distribution oscillates at frequency k
    across the answer axis — the signature of a per-frequency answer circuit.
    """
    logits = model(X)                              # [N, vocab]
    logits = logits[:, :answer_size].float()       # restrict to the answer axis
    logits = logits - logits.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(logits, dim=1)           # [N, n_freq]
    power = (spec.real ** 2 + spec.imag ** 2).mean(dim=0)  # [n_freq]
    power[0] = 0.0
    return power


@torch.no_grad()
def band_ablation(model, band: tuple[int, int], n_int: int, eval_fn) -> dict:
    """Zero a frequency band [lo, hi) in the token-embedding (FFT->mask->iFFT).

    Operates on a DEEP COPY — the live model is never mutated. Returns a dict with
    the eval_fn result before/after and the accuracy drop. `eval_fn(m) -> float`
    must return an accuracy (or any scalar metric) for model `m`.
    """
    lo, hi = band
    base_metric = float(eval_fn(model))

    ablated = copy.deepcopy(model)
    w = ablated.tok_emb.weight.data[:n_int].float()    # [n_int, d]
    mean = w.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(w - mean, dim=0)             # [n_freq, d]
    n_freq = spec.shape[0]
    lo_c = max(0, min(lo, n_freq))
    hi_c = max(0, min(hi, n_freq))
    spec[lo_c:hi_c] = 0.0
    w_new = torch.fft.irfft(spec, n=n_int, dim=0) + mean
    ablated.tok_emb.weight.data[:n_int] = w_new.to(ablated.tok_emb.weight.dtype)

    abl_metric = float(eval_fn(ablated))
    return {
        "band": (lo_c, hi_c),
        "metric_base": base_metric,
        "metric_ablated": abl_metric,
        "drop": base_metric - abl_metric,
    }


# ---------------------------------------------------------------------------
# Self-test: a synthetic embedding with pure cos(2*pi*k*x/n_int) integer rows
# must rank frequency k as the top spectral peak with sparsity ≈ 1.
# ---------------------------------------------------------------------------
class _ToyModel:
    """Minimal stand-in exposing the .tok_emb.weight / .unembed.weight the probe reads."""

    def __init__(self, tok_w: torch.Tensor, une_w: torch.Tensor):
        self.tok_emb = type("E", (), {"weight": tok_w})()
        self.unembed = type("U", (), {"weight": une_w})()


def _self_test() -> int:
    torch.manual_seed(0)
    n_int = 97
    d = 32
    k_true = 7

    x = torch.arange(n_int).float()
    # every embedding column is the SAME pure cosine at frequency k_true (plus a
    # tiny per-column phase so it isn't degenerate), so the FFT must spike at k.
    phases = torch.linspace(0, 0.5, d)
    rows = torch.cos(2 * torch.pi * k_true * x[:, None] / n_int + phases[None, :])
    tok_w = rows.clone()
    une_w = rows.clone()

    model = _ToyModel(tok_w, une_w)
    res = embedding_fft(model, n_int=n_int, top_k=5)

    top1 = res.top_freqs[0]
    print(f"SELF-TEST embedding_fft: true_k={k_true} top_freqs={res.top_freqs} "
          f"sparsity={res.sparsity:.4f} sparsity_unembed={res.sparsity_unembed:.4f}")

    ok_freq = (top1 == k_true)
    ok_sparsity = (res.sparsity > 0.99)
    print(f"  top1==k_true: {ok_freq}   sparsity>0.99: {ok_sparsity}")

    # band ablation sanity: zeroing the band containing k_true must collapse the
    # row-FFT power at k_true to ~0 on the copy, without mutating the original.
    def _power_at_k(m):
        p = embedding_fft(m, n_int=n_int, top_k=5).power
        return float(p[k_true])

    abl = band_ablation(model, band=(k_true, k_true + 1), n_int=n_int,
                        eval_fn=_power_at_k)
    ok_ablate = (abl["metric_ablated"] < 1e-6 * abl["metric_base"] + 1e-6)
    ok_nomutate = (float(embedding_fft(model, n_int=n_int).power[k_true]) > 0)
    print(f"  band_ablation drop={abl['drop']:.4e} ablated≈0: {ok_ablate} "
          f"live_model_unmutated: {ok_nomutate}")

    ok = ok_freq and ok_sparsity and ok_ablate and ok_nomutate
    print("PROBE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
