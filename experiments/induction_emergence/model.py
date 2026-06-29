"""Direction 007 — full-sequence transformer wrapper for the induction task.

The grokking GrokTransformer is a 2-layer / 4-head / d=128 causal decoder, but
its `forward` returns ONLY the last-position logits (`logits[:, -1, :]`) because
the grokking modular task predicts a single answer token. The induction / ICL
study needs next-token logits at EVERY position. We therefore reuse the grokking
architecture UNMODIFIED and add a thin local wrapper, `SeqTransformer`, that
re-runs the same embedding -> blocks -> ln_f -> unembed stack but returns the
full `[B, T, vocab]` logit tensor. The original grokking file is not touched.

We also expose the per-layer attention patterns (needed by the prefix-match
probe in probes.py) via `forward_with_attn`, which recomputes scaled-dot-product
attention weights from the same q/k as the blocks (read-only; no architecture or
weight change).

Muon hybrid split
-----------------
`split_params_for_muon` from grokking/muon.py is purely NAME-based (2-D params
whose name lacks "emb"/"unembed" go to Muon). The wrapper keeps the same module
names (`tok_emb`, `pos_emb`, `blocks.*`, `unembed`) so that split function works
unchanged. We re-export a local `split_params_for_muon` for a self-contained
import surface and to make the split explicit at this layer.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

# Import the grokking infra without modifying it: LOCAL dir first (so our
# `data`/`model`/`probes` win on name collisions), grokking dir APPENDED (we
# only pull `model`/`muon`, which do not collide).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

# `grokking/model.py` -> GrokTransformer; importing the module object avoids the
# name collision with THIS file (also called model.py) on the local-first path.
import importlib.util as _ilu

_grok_model_path = os.path.join(_GROKKING_DIR, "model.py")
_spec = _ilu.spec_from_file_location("grokking_model", _grok_model_path)
_grok_model = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_grok_model)  # type: ignore[union-attr]
GrokTransformer = _grok_model.GrokTransformer

from muon import split_params_for_muon as _grok_split  # noqa: E402 (grokking infra)


class SeqTransformer(GrokTransformer):
    """GrokTransformer that returns next-token logits at ALL positions.

    Identical parameters / init to GrokTransformer (so the Muon split and param
    count are unchanged); only the forward output slice differs.
    """

    def forward(self, idx):  # type: ignore[override]
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.unembed(x)        # [B, T, vocab]
        return logits                   # full sequence (vs grokking's [:, -1])

    @torch.no_grad()
    def forward_with_attn(self, idx):
        """Return (logits [B,T,vocab], attn_list) where attn_list[l] is the
        per-head causal attention weight tensor [B, n_heads, T, T] for layer l.

        Recomputes attention weights from each block's q/k (read-only probe);
        the forward output is identical to `forward`.
        """
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        attn_list = []
        causal = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        for blk in self.blocks:
            B, _, C = x.shape
            h = blk.ln1(x)
            qkv = blk.qkv(h).view(B, T, 3, blk.n_heads, blk.d_head)
            q, k, v = qkv.unbind(dim=2)              # each [B, T, n_heads, d_head]
            q = q.transpose(1, 2)                    # [B, n_heads, T, d_head]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (blk.d_head ** 0.5)  # [B,H,T,T]
            scores = scores.masked_fill(~causal[None, None], float("-inf"))
            attn = scores.softmax(dim=-1)            # [B, n_heads, T, T]
            attn_list.append(attn)
            ctx = attn @ v                           # [B, n_heads, T, d_head]
            ctx = ctx.transpose(1, 2).reshape(B, T, C)
            x = x + blk.proj(ctx)
            h2 = blk.ln2(x)
            x = x + blk.fc2(F.gelu(blk.fc1(h2)))
        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits, attn_list


def split_params_for_muon(model):
    """Partition params into (muon_2d, adamw_other). Re-exported from grokking.

    Name-based: 2-D weight matrices inside transformer blocks -> Muon; embeddings,
    unembed head, layernorm weights/biases -> AdamW. SeqTransformer keeps the same
    module names as GrokTransformer, so the grokking splitter applies unchanged.
    """
    return _grok_split(model)
