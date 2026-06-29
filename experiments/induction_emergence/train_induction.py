"""Direction 007 — induction / ICL emergence trainer (online fresh-batch).

One run = one (optimizer, seq_len, seed) configuration. Studies how the
induction-head / in-context-learning circuit EMERGES over training and whether
Muon's orthogonalized update changes the timing, abruptness, and cross-seed
variance of that emergence vs AdamW / SGDM.

Design
------
- FULL-SEQUENCE causal-LM next-token prediction (SeqTransformer wrapper; the
  grokking GrokTransformer is reused UNMODIFIED via import).
- ONLINE training: a fresh induction batch every step (no fixed train set), so
  there is no memorization phase — emergence is a pure feature/circuit-formation
  signal. Eval uses a FIXED held-out stream (fixed seeds) so the ICL-score curve
  is comparable across steps and runs.
- Same muon/adamw/sgdm hybrid split as grokking (Muon on 2-D block matrices;
  AdamW on embeddings / unembed / norms). SGDM = Muon's momentum+lr WITHOUT
  Newton-Schulz orthogonalization (isolates orthogonalization).
- Per-eval jsonl log (real runs only): loss, first-occ acc, repeat acc,
  icl_score, prefix_match per head (final layer). The summary records the
  emergence step / abruptness from the ICL-score curve.

Flags
-----
--smoke : print the labeled smoke lines, run <=1 step, write NO files, exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# --- import grokking infra without modifying it (LOCAL first, grokking back) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch
import torch.nn.functional as F

from muon import Muon  # noqa: E402  (grokking infra)

from data import InductionSpec, sample_batch, batch_seed  # noqa: E402
from model import SeqTransformer, split_params_for_muon     # noqa: E402
from probes import (  # noqa: E402
    position_accuracies, icl_score, prefix_match_score, detect_emergence,
)


@dataclass
class Config:
    # task / data
    vocab_size: int = 64         # V
    seq_len: int = 128           # L
    period: int = 0              # repeated-segment length; 0 => auto (L // 4)
    batch_size: int = 64         # fresh batch per step (online)
    eval_batch: int = 256        # held-out eval batch (fixed stream)
    n_eval_batches: int = 1      # eval batches averaged per eval
    # model (grokking spec: 2 layers, 4 heads, d=128)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization
    optimizer: str = "adamw"     # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3             # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02        # Muon/SGDM lr for hidden matrices
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 10000
    eval_every: int = 100
    emergence_threshold: float = 0.5   # ICL-score level marking emergence
    seed: int = 0
    device: str = "cuda"


def make_spec(cfg: Config) -> InductionSpec:
    return InductionSpec(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len,
                         period=cfg.period)


def build_model(cfg: Config, spec: InductionSpec, device: str) -> SeqTransformer:
    return SeqTransformer(
        vocab_size=spec.vocab_size,
        seq_len=spec.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """Same muon/adamw/sgdm hybrid split as grokking's train.py."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    elif cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opt_muon = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_muon, opt_adamw]
    elif cfg.optimizer == "sgdm":
        # Mechanistic control: same hybrid split & momentum/lr as Muon but plain
        # SGD-momentum on the hidden matrices (NO Newton-Schulz). Isolates the
        # effect of orthogonalization vs the optimizer family / lr.
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    else:
        raise ValueError(cfg.optimizer)


def make_eval_batches(spec: InductionSpec, cfg: Config, device: str):
    """Fixed held-out eval stream (constant across steps and runs of same cfg)."""
    batches = []
    for j in range(cfg.n_eval_batches):
        # fixed seeds in a disjoint range from the training stream
        bs = batch_seed(50_000_000 + cfg.seed, j)
        batches.append(sample_batch(spec, cfg.eval_batch, seed=bs, device=device))
    return batches


@torch.no_grad()
def evaluate(model, eval_batches, layer: int = -1):
    """Average per-role acc/loss + ICL score + per-head prefix-match over the
    fixed eval stream."""
    keys = ["loss", "repeat_acc", "first_acc", "repeat_loss", "first_loss"]
    agg = {k: 0.0 for k in keys}
    n_heads = None
    pm_sum = None
    for batch in eval_batches:
        pa = position_accuracies(model, batch)
        for k in keys:
            agg[k] += pa[k]
        pm = prefix_match_score(model, batch, layer=layer)
        ph = torch.tensor(pm["per_head"])
        pm_sum = ph if pm_sum is None else pm_sum + ph
        n_heads = len(pm["per_head"])
    nb = max(1, len(eval_batches))
    for k in keys:
        agg[k] /= nb
    agg["icl_score"] = agg["repeat_acc"] - agg["first_acc"]
    agg["prefix_match_per_head"] = [float(x / nb) for x in pm_sum]
    agg["prefix_match_max"] = float(max(agg["prefix_match_per_head"]))
    return agg


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = make_spec(cfg)

    model = build_model(cfg, spec, device)
    optimizers = build_optimizer(model, cfg)
    eval_batches = make_eval_batches(spec, cfg, device)

    history: list = []
    t0 = time.time()

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            model.eval()
            ev = evaluate(model, eval_batches)
            rec = {"step": step, **ev}
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        # online fresh-batch causal-LM step
        model.train()
        Xb, Yb, _, tmask = sample_batch(
            spec, cfg.batch_size, seed=batch_seed(cfg.seed, step), device=device)
        logits = model(Xb)                              # [B, T, V]
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V), Yb.reshape(-1),
                             reduction="none").reshape(Yb.shape)
        loss = (ce * tmask).sum() / tmask.sum().clamp(min=1)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    steps_seq = [r["step"] for r in history]
    icl_seq = [r["icl_score"] for r in history]
    em = detect_emergence(steps_seq, icl_seq, threshold=cfg.emergence_threshold)
    summary = {
        **asdict(cfg),
        "final_icl_score": history[-1]["icl_score"],
        "final_repeat_acc": history[-1]["repeat_acc"],
        "final_first_acc": history[-1]["first_acc"],
        "final_prefix_match_max": history[-1]["prefix_match_max"],
        "emergence_step": em["emergence_step"],
        "emergence_max_slope": em["max_slope"],
        "emergence_transition_width": em["transition_width"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# ---------------------------------------------------------------------------
# Smoke: labeled lines, <=1 training step, NO files, exit 0.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = Config(vocab_size=64, seq_len=128, optimizer="muon",
                 batch_size=64, seed=0)
    spec = make_spec(cfg)

    # 1. one online batch shape
    X, Y, repeat_mask, target_mask = sample_batch(spec, cfg.batch_size, seed=0,
                                                  device=device)
    print(f"SMOKE DATASET SHAPE: X={tuple(X.shape)}, Y={tuple(Y.shape)}, "
          f"repeat_mask={tuple(repeat_mask.shape)}")

    # 2. param count
    model = build_model(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss (full-sequence causal LM)
    logits = model(X)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                         reduction="none").reshape(Y.shape)
    loss = (ce * target_mask).sum() / target_mask.sum().clamp(min=1)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one Muon-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: ICL + prefix-match probes end-to-end on the (untrained) model
    model.eval()
    batch = (X, Y, repeat_mask, target_mask)
    s = icl_score(model, batch)
    pm = prefix_match_score(model, batch, layer=-1)
    print(f"SMOKE ICL PROBE: icl_score={s:.4f} prefix_match={pm['max_head']:.4f}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser(description="induction/ICL emergence trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    a = vars(ap.parse_args())
    smoke = a.pop("smoke")
    cfg = Config(**a)
    return cfg, smoke


if __name__ == "__main__":
    cfg, smoke = parse_args()
    if smoke:
        run_smoke()
        sys.exit(0)
    summary, _ = run(cfg, out_path=None)
    print(json.dumps(summary, indent=2))
