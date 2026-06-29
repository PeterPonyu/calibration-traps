"""Direction 006 — Fourier-boundary trainer (full-/large-batch AdamW).

One run = one (modular, tokenization, coverage, target, p_or_max, seed) cell.
Adapts the grokking trainer (Config dataclass + dynamic argparse) but:
  - FIXED optimizer = AdamW (NO optimizer sweep — scope discipline vs direction
    004, which owns the muon/adamw/sgdm comparison);
  - variable vocab_size / seq_len come from the factorized data.py meta;
  - per-eval logging adds the spectral SPARSITY index + the top-frequency list so
    the emergence (or non-emergence) of clean Fourier circuits is tracked online.

GrokTransformer is reused by import from the sibling grokking package. Per the
repo import discipline (README + degree_staircase pattern): THIS dir goes to the
FRONT of sys.path (so the local `data.py` / `probes.py` win over grokking's
same-named `data.py`), and the grokking dir is APPENDED to the back (we pull only
`model` from there, which does not collide).

Modules from grokking : model (GrokTransformer)
Modules local         : data (make_arith_dataset, DEFAULT_*), probes (embedding_fft)

Flags
-----
--smoke : print the labeled smoke lines, run <=1 AdamW step, write NO files, exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# --- import grokking infra without modifying it (spec-mandated pattern) -------
# Local dir FIRST (our data.py/probes.py must shadow grokking's data.py);
# grokking dir APPENDED last (we only pull `model`, which does not collide).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch
import torch.nn.functional as F

from model import GrokTransformer            # noqa: E402  (grokking infra)

from data import make_arith_dataset, DEFAULT_MODULAR_P, DEFAULT_NONMODULAR_MAX  # noqa: E402
from probes import embedding_fft                                                 # noqa: E402


@dataclass
class Config:
    # task / data (the four factorial axes + the operand range)
    p_or_max: int = DEFAULT_MODULAR_P     # modulus p (modular) or operand bound max
    modular: bool = True                  # wrap-around vs plain integer addition
    tokenization: str = "single"          # "single" | "digit"
    coverage: float = 0.4                 # train coverage fraction
    target: str = "mod"                   # "sum" | "mod"
    digit_base: int = 10
    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization (FIXED AdamW — no optimizer sweep here)
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    batch_size: int = 0                   # 0 => full-batch; else large-batch SGD
    steps: int = 20000
    eval_every: int = 100
    seed: int = 0
    device: str = "cuda"
    top_k: int = 5                        # top-k freqs for the sparsity index


def build_data(cfg: Config, device: str):
    return make_arith_dataset(
        p_or_max=cfg.p_or_max, modular=cfg.modular, tokenization=cfg.tokenization,
        coverage=cfg.coverage, target=cfg.target, seed=cfg.seed,
        digit_base=cfg.digit_base, device=device)


def build_model(cfg: Config, meta, device: str) -> GrokTransformer:
    return GrokTransformer(
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """FIXED AdamW (scope discipline: no optimizer family sweep in direction 006)."""
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)


@torch.no_grad()
def evaluate(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(-1) == Y).float().mean().item()
    return loss, acc


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    (Xtr, Ytr), (Xte, Yte), meta = build_data(cfg, device)
    model = build_model(cfg, meta, device)
    optimizer = build_optimizer(model, cfg)

    history: list = []
    t0 = time.time()

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": {**asdict(cfg), "data_meta": _meta_dict(meta)}}) + "\n")

    n_train = Xtr.shape[0]
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            tr_loss, tr_acc = evaluate(model, Xtr, Ytr)
            te_loss, te_acc = evaluate(model, Xte, Yte)
            fft = embedding_fft(model, n_int=meta.n_int, top_k=cfg.top_k)
            rec = {
                "step": step,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "test_loss": te_loss, "test_acc": te_acc,
                "sparsity": fft.sparsity,
                "sparsity_unembed": fft.sparsity_unembed,
                "top_freqs": fft.top_freqs,
            }
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        # gradient step: full-batch (batch_size==0) or a fresh large minibatch
        model.train()
        if cfg.batch_size and cfg.batch_size < n_train:
            g = torch.Generator(device=Xtr.device).manual_seed(step * 100003 + cfg.seed)
            sel = torch.randperm(n_train, generator=g, device=Xtr.device)[:cfg.batch_size]
            xb, yb = Xtr[sel], Ytr[sel]
        else:
            xb, yb = Xtr, Ytr
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    summary = {
        **asdict(cfg),
        "data_meta": _meta_dict(meta),
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "final_sparsity": history[-1]["sparsity"],
        "final_sparsity_unembed": history[-1]["sparsity_unembed"],
        "final_top_freqs": history[-1]["top_freqs"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


def _meta_dict(meta) -> dict:
    return {
        "vocab_size": meta.vocab_size, "seq_len": meta.seq_len,
        "n_int": meta.n_int, "answer_size": meta.answer_size,
        "n_train": meta.n_train, "n_test": meta.n_test,
        "digit_width": meta.digit_width,
    }


# ---------------------------------------------------------------------------
# Smoke: labeled lines, <=1 AdamW step, NO files, exit 0, <60s.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = Config(p_or_max=DEFAULT_MODULAR_P, modular=True, tokenization="single",
                 coverage=0.4, target="mod", seed=0)

    # 1. dataset shape
    (Xtr, Ytr), (Xte, Yte), meta = build_data(cfg, device)
    print(f"SMOKE DATASET SHAPE: Xtr={tuple(Xtr.shape)} Ytr={tuple(Ytr.shape)} "
          f"Xte={tuple(Xte.shape)} vocab={meta.vocab_size} seq_len={meta.seq_len} "
          f"n_int={meta.n_int}")

    # 2. param count
    model = build_model(cfg, meta, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss (full-batch train slice)
    logits = model(Xtr)
    loss = F.cross_entropy(logits, Ytr)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one AdamW step
    optimizer = build_optimizer(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: embedding-FFT probe on the (untrained) model
    fft = embedding_fft(model, n_int=meta.n_int, top_k=cfg.top_k)
    print(f"SMOKE FFT PROBE: sparsity={fft.sparsity:.4f} top_freqs={fft.top_freqs}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser(description="Direction 006 Fourier-boundary trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=_str2bool, default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    a = vars(ap.parse_args())
    smoke = a.pop("smoke")
    cfg = Config(**a)
    return cfg, smoke


def _str2bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "t")


if __name__ == "__main__":
    cfg, smoke = parse_args()
    if smoke:
        run_smoke()
    else:
        summary, _ = run(cfg, out_path=None)
        print(json.dumps({k: summary[k] for k in
                          ("final_train_acc", "final_test_acc", "final_sparsity",
                           "final_top_freqs", "n_params", "stopped_step")}, indent=2))
