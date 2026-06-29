"""Direction 011 — curriculum-schedule trainer (data-presentation curriculum).

One run = one (family, ordering, optimizer, seed) configuration. It trains a
GrokTransformer-family model under a ``CurriculumSchedule`` (curriculum.py):
stage-based dataset switching (easy->hard across stages) + ordered batch
presentation WITHIN each stage ({iid, easy_to_hard, hard_first, structured}).
Eval is ALWAYS on the FINAL/hard target (the last stage's spec), so the question
"does a data curriculum substitute for the 004 target ladder?" reduces to: does
any ordering arm let the weak optimizer (sgdm) reach the hard target that the
single-target run could not?

Reuse (all UNMODIFIED, loaded by file path via importlib to dodge the triple
``data.py`` / double ``model.py`` name collisions):
  * grokking/model.py    -> GrokTransformer  (walsh/modadd: predict at EQ position)
  * induction_emergence/model.py -> SeqTransformer (copy: full-sequence next-token)
  * grokking/muon.py     -> Muon + split_params_for_muon (the optimizer hybrid)
  * degree_staircase/probes.py -> model_scalar, degree_correlations
  * data families via data_router.py (sibling module; uniform interface).

Optimizers: {adamw, sgdm}. sgdm is the WEAK-OPTIMIZER probe (the 004 failure
case). muon is deliberately NOT an arm here — scope discipline (011 is about the
data curriculum, not the optimizer geometry).

Flags
-----
--smoke : print the labeled smoke lines, run <=1 step, write NO files, exit 0.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

# --- local-first import surface (data_router / curriculum are siblings) ------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)

import curriculum as curr            # noqa: E402  (local)
import data_router                   # noqa: E402  (local)


def _load_by_path(unique_name: str, path: str):
    """Load an external module by file path (sys.modules-registered for py3.13)."""
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = _ilu.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {unique_name} from {path}")
    module = _ilu.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


# grokking infra + the two model heads + the Walsh probe, all by path.
_GROKKING_DIR = os.path.join(_EXPERIMENTS_DIR, "grokking")
_INDUCTION_DIR = os.path.join(_EXPERIMENTS_DIR, "induction_emergence")
_DEGREE_DIR = os.path.join(_EXPERIMENTS_DIR, "degree_staircase")

_gk_model = _load_by_path("curr011_gk_model", os.path.join(_GROKKING_DIR, "model.py"))
_gk_muon = _load_by_path("curr011_gk_muon", os.path.join(_GROKKING_DIR, "muon.py"))
_ds_probes = _load_by_path("curr011_ds_probes",
                           os.path.join(_DEGREE_DIR, "probes.py"))

GrokTransformer = _gk_model.GrokTransformer
Muon = _gk_muon.Muon
split_params_for_muon = _gk_muon.split_params_for_muon
model_scalar = _ds_probes.model_scalar
degree_correlations = _ds_probes.degree_correlations


def _seq_transformer_cls():
    """Load SeqTransformer (full-sequence head) for the copy family on demand.

    induction_emergence/model.py imports `muon`/`model` off sys.path; we add the
    induction + grokking dirs to sys.path TAIL just for this load so its internal
    `from muon import ...` and its own importlib of grokking/model.py resolve,
    without letting them shadow our local `data`/`model`/`curriculum`.
    """
    for d in (_INDUCTION_DIR, _GROKKING_DIR):
        if d not in sys.path:
            sys.path.append(d)
    mod = _load_by_path("curr011_ind_model",
                        os.path.join(_INDUCTION_DIR, "model.py"))
    return mod.SeqTransformer


@dataclass
class Config:
    # task / family
    family: str = "walsh"        # "walsh" | "modadd" | "copy"
    ordering: str = "iid"        # "iid" | "easy_to_hard" | "hard_first" | "structured"
    # schedule
    steps_per_stage: int = 800   # optimizer steps per curriculum stage
    # walsh ladder
    walsh_final_degree: int = 3  # top pure degree of the easy->hard ramp
    walsh_target: str = "staircase"  # 2026-06-13 REDESIGN: eval target =
    # learnable degree-staircase (004/017-proven, fit ~0.9). "pure" = the
    # original unlearnable pure-deg-3 target (kept for the negative record).
    L: int = 16
    D: int = 4
    # modadd ladder
    modadd_p: int = 23
    modadd_stages: int = 3
    # copy ladder
    copy_final_period: int = 8
    copy_vocab: int = 32
    copy_seq_len: int = 32
    copy_stages: int = 3
    # batch
    batch_size: int = 4096       # large-batch / full-batch-ish per step
    eval_n: int = 8192           # eval-set size on the FINAL target
    # model (grokking spec)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization (011 arms: adamw | sgdm ; NO muon)
    optimizer: str = "adamw"
    lr: float = 1e-3             # AdamW lr (and AdamW side of the sgdm hybrid)
    sgdm_lr: float = 0.02        # SGD-momentum lr for hidden matrices
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    eval_every: int = 50
    emergence_threshold: float = 0.5  # fit-corr / acc level marking emergence
    seed: int = 0
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Schedule construction (delegates to curriculum.py ladders per family)
# ---------------------------------------------------------------------------
def make_schedule(cfg: Config) -> curr.CurriculumSchedule:
    if cfg.family == "walsh":
        return curr.build_schedule(
            "walsh", ordering=cfg.ordering, steps_per_stage=cfg.steps_per_stage,
            final_degree=cfg.walsh_final_degree, L=cfg.L, D=cfg.D,
            final_target=cfg.walsh_target)
    if cfg.family == "modadd":
        return curr.build_schedule(
            "modadd", ordering=cfg.ordering, steps_per_stage=cfg.steps_per_stage,
            p=cfg.modadd_p, n_stages=cfg.modadd_stages)
    if cfg.family == "copy":
        return curr.build_schedule(
            "copy", ordering=cfg.ordering, steps_per_stage=cfg.steps_per_stage,
            final_period=cfg.copy_final_period, vocab_size=cfg.copy_vocab,
            seq_len=cfg.copy_seq_len, n_stages=cfg.copy_stages)
    raise ValueError(f"unknown family {cfg.family!r}")


def family_dims(cfg: Config, sched: curr.CurriculumSchedule, device: str):
    """vocab_size / seq_len / final-target loader bits, from one FINAL-spec batch."""
    final = sched.final_spec
    probe = data_router.load_family(cfg.family, n=8, seed=cfg.seed,
                                    device=device, **final)
    return probe.meta["vocab_size"], probe.meta["seq_len"], final


def build_model(cfg: Config, vocab_size: int, seq_len: int, device: str):
    """GrokTransformer for walsh/modadd (EQ head); SeqTransformer for copy."""
    kw = dict(vocab_size=vocab_size, seq_len=seq_len, d_model=cfg.d_model,
              n_heads=cfg.n_heads, n_layers=cfg.n_layers, mlp_ratio=cfg.mlp_ratio,
              init_scale=cfg.init_scale)
    if cfg.family == "copy":
        return _seq_transformer_cls()(**kw).to(device)
    return GrokTransformer(**kw).to(device)


def build_optimizer(model, cfg: Config):
    """011 arms only: adamw (single) | sgdm (hybrid split, weak-optimizer probe)."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    if cfg.optimizer == "sgdm":
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.sgdm_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    raise ValueError(f"011 optimizer arm must be adamw|sgdm, got {cfg.optimizer!r}")


# ---------------------------------------------------------------------------
# Per-family loss + eval (on the FINAL/hard target always)
# ---------------------------------------------------------------------------
def _walsh_loss(model, batch):
    f = model(batch.X)
    pred = f[:, 1] - f[:, 0]            # scalar regression readout at EQ
    return F.mse_loss(pred, batch.Y)


def _modadd_loss(model, batch):
    logits = model(batch.X)            # [B, vocab] at EQ
    return F.cross_entropy(logits, batch.Y)


def _copy_loss(model, batch):
    logits = model(batch.X)            # [B, T, vocab]
    tm = batch.meta["target_mask"]
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), batch.Y.reshape(B * T),
                           reduction="none").reshape(B, T)
    return (loss * tm.float()).sum() / tm.float().sum().clamp(min=1.0)


def family_loss(cfg: Config, model, batch):
    return {"walsh": _walsh_loss, "modadd": _modadd_loss,
            "copy": _copy_loss}[cfg.family](model, batch)


@torch.no_grad()
def evaluate(cfg: Config, model, sched, device, stage_idx):
    """Loss/acc + (walsh) per-degree correlations on the FIXED FINAL target."""
    final = sched.final_spec
    batch = data_router.load_family(cfg.family, n=cfg.eval_n,
                                    seed=10_000 + cfg.seed, device=device, **final)
    model.eval()
    rec = {"stage": stage_idx}
    if cfg.family == "walsh":
        f = model_scalar(model, batch.X)
        rec["eval_loss"] = F.mse_loss(f, batch.Y).item()
        fc = f - f.mean(); yc = batch.Y - batch.Y.mean()
        denom = fc.norm() * yc.norm()
        rec["eval_fit_corr"] = float(torch.dot(fc, yc) / denom) if denom > 1e-12 else 0.0
        sets = batch.meta["sets"]
        dc = degree_correlations(f, batch.meta["signs"], sets)
        rec["deg_corr"] = {str(k): v for k, v in dc.items()}
        rec["eval_acc"] = rec["eval_fit_corr"]   # progress key reuses fit-corr
    elif cfg.family == "modadd":
        logits = model(batch.X)
        rec["eval_loss"] = F.cross_entropy(logits, batch.Y).item()
        rec["eval_acc"] = float((logits.argmax(-1) == batch.Y).float().mean())
    else:  # copy
        logits = model(batch.X)
        tm = batch.meta["target_mask"]; rm = batch.meta["repeat_mask"]
        B, T, V = logits.shape
        ce = F.cross_entropy(logits.reshape(B * T, V), batch.Y.reshape(B * T),
                             reduction="none").reshape(B, T)
        rec["eval_loss"] = float((ce * tm.float()).sum() / tm.float().sum().clamp(min=1.0))
        correct = (logits.argmax(-1) == batch.Y)
        rep_n = rm.float().sum().clamp(min=1.0)
        rec["eval_acc"] = float((correct & rm).float().sum() / rep_n)  # repeat acc
    return rec


# ---------------------------------------------------------------------------
# Training loop (stage switching + within-stage ordered presentation)
# ---------------------------------------------------------------------------
def _family_kw(cfg: Config, stage_spec: dict) -> dict:
    """Merge stage overrides with fixed family knobs for the data_router loader."""
    return dict(stage_spec)


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    sched = make_schedule(cfg)
    vocab_size, seq_len, _final = family_dims(cfg, sched, device)
    model = build_model(cfg, vocab_size, seq_len, device)
    optimizers = build_optimizer(model, cfg)

    history: list = []
    emergence_step = None
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    boundaries = sched.stage_boundaries()
    total = sched.total_steps
    for step in range(total + 1):
        stage_idx = sched.stage_at_step(min(step, total - 1))
        if step % cfg.eval_every == 0 or step == total:
            rec = evaluate(cfg, model, sched, device, stage_idx)
            rec["step"] = step
            if emergence_step is None and rec["eval_acc"] >= cfg.emergence_threshold:
                emergence_step = step
            rec["emergence_step"] = emergence_step
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n"); f.flush()
        if step == total:
            break

        # --- present one ordered batch from the CURRENT stage's dataset ---
        stage_spec = sched.stages[stage_idx].dataset_spec
        batch = data_router.load_family(
            cfg.family, n=cfg.batch_size, seed=step * 100003 + cfg.seed,
            device=device, **_family_kw(cfg, stage_spec))
        order = curr.order_indices(batch.scores(), sched.ordering,
                                   seed=step * 7919 + cfg.seed)
        batch.X = batch.X[order]
        batch.Y = batch.Y[order]
        if "target_mask" in batch.meta:
            batch.meta["target_mask"] = batch.meta["target_mask"][order]
            batch.meta["repeat_mask"] = batch.meta["repeat_mask"][order]

        model.train()
        loss = family_loss(cfg, model, batch)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    summary = {
        **asdict(cfg),
        "n_stages": len(sched.stages),
        "stage_boundaries": boundaries,
        "final_eval_loss": history[-1]["eval_loss"],
        "final_eval_acc": history[-1]["eval_acc"],
        "emergence_step": emergence_step,
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if cfg.family == "walsh":
        summary["final_deg_corr"] = history[-1]["deg_corr"]
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n"); f.close()
    return summary, history


# ---------------------------------------------------------------------------
# Smoke: labeled lines, <=1 training step, NO files, exit 0.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # tiny: one Walsh stage batch, single step, no eval/jsonl.
    cfg = Config(family="walsh", ordering="easy_to_hard", optimizer="sgdm",
                 steps_per_stage=1, batch_size=256, seed=0, device=device)
    sched = make_schedule(cfg)
    vocab_size, seq_len, _ = family_dims(cfg, sched, device)

    # 1. one Walsh stage batch shape (first stage of the schedule)
    stage0 = sched.stages[0].dataset_spec
    batch = data_router.load_family("walsh", n=cfg.batch_size, seed=0,
                                    device=device, **stage0)
    order = curr.order_indices(batch.scores(), sched.ordering, seed=0)
    Xo = batch.X[order]
    print(f"SMOKE DATASET SHAPE: X={tuple(Xo.shape)}, "
          f"y={tuple(batch.Y.shape)}, stage={sched.stages[0].name}")

    # 2. param count
    model = build_model(cfg, vocab_size, seq_len, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss (FINAL-target readout on the presented batch)
    batch.X = Xo; batch.Y = batch.Y[order]
    loss = family_loss(cfg, model, batch)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one sgdm-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: schedule self-check + all three family adapters load
    n_stages = len(sched.stages)
    diff = batch.scores()
    orderings_ok = True
    for o in curr.ORDERINGS:
        idx = curr.order_indices(diff, o, seed=1)
        if int(torch.sort(idx).values[-1]) != idx.shape[0] - 1 or \
                idx.numel() != diff.numel():
            orderings_ok = False
    families_loaded = 0
    probe_cases: list[tuple[str, int, dict]] = [
        ("walsh", 8, dict(profile="pure", pure_degree=1)),
        ("modadd", 8, dict(p=cfg.modadd_p)),
        ("copy", 4, dict(vocab_size=cfg.copy_vocab, seq_len=cfg.copy_seq_len)),
    ]
    for fam, n, kw in probe_cases:
        b = data_router.load_family(fam, n, 0, device=device, **kw)
        if b.X.ndim == 2 and b.difficulty_scores is not None:
            families_loaded += 1
    print(f"SMOKE CURRICULUM PROBE: stages={n_stages} "
          f"orderings_ok={orderings_ok} families_loaded={families_loaded}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser()
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
