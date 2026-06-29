"""Direction 016 trainer — in-context TD emergence on random-MRP streams.

Reuses SeqTransformer from experiments/induction_emergence/model.py via the
root-README import convention (THIS dir first on sys.path; provenance:
induction_emergence is CLOSED infra for 007/010 — HARD RULE: no file there or
in grokking/ may be modified). Online fresh-MRP batches per 007's protocol.
"""

import argparse
import importlib.util as _ilu
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

import data as D  # noqa: E402
import probes as PR  # noqa: E402

_IE_MODEL = os.path.abspath(os.path.join(_THIS_DIR, "..",
                                         "induction_emergence", "model.py"))
_spec = _ilu.spec_from_file_location("ie_model", _IE_MODEL)
_ie = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ie)  # type: ignore[union-attr]
SeqTransformer = _ie.SeqTransformer
split_params_for_muon = _ie.split_params_for_muon
from muon import Muon  # noqa: E402  (grokking dir appended by ie_model)

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "results", "icrl_td")


class Config:
    def __init__(self, **kw):
        self.n_states = 10
        self.T = 20
        self.optimizer = "adamw"
        self.lr = 1e-3
        self.muon_lr = 0.02
        self.beta1, self.beta2 = 0.9, 0.98
        self.weight_decay = 0.01
        self.d_model, self.n_heads, self.n_layers, self.mlp_ratio = 128, 4, 2, 4
        self.init_scale = 1.0
        self.batch = 64
        self.steps = 10000
        self.eval_every = 50
        self.eval_mrps = 32
        self.acc_thresh = 0.7          # pre-registered; calibrated at smoke
        self.seed = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for k, v in kw.items():
            assert hasattr(self, k), f"unknown config key {k}"
            setattr(self, k, v)

    def name(self):
        return f"{self.optimizer}_T{self.T}_s{self.seed}"

    def meta(self):
        return {k: v for k, v in self.__dict__.items()}


def build_model(cfg):
    m = SeqTransformer(
        vocab_size=D.vocab_size(cfg.n_states), seq_len=2 * cfg.T + 2,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio, init_scale=cfg.init_scale).to(cfg.device)
    return m


def build_optimizer(model, cfg):
    """001-standard hybrid split (mirrors induction_emergence verbatim)."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)]
    if cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        return [Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                     ns_steps=5, weight_decay=cfg.weight_decay),
                torch.optim.AdamW(adamw_p, lr=cfg.lr,
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)]
    if cfg.optimizer == "sgdm":
        sgd_p, adamw_p = split_params_for_muon(model)
        return [torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                nesterov=True, weight_decay=cfg.weight_decay),
                torch.optim.AdamW(adamw_p, lr=cfg.lr,
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)]
    raise ValueError(cfg.optimizer)


# ---------------------------------------------------------------- evaluation

def _value_slice(cfg):
    lo = D.value_token(0, cfg.n_states)
    return lo, lo + D.V_BUCKETS


@torch.no_grad()
def model_value_estimate(model, cfg, tokens_np):
    """Bucket-midpoint value estimate(s) for sequence(s) [B?, L]."""
    t = torch.as_tensor(np.atleast_2d(tokens_np), device=cfg.device)
    logits = model(t)[:, -1, :]
    lo, hi = _value_slice(cfg)
    buckets = logits[:, lo:hi].argmax(-1).cpu().numpy()
    return np.array([D.value_midpoint(int(b)) for b in buckets]), buckets


@torch.no_grad()
def evaluate(model, cfg, rng_eval):
    """Held-out fresh MRPs: acc (strict / +-1 tolerant), MAE, probe trio."""
    model.eval()
    toks, tgt_tokens, metas = D.make_batch(rng_eval, cfg.eval_mrps, cfg.T,
                                           cfg.n_states, with_meta=True)
    est, buckets = model_value_estimate(model, cfg, toks)
    lo, _ = _value_slice(cfg)
    true_b = tgt_tokens - lo
    acc = float((buckets == true_b).mean())
    acc_tol = float((np.abs(buckets - true_b) <= 1).mean())
    true_v = np.array([m["mrp"]["V"][m["s_q"]] for m in metas])
    mae = float(np.abs(est - true_v).mean())

    # Bellman residual: full V-hat vector on the first few eval MRPs
    res = []
    for m in metas[:4]:
        mrp, states = m["mrp"], m["states"]
        qs = []
        for s in range(cfg.n_states):
            seq = np.empty(2 * cfg.T + 2, dtype=np.int64)
            seq[0:2 * cfg.T:2] = states
            seq[1:2 * cfg.T:2] = mrp["r_tok"][states]
            seq[2 * cfg.T] = D.query_token(cfg.n_states)
            seq[2 * cfg.T + 1] = s
            qs.append(seq)
        v_hat, _ = model_value_estimate(model, cfg, np.stack(qs))
        res.append(PR.bellman_residual(v_hat, mrp["P"], mrp["r_disc"]))
    bellman = float(np.mean(res))

    # TD-alignment on the first few eval MRPs (prefix grid). Probe query =
    # modal state of the trajectory (instrument-side choice, see probes.py);
    # the task query s_q stays random.
    ks = [k for k in range(4, cfg.T + 1, max(2, cfg.T // 8))]
    aligns = []
    for m in metas[:4]:
        mrp, states = m["mrp"], m["states"]

        def estimator(k, probe_s, _mrp=mrp, _states=states):
            seq = np.empty(2 * k + 2, dtype=np.int64)
            seq[0:2 * k:2] = _states[:k]
            seq[1:2 * k:2] = _mrp["r_tok"][_states[:k]]
            seq[2 * k] = D.query_token(cfg.n_states)
            seq[2 * k + 1] = int(probe_s)
            v, _ = model_value_estimate(model, cfg, seq)
            return float(v[0])

        aligns.append(PR.td_alignment(estimator, m, ks))
    td_align = float(np.mean(aligns))

    # kernel-attention probe (layer/head-mean, query-position row)
    t = torch.as_tensor(toks[:4], device=cfg.device)
    _, attn = model.forward_with_attn(t)
    attn_mean = torch.stack(attn).mean(dim=(0, 2))  # [B, T, T] mean L,H
    kern = []
    for i, m in enumerate(metas[:4]):
        row = attn_mean[i, -1].cpu().numpy()
        kern.append(PR.kernel_attention_score(
            row, m["state_positions"], m["states"], m["mrp"]["P"][m["s_q"]]))
    kernel = float(np.mean(kern))

    # P4 kill arm (online): reward-permutation tracking + OOD-structure acc.
    # Imitation control: same trajectories, permuted state->reward map changes
    # the ground truth; an in-context algorithm tracks the NEW values.
    n_p4 = min(8, len(metas))
    dv_true, dv_hat, perm_hits = [], [], 0
    for i in range(n_p4):
        m = metas[i]
        toks2, tgt_tok2, m2 = D.permuted_reward_variant(rng_eval, m, cfg.T)
        est2, b2 = model_value_estimate(model, cfg, toks2)
        dv_true.append(m2["mrp"]["V"][m["s_q"]] - m["mrp"]["V"][m["s_q"]])
        dv_hat.append(float(est2[0]) - float(est[i]))
        perm_hits += int(b2[0] == tgt_tok2 - lo)
    dv_true, dv_hat = np.array(dv_true), np.array(dv_hat)
    if dv_true.std() > 1e-9 and dv_hat.std() > 1e-9:
        p4_tracking = float(np.corrcoef(dv_true, dv_hat)[0, 1])
    else:
        p4_tracking = 0.0
    p4_acc_perm = perm_hits / max(1, n_p4)
    toks_o, tgt_o = D.make_batch(rng_eval, cfg.eval_mrps, cfg.T, cfg.n_states,
                                 concentration=1.0)  # denser kernels = OOD
    _, b_o = model_value_estimate(model, cfg, toks_o)
    acc_ood = float((b_o == (tgt_o - lo)).mean())

    model.train()
    return {"val_acc": acc, "val_acc_tol": acc_tol, "val_mae": mae,
            "bellman_residual": bellman, "td_alignment": td_align,
            "kernel_attention": kernel, "p4_tracking": p4_tracking,
            "p4_acc_perm": p4_acc_perm, "val_acc_ood": acc_ood}


# ---------------------------------------------------------------- training

def train(cfg, out_path=None, log=print):
    torch.manual_seed(cfg.seed)
    rng_train = np.random.default_rng(1_000_000 + cfg.seed)
    rng_eval = np.random.default_rng(50_000_000 + cfg.seed)  # disjoint stream
    model = build_model(cfg)
    opts = build_optimizer(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())

    fh = open(out_path, "w") if out_path else None

    def emit(rec):
        if fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    emit({"_meta": {**cfg.meta(), "n_params": n_params}})
    eval_steps, eval_accs, history = [], [], []
    for step in range(cfg.steps + 1):
        toks, tgt = D.make_batch(rng_train, cfg.batch, cfg.T, cfg.n_states)
        t = torch.as_tensor(toks, device=cfg.device)
        y = torch.as_tensor(tgt, device=cfg.device)
        logits = model(t)[:, -1, :]
        loss = F.cross_entropy(logits, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        if step % cfg.eval_every == 0:
            ev = evaluate(model, cfg, rng_eval)
            rec = {"step": step, "train_loss": float(loss.item()), **ev}
            emit(rec)
            history.append(rec)
            eval_steps.append(step)
            eval_accs.append(ev["val_acc"])
            if not math.isfinite(loss.item()):
                break
    em = PR.emergence_step(eval_steps, eval_accs, cfg.acc_thresh)
    last = history[-1] if history else {}
    summary = {**cfg.meta(), "n_params": n_params, "emergence_step": em,
               "final_val_acc": eval_accs[-1] if eval_accs else None,
               "final_td_alignment": last.get("td_alignment"),
               "final_p4_tracking": last.get("p4_tracking"),
               "final_val_acc_ood": last.get("val_acc_ood")}
    emit({"_summary": summary})
    if fh:
        fh.close()
    return summary, history


# ---------------------------------------------------------------- smoke

def run_smoke():
    """<60s CPU, writes nothing: probe self-tests + end-to-end mini-train."""
    assert PR.self_test(), "probe self-test failed"
    cfg = Config(n_states=6, T=8, d_model=32, n_heads=2, n_layers=2,
                 batch=16, steps=60, eval_every=30, eval_mrps=8,
                 device="cpu", optimizer="muon")
    summary, history = train(cfg, out_path=None, log=lambda *a: None)
    assert len(history) >= 2, "eval pipeline produced no records"
    for rec in history:
        assert math.isfinite(rec["train_loss"])
        for k in ("val_acc", "td_alignment", "bellman_residual",
                  "kernel_attention", "p4_tracking", "p4_acc_perm",
                  "val_acc_ood"):
            assert k in rec and math.isfinite(rec[k]), k
    # P4 instrument wiring: permuted-reward variant produces a valid sequence
    rng = np.random.default_rng(0)
    mrp = D.make_mrp(rng, cfg.n_states)
    toks, tgt, meta = D.sample_sequence(rng, mrp, cfg.T)
    toks2, tgt2, meta2 = D.permuted_reward_variant(rng, meta, cfg.T)
    assert toks2.shape == toks.shape
    assert not np.allclose(meta2["mrp"]["V"], mrp["V"])
    print(f"SMOKE PASS: probes self-test OK; mini-train {cfg.steps} steps "
          f"(muon hybrid, {summary['n_params']} params) finite; "
          f"eval+3 probes wired; P4 instrument OK; zero writes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
    else:
        print("use run_icrl.py for grid runs; train_icrl.py --smoke for smoke")
