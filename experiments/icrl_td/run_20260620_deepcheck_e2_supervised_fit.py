#!/usr/bin/env python3
"""E2 same-pipeline supervised Bellman-label fit sanity.

Purpose: not a TD-emergence positive. It asks whether the exact E2 model,
loss/evaluator tokenization, optimizer, and LaTeX evidence path can fit a finite
Bellman-labelled support set. Fresh-MRP generalization is reported separately.
"""
import argparse, json, math, sys, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "icrl_td"))
sys.path.append(str(ROOT / "experiments"))
import data as D  # noqa:E402
import train_icrl as TI  # noqa:E402

OUT = ROOT / "experiments/results/ultragoal_20260620_deepcheck/e2_supervised_fit"
OUT.mkdir(parents=True, exist_ok=True)


def archive_incomplete(p: Path, reason: str) -> None:
    target = p.with_name(f"{p.name}.{reason}.{int(time.time())}")
    p.rename(target)
    print(f"archived incomplete evidence {p} -> {target}", flush=True)


def completed_summary(p: Path):
    if not p.exists():
        return None
    last = None
    try:
        for line in p.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
    except json.JSONDecodeError:
        archive_incomplete(p, 'corrupt_json')
        return None
    if isinstance(last, dict) and '_summary' in last:
        return last['_summary']
    archive_incomplete(p, 'missing_summary')
    return None


def make_support(seed, n, T, n_states):
    rng = np.random.default_rng(seed)
    toks, tgt = D.make_batch(rng, n, T, n_states)
    return toks, tgt


def acc_on(model, cfg, toks, tgt):
    model.eval()
    bs = 512
    hits = 0; total = 0; ce_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(toks), bs):
            x = torch.as_tensor(toks[i:i+bs], device=cfg.device)
            y = torch.as_tensor(tgt[i:i+bs], device=cfg.device)
            logits = model(x)[:, -1, :]
            ce = F.cross_entropy(logits, y, reduction='sum')
            pred = logits.argmax(-1)
            hits += int((pred == y).sum().item())
            total += len(x)
            ce_sum += float(ce.item())
    model.train()
    return {'acc': hits/max(1,total), 'ce': ce_sum/max(1,total), 'n': total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--support', type=int, default=512)
    ap.add_argument('--n-states', type=int, default=2)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=96)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--eval-every', type=int, default=100)
    args = ap.parse_args()
    summaries=[]
    for seed in args.seeds:
        path=OUT / f'supervised_fit_adamw_S{args.n_states}_T{args.T}_support{args.support}_s{seed}.jsonl'
        existing = completed_summary(path)
        if existing is not None:
            print(f'skip existing {path.name}', flush=True)
            summaries.append(existing); continue
        cfg = TI.Config(optimizer='adamw', seed=seed, n_states=args.n_states, T=args.T,
                        d_model=args.d_model, n_heads=2, n_layers=2, mlp_ratio=2,
                        batch=args.batch, steps=args.steps, eval_every=args.eval_every,
                        eval_mrps=256, acc_thresh=0.8, lr=args.lr, weight_decay=0.0)
        torch.manual_seed(seed)
        model = TI.build_model(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        toks, tgt = make_support(1_000_000+seed, args.support, args.T, args.n_states)
        val_toks, val_tgt = toks.copy(), tgt.copy()  # finite-support fit sanity, intentionally same support
        rng = np.random.default_rng(7_000_000+seed)
        t0=time.time(); hist=[]
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        with tmp_path.open('w') as fh:
            meta={'_meta': {**cfg.meta(), 'task':'finite_support_supervised_bellman_fit', 'support':args.support, 'note':'same finite support train/val; fresh-MRP eval reported separately'}}
            fh.write(json.dumps(meta)+'\n'); fh.flush()
            for step in range(args.steps+1):
                idx = rng.integers(0, len(toks), size=args.batch)
                x=torch.as_tensor(toks[idx], device=cfg.device)
                y=torch.as_tensor(tgt[idx], device=cfg.device)
                logits=model(x)[:, -1, :]
                loss=F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if step % args.eval_every == 0:
                    tr=acc_on(model,cfg,toks,tgt)
                    fresh=TI.evaluate(model,cfg,np.random.default_rng(50_000_000+seed+step))
                    rec={'step':step,'train_loss':float(loss.item()),'support_acc':tr['acc'],'support_ce':tr['ce'],
                         'fresh_val_acc':fresh['val_acc'],'fresh_val_acc_tol':fresh['val_acc_tol'],'fresh_mae':fresh['val_mae']}
                    fh.write(json.dumps(rec)+'\n'); fh.flush(); hist.append(rec)
                    print(f"seed={seed} step={step} support={tr['acc']:.3f} fresh={fresh['val_acc']:.3f} loss={loss.item():.3g}", flush=True)
                    if tr['acc'] >= 0.98 and step >= 300:
                        break
            summ={'seed':seed, **cfg.meta(), 'support':args.support, 'elapsed_wall_sec':time.time()-t0,
                  'max_support_acc':max(h['support_acc'] for h in hist), 'final_support_acc':hist[-1]['support_acc'],
                  'max_fresh_val_acc':max(h['fresh_val_acc'] for h in hist), 'final_fresh_val_acc':hist[-1]['fresh_val_acc'],
                  'pass_support_fit_0p98':max(h['support_acc'] for h in hist)>=0.98,
                  'interpretation':'supports optimizer/model/evaluator finite-support fit; does not establish fresh-MRP TD emergence'}
            fh.write(json.dumps({'_summary':summ})+'\n')
            summaries.append(summ)
        os.replace(tmp_path, path)
    verdict={'namespace':str(OUT.relative_to(ROOT)), 'task':'E2 finite-support supervised Bellman-label fit sanity',
             'criterion':'pass if any seed support_acc>=0.98; fresh-MRP accuracy reported separately and not required',
             'n_cells':len(summaries), 'n_support_fit_pass':sum(s.get('pass_support_fit_0p98') for s in summaries),
             'summaries':summaries}
    verdict['status']='supervised_fit_pass' if verdict['n_support_fit_pass'] else 'supervised_fit_not_passed'
    (OUT/'supervised_fit_verdict.json').write_text(json.dumps(verdict, indent=2, allow_nan=True)+'\n')
    print(json.dumps(verdict, indent=2, allow_nan=True), flush=True)

if __name__ == '__main__': main()
