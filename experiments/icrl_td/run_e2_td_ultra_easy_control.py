#!/usr/bin/env python3
"""Run the ultra-easy E2 TD appendix control in an isolated, resumable way."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'icrl_td'))
sys.path.append(str(ROOT / 'experiments'))
import train_icrl as TI  # noqa: E402
OUT = ROOT / 'experiments' / 'results' / 'ultragoal_20260618' / 'e2_td_positive_control'
OUT.mkdir(parents=True, exist_ok=True)
cells = []
for seed in range(3):
    cells.append(TI.Config(
        optimizer='adamw', seed=seed, n_states=2, T=4, d_model=96,
        n_heads=2, n_layers=2, mlp_ratio=2, batch=256, steps=4000,
        eval_every=50, eval_mrps=256, acc_thresh=0.8, lr=3e-3,
        weight_decay=0.0,
    ))


def completed_summary(path):
    if not path.exists():
        return None
    history = []
    last = None
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            last = rec
            if '_summary' not in rec:
                history.append(rec)
    if isinstance(last, dict) and '_summary' in last:
        return annotate_summary(last['_summary'], history)
    return None


def annotate_summary(summary, history=None, elapsed=None):
    if elapsed is not None:
        summary['elapsed_wall_sec'] = elapsed
    if history is None:
        return summary
    summary['max_val_acc'] = max((r.get('val_acc', 0) for r in history), default=0)
    summary['sustain2_0p8'] = any(
        history[j].get('val_acc', 0) >= 0.8 and history[j + 1].get('val_acc', 0) >= 0.8
        for j in range(max(0, len(history) - 1))
    )
    return summary


summaries = []
for i, cfg in enumerate(cells, 1):
    path = OUT / f'ultraeasy_{cfg.name()}.jsonl'
    summary = completed_summary(path)
    if summary is not None:
        print(f'[{i}/{len(cells)}] skip done {path.name}', flush=True)
        summaries.append(summary)
        continue
    print(f'[{i}/{len(cells)}] run {path.name} device={cfg.device}', flush=True)
    t0 = time.time()
    summary, hist = TI.train(cfg, out_path=str(path), log=lambda *_: None)
    annotate_summary(summary, hist, time.time() - t0)
    summaries.append(summary)
    print(f"[{i}/{len(cells)}] final={summary['final_val_acc']} max={summary['max_val_acc']} sustain2={summary['sustain2_0p8']}", flush=True)

appendix = {
    'criterion': 'n_states=2,T=4, positive if any AdamW seed sustain2>=0.8',
    'summaries': summaries,
    'n_positive_cells': sum(
        1 for s in summaries if s.get('max_val_acc', 0) >= 0.8 and s.get('sustain2_0p8')
    ),
}
(OUT / 'ultra_easy_positive_control_verdict.json').write_text(
    json.dumps(appendix, indent=2, allow_nan=True) + '\n'
)
print(json.dumps(appendix, indent=2, allow_nan=True), flush=True)
