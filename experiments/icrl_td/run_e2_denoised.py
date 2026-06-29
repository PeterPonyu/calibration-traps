from __future__ import annotations
import json, os, sys, math
from pathlib import Path
import numpy as np
ROOT=Path('/home/zeyufu/Desktop/dl-research')
ICRL=ROOT/'experiments'/'icrl_td'
sys.path.insert(0,str(ICRL))
import train_icrl as TI
OUT=ROOT/'experiments'/'results'/'icrl_td_ultragoal_denoised'
OUT.mkdir(parents=True, exist_ok=True)

def done(path: Path):
    if not path.exists(): return False
    last=''
    for line in path.read_text().splitlines():
        if line.strip(): last=line
    try: return '_summary' in json.loads(last)
    except Exception: return False

def smooth(vals,k=5):
    vals=np.asarray(vals,float)
    if len(vals)<k: return vals
    return np.convolve(vals, np.ones(k)/k, mode='valid')

def sustained(steps, vals, thresh=.8, k=2):
    vals=list(vals)
    for i in range(0, len(vals)-k+1):
        if all(v>=thresh for v in vals[i:i+k]):
            return steps[i]
    return None

def analyze_file(path: Path):
    hist=[]; meta={}; summary={}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        o=json.loads(line)
        if '_meta' in o: meta=o['_meta']
        elif '_summary' in o: summary=o['_summary']
        else: hist.append(o)
    steps=[h['step'] for h in hist]
    acc=[h['val_acc'] for h in hist]
    tol=[h['val_acc_tol'] for h in hist]
    sm=smooth(acc,5)
    sm_steps=steps[2:2+len(sm)] if len(steps)>=5 else steps
    return {
        'path': str(path.relative_to(ROOT)), 'meta': meta, 'summary': summary,
        'n_eval_points': len(hist), 'final_acc': acc[-1] if acc else None,
        'max_acc': max(acc) if acc else None, 'median_last5_acc': float(np.median(acc[-5:])) if acc else None,
        'single_ge_08_step': next((s for s,a in zip(steps,acc) if a>=.8), None),
        'sustained2_ge_08_step': sustained(steps,acc,.8,2),
        'sustained5_ge_08_step': sustained(steps,acc,.8,5),
        'smooth5_ge_08_step': next((s for s,a in zip(sm_steps,sm) if a>=.8), None),
        'final_tol_acc': tol[-1] if tol else None,
        'max_tol_acc': max(tol) if tol else None,
    }

def main():
    # Bounded reviewer-facing rerun: same hard T=40 cell family, stronger held-out eval.
    cells=[]
    for opt in ['adamw','muon','sgdm']:
        for seed in [5,6,7]:
            cells.append(TI.Config(optimizer=opt,T=40,seed=seed,eval_mrps=128,steps=3000,eval_every=100))
    manifest=[]
    for i,cfg in enumerate(cells,1):
        path=OUT/f'{cfg.name()}_denoised_eval128_steps3000.jsonl'
        if done(path):
            print(f'[{i}/{len(cells)}] skip {path.name}')
        else:
            summary,_=TI.train(cfg,out_path=str(path),log=lambda *a: None)
            print(f'[{i}/{len(cells)}] {cfg.name()} emergence={summary["emergence_step"]} final={summary["final_val_acc"]}')
        manifest.append(analyze_file(path))
    verdict={
        'namespace':'experiments/results/icrl_td_ultragoal_denoised',
        'config':{'optimizers':['adamw','muon','sgdm'],'T':[40],'seeds':[5,6,7],'eval_mrps':128,'steps':3000,'eval_every':100,'threshold':0.8},
        'runs':manifest,
    }
    verdict['counts']={
        'n':len(manifest),
        'single_ge_08':sum(r['single_ge_08_step'] is not None for r in manifest),
        'sustained2_ge_08':sum(r['sustained2_ge_08_step'] is not None for r in manifest),
        'sustained5_ge_08':sum(r['sustained5_ge_08_step'] is not None for r in manifest),
        'smooth5_ge_08':sum(r['smooth5_ge_08_step'] is not None for r in manifest),
        'median_final_acc':float(np.median([r['final_acc'] for r in manifest])),
        'max_acc':float(max(r['max_acc'] for r in manifest)),
    }
    (OUT/'denoised_verdict.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps(verdict['counts'],indent=2))
if __name__=='__main__': main()
