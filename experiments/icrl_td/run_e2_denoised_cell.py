from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path('/home/zeyufu/Desktop/dl-research')
ICRL=ROOT/'experiments'/'icrl_td'
sys.path.insert(0,str(ICRL))
import train_icrl as TI
OUT=ROOT/'experiments'/'results'/'icrl_td_ultragoal_denoised'
OUT.mkdir(parents=True, exist_ok=True)

def done(path):
    if not path.exists(): return False
    last=''
    for line in path.read_text().splitlines():
        if line.strip(): last=line
    try: return '_summary' in json.loads(last)
    except Exception: return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--optimizer',required=True); ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args()
    cfg=TI.Config(optimizer=args.optimizer,T=40,seed=args.seed,eval_mrps=128,steps=3000,eval_every=100)
    path=OUT/f'{cfg.name()}_denoised_eval128_steps3000.jsonl'
    if path.exists() and not done(path):
        path.unlink()
    if done(path):
        print(f'skip {path.name}', flush=True); return
    summary,_=TI.train(cfg,out_path=str(path),log=lambda *a: None)
    print(json.dumps({'cell':cfg.name(),'path':str(path.relative_to(ROOT)),'emergence':summary['emergence_step'],'final':summary['final_val_acc']}), flush=True)
if __name__=='__main__': main()
