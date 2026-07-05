#!/usr/bin/env python3
"""Consolidated analysis of the 2026-07-04/05 bolstering batches (1-3).

Prints every number the E1/E2 manuscript integration cites:
  [E2-LADDER] per-gamma: n, max-acc stats, single/sustained-2/sustained-3 counts
  [E2-M]      pure-noise single-cross rate vs M in {201,401,801} + prediction
  [E2-N]      pure-noise per-checkpoint exceedance q vs eval-set size N + Binom prediction
  [E1-ONSET]  small/{low,med} excess medians at n in {4,6,8,10,20}
  [E1-SHAPE]  med/med + large/high fine-grid excess vs n, log-log slope
  [E1-REAL]   WikiText vs code-bytes bridge: R_free + per-n excess (all seeds)
CPU only; safe to run while batches are in flight (prints current n per cell).
"""
from __future__ import annotations
import glob, json, math, statistics as st
from pathlib import Path

RES = Path('/home/zeyufu/Desktop/dl-research/experiments/results')
GL = RES / 'icrl_td_gamma_ladder'


def jlast(f):
    last = None
    for line in open(f):
        if line.strip():
            last = json.loads(line)
    return last


def jrows(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def accs_of(f):
    return [r['val_acc'] for r in jrows(f) if 'val_acc' in r]


def sus_k(accs, k, thr=0.8):
    run = 0
    for a in accs:
        run = run + 1 if a >= thr else 0
        if run >= k:
            return True
    return False


# ---------------- E2 ladder ----------------
print('=' * 30, 'E2-LADDER (T=40, M=201)', '=' * 30)
for g in ('0p0', '0p3', '0p5', '0p7', '0p9'):
    fs = sorted(glob.glob(str(GL / f'gamma{g}_T40_s*.jsonl')))
    fs = [f for f in fs if 'M401' not in f and 'M801' not in f and '_N' not in f]
    if not fs:
        continue
    n = len(fs)
    mx, single, s2, s3, qs = [], 0, 0, 0, []
    for f in fs:
        a = accs_of(f)
        mx.append(max(a))
        single += any(x >= 0.8 for x in a)
        s2 += sus_k(a, 2)
        s3 += sus_k(a, 3)
        qs.append(sum(x >= 0.8 for x in a) / len(a))
    M = len(accs_of(fs[0]))
    qbar = st.mean(qs)
    pred_single = 1 - (1 - qbar) ** M
    pred_s2 = (M - 1) * qbar ** 2
    pred_s3 = (M - 2) * qbar ** 3
    print(f"gamma={g.replace('p','.')}: n={n} M={M} max_acc med {st.median(mx):.3f} "
          f"range [{min(mx):.3f},{max(mx):.3f}] qbar={qbar:.4f}")
    print(f"   observed single {single}/{n}  sus2 {s2}/{n}  sus3 {s3}/{n}")
    print(f"   predicted per-run: single {pred_single:.3f}  sus2 {pred_s2:.3f}  sus3 {pred_s3:.4f}")

# ---------------- E2 M-axis ----------------
print('=' * 30, 'E2-M (gamma=0 pure noise)', '=' * 30)
base = sorted(glob.glob(str(GL / 'gamma0p0_T40_s*.jsonl')))
base = [f for f in base if 'M401' not in f and 'M801' not in f and '_N' not in f]
qpool = []
for f in base:
    a = accs_of(f)
    qpool += [x >= 0.8 for x in a]
q0 = sum(qpool) / len(qpool) if qpool else float('nan')
print(f'baseline q (M=201 cells, pooled checkpoints): {q0:.5f}')
for tag, pat in (('M=201', base),
                 ('M=401', sorted(glob.glob(str(GL / 'gamma0p0_M401_T40_s*.jsonl')))),
                 ('M=801', sorted(glob.glob(str(GL / 'gamma0p0_M801_T40_s*.jsonl'))))):
    fs = pat
    if not fs:
        continue
    singles, s2c, Ms = 0, 0, []
    for f in fs:
        a = accs_of(f)
        Ms.append(len(a))
        singles += any(x >= 0.8 for x in a)
        s2c += sus_k(a, 2)
    M = round(st.mean(Ms))
    pred = 1 - (1 - q0) ** M
    print(f'{tag}: n={len(fs)} M~{M} observed single {singles}/{len(fs)} '
          f'(pred {pred:.3f}/run)  sus2 {s2c}/{len(fs)}')

# ---------------- E2 N-axis ----------------
print('=' * 30, 'E2-N (gamma=0, exceedance q vs eval-set size)', '=' * 30)


def binom_tail(N, p, thr=0.8):
    k0 = math.ceil(thr * N)
    return sum(math.comb(N, k) * p ** k * (1 - p) ** (N - k) for k in range(k0, N + 1))


for N, pat in ((16, f'gamma0p0_N16_T40_s*.jsonl'), (32, None), (64, f'gamma0p0_N64_T40_s*.jsonl')):
    if pat is None:
        fs = base
    else:
        fs = sorted(glob.glob(str(GL / pat)))
    if not fs:
        continue
    qs, ps = [], []
    for f in fs:
        a = accs_of(f)
        qs.append(sum(x >= 0.8 for x in a) / len(a))
        ps.append(st.mean(a))
    pbar = st.mean(ps)
    print(f'N={N}: n={len(fs)} observed q={st.mean(qs):.5f}  mean acc p={pbar:.4f}  '
          f'Binom(N,p) tail>=0.8 pred q={binom_tail(N, pbar):.5f}')

# ---------------- E1 onsets + shape ----------------
def summ(f):
    return jlast(f)['_summary']


def cell_excess(cap, ent, E, U_tag):
    """median excess over available seeds; searches main+audit+finegrid dirs"""
    pats = [RES / 'repeated_data' / f'{cap}_{ent}_U{U_tag}_E{E}_s*.jsonl',
            RES / 'repeated_data_ultragoal_seed_audit' / f'{cap}_{ent}_U{U_tag}_E{E}_s*.jsonl',
            RES / 'repeated_data_finegrid' / f'{cap}_{ent}_U{U_tag}_E{E}_s*.jsonl']
    fs = sorted(set(sum([glob.glob(str(p)) for p in pats], [])))
    fresh = [summ(f)['final_val_loss'] for f in
             glob.glob(str(RES / 'repeated_data' / f'{cap}_{ent}_U20M_E1_s*.jsonl'))]
    if not fs or not fresh:
        return None
    fm = st.median(fresh)
    ex = [summ(f)['final_val_loss'] - fm for f in fs]
    cg = [summ(f)['final_canary_gap'] for f in fs]
    return dict(n=len(ex), med=st.median(ex), lo=min(ex), hi=max(ex),
                canary_med=st.median(cg))


print('=' * 30, 'E1-ONSET (small cells)', '=' * 30)
for ent in ('low', 'med'):
    for E, U_tag in ((4, '5M'), (6, '3p33333M'), (8, '2p5M'), (10, '2M'), (20, '1M')):
        r = cell_excess('small', ent, E, U_tag)
        if r:
            print(f"small/{ent} n={E:2d}: seeds={r['n']:2d} excess med {r['med']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] canary {r['canary_med']:+.4f}")

print('=' * 30, 'E1-SHAPE (fine grid)', '=' * 30)
for cap, ent in (('med', 'med'), ('large', 'high')):
    pts = []
    for E, U_tag in ((2, '10M'), (4, '5M'), (6, '3p33333M'), (8, '2p5M'),
                     (10, '2M'), (12, '1p66667M'), (16, '1p25M'), (20, '1M'), (40, '0p5M')):
        r = cell_excess(cap, ent, E, U_tag)
        if r:
            print(f"{cap}/{ent} n={E:2d}: seeds={r['n']:2d} excess med {r['med']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] canary {r['canary_med']:+.4f}")
            if r['med'] > 0.05:
                pts.append((E, r['med']))
    if len(pts) >= 3:
        xs = [math.log(e) for e, _ in pts]
        ys = [math.log(v) for _, v in pts]
        n = len(xs)
        b = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / \
            (n * sum(x * x for x in xs) - sum(xs) ** 2)
        print(f'  {cap}/{ent} post-threshold log-log slope: {b:.2f} '
              f'(points n={[e for e,_ in pts]})')

# ---------------- E1 real-text ----------------
print('=' * 30, 'E1-REAL (byte-level bridges, budget 50M, med cap)', '=' * 30)
for label, d in (('WikiText', RES / 'repeated_data_realtext'),
                 ('code', RES / 'repeated_data_realtext_code')):
    fs = sorted(glob.glob(str(d / 'med_b50M_s*.json')))
    if not fs:
        continue
    rfree, rowsets = [], []
    for f in fs:
        v = json.load(open(f))
        rfree.append(v['R_free'])
        rowsets.append({k: r['excess'] for k, r in v['rows'].items()})
    ns = sorted({int(k) for rs in rowsets for k in rs}, key=int)
    print(f'{label}: seeds={len(fs)} R_free={rfree}')
    for n in ns:
        vals = [rs[str(n)] for rs in rowsets if str(n) in rs]
        print(f'   n={n:2d}: excess med {st.median(vals):+.4f} '
              f'[{min(vals):+.4f},{max(vals):+.4f}] (n={len(vals)})')
