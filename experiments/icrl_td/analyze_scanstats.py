"""Deferred CPU analyses for the scan-statistic calibration paper (E2).

Computes, from the 45 in-context-TD logs in results/icrl_td/:
  1. Exact binomial q(0.8), q(0.7) (survival function, no normal approximation).
  2. P_single = 1-(1-q)^M and the exact sustained-K false-positive rate via
     finite-Markov-chain imbedding; the suppression factor.
  3. Lag-1 autocorrelation rho_1 of the centered validation signal per run
     (mean-centered and linearly detrended variants).
  4. sigma_run(T) and p_run(T): within-run std and mean of val_acc by horizon T,
     to adjudicate the N_eff(T) / p(T) mechanism behind Corollary tdep.
  5. Direct mixture computation E_p[P_single(p)] over the empirical 45-cell
     p-distribution, to replace the Jensen intuition.

Pure CPU; reads jsonl logs only.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results" / "icrl_td"
N = 32          # held-out sequences per eval
M = 201         # checkpoints (steps 0,50,...,10000)
TAU_MAIN = 0.8
TAU_PRE = 0.7


def exact_q(tau: float, p: float, n: int = N) -> float:
    """P(Binom(n,p)/n >= tau) with the >= tau accuracy criterion."""
    k = math.ceil(tau * n)
    return float(stats.binom.sf(k - 1, n, p))


def p_single(q: float, m: int = M) -> float:
    return 1.0 - (1.0 - q) ** m


def p_sustain_exact(q: float, k: int, m: int = M) -> float:
    """Exact P(success run >= k in m Bernoulli(q) trials) by Markov-chain imbedding."""
    # states 0..k-1 = current run length, k absorbing
    v = np.zeros(k + 1)
    v[0] = 1.0
    for _ in range(m):
        nv = np.zeros(k + 1)
        for s in range(k):
            nv[0] += v[s] * (1 - q)
            nv[min(s + 1, k)] += v[s] * q
        nv[k] += v[k]
        v = nv
    return float(v[k])


def load_runs():
    runs = {}
    for f in sorted(RESULTS.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        meta = rows[0]["_meta"] if "_meta" in rows[0] else None
        recs = [r for r in rows if "val_acc" in r]
        acc = np.array([r["val_acc"] for r in recs])
        runs[f.stem] = {"meta": meta, "acc": acc}
    return runs


def lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return float("nan")
    return float(np.dot(x[:-1], x[1:]) / denom)


def lag1_detrended(x: np.ndarray) -> float:
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return lag1_autocorr(x - (slope * t + intercept))


def main():
    runs = load_runs()
    assert len(runs) == 45, f"expected 45 logs, got {len(runs)}"
    lens = {len(v["acc"]) for v in runs.values()}
    print(f"runs: {len(runs)}, checkpoints per run: {sorted(lens)}")

    # ---- 1. exact binomial q ----
    p_med = float(np.median([np.median(v["acc"]) for v in runs.values()]))
    print(f"\nmedian per-run median val_acc (baseline p): {p_med:.4f}")
    for p_base in (0.562, p_med):
        q08 = exact_q(TAU_MAIN, p_base)
        q07 = exact_q(TAU_PRE, p_base)
        print(f"  p={p_base:.4f}: exact q(0.8)={q08:.6f}  exact q(0.7)={q07:.6f}")

    q08 = exact_q(TAU_MAIN, 0.562)
    q07 = exact_q(TAU_PRE, 0.562)

    # ---- 2. single-crossing + sustained-K, exact ----
    ps = p_single(q08)
    ps2_exact = p_sustain_exact(q08, 2)
    ps2_approx = (M - 1) * q08 ** 2
    print(f"\nexact q(0.8) = {q08:.6f}  (paper printed 0.00272 via normal-z)")
    print(f"P_single = {ps:.4f}  expected {45*ps:.1f}/45   (paper: 0.421, 18.9/45)")
    print(f"P_sustain-2 exact MC-imbed = {ps2_exact:.6f}; (M-1)q^2 = {ps2_approx:.6f}")
    print(f"expected sustained /45 = {45*ps2_exact:.3f}")
    print(f"suppression factor = {ps/ps2_exact:.1f}x  (paper: 286x)")
    ps07 = p_single(q07)
    print(f"q(0.7): exact={q07:.5f}, P_single={ps07:.6f} (paper: ~1.000, 45/45)")

    # ---- 3. lag-1 autocorrelation ----
    r1_mean = np.array([lag1_autocorr(v["acc"]) for v in runs.values()])
    r1_det = np.array([lag1_detrended(v["acc"]) for v in runs.values()])
    for name, arr in (("mean-centered", r1_mean), ("linearly detrended", r1_det)):
        print(f"\nlag-1 autocorr ({name}): median={np.median(arr):+.4f} "
              f"IQR=[{np.percentile(arr,25):+.4f},{np.percentile(arr,75):+.4f}] "
              f"min={arr.min():+.4f} max={arr.max():+.4f} "
              f"n>0: {(arr>0).sum()}/45")
    # 95% white-noise band for n=201: ~ +-1.96/sqrt(201)
    band = 1.96 / math.sqrt(M)
    print(f"white-noise 95% band: +-{band:.4f}; runs outside (detrended): "
          f"{(np.abs(r1_det) > band).sum()}/45")
    # inflation factor for sustained-K under AR(1)-like positive dependence:
    # P(next also exceeds | exceed) ~ q + rho*(1-q) crude upper bound -> Gamma(rho)
    rho_med = float(np.median(r1_det))
    if rho_med > 0:
        q_cond = q08 + rho_med * (1 - q08)
        ps2_infl = (M - 1) * q08 * q_cond
        print(f"rho_1>0: inflated sustained-2 bound (M-1)*q*(q+rho(1-q)) = {ps2_infl:.6f} "
              f"-> suppression {ps/ps2_infl:.1f}x")
    else:
        print("median rho_1 <= 0: no inflation of the sustained-K bound; "
              "the i.i.d. suppression factor stands")

    # ---- 4. sigma_run(T) and p_run(T) ----
    print("\nby horizon T:")
    byT = {}
    for v in runs.values():
        T = v["meta"]["T"]
        byT.setdefault(T, []).append(v["acc"])
    for T in sorted(byT):
        accs = byT[T]
        sig = [float(np.std(a, ddof=1)) for a in accs]
        mu = [float(np.mean(a)) for a in accs]
        print(f"  T={T:>2} (n={len(accs)} runs): sigma_run median={np.median(sig):.4f} "
              f"[{min(sig):.4f},{max(sig):.4f}]  p_run median={np.median(mu):.4f} "
              f"[{min(mu):.4f},{max(mu):.4f}]")
    sig_all = [float(np.std(v["acc"], ddof=1)) for v in runs.values()]
    print(f"  all runs: sigma_run median={np.median(sig_all):.4f} "
          f"(binomial prediction at p=0.562: {math.sqrt(0.562*0.438/N):.4f})")
    # Spearman: does sigma grow with T?
    Ts = np.array([v["meta"]["T"] for v in runs.values()])
    sig_arr = np.array([float(np.std(v["acc"], ddof=1)) for v in runs.values()])
    mu_arr = np.array([float(np.mean(v["acc"])) for v in runs.values()])
    rs_sig_r, rs_sig_p = stats.spearmanr(Ts, sig_arr)
    rs_mu_r, rs_mu_p = stats.spearmanr(Ts, mu_arr)
    print(f"  spearman sigma_run vs T: r={rs_sig_r:+.3f} p={rs_sig_p:.3f}")
    print(f"  spearman p_run    vs T: r={rs_mu_r:+.3f} p={rs_mu_p:.3f}")

    # ---- 5. direct mixture computation ----
    # per-run p estimated as the run's mean val_acc (the stationary-baseline estimate)
    crossings = 0
    ps_list = []
    for v in runs.values():
        p_run = float(np.mean(v["acc"]))
        q_run = exact_q(TAU_MAIN, p_run)
        ps_list.append(p_single(q_run))
        if (v["acc"] >= TAU_MAIN).any():
            crossings += 1
    mix = float(np.mean(ps_list))
    print(f"\nobserved single crossings: {crossings}/45 = {crossings/45:.3f}")
    print(f"mixture E_p[P_single(p)] over empirical 45-cell p-hat: {mix:.3f} "
          f"= {45*mix:.1f}/45")
    print(f"P_single at median p: {p_single(exact_q(TAU_MAIN, p_med)):.3f}")
    print(f"mixture >= observed? {mix >= crossings/45}  "
          f"mixture >= median-p value? {mix >= p_single(exact_q(TAU_MAIN, p_med))}")

    # sustained-2 observed
    sust = 0
    for v in runs.values():
        e = v["acc"] >= TAU_MAIN
        if (e[:-1] & e[1:]).any():
            sust += 1
    print(f"observed sustained-2 events: {sust}/45")


if __name__ == "__main__":
    main()
