#!/usr/bin/env python3
"""E2 Tier-0 revision analyses (2026-07-16), CPU-only, from raw jsonl logs.

T0-1: full tau=0.7 headline chain on the 45-run icrl_td grid
      (single-crossing counts, sustained-K ladder K=1..5, per-run mixture
      predictions, pooled-q nomogram reading, mixture-prescribed K, severity
      factors), with the tau=0.8 chain recomputed alongside for verification.
T0-2a: cluster-aware replacement for the 171-cell independent-Bernoulli 95%
      upper bound (Beta-Binomial profile over the dependence parameter ->
      the bound is unidentified between the independent and cluster-level
      endpoints; both endpoints computed).
T0-2b: k-fold (k=5) out-of-sample mixture validation on the gamma=0.9
      60-seed drift regime (replaces the 2-fold split-half), plus a
      reproduction of the published split-half numbers.
T0-2c: lag-1..5 autocorrelation audit on the mean-centered and linearly
      detrended validation traces of the 45 runs.

Outputs: t0_analysis_output.json + printed summary.
"""
import json
import math
import os
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = {}


def load_trace(path, key="val_acc"):
    vals = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if key not in d:  # _meta / _summary / _gamma_summary records
                continue
            vals.append(d[key])
    return vals


def max_runlen(bits):
    best = cur = 0
    for b in bits:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def p_sustain_exact(q, M, K):
    """Exact P(success run >= K in M Bernoulli(q) trials), FMCI recurrence."""
    # state = current run length 0..K-1, K absorbing
    probs = [1.0] + [0.0] * (K - 1)
    absorbed = 0.0
    for _ in range(M):
        new = [0.0] * K
        for s, ps in enumerate(probs):
            if ps == 0.0:
                continue
            new[0] += ps * (1 - q)
            if s + 1 >= K:
                absorbed += ps * q
            else:
                new[s + 1] += ps * q
        probs = new
    return absorbed


def mixture_pred(qhats, M, K):
    """Per-run mixture prediction sum_i [1 - exp(-(M-K+1) qhat_i^K)]
    (the paper's formula) and the exact-FMCI mixture alongside."""
    approx = sum(1 - math.exp(-(M - K + 1) * q ** K) for q in qhats)
    exact = sum(p_sustain_exact(q, M, K) for q in qhats)
    return approx, exact


def autocorr(x, lag):
    n = len(x)
    m = sum(x) / n
    xc = [v - m for v in x]
    denom = sum(v * v for v in xc)
    if denom == 0:
        return 0.0
    return sum(xc[i] * xc[i + lag] for i in range(n - lag)) / denom


def detrend(x):
    n = len(x)
    t = list(range(n))
    mt = (n - 1) / 2.0
    mx = sum(x) / n
    stt = sum((ti - mt) ** 2 for ti in t)
    b = sum((t[i] - mt) * (x[i] - mx) for i in range(n)) / stt
    a = mx - b * mt
    return [x[i] - (a + b * t[i]) for i in range(n)]


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------- T0-1
files = sorted(glob(os.path.join(RES, "icrl_td", "*.jsonl")))
assert len(files) == 45, f"expected 45 icrl_td runs, got {len(files)}"
traces = {os.path.basename(f): load_trace(f) for f in files}
for k, v in traces.items():
    assert len(v) == 201, (k, len(v))
M = 201

chain = {}
for tau in (0.7, 0.8):
    qhats = []
    ladder_obs = {K: 0 for K in range(1, 6)}
    for name, tr in traces.items():
        bits = [v >= tau for v in tr]
        qhats.append(sum(bits) / M)
        r = max_runlen(bits)
        for K in range(1, 6):
            if r >= K:
                ladder_obs[K] += 1
    qbar = sum(qhats) / len(qhats)
    ladder_pred = {}
    for K in range(1, 6):
        approx, exact = mixture_pred(qhats, M, K)
        ladder_pred[K] = {"mixture_approx": round(approx, 2),
                          "mixture_exact_fmci": round(exact, 2)}
    # pooled nomogram reading: K >= (ln M - ln alpha)/ln(1/qbar)
    alpha = 0.05
    K_nomogram = (math.log(M) - math.log(alpha)) / math.log(1 / qbar) if qbar > 0 else None
    # mixture-prescribed K: smallest K with mixture-predicted per-run rate <= alpha
    K_mix = None
    for K in range(1, 12):
        approx, _ = mixture_pred(qhats, M, K)
        if approx / len(qhats) <= alpha:
            K_mix = K
            break
    chain[str(tau)] = {
        "pooled_qbar": round(qbar, 4),
        "ladder_observed_K1to5": ladder_obs,
        "ladder_predicted_K1to5": ladder_pred,
        "nomogram_pooled_K": round(K_nomogram, 2),
        "mixture_prescribed_K_alpha0.05": K_mix,
    }

# exact binomial q at the median chance baseline p=0.562, N=32 (for reference)
def binom_tail(N, p, kmin):
    from math import comb
    return sum(comb(N, k) * p ** k * (1 - p) ** (N - k) for k in range(kmin, N + 1))

chain["exact_binomial_q"] = {
    "q(0.7)_N32_p0.562": round(binom_tail(32, 0.562, math.ceil(0.7 * 32)), 5),
    "q(0.8)_N32_p0.562": round(binom_tail(32, 0.562, math.ceil(0.8 * 32)), 6),
}

# severity/speedup factors at tau=0.7
obs7 = chain["0.7"]["ladder_observed_K1to5"]
pred7 = chain["0.7"]["ladder_predicted_K1to5"]
chain["0.7"]["severity"] = {
    "observed_single_over_K5": round(obs7[1] / max(obs7[5], 1), 1),
    "mixture_single_over_K5": round(
        pred7[1]["mixture_approx"] / pred7[5]["mixture_approx"], 1),
}
OUT["T0_1_tau_chain"] = chain

# ---------------------------------------------------------------- T0-2c
lags = {}
for kind in ("centered", "detrended"):
    per_lag = {}
    for lag in range(1, 6):
        vals = []
        for tr in traces.values():
            x = tr if kind == "centered" else detrend(tr)
            vals.append(autocorr(x, lag))
        band = 1.96 / math.sqrt(M)
        per_lag[lag] = {
            "median": round(median(vals), 3),
            "n_inside_band": sum(1 for v in vals if abs(v) <= band),
            "n_total": len(vals),
            "band": round(band, 3),
            "max_abs": round(max(abs(v) for v in vals), 3),
        }
    lags[kind] = per_lag
OUT["T0_2c_autocorr_lags1to5"] = lags

# ---------------------------------------------------------------- T0-2b
g9 = sorted(glob(os.path.join(RES, "icrl_td_gamma_ladder", "gamma0p9_T40_s*.jsonl")))
assert len(g9) == 60, len(g9)
tau = 0.8
g9_traces, g9_seeds = [], []
for f in g9:
    tr = load_trace(f)
    assert len(tr) == 201, (f, len(tr))
    g9_traces.append(tr)
    g9_seeds.append(int(os.path.basename(f).split("_s")[1].split(".")[0]))
g9_q = [sum(v >= tau for v in tr) / M for tr in g9_traces]
g9_sus2 = [max_runlen([v >= tau for v in tr]) >= 2 for tr in g9_traces]
g9_sus3 = [max_runlen([v >= tau for v in tr]) >= 3 for tr in g9_traces]
approx_all, exact_all = mixture_pred(g9_q, M, 2)
OUT["T0_2b_gamma09"] = {
    "n_runs": 60,
    "pooled_qbar": round(sum(g9_q) / 60, 4),
    "sustained2_observed": sum(g9_sus2),
    "sustained2_mixture_pred": round(approx_all, 2),
    "sustained2_mixture_pred_exact": round(exact_all, 2),
    "sustained3_observed": sum(g9_sus3),
    "sustained3_mixture_pred": round(mixture_pred(g9_q, M, 3)[0], 2),
}

# reproduce split-half (seeds sorted; halves = first 30 / last 30 by seed id)
order = sorted(range(60), key=lambda i: g9_seeds[i])
half_a, half_b = order[:30], order[30:]
def fold_pred(train_idx, test_idx):
    pr = [1 - math.exp(-(M - 1) * g9_q[i] ** 2) for i in train_idx]
    return sum(pr) / len(pr) * len(test_idx)
OUT["T0_2b_gamma09"]["splithalf_repro"] = {
    "pred_on_B_from_A": round(fold_pred(half_a, half_b), 1),
    "obs_B": sum(g9_sus2[i] for i in half_b),
    "pred_on_A_from_B": round(fold_pred(half_b, half_a), 1),
    "obs_A": sum(g9_sus2[i] for i in half_a),
}

# k=5 folds, deterministic assignment: fold j = ranks {j, j+5, j+10, ...} of the
# seed-sorted order (interleaved, no cherry-picking)
folds = [[order[r] for r in range(j, 60, 5)] for j in range(5)]
fold_rows = []
for j, te in enumerate(folds):
    trn = [i for i in range(60) if i not in te]
    fold_rows.append({
        "fold": j,
        "predicted": round(fold_pred(trn, te), 2),
        "observed": sum(g9_sus2[i] for i in te),
    })
tot_pred = sum(r["predicted"] for r in fold_rows)
tot_obs = sum(r["observed"] for r in fold_rows)
maxdev = max(abs(r["observed"] - r["predicted"]) for r in fold_rows)
OUT["T0_2b_gamma09"]["kfold5"] = {
    "folds": fold_rows,
    "total_predicted": round(tot_pred, 1),
    "total_observed": tot_obs,
    "max_per_fold_abs_dev": round(maxdev, 2),
}
# per-fold binomial check: is each observed count within the central 95% of
# Poisson-binomial approx N(pred, pred*(1-pred/12))? use simple 2*sqrt(pred)
for r in fold_rows:
    sd = math.sqrt(max(r["predicted"] * (1 - r["predicted"] / 12), 1e-9))
    r["within_2sd"] = abs(r["observed"] - r["predicted"]) <= 2 * sd

# ---------------------------------------------------------------- T0-2a
# 171-cell pool: 26 rescue + 85 standard-budget ladder (gamma>0) + 60 budget grid
n_total = 171
ub_indep = 1 - 0.05 ** (1 / n_total)
# cluster (arm) structure: rescue main / ultra / wide; ladder rungs gamma
# 0.3 / 0.5 / 0.7 / 0.9; budget grid 4x / 8x
clusters = {"rescue_main": 14, "rescue_ultra": 9, "rescue_wide": 3,
            "ladder_g0.3": 5, "ladder_g0.5": 10, "ladder_g0.7": 10,
            "ladder_g0.9": 60, "budget_4x": 30, "budget_8x": 30}
assert sum(clusters.values()) == n_total
J = len(clusters)
ub_cluster = 1 - 0.05 ** (1 / J)

# Beta-Binomial profile: p_j ~ Beta(mu*phi, (1-mu)*phi); zero successes in all
# clusters. L(mu, phi) = prod_j B(mu*phi, (1-mu)*phi + n_j) / B(mu*phi, (1-mu)*phi)
from math import lgamma
def loglik(mu, phi, ns):
    a, b = mu * phi, (1 - mu) * phi
    ll = 0.0
    for n in ns:
        ll += (lgamma(a + b) - lgamma(b)) + (lgamma(b + n) - lgamma(a + b + n))
    return ll

ns = list(clusters.values())
crit = -0.5 * 5.411  # one-sided 95% profile-likelihood cutoff? use chi2_1 90% two
# sided = 2.706 -> one-sided 95%: 0.5*chi2_1(0.90) = 1.353
cut = 1.353
profile = {}
for phi in (0.01, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6):
    # find largest mu with max-ll(mu,phi) - ll >= -cut  (ll(mu->0) -> 0 = max)
    lo, hi = 1e-8, 0.999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if loglik(mid, phi, ns) >= -cut:
            lo = mid
        else:
            hi = mid
    profile[phi] = round(lo, 4)
OUT["T0_2a_cluster_bound"] = {
    "n_total": n_total, "n_clusters": J, "cluster_sizes": clusters,
    "ub95_independent": round(ub_indep, 4),
    "ub95_fully_clustered_(0_of_J)": round(ub_cluster, 4),
    "betabinom_profile_ub95_by_phi": profile,
    "note": ("phi (within-cluster homogeneity) is unidentified from all-zero "
             "data; profile UB runs from the independent endpoint (phi->inf) "
             "to the cluster-level endpoint (phi->0)."),
}

with open(os.path.join(HERE, "t0_analysis_output.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
print(json.dumps(OUT, indent=1))
