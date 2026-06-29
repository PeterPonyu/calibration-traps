"""Direction 014 predictors, baselines, CV schemes, permutation tests.

Deliberately dependency-light: numpy-only ridge / L2-logistic on ≤~70-dim
features over ≤~350 runs. The science is in the comparisons, not the model:
headline quantity = Δ(signal − config-only) per CV scheme.
"""

import numpy as np


# ---------------------------------------------------------------- models

def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit_predict(Xtr, ytr, Xte, lam=1.0):
    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def logistic_fit_predict(Xtr, ytr, Xte, lam=1.0, iters=300, lr=0.5):
    Xtr, Xte = _standardize(Xtr, Xte)
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    w = np.zeros(Xtr.shape[1])
    n = len(ytr)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w, -30, 30)))
        g = Xtr.T @ (p - ytr) / n + lam * w / n
        w -= lr * g
    return 1.0 / (1.0 + np.exp(-np.clip(Xte @ w, -30, 30)))


# ---------------------------------------------------------------- metrics

def auroc(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based (handles ties)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    rpos = ranks[: len(pos)].sum()
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spearman(y, yhat):
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r
    ry, rh = rank(np.asarray(y)), rank(np.asarray(yhat))
    if ry.std() == 0 or rh.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rh)[0, 1])


# ---------------------------------------------------------------- CV splits

def cv_splits(groups, scheme, seeds=None):
    """Yield (train_idx, test_idx) index arrays.

    scheme="config": leave-one-config-cell-out (groups = cell keys)
    scheme="seed":   leave-one-seed-out (groups = cell keys, seeds = seed ids;
                     test = one seed across all cells, train = the rest)
    scheme="task":   groups = task family per run ("add"/"mul" vs "s5"):
                     train on non-s5, test on s5 (single split)
    """
    groups = list(groups)  # cell keys are tuples — keep them hashable, not ndarray
    idx = np.arange(len(groups))
    if scheme == "config":
        for g in sorted(set(groups), key=str):
            m = np.array([x == g for x in groups])
            if m.any() and (~m).any():
                yield idx[~m], idx[m]
    elif scheme == "seed":
        if seeds is None:
            raise ValueError("scheme='seed' requires seeds")
        seeds = list(seeds)
        for s in sorted(set(seeds), key=str):
            m = np.array([x == s for x in seeds])
            if m.any() and (~m).any():
                yield idx[~m], idx[m]
    elif scheme == "task":
        is_s5 = np.array([g == "s5" for g in groups])
        if is_s5.any() and (~is_s5).any():
            yield idx[~is_s5], idx[is_s5]
    else:
        raise ValueError(scheme)


# ---------------------------------------------------------------- evaluation

def evaluate(X, y, groups, scheme, task="clf", seeds=None, lam=1.0):
    """Pooled out-of-fold predictions → AUROC (clf) or spearman (reg)."""
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    for tr, te in cv_splits(groups, scheme, seeds):
        if task == "clf":
            if len(set(y[tr].tolist())) < 2:
                continue
            oof[te] = logistic_fit_predict(X[tr], y[tr], X[te], lam=lam)
        else:
            oof[te] = ridge_fit_predict(X[tr], y[tr], X[te], lam=lam)
    m = ~np.isnan(oof)
    if m.sum() < 4:
        return float("nan"), oof
    score = auroc(y[m], oof[m]) if task == "clf" else spearman(y[m], oof[m])
    return score, oof


def permutation_pvalue(X, y, groups, scheme, task="clf", seeds=None,
                       n_perm=200, rng=None, lam=1.0):
    """Permutation null: shuffle labels within the whole pool."""
    rng = rng or np.random.default_rng(0)
    obs, _ = evaluate(X, y, groups, scheme, task, seeds, lam)
    if np.isnan(obs):
        return obs, float("nan")
    null = []
    y = np.asarray(y, dtype=float)
    for _ in range(n_perm):
        yp = rng.permutation(y)
        s, _ = evaluate(X, yp, groups, scheme, task, seeds, lam)
        if not np.isnan(s):
            null.append(s)
    if not null:
        return obs, float("nan")
    null = np.asarray(null)
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    return obs, float(p)


def signal_increment(X_sig, X_cfg, y, groups, scheme, task="clf", seeds=None):
    """Headline quantity: Δ = score(signal) − score(config-only)."""
    s_sig, _ = evaluate(X_sig, y, groups, scheme, task, seeds)
    s_cfg, _ = evaluate(X_cfg, y, groups, scheme, task, seeds)
    return {"signal": s_sig, "config_only": s_cfg,
            "delta": (s_sig - s_cfg) if not (np.isnan(s_sig) or
                                             np.isnan(s_cfg)) else float("nan")}
