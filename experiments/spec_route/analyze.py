"""Direction 014 dedicated P1/P4 analysis pass.

P1 (seed-level signal increment): within high-variance config cells, does the
early-window trajectory of a run predict ITS OWN delay better than the cell
mean (the config-only baseline at seed granularity)? Operationalized as
leave-one-seed-out residual prediction pooled across cells.

P4 (predictability decomposition): per optimizer x task group, out-of-fold
variance shares of log-delay: config share (cell-mean LOO R^2), signal
increment (added R^2 of residual model), residual.
"""

import numpy as np

import predict


def _cell_groups(cells):
    groups = {}
    for i, c in enumerate(cells):
        groups.setdefault(c, []).append(i)
    return groups


def _loo_cell_means(y, cells):
    """Config-only seed-level prediction: mean of the OTHER seeds in the cell.
    Returns (pred, valid_mask); cells with <2 members are invalid."""
    y = np.asarray(y, dtype=float)
    pred = np.full(len(y), np.nan)
    for idx in _cell_groups(cells).values():
        if len(idx) < 2:
            continue
        s = y[idx].sum()
        for i in idx:
            pred[i] = (s - y[i]) / (len(idx) - 1)
    return pred, ~np.isnan(pred)


def p1_seed_increment(X, y, cells, min_seeds=4, n_perm=200, rng=None):
    """LOO-by-seed residual prediction within cells with >= min_seeds runs.

    Residual r_i = y_i - mean(y_cell minus i). Ridge maps features -> residual,
    evaluated out-of-fold (leave one run out). Returns spearman + permutation p
    (residuals shuffled WITHIN cells, preserving the cell structure) and the
    MAE comparison against the predict-zero (config-only) baseline.
    """
    rng = rng or np.random.default_rng(14)
    X, y = np.asarray(X, float), np.asarray(y, float)
    base_pred, valid = _loo_cell_means(y, cells)
    big = np.zeros(len(y), dtype=bool)
    for idx in _cell_groups(cells).values():
        if len(idx) >= min_seeds:
            big[idx] = True
    use = valid & big
    if use.sum() < 8:
        return {"n": int(use.sum()), "spearman": float("nan"),
                "perm_p": float("nan"), "mae_signal": float("nan"),
                "mae_config": float("nan")}
    resid = y - base_pred

    def loo_predict(r_target):
        out = np.full(len(y), np.nan)
        idx_use = np.where(use)[0]
        for i in idx_use:
            tr = idx_use[idx_use != i]
            out[i] = predict.ridge_fit_predict(X[tr], r_target[tr],
                                               X[i:i + 1])[0]
        return out[idx_use], r_target[idx_use]

    r_hat, r_true = loo_predict(resid)
    obs = predict.spearman(r_true, r_hat)
    mae_signal = float(np.abs(r_true - r_hat).mean())
    mae_config = float(np.abs(r_true).mean())  # config-only predicts resid 0

    null = []
    cg = {k: [i for i in v if use[i]] for k, v in _cell_groups(cells).items()}
    for _ in range(n_perm):
        rp = resid.copy()
        for idx in cg.values():
            if len(idx) >= 2:
                rp[idx] = rng.permutation(rp[idx])
        nh, nt = loo_predict(rp)
        s = predict.spearman(nt, nh)
        if not np.isnan(s):
            null.append(s)
    p = (1 + sum(s >= obs for s in null)) / (1 + len(null)) if null else float("nan")
    return {"n": int(use.sum()), "spearman": float(obs), "perm_p": float(p),
            "mae_signal": mae_signal, "mae_config": mae_config}


def p4_decomposition(X, y, cells, groups):
    """Per group (e.g. optimizer x task): out-of-fold variance shares.

    config_share   = R^2 of LOO cell-mean prediction
    signal_share   = R^2(config + residual ridge) - config_share
    residual_share = 1 - sum. Groups with <6 usable runs are skipped.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    out = {}
    for g in sorted(set(groups), key=str):
        gi = np.array([i for i in range(len(y)) if groups[i] == g])
        if len(gi) < 6:
            continue
        yg, Xg = y[gi], X[gi]
        cg = [cells[i] for i in gi]
        base, valid = _loo_cell_means(yg, cg)
        if valid.sum() < 6:
            continue
        yv, bv = yg[valid], base[valid]
        sst = ((yv - yv.mean()) ** 2).sum()
        if sst < 1e-12:
            # config-saturated group (e.g. muon mod-add zero variance):
            out[str(g)] = {"n": int(valid.sum()), "config_share": 1.0,
                           "signal_share": 0.0, "residual_share": 0.0,
                           "zero_variance": True}
            continue
        r2_cfg = 1.0 - ((yv - bv) ** 2).sum() / sst
        # residual model on top of config baseline (LOO)
        resid = yg - base
        iv = np.where(valid)[0]
        r_hat = np.full(len(yg), np.nan)
        for i in iv:
            tr = iv[iv != i]
            r_hat[i] = predict.ridge_fit_predict(Xg[tr], resid[tr],
                                                 Xg[i:i + 1])[0]
        comb = bv + r_hat[valid]
        r2_comb = 1.0 - ((yv - comb) ** 2).sum() / sst
        out[str(g)] = {"n": int(valid.sum()),
                       "config_share": float(max(0.0, r2_cfg)),
                       "signal_share": float(max(0.0, r2_comb - r2_cfg)),
                       "residual_share": float(max(0.0, 1.0 - max(r2_comb,
                                                                  r2_cfg))),
                       "zero_variance": False}
    return out
