"""Direction 016 probes — TD-alignment, Bellman residual, kernel-attention.

Kill-criterion discipline (direction doc): every probe is computed on
held-out MRPs against GROUND-TRUTH quantities (true P / true TD(0) trace),
never against the training loss — the probes cannot be tautological with CE
on value buckets. `self_test()` plants a TD oracle and a constant predictor
to verify the instruments separate algorithm from non-algorithm.
"""

import numpy as np

from data import GAMMA


def td_trace(states, rewards, s_q, n_states, alpha=0.2, gamma=GAMMA):
    """Run tabular TD(0) over the trajectory; return V_td[s_q] after each
    transition (length T-1 list) — the reference an in-context TD
    implementation should track."""
    V = np.zeros(n_states)
    out = []
    for t in range(len(states) - 1):
        s, s2 = states[t], states[t + 1]
        V[s] += alpha * (rewards[t] + gamma * V[s2] - V[s])
        out.append(V[s_q])
    return np.array(out)


def modal_state(states):
    """Most-visited state of a trajectory — the probe-side query choice.
    TD(0) only updates V[s] when s is visited as a transition source, so the
    TD-alignment comparison is only informative at a well-visited state. The
    TASK query stays random; this choice concerns the instrument only."""
    vals, counts = np.unique(np.asarray(states)[:-1], return_counts=True)
    return int(vals[counts.argmax()])


def td_alignment(estimator, meta, ks, alpha=0.2, probe_state=None):
    """Pearson corr between the model's value estimates across prefix lengths
    and the TD(0) trace at the same prefixes.

    estimator(k, probe_state) -> scalar V-hat for the probe state given the
    first k transitions (the trainer wraps the model; self_test plants
    oracles). probe_state defaults to the trajectory's modal state.
    """
    mrp = meta["mrp"]
    states = meta["states"]
    rewards = mrp["r_disc"][states]
    if probe_state is None:
        probe_state = modal_state(states)
    full = td_trace(states, rewards, probe_state, mrp["n_states"], alpha)
    xs, ys = [], []
    for k in ks:
        if k - 1 < 1 or k - 1 > len(full):
            continue
        xs.append(full[k - 2])          # TD estimate after k-1 transitions
        ys.append(estimator(k, probe_state))
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    if len(xs) < 3 or xs.std() < 1e-12 or ys.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def bellman_residual(V_hat, P, r, gamma=GAMMA):
    """sup-norm Bellman residual of a full value-vector estimate against the
    ground-truth kernel: || V_hat - (r + gamma P V_hat) ||_inf."""
    V_hat = np.asarray(V_hat, dtype=float)
    return float(np.max(np.abs(V_hat - (r + gamma * P @ V_hat))))


def kernel_attention_score(attn_query_row, state_positions, states, P_row):
    """Corr between the query position's PER-OCCURRENCE attention on each
    state's tokens and the true transition-kernel row of the query state.

    Per-occurrence (mass / visit count) — raw aggregated mass is confounded
    by visitation frequency (mass[s] ~ visits(s) * attention-per-visit), which
    dilutes the kernel correlation. Unvisited states carry no evidence and are
    excluded. attn_query_row: [T_seq] weights from the final position,
    averaged over heads/layers by the caller.
    """
    n = len(P_row)
    mass, count = np.zeros(n), np.zeros(n)
    for pos, s in zip(state_positions, states):
        mass[s] += float(attn_query_row[pos])
        count[s] += 1
    visited = count > 0
    if visited.sum() < 3:
        return 0.0
    per_occ = mass[visited] / count[visited]
    pr = np.asarray(P_row, dtype=float)[visited]
    if per_occ.std() < 1e-12 or pr.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(per_occ, pr)[0, 1])


def emergence_step(steps, accs, thresh):
    """First eval step where acc >= thresh (sustained: also >= at next eval
    if one exists) — mirrors the repo's emergence detector discipline."""
    for i, (st, a) in enumerate(zip(steps, accs)):
        if a >= thresh and (i + 1 >= len(accs) or accs[i + 1] >= thresh):
            return st
    return None


# ---------------------------------------------------------------- self-test

def self_test(seed=16):
    rng = np.random.default_rng(seed)
    import data as D

    # data validity: stochastic rows, Bellman-consistent V, bucket round-trip
    mrp = D.make_mrp(rng)
    assert np.allclose(mrp["P"].sum(1), 1.0, atol=1e-9)
    assert bellman_residual(mrp["V"], mrp["P"], mrp["r_disc"]) < 1e-9
    b = D.value_bucket(mrp["V"][0])
    assert abs(D.value_midpoint(b) - mrp["V"][0]) <= D.V_MAX / D.V_BUCKETS + 1e-9

    # TD-alignment separates a TD oracle from a constant predictor.
    # Probe query = modal state (well-visited; unvisited states give a
    # degenerate constant TD trace — exactly why modal_state exists).
    toks, tgt, meta = D.sample_sequence(rng, mrp, T=40)
    states, rewards = meta["states"], mrp["r_disc"][meta["states"]]
    sp = modal_state(states)
    assert (np.asarray(states)[:-1] == sp).sum() >= 3, \
        "modal state too rare for a meaningful self-test"
    full = td_trace(states, rewards, sp, mrp["n_states"])
    ks = list(range(4, 40, 4))
    oracle = lambda k, s: full[k - 2]                    # noqa: E731
    noisy_oracle = lambda k, s: full[k - 2] + rng.normal(0, 1e-3)  # noqa: E731
    const = lambda k, s: 0.37                            # noqa: E731
    a_or = td_alignment(noisy_oracle, meta, ks)
    a_ex = td_alignment(oracle, meta, ks)
    a_c = td_alignment(const, meta, ks)
    assert a_ex > 0.999 and a_or > 0.95, (a_ex, a_or)
    assert abs(a_c) < 0.5, a_c

    # Bellman residual: truth ~0, perturbed >> 0
    assert bellman_residual(mrp["V"] + 1.0, mrp["P"], mrp["r_disc"]) > 0.05

    # kernel-attention probe: planted attention on the true next-state
    # occurrences beats random attention
    T = 40
    s_q = meta["s_q"]
    P_row = mrp["P"][s_q]
    planted = np.zeros(2 * T + 2)
    for pos, s in zip(meta["state_positions"], states):
        planted[pos] = P_row[s]
    planted /= planted.sum() + 1e-12
    rnd = rng.random(2 * T + 2)
    rnd /= rnd.sum()
    k_pl = kernel_attention_score(planted, meta["state_positions"], states, P_row)
    k_rd = kernel_attention_score(rnd, meta["state_positions"], states, P_row)
    assert k_pl > 0.9, k_pl
    assert k_pl > k_rd + 0.2, (k_pl, k_rd)

    # emergence detector on a planted curve
    st = emergence_step([0, 100, 200, 300, 400],
                        [0.1, 0.2, 0.75, 0.9, 0.95], 0.7)
    assert st == 200, st

    # P4 instrument: permuted rewards change the truth
    toks2, tgt2, meta2 = D.permuted_reward_variant(rng, meta, T=40)
    assert not np.allclose(meta2["mrp"]["V"], mrp["V"])

    return True


if __name__ == "__main__":
    self_test()
    print("probes self-test PASS")
