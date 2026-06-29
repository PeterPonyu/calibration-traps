"""Direction 016 — random-MRP trajectory streams for in-context TD emergence.

Pure simulation, zero download: every batch is a fresh set of randomly
generated Markov Reward Processes (Boyan-chain-style task distribution per
2405.13861, local implementation). Sequence layout:

    [s_0, r~_0, s_1, r~_1, ..., s_{T-1}, r~_{T-1}, QUERY, s_q]  ->  V-bucket(s_q)

Token layout (vocab = S + R + 1 + K):
    states        [0, S)
    reward tokens [S, S + R)            (discretized state rewards)
    QUERY         S + R
    value buckets [S + R + 1, S + R + 1 + K)

Ground truth uses the DISCRETIZED rewards (the model sees only reward tokens,
so V* is computed from the same information): V = (I - gamma P)^{-1} r_disc.
"""

import numpy as np

GAMMA = 0.9
N_STATES = 10
R_BUCKETS = 7    # reward tokens over [-1, 1]
V_BUCKETS = 9    # value tokens over [-Vmax, Vmax], Vmax = 1/(1-gamma)
V_MAX = 1.0 / (1.0 - GAMMA)


def vocab_size(n_states=N_STATES):
    return n_states + R_BUCKETS + 1 + V_BUCKETS


def query_token(n_states=N_STATES):
    return n_states + R_BUCKETS


def reward_token(bucket, n_states=N_STATES):
    return n_states + bucket


def value_token(bucket, n_states=N_STATES):
    return n_states + R_BUCKETS + 1 + bucket


def reward_bucket(r):
    """Discretize r in [-1,1] to a bucket index; returns (bucket, midpoint)."""
    edges = np.linspace(-1.0, 1.0, R_BUCKETS + 1)
    b = int(np.clip(np.searchsorted(edges, r, side="right") - 1,
                    0, R_BUCKETS - 1))
    mid = (edges[b] + edges[b + 1]) / 2.0
    return b, mid


def value_bucket(v):
    edges = np.linspace(-V_MAX, V_MAX, V_BUCKETS + 1)
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1,
                       0, V_BUCKETS - 1))


def value_midpoint(bucket):
    edges = np.linspace(-V_MAX, V_MAX, V_BUCKETS + 1)
    return (edges[bucket] + edges[bucket + 1]) / 2.0


def make_mrp(rng, n_states=N_STATES, concentration=0.3):
    """Random MRP: Dirichlet rows (low concentration -> structured chains),
    per-state rewards in [-1,1] (then discretized), exact value vector."""
    P = rng.dirichlet(np.full(n_states, concentration), size=n_states)
    r_raw = rng.uniform(-1.0, 1.0, size=n_states)
    r_disc = np.empty(n_states)
    r_tok = np.empty(n_states, dtype=np.int64)
    for s in range(n_states):
        b, mid = reward_bucket(r_raw[s])
        r_tok[s] = reward_token(b, n_states)
        r_disc[s] = mid
    V = np.linalg.solve(np.eye(n_states) - GAMMA * P, r_disc)
    return {"P": P, "r_disc": r_disc, "r_tok": r_tok, "V": V,
            "n_states": n_states}


def sample_sequence(rng, mrp, T, query_state=None):
    """One trajectory + query. Returns (tokens [2T+2], target_bucket, meta)."""
    n = mrp["n_states"]
    states = np.empty(T, dtype=np.int64)
    s = rng.integers(n)
    for t in range(T):
        states[t] = s
        s = rng.choice(n, p=mrp["P"][s])
    tokens = np.empty(2 * T + 2, dtype=np.int64)
    tokens[0:2 * T:2] = states
    tokens[1:2 * T:2] = mrp["r_tok"][states]
    s_q = int(rng.integers(n)) if query_state is None else int(query_state)
    tokens[2 * T] = query_token(n)
    tokens[2 * T + 1] = s_q
    target = value_bucket(mrp["V"][s_q])
    meta = {"mrp": mrp, "states": states, "s_q": s_q,
            "state_positions": np.arange(0, 2 * T, 2)}
    return tokens, target, meta


def make_batch(rng, batch_size, T, n_states=N_STATES, with_meta=False,
               concentration=0.3):
    """Fresh-MRP batch: one new MRP per sequence (the multi-task stream).
    concentration: Dirichlet rows; 0.3 = training distribution, other values
    give the OOD-structure eval stream (P4)."""
    toks = np.empty((batch_size, 2 * T + 2), dtype=np.int64)
    tgts = np.empty(batch_size, dtype=np.int64)
    metas = []
    for i in range(batch_size):
        mrp = make_mrp(rng, n_states, concentration)
        toks[i], tgts[i], meta = sample_sequence(rng, mrp, T)
        if with_meta:
            metas.append(meta)
    # targets are value-bucket TOKEN ids (loss restricted to bucket slice)
    tgt_tokens = np.array([value_token(b, n_states) for b in tgts],
                          dtype=np.int64)
    if with_meta:
        return toks, tgt_tokens, metas
    return toks, tgt_tokens


def permuted_reward_variant(rng, meta, T):
    """P4 control: same state visitation sequence, permuted state->reward map.
    Ground-truth V changes; an in-context algorithm must track the NEW values,
    a label-imitating model will not."""
    mrp = meta["mrp"]
    n = mrp["n_states"]
    perm = rng.permutation(n)
    mrp2 = dict(mrp)
    mrp2["r_disc"] = mrp["r_disc"][perm]
    mrp2["r_tok"] = mrp["r_tok"][perm]
    mrp2["V"] = np.linalg.solve(np.eye(n) - GAMMA * mrp["P"], mrp2["r_disc"])
    states = meta["states"]
    tokens = np.empty(2 * T + 2, dtype=np.int64)
    tokens[0:2 * T:2] = states
    tokens[1:2 * T:2] = mrp2["r_tok"][states]
    tokens[2 * T] = query_token(n)
    tokens[2 * T + 1] = meta["s_q"]
    target = value_bucket(mrp2["V"][meta["s_q"]])
    return tokens, value_token(target, n), {**meta, "mrp": mrp2}
