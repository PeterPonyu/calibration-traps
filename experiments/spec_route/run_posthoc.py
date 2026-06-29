"""Direction 014 driver.

  python run_posthoc.py --smoke     # end-to-end self-test, <60 s, writes nothing
  python run_posthoc.py --dry-run   # read-only corpus inventory
  python run_posthoc.py             # formal analysis (executor session only)

The formal analysis writes to results/spec_route/ — never run it from the
design session; ROUND-1 verdicts belong to the executor per repo discipline.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
import corpus  # noqa: E402
import features  # noqa: E402
import labels  # noqa: E402
import predict  # noqa: E402

OUT_DIR = os.path.join(corpus.RESULTS_DIR, "spec_route")


# ---------------------------------------------------------------- smoke

def _synthetic_record(rng, slope, idx):
    """Fabricate a run record with a planted pre-mem wn_hidden slope signal.

    slope is continuous; route class = its sign, delay = linear in it — so the
    same planted quantity validates both the classifier and the regressor.
    """
    route_growth = slope > 0
    steps = []
    wn = 12.0 + rng.normal(0, 0.3)
    for t in range(0, 401, 50):
        wn = wn * (1 + slope / 8) + rng.normal(0, 0.02)
        steps.append({"step": t, "train_loss": 4.6 * np.exp(-t / 300),
                      "test_loss": 4.6, "train_acc": min(1.0, t / 400),
                      "test_acc": 0.01, "wn_total": wn * 2.2,
                      "wn_hidden": wn, "wn_embed": 3.2})
    # post-window tail up to grok
    for t in range(450, 2001, 50):
        steps.append({"step": t, "train_loss": 0.01, "test_loss": 1.0,
                      "train_acc": 1.0, "test_acc": 0.01,
                      "wn_total": wn * 2.2,
                      "wn_hidden": wn * (3.0 if route_growth else 0.5),
                      "wn_embed": 3.2})
    return {"namespace": "synthetic", "tier": "A", "path": f"<syn{idx}>",
            "seed": idx % 5,
            "config": {"optimizer": "adamw", "lr": 1e-3,
                       "weight_decay": 0.01 * (idx % 3), "init_scale": 1.0,
                       "train_frac": 0.4, "op": "add", "p": 97,
                       "muon_lr": 0.02},
            "eval_every": 50, "memorize_step": 400, "grok_step": 2000,
            "delay_ratio": 5.0 + 30.0 * slope + rng.normal(0, 0.05),
            "final_test_acc": 0.99, "final_train_acc": 1.0,
            "stopped_step": 2000, "steps": steps,
            "channels": list(corpus.NORM_CHANNELS)}


def smoke():
    rng = np.random.default_rng(14)
    planted = [(0.02 + 0.04 * rng.random()) * (1 if i % 2 == 0 else -1)
               for i in range(60)]
    recs = [_synthetic_record(rng, planted[i], i) for i in range(60)]

    # 1. pipeline recovers a planted route signal (trajectory-only, CV-config)
    X, y, groups, seeds = [], [], [], []
    for r in recs:
        vec, names = features.extract_features(r, "W-mem")
        assert vec is not None and len(vec) == len(names)
        lab = labels.make_labels(r)
        assert lab["route"] in ("growth", "contraction")
        X.append(vec)
        y.append(1.0 if lab["route"] == "growth" else 0.0)
        groups.append(corpus.config_cell_key(r) + (r["seed"] % 6,))
        seeds.append(r["seed"])
    X = np.vstack(X)
    score, _ = predict.evaluate(X, y, groups, "config", task="clf")
    assert score > 0.9, f"planted-signal AUROC too low: {score}"

    # 2. label-shuffle control sits near chance
    yp = rng.permutation(np.array(y))
    s_null, _ = predict.evaluate(X, yp, groups, "config", task="clf")
    assert 0.2 < s_null < 0.8, f"shuffled-label AUROC suspicious: {s_null}"

    # 3. regression recovers planted delay relation
    yd = np.array([r["delay_ratio"] for r in recs])
    s_reg, _ = predict.evaluate(X, yd, groups, "config", task="reg")
    assert s_reg > 0.9, f"planted regression spearman too low: {s_reg}"

    # 4. leakage guard: poison a post-window record → features unchanged
    r0 = recs[0]
    base, _ = features.extract_features(r0, "W-mem")
    poisoned = dict(r0)
    poisoned["steps"] = [dict(s) for s in r0["steps"]]
    for s in poisoned["steps"]:
        if s["step"] > r0["memorize_step"]:
            s["wn_hidden"] = 1e9
            s["train_loss"] = -1e9
    after, _ = features.extract_features(poisoned, "W-mem")
    assert np.array_equal(base, after), "LEAKAGE: post-window record changed features"

    # 4b. P1/P4 analysis pass recovers the planted seed-level signal
    cells_cfg = [corpus.config_cell_key(r) for r in recs]
    yd = np.array([r["delay_ratio"] for r in recs])
    p1 = analyze.p1_seed_increment(X, yd, cells_cfg, min_seeds=4, n_perm=50)
    assert p1["spearman"] > 0.5, f"P1 planted signal missed: {p1}"
    assert p1["perm_p"] < 0.05, f"P1 permutation null too strong: {p1}"
    assert p1["mae_signal"] < p1["mae_config"], "P1 no MAE gain over config"
    grp = [r["config"]["optimizer"] for r in recs]
    p4 = analyze.p4_decomposition(X, yd, cells_cfg, grp)
    assert "adamw" in p4 and p4["adamw"]["signal_share"] > 0.2, \
        f"P4 decomposition missed planted signal share: {p4}"

    # 5. forbidden-namespace guard
    try:
        corpus.load_namespace("grok_numerics")
        raise AssertionError("forbidden namespace was not rejected")
    except ValueError:
        pass

    # 6. real-corpus schema check (skipped gracefully if results/ absent)
    real_dir = os.path.join(corpus.RESULTS_DIR, "grid_main")
    if os.path.isdir(real_dir):
        fns = [f for f in sorted(os.listdir(real_dir)) if f.endswith(".jsonl")]
        if fns:
            run = corpus.parse_run(os.path.join(real_dir, fns[0]))
            assert run is not None, "failed to parse a real grid_main run"
            rec = corpus.run_to_record(run, "grid_main", "A")
            assert "wn_hidden" in rec["channels"]
            _ = labels.make_labels(rec)
            print(f"  real-corpus schema check OK ({fns[0]})")

    print("SMOKE PASS: pipeline, controls, leakage guard, namespace guard all OK")


# ---------------------------------------------------------------- dry-run

def dry_run():
    recs, inventory = corpus.load_corpus()
    print(f"{'namespace':18} {'loaded':>6} {'skipped':>7}")
    for ns, inv in inventory.items():
        print(f"{ns:18} {inv['loaded']:>6} {inv['skipped']:>7}")
    print(f"{'TOTAL':18} {len(recs):>6}")
    print("\nlabel coverage:")
    cov = labels.label_coverage(recs)
    hdr = ["runs", "grokked", "failed", "growth", "contraction", "no_memorize"]
    print(f"{'namespace':18} " + " ".join(f"{h:>11}" for h in hdr))
    for ns in inventory:
        c = cov.get(ns)
        if c:
            print(f"{ns:18} " + " ".join(f"{c[h]:>11}" for h in hdr))
    # window availability
    for w in features.WINDOWS:
        n = sum(features.window_records(r, w) is not None for r in recs)
        print(f"window {w:7}: defined for {n}/{len(recs)} runs")


# ---------------------------------------------------------------- formal

def formal(n_perm=200):
    """ROUND-1 analysis per the pre-registered P1–P4 grid (executor session)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    recs, inventory = corpus.load_corpus()
    out = {"inventory": inventory, "coverage": labels.label_coverage(recs)}

    # assemble the standard design matrices once per window
    for window in ("W-mem", "W-200"):
        rows = []
        for r in recs:
            vec = features.extract_features(r, window)[0]
            if vec is None:
                continue
            lab = labels.make_labels(r)
            rows.append((r, vec, lab))
        if len(rows) < 20:
            continue
        # pad Tier-A vectors to a common dimensionality per tier: analyze tiers
        # separately to avoid mixed-dim hacks
        for tier in ("A", "B"):
            sub = [(r, v, l) for r, v, l in rows if r["tier"] == tier]
            if len(sub) < 20:
                continue
            X = np.vstack([v for _, v, _ in sub])
            cells = [corpus.config_cell_key(r) for r, _, _ in sub]
            seeds = [r["seed"] for r, _, _ in sub]
            tasks = [r["config"]["op"] for r, _, _ in sub]
            vocab = features.build_config_vocab([r for r, _, _ in sub])
            Xc = np.vstack([features.config_features(r, vocab)
                            for r, _, _ in sub])
            key = f"{window}/tier{tier}"
            res = {}
            # y1 grok classification, three CV schemes
            y1 = np.array([l["grokked"] for _, _, l in sub], dtype=float)
            for scheme, grp in (("config", cells), ("seed", cells),
                                ("task", tasks)):
                inc = predict.signal_increment(
                    X, Xc, y1, grp, scheme, task="clf",
                    seeds=seeds if scheme == "seed" else None)
                obs, p = predict.permutation_pvalue(
                    X, y1, grp, scheme, task="clf",
                    seeds=seeds if scheme == "seed" else None, n_perm=n_perm)
                res[f"grok/{scheme}"] = {**inc, "perm_p": p}
            # y3 route classification (grokked runs only, trajectory-only)
            sub3 = [(i, l) for i, (_, _, l) in enumerate(sub)
                    if l["route"] is not None]
            if len(sub3) >= 20:
                idx = [i for i, _ in sub3]
                y3 = np.array([1.0 if l["route"] == "growth" else 0.0
                               for _, l in sub3])
                for scheme in ("config", "task"):
                    grp = ([cells[i] for i in idx] if scheme == "config"
                           else [tasks[i] for i in idx])
                    obs, p = predict.permutation_pvalue(
                        X[idx], y3, grp, scheme, task="clf", n_perm=n_perm)
                    res[f"route/{scheme}"] = {"auroc": obs, "perm_p": p}
                # P2 preregistered variant: WITHIN-AdamW only. The pooled task
                # above lets the classifier read optimizer identity off the
                # trajectory (growth ~ muon); P2's claim is route readability
                # with the optimizer axis frozen.
                idx_a = [i for i, _ in sub3
                         if sub[i][0]["config"]["optimizer"] == "adamw"]
                if len(idx_a) >= 20:
                    y3a = np.array([1.0 if l["route"] == "growth" else 0.0
                                    for i, l in sub3
                                    if sub[i][0]["config"]["optimizer"] == "adamw"])
                    grp_a = [cells[i] for i in idx_a]
                    obs, p = predict.permutation_pvalue(
                        X[idx_a], y3a, grp_a, "config", task="clf",
                        n_perm=n_perm)
                    res["route/config_adamw"] = {
                        "auroc": obs, "perm_p": p, "n": len(idx_a)}
            # P1/P4 dedicated pass (grokked runs with a delay label)
            sub_d = [(r, v, l) for r, v, l in sub
                     if l["grokked"] and l["log_delay"] is not None]
            if len(sub_d) >= 12:
                Xd = np.vstack([v for _, v, _ in sub_d])
                yd = np.array([l["log_delay"] for _, _, l in sub_d])
                cd = [corpus.config_cell_key(r) for r, _, _ in sub_d]
                gd = [(r["config"]["optimizer"], r["config"]["op"])
                      for r, _, _ in sub_d]
                res["p1_seed_increment"] = analyze.p1_seed_increment(
                    Xd, yd, cd, n_perm=n_perm)
                res["p4_decomposition"] = analyze.p4_decomposition(
                    Xd, yd, cd, gd)
            out[key] = res
    with open(os.path.join(OUT_DIR, "posthoc_round1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {os.path.join(OUT_DIR, 'posthoc_round1.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.dry_run:
        dry_run()
    else:
        formal(n_perm=args.n_perm)


if __name__ == "__main__":
    main()
