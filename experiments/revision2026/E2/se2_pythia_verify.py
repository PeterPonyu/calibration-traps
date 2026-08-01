#!/usr/bin/env python3
"""Independent verification of the S-E2 Pythia detector-calibration battery.

Recomputes, FROM THE RAW per-checkpoint jsonl curves only (results_v1_m12/),
every quantity needed for the paper integration, then diffs against the
(untrusted) killed-agent analysis se2_analysis_v1_m12.json.

Conventions (paper's, per MANIFEST_v1_m12.json + main.tex Box 1 / eq:nomogram):
  tau = chance + 0.10 (the BIG-Bench audit DELTA_PRINCIPAL)
  emerged_by_end: last two checkpoints >= tau
  terminal block: maximal suffix of consecutive >= tau checkpoints
  pre-emergence segment: checkpoints before the terminal block (whole curve if
    not emerged); M_pre = its length; qhat_pre = crossing rate on it
  K_nomogram = ceil((ln M_pre - ln alpha)/ln(1/qhat_pre)), alpha=0.05; qhat=0 -> K=1
  single-crossing fire (pre): any pre-emergence checkpoint >= tau
  sustained-K fire (pre): any run of >= K consecutive crossings inside pre segment
  emerged curves: caught iff terminal block >= K; detection idx = onset+K-1;
    added latency = detect_idx - (onset + K - 1); censored if block < K (block
    abuts trace end by construction)
  binomial null (MC tasks only): q_null = P(Binom(N, chance) >= ceil(tau*N));
    predicted single fires = sum over chance-level MC curves of 1-(1-q_null)^M_pre
  mixture-predicted sustained-K false fires = sum over chance-level curves of
    1 - exp(-(M_pre - K + 1) * qhat_pre^K)  [paper's per-run mixture form]

gen/lastword tasks (chance ~ 0) are outside the binomial-null calibration and
serve as emergence-preservation probes; wiki_bpb is a continuous control.
"""
import json, math, os, sys

EPS = 1e-9
ALPHA = 0.05
BASE = os.path.dirname(os.path.abspath(__file__))
SE2 = os.path.join(BASE, "..", "se2-pythia")
RES = os.path.join(SE2, "results_v1_m12")
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m"]


def binom_sf_ge(n, p, k):
    """P(X >= k), X ~ Binom(n,p), exact."""
    if k <= 0:
        return 1.0
    total = 0.0
    logp, log1p = math.log(p), math.log(1 - p)
    for x in range(k, n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(x + 1)
                          - math.lgamma(n - x + 1) + x * logp + (n - x) * log1p)
    return min(total, 1.0)


def analyze_curve(task, meta, steps, accs):
    tau = meta["chance"] + 0.10
    M = len(accs)
    cross = [a >= tau - EPS for a in accs]
    emerged = M >= 2 and cross[-1] and cross[-2]
    # terminal block = maximal suffix of consecutive crossings
    tb = 0
    for c in reversed(cross):
        if c:
            tb += 1
        else:
            break
    if emerged:
        onset = M - tb
        pre = cross[:onset]
    else:
        onset = None
        pre = cross  # whole curve is "pre" (flag trailing-crossing oddity)
    M_pre = len(pre)
    n_cross_pre = sum(pre)
    qhat = n_cross_pre / M_pre if M_pre > 0 else 0.0
    if qhat <= 0:
        K = 1
    else:
        K = max(1, math.ceil((math.log(M_pre) - math.log(ALPHA))
                             / math.log(1.0 / qhat) - 1e-12))
    single_fire_pre = any(pre)
    # sustained-K inside pre segment
    run = best_run = 0
    for c in pre:
        run = run + 1 if c else 0
        best_run = max(best_run, run)
    sustained_fire_pre = best_run >= K
    out = dict(task=task, type=meta["type"], N=meta["N"], chance=meta["chance"],
               tau=round(tau, 4), M=M, M_pre=M_pre,
               qhat_pre=round(qhat, 4), K_nomogram=K,
               emerged_by_end=emerged, terminal_block_len=tb,
               single_crossing_fire_pre=single_fire_pre,
               sustainedK_fire_pre=sustained_fire_pre,
               max_pre_run=best_run)
    if emerged:
        caught = tb >= K
        out.update(onset_idx=onset,
                   sustainedK_fires_overall=caught,
                   sustainedK_censored=not caught,
                   sustainedK_detect_idx=(onset + K - 1) if caught else None,
                   detect_latency_ckpts=0 if caught else None)
    if not emerged and cross and cross[-1]:
        out["FLAG_trailing_single_crossing"] = True
    # mixture per-curve predicted sustained-K false-fire prob (chance-level only)
    if not emerged:
        out["p_sustainedK_mixture"] = 1 - math.exp(-(M_pre - K + 1) * qhat ** K) \
            if qhat > 0 else 0.0
    if meta["type"] == "mc":
        k_needed = math.ceil(meta["N"] * tau - EPS)
        qn = binom_sf_ge(meta["N"], meta["chance"], k_needed)
        out["q_null_binomial"] = qn
        out["p_single_fire_null_Mpre"] = 1 - (1 - qn) ** M_pre
    return out


def main():
    with open(os.path.join(SE2, "MANIFEST_v1_m12.json")) as f:
        manifest = json.load(f)
    task_meta = {t["task"]: t for t in manifest["tasks"]}

    curves, controls = [], []
    for model in MODELS:
        rows = []
        with open(os.path.join(RES, f"{model}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("_meta"):
                    continue
                rows.append(r)
        rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        for task, meta in task_meta.items():
            accs = [r["tasks"][task]["acc"] for r in rows]
            ns = {r["tasks"][task]["n"] for r in rows}
            assert ns == {meta["N"]}, (model, task, ns)
            c = analyze_curve(task, meta, steps, accs)
            c["model"] = model
            c["steps"] = steps
            c["accs"] = [round(a, 4) for a in accs]
            curves.append(c)
        controls.append(dict(model=model, task="wiki_bpb",
                             bpb=[round(r["tasks"]["wiki_bpb"]["bpb"], 4) for r in rows]))

    # summary
    chance_level = [c for c in curves if not c["emerged_by_end"]]
    emerged = [c for c in curves if c["emerged_by_end"]]
    mc_chance = [c for c in chance_level if c["type"] == "mc"]
    summary = {
        "n_curves_total": len(curves),
        "n_chance_level": len(chance_level),
        "n_emerged_by_end": len(emerged),
        "chance_level": {
            "single_crossing_false_fires": sum(c["single_crossing_fire_pre"] for c in chance_level),
            "sustainedK_false_fires": sum(c["sustainedK_fire_pre"] for c in chance_level),
            "sustainedK_fires_mixture_pred": round(sum(c["p_sustainedK_mixture"] for c in chance_level), 3),
            "qhat_pre_values_nonzero": sorted(c["qhat_pre"] for c in chance_level if c["qhat_pre"] > 0),
            "qhat_pre_range": [min(c["qhat_pre"] for c in chance_level),
                               max(c["qhat_pre"] for c in chance_level)],
            "qhat_pre_median": sorted(c["qhat_pre"] for c in chance_level)[len(chance_level) // 2],
            "mc_only": {
                "n": len(mc_chance),
                "single_fires_observed": sum(c["single_crossing_fire_pre"] for c in mc_chance),
                "single_fires_predicted_binomial_null": round(
                    sum(c["p_single_fire_null_Mpre"] for c in mc_chance), 2),
                "sustainedK_fires_observed": sum(c["sustainedK_fire_pre"] for c in mc_chance),
            },
        },
        "emerged": [
            {k: c[k] for k in ("model", "task", "onset_idx", "K_nomogram",
                               "sustainedK_fires_overall", "sustainedK_censored",
                               "detect_latency_ckpts", "terminal_block_len")}
            for c in sorted(emerged, key=lambda c: (c["model"], c["task"]))
        ],
        "per_model_table": {},
        "single_fire_curves": [
            {"model": c["model"], "task": c["task"], "qhat_pre": c["qhat_pre"],
             "K_nomogram": c["K_nomogram"], "max_pre_run": c["max_pre_run"]}
            for c in chance_level if c["single_crossing_fire_pre"]],
    }
    for model in MODELS:
        mc = [c for c in curves if c["model"] == model]
        cl = [c for c in mc if not c["emerged_by_end"]]
        summary["per_model_table"][model] = {
            "curves": len(mc),
            "chance_level": len(cl),
            "emerged": len(mc) - len(cl),
            "single_false_fires": sum(c["single_crossing_fire_pre"] for c in cl),
            "sustainedK_false_fires": sum(c["sustainedK_fire_pre"] for c in cl),
        }

    out = {"summary": summary, "curves": curves, "wiki_bpb_controls": controls}
    outpath = os.path.join(BASE, "se2_pythia_verify_output.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=1, default=str)

    # ---- diff against the killed agent's analysis ----
    mismatches = []
    with open(os.path.join(SE2, "se2_analysis_v1_m12.json")) as f:
        ka = json.load(f)
    ka_curves = {(c["model"], c["task"]): c for c in ka["curves"] if c.get("type") != "control"}
    for c in curves:
        k = (c["model"], c["task"])
        if k not in ka_curves:
            mismatches.append(f"missing in killed-agent json: {k}")
            continue
        o = ka_curves[k]
        # raw accs
        if [round(a, 4) for a in o["accs"]] != c["accs"]:
            mismatches.append(f"{k}: accs differ")
        for field in ("M_pre", "qhat_pre", "K_nomogram", "emerged_by_end",
                      "single_crossing_fire_pre", "sustainedK_fire_pre"):
            if field in o and o[field] != c.get(field):
                mismatches.append(f"{k}: {field} ours={c.get(field)} theirs={o[field]}")
        for field in ("onset_idx", "sustainedK_fires_overall", "sustainedK_censored",
                      "sustainedK_detect_idx", "terminal_block_len"):
            if field in o and o[field] != c.get(field):
                mismatches.append(f"{k}: {field} ours={c.get(field)} theirs={o[field]}")
        if "q_null_binomial" in o:
            if abs(o["q_null_binomial"] - c["q_null_binomial"]) > 1e-9:
                mismatches.append(f"{k}: q_null ours={c['q_null_binomial']:.6g} theirs={o['q_null_binomial']:.6g}")
    kas = ka["summary"]
    checks = [
        ("n_curves_total", kas["n_curves_total"], summary["n_curves_total"]),
        ("n_chance_level", kas["n_chance_level"], summary["n_chance_level"]),
        ("n_emerged", kas["n_emerged_by_end"], summary["n_emerged_by_end"]),
        ("single_false_fires", kas["chance_level"]["single_crossing_false_fires"],
         summary["chance_level"]["single_crossing_false_fires"]),
        ("sustainedK_false_fires", kas["chance_level"]["sustainedK_false_fires"],
         summary["chance_level"]["sustainedK_false_fires"]),
        ("mc single pred", kas["chance_level"]["mc_only"]["single_fires_predicted_binomial_null"],
         summary["chance_level"]["mc_only"]["single_fires_predicted_binomial_null"]),
        ("mixture sustained pred", kas["chance_level"]["sustainedK_fires_mixture_pred"],
         summary["chance_level"]["sustainedK_fires_mixture_pred"]),
    ]
    for name, theirs, ours in checks:
        if isinstance(ours, float):
            ok = abs(theirs - ours) < 0.05
        else:
            ok = theirs == ours
        if not ok:
            mismatches.append(f"summary {name}: ours={ours} theirs={theirs}")
    # emerged-list latency diff
    ka_em = {(e["model"], e["task"]): e for e in kas["emerged"]}
    for e in summary["emerged"]:
        k = (e["model"], e["task"])
        o = ka_em.get(k)
        if o is None:
            mismatches.append(f"emerged {k} missing in killed-agent json")
            continue
        if (o["caught_by_sustainedK"] != e["sustainedK_fires_overall"]
                or o["censored_at_end"] != e["sustainedK_censored"]
                or o["detect_latency_ckpts"] != e["detect_latency_ckpts"]
                or o["onset_idx"] != e["onset_idx"] or o["K"] != e["K_nomogram"]):
            mismatches.append(f"emerged {k}: ours={e} theirs={o}")

    print(json.dumps(summary, indent=1))
    print("\n=== DIFF vs killed-agent analysis ===")
    if mismatches:
        for m in mismatches:
            print("MISMATCH:", m)
    else:
        print("ALL AGREE (curve-level fields, summary counts, emerged list, "
              "binomial nulls within tolerance)")
    print(f"\nwritten: {outpath}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
