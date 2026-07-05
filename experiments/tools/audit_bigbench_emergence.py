#!/usr/bin/env python3
"""
audit_bigbench_emergence.py
===========================
Scan-statistic audit of BIG-Bench "emergent ability" curves.

Real-world case study for paper E2 ("Apparent emergence timing ... can be
scan-statistic false positives"). We take the PUBLIC per-task, per-model-scale
accuracy data released with BIG-Bench (Srivastava et al. 2022), which underlies
the canonical emergence claims of Wei et al. (2022), and apply the same
preflight instrument developed in E2 for the small-scale grokking testbeds:

  For a multiple-choice task with chance accuracy p and eval-set size N, the
  probability that a *chance-level* model exceeds an accuracy threshold tau at a
  SINGLE scale point is the binomial tail
        q = P(Binom(N, p) >= ceil(tau * N)).
  A threshold-crossing "emergence detector" that scans M_scale model scales on
  that task fires under the all-chance null with probability
        P_single = 1 - (1 - q)^{M_scale}.
  Scanning T tasks, the expected number of tasks that light up by chance alone is
        E[false emergents] = sum_t P_single_t.
  A "look-elsewhere" (Bonferroni) corrected criterion asks whether a task's own
  achieved peak beats the whole-benchmark chance envelope:
        E_task = q(V_peak) * (T * M_scale)  <  alpha   ->  survives.

DATA SOURCE (public, cached locally):
  github.com/google/BIG-bench   (Apache-2.0)
  Per task: bigbench/benchmark_tasks/<task>/results/scores_BIG-G_<scale>_T=0.json
            gives multiple_choice_grade and the exact random baseline (low_score)
            at each of 12 dense BIG-G scales (2m ... 128b non-embedding params).
            bigbench/benchmark_tasks/<task>/task.json  gives N = #examples and
            the choice count per example.

We restrict the *binomial-null* audit to tasks whose preferred metric is
multiple_choice_grade, because only there is the chance model p exactly defined.
Generative tasks scored by exact_str_match (e.g. modified_arithmetic,
word_unscrambling, IPA transliterate) have chance p ~ 0 and fall under a
DIFFERENT mechanism (metric discontinuity, Schaeffer et al. 2023); they are
reported separately and NOT run through the binomial null.

Deterministic. Downloads are cached under results/bigbench_audit/cache/.
Outputs: results/bigbench_audit/audit_table.json  and  audit_summary.txt
"""

import os
import re
import json
import math
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import binom
import requests

# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "bigbench_audit"))
CACHE = os.path.join(RESULTS, "cache")
os.makedirs(CACHE, exist_ok=True)

RAW = "https://raw.githubusercontent.com/google/BIG-bench/main/"
TREE_URL = "https://api.github.com/repos/google/BIG-bench/git/trees/main?recursive=1"

MC_METRIC = "multiple_choice_grade"
GEN_METRICS = {"exact_str_match", "case_insensitive_str_match"}

# Emergence-detector thresholds expressed as (accuracy - chance) margins.
DELTA_GRID = [0.05, 0.10, 0.20]
DELTA_PRINCIPAL = 0.10          # principal "looks like it's working" detector
ALPHA = 0.10                    # family-wise false-emergent budget for adjudication

# Named tasks frequently cited as emergent in Wei et al. (2022) / BIG-Bench.
# Used only for labelling the output, not for any computation.
CITED_EMERGENT = {
    "hindu_knowledge", "logical_deduction", "logical_args", "english_proverbs",
    "figure_of_speech_detection", "misconceptions", "known_unknowns",
    "metaphor_understanding", "metaphor_boolean", "odd_one_out", "persian_idioms",
    "irony_identification", "international_phonetic_alphabet_nli",
    "physical_intuition", "crass_ai", "fantasy_reasoning", "analogical_similarity",
    "phrase_relatedness", "conceptual_combinations", "code_line_description",
    # generative headline emergent tasks (reported separately, chance ~ 0):
    "modified_arithmetic", "word_unscrambling",
    "international_phonetic_alphabet_transliterate", "gre_reading_comprehension",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "e2-bigbench-audit/1.0"})


# ----------------------------------------------------------------------------
def cached_get(rel_path):
    """Download a raw repo file (rel_path from repo root), cache by path."""
    local = os.path.join(CACHE, rel_path.replace("/", "__"))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        with open(local, "rb") as f:
            return f.read()
    url = RAW + urllib.parse.quote(rel_path)
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                with open(local, "wb") as f:
                    f.write(r.content)
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def get_tree():
    local = os.path.join(RESULTS, "repo_tree.json")
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    r = SESSION.get(TREE_URL, timeout=120)
    r.raise_for_status()
    d = r.json()
    with open(local, "w") as f:
        json.dump(d, f)
    return d


# ----------------------------------------------------------------------------
def inventory():
    """Map top-level task -> {scale: score_path}, and task.json path."""
    tree = get_tree()
    paths = [e["path"] for e in tree["tree"]]
    tasks = {}
    for p in paths:
        parts = p.split("/")
        # top-level score file: bigbench/benchmark_tasks/<task>/results/<file>
        if (len(parts) == 5 and parts[1] == "benchmark_tasks"
                and parts[3] == "results"
                and parts[4].startswith("scores_BIG-G_") and "T=0" in parts[4]):
            m = re.search(r"scores_BIG-G_([0-9a-z]+)_T=0\.json", parts[4])
            if m:
                tasks.setdefault(parts[2], {"scores": {}, "task_json": None})
                tasks[parts[2]]["scores"][m.group(1)] = p
    # attach top-level task.json
    for p in paths:
        parts = p.split("/")
        if (len(parts) == 4 and parts[1] == "benchmark_tasks"
                and parts[3] == "task.json" and parts[2] in tasks):
            tasks[parts[2]]["task_json"] = p
    return tasks


def read_score_file(path):
    raw = cached_get(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def pick_shot_entry(score_json, task_name):
    """From a scores JSON pick the aggregate entry at the MAX number_of_shots.
    Max shots gives emergence its best case (most in-context help)."""
    entries = score_json.get("scores", [])
    # prefer the aggregate whose subtask_description equals the task name
    agg = [e for e in entries if e.get("subtask_description") in (task_name, None)]
    pool = agg if agg else entries
    if not pool:
        return None
    return max(pool, key=lambda e: e.get("number_of_shots", 0))


def load_task_curve(task_name, info):
    """Return dict with per-scale accuracy, params, chance p, preferred metric."""
    scales = info["scores"]
    per_scale = []
    preferred = None
    chances = []
    for scale, path in scales.items():
        sj = read_score_file(path)
        if sj is None:
            continue
        entry = pick_shot_entry(sj, task_name)
        if entry is None:
            continue
        preferred = entry.get("preferred_score", preferred)
        params = (sj.get("model", {}).get("non_embedding_params")
                  or sj.get("model", {}).get("total_params"))
        sd = entry.get("score_dict", {})
        acc = sd.get(preferred, sd.get(MC_METRIC))
        low = entry.get("low_score")
        if low is not None and preferred == MC_METRIC:
            chances.append(low)
        per_scale.append({
            "scale": scale, "params": params, "acc": acc,
            "shots": entry.get("number_of_shots"),
            "low_score": low, "high_score": entry.get("high_score"),
        })
    per_scale = [x for x in per_scale if x["params"] is not None and x["acc"] is not None]
    per_scale.sort(key=lambda x: x["params"])
    chance = float(np.mean(chances)) if chances else None
    return {"preferred": preferred, "curve": per_scale, "chance": chance}


def task_N_and_choices(info):
    """N = #examples; choices distribution from task.json (top-level only)."""
    tj = info.get("task_json")
    if tj is None:
        return None, None
    raw = cached_get(tj)
    if raw is None:
        return None, None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    ex = d.get("examples")
    if not isinstance(ex, list) or not ex:
        return None, None
    n = len(ex)
    ch = []
    for e in ex:
        ts = e.get("target_scores")
        if isinstance(ts, dict) and ts:
            ch.append(len(ts))
    mean_choices = float(np.mean(ch)) if ch else None
    return n, mean_choices


# ----------------------------------------------------------------------------
def exceed_prob(N, p, tau):
    """q = P(Binom(N,p) >= ceil(tau*N)) : chance model beats tau at one point."""
    if N is None or p is None or N <= 0:
        return None
    k = math.ceil(tau * N)
    k = max(k, 0)
    if k > N:
        return 0.0
    # sf(k-1) = P(X >= k)
    return float(binom.sf(k - 1, N, p))


def analyze():
    inv = inventory()
    print(f"[inventory] {len(inv)} top-level BIG-G T=0 tasks")

    # Download all score files in parallel (cached).
    all_paths = []
    for t, info in inv.items():
        all_paths.extend(info["scores"].values())
    print(f"[download] {len(all_paths)} score files (cached) ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(cached_get, all_paths))

    # Build curves.
    curves = {}
    for t, info in inv.items():
        curves[t] = load_task_curve(t, info)

    mc_tasks = {t: c for t, c in curves.items()
                if c["preferred"] == MC_METRIC and len(c["curve"]) >= 6}
    gen_tasks = {t: c for t, c in curves.items()
                 if c["preferred"] in GEN_METRICS}
    print(f"[classify] {len(mc_tasks)} multiple-choice tasks, "
          f"{len(gen_tasks)} generative(str-match) tasks")

    # Fetch N for MC tasks (parallel).
    def _fetch_N(t):
        return t, task_N_and_choices(inv[t])
    N_map = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for t, res in ex.map(_fetch_N, list(mc_tasks.keys())):
            N_map[t] = res

    records = []
    for t, c in mc_tasks.items():
        curve = c["curve"]
        accs = np.array([x["acc"] for x in curve], dtype=float)
        params = np.array([x["params"] for x in curve], dtype=float)
        N, mean_choices = N_map.get(t, (None, None))
        # chance: prefer low_score baseline; fallback 1/choices
        p = c["chance"]
        if p is None and mean_choices:
            p = 1.0 / mean_choices
        if N is None or p is None:
            continue
        M_scale = len(curve)
        v_peak = float(np.max(accs))
        v_final = float(accs[-1])
        # early flatness: min accuracy over the smaller-scale half
        half = max(1, M_scale // 2)
        early_min = float(np.min(accs[:half]))
        tau_principal = p + DELTA_PRINCIPAL

        # apparent-emergence flag: near chance early, crosses principal tau at peak
        flagged = (early_min <= p + 0.05) and (v_peak >= tau_principal)

        q_grid = {f"delta_{d:.2f}": exceed_prob(N, p, p + d) for d in DELTA_GRID}
        q_principal = exceed_prob(N, p, tau_principal)
        p_single = 1.0 - (1.0 - q_principal) ** M_scale if q_principal is not None else None
        # peak-level exceedance (for look-elsewhere adjudication)
        q_peak = exceed_prob(N, p, v_peak)

        records.append({
            "task": t, "N": int(N), "chance_p": round(p, 4),
            "mean_choices": round(mean_choices, 3) if mean_choices else None,
            "M_scale": M_scale, "shots": curve[-1]["shots"],
            "v_peak": round(v_peak, 4), "v_final": round(v_final, 4),
            "early_min": round(early_min, 4),
            "jump_peak_minus_chance": round(v_peak - p, 4),
            "flagged_apparent_emergence": bool(flagged),
            "tau_principal": round(tau_principal, 4),
            "q_principal": q_principal, "P_single_principal": p_single,
            "q_peak": q_peak, "q_grid": q_grid,
            "cited_emergent": t in CITED_EMERGENT,
            "acc_by_scale": [round(a, 4) for a in accs.tolist()],
            "params_by_scale": params.tolist(),
        })

    T = len(records)
    # Family-wise expected false emergents at the principal detector threshold.
    E_false_principal = float(np.nansum([r["P_single_principal"] for r in records
                                         if r["P_single_principal"] is not None]))
    n_flagged = sum(r["flagged_apparent_emergence"] for r in records)

    # Sensitivity: E[false] across the delta grid (single-crossing over 12 scales,
    # summed over all T tasks under the all-chance null).
    E_false_grid = {}
    for d in DELTA_GRID:
        key = f"delta_{d:.2f}"
        tot = 0.0
        for r in records:
            q = r["q_grid"].get(key)
            if q is not None:
                tot += 1.0 - (1.0 - q) ** r["M_scale"]
        E_false_grid[key] = round(tot, 3)

    # Look-elsewhere adjudication for each flagged task.
    scan_opportunities = T * 12  # tasks x scale points actually scanned
    for r in records:
        qp = r["q_peak"]
        if qp is None:
            r["E_task_lookelsewhere"] = None
            r["verdict"] = "undetermined"
            continue
        E_task = qp * scan_opportunities
        r["E_task_lookelsewhere"] = E_task
        # within-task scan significance of the peak
        r["P_peak_withintask"] = 1.0 - (1.0 - qp) ** r["M_scale"]
        if not r["flagged_apparent_emergence"]:
            r["verdict"] = "not_flagged"
        elif E_task < ALPHA:
            r["verdict"] = "survives"          # peak beats whole-benchmark chance envelope
        elif r["P_peak_withintask"] >= 0.05:
            r["verdict"] = "scan_false_positive"   # not even significant on its own 12-pt scan
        else:
            r["verdict"] = "fragile"           # own-task sig but not look-elsewhere robust

    # Generative tasks: report chance~0 scope note (no binomial null).
    gen_report = []
    for t, c in gen_tasks.items():
        if not c["curve"]:
            continue
        accs = [x["acc"] for x in c["curve"]]
        gen_report.append({
            "task": t, "metric": c["preferred"],
            "v_peak": round(float(np.max(accs)), 4),
            "v_final": round(float(accs[-1]), 4),
            "cited_emergent": t in CITED_EMERGENT,
            "note": "chance p ~ 0; metric-discontinuity mechanism (Schaeffer 2023), "
                    "not scan-statistic; excluded from binomial null",
        })

    oos = out_of_scope_report(inv, set(mc_tasks.keys()), set(gen_tasks.keys()))

    summary = {
        "data_source": {
            "repo": "https://github.com/google/BIG-bench",
            "files": "bigbench/benchmark_tasks/<task>/results/scores_BIG-G_<scale>_T=0.json"
                     " and .../task.json",
            "model_series": "BIG-G dense, T=0, 12 scales (2m..128b non-embedding params)",
            "shots": "aggregate score at max available number_of_shots per task",
        },
        "scope": {
            "top_level_tasks": len(inv),
            "multiple_choice_tasks_audited": T,
            "generative_strmatch_tasks_reported_separately": len(gen_report),
        },
        "headline": {
            "detector": f"accuracy >= chance + {DELTA_PRINCIPAL} at some scale, "
                        f"near-chance (<= chance+0.05) early",
            "n_tasks_flagged_apparent_emergence": int(n_flagged),
            "E_false_emergents_predicted_by_chance": round(E_false_principal, 3),
            "E_false_by_threshold_grid": E_false_grid,
            "alpha_budget": ALPHA,
            "scan_opportunities_task_x_scale": scan_opportunities,
            "verdicts": _count([r["verdict"] for r in records]),
        },
    }

    out = {
        "summary": summary,
        "mc_tasks": sorted(records, key=lambda r: (-int(r["flagged_apparent_emergence"]),
                                                   r["q_peak"] if r["q_peak"] is not None else 1e9)),
        "generative_tasks": sorted(gen_report, key=lambda r: -r["v_peak"]),
        "cited_headline_out_of_scope": oos,
    }
    with open(os.path.join(RESULTS, "audit_table.json"), "w") as f:
        json.dump(out, f, indent=2)
    write_summary(out)
    return out


def out_of_scope_report(inv, mc_set, gen_set):
    """For famous cited-emergent tasks NOT in the MC binomial audit, record the
    reason (generative chance~0, continuous metric, or container/subtask task
    with no top-level example set) and a best-effort peak accuracy."""
    rows = []
    for t in sorted(CITED_EMERGENT):
        if t in mc_set or t not in inv:
            continue
        # scan every scale/shot entry for this task; collect all preferred metrics
        preferred_seen = set()
        subtask_labeled = False
        gen_peak = None   # peak of a string-match metric, if any
        mc_peak = None    # peak of multiple_choice_grade, if any
        for path in inv[t]["scores"].values():
            sj = read_score_file(path)
            if sj is None:
                continue
            for e in sj.get("scores", []):
                pf = e.get("preferred_score")
                if pf:
                    preferred_seen.add(pf)
                sd = e.get("score_dict", {})
                for gm in GEN_METRICS:
                    if sd.get(gm) is not None:
                        gen_peak = sd[gm] if gen_peak is None else max(gen_peak, sd[gm])
                if sd.get(MC_METRIC) is not None:
                    mc_peak = sd[MC_METRIC] if mc_peak is None else max(mc_peak, sd[MC_METRIC])
                if e.get("subtask_description") not in (t, None):
                    subtask_labeled = True
        preferred = sorted(preferred_seen)
        # reason
        tj = cached_get(inv[t]["task_json"]) if inv[t].get("task_json") else None
        has_examples = False
        if tj:
            try:
                has_examples = isinstance(json.loads(tj).get("examples"), list)
            except json.JSONDecodeError:
                pass
        if preferred_seen & GEN_METRICS:
            reason = "generative exact-match, chance~0 (metric-discontinuity mechanism)"
            peak = gen_peak
        elif preferred_seen and not (preferred_seen & {MC_METRIC}):
            reason = ("continuous metric (%s); no discrete chance baseline"
                      % ",".join(preferred))
            peak = None
        elif subtask_labeled or not has_examples:
            reason = "container task: scored via subtasks, no top-level example set / N"
            peak = mc_peak
        else:
            reason = "excluded (insufficient top-level data)"
            peak = mc_peak
        rows.append({
            "task": t, "preferred_metric": preferred,
            "best_peak_preferred_metric": round(peak, 4) if peak is not None else None,
            "excluded_reason": reason,
        })
    return rows


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def write_summary(out):
    s = out["summary"]
    recs = out["mc_tasks"]
    lines = []
    P = lines.append
    P("=" * 78)
    P("BIG-BENCH EMERGENCE SCAN-STATISTIC AUDIT  (case study for paper E2)")
    P("=" * 78)
    P("")
    P("DATA: %s" % s["data_source"]["repo"])
    P("      %s" % s["data_source"]["files"])
    P("      %s" % s["data_source"]["model_series"])
    P("      shots: %s" % s["data_source"]["shots"])
    P("")
    P("SCOPE:")
    P("  top-level BIG-G tasks              : %d" % s["scope"]["top_level_tasks"])
    P("  multiple-choice tasks audited (T)  : %d" % s["scope"]["multiple_choice_tasks_audited"])
    P("  generative str-match tasks (sep.)  : %d" % s["scope"]["generative_strmatch_tasks_reported_separately"])
    P("")
    h = s["headline"]
    P("HEADLINE (detector = %s):" % h["detector"])
    P("  tasks flagged as apparent emergence      : %d" % h["n_tasks_flagged_apparent_emergence"])
    P("  E[false emergents] predicted by chance   : %.2f" % h["E_false_emergents_predicted_by_chance"])
    P("  E[false] across detector grid (delta)    : %s" % h["E_false_by_threshold_grid"])
    P("  scan opportunities (tasks x 12 scales)   : %d" % h["scan_opportunities_task_x_scale"])
    P("  adjudication verdicts (alpha=%.2f)       : %s" % (h["alpha_budget"], h["verdicts"]))
    P("")
    P("-" * 78)
    P("SURVIVORS (peak beats whole-benchmark chance envelope; genuine emergence):")
    P("-" * 78)
    surv = [r for r in recs if r["verdict"] == "survives"]
    surv.sort(key=lambda r: -r["jump_peak_minus_chance"])
    P("  %-42s %5s %6s %7s %8s %8s" % ("task", "N", "chance", "peak", "jump", "q_peak"))
    for r in surv[:20]:
        P("  %-42s %5d %6.3f %7.3f %8.3f %8.1e%s" % (
            r["task"][:42], r["N"], r["chance_p"], r["v_peak"],
            r["jump_peak_minus_chance"], r["q_peak"],
            "  *cited" if r["cited_emergent"] else ""))
    P("  ... %d survivors total" % len(surv))
    P("")
    P("-" * 78)
    P("SCAN FALSE POSITIVES (apparent jump NOT distinguishable from chance scan):")
    P("-" * 78)
    fp = [r for r in recs if r["verdict"] == "scan_false_positive"]
    fp.sort(key=lambda r: -r["q_peak"])
    P("  %-42s %5s %6s %7s %8s %9s" % ("task", "N", "chance", "peak", "jump", "E_task"))
    for r in fp[:25]:
        P("  %-42s %5d %6.3f %7.3f %8.3f %9.1f%s" % (
            r["task"][:42], r["N"], r["chance_p"], r["v_peak"],
            r["jump_peak_minus_chance"], r["E_task_lookelsewhere"],
            "  *cited" if r["cited_emergent"] else ""))
    P("  ... %d scan-false-positive candidates total" % len(fp))
    P("")
    P("-" * 78)
    P("FRAGILE (own-task significant, not look-elsewhere robust):")
    P("-" * 78)
    fr = [r for r in recs if r["verdict"] == "fragile"]
    fr.sort(key=lambda r: -r["q_peak"])
    for r in fr[:15]:
        P("  %-42s N=%-5d chance=%.3f peak=%.3f E_task=%.2f%s" % (
            r["task"][:42], r["N"], r["chance_p"], r["v_peak"],
            r["E_task_lookelsewhere"], "  *cited" if r["cited_emergent"] else ""))
    P("  ... %d fragile total" % len(fr))
    P("")
    P("-" * 78)
    P("GENERATIVE HEADLINE TASKS (chance ~ 0; OUT OF binomial-null scope):")
    P("-" * 78)
    for r in out["generative_tasks"]:
        if r["cited_emergent"]:
            P("  %-46s metric=%-22s peak=%.3f final=%.3f" % (
                r["task"][:46], r["metric"], r["v_peak"], r["v_final"]))
    P("")
    P("-" * 78)
    P("CITED HEADLINE EMERGENT TASKS OUTSIDE THIS AUDIT'S SCOPE (with reason):")
    P("-" * 78)
    for r in out.get("cited_headline_out_of_scope", []):
        pk = ("%.3f" % r["best_peak_preferred_metric"]) if r["best_peak_preferred_metric"] is not None else "n/a"
        P("  %-46s peak=%s  [%s]" % (r["task"][:46], pk, r["excluded_reason"]))
    P("")
    P("Adjudication rule: flagged AND q(V_peak)*(T*12) < %.2f  -> survives;" % ALPHA)
    P("  flagged AND peak not significant on own 12-pt scan   -> scan_false_positive;")
    P("  in between                                           -> fragile.")
    P("=" * 78)
    with open(os.path.join(RESULTS, "audit_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    analyze()
