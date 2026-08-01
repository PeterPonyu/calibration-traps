#!/usr/bin/env python3
"""E2-T0-3ii: per-task item counts and chance baselines for the BIG-G audit,
summarised for the paper appendix. Reads results/bigbench_audit/audit_table.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
d = json.load(open(os.path.join(RES, "bigbench_audit", "audit_table.json")))

mc = d["mc_tasks"]
print("mc task record keys:", sorted(mc[0].keys()))
print("n mc tasks:", len(mc))


def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


groups = {}
for t in mc:
    groups.setdefault(t["verdict"], []).append(t)

summary = {}
for g, ts in sorted(groups.items()):
    Ns = [t["N"] for t in ts]
    ch = [t["chance_p"] for t in ts]
    summary[g] = {
        "n_tasks": len(ts),
        "N_min": min(Ns), "N_median": med(Ns), "N_max": max(Ns),
        "chance_min": round(min(ch), 3), "chance_max": round(max(ch), 3),
    }
print(json.dumps(summary, indent=1))

for g in ("scan_false_positive", "fragile"):
    print(f"\n--- {g} rows ---")
    for t in sorted(groups[g], key=lambda x: -x.get("E_task", 0)):
        print("  ", t["task"], "N=", t["N"], "chance=", round(t["chance_p"], 3),
              "peak=", round(t["v_peak"], 3), "jump=", round(t["jump_peak_minus_chance"], 3),
              "E_task=", round(t.get("E_task_lookelsewhere", float("nan")), 2),
              "cited=", t.get("cited_emergent", False))

out = {"summary_by_verdict": summary,
       "scan_false_positive_rows": groups["scan_false_positive"],
       "fragile_rows": groups["fragile"]}
json.dump(out, open(os.path.join(HERE, "t0_3_bigbench_appendix.json"), "w"),
          indent=1)
print("\nwrote t0_3_bigbench_appendix.json")
