"""Direction 014 labels — route/outcome targets per the repo's authority defs.

Route definition authority: 001 findings R4 + 002 findings P1 —
    norm_ratio = wn_hidden(T_grok) / wn_hidden(T_mem)
    > 1  → growth/directional route ("route-2")
    < 1  → contraction route ("route-1")
002 red-team clause: rot_rate alone is NOT a route criterion (binds the
interpretation of features, not the label).
"""

import math


def _wn_hidden_at(rec, step):
    """wn_hidden at the eval record nearest to `step` (grid-quantized)."""
    if step is None:
        return None
    best, best_d = None, None
    for r in rec["steps"]:
        v = r.get("wn_hidden")
        if v is None:
            continue
        d = abs(r["step"] - step)
        if best_d is None or d < best_d:
            best, best_d = v, d
    return best


def make_labels(rec):
    """Return the label dict for one run record."""
    grok = rec["grok_step"]
    mem = rec["memorize_step"]
    labels = {
        "grokked": int(grok is not None),
        "memorized": int(mem is not None),
        "log_delay": None,
        "route": None,        # "growth" | "contraction" | None
        "norm_ratio": None,
    }
    if rec["delay_ratio"] is not None and rec["delay_ratio"] > 0:
        labels["log_delay"] = math.log(rec["delay_ratio"])
    if grok is not None and mem is not None:
        wg, wm = _wn_hidden_at(rec, grok), _wn_hidden_at(rec, mem)
        if wg is not None and wm not in (None, 0):
            ratio = wg / wm
            labels["norm_ratio"] = ratio
            labels["route"] = "growth" if ratio > 1.0 else "contraction"
    return labels


def label_coverage(records):
    """Inventory table for --dry-run: counts per namespace."""
    cov = {}
    for rec in records:
        ns = rec["namespace"]
        c = cov.setdefault(ns, {"runs": 0, "grokked": 0, "failed": 0,
                                "growth": 0, "contraction": 0,
                                "no_memorize": 0})
        lab = make_labels(rec)
        c["runs"] += 1
        c["grokked"] += lab["grokked"]
        c["failed"] += 1 - lab["grokked"]
        c["no_memorize"] += 1 - lab["memorized"]
        if lab["route"] == "growth":
            c["growth"] += 1
        elif lab["route"] == "contraction":
            c["contraction"] += 1
    return cov
