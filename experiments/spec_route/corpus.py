"""Direction 014 corpus loader — read-only over existing results namespaces.

Parses the repo's standard run jsonl format: line 0 = {"_meta": {...}},
middle lines = per-eval step records, last line = {"_summary": {...}}.
No experiment code is imported; this module touches results/ strictly
read-only and never looks at grok_numerics* (009 avoidance clause).
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")

# Tier-A: norm channels only. Tier-B: + mech probe channels.
TIER_A = ["grid_main", "lr_control", "lr_control_sc3", "wd_sweep", "tf_sweep",
          "task_mul", "task_s5", "fine_eval", "s5_rescue"]
TIER_B = ["mech", "s5_mech"]

# Hard exclusions (never read, even if asked): 009 live namespaces + non-route
# metric systems. Kept as a guard list so a typo in TIER_* cannot leak them in.
FORBIDDEN_PREFIXES = ("grok_numerics", "calib", "degree_staircase",
                      "induction_", "sink_", "eos_tiny", "figures")

NORM_CHANNELS = ["wn_total", "wn_hidden", "wn_embed",
                 "train_loss", "test_loss", "train_acc", "test_acc"]
MECH_CHANNELS = ["eff_rank_mean", "stable_rank_mean", "cos_init", "rot_rate"]
# cos_mem is null until memorization → useless inside a pre-mem window; excluded.

CONFIG_KEYS = ["optimizer", "lr", "weight_decay", "init_scale", "train_frac",
               "op", "p", "muon_lr"]


def _check_allowed(namespace):
    for bad in FORBIDDEN_PREFIXES:
        if namespace.startswith(bad):
            raise ValueError(f"namespace '{namespace}' is on the forbidden "
                             f"list (009 avoidance / out-of-scope)")


def parse_run(path):
    """Parse one run jsonl → dict(meta, steps, channels, summary) or None."""
    meta, summary, steps = None, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return None
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "_summary" in rec:
                summary = rec["_summary"]
            elif "step" in rec:
                steps.append(rec)
    if meta is None or summary is None or not steps:
        return None  # incomplete / in-flight run: skip, never guess
    return {"path": path, "meta": meta, "summary": summary, "steps": steps}


def run_to_record(run, namespace, tier):
    """Flatten a parsed run into the analysis record used downstream."""
    s = run["summary"]
    rec = {
        "namespace": namespace,
        "tier": tier,
        "path": run["path"],
        "seed": run["meta"].get("seed"),
        "config": {k: run["meta"].get(k) for k in CONFIG_KEYS},
        "eval_every": run["meta"].get("eval_every"),
        "memorize_step": s.get("memorize_step"),
        "grok_step": s.get("grok_step"),
        "delay_ratio": s.get("delay_ratio"),
        "final_test_acc": s.get("final_test_acc"),
        "final_train_acc": s.get("final_train_acc"),
        "stopped_step": s.get("stopped_step"),
        "steps": run["steps"],
    }
    chans = list(NORM_CHANNELS)
    if tier == "B":
        chans += MECH_CHANNELS
    # keep only channels actually present in the first record
    present = set(run["steps"][0].keys())
    rec["channels"] = [c for c in chans if c in present]
    return rec


def load_namespace(namespace, results_dir=RESULTS_DIR):
    _check_allowed(namespace)
    tier = "B" if namespace in TIER_B else "A"
    d = os.path.join(results_dir, namespace)
    records, skipped = [], 0
    if not os.path.isdir(d):
        return [], 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        run = parse_run(os.path.join(d, fn))
        if run is None:
            skipped += 1
            continue
        records.append(run_to_record(run, namespace, tier))
    return records, skipped


def load_corpus(namespaces=None, results_dir=RESULTS_DIR):
    """Load all (default) namespaces → list of records + per-ns inventory."""
    if namespaces is None:
        namespaces = TIER_A + TIER_B
    corpus, inventory = [], {}
    for ns in namespaces:
        recs, skipped = load_namespace(ns, results_dir)
        corpus.extend(recs)
        inventory[ns] = {"loaded": len(recs), "skipped": skipped}
    return corpus, inventory


def config_cell_key(rec):
    """Cell identity = full config sans seed (groups seeds of one cell)."""
    c = rec["config"]
    return (rec["namespace"],) + tuple(
        (k, c.get(k)) for k in CONFIG_KEYS)
