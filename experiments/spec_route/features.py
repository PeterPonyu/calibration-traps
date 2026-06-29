"""Direction 014 early-window feature extraction.

Leakage discipline lives HERE and only here: a window function returns the
list of step records allowed into features; everything downstream consumes
that list blindly. Records after the window boundary must never influence
any feature value (smoke test poisons a post-window record to verify).
"""

import math

import numpy as np

WINDOWS = ("W-mem", "W-200", "W-500")
# Per-channel statistics extracted from the window (names for reporting).
STAT_NAMES = ("first", "last", "log_ratio", "slope", "curvature", "osc_amp")
MIN_WINDOW_RECORDS = 3  # below this the run is excluded from that window


def window_records(rec, window):
    """Return the step records inside the window, or None if undefined."""
    steps = rec["steps"]
    if window == "W-mem":
        m = rec["memorize_step"]
        if m is None:
            return None  # never memorized → W-mem undefined, use W-K instead
        cut = [r for r in steps if r["step"] <= m]
    elif window.startswith("W-"):
        k = int(window.split("-")[1])
        cut = [r for r in steps if r["step"] <= k]
    else:
        raise ValueError(window)
    if len(cut) < MIN_WINDOW_RECORDS:
        return None
    return cut


def _series(cut, channel):
    xs, ys = [], []
    for r in cut:
        v = r.get(channel)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs.append(float(r["step"]))
        ys.append(float(v))
    if len(ys) < MIN_WINDOW_RECORDS:
        return None, None
    return np.array(xs), np.array(ys)


def _channel_stats(xs, ys):
    eps = 1e-12
    first, last = ys[0], ys[-1]
    log_ratio = math.log(abs(last) + eps) - math.log(abs(first) + eps)
    # least-squares slope on normalized step axis
    xn = (xs - xs[0]) / max(xs[-1] - xs[0], 1.0)
    slope = float(np.polyfit(xn, ys, 1)[0]) if len(ys) >= 2 else 0.0
    curvature = float(np.polyfit(xn, ys, 2)[0]) if len(ys) >= 3 else 0.0
    # oscillation amplitude: std of first differences (2306.13253 homage —
    # their loss-oscillation signal, as a per-channel feature family here)
    osc_amp = float(np.std(np.diff(ys))) if len(ys) >= 2 else 0.0
    return [first, last, log_ratio, slope, curvature, osc_amp]


def extract_features(rec, window, drop_last_record=False):
    """Feature vector + names for one run, or (None, names) if undefined.

    drop_last_record: robustness arm for the documented W-mem overlap
    (label denominator wn_hidden(T_mem) coincides with the window endpoint).
    """
    cut = window_records(rec, window)
    if cut is None:
        return None, None
    if drop_last_record:
        if len(cut) - 1 < MIN_WINDOW_RECORDS:
            return None, None
        cut = cut[:-1]
    vec, names = [], []
    for ch in rec["channels"]:
        xs, ys = _series(cut, ch)
        if ys is None:
            # channel unusable in window → all-zero block keeps dims aligned
            vec.extend([0.0] * len(STAT_NAMES))
        else:
            vec.extend(_channel_stats(xs, ys))
        names.extend(f"{ch}.{s}" for s in STAT_NAMES)
    return np.array(vec, dtype=float), names


def config_features(rec, vocab):
    """One-hot-ish config vector for the config-only baseline.

    vocab: dict field -> sorted list of observed values (built by caller on
    the train split only, so unseen test values map to the zero vector).
    """
    vec = []
    for field, values in vocab.items():
        v = rec["config"].get(field)
        vec.extend(1.0 if v == val else 0.0 for val in values)
    return np.array(vec, dtype=float)


def build_config_vocab(records, fields=("optimizer", "lr", "weight_decay",
                                        "init_scale", "train_frac", "op")):
    vocab = {}
    for f in fields:
        vocab[f] = sorted({r["config"].get(f) for r in records},
                          key=lambda x: (x is None, str(x)))
    return vocab
