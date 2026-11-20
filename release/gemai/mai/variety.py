"""Variety statistics of a chart (numpy only), to compare generated charts with official ones.

``chart_variety(notes)`` -> flat dict of floats:
  bar_distinct_ratio      distinct bar signatures / non-empty bars   (1.0 = no bar ever repeats)
  bar_entropy             Shannon entropy (bits) of the bar-signature distribution, normalised by log2(#bars)
  rhythm_bar_distinct     the same ratio for the rhythm alone (onset ticks inside the bar, lanes ignored)
  rhythm_<class>          share of onsets by grid class: 4th, 8th, 12th (triplet), 16th, 24th, 32nd, other
  kind_<k>                share of tap / hold / slide / touch (touch + touch hold) notes; brk, ex: share of notes with it
  slide_shape_entropy     entropy (bits) of the first-segment shape distribution of slides; slide_shapes: distinct shapes
  lane_entropy            entropy (bits) of the button lane distribution (max 3)
  double_rate             share of button onset ticks with two simultaneous button notes
  dist_<0..4>             share of consecutive button onsets by circular lane distance (0 = same lane, 4 = opposite)
  trill_rate              share of lane triples A B A (A != B);  stair_rate: A, A+1, A+2 or A, A-1, A-2
  delta_4gram_distinct    distinct 4-grams / total of the lane-delta sequence (low = repetitive stream patterns)
  shape_run3, shape_run4  share of bars that are the 3rd / 4th+ consecutive bar of the same *shape* (rhythm + kinds +
                          relative button motion, see ``barsig``);  rk_run3 / rk_run4 the same for rhythm + kinds alone;
                          shape_maxrun / rk_maxrun the longest run
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from . import barsig

TPB = 48
BAR = 4 * TPB
SHAPES_N = 14

RHYTHM_CLASSES = ("4th", "8th", "12th", "16th", "24th", "32nd", "other")


def rhythm_class(tick):
    r = tick % TPB
    if r == 0:
        return "4th"
    if r == 24:
        return "8th"
    if r in (16, 32):
        return "12th"
    if r in (12, 36):
        return "16th"
    if r in (8, 40):
        return "24th"
    if r % 6 == 0:
        return "32nd"
    return "other"


def _entropy(counts):
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def _bar_sig(notes, bar_start, rhythm_only=False):
    out = []
    for n in notes:
        if rhythm_only:
            out.append(int(n.beat) - bar_start)
            continue
        arm = (tuple((a.wait, tuple((s.shape, s.end) for s in a.segs), tuple(a.groups)) for a in n.arms)
               if n.kind == "slide" else ())
        out.append((int(n.beat) - bar_start, n.kind, n.pos, int(n.dur), arm, n.brk, n.ex))
    return tuple(sorted(set(out), key=repr))


def _circ(a, b):
    d = abs(a - b) % 8
    return min(d, 8 - d)


def chart_variety(notes):
    notes = list(notes)
    n = len(notes)
    out = {}
    if n == 0:
        return out
    bars = {}
    for x in notes:
        bars.setdefault(int(x.beat) // BAR, []).append(x)
    keys = sorted(bars)
    sigs = [_bar_sig(bars[b], b * BAR) for b in keys]
    rsigs = [_bar_sig(bars[b], b * BAR, True) for b in keys]
    cnt = Counter(sigs)
    out["bar_distinct_ratio"] = len(cnt) / len(sigs)
    out["bar_entropy"] = _entropy(cnt.values()) / math.log2(len(sigs)) if len(sigs) > 1 else 1.0
    out["rhythm_bar_distinct"] = len(set(rsigs)) / len(rsigs)
    onset_ticks = sorted({int(x.beat) for x in notes})
    rc = Counter(rhythm_class(t) for t in onset_ticks)
    for name in RHYTHM_CLASSES:
        out[f"rhythm_{name}"] = rc.get(name, 0) / len(onset_ticks)
    kinds = Counter(x.kind for x in notes)
    out["kind_tap"] = kinds["tap"] / n
    out["kind_hold"] = kinds["hold"] / n
    out["kind_slide"] = kinds["slide"] / n
    out["kind_touch"] = (kinds["touch"] + kinds["thold"]) / n
    out["kind_brk"] = sum(1 for x in notes if x.brk or any(a.brk for a in x.arms)) / n
    out["kind_ex"] = sum(1 for x in notes if x.ex) / n
    shapes = Counter(x.arms[0].segs[0].shape for x in notes if x.kind == "slide" and x.arms and x.arms[0].segs)
    out["slide_shape_entropy"] = _entropy(shapes.values())
    out["slide_shapes"] = float(len(shapes))
    # button lanes per onset tick
    by_tick = {}
    for x in notes:
        if x.kind in ("tap", "hold", "slide"):
            by_tick.setdefault(int(x.beat), []).append(int(x.pos))
    ticks = sorted(by_tick)
    lanes = [min(by_tick[t]) for t in ticks]
    out["double_rate"] = (sum(1 for t in ticks if len(by_tick[t]) >= 2) / len(ticks)) if ticks else 0.0
    out["lane_entropy"] = _entropy(Counter(int(x.pos) for x in notes if x.kind in ("tap", "hold", "slide")).values())
    if len(lanes) >= 4:
        d = Counter(_circ(a, b) for a, b in zip(lanes[:-1], lanes[1:]))
        for k in range(5):
            out[f"dist_{k}"] = d.get(k, 0) / (len(lanes) - 1)
        trip = list(zip(lanes[:-2], lanes[1:-1], lanes[2:]))
        out["trill_rate"] = sum(1 for a, b, c in trip if a == c and a != b) / len(trip)
        def step(a, b):
            return ((b - a + 4) % 8) - 4
        out["stair_rate"] = sum(1 for a, b, c in trip if abs(step(a, b)) == 1 and step(a, b) == step(b, c)) / len(trip)
        deltas = [step(a, b) for a, b in zip(lanes[:-1], lanes[1:])]
        grams = [tuple(deltas[i:i + 4]) for i in range(len(deltas) - 3)]
        out["delta_4gram_distinct"] = len(set(grams)) / len(grams) if grams else 1.0
    for kind in ("shape", "rk"):
        runs = barsig.run_positions(notes, kind)
        if runs:
            out[f"{kind}_run3"] = runs[f"{kind}_run3"]
            out[f"{kind}_run4"] = runs[f"{kind}_run4"]
            out[f"{kind}_maxrun"] = runs[f"{kind}_maxrun"]
    return out


HISTOGRAM_GROUPS = {
    "rhythm": [f"rhythm_{c}" for c in RHYTHM_CLASSES],
    "kind": ["kind_tap", "kind_hold", "kind_slide", "kind_touch"],
    "dist": [f"dist_{k}" for k in range(5)],
}


def mean_stats(rows):
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def js_divergence(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def variety_distance(gen_rows, ref_rows):
    """Compare two sets of charts.  Returns (summary dict, per-stat table).

    ``scalar_z``: mean over the scalar stats of |mean_gen - mean_ref| / std_ref (std across the reference charts);
    ``js_<group>``: JS divergence (bits) between the mean histograms of the group.
    """
    g, r = mean_stats(gen_rows), mean_stats(ref_rows)
    table = {}
    zs = []
    scalar = [k for k in r if not any(k in v for v in HISTOGRAM_GROUPS.values())]
    for k in sorted(r):
        std = float(np.std([x[k] for x in ref_rows if k in x])) or 1e-6
        z = (g.get(k, 0.0) - r[k]) / std
        table[k] = (g.get(k, float("nan")), r[k], z)
        if k in scalar:
            zs.append(abs(z))
    summary = {"scalar_z": float(np.mean(zs))}
    for name, keys in HISTOGRAM_GROUPS.items():
        summary[f"js_{name}"] = js_divergence([g.get(k, 0.0) for k in keys], [r[k] for k in keys])
    return summary, table
