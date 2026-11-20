"""Per-token marginal calibration of the sampler.

Sampling with temperature / top-p, likelihood reranking and a small amount of model bias under-represent rare tokens:
slides are almost all '-' / CW / CCW, connected slides are rare, doubles, breaks and 16th rhythms are scarce.  The
calibration adds a fixed logit offset to each token group so that the *marginal* statistics of generated charts match
those of official MASTER / Re:MASTER charts:

  shape    slide segment shapes (one offset per shape, first segments and continuations alike)
  cont     continuing a slide arm with another segment (odds; applied only with --fest on, which is what official
           connected slides belong to)
  kind     N_HOLD / N_SLIDE relative to N_TAP; ``touch`` / ``thold`` for N_TOUCH / N_THOLD (touch charts)
  double   another note on the same tick (odds)
  brk, sbrk, ex   BRK / SBRK / EX modifier tokens (odds)
  gap      the time token that follows a note, by the gap to it (3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 72, 96, 144, 192
           ticks, other short / long)

Offsets are fitted by ``tools/fit_calibration.py``: generate train songs, measure the marginals, move every offset by
``rate * log(target / generated)`` (log-odds for the binary groups), repeat.  ``work/calibration_v1.json`` belongs to
the v1 checkpoint family; refit for a new model.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os

import numpy as np

from . import simai
from .tokens import T_MIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(ROOT, "work", "calibration_v1.json")

SHAPES = list(simai.SHAPES)
GAPS = [3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 72, 96, 144, 192]
GAP_NAMES = [str(g) for g in GAPS] + ["short", "long"]
BUTTONS = ("tap", "hold", "slide")
MAX_BIAS = 4.0


def gap_class(delta):
    if delta in _GAP_SET:
        return str(delta)
    return "short" if delta < 48 else "long"


_GAP_SET = set(GAPS)


def count_marginals(notes):
    """Additive counts of one chart (``Counter``); see ``derive`` for how they become probabilities."""
    c = Counter()
    notes = sorted(notes, key=lambda n: (int(n.beat), n.pos))
    ticks = {}
    has_touch = has_fest = False
    for n in notes:
        c["notes"] += 1
        c[f"kind_{n.kind}"] += 1
        ticks[int(n.beat)] = ticks.get(int(n.beat), 0) + 1
        if n.kind in BUTTONS:
            c["buttons"] += 1
            c["brk"] += bool(n.brk)
            c["ex"] += bool(n.ex)
        else:
            has_touch = True
        if n.kind == "slide":
            for a in n.arms:
                c["arms"] += 1
                c["segments"] += len(a.segs)
                c["sbrk"] += bool(a.brk)
                if len(a.segs) > 1:
                    has_fest = True
                for s in a.segs:
                    c[f"shape_{s.shape}"] += 1
    c["doubles"] = sum(v - 1 for v in ticks.values())
    order = sorted(ticks)
    for a, b in zip(order[:-1], order[1:]):
        c[f"gap_{gap_class(b - a)}"] += 1
    if c["ex"]:
        c["ec_buttons"] = c["buttons"]
        c["ec_ex"] = c["ex"]
    if has_touch:
        c["tc_notes"] = c["notes"]
        c["tc_touch"] = c["kind_touch"]
        c["tc_thold"] = c["kind_thold"]
    if has_fest:
        c["fc_arms"] = c["arms"]
        c["fc_segments"] = c["segments"]
    return c


def _norm(counter, names, prefix, smooth=1.0):
    v = np.asarray([counter.get(prefix + n, 0.0) + smooth for n in names], float)
    return v / v.sum()


def _odds(num, den, smooth=1.0):
    p = (num + 0.5 * smooth) / (den + smooth)
    return min(max(p, 1e-5), 1 - 1e-5)


def derive(c):
    """Aggregated counts -> the probabilities the calibration matches."""
    buttons = max(c.get("buttons", 0.0), 1.0)
    shares = {k: c.get(f"kind_{k}", 0.0) for k in BUTTONS}
    total = max(sum(shares.values()), 1.0)
    out = dict(
        shape=dict(zip(SHAPES, _norm(c, SHAPES, "shape_", 2.0))),
        cont=_odds(c.get("fc_segments", 0.0) - c.get("fc_arms", 0.0), c.get("fc_segments", 0.0), 4.0),
        kind={k: (shares[k] + 1.0) / (total + 3.0) for k in BUTTONS},
        touch=_odds(c.get("tc_touch", 0.0) + c.get("tc_thold", 0.0), c.get("tc_notes", 0.0), 8.0),
        thold=_odds(c.get("tc_thold", 0.0), c.get("tc_touch", 0.0) + c.get("tc_thold", 0.0), 8.0),
        double=_odds(c.get("doubles", 0.0), c.get("notes", 0.0), 8.0),
        brk=_odds(c.get("brk", 0.0), buttons, 8.0),
        sbrk=_odds(c.get("sbrk", 0.0), max(c.get("arms", 0.0), 1.0), 8.0),
        ex=_odds(c.get("ec_ex", 0.0), max(c.get("ec_buttons", 0.0), 1.0), 8.0),
        gap=dict(zip(GAP_NAMES, _norm(c, GAP_NAMES, "gap_", 2.0))),
    )
    return out


def _logit(p):
    return math.log(p / (1.0 - p))


def blank():
    return dict(shape={s: 0.0 for s in SHAPES}, cont=0.0, kind={"hold": 0.0, "slide": 0.0}, touch=0.0, thold=0.0,
                double=0.0, brk=0.0, sbrk=0.0, ex=0.0, gap={g: 0.0 for g in GAP_NAMES}, meta={})


def update(bias, generated, target, rate=0.7, clip=1.5):
    """One fitting step: move every offset towards matching ``target`` (both from ``derive``)."""
    step = lambda delta: float(np.clip(rate * delta, -clip, clip))
    out = json.loads(json.dumps(bias))
    for s in SHAPES:
        out["shape"][s] += step(math.log(target["shape"][s] / generated["shape"][s]))
    for g in GAP_NAMES:
        out["gap"][g] += step(math.log(target["gap"][g] / generated["gap"][g]))
    for k in ("hold", "slide"):
        out["kind"][k] += step(math.log(target["kind"][k] / generated["kind"][k]))
    for k in ("cont", "touch", "thold", "double", "brk", "sbrk", "ex"):
        out[k] += step(_logit(target[k]) - _logit(generated[k]))

    def clamp(v):
        if isinstance(v, dict):
            return {k: clamp(x) for k, x in v.items()}
        return v if isinstance(v, (list, str)) else float(np.clip(v, -MAX_BIAS, MAX_BIAS))
    keep = out.get("meta", {})
    out = clamp({k: v for k, v in out.items() if k != "meta"})
    out["meta"] = keep
    return out


def load_bias(path=None):
    with open(path or CALIB_PATH, encoding="utf-8") as stream:
        data = json.load(stream)
    base = blank()
    for key, value in data.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


class Calibration:
    """Applies a bias dictionary (``blank()`` layout) to the decoder's dense logit offsets."""

    def __init__(self, V, bias, *, use_cont=True, skip=()):
        self.V = V
        bias = json.loads(json.dumps(bias))
        for group in skip:                         # ablation: leave a group uncalibrated
            if group not in bias or group == 'meta':
                raise ValueError(f'unknown calibration group {group!r}')
            bias[group] = ({k: 0.0 for k in bias[group]} if isinstance(bias[group], dict) else 0.0)
        self.bias = bias
        self.use_cont = use_cont
        self.shape_ids = np.asarray([V[f"SH_{s}"] for s in SHAPES])
        self.shape_off = np.asarray([bias["shape"][s] for s in SHAPES])
        k = bias["kind"]
        self.kind_ids = np.asarray([V["N_HOLD"], V["N_SLIDE"], V["N_TOUCH"], V["N_THOLD"]])
        self.kind_off = np.asarray([k["hold"], k["slide"], bias["touch"], bias["touch"] + bias["thold"]])
        self.all_n = np.asarray([V[f"N_{n}"] for n in ("TAP", "HOLD", "SLIDE", "TOUCH", "THOLD")])
        self.double = bias["double"]
        self.cont = bias["cont"] if use_cont else 0.0
        self.brk_id, self.sbrk_id, self.ex_id = V["BRK"], V["SBRK"], V["EX"]
        self.brk, self.sbrk, self.ex = bias["brk"], bias["sbrk"], bias["ex"]
        by_delta = np.zeros(1024)
        for delta in range(1, 1024):
            by_delta[delta] = bias["gap"][gap_class(delta)]
        self.gap_by_delta = by_delta
        self.time_ids = V.time0 - T_MIN + np.arange(768)

    def add(self, grammar, out):
        state = grammar.state
        boundary = grammar.at_boundary
        if state == "time" or (boundary and grammar.has_time):
            out[self.kind_ids] += self.kind_off
            if boundary:
                out[self.all_n] += self.double
        if boundary and grammar.has_time:
            cur, hi = grammar.cur_time, grammar.max_tick
            if hi > cur + 1:
                out[self.time_ids[cur + 1:hi]] += self.gap_by_delta[1:hi - cur]
        if state in ("arm_start", "arm_shape"):
            out[self.shape_ids] += self.shape_off
        elif state == "segment_end":
            out[self.shape_ids] += self.shape_off + self.cont
        elif state == "head":
            out[self.brk_id] += self.brk
            out[self.ex_id] += self.ex
        elif state == "slide_group_end":
            out[self.sbrk_id] += self.sbrk
