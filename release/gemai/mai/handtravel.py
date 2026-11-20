"""Hand-travel / crossing heuristic for maimai charts (numpy only; geometry from mai.playcheck).

A deliberately simple two-hand model, used as one soft rule of the quality mode (candidate scoring, lookahead, self-refine) and
as an evaluation statistic.  It is NOT the playability check (mai.playcheck is): every chart it sees is already feasible.

Model
  * contacts come from ``playcheck.build_contacts`` (tap / touch instant, hold / touch-hold and slides sustained; a slide is one
    sustained contact from its head tap to the end of its first arm, a headless slide starts at its arm; extra arms are ignored).
  * two hands (0 = left, 1 = right).  A hand is busy until its sustained contact ends.  The contacts of one tick (1 or 2 used) are
    assigned to the free hands by the cheapest assignment: cost = excess speed + small distance term + crossing penalty.
  * speed of a hand move = distance (screen radii) / seconds since that hand's last contact ended (>= MIN_DT).
Outputs (``HandModel.feed``): per instant/sustained contact start the speed, and the crossing flag (left hand right of the right
hand by more than CROSS_MARGIN while both hands acted within CROSS_WINDOW seconds).
Window penalty: mean over the window's contacts of ``max(0, speed - V_OK) / V_OK`` (capped at 3 per contact) plus ``W_CROSS`` per
crossing, divided by the number of contacts (the unit is "typical excess"; 0 = unremarkable).
"""
from __future__ import annotations

import math

import numpy as np

from . import playcheck
from .playcheck import BTN_XY, SEN_XY

V_OK = 14.0          # screen radii per second; set from the official charts (scratch/refine/tools/calib_hand.py)
W_CROSS = 0.6
CROSS_MARGIN = 0.35
CROSS_WINDOW = 0.8
MIN_DT = 0.04
CAP = 3.0
HOME = ((-0.6, -0.3), (0.6, -0.3))


class HandModel:
    __slots__ = ("x", "y", "t_end", "busy", "last_t", "n", "tick_done")

    def __init__(self):
        self.x = [HOME[0][0], HOME[1][0]]
        self.y = [HOME[0][1], HOME[1][1]]
        self.t_end = [-1e9, -1e9]        # when the hand last stopped touching (seconds)
        self.busy = [-1e9, -1e9]         # sustained contact running until
        self.last_t = [-1e9, -1e9]       # when the hand last started a contact
        self.n = 0
        self.tick_done = -1              # last tick fed (contacts of a tick are fed together)

    def copy(self):
        o = HandModel.__new__(HandModel)
        o.x, o.y, o.t_end, o.busy, o.last_t = list(self.x), list(self.y), list(self.t_end), list(self.busy), list(self.last_t)
        o.n, o.tick_done = self.n, self.tick_done
        return o

    # ------------------------------------------------------------------ one tick group
    def _end_pos(self, c):
        if c.poly is not None:
            return c.poly[-1]
        return (c.x, c.y)

    def _cost(self, h, c):
        s = c.s0
        if self.busy[h] > s + 1e-6:
            return None
        d = math.hypot(c.x - self.x[h], c.y - self.y[h])
        dt = max(s - self.t_end[h], MIN_DT)
        speed = d / dt
        return speed, d

    def feed_group3(self, group):
        """group: contacts starting at the same tick.  Returns a list of (speed, crossed, dist) in the order of ``group`` (at most 2 are used)."""
        if len(group) > 2:
            group = group[:2]
        out = []
        opts = []
        n = len(group)
        if n == 1:
            for h in (0, 1):
                r = self._cost(h, group[0])
                if r is not None:
                    opts.append(((h,), r[0] / V_OK + 0.05 * r[1] / V_OK, [r[0]], [r[1]]))
        else:
            for a in ((0, 1), (1, 0)):
                rs = [self._cost(a[i], group[i]) for i in range(2)]
                if any(r is None for r in rs):
                    continue
                opts.append((a, sum(r[0] for r in rs) / V_OK + 0.05 * sum(r[1] for r in rs) / V_OK, [r[0] for r in rs], [r[1] for r in rs]))
        if not opts:
            return [(0.0, False, 0.0)] * n
        best = None
        for a, cost, speeds, dists in opts:
            # crossing check after the move (the idle hand counts only if it acted recently)
            xs = list(self.x)
            moved = set()
            for i, h in enumerate(a):
                xs[h] = group[i].x
                moved.add(h)
            s_now = group[0].s0
            live = all(h in moved or (s_now - self.t_end[h]) < CROSS_WINDOW for h in (0, 1))
            crossed = live and (xs[0] - xs[1]) > CROSS_MARGIN
            ctot = cost + (W_CROSS if crossed else 0.0)
            if best is None or ctot < best[0] - 1e-9:
                best = (ctot, a, speeds, crossed, dists)
        _, a, speeds, crossed, dists = best
        for i, h in enumerate(a):
            c = group[i]
            self.x[h], self.y[h] = self._end_pos(c)
            self.last_t[h] = c.s0
            if c.sustained:
                self.busy[h] = c.s1
                self.t_end[h] = c.s1
            else:
                self.busy[h] = c.s0
                self.t_end[h] = c.s0
            self.n += 1
            out.append((speeds[i], bool(crossed), dists[i]))
        return out

    def feed_group(self, group):
        """as ``feed_group3`` without the distance: list of (speed, crossed)"""
        return [(s, c) for s, c, _ in self.feed_group3(group)]


def contacts_for_model(notes, tempo):
    """playcheck contacts merged for the hand model: a slide's head and first arm become one sustained contact."""
    cs = playcheck.build_contacts(notes, tempo)
    # merge head + arm of the same slide note (ref[0] is the note); keep only the first arm
    seen_arm = set()
    heads = {}
    out = []
    for c in cs:
        if c.kind == "head":
            heads[id(c.ref[0])] = c
    for c in cs:
        if c.kind == "head":
            continue
        if c.kind == "arm":
            nid = id(c.ref[0])
            if nid in seen_arm:
                continue
            seen_arm.add(nid)
            h = heads.get(nid)
            if h is not None:
                m = playcheck.Contact(h.t0, c.t1, 0, "arm", h.x, h.y, poly=c.poly, ref=c.ref, cum=c.cum)
                m.s0, m.s1, m.e = h.s0, c.s1, c.e
                out.append(m)
            else:
                out.append(c)
            continue
        out.append(c)
    out.sort(key=lambda c: (c.t0, c.cls))
    return out


def window_penalty(speeds, crossed):
    """(penalty, stats) of a list of contact speeds / crossing flags"""
    if not len(speeds):
        return 0.0, dict(n=0, mean_speed=0.0, p90=0.0, cross=0, pen=0.0)
    sp = np.asarray(speeds, float)
    ex = np.minimum(np.maximum(sp - V_OK, 0.0) / V_OK, CAP)
    cr = int(sum(crossed))
    pen = float((ex.sum() + W_CROSS * cr) / len(sp))
    return pen, dict(n=len(sp), mean_speed=float(sp.mean()), p90=float(np.percentile(sp, 90)), cross=cr, pen=pen)


class ChartHands:
    """Incremental driver: ``advance(window_notes)`` feeds the notes of the next window (notes in canonical order) and returns the
    window cost/stats; ``peek(window_notes)`` does the same on a copy (candidate scoring).

    Two modes.  ``budget is None`` (legacy "travel" mode): the cost is the window penalty above (less travel = better).  With a
    ``handtarget.HandBudget``: the cost is the change of the level-target budget penalty (two-sided, matches the official distribution) plus the
    implausible-event terms; it may be negative (the window moves the chart towards the target).  ``term`` turns the value into the amount that
    is subtracted (times ``w_hand``) from a window score, ``bad`` into a non-negative badness for the self-refine target selection."""

    HAND_FREE = 0.10

    def __init__(self, tempo, hm=None, budget=None):
        self.tempo = tempo
        self.hm = hm or HandModel()
        self.budget = budget

    def copy(self):
        return ChartHands(self.tempo, self.hm.copy(), None if self.budget is None else self.budget.copy())

    def term(self, pen):
        return float(pen) if self.budget is not None else max(0.0, float(pen) - self.HAND_FREE)

    def bad(self, pen):
        return max(0.0, float(pen)) if self.budget is not None else max(0.0, float(pen) - self.HAND_FREE)

    def _run(self, hm, notes, start_tick, end_tick):
        cs = [c for c in contacts_for_model(notes, self.tempo) if start_tick <= c.t0 < end_tick]
        speeds, crossed, dists = [], [], []
        i = 0
        while i < len(cs):
            j = i
            while j < len(cs) and cs[j].t0 == cs[i].t0:
                j += 1
            for sp, cr, d in hm.feed_group3(cs[i:j]):
                speeds.append(sp)
                crossed.append(cr)
                dists.append(d)
            i = j
        return speeds, crossed, dists

    def _result(self, speeds, crossed, dists):
        pen, st = window_penalty(speeds, crossed)
        st["travel_pen"] = pen
        if self.budget is None:
            return pen, st
        cost, info = self.budget.cost(speeds, dists, crossed)
        st.update(info)
        st["pen"] = cost
        return cost, st

    def advance(self, notes, start_tick, end_tick):
        sp, cr, ds = self._run(self.hm, notes, start_tick, end_tick)
        out = self._result(sp, cr, ds)
        if self.budget is not None:
            self.budget.add(sp, ds, cr)
        return out

    def peek(self, notes, start_tick, end_tick):
        hm = self.hm.copy()
        sp, cr, ds = self._run(hm, notes, start_tick, end_tick)
        return self._result(sp, cr, ds)


def chart_hand_stats(notes, tempo):
    """whole-chart statistics: dict(mean_speed, p90, p99, cross_rate, pen, n)"""
    ch = ChartHands(tempo)
    last = max([int(n.end_beat()) for n in notes] + [0])
    cs = contacts_for_model(notes, tempo)
    speeds, crossed = [], []
    i = 0
    hm = ch.hm
    while i < len(cs):
        j = i
        while j < len(cs) and cs[j].t0 == cs[i].t0:
            j += 1
        for sp, cr in hm.feed_group(cs[i:j]):
            speeds.append(sp)
            crossed.append(cr)
        i = j
    pen, st = window_penalty(speeds, crossed)
    sp = np.asarray(speeds, float) if speeds else np.zeros(1)
    st.update(p99=float(np.percentile(sp, 99)), cross_rate=float(sum(crossed) / max(len(crossed), 1)),
              frac_fast=float((sp > 2 * V_OK).mean()))
    return st
