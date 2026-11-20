"""Level-conditioned hand-movement targets for the quality mode (numpy only).

The previous quality mode minimised hand travel (``handtravel.window_penalty``): it removed the fast jumps and the crossing patterns the
official charts of that level contain.  This module makes the hand statistics MATCH the official distribution instead.

Targets (``work/hand_targets.json``, fitted by ``scratch/refine2/tools/hand_fit.py`` on the TRAIN split of the official Master / Re:MASTER charts of
versionid >= 10, per level band <=12 / 12+ / 13 / 13+ / 14 / 14+): the distribution of the speed of every hand move (screen radii / s, 12 bins whose
edges separate the "floor" speeds of moves that start the moment the previous contact ends), of the jump size (6 bins), the crossing rate, plus the
chart-to-chart overdispersion (phi) of each histogram and the window-level extremes seen in official charts.

Budget (``HandBudget``): the counts of the chart accepted so far (plus ``n0`` pseudo-contacts distributed like the target).  For every bin the
cumulative count O is compared with the count the official distribution predicts for that many contacts, E = N p, in units of the chart-to-chart
spread, z = (O - E) / sqrt(phi E + 1).  The penalty is two-sided with a dead band, ``max(0, |z| - z0) ** 2`` summed over the bins: a chart that is
too smooth (too few fast moves, too few crossings) is penalised like a chart that is too wild, a chart inside the official spread is not touched,
so fast jumps and crossings keep occurring at the official rate and nothing pushes them to zero.  The cost of a candidate window is the change
of this budget penalty when its contacts are added; a few "implausible" events (speed above anything in official charts, more fast moves in one
window than 99.5 % of official windows) cost extra.
"""
from __future__ import annotations

import json
import os

import numpy as np

BANDS = ("<=12", "12+", "13", "13+", "14", "14+")


def band_of(level):
    """display level band of a level constant (None -> 'all')"""
    if level is None:
        return "all"
    x = float(level)
    if x < 12.55:
        return "<=12"
    if x < 12.95:
        return "12+"
    if x < 13.55:
        return "13"
    if x < 13.95:
        return "13+"
    if x < 14.55:
        return "14"
    return "14+"


_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_PKG, "work", "hand_targets.json")
if not os.path.exists(DEFAULT_PATH):             # stock layout (release/v6_cal): the project's work/ two levels up
    DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(_PKG)), "work", "hand_targets.json")
_CACHE = {}


def load(path=None):
    path = os.path.abspath(path or DEFAULT_PATH)
    if path not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE[path] = json.load(f)
    return _CACHE[path]


class BandTarget:
    def __init__(self, data, band):
        d = data["bands_data"][band]
        self.band = band
        self.speed_edges = np.asarray(data["speed_edges"], float)
        self.dist_edges = np.asarray(data["dist_edges"], float)
        self.tail_speed = float(data.get("tail_speed", 27.0))
        self.p_speed = np.asarray(d["speed_p"], float)
        self.p_dist = np.asarray(d["dist_p"], float)
        self.p_cross = float(d["cross_p"])
        self.phi_speed = max(1.0, float(d["phi"]["speed"]))
        self.phi_dist = max(1.0, float(d["phi"]["dist"]))
        self.phi_cross = max(1.0, float(d["phi"]["cross"]))
        self.v_hard = float(d["v_hard"])
        self.cap_tail = float(d["window"]["tail_count"]["p99.5"])
        self.cap_cross = float(d["window"]["cross_count"]["p99.5"])
        self.info = d


def target_for(level, path=None):
    data = load(path)
    band = band_of(level)
    if band not in data["bands_data"]:
        band = "all"
    return BandTarget(data, band)


def _bin_counts(values, edges, n_bins):
    if len(values) == 0:
        return np.zeros(n_bins)
    return np.bincount(np.searchsorted(edges, np.asarray(values, float), side="right"), minlength=n_bins).astype(float)


class HandBudget:
    """Cumulative hand statistics of the chart so far against a ``BandTarget`` (see module doc)."""

    __slots__ = ("t", "n0", "z0", "kappa", "w_hard", "w_speed", "w_dist", "w_cross", "clip", "S", "D", "X", "n", "_pen")

    def __init__(self, target, n0=120.0, z0=0.25, kappa=1.0, w_hard=1.0, w_speed=1.0, w_dist=1.0, w_cross=2.0, clip=0.15):
        self.t = target
        self.clip = float(clip)          # the reward for moving towards the target is capped (it must never outweigh a muri / activity / density penalty)
        self.n0, self.z0, self.kappa, self.w_hard = float(n0), float(z0), float(kappa), float(w_hard)
        self.w_speed, self.w_dist, self.w_cross = float(w_speed), float(w_dist), float(w_cross)
        self.S = np.zeros(len(target.p_speed))
        self.D = np.zeros(len(target.p_dist))
        self.X = 0.0
        self.n = 0
        self._pen = self._penalty(self.S, self.D, self.X, 0)

    def copy(self):
        o = HandBudget.__new__(HandBudget)
        for k in ("t", "n0", "z0", "kappa", "w_hard", "w_speed", "w_dist", "w_cross", "clip", "X", "n", "_pen"):
            setattr(o, k, getattr(self, k))
        o.S, o.D = self.S.copy(), self.D.copy()
        return o

    # ------------------------------------------------------------------ penalty of a count state
    def _hinge(self, O, p, phi, N):
        E = N * p
        z = (O - E) / np.sqrt(phi * E + 1.0)
        return float(np.sum(np.maximum(0.0, np.abs(z) - self.z0) ** 2))

    def _penalty(self, S, D, X, n):
        t = self.t
        N = n + self.n0
        pen = self.w_speed * self._hinge(S + self.n0 * t.p_speed, t.p_speed, t.phi_speed, N)
        pen += self.w_dist * self._hinge(D + self.n0 * t.p_dist, t.p_dist, t.phi_dist, N)
        pc = t.p_cross
        pen += self.w_cross * self._hinge(np.array([X + self.n0 * pc, (n - X) + self.n0 * (1 - pc)]), np.array([pc, 1 - pc]), t.phi_cross, N)
        return pen

    def penalty(self):
        return self._pen

    def z_scores(self):
        """diagnostics: (z speed bins, z dist bins, z cross) of the current state"""
        t, N = self.t, self.n + self.n0
        zs = lambda O, p, phi: (O - N * p) / np.sqrt(phi * N * p + 1.0)
        return (zs(self.S + self.n0 * t.p_speed, t.p_speed, t.phi_speed), zs(self.D + self.n0 * t.p_dist, t.p_dist, t.phi_dist),
                float(zs(np.array([self.X + self.n0 * t.p_cross]), np.array([t.p_cross]), t.phi_cross)[0]))

    # ------------------------------------------------------------------ candidate window
    def _counts(self, speeds, dists, crossed):
        t = self.t
        S = _bin_counts(speeds, t.speed_edges, len(t.p_speed))
        D = _bin_counts(dists, t.dist_edges, len(t.p_dist))
        return S, D, float(sum(bool(c) for c in crossed)), len(speeds)

    def cost(self, speeds, dists, crossed):
        """(cost, info) of adding these contacts: kappa x (budget penalty after - before) + w_hard x implausible events."""
        t = self.t
        S, D, X, m = self._counts(speeds, dists, crossed)
        after = self._penalty(self.S + S, self.D + D, self.X + X, self.n + m)
        delta = after - self._pen
        sp = np.asarray(speeds, float) if len(speeds) else np.zeros(0)
        over = float(np.sum(np.maximum(0.0, sp - t.v_hard) / t.v_hard)) * 4.0
        tails = int((sp >= t.tail_speed).sum())
        hard = over + max(0.0, tails - t.cap_tail) + 0.5 * max(0.0, X - t.cap_cross)
        cost = max(self.kappa * delta, -self.clip) + self.w_hard * hard
        return float(cost), dict(cost=float(cost), dev=float(delta), hard=float(hard), tail=tails, cross=int(X), budget=float(after))

    def add(self, speeds, dists, crossed):
        S, D, X, m = self._counts(speeds, dists, crossed)
        self.S += S
        self.D += D
        self.X += X
        self.n += m
        self._pen = self._penalty(self.S, self.D, self.X, self.n)
