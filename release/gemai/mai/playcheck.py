"""Two-hand playability checker for maimai charts (canonical tick notes + NpTempo).

Model
  * every note becomes one or more *contacts* in the touch-screen plane:
      tap / touch            instant contact at its button / sensor
      hold / touch-hold      sustained contact [t, t+dur]  (hand may leave `slack` seconds early)
      slide                  head = instant tap at t (unless headless); every arm is a sustained *moving*
                             contact on [t+wait, t+wait+dur] that follows the slide path
  * two contacts CONFLICT (cannot be played by the same hand) when
      - they overlap in time and are farther apart than a span (button/button, button/touch, touch/touch), or
      - they do not overlap but the gap between them is shorter than the time a hand needs to get from the first
        position to the second (t_same for the same spot, t_adj for neighbours, t_far otherwise)  -> "fast jump".
    A chart is playable iff the conflict graph is 2-colourable (colours = hands).  Additionally (chart convention, true
    for all ~6400 official charts) at most `max_simul_buttons` = 2 tap/hold/slide-head notes may start on one tick.  Odd cycles are violations: three
    far contacts at once ("hands"), a tap while both hands hold, a far jump on the free hand in less than t_far while
    the other hand is holding ("jump"), ...
  * a group of >= slap_n touches at one tick is a palm/arm slap and is not counted.
Distances are in units of the touch-screen radius (button ring radius 0.85).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from .simai import SENSORS, TPB

# ----------------------------------------------------------------------------- geometry
R_A, R_B, R_D, R_E = 0.85, 0.5, 1.0, 0.7

def _polar(angle_deg, r):
    a = math.radians(angle_deg)
    return (r * math.sin(a), r * math.cos(a))

BTN_XY = {k: _polar((k - 0.5) * 45.0, R_A) for k in range(1, 9)}
RING_E = {k: _polar((k - 1) * 45.0, R_E) for k in range(1, 9)}
SEN_XY = []
for _s in SENSORS:
    if _s == "C":
        SEN_XY.append((0.0, 0.0))
    else:
        _z, _k = _s[0], int(_s[1:])
        if _z == "A": SEN_XY.append(_polar((_k - 0.5) * 45.0, R_A))
        elif _z == "B": SEN_XY.append(_polar((_k - 0.5) * 45.0, R_B))
        elif _z == "D": SEN_XY.append(_polar((_k - 1) * 45.0, R_D))
        else: SEN_XY.append(_polar((_k - 1) * 45.0, R_E))

def _ring(k, r):
    return _polar((k - 0.5) * 45.0, r)

_WAYPOINTS = {}

def arm_waypoints(start, segs):
    """polyline (list of xy) followed by the finger along a slide arm (memoised; callers must not mutate the list)"""
    key = (start, tuple((sg.shape, sg.end, sg.via) for sg in segs))
    pts = _WAYPOINTS.get(key)
    if pts is None:
        if len(_WAYPOINTS) > 50000:
            _WAYPOINTS.clear()
        pts = _WAYPOINTS[key] = _arm_waypoints(start, segs)
    return pts

_CUMS = {}

def _arm_cum(start, segs, poly):
    key = (start, tuple((sg.shape, sg.end, sg.via) for sg in segs))
    cum = _CUMS.get(key)
    if cum is None:
        if len(_CUMS) > 50000:
            _CUMS.clear()
        cum = [0.0]
        for i in range(len(poly) - 1):
            cum.append(cum[-1] + math.dist(poly[i], poly[i + 1]))
        _CUMS[key] = cum
    return cum

def _arm_waypoints(start, segs):
    pts = [BTN_XY[start]]
    cur = start
    for sg in segs:
        e, sh = sg.end, sg.shape
        if sh == "STRAIGHT":
            pts.append(BTN_XY[e])
        elif sh in ("CW", "CCW"):
            step = 1 if sh == "CW" else -1
            n = ((e - cur) * step) % 8
            if n == 0: n = 8
            k = cur
            for _ in range(n):
                k = (k - 1 + step) % 8 + 1
                pts.append(BTN_XY[k])
        elif sh in ("VEE", "S", "Z", "FAN"):
            pts.append((0.0, 0.0)); pts.append(BTN_XY[e])
        elif sh == "GRANDV":
            pts.append(BTN_XY[sg.via]); pts.append(BTN_XY[e])
        elif sh in ("P", "Q", "PP", "QQ"):
            r = R_B if sh in ("P", "Q") else R_E
            step = -1 if sh in ("P", "PP") else 1
            pts.append(_ring(cur, r))
            n = ((e - cur) * step) % 8
            k = cur
            for _ in range(n):
                k = (k - 1 + step) % 8 + 1
                pts.append(_ring(k, r))
            pts.append(BTN_XY[e])
        else:
            pts.append(BTN_XY[e])
        cur = e
    return pts

# ----------------------------------------------------------------------------- parameters
@dataclass
class Params:
    sbb: float = 0.7      # max distance for two button-type contacts (tap/hold/head/slide arm) to share a hand
    sbt: float = 1.3      # button-type <-> touch-type
    stt: float = 1.6      # touch <-> touch
    slap_n: int = 8       # >= slap_n instant touches at one tick are a palm/arm "slap": ignored for hand counting
    max_simul_buttons: int = 2   # official charts never start more than 2 tap/hold/slide-head notes on one tick (0 = off)
    slack: float = 0.10   # seconds a hand may leave a hold / slide before its nominal end
    t_same: float = 0.06  # min seconds between two contacts of one hand at the same spot (jack)
    v_max: float = 45.0   # max hand speed between two button contacts (screen radii per second)
    dyn_touch: bool = False   # also apply the speed rule to pairs involving touch sensors
    def span(self, c1, c2):
        return (self.sbb, self.sbt, self.stt)[c1 + c2]
    def tmin(self, d):
        if d < 0.01: return self.t_same
        return max(d / self.v_max, 0.0)
    @property
    def tmax(self):
        return max(self.t_same, 2.0 / self.v_max)

DEFAULT = Params()

# ----------------------------------------------------------------------------- contacts
class Contact:
    __slots__ = ("t0", "t1", "s0", "s1", "e", "cls", "kind", "x", "y", "poly", "cum", "sustained", "ref")
    def __init__(self, t0, t1, cls, kind, x, y, poly=None, ref=None, cum=None):
        self.t0, self.t1, self.cls, self.kind, self.x, self.y, self.poly, self.ref = t0, t1, cls, kind, x, y, poly, ref
        self.sustained = kind in ("hold", "thold", "arm")
        self.cum = None
        if poly is not None:
            if cum is None:
                cum = [0.0]
                for i in range(len(poly) - 1):
                    cum.append(cum[-1] + math.dist(poly[i], poly[i + 1]))
            self.cum = cum
    def pos(self, s):
        poly = self.poly
        if poly is None:
            return self.x, self.y
        if self.s1 <= self.s0 or s <= self.s0:
            return poly[0]
        if s >= self.s1:
            return poly[-1]
        tot = self.cum[-1]
        if tot <= 0:
            return poly[0]
        d = (s - self.s0) / (self.s1 - self.s0) * tot
        cum = self.cum
        i = 1
        while i < len(cum) - 1 and cum[i] < d:
            i += 1
        l = cum[i] - cum[i - 1]
        u = (d - cum[i - 1]) / l if l > 0 else 0.0
        p0, p1 = poly[i - 1], poly[i]
        return p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1])
    def label(self):
        if self.cls == 1:
            return SENSORS[self.ref[1]]
        return str(self.ref[1])

def note_contacts(n):
    """contacts (without seconds) of one tick note"""
    t = int(n.beat)
    k = n.kind
    if k == "tap":
        x, y = BTN_XY[n.pos]; return [Contact(t, t, 0, "tap", x, y, ref=(n, n.pos))]
    if k == "touch":
        x, y = SEN_XY[n.pos]; return [Contact(t, t, 1, "touch", x, y, ref=(n, n.pos))]
    if k == "hold":
        x, y = BTN_XY[n.pos]; return [Contact(t, t + max(1, int(n.dur)), 0, "hold", x, y, ref=(n, n.pos))]
    if k == "thold":
        x, y = SEN_XY[n.pos]; return [Contact(t, t + max(1, int(n.dur)), 1, "thold", x, y, ref=(n, n.pos))]
    out = []
    if not n.headless:
        x, y = BTN_XY[n.pos]; out.append(Contact(t, t, 0, "head", x, y, ref=(n, n.pos)))
    for a in n.arms:
        poly = arm_waypoints(n.pos, a.segs)
        t0 = t + int(a.wait); t1 = t0 + max(1, int(a.dur))
        out.append(Contact(t0, t1, 0, "arm", poly[0][0], poly[0][1], poly=poly, ref=(n, n.pos),
                           cum=_arm_cum(n.pos, a.segs, poly)))
    return out

def _set_seconds(cs, tempo, P):
    if not cs:
        return
    sl = P.slack
    scalar = getattr(tempo, "sec_scalar", None)
    if scalar is not None:
        # same IEEE operations as the vectorised branch below, without numpy call overhead
        for c in cs:
            c.s0 = s0 = scalar(c.t0 / TPB)
            c.s1 = s1 = scalar(c.t1 / TPB)
            c.e = max(s0, s1 - sl) if c.sustained else s0
        return
    t = np.empty(2 * len(cs))
    for i, c in enumerate(cs):
        t[2 * i] = c.t0; t[2 * i + 1] = c.t1
    s = tempo.to_sec(t / TPB)
    for i, c in enumerate(cs):
        c.s0 = float(s[2 * i]); c.s1 = float(s[2 * i + 1])
        c.e = max(c.s0, c.s1 - sl) if c.sustained else c.s0

def build_contacts(notes, tempo, P=DEFAULT):
    cs = []
    for n in notes:
        cs += note_contacts(n)
    _set_seconds(cs, tempo, P)
    cs.sort(key=lambda c: c.t0)
    return cs

# ----------------------------------------------------------------------------- static two-hand cover (statistics)
def hands_needed(xs, ys, cls, P):
    """minimal number of hands for simultaneous points (2-colouring of the incompatibility graph; >2 -> greedy bound)"""
    n = len(xs)
    if n == 1:
        return 1
    inc = [[] for _ in range(n)]
    any_inc = False
    for i in range(n):
        for j in range(i + 1, n):
            sp = P.span(cls[i], cls[j])
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx * dx + dy * dy > sp * sp + 1e-9:
                inc[i].append(j); inc[j].append(i); any_inc = True
    if not any_inc:
        return 1
    col = [-1] * n
    ok = True
    for s0 in range(n):
        if col[s0] != -1: continue
        col[s0] = 0; st = [s0]
        while st and ok:
            u = st.pop()
            for v in inc[u]:
                if col[v] == -1:
                    col[v] = 1 - col[u]; st.append(v)
                elif col[v] == col[u]:
                    ok = False; break
        if not ok: break
    if ok:
        return 2
    c = [-1] * n
    for u in sorted(range(n), key=lambda x: -len(inc[x])):
        used = {c[v] for v in inc[u] if c[v] != -1}
        k = 0
        while k in used: k += 1
        c[u] = k
    return max(c) + 1

# ----------------------------------------------------------------------------- conflict graph with parity union-find
class HandGraph:
    """union-find with parity (no path compression -> cheap rollback). add_edge(a,b): a and b must be on different hands."""
    def __init__(self):
        self.parent = []; self.par = []; self.size = []; self.log = []
    def copy(self):
        o = HandGraph(); o.parent = list(self.parent); o.par = list(self.par); o.size = list(self.size); o.log = []
        return o
    def add_node(self):
        i = len(self.parent); self.parent.append(i); self.par.append(0); self.size.append(1); return i
    def pop_node(self):
        self.parent.pop(); self.par.pop(); self.size.pop()
    def find(self, x):
        p = 0
        parent = self.parent; par = self.par
        while parent[x] != x:
            p ^= par[x]; x = parent[x]
        return x, p
    def add_edge(self, a, b):
        """returns True if consistent (edge added/implied), False if it would close an odd cycle (edge NOT added)"""
        ra, pa = self.find(a); rb, pb = self.find(b)
        if ra == rb:
            return pa != pb
        if self.size[ra] > self.size[rb]:
            ra, rb = rb, ra; pa, pb = pb, pa
        self.parent[ra] = rb; self.par[ra] = pa ^ pb ^ 1; self.size[rb] += self.size[ra]
        self.log.append(ra)
        return True
    def mark(self):
        return len(self.log)
    def rollback(self, m):
        while len(self.log) > m:
            ra = self.log.pop(); rb = self.parent[ra]
            self.size[rb] -= self.size[ra]; self.parent[ra] = ra; self.par[ra] = 0

def conflict(a, b, P):
    """None, or ('static'|'dynamic', dist, gap) if contacts a and b cannot be played by the same hand"""
    if a.ref[0] is b.ref[0]:
        return None                       # head and arms of one slide: same finger
    if a.s0 > b.s0 or (a.s0 == b.s0 and a.t0 > b.t0):
        a, b = b, a
    sj = b.s0
    xj, yj = b.x, b.y
    if a.e >= sj - 1e-9 and (a.sustained or a.s0 == sj or a.e > sj + 1e-9):
        # overlap in time (a still active at the start of b, or same instant)
        px, py = a.pos(sj)
        sp = P.span(a.cls, b.cls)
        d = math.hypot(px - xj, py - yj)
        if d > sp + 1e-9:
            return ("static", d, 0.0)
        return None
    gap = sj - a.e
    if gap >= P.tmax - 1e-9 or ((a.cls or b.cls) and not P.dyn_touch):
        return None
    px, py = a.pos(a.e) if a.poly is not None else (a.x, a.y)
    d = math.hypot(px - xj, py - yj)
    if gap < P.tmin(d) - 1e-9:
        return ("dynamic", d, gap)
    return None

def _viol(kind, a, b, d, gap):
    if a.s0 > b.s0: a, b = b, a
    return dict(tick=b.t0, kind="hands" if kind == "static" else "jump", n=3, dist=d, dt=gap,
                detail=[(a.kind, a.label(), a.t0), (b.kind, b.label(), b.t0)])

def _slap_filter(cs, P):
    """drop instant touches of ticks that hold >= slap_n touches"""
    if not P.slap_n:
        return cs
    cnt = {}
    for c in cs:
        if c.kind == "touch":
            cnt[c.t0] = cnt.get(c.t0, 0) + 1
    if not any(v >= P.slap_n for v in cnt.values()):
        return cs
    return [c for c in cs if not (c.kind == "touch" and cnt[c.t0] >= P.slap_n)]

def check(notes, tempo, P=DEFAULT, contacts=None, stats=True):
    """Full chart check. notes: canonical tick notes (tempo: NpTempo).
    Returns dict(violations=[dict(tick, kind 'hands'|'jump', ...)], n_viol, n_hands_viol, n_jump_viol,
    per_bar (violations per 4-beat bar), n_events, and (stats=True) per-event arrays ticks, hands_needed, n_active,
    n_sus (sustained contacts running before the event), anchored; n_forced = events where a new instant contact can
    only be played by the free hand while the other one is anchored, fh = (dt, dist) of consecutive such events)."""
    cs = contacts if contacts is not None else build_contacts(notes, tempo, P)
    cs = _slap_filter(cs, P)
    g = HandGraph()
    for _ in cs:
        g.add_node()
    viol = []
    win = []
    tmax = P.tmax
    n_edges = [0, 0]
    for j, cj in enumerate(cs):
        sj = cj.s0
        lim = sj - tmax - 1e-9
        win = [i for i in win if cs[i].e > lim]
        for i in win:
            r = conflict(cs[i], cj, P)
            if r is None:
                continue
            n_edges[r[0] == "dynamic"] += 1
            if not g.add_edge(i, j):
                viol.append(_viol(r[0], cs[i], cj, r[1], r[2]))
        win.append(j)
    if P.max_simul_buttons:
        nb = {}
        for c in cs:
            if c.kind in ("tap", "head", "hold"):
                nb.setdefault(c.t0, []).append(c)
        for t, lst in nb.items():
            if len(lst) > P.max_simul_buttons:
                viol.append(dict(tick=t, kind="hands", n=len(lst), dist=0.0, dt=0.0, rule="max_simul_buttons",
                                 detail=[(c.kind, c.label(), c.t0) for c in lst]))
        viol.sort(key=lambda v: v["tick"])
    ticks, hn, na, nsus = [], [], [], []
    n_forced = 0; fh = []; last_forced = None
    if stats:
        i = 0; N = len(cs); active = []
        while i < N:
            t = cs[i].t0
            k = i
            while k < N and cs[k].t0 == t:
                k += 1
            s = cs[i].s0
            old = [c for c in active if c.e > s + 1e-9]
            mem = old + cs[i:k]
            if len(mem) == 1:
                h = 1
            else:
                pos = [c.pos(s) for c in mem]
                h = hands_needed([p[0] for p in pos], [p[1] for p in pos], [c.cls for c in mem], P)
            sus_old = [c for c in old if c.sustained]
            ticks.append(t); hn.append(h); na.append(len(mem)); nsus.append(len(sus_old))
            if sus_old:
                fp = []
                for c in cs[i:k]:
                    if c.sustained:
                        continue
                    ok = True
                    for o in sus_old:
                        ox, oy = o.pos(s)
                        sp = P.span(c.cls, o.cls)
                        if (c.x - ox) ** 2 + (c.y - oy) ** 2 <= sp * sp + 1e-9:
                            ok = False; break
                    if ok:
                        fp.append((c.x, c.y))
                if fp:       # a new instant contact that only the free hand can play
                    n_forced += 1
                    if last_forced is not None:
                        fh.append((s - last_forced[0], min(math.dist(a, b) for a in last_forced[1] for b in fp)))
                    last_forced = (s, fp)
            else:
                last_forced = None
            active = old + [c for c in cs[i:k] if c.sustained]
            i = k
    last = max([c.t1 for c in cs] + [0])
    nbar = last // (4 * TPB) + 1
    per_bar = np.zeros(nbar, np.int32)
    for v in viol:
        per_bar[min(nbar - 1, v["tick"] // (4 * TPB))] += 1
    nh = sum(1 for v in viol if v["kind"] == "hands")
    out = dict(violations=viol, n_viol=len(viol), n_hands_viol=nh, n_jump_viol=len(viol) - nh, per_bar=per_bar,
               n_events=len(ticks) if stats else 0, n_static_edges=n_edges[0], n_dyn_edges=n_edges[1])
    if stats:
        nsus = np.array(nsus, np.int16)
        out.update(ticks=np.array(ticks, np.int32), hands_needed=np.array(hn, np.int8), n_active=np.array(na, np.int16),
                   n_sus=nsus, anchored=nsus > 0, n_forced=n_forced, fh=np.array(fh, np.float64).reshape(-1, 2))
    return out

# ----------------------------------------------------------------------------- incremental use (decoding)
class HandTracker:
    """Incremental hand-feasibility state. Feed notes in non-decreasing tick order (any order inside a tick).
        tr = HandTracker(tempo)
        ok = tr.can_add(note)       # would adding `note` make the chart unplayable? (no state change)
        tr.add(note)                # register note (returns False if it introduced a violation; the offending
                                    # conflict edge is then skipped so that later checks stay meaningful)
    Only contacts of the last few seconds (+ all still-running holds/slides) are searched for conflicts."""
    def __init__(self, tempo, P=DEFAULT):
        self.tempo = tempo; self.P = P
        self.g = HandGraph()
        self.cs = []            # registered recent contacts: (node_id, contact)
        self.n_viol = 0
        self.last_tick = -10 ** 9

    def copy(self):
        o = HandTracker.__new__(HandTracker)
        o.tempo = self.tempo; o.P = self.P; o.g = self.g.copy(); o.cs = list(self.cs); o.n_viol = self.n_viol
        o.last_tick = self.last_tick
        return o

    def _prune(self, s_now):
        lim = s_now - 4.0
        if self.cs and any(c.e < lim for _, c in self.cs):
            self.cs = [(i, c) for i, c in self.cs if c.e >= lim]

    def _try(self, note, commit):
        P = self.P
        new = note_contacts(note)
        _set_seconds(new, self.tempo, P)
        g = self.g
        m = g.mark(); n0 = len(g.parent)
        ids = [g.add_node() for _ in new]
        bad = []
        tmax = P.tmax
        if P.max_simul_buttons and note.kind in ("tap", "hold", "slide") and not (note.kind == "slide" and note.headless):
            tk = int(note.beat)
            nb = 1 + sum(1 for _, c in self.cs if c.t0 == tk and c.kind in ("tap", "head", "hold"))
            if nb > P.max_simul_buttons:
                bad.append(dict(tick=tk, kind="hands", n=nb, dist=0.0, dt=0.0, rule="max_simul_buttons",
                                detail=[(note.kind, str(note.pos), tk)]))
        if note.kind == "touch" and P.slap_n:
            nt = sum(1 for _, c in self.cs if c.kind == "touch" and c.t0 == int(note.beat))
            if nt >= P.slap_n - 1:      # palm slap: no conflicts for the 8th and later simultaneous touch
                new_iter = []
            else:
                new_iter = list(zip(ids, new))
        else:
            new_iter = list(zip(ids, new))
        for a, ca in new_iter:
            for i, ci in self.cs:
                if ci.e <= ca.s0 - tmax - 1e-9 or ci.s0 >= ca.e + tmax + 1e-9:
                    continue
                r = conflict(ci, ca, P)
                if r is not None and not g.add_edge(i, a):
                    bad.append(_viol(r[0], ci, ca, r[1], r[2]))
        if commit:
            self.cs += list(zip(ids, new))
            self.n_viol += len(bad)
            self.last_tick = max(self.last_tick, int(note.beat))
            # A headless slide's first contact can start several beats later.
            # Prune at its chart onset so intervening notes still see active holds.
            self._prune(float(self.tempo.to_sec(int(note.beat) / TPB)))
        else:
            g.rollback(m)
            while len(g.parent) > n0:
                g.pop_node()
        return bad

    def can_add_variants(self, notes):
        """Exactly ``[self.can_add(n) for n in notes]`` for notes that differ only in the duration of their last
        sustained contact (a hold / touch-hold, or the last duration group of a slide's last arm) -- the 48 candidate
        DT tokens of one note.  The conflicts of every other contact pair do not depend on that duration, so they are
        evaluated once and only the pairs in which the new contact starts first are re-evaluated per variant."""
        n = len(notes)
        if n < 3:
            return [self.can_add(x) for x in notes]
        P = self.P
        tmax = P.tmax
        base = notes[0]
        if base.kind not in ("hold", "thold", "slide") or any(
                x.kind != base.kind or int(x.beat) != int(base.beat) or x.pos != base.pos for x in notes):
            return [self.can_add(x) for x in notes]
        sets = []
        for x in notes:
            cs = note_contacts(x)
            _set_seconds(cs, self.tempo, P)
            sets.append(cs)
        m = len(sets[0])
        if any(len(cs) != m for cs in sets) or any(
                (a.t0, a.s0, a.cls, a.x, a.y) != (b.t0, b.s0, b.cls, b.x, b.y)
                for cs in sets[1:] for a, b in zip(sets[0], cs)):
            return [self.can_add(x) for x in notes]
        if P.max_simul_buttons and base.kind in ("hold", "slide") and not (base.kind == "slide" and base.headless):
            tk = int(base.beat)
            if 1 + sum(1 for _, c in self.cs if c.t0 == tk and c.kind in ("tap", "head", "hold")) > P.max_simul_buttons:
                return [False] * n
        g = self.g
        mark, n0 = g.mark(), len(g.parent)
        ids = [g.add_node() for _ in range(m)]
        last = m - 1                      # the contact whose end moves with the duration
        fixed, early, late = [], [], []
        for u, ca in enumerate(sets[0]):
            for i, ci in self.cs:
                if u < last or (ci.s0 < ca.s0 or (ci.s0 == ca.s0 and ci.t0 <= ca.t0)):
                    if ci.e <= ca.s0 - tmax - 1e-9 or ci.s0 >= ca.e + tmax + 1e-9:
                        continue
                    if conflict(ci, ca, P) is not None:
                        (fixed if u < last else early).append((i, ids[u]))
                else:
                    late.append((i, ci))
        out = []
        try:
            for vi in range(n):
                edges = fixed + early
                if late:
                    ca = sets[vi][last]
                    edges = edges + [(i, ids[last]) for i, ci in late
                                     if not (ci.e <= ca.s0 - tmax - 1e-9 or ci.s0 >= ca.e + tmax + 1e-9)
                                     and conflict(ci, ca, P) is not None]
                ok = True
                for a, b in edges:
                    if not g.add_edge(a, b):
                        ok = False
                        break
                g.rollback(mark)
                out.append(ok)
        finally:
            g.rollback(mark)
            while len(g.parent) > n0:
                g.pop_node()
        return out

    def violations_if_added(self, note):
        return self._try(note, False)
    def can_add(self, note):
        return not self._try(note, False)
    def add(self, note):
        return not self._try(note, True)

def can_add(state, note):
    """functional alias: state is a HandTracker"""
    return state.can_add(note)
