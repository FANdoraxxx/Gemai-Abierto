"""simai / maidata.txt parser and writer.

Internal representation:
  * time is measured in grid beats (float) from chart beat 0 (= &first seconds).
  * a TempoMap converts beats <-> seconds (piecewise constant BPM in beat domain).
  * Note objects use beats; durations in beats.
"""
from __future__ import annotations
import re, math
from dataclasses import dataclass, field
from fractions import Fraction

TPB = 48  # ticks per beat used by tokenizer / writer

# ---------------------------------------------------------------- tempo map
class TempoMap:
    def __init__(self, changes, first=0.0):
        # changes: list of (beat, bpm) sorted, first at beat 0
        ch = sorted(changes, key=lambda x: x[0])
        if not ch or ch[0][0] > 1e-9:
            ch = [(0.0, ch[0][1] if ch else 120.0)] + ch
        # dedupe same-beat entries (keep last)
        out = []
        for b, bpm in ch:
            if out and abs(out[-1][0] - b) < 1e-9:
                out[-1] = (b, bpm)
            elif out and abs(out[-1][1] - bpm) < 1e-9:
                continue
            else:
                out.append((b, bpm))
        self.beats = [b for b, _ in out]
        self.bpms = [p for _, p in out]
        self.first = first
        self.secs = [0.0]
        for i in range(1, len(out)):
            self.secs.append(self.secs[-1] + (self.beats[i] - self.beats[i - 1]) * 60.0 / self.bpms[i - 1])

    def _seg_b(self, beat):
        lo, hi = 0, len(self.beats) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.beats[mid] <= beat + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _seg_s(self, sec):
        lo, hi = 0, len(self.secs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.secs[mid] <= sec + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def bpm_at(self, beat):
        return self.bpms[self._seg_b(beat)]

    def to_sec(self, beat):
        """beat -> seconds relative to beat 0 (add self.first for absolute audio time)"""
        i = self._seg_b(beat)
        return self.secs[i] + (beat - self.beats[i]) * 60.0 / self.bpms[i]

    def to_beat(self, sec):
        i = self._seg_s(sec)
        return self.beats[i] + (sec - self.secs[i]) * self.bpms[i] / 60.0

    def changes(self):
        return list(zip(self.beats, self.bpms))

# ---------------------------------------------------------------- notes
SENSORS = [f"A{i}" for i in range(1, 9)] + [f"B{i}" for i in range(1, 9)] + ["C"] + \
          [f"D{i}" for i in range(1, 9)] + [f"E{i}" for i in range(1, 9)]
SENSOR_IDX = {s: i for i, s in enumerate(SENSORS)}

# canonical slide shapes (direction made absolute)
SHAPES = ["STRAIGHT", "CW", "CCW", "VEE", "P", "Q", "PP", "QQ", "S", "Z", "FAN", "GRANDV"]

@dataclass
class Seg:
    shape: str          # one of SHAPES
    end: int            # 1..8
    via: int = 0        # for GRANDV

@dataclass
class Arm:
    wait: float                          # beats between head and start of movement
    segs: list                           # list[Seg]
    groups: list                         # list of (nseg, dur_beats): durations covering consecutive segs
    brk: bool = False
    ex: bool = False

    @property
    def dur(self):
        return sum(d for _, d in self.groups)

@dataclass
class Note:
    kind: str            # tap | hold | slide | touch | thold
    beat: float
    pos: int             # button 1..8 or sensor index 0..32
    brk: bool = False
    ex: bool = False
    fw: bool = False     # touch firework
    dur: float = 0.0     # hold / touch-hold duration (beats)
    headless: bool = False
    arms: list = field(default_factory=list)

    def end_beat(self):
        if self.kind in ("hold", "thold"):
            return self.beat + self.dur
        if self.kind == "slide":
            return max(self.beat + a.wait + a.dur for a in self.arms)
        return self.beat

def is_upper(b):
    return b in (1, 2, 7, 8)

def arc_dir(shape_ch, start):
    """convert simai '<' '>' '^' to CW/CCW"""
    raise NotImplementedError

# ---------------------------------------------------------------- parsing
class SimaiError(Exception):
    pass

_SHAPE_RE = re.compile(r"pp|qq|V|[-<>^vpqszw]")

def _parse_bracket(txt, bpm):
    """returns (wait_sec or None, dur_sec)"""
    t = txt.strip()
    if "##" in t:
        w, d = t.split("##", 1)
        wait = float(w) if w else None
        if ":" in d:
            a, b = d.split(":")
            dur = 4.0 * float(b) / float(a) * 60.0 / bpm
        else:
            dur = float(d)
        return wait, dur
    if "#" in t:
        p, d = t.split("#", 1)
        lbpm = float(p) if p else bpm
        if ":" in d:
            a, b = d.split(":")
            dur = 4.0 * float(b) / float(a) * 60.0 / lbpm
        else:
            dur = float(d)
        return (60.0 / lbpm if p else None), dur
    a, b = t.split(":")
    return None, 4.0 * float(b) / float(a) * 60.0 / bpm


def _shape_canon(ch, start, end):
    if ch == "-":
        return "STRAIGHT"
    if ch in "<>":
        up = is_upper(start)
        cw = (ch == ">") == up
        return "CW" if cw else "CCW"
    if ch == "^":
        d = (end - start) % 8
        if d == 0 or d == 4:
            return "CW"
        return "CW" if d < 4 else "CCW"
    return {"v": "VEE", "p": "P", "q": "Q", "pp": "PP", "qq": "QQ", "s": "S", "z": "Z", "w": "FAN", "V": "GRANDV"}[ch]


def _parse_arm(s, start, bpm, beat, tm):
    """parse arm text like '-4[8:1]' or '-4-7[4:1]b' ; returns Arm"""
    i = 0
    segs, groups, pending = [], [], 0
    brk = ex = False
    wait_beats = None
    cur = start
    while i < len(s):
        m = _SHAPE_RE.match(s, i)
        if not m:
            raise SimaiError(f"bad slide arm '{s}' at {i}")
        ch = m.group(0); i = m.end()
        via = 0
        if ch == "V":
            via = int(s[i]); i += 1
        if i >= len(s) or not s[i].isdigit():
            raise SimaiError(f"slide missing end '{s}'")
        end = int(s[i]); i += 1
        segs.append(Seg(_shape_canon(ch, cur, end), end, via))
        pending += 1
        cur = end
        while i < len(s) and s[i] in "bx":
            if s[i] == "b": brk = True
            else: ex = True
            i += 1
        if i < len(s) and s[i] == "[":
            j = s.index("]", i)
            wsec, dsec = _parse_bracket(s[i + 1:j], bpm)
            i = j + 1
            # convert durations to beats via tempo map (relative to movement start)
            if not groups:
                hs = tm.to_sec(beat)
                w = wsec if wsec is not None else 60.0 / bpm
                wait_beats = tm.to_beat(hs + w) - beat
                mstart_sec = hs + w
            else:
                mstart_sec = tm.to_sec(beat + wait_beats + sum(d for _, d in groups))
            mstart_beat = tm.to_beat(mstart_sec)
            dbeats = tm.to_beat(mstart_sec + dsec) - mstart_beat
            groups.append((pending, dbeats)); pending = 0
            while i < len(s) and s[i] in "bx":
                if s[i] == "b": brk = True
                else: ex = True
                i += 1
    if pending:
        if not groups:
            raise SimaiError(f"slide without duration '{s}'")
        n, d = groups[-1]
        groups[-1] = (n + pending, d)
    return Arm(wait_beats, segs, groups, brk, ex)


def _parse_note(g, beat, bpm, tm):
    """parse one note group element (no '/'), returns list[Note]"""
    g = g.strip()
    if not g:
        return []
    c0 = g[0]
    # touch
    if c0 in "ABCDE":
        m = re.match(r"^([ABDE][1-8]|C[12]?)([hf]*)(\[[^\]]*\])?([hf]*)$", g)
        if not m:
            raise SimaiError(f"bad touch '{g}'")
        sname = m.group(1)
        if sname.startswith("C"): sname = "C"
        mods = m.group(2) + m.group(4)
        fw = "f" in mods
        if "h" in mods:
            if not m.group(3):
                raise SimaiError(f"touch hold w/o duration '{g}'")
            _, dsec = _parse_bracket(m.group(3)[1:-1], bpm)
            hs = tm.to_sec(beat)
            dur = tm.to_beat(hs + dsec) - beat
            return [Note("thold", beat, SENSOR_IDX[sname], fw=fw, dur=dur)]
        return [Note("touch", beat, SENSOR_IDX[sname], fw=fw)]
    if not c0.isdigit():
        raise SimaiError(f"bad note '{g}'")
    # multi-digit shorthand taps e.g. '18'
    if re.fullmatch(r"[1-8]{2,}", g):
        return [Note("tap", beat, int(c)) for c in g]
    btn = int(c0)
    if not 1 <= btn <= 8:
        raise SimaiError(f"bad button '{g}'")
    i = 1
    mods = ""
    while i < len(g) and g[i] in "bx$@?!hf":
        mods += g[i]; i += 1
    rest = g[i:]
    brk = "b" in mods; ex = "x" in mods
    if rest and _SHAPE_RE.match(rest):
        headless = "?" in mods or "!" in mods
        arms = []
        for k, a in enumerate(rest.split("*")):
            if not a:
                continue
            arms.append(_parse_arm(a, btn, bpm, beat, tm))
        return [Note("slide", beat, btn, brk=brk, ex=ex, headless=headless, arms=arms)]
    if "h" in mods:
        m = re.match(r"^(\[[^\]]*\])?([bx]*)$", rest)
        if not m:
            raise SimaiError(f"bad hold '{g}'")
        if m.group(2):
            brk = brk or "b" in m.group(2); ex = ex or "x" in m.group(2)
        if not m.group(1):
            return [Note("tap", beat, btn, brk=brk, ex=ex)]
        _, dsec = _parse_bracket(m.group(1)[1:-1], bpm)
        hs = tm.to_sec(beat)
        dur = tm.to_beat(hs + dsec) - beat
        return [Note("hold", beat, btn, brk=brk, ex=ex, dur=dur)]
    if rest:
        m = re.fullmatch(r"[bx$@]*", rest)
        if not m:
            raise SimaiError(f"bad tap '{g}'")
        brk = brk or "b" in rest; ex = ex or "x" in rest
    return [Note("tap", beat, btn, brk=brk, ex=ex)]


def _strip_comments(body):
    return "\n".join(l.split("||")[0] for l in body.split("\n"))


def parse_chart(body, default_bpm=120.0, first=0.0):
    """parse an inote body. Returns (notes, TempoMap)."""
    body = _strip_comments(body)
    body = re.sub(r"\s+", "", body)
    tokens = body.split(",")
    # pass 1: tempo map (beats advance per comma by 4/div at current bpm)
    def walk(callback):
        beat = 0.0; bpm = None; div = 4.0; secdiv = None
        changes = []
        for tok in tokens:
            t = tok
            while True:
                m = re.match(r"^\(([\d.]+)\)", t)
                if m:
                    bpm = float(m.group(1)); changes.append((beat, bpm)); t = t[m.end():]; continue
                m = re.match(r"^\{(#?)([\d.]+)\}", t)
                if m:
                    if m.group(1):
                        secdiv = float(m.group(2))
                    else:
                        div = float(m.group(2)); secdiv = None
                    t = t[m.end():]; continue
                break
            if bpm is None:
                bpm = default_bpm; changes.append((0.0, bpm))
            if callback and t and t != "E":
                callback(t, beat, bpm)
            if secdiv is not None:
                beat += secdiv * bpm / 60.0
            else:
                beat += 4.0 / div
        return changes, beat
    changes, total = walk(None)
    tm = TempoMap(changes, first)
    notes = []
    def cb(t, beat, bpm):
        if t == "E":
            return
        for g in re.split(r"[/`]", t):
            if g == "E":
                continue
            notes.extend(_parse_note(g, beat, bpm, tm))
    walk(cb)
    return notes, tm


def read_maidata(path):
    with open(path, encoding="utf-8-sig", errors="replace") as stream:
        raw = stream.read()
    raw = raw.replace("\r\n", "\n")
    meta, charts = {}, {}
    for m in re.finditer(r"^&([^=\n]+)=(.*?)(?=\n&|\Z)", raw, re.S | re.M):
        k, v = m.group(1).strip(), m.group(2)
        if k.startswith("inote_"):
            charts[int(k[6:])] = v
        else:
            meta[k] = v.strip()
    return meta, charts


# ---------------------------------------------------------------- writing
DIVS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]

def _frac_str(ticks):
    """ticks (at TPB per beat) -> 'a:b' measure fraction"""
    g = math.gcd(int(ticks), 4 * TPB)
    return f"{4 * TPB // g}:{int(ticks) // g}"

def _fmt_sec(x):
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _shape_out(seg, cur):
    sh = seg.shape
    if sh == "STRAIGHT": return f"-{seg.end}"
    if sh in ("CW", "CCW"):
        up = is_upper(cur)
        ch = ">" if (sh == "CW") == up else "<"
        return f"{ch}{seg.end}"
    if sh == "GRANDV": return f"V{seg.via}{seg.end}"
    return {"VEE": "v", "P": "p", "Q": "q", "PP": "pp", "QQ": "qq", "S": "s", "Z": "z", "FAN": "w"}[sh] + str(seg.end)

def note_to_simai(n, tm):
    """n uses tick units (beat, dur, wait in ticks as ints)."""
    def const_bpm(t0, t1):
        b0 = tm.bpm_at(t0 / TPB)
        i0 = tm._seg_b(t0 / TPB); i1 = tm._seg_b(max(t0, t1 - 1e-6) / TPB)
        return i0 == i1, b0
    def secs(t0, t1):
        return tm.to_sec(t1 / TPB) - tm.to_sec(t0 / TPB)
    if n.kind in ("touch", "thold"):
        s = SENSORS[n.pos]
        if n.kind == "thold":
            s += "h"
            if n.fw: s += "f"
            ok, _ = const_bpm(n.beat, n.beat + n.dur)
            s += f"[{_frac_str(n.dur)}]" if ok else f"[#{_fmt_sec(secs(n.beat, n.beat + n.dur))}]"
        elif n.fw:
            s += "f"
        return s
    s = str(n.pos)
    if n.brk: s += "b"
    if n.ex: s += "x"
    if n.kind == "tap":
        return s
    if n.kind == "hold":
        ok, _ = const_bpm(n.beat, n.beat + n.dur)
        return s + "h" + (f"[{_frac_str(n.dur)}]" if ok else f"[#{_fmt_sec(secs(n.beat, n.beat + n.dur))}]")
    # slide
    if n.headless: s += "?"
    arms_txt = []
    for a in n.arms:
        t0 = n.beat; tw = n.beat + a.wait; te = tw + a.dur
        ok, bpm = const_bpm(t0, te)
        default_wait = abs(a.wait - TPB) < 1e-6
        txt = ""; cur = n.pos; k = 0
        if ok and default_wait:
            for (ns, d) in a.groups:
                for sg in a.segs[k:k + ns]:
                    txt += _shape_out(sg, cur); cur = sg.end
                k += ns
                txt += f"[{_frac_str(d)}]"
        else:
            # Preserve each duration group, including across tempo changes.
            # Only the first group carries the head-to-movement wait.
            # Difference rounded cumulative endpoints to avoid per-group drift.
            movement = tw
            elapsed = round(secs(t0, tw), 4)
            for gi, (ns, d) in enumerate(a.groups):
                for sg in a.segs[k:k + ns]:
                    txt += _shape_out(sg, cur); cur = sg.end
                k += ns
                endpoint = round(secs(t0, movement + d), 4)
                duration = _fmt_sec(round(endpoint - elapsed, 4))
                txt += (f"[{_fmt_sec(elapsed)}##{duration}]" if gi == 0
                        else f"[#{duration}]")
                movement += d
                elapsed = endpoint
        if a.brk: txt += "b"
        arms_txt.append(txt)
    return s + "*".join(arms_txt)


def write_chart(notes, tm, end_pad_beats=4):
    """notes in tick units. returns inote body string."""
    by_t = {}
    for n in notes:
        by_t.setdefault(int(n.beat), []).append(n)
    bpm_t = {int(round(b * TPB)): p for b, p in tm.changes()}
    last = max([int(n.end_beat()) for n in notes] + [0] + list(bpm_t.keys()))
    nmeas = last // (4 * TPB) + 1 + end_pad_beats // 4
    lines = []
    cur_div = None
    for mi in range(nmeas):
        m0 = mi * 4 * TPB
        evs = [t for t in list(by_t.keys()) + list(bpm_t.keys()) if m0 <= t < m0 + 4 * TPB]
        div = 1
        for d in DIVS:
            step = 4 * TPB // d
            if all((t - m0) % step == 0 for t in evs):
                div = d; break
        else:
            div = 192
        step = 4 * TPB // div
        parts = []
        for k in range(div):
            t = m0 + k * step
            cell = ""
            if k == 0 and div != cur_div:
                cell += f"{{{div}}}"
            if t in bpm_t:
                b = bpm_t[t]
                cell = (f"({_fmt_sec(b)})" + cell)
            if t in by_t:
                cell += "/".join(note_to_simai(n, tm) for n in by_t[t])
            parts.append(cell)
        cur_div = div
        lines.append(",".join(parts) + ",")
    lines.append("E")
    return "\n".join(lines)


# ---------------------------------------------------------------- quantization
import copy

def quantize(notes, tpb=TPB):
    """returns (tick_notes, max_err_ticks). Beats/durations become int ticks."""
    out = []; err = 0.0
    for n in notes:
        q = copy.deepcopy(n)
        b = n.beat * tpb; q.beat = int(round(b)); err = max(err, abs(b - q.beat))
        if n.kind in ("hold", "thold"):
            e = (n.beat + n.dur) * tpb
            q.dur = max(1, int(round(e)) - q.beat)
        if n.kind == "slide":
            for a, qa in zip(n.arms, q.arms):
                ws = (n.beat + a.wait) * tpb
                qa.wait = max(0, int(round(ws)) - q.beat)
                acc = ws; qacc = q.beat + qa.wait; g2 = []
                for ns, d in a.groups:
                    acc += d * tpb
                    e = int(round(acc))
                    g2.append((ns, max(1, e - qacc))); qacc = e
                qa.groups = g2
        out.append(q)
    return out, err


def canon_key(n):
    """hashable canonical description of a tick note (for comparisons / dedupe)"""
    base = (n.kind, int(n.beat), n.pos, n.brk, n.ex, n.fw, int(n.dur), n.headless)
    arms = tuple((a.wait, tuple((s.shape, s.end, s.via) for s in a.segs), tuple(a.groups), a.brk) for a in n.arms)
    return base + (arms,)


def load_song(folder):
    import os
    meta, charts = read_maidata(os.path.join(folder, "maidata.txt"))
    first = float(meta.get("first", "0") or 0)
    wbpm = float(meta.get("wholebpm", "120") or 120)
    res = {}
    for k, body in charts.items():
        notes, tm = parse_chart(body, wbpm, first)
        res[k] = (notes, tm)
    return meta, res
