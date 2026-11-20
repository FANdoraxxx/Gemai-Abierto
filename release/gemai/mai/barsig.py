"""Bar signatures at three granularities, shared by the window reranker, the steering bias and the variety metrics.

* ``exact``   everything in the bar (offsets, kinds, absolute lanes, durations, slide arms, break / ex flags)
* ``shape``   rhythm + kinds + *relative* button motion: the same pattern played from another starting lane (or a
              rotation of it) has the same shape.  Lane steps are measured inside the bar, so the first note of a bar
              carries no step.  Holds keep their duration, slides their (shape, length) segments.
* ``rk``      rhythm + kinds only (offsets and note kinds)

Notes are canonical tick notes (``mai.simai.Note``); a bar is 4 beats = 192 ticks.
"""
from __future__ import annotations

TPB = 48
BAR = 4 * TPB


def _slide_desc(n):
    return tuple(tuple((s.shape, (int(s.end) - int(n.pos)) % 8) for s in a.segs) for a in n.arms)


def exact_signature(notes, bar_start):
    out = []
    for n in notes:
        arm = (tuple((a.wait, tuple((s.shape, s.end) for s in a.segs), tuple(a.groups)) for a in n.arms)
               if n.kind == "slide" else ())
        out.append((int(n.beat) - bar_start, n.kind, n.pos, int(n.dur), arm, n.brk, n.ex))
    return tuple(sorted(out, key=repr))


def shape_events(notes, bar_start):
    """Per-note shape keys (offset, kind, step, extra) in (tick, lane) order; step is None for touch notes and 0 for
    the first button note of the bar."""
    out, prev = [], None
    for n in sorted(notes, key=lambda x: (int(x.beat), x.pos)):
        off = int(n.beat) - bar_start
        if n.kind in ("touch", "thold"):
            out.append((off, n.kind, None, int(n.dur)))
            continue
        step = 0 if prev is None else (int(n.pos) - prev) % 8
        prev = int(n.pos)
        extra = int(n.dur) if n.kind == "hold" else (_slide_desc(n) if n.kind == "slide" else 0)
        out.append((off, n.kind, step, extra))
    return tuple(out)


def shape_signature(notes, bar_start):
    return shape_events(notes, bar_start)


def rk_signature(notes, bar_start):
    return tuple(sorted((int(n.beat) - bar_start, n.kind) for n in notes))


def group_bars(notes):
    bars = {}
    for n in notes:
        bars.setdefault(int(n.beat) // BAR, []).append(n)
    return bars


def run_lengths(sigs, min_notes=2):
    """Lengths of maximal runs of equal consecutive signatures (``None`` / short bars break a run)."""
    out, r, prev = [], 0, None
    for s in sigs:
        ok = s is not None and len(s) >= min_notes
        if ok and s == prev:
            r += 1
        else:
            if r:
                out.append(r)
            r = 1 if ok else 0
        prev = s if ok else None
    if r:
        out.append(r)
    return out


def run_positions(notes, kind="shape", min_notes=2):
    """Share of non-empty bars that sit at position >= 2, 3, 4, 6 of a run of equal signatures, and the longest run."""
    fn = {"exact": exact_signature, "shape": shape_signature, "rk": rk_signature}[kind]
    bars = group_bars(notes)
    if not bars:
        return None
    sigs = [fn(bars[b], b * BAR) if b in bars else None for b in range(min(bars), max(bars) + 1)]
    runs = run_lengths(sigs, min_notes)
    nb = max(1, sum(1 for s in sigs if s))
    out = {f"{kind}_run{p}": sum(max(0, r - (p - 1)) for r in runs) / nb for p in (2, 3, 4, 6)}
    out[f"{kind}_maxrun"] = float(max(runs)) if runs else 0.0
    return out
