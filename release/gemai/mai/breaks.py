"""Musical break (BRK) timing helpers.

The generator, window reranker and the DPO Round-4 scorer all need the same
definition of a well-placed break.  Chart times are integer ticks (TPB=48), a
bar is 192 ticks and the four-bar decoder window is 768 ticks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

TPB = 48
BAR = 4 * TPB
WINDOW = 4 * BAR
TARGET_TICKS = WINDOW


def note_has_break(n) -> bool:
    if bool(getattr(n, "brk", False)):
        return True
    return any(bool(getattr(a, "brk", False)) for a in (getattr(n, "arms", None) or ()))


def break_notes(notes) -> list:
    return [n for n in (notes or ()) if note_has_break(n)]


def break_timestamps(notes) -> list[int]:
    return sorted({int(round(float(n.beat))) for n in (notes or ()) if note_has_break(n)})


def tap_chord_groups(notes):
    groups = defaultdict(list)
    for n in notes or ():
        if str(getattr(n, "kind", "")) == "tap":
            groups[int(round(float(n.beat)))].append(n)
    return {t: ns for t, ns in groups.items() if len(ns) >= 2}


def asymmetric_tap_chords(notes) -> list[int]:
    """Ticks of simultaneous tap chords whose BRK flags are mixed."""
    out = []
    for t, ns in tap_chord_groups(notes).items():
        flags = [note_has_break(n) for n in ns]
        if any(flags) and not all(flags):
            out.append(int(t))
    return out


def max_break_per_bar(notes) -> int:
    c = Counter(t // BAR for t in break_timestamps(notes))
    return int(max(c.values())) if c else 0


def break_count_per_bar(notes) -> dict[int, int]:
    return dict(Counter(t // BAR for t in break_timestamps(notes)))


def phrase_edge(tick: int, start_tick: int = 0) -> bool:
    """First beat of the four-bar phrase, or its final three beats (bar-3 cadence)."""
    rel = int(tick) - int(start_tick)
    return rel < TPB or rel >= 13 * TPB


def weak_off16(tick: int) -> bool:
    return int(tick) % 24 != 0


def _audio_rank(ticks, audio):
    if not ticks or not audio:
        return 0.0
    env = np.asarray(audio.get("onset_env", []), np.float32)
    if env.size < 2:
        return 0.0
    nf = env.size
    duration_ticks = float(audio.get("duration_ticks", TARGET_TICKS) or TARGET_TICKS)
    order = np.sort(env)
    vals = []
    for t in ticks:
        r = max(0, min(TARGET_TICKS - 1, int(round(float(t)))))
        frame = int(r / max(duration_ticks, 1e-6) * (nf - 1))
        frame = max(0, min(nf - 1, frame))
        vals.append(float(np.searchsorted(order, env[frame], side="right")) / max(1, nf))
    return float(np.mean(vals)) if vals else 0.0


def _loud_jump(ticks, audio):
    loud = list((audio or {}).get("bar_loud", []))
    if not ticks or len(loud) < 2:
        return 0.0
    vals = []
    for t in ticks:
        b = max(0, min(len(loud) - 1, int(round(float(t))) // BAR))
        others = [float(x) for i, x in enumerate(loud) if i != b]
        if others:
            vals.append(abs(float(loud[b]) - float(np.mean(others))))
    return float(np.mean(vals)) if vals else 0.0


def window_break_features(notes, start_tick: int, audio=None, total_notes=None) -> dict:
    """Break count/timing features of a four-bar candidate window."""
    notes = list(notes or ())
    breaks = break_notes(notes)
    ts = sorted({int(round(float(n.beat))) for n in breaks})
    n_all = int(total_notes if total_notes is not None else len(notes))
    rel = [t - int(start_tick) for t in ts]
    return dict(
        brk_n=float(len(breaks)),
        brk_ts_n=float(len(ts)),
        brk_rate=float(len(breaks) / max(n_all, 1)),
        brk_downbeat_frac=float(sum(1 for t in ts if t % TPB == 0) / max(len(ts), 1)),
        brk_phrase_edge_frac=float(sum(1 for t in rel if t < TPB or t >= 13 * TPB) / max(len(ts), 1)),
        brk_off16_frac=float(sum(1 for t in ts if t % 24 != 0) / max(len(ts), 1)),
        brk_asym_chord_n=float(len(asymmetric_tap_chords(notes))),
        brk_max_per_bar=float(max_break_per_bar(notes)),
        brk_audio_onset_rank=_audio_rank(rel, audio),
        brk_audio_loud_jump=_loud_jump(rel, audio),
    )


def break_f1(cand_notes, gt_notes, tol: int = 6) -> float:
    """F1 of distinct BRK onset ticks within ``tol`` ticks (one-to-one greedy)."""
    a = break_timestamps(cand_notes)
    b = break_timestamps(gt_notes)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    i = j = hit = 0
    while i < len(a) and j < len(b):
        if abs(int(a[i]) - int(b[j])) <= int(tol):
            hit += 1
            i += 1
            j += 1
        elif a[i] < b[j] - tol:
            i += 1
        else:
            j += 1
    return 2.0 * hit / (len(a) + len(b))


def count_close(cand_n: int, gt_n: int) -> float:
    """1 for an exact match, exponentially smaller when BRKs are inflated."""
    c, g = max(0, int(cand_n)), max(0, int(gt_n))
    if c == g:
        return 1.0
    return float(math.exp(-abs(c - g) / max(1.0, 0.55 * g + 1.0)))


def timing_penalty(notes, start_tick: int, gt_notes=None, audio=None) -> dict:
    """Return break timing features plus a signed penalty/score bundle.

    ``penalty`` is subtracted from a candidate score (positive is bad).
    ``alignment`` is in [0, 1]-ish and used by the DPO scorer/relation gate.
    """
    f = window_break_features(notes, start_tick, audio)
    gt = window_break_features(gt_notes or [], start_tick, audio=None)
    f1 = break_f1(notes, gt_notes or [])
    c_n, g_n = int(f["brk_n"]), int(gt["brk_n"])
    c_off, g_off = int(sum(1 for t in break_timestamps(notes) if t % 24 != 0)), int(sum(1 for t in break_timestamps(gt_notes or []) if t % 24 != 0))
    c_max, g_max = int(f["brk_max_per_bar"]), int(gt["brk_max_per_bar"])
    over = max(0, c_n - g_n - 1)
    penalty = 0.0
    penalty += 2.0 * f["brk_asym_chord_n"]
    penalty += 0.8 * max(0, int(f["brk_ts_n"]) - 6)
    for b, cnt in break_count_per_bar(notes).items():
        if int(b) != 3 and cnt > 2:
            penalty += 1.2 * (cnt - 2)
    penalty += 0.7 * max(0, c_off - max(0, g_off))
    penalty += 1.2 * max(0, c_max - max(2, g_max))
    penalty += 0.55 * over
    align = (0.52 * f1 + 0.22 * float(f["brk_phrase_edge_frac"])
             + 0.18 * float(f["brk_audio_onset_rank"]) + 0.08 * count_close(c_n, g_n))
    # A sparse but well-placed set should not lose to a dense spam set.
    if c_n > g_n + 2:
        align -= 0.18 * min(1.0, (c_n - g_n - 2) / max(1.0, g_n + 1.0))
    f.update(gt_brk_n=float(g_n), gt_brk_off16_n=float(g_off), gt_brk_max_per_bar=float(g_max),
             brk_f1=float(f1), brk_count_close=float(count_close(c_n, g_n)),
             brk_penalty=float(penalty), brk_alignment=float(align))
    return f
