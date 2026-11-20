"""Which slide segments exist in the game.

A segment is (start button, canonical shape, end button[, via button for GRANDV]); the start of a segment is the
head button for the first segment of an arm and the end of the previous segment for chained (festival) segments.
The table of the game's slide data (scratch/muri/slide_valid_canon.json, generated with the MaiMuriDX parser and
verified against the 266,884 segments of all non-utage official charts: 0 of them invalid) is rotation invariant, so it
reduces to the set of allowed offsets d = (end - start) mod 8 per shape:

    STRAIGHT       d in 2..6                      (no straight slide to a neighbour key)
    VEE            d in {1, 2, 3, 5, 6, 7}        (no V to the same or the opposite key)
    S, Z, FAN      d == 4                         (and a FAN / wifi segment cannot be chained with another segment)
    CW CCW P Q PP QQ   any d (0..7)
    GRANDV         via offset (via - start) mod 8 in {2, 6}; end offset from the via key (end - via) mod 8 in 2..5 when the
                   via offset is 2 and in 3..6 when it is 6

tools/test_slidetable.py checks these rules against the json table and against every official segment.
"""
from __future__ import annotations

ALL = tuple(range(8))
SHAPE_OFFSETS = {
    "STRAIGHT": (2, 3, 4, 5, 6),
    "VEE": (1, 2, 3, 5, 6, 7),
    "S": (4,), "Z": (4,), "FAN": (4,),
    "CW": ALL, "CCW": ALL, "P": ALL, "Q": ALL, "PP": ALL, "QQ": ALL,
}
GRANDV_VIA_OFFSETS = (2, 6)
GRANDV_END_OFFSETS = {2: (2, 3, 4, 5), 6: (3, 4, 5, 6)}
NO_CHAIN = frozenset({"FAN"})          # wifi slides cannot be chained (parser: "wifi slide does not support chaining yet")


def _btn(start: int, offset: int) -> int:
    return (start - 1 + offset) % 8 + 1


def segment_ends(shape: str, start: int) -> tuple:
    """Valid end buttons (1..8) of a non-GRANDV segment from ``start``."""
    return tuple(_btn(start, d) for d in SHAPE_OFFSETS[shape])


def grandv_vias(start: int) -> tuple:
    return tuple(_btn(start, d) for d in GRANDV_VIA_OFFSETS)


def grandv_ends(start: int, via: int) -> tuple:
    """Valid end buttons of a GRANDV segment (empty when ``via`` is not a valid via key)."""
    v = (via - start) % 8
    return tuple(_btn(via, d) for d in GRANDV_END_OFFSETS.get(v, ()))


def is_valid(start: int, shape: str, end: int, via: int = 0) -> bool:
    if shape == "GRANDV":
        return end in grandv_ends(start, via)
    return end in segment_ends(shape, start)


def arm_problems(start: int, segs) -> list:
    """Invalid segments of a slide arm (``segs`` = iterable of objects with shape / end / via): list of strings."""
    out, cur = [], start
    segs = list(segs)
    for i, s in enumerate(segs):
        if not is_valid(cur, s.shape, s.end, getattr(s, "via", 0) or 0):
            out.append(f"{cur}:{s.shape}:{s.end}:{getattr(s, 'via', 0) or 0}")
        if s.shape in NO_CHAIN and len(segs) > 1:
            out.append(f"chained {s.shape}")
        cur = s.end
    return out


def note_problems(note) -> list:
    if note.kind != "slide":
        return []
    out = []
    for a in note.arms:
        out += arm_problems(note.pos, a.segs)
    return out
