"""Incremental grammar for the target event stream emitted by ``encode_note``.

Only event tokens and the terminating EOS belong here; the model's condition
prefix and history are outside the grammar.  ``at_boundary`` means that EOS is
legal *now*.  Optional modifiers can still extend a complete tap/touch/slide, so
the current note is committed only when the next N_*, T_* or EOS is accepted.
``pending_note_tokens`` exposes that complete but uncommitted note without its
time token; ``note_tokens`` also exposes incomplete notes for backtracking.

This module is CPU-only and deliberately independent of playability checks.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from operator import index
from weakref import WeakKeyDictionary

import numpy as np

from . import slidetable
from .simai import SENSORS, SHAPES, TPB
from .tokens import MAX_DUR_B, MAX_WAIT, T_MAX


BAR = 4 * TPB          # seq_version 6 (interleaved BD): the 4 target bars are [0, BAR), [BAR, 2 BAR), ...


@dataclass(frozen=True)
class CompletedNote:
    """A committed note; start/end are offsets in accepted target tokens."""

    tick: int
    start: int
    end: int
    tokens: tuple[int, ...]


class _Vocabulary:
    def __init__(self, vocab):
        self.eos = vocab.EOS
        self.kinds = {
            vocab["N_TAP"]: "tap",
            vocab["N_HOLD"]: "hold",
            vocab["N_SLIDE"]: "slide",
            vocab["N_TOUCH"]: "touch",
            vocab["N_THOLD"]: "thold",
        }
        self.button_kinds = tuple(list(self.kinds)[:3])
        self.touch_kinds = tuple(list(self.kinds)[3:])
        self.buttons = tuple(vocab[f"BTN_{b}"] for b in range(1, 9))
        self.sensors = tuple(vocab[f"SEN_{s}"] for s in SENSORS)
        self.shapes = tuple(vocab[f"SH_{s}"] for s in SHAPES)
        self.grandv = vocab["SH_GRANDV"]
        self.straight = vocab["SH_STRAIGHT"]
        # WAIT_48 is never emitted by encode_note: absence means one beat.
        self.waits = tuple(vocab[f"WAIT_{w}"] for w in range(MAX_WAIT) if w != TPB)
        self.dbs = tuple(vocab[f"DB_{b}"] for b in range(MAX_DUR_B))
        self.dbset = frozenset(self.dbs)
        self.shapeset = frozenset(self.shapes)
        self.dts = tuple(vocab[f"DT_{t}"] for t in range(TPB))
        self.times = tuple(vocab.time_tok(t) for t in range(T_MAX))
        self.times_arr = np.asarray(self.times, np.int64)
        self.brk = vocab["BRK"]
        self.ex = vocab["EX"]
        self.fw = vocab["FW"]
        self.headless = vocab["HEADLESS"]
        self.sbrk = vocab["SBRK"]
        self.arm = vocab["ARM"]
        # slide-segment validity masks (mai/slidetable.py): token tuples per (shape, start button)
        self.shape_name = {vocab[f"SH_{s}"]: s for s in SHAPES}
        self.fan = vocab["SH_FAN"]
        self.chain_shapes = tuple(t for t in self.shapes if self.shape_name[t] not in slidetable.NO_CHAIN)
        btn = lambda b: self.buttons[b - 1]
        self.end_ok = {(self.shape_name[t], a): tuple(btn(b) for b in slidetable.segment_ends(self.shape_name[t], a))
                       for t in self.shapes if self.shape_name[t] != "GRANDV" for a in range(1, 9)}
        self.via_ok = {a: tuple(btn(b) for b in slidetable.grandv_vias(a)) for a in range(1, 9)}
        self.gend_ok = {(a, via): tuple(btn(b) for b in slidetable.grandv_ends(a, via))
                        for a in range(1, 9) for via in range(1, 9)}


_VOCAB_CACHE = WeakKeyDictionary()


def _feature(value, name):
    if isinstance(value, bool):
        return value
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{name} must be 'auto', 'on', or 'off'")
    return value != "off"


class EventGrammar:
    """Mask and consume canonical target tokens one at a time.

    ``max_arms=None`` disables only the generation arm cap, for validation of
    unusual official charts.  Feature ``on`` and ``auto`` both permit a feature;
    ``on`` conditions the model elsewhere and does not force its occurrence.
    """

    def __init__(self, V, *, touch="auto", ex="auto", fest="auto",
                 max_arms=3, max_tick=768, max_segments=None, chain_groups=True, slide_mask=True, bars=None):
        max_tick = index(max_tick)
        if not 1 <= max_tick <= T_MAX:
            raise ValueError(f"max_tick must be in [1, {T_MAX}]")
        if max_arms is not None:
            max_arms = index(max_arms)
            if max_arms < 1:
                raise ValueError("max_arms must be positive or None")
        if max_segments is not None:
            max_segments = index(max_segments)
            if max_segments < 1:
                raise ValueError("max_segments must be positive or None")
        self.max_segments = max_segments
        # slide_mask: only slide segments that exist in the game (mai/slidetable.py) are legal
        self.slide_mask = bool(slide_mask)
        # chain_groups=False: a duration closes an arm's only group (the official charts never continue after one)
        self.chain_groups = bool(chain_groups)
        self.V = V
        if V not in _VOCAB_CACHE:
            _VOCAB_CACHE[V] = _Vocabulary(V)
        self._v = _VOCAB_CACHE[V]
        self.touch = _feature(touch, "touch")
        self.ex = _feature(ex, "ex")
        self.fest = _feature(fest, "fest")
        self.max_arms = max_arms
        self.max_tick = max_tick
        self.cur_time = 0
        self.has_time = False
        self.finished = False
        self.token_count = 0
        self.completed_count = 0
        self.last_completed_note = None
        self.note_tokens = []
        self._note_start = 0
        self._state = "start"
        self._kind = None
        self._last_button = 0
        self._last_sensor = -1
        self._mod_phase = 0
        self._firework = False
        self._arms = 0
        self._segments = 0
        self._duration_db = 0
        # timing floors (ticks after the note start) used to prune tokens that cannot fit before a deadline
        self._wait = TPB             # wait of the current arm (default one beat)
        self._arm_dur = 0            # sum of completed group durations of the current arm
        self._prev_end = 0           # max wait+duration over completed arms
        self._hold_floor = 0
        self._seg_start = 0          # start button of the current slide segment (head button, then the previous segment's end)
        self._seg_shape = None       # shape name of the current / last slide segment
        self._seg_via = 0
        # seq_version 6 bar mode: bars = the 4 BDI token ids (forced by the plan).  Target = BDI_0 <bar 0> BARM BDI_1 <bar 1> BARM BDI_2 <bar 2> BARM BDI_3 <bar 3> EOS;
        # a bar's time tokens stay inside the bar, BARM only at a note boundary (never right after a time token) and only before bar 3, EOS only in the last bar
        # (the last bar that fits before max_tick).  bars=None -> exactly the old grammar.
        self._bdi = None if bars is None else tuple(int(x) for x in bars)
        self._bar = -1
        self._fresh = False                 # no time token yet in the current bar
        if self._bdi is not None:
            if len(self._bdi) != 4:
                raise ValueError("bars must hold 4 BDI token ids")
            self._state = "need_bdi"

    def _bar_hi(self):
        return min(self.max_tick, BAR * (self._bar + 1))

    def _eos_ok(self):
        return self._bar >= 3 or BAR * (self._bar + 1) >= self.max_tick

    def _barm_ok(self):
        return self._bar < 3 and BAR * (self._bar + 1) < self.max_tick

    def _pairs_from(self, bar):
        n = 0
        while not (bar >= 3 or BAR * (bar + 1) >= self.max_tick):
            n += 1
            bar += 1
        return n

    @property
    def state(self):
        return self._state

    @property
    def current_kind(self):
        return self._kind

    @property
    def feed_time(self):
        """cur_time of the token that was just accepted (BARM / BDI sit on the bar line, everything else on the current tick)"""
        if self._bdi is not None:
            if self._state == "need_bdi":
                return BAR * (self._bar + 1)
            if self._fresh:
                return BAR * self._bar
        return self.cur_time

    @property
    def at_boundary(self):
        if self.finished:
            return False
        return (
            self._state in ("start", "done", "slide_group_end", "slide_arm_end")
            or self._state == "head" and self._kind == "tap"
            or self._state == "touch_head" and self._kind == "touch"
        )

    @property
    def pending_note_tokens(self):
        """Complete current note (tuple), or None if absent/incomplete."""
        if self.note_tokens and self.at_boundary:
            return tuple(self.note_tokens)
        return None

    def clone(self):
        """Independent parser state; immutable vocabulary tables are shared."""
        result = copy(self)
        result.note_tokens = self.note_tokens.copy()
        return result

    def _note_kinds(self):
        v = self._v
        buttons = v.button_kinds if self._last_sensor < 0 and self._last_button < 8 else ()
        touches = v.touch_kinds if self.touch and self._last_sensor < len(SENSORS) - 1 else ()
        return buttons + touches

    def _more_arms(self):
        return self.max_arms is None or self._arms < self.max_arms

    def _more_segments(self):
        return self.max_segments is None or self._segments < self.max_segments

    def _chain_shapes(self):
        """Shapes that may start a further (connected) segment of the current arm."""
        if not (self.fest and self._more_segments()):
            return ()
        v = self._v
        if not self.slide_mask:
            return v.shapes
        return () if self._seg_shape in slidetable.NO_CHAIN else v.chain_shapes

    def _local_allowed(self):
        v = self._v
        st = self._state
        if st == "need_bdi":
            return (self._bdi[self._bar + 1],)
        if st in ("start", "done", "finished"):
            return ()
        if st == "time":
            return self._note_kinds()
        if st == "position":
            if self._kind in ("touch", "thold"):
                return v.sensors[self._last_sensor + 1:]
            return v.buttons[self._last_button:]
        if st == "head":
            mods = ()
            if self._mod_phase == 0:
                mods += (v.brk,)
            if self._mod_phase < 2 and self.ex:
                mods += (v.ex,)
            if self._kind == "tap":
                return mods
            if self._kind == "hold":
                return mods + v.dbs
            if self._mod_phase < 3:
                mods += (v.headless,)
            return mods + (v.arm,)
        if st == "touch_head":
            mods = () if self._firework else (v.fw,)
            return mods + (v.dbs if self._kind == "thold" else ())
        if st == "arm_start":
            return v.shapes + v.waits
        if st == "arm_shape":
            return v.shapes
        if st == "slide_via":
            return v.via_ok[self._seg_start] if self.slide_mask else v.buttons
        if st == "slide_end":
            if not self.slide_mask:
                return v.buttons
            if self._seg_shape == "GRANDV":
                return v.gend_ok[(self._seg_start, self._seg_via)]
            return v.end_ok[(self._seg_shape, self._seg_start)]
        if st == "segment_end":
            return v.dbs + self._chain_shapes()
        if st == "duration_tick":
            return v.dts[1:] if self._duration_db == 0 else v.dts
        if st == "slide_group_end":
            return ((self._chain_shapes() if self.chain_groups else ()) + (v.sbrk,)
                    + ((v.arm,) if self._more_arms() else ()))
        if st == "slide_arm_end":
            return (v.arm,) if self._more_arms() else ()
        raise AssertionError(f"unknown grammar state {st}")

    def allowed(self):
        """Return legal next token IDs; an empty tuple follows accepted EOS."""
        local = self._local_allowed()
        if not self.at_boundary:
            return local
        v = self._v
        if self._bdi is not None:
            local2, tail, (lo, hi) = self.allowed_parts()
            return local2 + tail + v.times[lo:hi]
        kinds = self._note_kinds() if self.has_time else ()
        first_time = self.cur_time + 1 if self.has_time else 0
        return local + (v.eos,) + kinds + v.times[first_time:self.max_tick]

    def allowed_parts(self):
        """``allowed()`` split for bulk handling of the time tokens: ``(local, tail, (lo, hi))`` such that
        ``allowed() == local + tail + times[lo:hi]`` (``hi == lo`` when no time token is legal)."""
        local = self._local_allowed()
        if not self.at_boundary:
            return local, (), (0, 0)
        v = self._v
        if self._bdi is not None:
            kinds = self._note_kinds() if (self.has_time and not self._fresh) else ()
            tail = ((v.eos,) if self._eos_ok() else ()) + ((self.V.barm,) if self._barm_ok() else ()) + kinds
            lo = BAR * self._bar if self._fresh else self.cur_time + 1
            return local, tail, (lo, max(lo, self._bar_hi()))
        kinds = self._note_kinds() if self.has_time else ()
        first_time = self.cur_time + 1 if self.has_time else 0
        return local, (v.eos,) + kinds, (first_time, max(first_time, self.max_tick))

    def _commit_note(self):
        if not self.note_tokens:
            return
        self.last_completed_note = CompletedNote(
            tick=self.cur_time, start=self._note_start, end=self.token_count,
            tokens=tuple(self.note_tokens),
        )
        self.completed_count += 1
        self.note_tokens = []
        self._kind = None

    def _start_note(self, token_id):
        self._kind = self._v.kinds[token_id]
        self.note_tokens = [token_id]
        self._note_start = self.token_count
        self._mod_phase = 0
        self._firework = False
        self._arms = 0
        self._segments = 0
        self._wait = TPB
        self._arm_dur = 0
        self._prev_end = 0
        self._state = "position"

    def _start_arm(self):
        if self._arms:
            self._prev_end = max(self._prev_end, self._wait + self._arm_dur)
        self._arms += 1
        self._segments = 0
        self._wait = TPB
        self._arm_dur = 0
        self._seg_start = self._last_button
        self._seg_shape = None
        self._seg_via = 0
        self._state = "arm_start"

    def _start_segment(self, token_id):
        self._segments += 1
        self._seg_shape = self._v.shape_name[token_id]
        self._seg_via = 0
        self._state = "slide_via" if token_id == self._v.grandv else "slide_end"

    def _reject(self, token_id):
        name = self.V.toks[token_id] if 0 <= token_id < self.V.size else repr(token_id)
        raise ValueError(
            f"illegal {name} in {self._state} at tick {self.cur_time}, "
            f"target offset {self.token_count}"
        )

    def accept(self, token_id):
        """Consume one legal ID, raising ValueError without mutation otherwise."""
        try:
            token_id = index(token_id)
        except TypeError as exc:
            raise ValueError("token_id must be an integer") from exc
        if self.finished or not 0 <= token_id < self.V.size:
            self._reject(token_id)
        v = self._v
        if self._bdi is not None:
            if token_id == self.V.barm:
                if not (self.at_boundary and self._barm_ok()):
                    self._reject(token_id)
                self._commit_note()
                self._state = "need_bdi"
                self.token_count += 1
                return
            if self._state == "need_bdi":
                if token_id != self._bdi[self._bar + 1]:
                    self._reject(token_id)
                self._bar += 1
                self._state = "done"
                self._fresh = True
                self.token_count += 1
                return
        is_time = self.V.is_time(token_id)
        is_kind = token_id in v.kinds
        if token_id == v.eos or is_time or is_kind:
            # A freshly emitted T requires a note before EOS or another T.
            if is_kind and self._state == "time" and token_id in self._note_kinds():
                self._start_note(token_id)
            elif self.at_boundary:
                if is_time:
                    tick = self.V.tok_time(token_id)
                    if not (0 <= tick < self.max_tick
                            and (not self.has_time or tick > self.cur_time)):
                        self._reject(token_id)
                    if self._bdi is not None:
                        if not (BAR * self._bar <= tick < BAR * (self._bar + 1)):
                            self._reject(token_id)
                        self._fresh = False
                    self._commit_note()
                    self.cur_time = tick
                    self.has_time = True
                    self._last_button = 0
                    self._last_sensor = -1
                    self._state = "time"
                elif is_kind:
                    if not self.has_time or token_id not in self._note_kinds() or self._fresh:
                        self._reject(token_id)
                    self._commit_note()
                    self._start_note(token_id)
                else:
                    if self._bdi is not None and not self._eos_ok():
                        self._reject(token_id)
                    self._commit_note()
                    self.finished = True
                    self._state = "finished"
            else:
                self._reject(token_id)
            self.token_count += 1
            return

        if token_id not in self._local_allowed():
            self._reject(token_id)
        st = self._state
        self.note_tokens.append(token_id)
        if st == "position":
            if self._kind in ("touch", "thold"):
                self._last_sensor = token_id - v.sensors[0]
                self._state = "touch_head"
            else:
                self._last_button = token_id - v.buttons[0] + 1
                self._state = "head"
        elif st == "head":
            if token_id == v.brk:
                self._mod_phase = 1
            elif token_id == v.ex:
                self._mod_phase = 2
            elif token_id == v.headless:
                self._mod_phase = 3
            elif token_id == v.arm:
                self._start_arm()
            else:
                self._duration_db = token_id - v.dbs[0]
                self._state = "duration_tick"
        elif st == "touch_head":
            if token_id == v.fw:
                self._firework = True
            else:
                self._duration_db = token_id - v.dbs[0]
                self._state = "duration_tick"
        elif st in ("arm_start", "arm_shape"):
            if token_id in v.shapes:
                self._start_segment(token_id)
            else:
                self._wait = token_id - v.waits[0]      # WAIT_w ids are contiguous in w
                self._state = "arm_shape"
        elif st == "slide_via":
            self._seg_via = token_id - v.buttons[0] + 1
            self._state = "slide_end"
        elif st == "slide_end":
            self._seg_start = token_id - v.buttons[0] + 1      # a connected segment starts where this one ends
            self._state = "segment_end"
        elif st == "segment_end":
            if token_id in v.shapes:
                self._start_segment(token_id)
            else:
                self._duration_db = token_id - v.dbs[0]
                self._state = "duration_tick"
        elif st == "duration_tick":
            dur = self._duration_db * TPB + (token_id - v.dts[0])
            if self._kind == "slide":
                self._arm_dur += dur
            self._state = "slide_group_end" if self._kind == "slide" else "done"
        elif st == "slide_group_end":
            if token_id == v.sbrk:
                self._state = "slide_arm_end"
            elif token_id == v.arm:
                self._start_arm()
            else:
                self._start_segment(token_id)
        elif st == "slide_arm_end":
            self._start_arm()
        else:
            raise AssertionError(f"unhandled grammar state {st}")
        self.token_count += 1

    # ------------------------------------------------------------------ timing floors
    def _floor_now(self):
        """Lower bound of the note's end (ticks after its start) over all legal completions from the current state."""
        kind, st = self._kind, self._state
        if kind in ("hold", "thold"):
            if st == "duration_tick":
                return self._duration_db * TPB + (1 if self._duration_db == 0 else 0)
            return 1
        if kind != "slide":
            return 0
        if st in ("position", "head"):
            return 1
        if st == "arm_start":
            return max(self._prev_end, 1)
        if st in ("arm_shape", "slide_via", "slide_end", "segment_end"):
            return max(self._prev_end, self._wait + self._arm_dur + 1)
        if st == "duration_tick":
            return max(self._prev_end, self._wait + self._arm_dur + self._duration_db * TPB
                       + (1 if self._duration_db == 0 else 0))
        return max(self._prev_end, self._wait + self._arm_dur)      # arm complete

    def end_floor(self, token_id):
        """Lower bound of the end offset (ticks after the note start) once ``token_id`` has been accepted.

        Exact for the tokens that fix timing (DB_k of a hold / touch hold / slide segment, WAIT_w, the default wait implied
        by the first SH of an arm, ARM); otherwise the bound of the current state.  Pure arithmetic, no state change.
        """
        v = self._v
        kind, st = self._kind, self._state
        if kind not in ("hold", "thold", "slide"):
            return 0
        if token_id in v.dbset:
            k = token_id - v.dbs[0]
            extra = k * TPB + (1 if k == 0 else 0)
            if kind == "slide":
                if st == "segment_end":
                    return max(self._prev_end, self._wait + self._arm_dur + extra)
            else:
                return extra
        elif kind == "slide":
            if token_id == v.arm:
                prev = self._prev_end
                if st in ("slide_group_end", "slide_arm_end"):
                    prev = max(prev, self._wait + self._arm_dur)
                return max(prev, 1)
            if st == "arm_start":
                if token_id in v.shapeset:
                    return max(self._prev_end, TPB + 1)
                if v.waits[0] <= token_id <= v.waits[-1]:
                    return max(self._prev_end, token_id - v.waits[0] + 1)
        return self._floor_now()

    def completion_distance(self):
        """Minimum additional token count needed to reach accepted EOS."""
        d = self._completion_base()
        if self._bdi is None or self.finished:
            return d
        if self._state == "need_bdi":
            return 1 + d + 2 * self._pairs_from(self._bar + 1)         # BDI, then the boundary distance of the next bar
        return d + 2 * self._pairs_from(self._bar)                      # + one BARM BDI pair per remaining bar

    def _completion_base(self):
        if self.finished:
            return 0
        if self._state == "need_bdi":
            return 1
        if self.at_boundary:
            return 1
        st = self._state
        if st == "time":
            return 3  # N_TAP, BTN, EOS
        if st == "position":
            return {"tap": 2, "touch": 2, "hold": 4, "thold": 4, "slide": 7}[self._kind]
        if st == "head":
            return 3 if self._kind == "hold" else 6
        if st == "touch_head":
            return 3
        return {"arm_start": 5, "arm_shape": 5, "slide_via": 5,
                "slide_end": 4, "segment_end": 3, "duration_tick": 2}[st]

    def finishing_tokens(self):
        """A shortest legal suffix ending in EOS, without changing this state."""
        result = []
        probe = self.clone()
        v = self._v
        while not probe.finished:
            st = probe._state
            if st == "need_bdi":
                token_id = probe._local_allowed()[0]
            elif probe.at_boundary:
                token_id = v.eos if (probe._bdi is None or probe._eos_ok()) else self.V.barm
            elif st == "time":
                token_id = v.button_kinds[0]
            elif st in ("position", "slide_via", "slide_end"):
                token_id = probe._local_allowed()[0]
            elif st == "head":
                token_id = v.arm if probe._kind == "slide" else v.dbs[0]
            elif st in ("touch_head", "segment_end"):
                token_id = v.dbs[0]
            elif st in ("arm_start", "arm_shape"):
                token_id = v.straight
            elif st == "duration_tick":
                token_id = v.dts[1 if probe._duration_db == 0 else 0]
            else:
                raise AssertionError(f"cannot finish {st}")
            probe.accept(token_id)
            result.append(token_id)
        return tuple(result)

    def allowed_with_budget(self, remaining):
        """Legal next IDs that can finish within ``remaining`` tokens including EOS."""
        remaining = index(remaining)
        result = []
        if remaining < self.completion_distance() or self.finished:
            return ()
        for token_id in self.allowed():
            probe = self.clone()
            probe.accept(token_id)
            if 1 + probe.completion_distance() <= remaining:
                result.append(token_id)
        return tuple(result)
