"""Closed-loop steering of the decoder towards the statistics of official charts.

The model's samples are conservative: too few notes for the requested level, too few slides / holds / doubles, and
bar patterns that lock in and repeat for many bars.  ``Steer`` adds small logit offsets at the decisions that control
exactly these quantities; every offset is either fixed by construction or steered between windows by a log-ratio
feedback loop on the notes accepted so far, so a song that already matches the target is left alone.

Offsets (all added to the model's logits after guidance, before temperature):

* density   ``-beta * gap / TPB`` on the time tokens that follow a note (shorter gaps are favoured when ``beta > 0``);
            ``beta`` tracks notes-per-window against the count expected for the requested level and the audio activity
            (``DensityModel``).  This is where a future density-conditioning token plugs in: ``expected_fn`` is the only
            thing the loop needs.
* kinds     N_HOLD / N_SLIDE (touch with ``--touch on``) towards the official share at the requested level
            (``StyleProfile``, fitted by ``tools/fit_style.py``); BRK / EX modifiers likewise.
* doubles   all N_* tokens at a note boundary (a second note on the same tick) towards the official double rate.
* repeats   an anti-copy penalty on the continuation of a bar pattern that already repeated: when the two previous bars
            have the same *shape* (rhythm, note kinds, relative button motion) the token that would extend the copy gets a
            negative offset (time token of the next note, button lane of the next note); the same for the coarser
            rhythm-and-kinds pattern after four equal bars.
* features  ``--touch on`` / ``--fest on`` (FeatureBias).
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os

import numpy as np

from . import barsig, breaks as brk_utils, simai
from .tokens import T_MIN, TPB, encode_note

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_PATH = os.path.join(ROOT, "work", "style_model.json")
BAR = barsig.BAR
RHYTHM_CLASSES = ("12th", "16th", "24th", "32nd")
FEST_AUTO_SHARE = 0.029          # official charts overall: connected (multi-segment) share of slide arms (fest on and off together)
SHAPE_TARGETS_PATH = os.path.join(ROOT, "work", "shape_targets.json")
SHAPE_CTRL = ("STRAIGHT", "CW", "CCW", "VEE", "P", "Q", "PP", "QQ", "GRANDV")     # shapes with a feedback knob (S / Z / FAN stay with the calibration)
_SHAPE_TABLES = {}


def shape_target(level, path=None):
    """{shape: official MiLK+ share of all slide segments} for a level constant (linear interpolation, clamped to 10.5 .. 15.5)."""
    path = path or SHAPE_TARGETS_PATH
    if path not in _SHAPE_TABLES:
        with open(path, encoding="utf-8") as stream:
            d = json.load(stream)
        _SHAPE_TABLES[path] = (np.asarray(d["levels"], float), list(d["shapes"]), np.asarray(d["table"], float))
    lv, names, tab = _SHAPE_TABLES[path]
    x = float(np.clip(13.5 if level is None else level, 10.5, lv[-1]))
    return {n: float(np.interp(x, lv, tab[:, j])) for j, n in enumerate(names)}


def rhythm_class_by_rel():
    """Grid class of every tick offset 0..767 of a window (windows start on beats, TPB = 48)."""
    from .variety import rhythm_class
    return [rhythm_class(t) for t in range(768)]


MOTIF_LAG_RANK = {4: 0, 8: 0, 16: 0, 2: 1, 6: 1, 12: 1, 3: 2, 5: 2, 7: 2}      # musical phrase lengths first


def note_sig(n, bar_start):
    """Exact per-note signature (same fields as rerank.bar_signature)."""
    arm = (tuple((a.wait, tuple((s.shape, s.end) for s in a.segs), tuple(a.groups)) for a in n.arms)
           if n.kind == "slide" else ())
    return (int(n.beat) - bar_start, n.kind, n.pos, int(n.dur), arm, n.brk, n.ex)


class StyleProfile:
    """Official-chart statistics by level constant (``work/style_model.json``)."""

    def __init__(self, d):
        self.levels = np.asarray(d["levels"], np.float64)
        self.names = list(d["names"])
        self.table = np.asarray(d["table"], np.float64)
        self.std = np.asarray(d["std"], np.float64)

    @classmethod
    def load(cls, path=STYLE_PATH):
        with open(path, encoding="utf-8") as stream:
            return cls(json.load(stream))

    def target(self, name, level):
        j = self.names.index(name)
        return float(np.interp(level, self.levels, self.table[:, j]))


_PROFILE = None


def get_style_profile():
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = StyleProfile.load()
    return _PROFILE


class Knob:
    """A scalar logit offset adjusted by log-ratio feedback."""
    __slots__ = ("value", "lo", "hi", "gain", "dead", "gain_down")

    def __init__(self, init=0.0, lo=-2.0, hi=3.0, gain=0.8, dead=0.06, gain_down=None):
        self.value, self.lo, self.hi, self.gain, self.dead = float(init), lo, hi, gain, dead
        self.gain_down = gain if gain_down is None else gain_down      # gain when ``actual`` is above ``target``

    def feedback(self, actual, target):
        """Raise the offset when ``actual`` is below ``target`` (log ratio), lower it when above."""
        err = math.log(max(target, 1e-4) / max(actual, 1e-4))
        if abs(err) > self.dead:
            self.value = float(np.clip(self.value + (self.gain if err > 0 else self.gain_down) * err, self.lo, self.hi))
        return err


class FeatureBias:
    """Logit offsets for ``--touch on`` / ``--fest on``, steered by a small feedback controller.

    The unconditional effect of the TOUCH_Y / FEST_Y tokens is weak and song dependent (the tokens only say "at least one
    in the chart"), while a fixed logit offset ranges from nothing to 70 % touches.  So each offset starts at a modest
    value and is adjusted between windows to track a target the way a human charter would use the feature:
      * touch: share of touch / touch-hold notes among all notes (``TOUCH_TARGET`` * strength; official charts that use
        touch have 7 % median, 11 % 75th percentile);
      * fest: connected (multi-segment) slides per minute of audio (``FEST_PER_MINUTE`` * strength; official FESTiVAL
        charts have a median of 6 per chart, about 3 per minute).
    The offset only acts where the choice is made: N_TOUCH / N_THOLD right after a time token (touches replace other
    note kinds instead of adding notes), SH_ tokens when a slide arm may continue with another segment.
    """
    TOUCH_START = 1.0
    TOUCH_TARGET = 0.08
    FEST_START = 1.5
    FEST_PER_MINUTE = 3.0            # official FESTiVAL charts: median 2.7 / mean 3.3 connected slides per minute (median 6 per chart)
    MAX_OFFSET = 6.0

    def __init__(self, V, options, song_seconds=120.0, n_windows=1, calibrated=False):
        self.V = V
        self.touch_offset = self.fest_offset = 0.0
        self._touch_ids = self._fest_ids = None
        self.song_minutes = max(song_seconds, 1.0) / 60.0
        self.n_windows = max(1, n_windows)
        self.touch_target = float(options.touch_target) * options.touch_strength
        self.fest_rate = float(options.fest_rate) * options.fest_strength
        if options.touch == "on" and options.touch_strength > 0:
            self._touch_ids = np.asarray([V["N_TOUCH"], V["N_THOLD"]])
            self.touch_offset = 0.0 if calibrated else self.TOUCH_START * options.touch_strength
        if options.fest == "on" and options.fest_strength > 0:
            self._fest_ids = np.asarray([V[f"SH_{s}"] for s in simai.SHAPES])
            self.fest_offset = 0.0 if calibrated else self.FEST_START * options.fest_strength

    @property
    def active(self):
        return self._touch_ids is not None or self._fest_ids is not None

    def update(self, notes, window_index):
        """Adjust the offsets before window ``window_index`` from the notes accepted so far."""
        if self._touch_ids is not None and len(notes) >= 25:
            touches = sum(1 for n in notes if n.kind in ("touch", "thold"))
            ratio = touches / len(notes)
            if ratio < 0.8 * self.touch_target:
                self.touch_offset = min(self.touch_offset + 0.5, self.MAX_OFFSET)
            elif ratio > 1.25 * self.touch_target:
                self.touch_offset = max(self.touch_offset - 0.5, 0.0)
        if self._fest_ids is not None and window_index > 0:
            connected = sum(1 for n in notes if n.kind == "slide" for a in n.arms if len(a.segs) > 1)
            expected = self.fest_rate * self.song_minutes * window_index / self.n_windows
            if expected - connected > 0.8:
                self.fest_offset = min(self.fest_offset + 0.5, self.MAX_OFFSET)
            elif connected - expected > 2.0:
                self.fest_offset = max(self.fest_offset - 0.5, 0.0)

    def add(self, grammar, out):
        """Add this controller's offsets for the grammar's current state to the dense vector ``out``."""
        state = grammar.state
        if state == "time" and self._touch_ids is not None:
            out[self._touch_ids[0]] += self.touch_offset
            out[self._touch_ids[1]] += 0.5 * self.touch_offset
        elif state == "segment_end" and self._fest_ids is not None:
            out[self._fest_ids] += self.fest_offset

    def __call__(self, grammar):
        out = np.zeros(self.V.size)
        self.add(grammar, out)
        return out if out.any() else None


class StepBias:
    """Per-step steering output: ``dense`` logit offsets [V] (or None) and ``tilt`` = (time token ids, offsets) that the
    sampler applies as a mass-preserving re-weighting among the time tokens."""
    __slots__ = ("dense", "tilt")

    def __init__(self, dense=None, tilt=None):
        self.dense, self.tilt = dense, tilt


def button_pos(note):
    return int(note.pos) if note.kind in ("tap", "hold", "slide") else None


def _count_kinds(notes):
    n = len(notes)
    c = Counter()
    for x in notes:
        c[x.kind] += 1
        if x.brk or any(a.brk for a in x.arms):
            c["brk"] += 1
        if x.ex:
            c["ex"] += 1
    ticks = {}
    for x in notes:
        if x.kind in ("tap", "hold", "slide"):
            ticks[int(x.beat)] = ticks.get(int(x.beat), 0) + 1
    c["button_ticks"] = len(ticks)
    c["double_ticks"] = sum(1 for v in ticks.values() if v >= 2)
    c["n"] = n
    return c


class Steer:
    """Per-chart steering state.  Call ``begin_window`` before each window and use the instance as ``bias(grammar, guard)``."""
    # starting offsets (reached from the first window on; the feedback takes over afterwards)
    DENSITY_START = 0.5
    MEMORY = 3                   # windows of history for the density / share measurements
    RHYTHM_MEMORY = 6            # ... for the rhythm shares (onset ticks are sparse: 12th / 24th are a few per window)
    RHYTHM_PRIOR = 20.0          # pseudo-onsets at the target share
    PRIOR_NOTES = 24.0           # pseudo-observations at the target share

    def __init__(self, V, options, *, level=None, profile=None, expected_fn=None, calibration=None, song_seconds=120.0,
                 n_windows=1, intensity_fn=None, intensity_mean=None, kind_prior=None, plan=None, chorus=None):
        self.V = V
        self.kind_prior = kind_prior      # dict name -> per-bar logit offsets (mai/audiobars.py) or None
        self.calibration = calibration
        self.o = options
        self.level = level
        self.style_level = level if level is not None else 13.5
        self.profile = profile
        self.expected_fn = expected_fn if (level is not None and options.density_steer > 0) else None
        self.features = FeatureBias(V, options, song_seconds, n_windows, calibrated=calibration is not None)
        s = float(options.style_steer)
        self.style_on = s > 0 and profile is not None
        d = float(options.density_steer)
        self.kind_ids = {k: V[f"N_{k.upper()}"] for k in ("tap", "hold", "slide", "touch", "thold")}
        self.n_ids = np.asarray(list(self.kind_ids.values()))
        self.brk_id, self.ex_id = V["BRK"], V["EX"]
        self.tilt = Knob(self.DENSITY_START * d if self.expected_fn else 0.0, lo=-0.1 * d, hi=2.5 * max(d, 1.0),
                         gain=3.0 * d, dead=0.07, gain_down=1.0 * d)
        mg = float(options.mod_gain)
        self.knobs = {"hold": Knob(0.0, -2.0, 2.5 * s, 0.7 * s), "slide": Knob(0.0, -2.0, 3.0 * s, 0.9 * s),
                      "brk": Knob(0.0, -1.5, 2.0 * s, 0.6 * s * mg), "ex": Knob(0.0, -1.5, 2.0 * s, 0.6 * s * mg),
                      "double": Knob(0.0, -2.0, 2.5 * s, 0.9 * s),
                      "tch": Knob(0.0, -2.0, 2.0 * float(options.touch_steer), 0.6 * float(options.touch_steer) * mg)}
        self.targets = {}
        self.mod_memory = int(options.mod_memory)
        self.style_memory = int(getattr(options, "style_memory", self.MEMORY))   # windows of history of the hold / slide / double share feedback (0 = whole chart)
        # touch knob: only where the flag is unknown (touch on has FeatureBias, touch off is masked)
        self.touch_knob_on = options.touch == "auto" and float(options.touch_steer) > 0
        if self.style_on:
            lv = self.style_level
            ms = float(options.mod_target_scale)
            self.targets = {"hold": profile.target("kind_hold", lv), "slide": profile.target("kind_slide", lv),
                            "brk": ms * profile.target("kind_brk", lv), "ex": ms * profile.target("kind_ex", lv),
                            "double": profile.target("double_rate", lv)}
            if self.touch_knob_on:
                self.targets["tch"] = ms * profile.target("kind_touch", lv)
        # rhythm shares (12th / 16th / 24th / 32nd of the distinct onset ticks) towards the official shares of the level
        self.rhythm_on = float(options.rhythm_steer) > 0 and profile is not None
        self.rhythm_knobs, self.rhythm_targets, self.rhythm_vec = {}, {}, None
        if self.rhythm_on:
            r = float(options.rhythm_steer)
            self.rhythm_names = [c for c in str(options.rhythm_classes).split(",")
                                if c in RHYTHM_CLASSES and f"rhythm_{c}" in profile.names]
            self.rhythm_knobs = {c: Knob(0.0, -1.0, 2.5 * r, 0.8 * r, dead=0.15) for c in self.rhythm_names}
            self.rhythm_targets = {c: float(options.rhythm_target_scale) * profile.target(f"rhythm_{c}", self.style_level)
                                   for c in self.rhythm_names}
            self.rhythm_cls = np.asarray([self.rhythm_names.index(c) if c in self.rhythm_names else -1
                                          for c in rhythm_class_by_rel()])
            self.rhythm_vec = np.zeros(768)
        # connected slides while the fest flag is unknown: feedback on the multi-segment share of slide arms
        self.cont_on = float(options.cont_steer) > 0 and options.fest == "auto"
        self.cont_knob = Knob(0.0, -2.0, 2.0 * float(options.cont_steer), 0.8 * float(options.cont_steer), dead=0.2)
        self.cont_target = FEST_AUTO_SHARE * float(options.cont_target_scale)
        self.shape_ids = np.asarray([V[f"SH_{sh}"] for sh in simai.SHAPES])
        # slide shapes: integrator knobs on the running (whole-chart) share of every controlled shape, towards the official MiLK+ share of the level
        ss = float(getattr(options, "shape_steer", 0.0))
        self.shape_on = ss > 0
        self.shape_knobs, self.shape_tgt = {}, {}
        self.shape_vec = np.zeros(len(simai.SHAPES))
        if self.shape_on:
            self.shape_prior = float(getattr(options, "shape_prior", 30.0))
            tgt = shape_target(self.style_level, getattr(options, "shape_targets", "") or None)
            g = float(getattr(options, "shape_gain", 0.4)) * ss
            self.shape_tgt = {sh: tgt[sh] for sh in SHAPE_CTRL}
            self.shape_knobs = {sh: Knob(0.0, -1.5, 1.5, g, dead=0.05) for sh in SHAPE_CTRL}
        # motif recurrence: continue a bar that so far equals the beginning of an earlier bar (official charts repeat bars)
        self.motif = float(options.motif)
        self.motif_min = max(1, int(options.motif_min))
        self.motif_end = float(options.motif_end)
        self.motif_lookback = int(options.motif_lookback)
        self.prior_bars = {}              # bar -> notes of the previous windows (canonical order)
        self._mcache = {}
        self._sigc = {}
        self.copy_delta = float(options.anti_repeat)
        self.window_start = 0
        self.tilt_shift = 0.0             # temporary extra density tilt of a guard re-decode round (never fed back into the knob)
        self.act_gain = float(options.act_prior) if (intensity_fn is not None and intensity_mean) else 0.0
        self.intensity_fn, self.intensity_mean = intensity_fn, intensity_mean
        self.prior_vec = None            # per-tick onset prior of the current window (audio intensity), see ACT_CLIP
        self.window_expected = {}        # window start tick -> expected note count
        self.log = []                    # per window diagnostics
        self.counts = Counter()
        self.time_ids = V.time0 - T_MIN + np.arange(768)
        # whole-song plan (mai/plan.py): per-window share targets / feed-forward offsets; None = nothing below ever runs
        self.plan = plan
        # chorus helper (mai/chorus.py): per-window rhythm logit bias of the chorus windows; None = nothing below ever runs
        self.chorus = chorus
        self.chorus_vec = None
        self.chorus_stream = 0.0
        self.n_windows = int(n_windows)
        self._plan_tilt = 0.0
        self._E_mean = None
        self._ff = {}
        self._kind_ff = False
        self._plan_rhythm_vec = None
        self._plan_log = None
        if plan is not None:
            self._base_targets = dict(self.targets)
            self._base_rhythm_targets = dict(self.rhythm_targets)
            self._base_cont_target = self.cont_target
            self._base_copy_delta, self._base_motif = self.copy_delta, self.motif
            self._fine_mask = (np.arange(768) % (TPB // 2)) != 0

    # ------------------------------------------------------------------ between windows
    def _apply_plan(self, i):
        """per-window plan: scaled share targets (the feedback tracks them), feed-forward logit offsets, creativity"""
        P = self.plan
        parts = P.parts
        log = dict(label=P.label(i), rel=round(P.rel(i), 3))
        if "share" in parts:
            if self._base_targets:
                t = dict(self._base_targets)
                for k in t:
                    t[k] = self._base_targets[k] * P.mult(k, i)
                self.targets = t
                log["targets"] = {k: round(v, 4) for k, v in t.items()}
            self.cont_target = self._base_cont_target * P.mult("cont", i)
        if "rhythm" in parts and self.rhythm_targets:
            m = P.mult("fine", i)
            self.rhythm_targets = {c: v * m for c, v in self._base_rhythm_targets.items()}
            log["rhythm_mult"] = round(m, 3)
        self._ff = {k: 0.0 for k in ("hold", "slide", "brk", "ex", "double", "tch", "stream", "wide")}
        self._plan_rhythm_vec = None
        self._kind_ff = "ff" in parts
        if self._kind_ff:
            for k in ("hold", "slide", "brk", "ex", "double", "tch"):
                self._ff[k] = P.log_ff(k, i)
        if "wide" in parts:
            self._ff["wide"] = P.log_ff("wide", i)
        if "rhythm" in parts:
            self._ff["stream"] = P.log_ff("stream", i)
            vec = np.zeros(768)
            vec[self._fine_mask] = P.log_ff("fine", i)
            self._plan_rhythm_vec = vec if vec.any() else None
            log["fine_ff"] = round(P.log_ff("fine", i), 3)
        log["ff"] = {k: round(v, 3) for k, v in self._ff.items() if v}
        self._plan_tilt = 0.0
        if "tilt" in parts and self.expected_fn is not None:
            if self._E_mean is None:
                vals = [float(self.expected_fn(w * 768) or 0.0) for w in range(self.n_windows)]
                vals = [v for v in vals if v >= 3.0]
                self._E_mean = float(np.mean(vals)) if vals else 0.0
            E = float(self.expected_fn(i * 768) or 0.0)
            if self._E_mean > 0:
                self._plan_tilt = float(np.clip(P.ff * (E - self._E_mean) / 16.0, -1.5, 1.5))
                log["tilt_ff"] = round(self._plan_tilt, 3)
        if "creat" in parts:
            c = P.creativity(i)
            self.copy_delta = self._base_copy_delta * c["anti_repeat"]
            self.motif = self._base_motif * c["motif"]
            log["creat"] = {k: round(v, 3) for k, v in c.items()}
        self._plan_log = log

    def begin_window(self, window_index, start_tick, notes):
        if self.plan is not None:
            self._apply_plan(int(window_index))
        if self.chorus is not None:
            self.chorus.set_window(int(window_index))
            self.chorus.observe(notes)        # rw: realised onset classes of the chorus windows so far
            self.chorus_vec = self.chorus.fine_vec()
            self.chorus_stream = self.chorus.stream_gain()
        self.window_start = start_tick
        if self.motif > 0:
            self._mcache = {}
            self.prior_bars = {}
            for n in notes:
                self.prior_bars.setdefault(int(n.beat) // BAR, []).append(n)
        if self.expected_fn is not None:
            self.window_expected[start_tick] = float(self.expected_fn(start_tick) or 0.0)
        if self.act_gain > 0:
            self.prior_vec = self._intensity_prior(start_tick)
        self.features.update(notes, window_index)
        entry = dict(window=window_index, tilt=self.tilt.value, **{k: round(v.value, 3) for k, v in self.knobs.items()},
                     touch=self.features.touch_offset, fest=self.features.fest_offset)
        if window_index > 0:
            if self.expected_fn is not None:
                self._update_density(start_tick, notes, entry)
            if self.style_on:
                self._update_shares(start_tick, notes, entry)
            if self.rhythm_on:
                self._update_rhythm(start_tick, notes, entry)
            if self.cont_on:
                self._update_cont(notes, entry)
            if self.shape_on:
                self._update_shape(notes, entry)
        entry["tilt_after"] = self.tilt.value
        if self._plan_log is not None:
            entry["plan"] = self._plan_log
        if self.chorus is not None and (self.chorus_vec is not None or self.chorus_stream):
            entry["chorus"] = dict(fine=round(float(self.chorus_vec.max()), 3) if self.chorus_vec is not None else 0.0,
                                   stream=round(float(self.chorus_stream), 3),
                                   rw=[round(float(v), 3) for v in self.chorus.stats.get("rw_offsets", [])])
        self.log.append(entry)

    ACT_CLIP = 1.5            # |log ratio| cap of the intensity tilt
    ACT_EPS = 0.5             # notes/s added before the log ratio (quiet passages)

    def _intensity_prior(self, start_tick):
        """gain * log((nps(tick) + eps) / (song mean nps + eps)) for the 768 ticks of the window, added to the density tilt (slope on the gap length) of a gap that starts there:
        louder half-bars (for the level) get shorter gaps, quiet ones longer.  (A constant per-tick offset on the time tokens does nothing inside a half-bar: only the slope changes the rate.)"""
        prof = self.intensity_fn(start_tick)
        if prof is None:
            return None
        v = np.log((np.nan_to_num(prof, nan=self.intensity_mean) + self.ACT_EPS) / (self.intensity_mean + self.ACT_EPS))
        return self.act_gain * np.clip(v, -self.ACT_CLIP, self.ACT_CLIP)

    def _recent(self, start_tick, notes, memory=None):
        memory = self.MEMORY if memory is None else memory
        lo = start_tick - memory * 768 if memory > 0 else -1
        return [n for n in notes if n.beat >= lo]

    def _update_rhythm(self, start_tick, notes, entry):
        ticks = sorted({int(n.beat) for n in self._recent(start_tick, notes, self.RHYTHM_MEMORY)})
        total = len(ticks)
        prior = self.RHYTHM_PRIOR
        counts = np.zeros(len(self.rhythm_names))
        for t in ticks:
            k = self.rhythm_cls[t % 768]
            if k >= 0:
                counts[k] += 1
        vec = np.zeros(768)
        for i, c in enumerate(self.rhythm_names):
            target = self.rhythm_targets[c]
            share = (counts[i] + prior * target) / (total + prior)
            self.rhythm_knobs[c].feedback(share, target)
            vec[self.rhythm_cls == i] = self.rhythm_knobs[c].value
        self.rhythm_vec = vec
        entry["rhythm"] = {c: round(self.rhythm_knobs[c].value, 2) for c in self.rhythm_names}

    def _update_cont(self, notes, entry):
        arms = multi = 0
        for n in notes:
            if n.kind == "slide":
                for a in n.arms:
                    arms += 1
                    multi += len(a.segs) > 1
        prior = 12.0
        share = (multi + prior * self.cont_target) / (arms + prior)
        self.cont_knob.feedback(share, self.cont_target)
        entry["cont"] = round(self.cont_knob.value, 2)

    def _update_shape(self, notes, entry):
        c = Counter()
        for n in notes:
            if n.kind == "slide":
                for a in n.arms:
                    for s in a.segs:
                        c[s.shape] += 1
        total = float(sum(c.values()))
        for sh, knob in self.shape_knobs.items():
            t = self.shape_tgt[sh]
            knob.feedback((c[sh] + self.shape_prior * t) / (total + self.shape_prior), t)
            self.shape_vec[simai.SHAPES.index(sh)] = knob.value
        entry["shape"] = {sh: round(k.value, 2) for sh, k in self.shape_knobs.items()}
        entry["shape_n"] = int(total)

    def _update_density(self, start_tick, notes, entry):
        n = e = 0.0
        for w in range(max(0, start_tick // 768 - self.MEMORY), start_tick // 768):
            s = w * 768
            exp = self.window_expected.get(s)
            if exp is None or exp < 3.0:
                continue
            e += exp
            n += sum(1 for x in notes if s <= x.beat < s + 768)
        if e >= 8.0:
            err = self.tilt.feedback((n + 1.0) / (e + 1.0), 1.0)
            entry["density_ratio"] = round((n + 1.0) / (e + 1.0), 3)
            self.counts["density_updates"] += 1

    def _update_shares(self, start_tick, notes, entry):
        recent = self._recent(start_tick, notes, self.style_memory)
        c = _count_kinds(recent)
        n = c["n"]
        prior = self.PRIOR_NOTES
        if self.mod_memory != self.style_memory:
            cm = _count_kinds(self._recent(start_tick, notes, self.mod_memory))
            nm = cm["n"]
        else:
            cm, nm = c, n
        for name in ("hold", "slide", "brk", "ex", "tch"):
            if name == "ex" and self.o.ex == "off":
                continue
            if name == "tch" and not self.touch_knob_on:
                continue
            mod = name in ("brk", "ex", "tch")
            cc, nn = (cm, nm) if mod else (c, n)
            count = cc["touch"] + cc["thold"] if name == "tch" else cc[name]
            share = (count + prior * self.targets[name]) / (nn + prior)
            self.knobs[name].feedback(share, self.targets[name])
        bt = c["button_ticks"]
        rate = (c["double_ticks"] + 8.0 * self.targets["double"]) / (bt + 8.0)
        self.knobs["double"].feedback(rate, self.targets["double"])
        entry["shares"] = dict(n=n, hold=round(c["hold"] / max(n, 1), 3), slide=round(c["slide"] / max(n, 1), 3),
                               double=round(c["double_ticks"] / max(bt, 1), 3))

    # ------------------------------------------------------------------ decoding
    def __call__(self, grammar, guard=None):
        state = grammar.state
        boundary = grammar.at_boundary
        out = np.zeros(self.V.size)
        tilt = None
        if self.calibration is not None:
            self.calibration.add(grammar, out)
        if state == "time" or boundary:
            if self.style_on:
                self._kinds(out, double=boundary and grammar.has_time)
            if self.kind_prior is not None:
                self._kind_prior(out, grammar, double=boundary and grammar.has_time)
            if boundary:
                tilt = self._time_tokens(out, grammar, guard)
        gate_on = bool(getattr(self.o, "brk_timing_gate", True))
        if state == "head" and not gate_on:
            # Keep the pre-R4 addition order byte-for-byte identical when the
            # gate is explicitly disabled.
            if self.kind_prior is not None and "brk" in self.kind_prior:
                out[self.brk_id] += self._kp("brk", grammar)
            out[self.brk_id] += self.o.brk_bias
            if self.o.ex == "auto":
                out[self.ex_id] += self.o.ex_auto
            if self._kind_ff:
                out[self.brk_id] += self._ff["brk"]
                if self.o.ex != "off":
                    out[self.ex_id] += self._ff["ex"]
        elif state == "head":
            brk_boost = float(self.o.brk_bias)
            if self.kind_prior is not None and "brk" in self.kind_prior:
                brk_boost += self._kp("brk", grammar)
            if self._kind_ff:
                brk_boost += self._ff["brk"]
            if self.style_on:
                brk_boost += self.knobs["brk"].value
            brk_boost = self._brk_timing_adjust(grammar, guard, brk_boost)
            out[self.brk_id] += brk_boost
            if self.o.ex == "auto":
                out[self.ex_id] += self.o.ex_auto
            if self._kind_ff and self.o.ex != "off":
                out[self.ex_id] += self._ff["ex"]
        if state == "position" and self._ff.get("wide") and guard is not None and grammar.current_kind in ("tap", "hold", "slide"):
            self._plan_wide(out, grammar, guard)
        if state == "head" and gate_on and self.style_on and self.o.ex != "off":
            out[self.ex_id] += self.knobs["ex"].value
        elif state == "head" and not gate_on and self.style_on:
            out[self.brk_id] += self.knobs["brk"].value
            if self.o.ex != "off":
                out[self.ex_id] += self.knobs["ex"].value
        elif state == "position" and guard is not None and self.copy_delta > 0 and grammar.current_kind in (
                "tap", "hold", "slide"):
            self._lanes(out, grammar, guard)
        seg_gate = bool(getattr(self.o, "fest_seg_gate", True))
        if state == "time":
            self.features.add(grammar, out)
        elif state == "segment_end":
            if not seg_gate:
                self.features.add(grammar, out)
            elif getattr(grammar, "_segments", 1) == 1:
                if self.features._fest_ids is not None and self.features.fest_offset > 0:
                    out[self.features._fest_ids] += self.features.fest_offset
            elif getattr(grammar, "_segments", 1) >= int(getattr(self.o, "fest_seg_max", 2)):
                out[self.shape_ids] -= 4.5
                if self.calibration is not None and self.calibration.cont > 0:
                    out[self.shape_ids] -= self.calibration.cont
        if self.motif > 0 and guard is not None and grammar.has_time:
            self._motif(out, grammar, guard)
        if state == "segment_end":
            if not seg_gate:
                if self.cont_on and self.cont_knob.value:
                    out[self.shape_ids] += self.cont_knob.value
            elif getattr(grammar, "_segments", 1) == 1 and self.cont_on and self.cont_knob.value > 0:
                out[self.shape_ids] += min(float(self.cont_knob.value), 0.6)
        if self.shape_on:
            if state in ("arm_start", "arm_shape"):
                out[self.shape_ids] += self.shape_vec
            elif state == "segment_end":
                if seg_gate:
                    out[self.shape_ids] += self.shape_vec - float(np.mean(self.shape_vec))
                else:
                    out[self.shape_ids] += self.shape_vec
        dense = out if out.any() else None
        return StepBias(dense, tilt) if (dense is not None or tilt is not None) else None

    def _kp(self, name, grammar):
        arr = self.kind_prior[name]
        b = (self.window_start + max(int(grammar.cur_time), 0)) // BAR
        return float(arr[min(max(b, 0), len(arr) - 1)])

    def _brk_timing_adjust(self, grammar, guard, base):
        """Musical beat/phrase gate, per-bar cooldown and tap-chord symmetry.

        Positive steering is concentrated on phrase starts/cadences; weak
        mid-bar 16th/24th/32nd offbeats are actively pushed away from BRK.
        """
        cur_t = max(int(grammar.cur_time), 0)
        bar_in_win = cur_t // brk_utils.BAR
        beat_rem = cur_t % brk_utils.TPB
        bar_rem = cur_t % brk_utils.BAR
        pos = max(0.0, float(base))
        neg = min(0.0, float(base))
        penalty = 0.0

        final_bar3_beats = cur_t >= 14 * brk_utils.TPB
        weak = (beat_rem % 24 != 0) and not final_bar3_beats
        mid_phrase = bar_in_win in (1, 2) and bar_rem not in (0, 144)
        if weak:
            pos = 0.0
            penalty -= 1.5
        elif mid_phrase:
            pos *= 0.35

        if guard is not None and getattr(guard, "recent", None):
            bar_start = self.window_start + bar_in_win * brk_utils.BAR
            bar_end = bar_start + brk_utils.BAR
            prior_ticks = sorted({int(round(float(n.beat))) for n in guard.recent
                                  if bar_start <= float(n.beat) < bar_end and brk_utils.note_has_break(n)})
            cur_abs = self.window_start + cur_t
            if len(prior_ticks) >= 2 and (not prior_ticks or cur_abs != prior_ticks[-1]):
                penalty -= 3.5

            prev = guard.recent[-1]
            if (int(round(float(prev.beat))) == cur_abs
                    and str(getattr(prev, "kind", "")) == "tap"
                    and grammar.current_kind == "tap"):
                penalty += 2.5 if brk_utils.note_has_break(prev) else -3.5

        return neg + pos + penalty

    def _kind_prior(self, out, grammar, double):
        kp = self.kind_prior
        out[self.kind_ids["hold"]] += self._kp("hold", grammar)
        out[self.kind_ids["slide"]] += self._kp("slide", grammar)
        if self.o.touch == "auto":
            t = self._kp("touch", grammar)
            out[self.kind_ids["touch"]] += t
            out[self.kind_ids["thold"]] += t
        if double:
            out[self.n_ids] += self._kp("double", grammar)

    def _kinds(self, out, double):
        k = self.knobs
        out[self.kind_ids["hold"]] += k["hold"].value
        out[self.kind_ids["slide"]] += k["slide"].value
        if self.o.touch == "auto" and self.o.touch_auto:
            out[self.kind_ids["touch"]] += self.o.touch_auto
            out[self.kind_ids["thold"]] += self.o.touch_auto
        if self.touch_knob_on and k["tch"].value:
            out[self.kind_ids["touch"]] += k["tch"].value
            out[self.kind_ids["thold"]] += k["tch"].value
        if double:
            out[self.n_ids] += k["double"].value
        if self._kind_ff:
            ff = self._ff
            out[self.kind_ids["hold"]] += ff["hold"]
            out[self.kind_ids["slide"]] += ff["slide"]
            if self.o.touch == "auto" and ff["tch"]:
                out[self.kind_ids["touch"]] += ff["tch"]
                out[self.kind_ids["thold"]] += ff["tch"]
            if double:
                out[self.n_ids] += ff["double"]

    def _time_tokens(self, out, grammar, guard):
        """Anti-copy penalties go into ``out``; returns the density tilt (ids, offsets) or None."""
        if not grammar.has_time:
            return None
        cur = grammar.cur_time
        beta = self.tilt.value + self.tilt_shift + self._plan_tilt
        if self.prior_vec is not None:       # audio intensity at the start of the gap: denser tilt in loud half-bars, sparser in quiet ones
            beta += float(self.prior_vec[min(max(cur, 0), len(self.prior_vec) - 1)])
        lo, hi = cur + 1, grammar.max_tick
        if hi <= lo:
            return None
        if self.copy_delta > 0 and guard is not None:
            plan = self._time_plan(guard, grammar)
            for tick, delta in plan.items():
                rel = tick - guard.start_tick
                if lo <= rel < hi:
                    out[self.time_ids[rel]] -= delta
                    self.counts["time_penalties"] += 1
        if self.rhythm_vec is not None:
            out[self.time_ids[lo:hi]] += self.rhythm_vec[lo:hi]
        if self._plan_rhythm_vec is not None:
            out[self.time_ids[lo:hi]] += self._plan_rhythm_vec[lo:hi]
        if self.chorus_vec is not None:
            out[self.time_ids[lo:hi]] += self.chorus_vec[lo:hi]
        if self.chorus_stream and guard is not None:
            self._chorus_stream_bias(out, grammar, guard, lo, hi)
        if self._ff.get("stream") and guard is not None:
            self._plan_stream(out, grammar, guard, lo, hi)
        if beta:
            ticks = np.arange(lo, hi)
            return self.time_ids[lo:hi], -beta * (ticks - cur) / TPB
        return None

    # ------------------------------------------------------------------ plan: stream / wide feed-forward
    STREAM_GAP = TPB // 4         # a 16th: cmet "stream" = runs of >= 4 button onsets at most this far apart
    WIDE_GAP = TPB // 2           # an 8th: cmet "wide" = button jump of >= 3 lanes at most this far apart

    def _plan_stream(self, out, grammar, guard, lo, hi):
        """continue (planned multiplier > 1) or cut (< 1) a running 16th stream: offset on the time token one 16th after the last onset"""
        ev = self._events(guard, grammar)
        ticks = sorted({int(n.beat) for n in ev[-12:] if button_pos(n) is not None})
        if len(ticks) < 2 or ticks[-1] - ticks[-2] > self.STREAM_GAP:
            return
        rel = ticks[-1] + self.STREAM_GAP - guard.start_tick
        if lo <= rel < hi:
            out[self.time_ids[rel]] += self._ff["stream"]

    def _chorus_stream_bias(self, out, grammar, guard, lo, hi):
        """chorus ``stream`` part: keep a running 16th (12-tick) stream going (same rule as cmet's stream runs)."""
        ev = self._events(guard, grammar)
        ticks = sorted({int(n.beat) for n in ev[-12:] if button_pos(n) is not None})
        if len(ticks) < 2 or ticks[-1] - ticks[-2] > self.STREAM_GAP:
            return
        rel = ticks[-1] + self.STREAM_GAP - guard.start_tick
        if lo <= rel < hi:
            out[self.time_ids[rel]] += self.chorus_stream

    def _plan_wide(self, out, grammar, guard):
        """offset on the buttons >= 3 lanes away from the previous button within an 8th (planned multiplier > 1 favours big jumps)"""
        tau = guard.start_tick + grammar.cur_time
        prev = None
        for n in reversed(guard.recent):
            if button_pos(n) is not None:
                prev = n
                break
        if prev is None:
            return
        dt = tau - int(prev.beat)
        if dt <= 0 or dt > self.WIDE_GAP:
            return
        p = button_pos(prev)
        g = self._ff["wide"]
        for lane in range(1, 9):
            d = abs(lane - p)
            if min(d, 8 - d) >= 3:
                out[self.V[f"BTN_{lane}"]] += g

    # ------------------------------------------------------------------ anti-copy
    @staticmethod
    def _events(guard, grammar):
        ev = guard.recent
        if grammar.pending_note_tokens and grammar.completed_count >= guard.committed_count:
            pending = guard._note(grammar)
            if pending is not None:
                ev = ev + [pending]
        return ev

    @staticmethod
    def _bars(events, last_bar, depth=5):
        bars = {}
        for n in events:
            b = int(n.beat) // BAR
            if b > last_bar - depth:
                bars.setdefault(b, []).append(n)
        return bars

    @staticmethod
    def _sig(bars, b, fn, min_notes=3):
        ns = bars.get(b)
        if not ns or len(ns) < min_notes:
            return None
        return fn(ns, b * BAR)

    def _run(self, bars, b, fn):
        """(signature of bar b, number of consecutive equal bars ending at b)."""
        s = self._sig(bars, b, fn)
        if s is None:
            return None, 0
        r, j = 1, b - 1
        while j > b - 5 and self._sig(bars, j, fn) == s:
            r += 1
            j -= 1
        return s, r

    def _time_plan(self, guard, grammar):
        """{absolute tick: penalty} of the time tokens that would extend a repeating bar pattern."""
        events = self._events(guard, grammar)
        if len(events) < 6:
            return {}
        last = events[-1]
        cb = int(last.beat) // BAR
        bars = self._bars(events, cb)
        plan = {}
        d0 = self.copy_delta
        for fn, shape_run, scale in ((barsig.shape_events, 2, 1.0), (barsig.rk_signature, 4, 0.5)):
            ref, r = self._run(bars, cb - 1, fn)
            cur_ns = bars.get(cb, [])
            if ref is not None and r >= shape_run and cur_ns:
                cur = fn(cur_ns, cb * BAR)
                if len(cur) < len(ref) and ref[:len(cur)] == cur:
                    tick = cb * BAR + ref[len(cur)][0]
                    if tick > int(last.beat):
                        plan[tick] = max(plan.get(tick, 0.0), scale * d0 * min(r - shape_run + 1, 3))
            sig, r2 = self._run(bars, cb, fn)
            if sig is not None and r2 >= shape_run:
                tick = (cb + 1) * BAR + sig[0][0]
                plan[tick] = max(plan.get(tick, 0.0), scale * d0 * min(r2 - shape_run + 1, 3))
        return plan

    def _lanes(self, out, grammar, guard):
        tau = guard.start_tick + grammar.cur_time
        b = tau // BAR
        events = guard.recent
        if len(events) < 6:
            return
        bars = self._bars(events, b)
        ref, r = self._run(bars, b - 1, barsig.shape_events)
        if ref is None or r < 2:
            return
        cur_ns = [n for n in bars.get(b, []) if int(n.beat) <= tau]
        cur = barsig.shape_events(cur_ns, b * BAR)
        j = len(cur)
        if j == 0 or j >= len(ref) or ref[:j] != cur:
            return
        nxt = ref[j]
        if nxt[0] != tau - b * BAR or nxt[1] != grammar.current_kind or nxt[2] is None:
            return
        prev = [button_pos(n) for n in sorted(cur_ns, key=lambda x: (int(x.beat), x.pos)) if button_pos(n) is not None]
        if not prev:
            return
        lane = (prev[-1] - 1 + nxt[2]) % 8 + 1
        out[self.V[f"BTN_{lane}"]] -= self.copy_delta * min(r, 3)
        self.counts["lane_penalties"] += 1

    # ------------------------------------------------------------------ motif recurrence
    def _bar_data(self, a, recent, window_bar):
        """(sorted signatures, notes) of bar ``a`` (>= 3 notes) or None; finished bars are cached."""
        if a < window_bar:
            hit = self._sigc.get(a, 0)
            if hit != 0:
                return hit
            ref = self.prior_bars.get(a, ())
        else:
            hit, ref = 0, recent.get(a, ())
        if len(ref) < 3:
            out = None
        else:
            pairs = sorted(((note_sig(n, a * BAR), n) for n in ref), key=lambda t: (t[0][0], t[0][2]))
            out = ([t[0] for t in pairs], [t[1] for t in pairs])
        if a < window_bar:
            self._sigc[a] = out
        return out

    def _motif_ref(self, guard, events, cb):
        """(reference notes sorted canonically, number of matched leading notes, bar) of the best earlier bar, or None."""
        key = (id(guard), guard.committed_count, cb, len(events), note_sig(events[-1], 0) if events else None)
        hit = self._mcache.get(key, 0)
        if hit != 0:
            return hit
        recent = {}
        for n in events:
            if int(n.beat) >= guard.start_tick:
                recent.setdefault(int(n.beat) // BAR, []).append(n)
        window_bar = guard.start_tick // BAR
        cur_notes = recent.get(cb) if cb >= window_bar else [n for n in events if int(n.beat) // BAR == cb]
        cur = sorted((note_sig(n, cb * BAR) for n in (cur_notes or ())), key=lambda x: (x[0], x[2]))
        best = None
        if len(cur) >= self.motif_min:
            j = len(cur)
            for a in range(cb - 2, max(-1, cb - 1 - self.motif_lookback), -1):
                data = self._bar_data(a, recent, window_bar)
                if data is None or len(data[0]) <= j or data[0][:j] != cur:
                    continue
                rank = (MOTIF_LAG_RANK.get(cb - a, 3), cb - a)
                if best is None or rank < best[0]:
                    best = (rank, data[1], a)
        out = (best[1], len(cur), best[2]) if best else None
        if len(self._mcache) > 4096:
            self._mcache.clear()
        self._mcache[key] = out
        return out

    def _motif(self, out, grammar, guard):
        state = grammar.state
        if state in ("start", "finished"):
            return
        events = self._events(guard, grammar)
        inprog = bool(grammar.note_tokens) and not grammar.at_boundary
        tau = guard.start_tick + grammar.cur_time
        if inprog or state == "time":
            cb = tau // BAR
        elif events:
            cb = int(events[-1].beat) // BAR
        else:
            return
        found = self._motif_ref(guard, events, cb)
        self.counts["motif_calls"] += 1
        if found is None:
            return
        self.counts["motif_found"] += 1
        ref, j, a = found
        d = self.motif
        off = lambda n: int(n.beat) - a * BAR
        if inprog:
            if j >= len(ref) or off(ref[j]) != tau - cb * BAR:
                return
            expected = encode_note(self.V, ref[j])
            k = len(grammar.note_tokens)
            if k < len(expected) and list(grammar.note_tokens) == expected[:k]:
                out[expected[k]] += d
                self.counts["motif_token"] += 1
            return
        if state == "time":
            if j < len(ref) and off(ref[j]) == tau - cb * BAR:
                out[encode_note(self.V, ref[j])[0]] += d
            return
        # note boundary: the next onset (time token, or another note on the same tick)
        last_tick = int(events[-1].beat)
        if j < len(ref):
            nxt_tick = cb * BAR + off(ref[j])
            if nxt_tick == last_tick:
                out[encode_note(self.V, ref[j])[0]] += d
            elif nxt_tick > last_tick:
                rel = nxt_tick - guard.start_tick
                if grammar.cur_time < rel < grammar.max_tick:
                    out[self.time_ids[rel]] += d
        else:                                 # the reference bar is complete: end the bar here
            e = d * self.motif_end
            lo = max(grammar.cur_time + 1, 0)
            hi = min((cb + 1) * BAR - guard.start_tick, grammar.max_tick)
            if hi > lo:
                out[self.time_ids[lo:hi]] -= e
            out[[self.kind_ids[k] for k in ("tap", "hold", "slide", "touch", "thold")]] -= e

    def summary(self):
        return dict(counts=dict(self.counts), tilt=self.tilt.value,
                    knobs={k: round(v.value, 3) for k, v in self.knobs.items()},
                    shape_knobs={k: round(v.value, 3) for k, v in self.shape_knobs.items()},
                    touch=self.features.touch_offset, fest=self.features.fest_offset)
