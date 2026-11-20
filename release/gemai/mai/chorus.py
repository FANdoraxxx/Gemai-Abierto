"""Chorus / highlight helper of the generator: ``--chorus S`` (strength, 0 = off = byte identical) x ``--chorus-parts``.

Problem (out/plan/e25_report_plan_seeds.md, section 3): the fine rhythms (16th / 24th / 32nd), the peak bar density
(npb_p90), the density (npb) and the 16th streams (stream) of the generated charts are clearly below the official charts
*inside the chorus / drop windows*, while the verse windows are fine.

Sections: audio-only section predictor (scratch/plan/sectionpred.py through ``mai.plan.predicted_record``) -> v4 annotation
record {nb, starts, labels}; a generation window (4 bars) is a chorus window when at least ``--chorus-min-frac`` of its bars
is chorus or drop.  ``--chorus-section file:<json>`` feeds a stored record instead (diagnostics only).

Targets: ``out/plan/targets.json`` -- median chorus / verse values of the official MiLK+ (versionid >= 10) MASTER /
Re:MASTER charts of the level band, fitted by scratch/plan/fit_targets.py on the TRAINING songs (val_songs / e25 never
enter it).  Only the ratio and the medians are used; nothing is fitted here.

Parts (``--chorus-parts``, comma separated; each is inert while its knob is 0):
  dens    the expected note count of the chorus windows x the official chorus / verse density ratio, renormalised so the
          song total stays the level-calibrated one (acts on the rerank density penalty + guard and on the density feedback
          of the steering).
  pool    more window candidates in the chorus windows (``--chorus-k``) and a logit bias on the non-8th-grid time tokens
          (``--chorus-fine-ff``) so dense writing is actually in the candidate pool.
  stream  ... plus a bias on the time token that continues a running 16th stream (``--chorus-stream-ff``).
  pick    rerank reward for a chorus window whose fine-rhythm share / p90 bar density is close to the official chorus
          median (``--chorus-pick``); the hand and muri terms are untouched (no density is bought with playability).
  rw      mass-preserving re-weighting of the time tokens of the chorus windows towards the official chorus onset-class
          shares (8th / 12th / 16th / 24th / 32nd): per-class offsets from a log-ratio feedback against the realised
          onset shares of the chorus windows so far.  The sampler renormalises among the time tokens, so the total
          probability mass of the time tokens is conserved; only the musical grids are touched (the 48th / 64th fuzz
          ticks stay at 0).  Replaces the ``pool`` feed-forward offset when both parts are on (mutually exclusive;
          ``pool`` then keeps its extra candidates only).
  rwo     onset-gated ``rw`` (mutually exclusive with it): the same log-ratio re-weighting, but an offset is only placed
          on a fine-grid tick that carries an audio onset.  Onset strength of a tick = max positive spectral flux
          (``mai.activity.flux_frames``, the existing 200 fps log-mel flux) over the frames within
          ``--chorus-onset-tol`` ticks (default +-1) of the tick time; a tick is allowed when its strength reaches the
          ``--chorus-onset-q`` quantile (default 0.7) of the song's own per-tick onset strengths.  A fine tick without an
          onset keeps the model's own probability, so the re-weighting only moves mass between positions the music marks.
          The sampler renormalises among the time tokens (total time-token mass conserved), exactly as ``rw``.
  hand    the rerank hand penalty of a chorus window is halved while the candidate does not exceed the expected note count
          of that window (``hand_scale``): inside dense windows the hand budget (with the intensity term) is what pushes
          the selection back to the sparser half of the candidate pool (chorus_diag 2.1).  A candidate that is not yet at
          the level / audio expectation is not "too dense" and must not pay the full playability price for it.  Windows at
          or above the expected count, and everything outside the chorus windows, are untouched.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import paths as _paths
from .audio import FPS as _FPS

TPB = 48
BAR = 4 * TPB
EIGHTH = TPB // 2               # 24 ticks = one 8th note: cmet "rhythm_fine" counts the onsets off this grid
WINDOW = 4 * BAR                # 768 ticks
BARS_PER_WINDOW = 4
LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "break", "build-up", "drop", "outro", "other")
CHORUS_LABELS = (LABELS.index("chorus"), LABELS.index("drop"))
ALL_PARTS = ("dens", "pool", "stream", "pick", "rw", "rwo", "hand")
DEFAULT_PARTS = "dens,pool,pick"
NEEDED = ("npb", "npb_p90", "rhythm_fine", "stream")
# rw: onset class of a window tick (0 = 8th grid, 1..4 = 12th / 16th / 24th / 32nd grid, -1 = fuzz)
RW_CLASSES = ("8th", "12th", "16th", "24th", "32nd")
RW_FINE = RW_CLASSES[1:]
RW_FUZZ = -1
STYLE_PATH = "work/style_model.json"
_RW_SPLIT_CACHE = {}


def rw_class(rel):
    """Class of a window tick (0 = 8th grid, 1..4 = 12th / 16th / 24th / 32nd grid, -1 = fuzz)."""
    rel = int(rel) % WINDOW
    if rel % EIGHTH == 0:
        return 0
    if rel % 16 == 0:
        return 1
    if rel % 12 == 0:
        return 2
    if rel % 8 == 0:
        return 3
    if rel % 6 == 0:
        return 4
    return RW_FUZZ


def rw_split(level):
    """Official split of the *fine* onsets among the 12th / 16th / 24th / 32nd classes at a level (work/style_model.json)."""
    path = _paths.project_root(__file__) / STYLE_PATH
    key = (str(path), round(float(level), 2))
    if key not in _RW_SPLIT_CACHE:
        try:
            with open(path, encoding="utf-8") as stream:
                d = json.load(stream)
            lv = np.asarray(d["levels"], np.float64)
            tab = np.asarray(d["table"], np.float64)
            w = np.asarray([float(np.interp(level, lv, tab[:, d["names"].index("rhythm_" + c)])) for c in RW_FINE])
        except Exception:                    # a missing / malformed style table must not kill a generation run
            w = np.ones(len(RW_FINE))
        w = np.maximum(w, 1e-4)
        _RW_SPLIT_CACHE[key] = w / w.sum()
    return _RW_SPLIT_CACHE[key]


def parse_parts(text):
    parts = {p.strip() for p in str(text).split(",") if p.strip()}
    bad = parts - set(ALL_PARTS)
    if bad:
        raise ValueError(f"chorus_parts: unknown {sorted(bad)}; choose from {ALL_PARTS}")
    if "rw" in parts and "rwo" in parts:
        raise ValueError("chorus_parts: rw and rwo are mutually exclusive (rwo = onset-gated rw)")
    return parts


def onset_strength(mel, tempo, first, n_ticks, tol=1.0):
    """Onset strength of every tick of the chart grid (float64 [n_ticks]).

    ``mel`` = 200 fps log-mel [T, 96] of the song (``mai.audio.mel_db``), ``tempo`` needs ``to_sec(beats)`` and ``first``
    is the audio offset of beat 0 in seconds.  Strength of a tick = max positive spectral flux (``activity.flux_frames``,
    the same onset measure the density model uses) over the mel frames within ``tol`` ticks of that tick's time.  Ticks
    whose range falls outside the audio get 0.
    """
    from . import activity
    mel32 = np.asarray(mel, np.float32)
    n = max(0, int(n_ticks))
    out = np.zeros(n, np.float64)
    if n == 0 or mel32.ndim != 2 or mel32.shape[0] < 3:
        return out
    flux = np.asarray(activity.flux_frames(mel32), np.float64)
    T = int(len(flux))
    if T < 3:
        return out
    t = np.arange(n, dtype=np.float64)
    beats = t / TPB                                            # tempo maps beat -> sec, the grid counts ticks
    f0 = (np.asarray(tempo.to_sec(beats), np.float64) + float(first)) * _FPS
    f1 = (np.asarray(tempo.to_sec((t + 1.0) / TPB), np.float64) + float(first)) * _FPS
    tick_frames = np.maximum(f1 - f0, 1e-3)                     # frames per tick at this tick's tempo
    c = np.rint(f0).astype(np.int64)
    rad = np.clip(np.rint(max(0.0, float(tol)) * tick_frames), 1, 512).astype(np.int64)
    R = int(rad.max()) if n else 1
    best = np.zeros(n, np.float64)
    for d in range(-R, R + 1):
        idx = np.clip(c + d, 0, T - 1)
        best = np.where(np.abs(d) <= rad, np.maximum(best, flux[idx]), best)
    out = best
    return out


def band_of(lv):
    """Display level band of a decimal level constant (same thresholds as scratch/chorus/mai/cmet.band_of)."""
    c = float(lv) + 1e-9
    base = math.floor(c)
    disp = base + (0.5 if base >= 7 and round(c - base, 1) >= 0.6 else 0.0)
    for name, lo, hi in (("12", 12.0, 12.5), ("12+", 12.5, 13.0), ("13", 13.0, 13.5), ("13+", 13.5, 14.0),
                         ("14", 14.0, 14.5), ("14+", 14.5, 99.0)):
        if lo - 1e-9 <= disp < hi - 1e-9:
            return name
    return "14+" if disp >= 14.5 else "12"


def targets_path():
    return Path(os.environ.get("MAI_CHORUS_TARGETS") or (_paths.project_root(__file__) / "out" / "plan" / "targets.json"))


_TARGET_CACHE = {}


def load_targets(band, path=None):
    """{metric: (chorus median, verse median)} of the level band from out/plan/targets.json."""
    path = Path(path or targets_path())
    if path not in _TARGET_CACHE:
        with open(path, encoding="utf-8") as stream:
            _TARGET_CACHE[path] = json.load(stream)
    bands = _TARGET_CACHE[path]["bands"]
    b = bands.get(band) or bands[sorted(bands)[0]]
    out = {}
    for name in NEEDED:
        c = b.get("chorus", {}).get(name)
        v = b.get("verse", {}).get(name)
        c = float(c["p50"]) if isinstance(c, dict) else float(c or 0.0)
        v = float(v["p50"]) if isinstance(v, dict) else float(v or 0.0)
        out[name] = (c, v)
    return out


def _labels_of(rec, nb):
    """v4 annotation record -> per-bar label id (-1 where not covered)."""
    starts = list(np.asarray(rec["starts"]).astype(int))
    labels = list(np.asarray(rec["labels"]).astype(int))
    lab = np.full(int(nb), -1, np.int64)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(lab)
        lab[max(s, 0):min(int(e), len(lab))] = labels[i]
    return lab


class ChorusState:
    """Per-song chorus windows, their targets and the four parts.  Created once per song (``build``), attached to the
    window scorer (``pick``), the steering (``pool`` / ``stream``) and the expectation (``dens``)."""

    def __init__(self, options, nw, frac, level, band, targets, labels, source):
        self.o = options
        self.S = float(options.chorus)
        self.parts = parse_parts(options.chorus_parts)
        self.nw = int(nw)
        self.frac = np.asarray(frac, np.float64)[: self.nw]
        self.cmask = self.frac >= float(options.chorus_min_frac)
        self.level = level
        self.band = band
        self.tg = targets
        self.labels = list(labels)[: self.nw]
        self.source = source
        self.mults = np.ones(self.nw)          # dens: per-window expected-note multiplier (1 = untouched)
        self.fine_ff = float(options.chorus_fine_ff) * self.S
        self.stream_ff = float(options.chorus_stream_ff) * self.S
        self.pick_w = float(options.chorus_pick) * self.S
        self.pick_band = float(options.chorus_pick_band)
        band = float(getattr(options, "chorus_band", 0.0) or 0.0)
        self.band_override = band if band > 0 else None      # dens: narrower density dead band inside the chorus windows
        self.k_extra = float(options.chorus_k) * min(1.0, max(0.0, self.S))
        # rw / rwo: mass-preserving re-weighting of the time tokens of the chorus windows (replaces pool's fine-ff when on)
        self.rwo_on = "rwo" in self.parts                       # rwo = the same offsets, only on onset-bearing ticks
        self.rw_on = ("rw" in self.parts) or self.rwo_on
        self.onset_q = float(getattr(options, "chorus_onset_q", 0.7) if getattr(options, "chorus_onset_q", None) is not None else 0.7)
        self.onset_tol = float(getattr(options, "chorus_onset_tol", 1.0) if getattr(options, "chorus_onset_tol", None) is not None else 1.0)
        self.onset = None            # per-tick onset strength of the whole grid (attach_onset, rwo only)
        self.onset_allowed = None    # bool [n_ticks] gate, rwo only
        self.onset_thr = 0.0
        self.rw_gain = float(getattr(options, "chorus_rw_gain", 1.0) or 0.0) * self.S
        self.rw_max = float(getattr(options, "chorus_rw_max", 1.0) or 1.0)
        self.rw_rel = float(getattr(options, "chorus_rw_rel", 0.35) or 0.0)
        self.rw_dead = float(getattr(options, "chorus_rw_dead", 0.05) or 0.0)
        self.rw_prior = float(getattr(options, "chorus_rw_prior", 8.0) or 0.0)
        self.rw_cls = np.asarray([rw_class(t) for t in range(WINDOW)], np.int8)
        self.rw_counts = np.zeros(len(RW_CLASSES), np.float64)
        self.rw_target = self._rw_target() if self.rw_on else None
        self.rw_vec = None
        self._cur = -1
        self.fine_mode = str(getattr(options, "chorus_fine_mode", "any") or "any")
        ticks = np.arange(WINDOW)
        off8 = (ticks % EIGHTH) != 0
        musical = ((ticks % 6 == 0) | (ticks % 8 == 0)) & off8      # 16th / 24th / 32nd grid (no 48th / 64th fuzz)
        self._fine_mask = musical if self.fine_mode == "grid" else off8
        self.stats = dict(dens_windows=0, k_windows=0, pick_onsets=0, pick_hits=0,
                          rw_windows=0, rw_onsets=0, rw_offsets=[0.0] * len(RW_CLASSES),
                          rwo_windows=0, rwo_hits=0, rwo_allowed=0, rwo_ticks=0,
                          hand_windows=0, hand_hits=0)
        self.starts = None                     # set by the generator: window start ticks of the song

    # ------------------------------------------------------------------ sections
    def is_chorus(self, i):
        return 0 <= int(i) < self.nw and bool(self.cmask[int(i)])

    # ------------------------------------------------------------------ dens
    def apply_dens(self, expect):
        """Scale ``expect.expected`` of the chorus windows by the official chorus/verse density ratio, song total kept."""
        if "dens" not in self.parts or expect is None:
            return
        npb_c, npb_v = self.tg["npb"]
        ratio = float(np.clip(npb_c / max(npb_v, 0.5), 1.0, 2.0))
        m = 1.0 + (ratio - 1.0) * self.S * self.frac
        if not np.any(m != 1.0):
            return
        e = np.asarray(expect.expected, np.float64)
        s0 = float(e.sum())
        s1 = float((e * m).sum())
        if s0 > 1e-6 and s1 > 1e-6:
            m = m * (s0 / s1)                    # the level-calibrated song total stays the same; the verse gets slightly sparser
        expect.expected = e * m
        try:
            expect.ratio = np.asarray(expect.ratio, np.float64) * m
        except Exception:
            pass
        self.mults = np.asarray(m, np.float64)
        self.stats["dens_windows"] = int(np.count_nonzero(self.mults > 1.0 + 1e-9))

    def wrap_expected(self, fn):
        """density feedback of the steering: same per-window multiplier as the rerank / guard expectation."""
        if "dens" not in self.parts or not np.any(self.mults != 1.0) or fn is None:
            return fn
        mults = self.mults

        def wrapped(tick):
            v = fn(tick)
            if v is None:
                return v
            w = int(tick) // WINDOW
            return float(v) * (float(mults[w]) if 0 <= w < len(mults) else 1.0)

        return wrapped

    # ------------------------------------------------------------------ pool / stream
    def k_for(self, i, K):
        """window candidates of window ``i`` (more in the chorus windows while ``pool`` is on)."""
        K = int(K)
        if "pool" not in self.parts or self.k_extra <= 0 or not self.is_chorus(i):
            return K
        kw = int(round(K * (1.0 + self.k_extra)))
        self.stats["k_windows"] += int(kw > K)
        return max(K, kw)

    def set_window(self, i):
        self._cur = int(i)

    def fine_vec(self):
        """per-tick logit offsets of the current window: the non-8th grid time tokens (cmet rhythm_fine)."""
        if not self.is_chorus(self._cur):
            return None
        if self.rw_on:                       # rw replaces the flat pool offset (mutually exclusive)
            return self._rw_vec()
        if "pool" not in self.parts or not self.fine_ff:
            return None
        vec = np.zeros(WINDOW)
        vec[self._fine_mask] = self.fine_ff
        return vec

    def stream_gain(self):
        """logit offset that continues a running 16th stream in a chorus window (0 = off)."""
        if "stream" not in self.parts or not self.stream_ff or not self.is_chorus(self._cur):
            return 0.0
        return self.stream_ff

    # ------------------------------------------------------------------ hand
    def hand_scale(self, i, n_notes, expected):
        """Factor on the rerank hand penalty of a chorus window (1 = untouched).

        ``hand``: while the candidate does not exceed the expected note count of the window it cannot be "too dense", so
        the playability price is halved (1 - 0.5 * S) instead of pushing the selection towards the sparser pool half.
        """
        if "hand" not in self.parts or expected is None or not self.is_chorus(i):
            return 1.0
        exp = float(expected)
        if not np.isfinite(exp) or float(n_notes) > exp + 1e-9:
            return 1.0
        self.stats["hand_hits"] += 1
        return max(0.0, 1.0 - 0.5 * min(1.0, max(0.0, self.S)))

    # ------------------------------------------------------------------ rw
    def _rw_target(self):
        """Target share of every onset class of a chorus window: the official chorus fine share (targets.json) split by
        the official 12th / 16th / 24th / 32nd split at this level; the remainder is the 8th grid."""
        fine = float(np.clip(float(self.tg["rhythm_fine"][0]), 0.0, 0.9))
        q = np.zeros(len(RW_CLASSES), np.float64)
        q[0] = 1.0 - fine
        q[1:] = fine * rw_split(self.level)
        total = float(q.sum())
        return q / total if total > 0 else None

    def observe(self, notes):
        """Recount the button onsets of the chorus windows among the notes accepted so far (rw feedback)."""
        if not self.rw_on:
            return
        self.rw_counts[:] = 0.0
        for n in notes:
            if n.kind in ("touch", "thold") or (n.kind == "slide" and n.headless):
                continue
            i = int(n.beat) // WINDOW
            if not self.is_chorus(i):
                continue
            k = int(self.rw_cls[(int(n.beat) - i * WINDOW) % WINDOW])
            if k >= 0:
                self.rw_counts[k] += 1.0
        self.stats["rw_onsets"] = int(self.rw_counts.sum())

    # ------------------------------------------------------------------ rwo (onset gate)
    def attach_onset(self, mel, tempo, first, n_ticks=None, log=print):
        """rwo: per-tick onset gate of the whole song grid (computed once per song; inert unless ``rwo`` is on).

        The gate keeps a fine-grid tick only when the audio carries an onset there -- the max spectral flux within
        ``onset_tol`` ticks of the tick reaches the ``onset_q`` quantile of the song's per-tick onset strengths.  Without
        mel / tempo (or on any error) the part degrades to plain ``rw``.
        """
        if not self.rwo_on:
            return
        n = int(n_ticks) if n_ticks else int(self.nw) * WINDOW
        if mel is None or tempo is None or n <= 0:
            log("warning: chorus rwo without audio features; falling back to plain rw offsets", flush=True)
            self.rwo_on = False
            return
        try:
            s = onset_strength(mel, tempo, first, n, self.onset_tol)
        except Exception as exc:                       # a missing / odd mel must not kill a generation run
            log(f"warning: chorus rwo onset gate unavailable ({exc}); falling back to plain rw offsets", flush=True)
            self.rwo_on = False
            return
        self.onset = s
        q = float(np.clip(self.onset_q, 0.0, 1.0))
        self.onset_thr = float(np.quantile(s, q)) if len(s) else 0.0
        # an all-silent grid (threshold 0) allows nothing: the part is then inert instead of boosting every tick
        self.onset_allowed = s >= max(self.onset_thr, 1e-9)
        self.stats["rwo_allowed"] = int(self.onset_allowed.sum())
        self.stats["rwo_ticks"] = int(len(s))
        allowed_fine = int(np.count_nonzero(self.onset_allowed & (self.rw_cls[np.arange(len(s)) % WINDOW] > 0)))
        log(f"chorus rwo: onset gate q={q:.2f} tol=+-{self.onset_tol:g} tick, threshold {self.onset_thr:.4g}, "
            f"{int(self.onset_allowed.sum())}/{len(s)} ticks allowed, {allowed_fine} of them on the 12/16/24/32 grid",
            flush=True)

    def _win_allowed(self, i):
        """Bool [WINDOW] onset gate of window ``i`` (all False when rwo is off / not attached)."""
        if not self.rwo_on or self.onset_allowed is None:
            return None
        lo = int(i) * WINDOW
        out = np.zeros(WINDOW, bool)
        if lo >= len(self.onset_allowed) or lo < 0:
            return out
        seg = self.onset_allowed[lo:lo + WINDOW]
        out[:len(seg)] = seg
        return out

    def _rw_vec(self):
        """Per-tick offsets of the current chorus window: one shared log-ratio feedback of the realized fine-rhythm share
        against the official chorus target, plus a bounded reallocation among the four fine classes towards the official
        12th / 16th / 24th / 32nd split.  The 8th grid keeps 0 (its share falls by renormalisation only) and the fuzz
        ticks (48th / 64th) are never touched; the sampler renormalises among the time tokens, so the total probability
        mass of the time tokens is conserved.

        ``rwo``: the same offsets, but only on the ticks whose audio carries an onset (``_win_allowed``); a fine tick
        without an onset keeps its model probability, so mass only moves between positions the music marks."""
        q = self.rw_target
        if q is None or not self.rw_gain:
            return None
        allow = self._win_allowed(self._cur) if self.rwo_on else None
        if self.rwo_on and allow is None:
            return None
        total, prior = float(self.rw_counts.sum()), self.rw_prior
        t_fine = float(q[1:].sum())
        obs_fine = float(self.rw_counts[1:].sum())
        fine_share = (obs_fine + prior * t_fine) / (total + prior) if (total + prior) > 0 else t_fine
        err = math.log(max(t_fine, 1e-6) / max(fine_share, 1e-6))
        boost = float(np.clip(self.rw_gain * err, -self.rw_max, self.rw_max)) if abs(err) > self.rw_dead else 0.0
        offs = np.zeros(len(RW_CLASSES), np.float64)
        offs[1:] = boost
        if self.rw_rel > 0 and obs_fine > 0:      # intra-fine split: only reallocates among the four fine classes
            for k in range(1, len(RW_CLASSES)):
                share = (self.rw_counts[k] + prior * q[k]) / (obs_fine + prior * t_fine)
                rel = math.log(max(float(q[k]), 1e-6) / max(share, 1e-6))
                offs[k] += float(np.clip(self.rw_gain * rel, -self.rw_rel, self.rw_rel))
        self.stats["rw_offsets"] = [round(float(v), 3) for v in offs]
        if not offs.any():
            self.rw_vec = None
            return None
        vec = np.zeros(WINDOW)
        if allow is not None:                     # rwo: mask to the ticks that carry an onset
            free = int(np.count_nonzero(allow))
            if not free:
                self.rw_vec = None
                return None
            for k in range(len(RW_CLASSES)):
                if offs[k]:
                    vec[(self.rw_cls == k) & allow] = offs[k]
            self.stats["rwo_windows"] += 1
            self.stats["rwo_hits"] += int(np.count_nonzero(vec))
        else:
            for k in range(len(RW_CLASSES)):
                if offs[k]:
                    vec[self.rw_cls == k] = offs[k]
        if not vec.any():
            self.rw_vec = None
            return None
        self.stats["rw_windows"] += 1
        self.rw_vec = vec
        return vec

    # ------------------------------------------------------------------ pick
    def pick_penalty(self, notes, start_tick):
        """Rerank penalty (subtracted from the candidate score) of a chorus window far from the official chorus medians
        of its fine-rhythm share and its p90 bar density; 0 outside the chorus windows and below a dead band."""
        if "pick" not in self.parts or self.pick_w <= 0:
            return 0.0
        i = int(start_tick) // WINDOW
        if not self.is_chorus(i):
            return 0.0
        ticks = set()
        bars = np.zeros(BARS_PER_WINDOW, np.int32)
        for n in notes:
            rel = int(n.beat) - int(start_tick)
            b = rel // BAR
            if 0 <= b < BARS_PER_WINDOW:
                bars[b] += 1
            if n.kind in ("tap", "hold") or (n.kind == "slide" and not n.headless):
                ticks.add(rel)
        n_on = len(ticks)
        if n_on < 8:
            return 0.0
        self.stats["pick_onsets"] += 1
        fine = sum(1 for t in ticks if t % EIGHTH)
        t_fine = max(self.tg["rhythm_fine"][0], 1e-3)
        prior = 12.0
        share = (fine + prior * t_fine) / (n_on + prior)              # empirical-Bayes: sparse windows are pulled to the target
        t_p90 = max(self.tg["npb_p90"][0], 1.0)
        p90 = float(np.percentile(bars, 90))
        dev_f = max(0.0, abs(math.log((share + 1e-3) / (t_fine + 1e-3))) - self.pick_band)
        dev_p = max(0.0, math.log((t_p90 + 0.5) / (p90 + 0.5)) - self.pick_band)   # peak density: the shortfall is the defect (a denser bar is not)
        # linear, so that the term is on the scale of the other per-window penalties (official shortfall ~.2-.4 log -> .3-.7 score)
        pen = self.pick_w * (dev_f + dev_p)
        if pen > 0:
            self.stats["pick_hits"] += 1
        return float(min(pen, 1.5))

    # ------------------------------------------------------------------ diagnostics
    def info(self):
        return dict(strength=self.S, parts=sorted(self.parts), band=self.band, level=self.level, source=self.source,
                    chorus_windows=int(self.cmask.sum()), n_windows=self.nw,
                    frac=[round(float(v), 3) for v in self.frac], labels=self.labels,
                    targets={k: [round(v[0], 4), round(v[1], 4)] for k, v in self.tg.items()},
                    mults=[round(float(v), 3) for v in self.mults],
                    fine_ff=round(self.fine_ff, 3), stream_ff=round(self.stream_ff, 3),
                    pick=round(self.pick_w, 3), k_extra=round(self.k_extra, 3),
                    rw=None if self.rw_target is None else [round(float(v), 4) for v in self.rw_target],
                    rw_counts=[int(v) for v in self.rw_counts],
                    rwo=None if not self.rwo_on else dict(q=self.onset_q, tol=self.onset_tol,
                                                          thr=round(float(self.onset_thr), 5),
                                                          allowed=int(self.stats["rwo_allowed"]),
                                                          ticks=int(self.stats["rwo_ticks"])),
                    stats=dict(self.stats))


def build(gen, v3ctx, level, options, nw, log=print):
    """ChorusState for this song, or None (``--chorus 0`` -> the untouched generator)."""
    S = float(options.chorus)
    if S <= 0:
        return None
    if v3ctx is None:
        log("warning: --chorus needs the song context (bar table); chorus parts switched off", flush=True)
        return None
    tab = np.asarray(v3ctx.tab)
    nb = len(tab)
    spec = str(getattr(options, "chorus_section", "") or "pred").strip()
    if spec.startswith("file:"):
        path = Path(spec[5:])
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "starts" in raw and "labels" in raw:
            rec = raw
            source = "file:" + str(path)
        else:
            raise ValueError(f"--chorus-section {path}: needs a record {{nb, starts, labels}}")
    else:
        from . import plan as plan_mod
        rec = plan_mod.predicted_record(tab, nb)
        source = "pred"
    lab = _labels_of(rec, nb)
    chorus_bar = np.isin(lab, CHORUS_LABELS)
    nw = int(nw)
    frac = np.zeros(nw, np.float64)
    labels = []
    for i in range(nw):
        bars = chorus_bar[i * BARS_PER_WINDOW:(i + 1) * BARS_PER_WINDOW]
        frac[i] = float(bars.mean()) if len(bars) else 0.0
        ids = lab[i * BARS_PER_WINDOW:(i + 1) * BARS_PER_WINDOW]
        ids = [int(x) for x in ids if x >= 0]
        labels.append(LABELS[max(set(ids), key=ids.count)] if ids else "")
    lvl = float(level) if level is not None else 13.5
    if level is None:
        log("warning: --chorus without a level: using the 13.5 targets", flush=True)
    band = band_of(lvl)
    try:
        tg = load_targets(band, getattr(options, "chorus_targets", "") or None)
    except Exception as exc:                       # a missing targets.json must not kill a generation run
        log(f"warning: chorus targets unavailable ({exc}); dens / pick switched off", flush=True)
        tg = {k: (0.0, 0.0) for k in NEEDED}
    st = ChorusState(options, nw, frac, lvl, band, tg, labels, source)
    if not np.any(st.cmask):
        log(f"chorus: no chorus window predicted on this grid ({source}); parts inert", flush=True)
    log(f"chorus: strength {S:g} parts {','.join(sorted(st.parts))} band {band} source {source}, "
        f"{int(st.cmask.sum())}/{nw} chorus windows, targets npb {tg['npb'][0]:.2f}/{tg['npb'][1]:.2f} "
        f"fine {tg['rhythm_fine'][0]:.3f}/{tg['rhythm_fine'][1]:.3f} p90 {tg['npb_p90'][0]:.1f}/{tg['npb_p90'][1]:.1f} "
        f"(chorus/verse)", flush=True)
    return st
