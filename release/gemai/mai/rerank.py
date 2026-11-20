"""Per-window candidate scoring: audio-activity fit, repetition and density-versus-level penalties.

``WindowScorer`` keeps the chart-level history (bar signatures, trailing silence) and scores each candidate window as
``w_logprob * mean_logprob - w_activity * P_activity - w_repeat * P_repeat - w_density * P_density`` (higher is
better).  All three penalties are 0 for an unremarkable window, so the (down-weighted) likelihood decides unless a
candidate has a visible defect:

* P_activity  (half-bars = 2 beats, 8 per window) a half-bar with no note starting in it and no hold/slide running through it
  while the song-relative audio activity is high; consecutive empty half-bars (also across the previous window) cost more.
* P_repeat    identical consecutive bars (a pair is mildly penalised, three or more strongly), ABAB two-bar loops, and
  mildly any non-adjacent repeat of a bar with at least ``min_notes`` notes; plus runs of bars with the same *shape*
  (rhythm, note kinds and relative button motion, a third consecutive bar is already penalised) and of bars with the
  same rhythm and kinds (from the fourth on) -- see ``barsig``.
* P_density   squared log-ratio between the window's note count and the count expected for the requested level
  (``DensityModel``) beyond a dead band.  Off when no level is requested.

v3_sens3 (``expect`` given): the expected count is ``activity.WindowExpectation`` (official nps of the level at this tempo x the window's intensity
relative to the song, audio + PLAN); the density penalty is two-sided with a wide dead band (the official window counts scatter by +-.37 around any
such estimate) and capped; the rank-correlation term (``intensity``) gets no credit for windows far below the expected count; and a GUARD ranks
candidates before the score is looked at (``guard`` in the parts): a candidate is in violation when
  * ``empty``  >= 3 of the 8 half-bars are empty (no note start, no hold / slide running) while the audio plays there (loudness within 3 dB of the
               song median; official charts: 0.34 % of the windows), or an empty-and-playing run of >= 4 half-bars crosses the window border,
  * ``floor``  the window plays and has fewer than 40 % of the expected notes (expected >= 8),
  * ``streak`` the previous window was below 60 % of its expected count and this one is too.
The generator picks the candidate with the fewest violations first (muri / score second) and decodes one more round (denser tilt) when every
candidate of a window violates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from . import barsig, breaks as brk_utils
from .activity import HALF_BAR, TPB

BAR = 4 * TPB
N_HALF = 8
N_BARS = 4


def bar_signature(notes, bar_start):
    """Hashable description of a bar's notes (identical to scratch/ctl/defects.py so that metric and reranker agree)."""
    out = []
    for n in notes:
        arm = (tuple((a.wait, tuple((s.shape, s.end) for s in a.segs), tuple(a.groups)) for a in n.arms)
               if n.kind == "slide" else ())
        out.append((int(n.beat) - bar_start, n.kind, n.pos, int(n.dur), arm, n.brk, n.ex))
    return tuple(sorted(out, key=repr))


def window_bars(notes, start_tick, n_bars=N_BARS):
    bars = [[] for _ in range(n_bars)]
    for n in notes:
        b = (int(n.beat) - start_tick) // BAR
        if 0 <= b < n_bars:
            bars[b].append(n)
    return bars


def window_bar_signatures(notes, start_tick, n_bars=N_BARS):
    bars = window_bars(notes, start_tick, n_bars)
    return [bar_signature(b, start_tick + i * BAR) if b else () for i, b in enumerate(bars)]


def window_pattern_signatures(notes, start_tick, n_bars=N_BARS):
    """(exact, shape, rk) signature lists of the window's bars; empty bars are ``()``."""
    bars = window_bars(notes, start_tick, n_bars)
    exact = [bar_signature(b, start_tick + i * BAR) if b else () for i, b in enumerate(bars)]
    shape = [barsig.shape_signature(b, start_tick + i * BAR) if b else () for i, b in enumerate(bars)]
    rk = [barsig.rk_signature(b, start_tick + i * BAR) if b else () for i, b in enumerate(bars)]
    return exact, shape, rk


@dataclass
class RerankWeights:
    logprob: float = 0.0         # weight of the mean token log-probability (likelihood alone prefers dense, looping windows)
    activity: float = 0.5
    repeat: float = 1.0
    density: float = 1.0
    intensity: float = 0.0       # weight of the note-count vs audio-flux rank correlation penalty over the 8 half-bars (0 = off)
    act_lo: float = 0.4          # relative activity below which an empty half-bar is not penalised
    act_hi: float = 0.8          # ... and above which it is penalised in full
    density_dead_band: float = 0.15
    min_notes: int = 2           # bars with fewer notes are never "repeats"
    pattern_min_notes: int = 3   # ... and never part of a shape / rhythm run
    pair: float = 1.0            # cost of a bar identical to the previous one (a third in a row costs 2+, see repetition_penalty)
    far: float = 0.6             # cost of a bar identical to an earlier, non-adjacent one (motif repetition; official charts: 7 %)
    # --- v3_sens3 (only used when the scorer has an ``expect`` model)
    band: float = 0.35           # dead band of the two-sided density penalty (log units)
    density_cap: float = 1.5     # upper limit of the density penalty
    credit_floor: float = 0.7    # the rank-correlation credit is scaled by min(1, notes / (credit_floor * expected))
    guard_empty: int = 3         # empty playing half-bars (of 8) that make a candidate a violation (0 = rule off)
    guard_run: int = 4           # ... empty playing run across the window border (0 = off)
    guard_floor: float = 0.4     # fewer than this share of the expected notes in a playing window is a violation (0 = off)
    guard_streak: float = 0.6    # ... two windows in a row below this share (0 = off)
    guard_play_db: float = 3.0   # a half-bar plays when its loudness is within this many dB of the song median
    guard_min_expected: float = 8.0   # the floor / streak rules need at least this many expected notes
    brk_timing: float = 1.0      # musical BRK timing penalty / phrase-edge reward


class WindowScorer:
    def __init__(self, activity=None, density=None, level=None, weights=None, brk_timing=True):
        self.activity = activity
        self.density = density
        self.level = level
        self.w = weights or RerankWeights()
        self.sigs = []            # one signature per bar of the chart so far
        self.shapes = []          # ... shape signatures
        self.rks = []             # ... rhythm + kind signatures
        self.seen = Counter()
        self.empty_run = 0        # trailing empty (uncovered, non-quiet) half-bars of the accepted chart
        self.density_bias = 0.0   # EMA of log((n+1)/(E+1)) of accepted windows (diagnostic)
        self.scale = 1.0          # multiplier on the expected note count (--density-scale); applies to the penalty and to the steering
        self.expect = None        # activity.WindowExpectation (v3_sens3); None = the table-based expectation above
        self.play_run = 0         # trailing empty-and-playing half-bars of the accepted chart (guard)
        self.prev_sparse = False  # the previous accepted window was below ``guard_streak`` of its expected count
        self.chorus = None        # mai.chorus.ChorusState (--chorus > 0): extra ``pick`` penalty of the chorus windows
        self.brk_timing = bool(brk_timing)

    # ------------------------------------------------------------------ pieces
    def _half_bars(self, notes, start_tick, carried_hold_end=0):
        occupied = np.zeros(N_HALF, bool)
        count = np.zeros(N_HALF, np.int32)
        for n in notes:
            h = (int(n.beat) - start_tick) // HALF_BAR
            if 0 <= h < N_HALF:
                count[h] += 1
                occupied[h] = True
            if n.kind in ("hold", "thold", "slide"):
                end = int(n.end_beat())
                h0 = max(0, (int(n.beat) - start_tick) // HALF_BAR)
                h1 = min(N_HALF - 1, (end - 1 - start_tick) // HALF_BAR)
                if h1 >= h0:
                    occupied[h0:h1 + 1] = True
        if carried_hold_end > start_tick:
            h1 = min(N_HALF - 1, (carried_hold_end - 1 - start_tick) // HALF_BAR)
            occupied[:h1 + 1] = True
        return count, occupied

    def activity_penalty(self, notes, start_tick, carried_hold_end=0):
        if self.activity is None or self.w.activity <= 0:
            return 0.0, 0, 0
        win = self.activity.window(start_tick, N_HALF)
        count, occupied = self._half_bars(notes, start_tick, carried_hold_end)
        rel = win["rel"]
        weight = np.clip((rel - self.w.act_lo) / max(self.w.act_hi - self.w.act_lo, 1e-6), 0.0, 1.0)
        empty = (~occupied) & win["valid"]
        run = self.empty_run
        pen = 0.0
        for h in range(N_HALF):
            if empty[h] and weight[h] > 0:
                run += 1
                pen += weight[h] * (1.0 + 0.5 * (run - 1))
            elif not empty[h]:
                run = 0                                         # a quiet empty half-bar neither extends nor resets
        return pen / N_HALF, run, int(empty.sum())

    def repetition_penalty(self, cand_sigs):
        """Per bar: identical to the previous bar 1.0 (three in a row 2.0 and growing), ABAB loop 1.0, any earlier
        identical bar 0.6; averaged over the window's bars."""
        pen = 0.0
        seq = self.sigs + list(cand_sigs)
        n0 = len(self.sigs)
        min_notes = self.w.min_notes
        for i in range(n0, len(seq)):
            s = seq[i]
            if len(s) < min_notes:
                continue
            r, j = 1, i - 1
            while j >= 0 and seq[j] == s:
                r += 1
                j -= 1
            if r == 2:
                pen += self.w.pair
            elif r >= 3:
                pen += 2.0 + (r - 3)
            elif i >= 3 and seq[i - 2] == s and seq[i - 3] == seq[i - 1] and len(seq[i - 1]) >= min_notes:
                pen += 1.0
            elif self.seen[s] > 0:
                pen += self.w.far
        return pen / max(1, len(cand_sigs))

    def pattern_penalty(self, cand_shapes, cand_rks):
        """Runs of bars with equal shape (3rd consecutive bar on) or equal rhythm and kinds (4th on)."""
        pen = 0.0
        m = self.w.pattern_min_notes

        def shape_cost(r):           # r-th consecutive bar with the same shape
            return 0.3 if r == 2 else (1.2 if r == 3 else 2.0 + 0.7 * (r - 4))

        def rk_cost(r):              # r-th consecutive bar with the same rhythm and kinds
            return min(1.5, 0.25 * (r - 3))
        for hist, cand, first_run, cost in ((self.shapes, cand_shapes, 2, shape_cost), (self.rks, cand_rks, 4, rk_cost)):
            seq = hist + list(cand)
            for i in range(len(hist), len(seq)):
                s = seq[i]
                if len(s) < m:
                    continue
                r, j = 1, i - 1
                while j >= 0 and seq[j] == s:
                    r += 1
                    j -= 1
                if r >= first_run:
                    pen += cost(r)
        return pen / max(1, len(cand_shapes))

    def expected_notes(self, start_tick):
        """Expected notes of the window: the scorer's (penalty / guard) expectation."""
        if self.expect is not None:
            return self.expect.window(start_tick)
        return self.expected_notes_table(start_tick)

    def expected_notes_table(self, start_tick):
        """The flux-table expectation (density feedback of the steering; ``scale`` applies)."""
        if self.level is None or self.density is None or self.activity is None:
            return None
        win = self.activity.window(start_tick, N_HALF)
        ok = win["valid"]
        if not ok.any():
            return None
        return self.scale * self.density.expected_notes(self.level, win["rel"][ok], win["seconds"][ok])

    def intensity_penalty(self, notes, start_tick):
        """(1 - Spearman(note starts per half-bar, audio flux per half-bar)) / 2 over the window's 8 half-bars: 0 = counts follow the audio exactly, 0.5 =
        unrelated (also when either side is constant), 1 = inverted."""
        if self.activity is None or self.w.intensity <= 0:
            return 0.0
        win = self.activity.window(start_tick, N_HALF)
        ok = win["valid"]
        if ok.sum() < 4:
            return 0.0
        count, _ = self._half_bars(notes, start_tick)
        a, b = count[ok].astype(float), np.asarray(win["rel"], float)[ok]
        if a.std() == 0 or b.std() == 0:
            return 0.5
        from scipy.stats import rankdata
        c = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
        return (1.0 - c) / 2.0

    def brk_timing_penalty(self, notes, start_tick):
        """BRK placement defects plus a reward for phrase-edge / audio-peak breaks.

        Positive values are penalties.  This term deliberately works on the
        candidate notes only; the official GT is used by the DPO scorer.
        """
        if not self.brk_timing or self.w.brk_timing <= 0:
            return 0.0, {}
        f = brk_utils.window_break_features(notes, start_tick)
        ts = brk_utils.break_timestamps(notes)
        if ts and self.activity is not None:
            vals = self.activity.flux
            order = np.sort(vals)
            ranks = []
            for t in ts:
                h = max(0, min(self.activity.n_half - 1, int(t) // (2 * TPB)))
                ranks.append(float(np.searchsorted(order, vals[h], side="right")) / max(1, len(order)))
            f["brk_audio_onset_rank"] = float(np.mean(ranks)) if ranks else 0.0
            loud = self.activity.loud
            jumps = []
            for t in ts:
                h = max(0, min(self.activity.n_half - 1, int(t) // (2 * TPB)))
                lo, hi = max(0, h - 2), min(self.activity.n_half, h + 3)
                others = np.concatenate([loud[lo:h], loud[h + 1:hi]])
                if len(others):
                    jumps.append(abs(float(loud[h]) - float(np.mean(others))))
            f["brk_audio_loud_jump"] = float(np.mean(jumps)) if jumps else 0.0

        pen = 2.0 * float(f.get("brk_asym_chord_n", 0.0))
        bars = brk_utils.break_count_per_bar(notes)
        for b, cnt in bars.items():
            if int(b) != 3 and cnt > 2:
                pen += 1.2 * (cnt - 2)
            if int(b) == 3 and cnt > 3:
                pen += 0.8 * (cnt - 3)
        pen += 0.8 * (int(f.get("brk_ts_n", 0)) - 6) if int(f.get("brk_ts_n", 0)) > 6 else 0.0
        pen += 0.6 * float(f.get("brk_off16_frac", 0.0))
        if f.get("brk_ts_n", 0):
            pen -= 0.55 * float(f.get("brk_phrase_edge_frac", 0.0))
            pen -= 0.20 * float(f.get("brk_downbeat_frac", 0.0))
            pen -= 0.35 * min(1.0, max(0.0, float(f.get("brk_audio_onset_rank", 0.0)) - 0.45) / 0.55)
        return float(pen), f

    def nps_profile(self, start_tick, n=768):
        """Expected notes/second at each of ``n`` ticks from ``start_tick`` (the half-bar's activity through the level's density table); NaN past the song."""
        if self.level is None or self.density is None or self.activity is None:
            return None
        h = (start_tick + np.arange(n)) // HALF_BAR
        ok = h < self.activity.n_half
        rel = self.activity.rel[np.minimum(h, self.activity.n_half - 1)]
        return np.where(ok, self.scale * self.density.nps(self.level, rel), np.nan)

    def mean_nps(self):
        """Time-weighted mean of the expected notes/second over the whole song."""
        if self.level is None or self.density is None or self.activity is None:
            return None
        nps = self.scale * self.density.nps(self.level, self.activity.rel)
        sec = self.activity.seconds
        return float((nps * sec).sum() / max(sec.sum(), 1e-6))

    def density_penalty(self, n_notes, start_tick):
        e = self.expected_notes(start_tick)
        if e is None or self.w.density <= 0:
            return 0.0, e
        dev = abs(math.log((n_notes + 1.0) / (e + 1.0)))
        if self.expect is not None:
            band = self.w.band
            if self.chorus is not None and self.chorus.band_override is not None and self.chorus.is_chorus(int(start_tick) // 768):
                band = self.chorus.band_override          # dens: the chorus windows hold the official chorus density much tighter
            return min(max(0.0, dev - band) ** 2, self.w.density_cap), e
        return max(0.0, dev - self.w.density_dead_band) ** 2, e

    def guard_check(self, notes, start_tick, carried_hold_end, n_notes, expected):
        """-> dict(empty_play, lead, trail, playing, empty, run, floor, streak, viol): see the module docstring."""
        out = dict(empty_play=0, playing=0.0, empty=False, run=False, floor=False, streak=False, viol=0, trail=0, sparse=False)
        if self.activity is None:
            return out
        win = self.activity.window(start_tick, N_HALF)
        count, occupied = self._half_bars(notes, start_tick, carried_hold_end)
        ok = win["valid"]
        play = ok & (win["loud_rel"] >= -self.w.guard_play_db)
        epl = (~occupied) & play
        out["empty_play"] = int(epl.sum())
        out["playing"] = float(play.sum() / max(1, ok.sum()))
        lead = 0
        for h in range(N_HALF):
            if epl[h]:
                lead += 1
            else:
                break
        trail = 0
        for h in range(N_HALF - 1, -1, -1):
            if epl[h]:
                trail += 1
            else:
                break
        out["trail"] = (self.play_run + trail) if trail == N_HALF else trail     # all 8 empty: the earlier run continues
        if self.w.guard_empty > 0 and out["empty_play"] >= self.w.guard_empty:
            out["empty"] = True
        if self.w.guard_run > 0 and (self.play_run + lead >= self.w.guard_run and lead > 0):
            out["run"] = True
        playing = out["playing"] >= 0.75
        if expected is not None and expected >= self.w.guard_min_expected and playing:
            if self.w.guard_floor > 0 and n_notes < self.w.guard_floor * expected:
                out["floor"] = True
            if self.w.guard_streak > 0 and n_notes < self.w.guard_streak * expected:
                out["sparse"] = True
                if self.prev_sparse:
                    out["streak"] = True
        out["viol"] = int(out["empty"]) + int(out["run"]) + int(out["floor"]) + int(out["streak"])
        return out

    # ------------------------------------------------------------------ API
    def score(self, notes, start_tick, mean_logprob, carried_hold_end=0, muri=None):
        """Return (score, parts).  ``notes`` are the candidate's new notes (absolute ticks, canonical order).
        ``muri`` = the candidate's ``MuriGate.assess`` entry (None = no muri term): its ``penalty`` is subtracted."""
        p_act, run, n_empty = self.activity_penalty(notes, start_tick, carried_hold_end)
        sigs, shapes, rks = window_pattern_signatures(notes, start_tick)
        p_rep = (self.repetition_penalty(sigs) + self.pattern_penalty(shapes, rks)) if self.w.repeat > 0 else 0.0
        p_den, expected = self.density_penalty(len(notes), start_tick)
        p_int = self.intensity_penalty(notes, start_tick) if self.w.intensity > 0 else 0.0
        p_brk, brk_parts = self.brk_timing_penalty(notes, start_tick)
        if self.expect is not None and expected is not None and self.w.intensity > 0 and self.w.credit_floor > 0:
            # a sparse window must not win on the correlation term: its credit (0.5 - p_int) shrinks with notes / expected
            adequacy = min(1.0, len(notes) / max(self.w.credit_floor * expected, 1.0))
            p_int = 0.5 - (0.5 - p_int) * adequacy
        guard = self.guard_check(notes, start_tick, carried_hold_end, len(notes), expected) if self.expect is not None else None
        total = (self.w.logprob * mean_logprob - self.w.activity * p_act - self.w.repeat * p_rep - self.w.density * p_den
                 - self.w.intensity * p_int - self.w.brk_timing * p_brk)
        p_muri = float(muri["penalty"]) if muri is not None and muri.get("ok") else 0.0
        total -= p_muri
        parts = dict(mean_logprob=float(mean_logprob), p_activity=float(p_act), p_repeat=float(p_rep),
                     p_density=float(p_den), p_intensity=float(p_int), n_notes=len(notes), expected=None if expected is None else float(expected),
                     empty_half_bars=n_empty, p_brk_timing=float(p_brk), score=float(total))
        for k, v in brk_parts.items():
            parts[k] = float(v)
        if self.chorus is not None:
            # pick: keep the chorus windows near the official chorus medians of the fine-rhythm share / p90 bar density
            p_ch = float(self.chorus.pick_penalty(notes, start_tick))
            total -= p_ch
            parts["p_chorus"] = p_ch
            parts["score"] = float(total)
        if muri is not None:
            parts["p_muri"] = p_muri
            parts["muri_counts"] = dict(muri["counts"])
        if guard is not None:
            parts["guard"] = int(guard["viol"])
            parts["guard_detail"] = dict(empty_play=guard["empty_play"], playing=round(guard["playing"], 2), empty=guard["empty"],
                                         run=guard["run"], floor=guard["floor"], streak=guard["streak"])
            parts["_trail"] = guard["trail"]
            parts["_sparse"] = guard["sparse"]
        parts["_run"] = run
        parts["_sigs"] = sigs
        parts["_shapes"] = shapes
        parts["_rks"] = rks
        return float(total), parts

    def accept(self, parts):
        """Fold the chosen candidate into the chart history."""
        for s in parts["_sigs"]:
            self.sigs.append(s)
            if len(s) >= self.w.min_notes:
                self.seen[s] += 1
        self.shapes.extend(parts.get("_shapes", ()))
        self.rks.extend(parts.get("_rks", ()))
        self.empty_run = int(parts["_run"])
        if "_trail" in parts:
            self.play_run = int(parts["_trail"])
            self.prev_sparse = bool(parts.get("_sparse", False))
        if parts.get("expected") is not None:
            ratio = math.log((parts["n_notes"] + 1.0) / (parts["expected"] + 1.0))
            self.density_bias = 0.7 * self.density_bias + 0.3 * ratio
