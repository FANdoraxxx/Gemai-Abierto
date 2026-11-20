"""Audio activity curves on the chart grid, and the activity -> note-density model used for candidate reranking.

All functions are numpy-only (no torch).  ``mel`` is the 200 fps log-mel array of ``mai.audio.mel_db`` ([T, 96], dB).

* ``flux_frames(mel)``         positive spectral flux per mel frame (rise of the linear-amplitude mel spectrum versus the max
                               of the three previous frames; correlates 0.36 (Spearman, within song) with the per-half-bar
                               note count of official charts, versus 0.27 for mean level and 0.14 for dB flux).
* ``ActivityCurve``            integrates flux / energy / flux peaks over half-bars (2 beats = 96 ticks) of a tempo map and
                               exposes the song-relative activity ``rel`` (flux divided by the song's 90th percentile).
* ``DensityModel``             expected notes per second as a function of (level constant, relative activity); fitted from
                               the official MASTER / Re:MASTER charts by ``tools/fit_density.py``.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .audio import FPS

TPB = 48
HALF_BAR = 2 * TPB            # 96 ticks
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSITY_PATH = os.path.join(ROOT, "work", "density_model.json")


def flux_frames(mel):
    """float32 [T] onset strength per 5 ms frame: positive rise of the (linear amplitude) mel spectrum, summed over
    bands, versus the max of the three previous frames (sustained tones / vibrato do not count; loud onsets dominate)."""
    m = np.asarray(mel, np.float32)
    T = m.shape[0]
    if T < 5:
        return np.zeros(T, np.float32)
    lin = np.power(10.0, m / 20.0, dtype=np.float32)
    ref = np.empty_like(lin)
    ref[3:] = np.maximum(np.maximum(lin[2:-1], lin[1:-2]), lin[:-3])
    ref[:3] = lin[:3]
    return np.maximum(lin - ref, 0.0).sum(axis=1).astype(np.float32)


def energy_frames(mel):
    """float32 [T] mean mel level (dB)."""
    return np.asarray(mel, np.float32).mean(axis=1)


def power_frames(mel):
    """float32 [T] mean linear mel power per frame (an RMS^2 proxy; its dB value is the 'loudness' used by the playing / quiet test)."""
    return np.power(10.0, np.asarray(mel, np.float32) / 10.0, dtype=np.float32).mean(axis=1)


def peak_frames(flux, thresh_scale=1.5, min_gap=8):
    """0/1 peak indicator on flux: local maximum above thresh_scale * median of positive flux, >= min_gap frames apart."""
    flux = np.asarray(flux, np.float32)
    if len(flux) < 3:
        return np.zeros_like(flux)
    pos = flux[flux > 0]
    thr = thresh_scale * (float(np.median(pos)) if len(pos) else 0.0)
    is_peak = np.zeros(len(flux), bool)
    is_peak[1:-1] = (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > thr)
    idx = np.flatnonzero(is_peak)
    keep = []
    last = -10 ** 9
    for i in idx:
        if i - last >= min_gap:
            keep.append(i)
            last = i
        elif flux[i] > flux[last]:
            keep[-1] = i
            last = i
    out = np.zeros_like(flux)
    out[keep] = 1.0
    return out


def _integrate(series, f0, f1):
    """Mean of ``series`` over fractional frame intervals [f0, f1) (vectorised, linear interpolation)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(series, np.float64))])
    grid = np.arange(len(c), dtype=np.float64)
    s0 = np.interp(f0, grid, c)
    s1 = np.interp(f1, grid, c)
    return (s1 - s0) / np.maximum(f1 - f0, 1e-6)


class ActivityCurve:
    """Per-half-bar audio activity of one song on a given tempo map.

    ``tempo`` needs ``to_sec(beats)``; ``first`` is the chart's audio offset in seconds.
    """

    def __init__(self, mel, tempo, first, song_beats=None):
        flux = flux_frames(mel)
        energy = energy_frames(mel)
        peaks = peak_frames(flux)
        power = power_frames(mel)
        duration = len(flux) / FPS
        if song_beats is None:
            song_beats = float(tempo.to_beat(max(duration - first, 0.0)))
        self.n_half = max(1, int(math.ceil(song_beats * TPB / HALF_BAR)))
        edges_beats = np.arange(self.n_half + 1, dtype=np.float64) * (HALF_BAR / TPB)
        sec = tempo.to_sec(edges_beats) + float(first)
        frames = sec * FPS
        f0 = np.clip(frames[:-1], 0, len(flux))
        f1 = np.clip(frames[1:], 0, len(flux))
        valid = f1 - f0 > 1e-6
        self.seconds = np.maximum(np.diff(sec), 1e-6)
        self.flux = np.where(valid, _integrate(flux, f0, np.maximum(f1, f0 + 1e-3)), 0.0)
        self.peaks = np.where(valid, _integrate(peaks, f0, np.maximum(f1, f0 + 1e-3)) * FPS, 0.0)   # peaks per second
        self.energy = np.where(valid, _integrate(energy, f0, np.maximum(f1, f0 + 1e-3)), float(energy.min()))
        # loudness of the half-bar (dB of the mean mel power) and its distance to the song's median: "playing" = within PLAY_DB of the median
        self.loud = np.where(valid, 10.0 * np.log10(np.maximum(_integrate(power, f0, np.maximum(f1, f0 + 1e-3)), 1e-12)), -120.0)
        self.loud_med = float(np.median(self.loud[valid])) if valid.any() else 0.0
        self.loud_rel = self.loud - self.loud_med
        scale = float(np.percentile(self.flux[valid], 90)) if valid.any() else 0.0
        self.scale = max(scale, 1e-6)
        self.rel = self.flux / self.scale
        pk_scale = float(np.percentile(self.peaks[valid], 90)) if valid.any() else 0.0
        self.peaks_rel = self.peaks / max(pk_scale, 1e-6)

    def window(self, start_tick, n_half=8):
        """Slice for the half-bars of the window starting at ``start_tick`` (zero padded past the song end)."""
        h0 = start_tick // HALF_BAR
        idx = np.arange(h0, h0 + n_half)
        ok = idx < self.n_half
        out = {}
        for name in ("rel", "peaks_rel", "seconds", "energy", "loud_rel"):
            arr = getattr(self, name)
            out[name] = np.where(ok, arr[np.minimum(idx, self.n_half - 1)], 0.0 if name != "seconds" else 1.0)
        out["valid"] = ok
        return out


class DensityModel:
    """Expected note density: nps(lv, rel) from a table fitted on official charts.

    ``levels``: ascending level knots; ``rel_edges``: bin edges of relative activity; ``table[l][b]``: mean notes per
    second of half-bars in activity bin b for charts of level knot l (smooth in level by linear interpolation).
    """

    def __init__(self, d):
        self.d = d
        self.levels = np.asarray(d["levels"], np.float64)
        self.rel_edges = np.asarray(d["rel_edges"], np.float64)
        self.table = np.asarray(d["table"], np.float64)        # [L, B]
        self.song_scale = d.get("song_scale")

    @classmethod
    def load(cls, path=DENSITY_PATH):
        with open(path, encoding="utf-8") as stream:
            return cls(json.load(stream))

    def nps(self, level, rel):
        """Expected notes/second for activity ``rel`` (array) at level constant ``level``."""
        rel = np.minimum(np.asarray(rel, np.float64), float(self.rel_edges[-2]) * 1.33)   # last bin is sparse: saturate
        b = np.clip(np.searchsorted(self.rel_edges, rel, side="right") - 1, 0, self.table.shape[1] - 2)
        lv = float(np.clip(level, self.levels[0], self.levels[-1]))
        i = int(np.clip(np.searchsorted(self.levels, lv, side="right") - 1, 0, len(self.levels) - 2))
        w = (lv - self.levels[i]) / max(self.levels[i + 1] - self.levels[i], 1e-9)
        col = (1 - w) * self.table[i] + w * self.table[i + 1]
        return col[b]

    def expected_notes(self, level, rel, seconds):
        return float(np.sum(self.nps(level, rel) * np.asarray(seconds, np.float64)))


# ----------------------------------------------------------------------------------------------
# Per-window expected note count used by the window scorer, the guard and the density feedback.
#
#   E_w = S(level, bpm) * seconds_w * ratio_w
#   S(level, bpm)  official MiLK+ MASTER / Re:MASTER notes per second:  log S = A + B (level - 13.5) + C log(bpm / 160)
#                  (1176 train charts with versionid >= 10, chart-level residual std .155 in log; scratch/density_fix/an4.py)
#   ratio_w        the window's intensity relative to the song's own mean:  exp(BETA_A x_audio + BETA_P x_plan) normalised to a
#                  seconds-weighted mean of 1 over the song and clipped to [RATIO_LO, RATIO_HI];  x_audio = log nps_table(level, rel)
#                  of the window minus its song mean (the flux-based table of ``DensityModel``), x_plan = log of the PLAN bucket's
#                  window / song-mean ratio minus its song mean (seq_version 3 only).
#   The within-song window counts of official charts are log-normal around this with std ~.37 (audio only .363 train / .369 val, audio + PLAN
#   .369 on the 25 validation songs), so no single window is "wrong" by less than ~ +-.4 in log: the penalties built on E use a dead band.
EXPECT_PARAMS = dict(A=1.618, B=0.203, C=0.309, BETA_AUDIO_ONLY=1.40, BETA_A=0.95, BETA_P=0.45, RATIO_LO=0.12, RATIO_HI=2.0)
WINDOW_HALF = 8


def level_nps(level, bpm, params=None):
    """Official (MiLK+ MASTER / Re:MASTER) notes per second of a chart of this level constant and tempo."""
    p = params or EXPECT_PARAMS
    return math.exp(p["A"] + p["B"] * (float(level) - 13.5) + p["C"] * math.log(max(float(bpm), 40.0) / 160.0))


class WindowExpectation:
    """Expected note count of every 4-bar window (8 half-bars) of one song, see the block comment above.

    ``curve`` = ActivityCurve, ``density`` = DensityModel, ``level`` the requested / predicted level constant,
    ``plan_ratios`` = optional list (window index -> typical window / song-mean density ratio of the PLAN bucket),
    ``scale`` multiplies every E (``--density-scale``)."""

    def __init__(self, curve, density, level, plan_ratios=None, scale=1.0, params=None):
        p = dict(EXPECT_PARAMS)
        if params:
            p.update(params)
        self.p = p
        nh = curve.n_half
        nw = -(-nh // WINDOW_HALF)
        pad = nw * WINDOW_HALF - nh
        sec_h = np.concatenate([curve.seconds, np.zeros(pad)])
        rel_h = np.concatenate([curve.rel, np.zeros(pad)])
        nps_h = np.where(sec_h > 0, density.nps(level, rel_h), 0.0)
        self.sec = sec_h.reshape(nw, WINDOW_HALF).sum(1)                       # valid seconds per window
        notes_tab = (nps_h * sec_h).reshape(nw, WINDOW_HALF).sum(1)
        ok = self.sec > 1e-6
        nps_w = np.where(ok, notes_tab / np.maximum(self.sec, 1e-6), 1.0)
        wsum = max(float(self.sec[ok].sum()), 1e-6)
        x_a = np.log(np.maximum(nps_w, 1e-3))
        x_a = x_a - float((x_a * self.sec)[ok].sum() / wsum)
        x = p["BETA_AUDIO_ONLY"] * x_a
        self.has_plan = plan_ratios is not None and len(plan_ratios) >= nw
        if self.has_plan:
            x_p = np.log(np.maximum(np.asarray(plan_ratios[:nw], np.float64), 0.05))
            x_p = x_p - float((x_p * self.sec)[ok].sum() / wsum)
            x = p["BETA_A"] * x_a + p["BETA_P"] * x_p
        ratio = np.exp(x)
        ratio = ratio / float((ratio * self.sec)[ok].sum() / wsum)           # seconds-weighted mean 1
        self.ratio = np.clip(ratio, p["RATIO_LO"], p["RATIO_HI"])
        ratio_mean = float((self.ratio * self.sec)[ok].sum() / wsum)
        self.ratio = self.ratio / ratio_mean                                  # keep the mean at 1 after the clip
        bpm = 120.0 * float(np.count_nonzero(curve.seconds > 0)) / max(float(curve.seconds.sum()), 1e-6)
        self.bpm = bpm
        self.level = float(level)
        constant = bool(ok.any() and np.ptp(self.sec[ok]) < 1e-6)
        if constant:
            self.nps = level_nps(level, bpm, p) * float(scale)                # song mean notes per second
            self.expected = self.nps * self.sec * self.ratio
        else:
            # 8 half-bars = 16 beats: bpm_w = 16 * 60 / sec_w.  Score every window
            # against its own tempo baseline, then apply the (per-bar) intensity ratio once.
            bpm_w = np.where(ok, 960.0 / np.maximum(self.sec, 1e-6), bpm)
            base_w = np.exp(p["A"] + p["B"] * (float(level) - 13.5)
                            + p["C"] * np.log(np.maximum(bpm_w, 40.0) / 160.0)) * float(scale)
            self.nps_w = np.where(ok, base_w, 0.0)
            self.expected = self.nps_w * self.sec * self.ratio
            self.nps = float(self.expected[ok].sum() / max(float(self.sec[ok].sum()), 1e-6))
        self.n_windows = nw
        self.ok = ok

    def window(self, start_tick):
        """Expected notes of the window starting at ``start_tick`` (a multiple of 768) or None past the song."""
        w = int(start_tick) // (WINDOW_HALF * HALF_BAR)
        if w < 0 or w >= self.n_windows or not self.ok[w]:
            return None
        return float(self.expected[w])


_MODEL = None


def get_density_model(path=None):
    global _MODEL
    if path is not None:
        return DensityModel.load(path)
    if _MODEL is None:
        _MODEL = DensityModel.load()
    return _MODEL
