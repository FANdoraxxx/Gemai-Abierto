"""Audio -> level constant predictor for ``--level auto`` (numpy only).

The model's LV token is unknown in auto mode, so generated charts follow the average density of all training charts.  A
charter would pick the difficulty from the song: ``predict_level`` regresses the MASTER level constant of the official
charts from song-level audio statistics (onset-peak rates, spectral-flux level and spread, band energies, BPM) with a ridge
regression (``tools/fit_levelpred.py`` -> ``work/levelpred.json``; fitted on the train split only).  The generator then
conditions on the predicted level exactly as if it had been requested.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import activity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "work", "levelpred.json")
FPS = 200.0
KEYS = ["dur", "act_frac", "peaks_ps_1.0", "peaks_ps_1.5", "peaks_ps_2.5", "flux_mean", "flux_p90", "flux_p50", "flux_cv",
        "energy_mean", "energy_std", "flux_band0", "flux_band1", "flux_band2", "pk_s_p50", "pk_s_p90", "pk_s_p99", "bpm"]


def song_features(mel, bpm):
    """Flat dict of song-level audio statistics of the 200 fps log-mel array ``mel`` [T, 96] (dB) and the chart BPM."""
    mel = np.asarray(mel, np.float32)
    flux = activity.flux_frames(mel)
    en = activity.energy_frames(mel)
    T = len(flux)
    dur = T / FPS
    f = {}
    act = en > np.percentile(en, 15)                    # active frames: above the 15th percentile of the mean level
    secs_act = max(act.sum() / FPS, 1.0)
    f["dur"] = dur
    f["act_frac"] = float(act.mean())
    for sc in (1.0, 1.5, 2.5):
        pk = activity.peak_frames(flux, thresh_scale=sc)
        f[f"peaks_ps_{sc}"] = float(pk.sum() / secs_act)
    pos = flux[act]
    f["flux_mean"] = float(pos.mean()) if len(pos) else 0.0
    f["flux_p90"] = float(np.percentile(flux[act], 90)) if act.any() else 0.0
    f["flux_p50"] = float(np.percentile(flux[act], 50)) if act.any() else 0.0
    f["flux_cv"] = float(pos.std() / max(pos.mean(), 1e-9)) if len(pos) else 0.0
    f["energy_mean"] = float(en[act].mean()) if act.any() else 0.0
    f["energy_std"] = float(en[act].std()) if act.any() else 0.0
    lin = np.power(10.0, mel / 20.0)
    d = np.maximum(np.diff(lin, axis=0), 0.0)
    for i, (a, b) in enumerate(((0, 32), (32, 64), (64, 96))):
        f[f"flux_band{i}"] = float(d[:, a:b].sum() / max(secs_act, 1.0))
    pk = activity.peak_frames(flux, thresh_scale=1.5)
    n_s = int(dur)
    per_s = pk[:n_s * 200].reshape(n_s, 200).sum(1) if n_s > 4 else np.zeros(1)
    f["pk_s_p50"] = float(np.percentile(per_s, 50))
    f["pk_s_p90"] = float(np.percentile(per_s, 90))
    f["pk_s_p99"] = float(np.percentile(per_s, 99))
    f["bpm"] = float(bpm)
    return f


class LevelPredictor:
    def __init__(self, d):
        self.d = d
        self.keys = list(d["keys"])
        self.mu = np.asarray(d["mu"], np.float64)
        self.sd = np.asarray(d["sd"], np.float64)
        self.w = np.asarray(d["w"], np.float64)
        self.b = float(d["b"])
        self.lo, self.hi = float(d.get("lo", 11.0)), float(d.get("hi", 15.0))

    @classmethod
    def load(cls, path=PATH):
        with open(path, encoding="utf-8") as stream:
            return cls(json.load(stream))

    def predict_features(self, f):
        x = np.asarray([f[k] for k in self.keys], np.float64)
        return float(np.clip(((x - self.mu) / self.sd) @ self.w + self.b, self.lo, self.hi))

    def predict(self, mel, bpm):
        return self.predict_features(song_features(mel, bpm))


_MODEL = None


def get_predictor():
    global _MODEL
    if _MODEL is None:
        _MODEL = LevelPredictor.load()
    return _MODEL
