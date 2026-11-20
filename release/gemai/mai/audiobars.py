"""Audio-conditional kind prior (``--kind-prior G``).

A small ridge probe (fitted by scratch/sens/build_model.py on TRAIN official Master / Re:MASTER charts, exported to
``work/audio_kind_prior.npz``) predicts, from the audio of one bar (MERT-L8 mean embedding -> PCA-32 -> 16 clusters, plus flux / centroid /
percussive ratio / stationarity / loudness) how far the bar's hold / slide / BRK / touch / double shares deviate from the song's own mean.
The deviations are turned into logit offsets on the corresponding tokens, so that e.g. sustained, legato passages get more holds and slides and
percussive ones fewer, instead of the song-wide share feedback and the marginal calibration pulling every window towards the population mean.
Everything is relative to the song (features are song-demeaned), so the song-level shares stay under the existing feedback.
"""
from __future__ import annotations

import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR_PATH = os.path.join(ROOT, "work", "audio_kind_prior.npz")
KEYS = ["logn", "hold", "slide", "long", "brk", "touch", "double", "ex"]
BASE_SHARE = {"hold": 0.08, "slide": 0.14, "brk": 0.04, "touch": 0.06, "double": 0.26}
FPS_MEL = 200.0
FPS_FEAT = 75.0


def _cum(a):
    c = np.zeros((a.shape[0] + 1,) + a.shape[1:], np.float64)
    np.cumsum(a, axis=0, out=c[1:])
    return c


def _pool(c, f0, f1):
    f0 = np.clip(f0, 0, len(c) - 1)
    f1 = np.clip(np.maximum(f1, f0 + 1), 0, len(c) - 1)
    n = np.maximum(f1 - f0, 1).astype(np.float64)
    return (c[f1] - c[f0]) / (n[:, None] if c.ndim == 2 else n)


def bar_audio(mel, mert, tempo, first, nb):
    """raw per-bar audio features: emb [nb, 1024] (zeros when ``mert`` is None), desc [nb, 5] (log flux, centroid, perc, stat, loud)"""
    from . import activity
    mel = np.asarray(mel, np.float32)
    edges = tempo.to_sec(np.arange(nb + 1) * 4.0) + float(first)
    lin = np.power(10.0, mel / 20.0, dtype=np.float32)
    flux = activity.flux_frames(mel)
    e = lin.sum(1)
    cent = (lin * np.arange(lin.shape[1], dtype=np.float32)[None]).sum(1) / np.maximum(e, 1e-9)
    stride = 4
    u = lin[::stride] / np.maximum(np.linalg.norm(lin[::stride], axis=1, keepdims=True), 1e-9)
    st = np.zeros(len(u), np.float32)
    st[:-1] = (u[:-1] * u[1:]).sum(1)
    st[-1] = st[-2] if len(st) > 1 else 0
    db = mel.mean(1)
    f200 = np.clip(np.round(edges * FPS_MEL).astype(np.int64), 0, len(mel))
    fst = np.clip(np.round(edges * FPS_MEL / stride).astype(np.int64), 0, len(st))
    cf, ce, ccent, cdb, cst = _cum(flux), _cum(e), _cum(cent * e), _cum(db), _cum(st)
    a, b = f200[:-1], f200[1:]
    bb = np.maximum(b, a + 1)
    fl = (cf[bb.clip(0, len(cf) - 1)] - cf[a]) / np.maximum(b - a, 1)
    en = ce[bb.clip(0, len(ce) - 1)] - ce[a]
    ct = (ccent[bb.clip(0, len(ccent) - 1)] - ccent[a]) / np.maximum(en, 1e-9)
    perc = fl * np.maximum(b - a, 1) / np.maximum(en, 1e-9)
    s0, s1 = fst[:-1], fst[1:]
    stat = (cst[np.maximum(s1, s0 + 1).clip(0, len(cst) - 1)] - cst[s0]) / np.maximum(s1 - s0, 1)
    loud = (cdb[bb.clip(0, len(cdb) - 1)] - cdb[a]) / np.maximum(b - a, 1)
    desc = np.stack([np.log(fl + 1e-6), ct, perc, stat, loud], 1).astype(np.float32)
    if mert is not None:
        cm = _cum(np.asarray(mert, np.float32))
        g0 = np.clip(np.round(edges[:-1] * FPS_FEAT).astype(np.int64), 0, len(mert))
        g1 = np.clip(np.round(edges[1:] * FPS_FEAT).astype(np.int64), 0, len(mert))
        emb = _pool(cm, g0, g1).astype(np.float32)
    else:
        emb = np.zeros((nb, 1024), np.float32)
    return dict(emb=emb, desc=desc, nb=nb, flux=fl.astype(np.float32))


class AudioKindPrior:
    def __init__(self, path=None):
        d = np.load(path or PRIOR_PATH)
        self.mu, self.pc, self.zs, self.cent = d["mu"], d["pc"], d["zs"], d["cent"]
        self.dmu, self.dsd, self.W = d["dmu"], d["dsd"], d["W"]

    def predict(self, ab):
        """song-demeaned predictions [nb, len(KEYS)]"""
        Z = ((ab["emb"] - self.mu) @ self.pc.T) / self.zs
        d2 = (Z ** 2).sum(1, keepdims=True) - 2 * Z @ self.cent.T + (self.cent ** 2).sum(1)[None]
        cl = d2.argmin(1)
        D = (ab["desc"] - self.dmu) / self.dsd
        Zd = Z - Z.mean(0, keepdims=True)
        Dd = D - D.mean(0, keepdims=True)
        oh = np.eye(len(self.cent))[cl]
        oh = oh - oh.mean(0, keepdims=True)
        X = np.concatenate([Zd, oh, Dd], 1)
        return np.concatenate([X, np.ones((len(X), 1))], 1) @ self.W.T

    def offsets(self, mel, mert, tempo, first, nb, gain, clip=1.5):
        """dict name -> logit offsets per bar (array [nb])"""
        P = self.predict(bar_audio(mel, mert, tempo, first, nb))
        out = {}
        for name, s in BASE_SHARE.items():
            p = np.clip(s + P[:, KEYS.index(name)], 0.005, 0.95)
            out[name] = gain * np.clip(np.log(p / (1 - p)) - math.log(s / (1 - s)), -clip, clip)
        return out
