"""Chart level estimator: predicts the official level constant (e.g. 13.7) from chartstats features.

    from mai.levelest import estimate_level
    lv = estimate_level(notes, tempo)          # notes: canonical tick notes, tempo: NpTempo
Model parameters live in work/levelest.json (written by tools/fit_levelest.py); inference needs numpy only
(gradient-boosted trees stored as arrays, optional ridge blend).
"""
from __future__ import annotations
import json, math, os
import numpy as np
from . import chartstats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "work", "levelest.json")

DROP_SCALARS = {"t_first", "len_sec", "pc_viol", "pc_hands_viol", "pc_jump_viol", "pc_viol_rate", "n_fw"}
HIST_KEYS = ["h_simul_dist", "h_cons_dist", "h_slide_shape", "h_slide_speed", "h_hold_dur", "h_slide_dur", "h_rhythm"]

def feature_names(sample):
    names = sorted(k for k in chartstats.scalar_keys(sample) if k not in DROP_SCALARS)
    for h in HIST_KEYS:
        names += [f"{h}:{i}" for i in range(len(sample[h]))]
    return names

def to_vector(f, names):
    out = np.zeros(len(names), np.float64)
    hist_frac = {}
    for i, nm in enumerate(names):
        if ":" in nm:
            h, j = nm.split(":")
            if h not in hist_frac:
                a = np.asarray(f.get(h, np.zeros(1)), np.float64)
                hist_frac[h] = a / max(a.sum(), 1.0)
            j = int(j)
            out[i] = hist_frac[h][j] if j < len(hist_frac[h]) else 0.0
        else:
            out[i] = f.get(nm, 0.0)
    return out

def build_matrix(feat_dicts, names=None):
    names = names or feature_names(feat_dicts[0])
    return np.stack([to_vector(f, names) for f in feat_dicts]), names

# ------------------------------------------------------------------ display level helpers
def level_class(lv):
    """display level class as a number: x for x.0-x.5, x+0.5 for 'x+' (decimal >= .6, levels >= 7)"""
    lv = round(float(lv) + 1e-9, 1)
    base = math.floor(lv + 1e-9)
    return base + (0.5 if lv >= 7 and round(lv - base, 1) >= 0.6 else 0.0)

def display_level(lv):
    c = level_class(lv)
    return f"{int(c)}+" if c != int(c) else f"{int(c)}"

# ------------------------------------------------------------------ model
def _eval_tree(t, x):
    f, th, l, r, v = t["f"], t["t"], t["l"], t["r"], t["v"]
    n = 0
    while f[n] >= 0:
        n = l[n] if x[f[n]] <= th[n] else r[n]
    return v[n]

class LevelModel:
    def __init__(self, d):
        self.d = d
        self.names = d["names"]
        self.trees = d.get("trees", [])
        ri = d.get("ridge")
        self.ridge = None if ri is None else (np.array(ri["mean"]), np.array(ri["std"]), np.array(ri["w"]), ri["b"])
        self.w_ridge = d.get("w_ridge", 0.0)
        self.lo, self.hi = d.get("clip", [1.0, 15.0])
        self.special = d.get("special")      # specialist trees for MASTER / Re:MASTER range, used when general estimate >= switch
    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(json.load(open(path, encoding="utf-8")))
    def predict_vec(self, x, slot=None):
        """slot 5/6 (MASTER / Re:MASTER): specialist model; slot 2-4: general model; slot None: general model that
        hands over to the specialist when its estimate is >= special['switch']"""
        p = 0.0
        if self.trees:
            if self.special is not None and slot in (5, 6):
                p = sum(_eval_tree(t, x) for t in self.special["trees"])
            else:
                p = sum(_eval_tree(t, x) for t in self.trees)
                if self.special is not None and slot is None and p >= self.special["switch"]:
                    p = sum(_eval_tree(t, x) for t in self.special["trees"])
        if self.ridge is not None and self.w_ridge > 0:
            m, s, w, b = self.ridge
            pr = float(((x - m) / s) @ w + b)
            p = (1 - self.w_ridge) * p + self.w_ridge * pr if self.trees else pr
        return float(min(self.hi, max(self.lo, p)))
    def predict_feats(self, f, slot=None):
        return self.predict_vec(to_vector(f, self.names), slot)
    def predict_matrix(self, X, slots=None):
        return np.array([self.predict_vec(x, None if slots is None else slots[i]) for i, x in enumerate(X)])

_MODEL = None
def get_model(path=None):
    global _MODEL
    if path is not None:
        return LevelModel.load(path)
    if _MODEL is None:
        _MODEL = LevelModel.load()
    return _MODEL

def estimate_level(notes, tempo, model=None, feats=None, slot=None):
    """predicted level constant of a chart (clipped to [1, 15]). Pass slot=5 / 6 when the chart is a MASTER /
    Re:MASTER (uses the specialist model, MAE ~0.22 on official val charts)."""
    m = model or get_model()
    f = feats if feats is not None else chartstats.extract(notes, tempo)
    return m.predict_feats(f, slot)
