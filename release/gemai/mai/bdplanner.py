"""Per-bar density planner (BD plan) inference: audio bar table -> one density bucket (0..11) per 4-beat bar.

This is the inference side of scratch/bdplan (build_ds.py features, train_nn.py `ar` net, post3.py post-processing).  It consumes the
bar table of the generator itself (mai.v3.compute_bar_table on the generation beat grid), so planner bars and writer bars are the same bars.

    pl = BDPlanner(root / "planner", device="cpu")
    res = pl.plan(tab, bpm=float, lv=14.2, slot=5, seed=0)      # res["bd"] = int array [nb], bucket per bar (0 = silent / inactive bar)

Pipeline (identical to the evaluated cal3_g4 setting):
  1. X = planfeat.bar_feats(tab) (+ neighbour-context features of 13 columns), cast to float16 like the training set, standardised with the
     TRAIN-set mean/std (planner/norm.npz), + level / Re:MASTER condition columns.
  2. 5 fold models of the `ar` net (BiGRU context + autoregressive GRU decoder over previous buckets), 16 samples each at T=1.0 -> 80 samples.
  3. post-processing (post3.apply): expectation E and the MBR (most central) sample, slow component of the plan (9-bar moving average) with
     contrast g, residual injection alpha*(MBR - E), tail damping of the last 16 active bars, rank-remap onto the official relative-bucket
     distribution (QREL), round and clip to 1..11 on active bars.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import planfeat as PF

NB = 12                                     # buckets 0..11 (bucket 0 = empty bar)
CTX_COLS = ("loud_z", "flux_z", "bright", "std_z", "nov", "rec_top", "mel_lo", "mel_mid", "mel_hi", "pca0", "pca1", "pca2", "pca3")


def _ctx(F, cols):
    """context features: smoothed neighbours of selected columns (verbatim from scratch/bdplan/build_ds.py)"""
    X = F[:, cols]
    nb = len(X)
    res = []
    for k in (1, 2, 4, 8):
        pad = np.pad(X, ((k, k), (0, 0)), mode="edge")
        res.append(pad[2 * k:2 * k + nb] - X)      # lead - cur
        res.append(pad[0:nb] - X)                  # lag - cur
    sm = lambda w: np.stack([np.convolve(np.pad(X[:, j], (w, w), mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "valid") for j in range(X.shape[1])], 1)
    res += [sm(4), sm(16), np.broadcast_to(X.mean(0), X.shape), np.broadcast_to(X.std(0), X.shape)]
    return np.concatenate(res, 1).astype(np.float32)


def dominant_bpm(beats, bpms, total_beats):
    """bpm that covers most of the song (time-weighted); = the only bpm of a constant-tempo song.  Stand-in for the chart header `wholebpm`
    that the training features used (only enters as the constant column log(bpm)/5)."""
    b = np.asarray(beats, float)
    p = np.asarray(bpms, float)
    if len(p) == 1:
        return float(p[0])
    ends = np.append(b[1:], max(float(total_beats), float(b[-1]) + 1e-6))
    secs = (ends - b) * 60.0 / p
    r = {}
    for bpm, s in zip(np.round(p, 2), secs):
        r[float(bpm)] = r.get(float(bpm), 0.0) + float(s)
    return max(r.items(), key=lambda kv: kv[1])[0]


def make_net(D, H=128):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, H), nn.GELU(), nn.Dropout(0.3))
            s.gru = nn.GRU(H, H, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            s.drop = nn.Dropout(0.3)
            s.emb = nn.Embedding(NB + 1, 16)
            s.dec = nn.GRU(2 * H + 16, H, batch_first=True)
            s.head = nn.Linear(H + 2 * H, NB)

        def ctx(s, x):
            return s.drop(s.gru(s.inp(x))[0])

        def sample(s, x, n, temp=1.0, g=None):
            import torch
            import torch.nn.functional as Fn
            with torch.no_grad():
                c = s.ctx(x.expand(n, -1, -1))
                T = x.shape[1]
                prev = torch.full((n, 1), NB, dtype=torch.long, device=x.device)
                h = None
                out = []
                for t in range(T):
                    o, h = s.dec(torch.cat([c[:, t:t + 1], s.emb(prev)], -1), h)
                    lg = s.head(torch.cat([o, c[:, t:t + 1]], -1))[:, 0] / temp
                    prev = torch.multinomial(Fn.softmax(lg, -1), 1, generator=g)
                    out.append(prev[:, 0])
                return torch.stack(out, 1)

    return Net()


class BDPlanner:
    def __init__(self, root, device="cpu", threads=None):
        import torch
        self.torch = torch
        self.root = str(root)
        self.cfg = json.load(open(os.path.join(self.root, "config.json"), encoding="utf-8"))
        z = np.load(os.path.join(self.root, "norm.npz"))
        self.mu, self.sd, self.qrel = z["mu"].astype(np.float32), z["sd"].astype(np.float32), z["qrel"].astype(np.float64)
        self.pmean, self.pcomp = z["pca_mean"].astype(np.float32), z["pca_comp"].astype(np.float32)
        names = PF.feature_names()
        idx = {n: i for i, n in enumerate(names)}
        self.ccols = [idx[n] for n in CTX_COLS]
        self.device = torch.device(device)
        if threads:
            torch.set_num_threads(int(threads))
        self.D = int(self.mu.shape[0]) + 3
        self.nets = []
        for f in self.cfg["folds"]:
            net = make_net(self.D, int(self.cfg["hid"]))
            net.load_state_dict(torch.load(os.path.join(self.root, self.cfg["weights"].format(f=f)), map_location="cpu"))
            self.nets.append(net.to(self.device).eval())

    # ------------------------------------------------------------------ features
    def features(self, tab, bpm):
        """bar table [nb, 1218] -> float16 [nb, D-3]; same chain as scratch/bdplan/build_ds.py"""
        tab = np.asarray(tab, np.float32)
        F = PF.clip_feats(PF.bar_feats(tab, self.pmean, self.pcomp, bpm=float(bpm) if bpm else None))
        Fc = _ctx(F, self.ccols)
        return np.concatenate([F, Fc], 1).astype(np.float16)

    def model_input(self, X, lv, slot):
        X = (X.astype(np.float32) - self.mu) / self.sd
        X = np.clip(X, -6, 6)
        n = len(X)
        c = np.zeros((n, 3), np.float32)
        c[:, 0] = (float(lv) - 13.5) / 1.5
        c[:, 1] = float(int(slot) == 6)
        c[:, 2] = c[:, 0] * c[:, 1]
        return np.concatenate([X, c], 1)

    # ------------------------------------------------------------------ sampling
    def sample(self, x, seed=0):
        torch = self.torch
        xt = torch.tensor(x[None], device=self.device)
        out = []
        for f, net in zip(self.cfg["folds"], self.nets):
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed) * 1000 + 100 + int(f))
            out.append(net.sample(xt, int(self.cfg["nsamp"]), float(self.cfg["temp"]), g).cpu().numpy())
        return np.concatenate(out, 0)                                   # [n_models * nsamp, nb]

    # ------------------------------------------------------------------ post-processing (scratch/bdplan/post3.py comp/apply, MODE default)
    def _remap(self, v, idx, mb, spread, rng):
        out = np.zeros(len(v))
        r = np.argsort(np.argsort(v[idx] + 1e-6 * rng.standard_normal(len(idx))))
        u = (r + .5) / len(idx)
        out[idx] = mb + spread * np.quantile(self.qrel, u)
        return out

    def postprocess(self, S, seed=0):
        p = self.cfg["post"]
        g, al, sp_, tail, lvl = float(p["g"]), float(p["alpha"]), float(p["spread"]), float(p["tail"]), float(p["lvl"])
        rng = np.random.default_rng(int(seed))
        S = S.astype(float)
        Em = S.mean(0)
        act = (S > 0).mean(0) > 0.5
        ia = np.flatnonzero(act)
        if len(ia) < 8:
            return None
        mb = S[:, act].mean()
        Ra = np.argsort(np.argsort(S[:, ia], 1), 1).astype(float)
        Ra -= Ra.mean(1, keepdims=True)
        Ra /= (np.linalg.norm(Ra, axis=1, keepdims=True) + 1e-9)
        C = Ra @ Ra.T
        sm = S[int(np.argmax(C.sum(1)))]                                # MBR sample = most central
        q = self._remap(Em, ia, mb, 1.0, rng)                           # qE
        sl = np.zeros(len(q))
        z = q[ia]
        sl[ia] = np.convolve(np.pad(z, (4, 4), mode="edge"), np.ones(9) / 9, "valid")      # slow (section-level) component
        q = np.where(act, mb + g * (sl - mb) + (q - sl), 0)
        v = q + al * (sm - Em)
        if tail < 1:
            last = ia[-16:]
            v[last] = mb + tail * (v[last] - mb)
        w = self._remap(v, ia, mb, sp_, rng) * lvl
        bd = np.where(act, np.clip(np.round(w), 1, 11), 0).astype(int)
        return bd

    def plan(self, tab, bpm, lv, slot=5, seed=0):
        t0 = time.perf_counter()
        tab = np.asarray(tab)
        nb = len(tab)
        if nb < 8:
            return dict(bd=None, reason=f"song too short for the planner ({nb} bars)")
        X = self.features(tab, bpm)
        t1 = time.perf_counter()
        x = self.model_input(X, lv, slot)
        S = self.sample(x, seed)
        t2 = time.perf_counter()
        bd = self.postprocess(S, seed)
        if bd is None:
            return dict(bd=None, reason="planner found < 8 active bars")
        return dict(bd=bd, n_bars=nb, lv=float(lv), slot=int(slot), bpm=float(bpm), seed=int(seed), X=X, E=S.mean(0),
                    seconds=dict(features=round(t1 - t0, 2), sampling=round(t2 - t1, 2), post=round(time.perf_counter() - t2, 2)))
