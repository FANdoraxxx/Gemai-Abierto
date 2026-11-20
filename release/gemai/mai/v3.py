"""seq_version 3 helpers (numpy only): song-level context, intensity plan, section-repeat retrieval (REF segment).

Everything here is pure numpy so it can be unit-tested without torch and reused 1:1 by the generator.

Per-chart *bar table* (tools/prep_bars.py -> <MAI_BARS_ROOT>/{sid:05d}_{slot}.npy, float16 [n_bars, 1218]), one row per 4-beat bar of
the chart's own tempo map (bar b = beats [4b, 4b+4), audio time first + tempo.to_sec(.)):
    cols [0, 1024)    mean MERT-330M L8 feature of the bar (75 fps frames inside the bar)
    cols [1024, 1120) mean of the normalized log-mel (96 bands, 200 fps; (x - chart mel_mean) / chart mel_std)
    cols [1120, 1216) temporal std of the same
    col  1216         mean positive spectral flux (sum over bands of max(0, x[t] - x[t-1]))
    col  1217         fraction of the bar's frames that lie inside the audio (0 = bar is past the end of the track)
The table depends on AUDIO + TEMPO MAP only (never on the chart's notes).

Three things are built from it:
  * song arrays    : the whole song (pooled to <= SONG_CAP tokens) -> inputs of the model's song encoder (+ aux targets in training)
  * plan           : per-window planned density bucket relative to the song mean (PLAN_k tokens)
  * retrieval (REF): the earlier block of 4 bars whose MERT bars are most similar to the current window's 4 bars; its chart notes
                     (lite form, time-shifted to the current window) are put in front of the history as a context segment.
"""
from __future__ import annotations
import math
import os
import numpy as np

from .tokens import TPB, T_MIN, T_MAX
from . import feats as FE

BAR = 4 * TPB                      # 192 ticks
WIN_BARS = 4                       # a training window's target = 4 bars (768 ticks)
SONG_CAP = 160                     # max memory tokens (longer songs: consecutive bars are mean-pooled by a factor g)
D_MERT, D_MEL = 1024, 96
C_MEL0 = D_MERT
C_STD0 = D_MERT + D_MEL
C_FLUX = D_MERT + 2 * D_MEL        # 1216
C_VF = C_FLUX + 1                  # 1217
BAR_COLS = C_VF + 1                # 1218
N_PLAN = 9                         # PLAN_0 .. PLAN_8 (+ PLAN_UNK)
N_REFS = 4                         # REF_0 .. REF_3 (+ REF_NONE)
REL_HALF = 48                      # relative-bar embedding index = bucket(delta bars) + REL_HALF  (0 .. 2 * REL_HALF)
REL_N = 2 * REL_HALF + 1
IDX_MAX = 255
AUX_COLS = 5                       # [log1p(mean notes per bar), hold share, slide share, touch share, share-valid flag]

# fallbacks only; the fitted values live in work/v3_stats.json (tools/prep_bars.py --stats) and in every v3 checkpoint's seq_meta["v3"]
PLAN_EDGES = np.array([0.35, 0.6, 0.8, 0.95, 1.1, 1.25, 1.5, 2.0])      # 8 edges -> 9 buckets of the window / song-mean density ratio
# BD (seq_version 5): bucket of the number of notes of ONE bar.  Counting rule: every note entry of the chart (tap / hold / slide / touch, each
# member of a chord counted) = what a player has to hit; same count as bar_counts()[:, 0] and window_ratio.  Lower bounds of buckets 1..11
# (bucket 0 = empty bar); fitted on TRAINING charts only (tools/fit_bd_edges.py): deciles of the non-empty bars, then widened to <= 2 notes in 4..12.
BD_EDGES = (1, 3, 4, 6, 7, 8, 9, 10, 12, 14, 17)
N_BD_B = len(BD_EDGES) + 1
REF_EDGES = (0.35, 0.5, 0.65, 0.8)                                       # centered-cosine of the best block: < e0 -> REF_NONE, then REF_0..3
REF_BUDGET = 150


def bars_root():
    return os.environ.get("MAI_BARS_ROOT") or os.path.join(FE.FEATS_ROOT, "bars_v3")


FAN_BARS_FALLBACK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "work_fan2", "bars_v3")
FAN_BARS_ROOT = os.environ.get("MAI_FAN_BARS_ROOT") or FAN_BARS_FALLBACK   # SSD cache; a missing file falls back to the G: original


def bars_path(sid, slot, root=None):
    if root is None and int(sid) >= FE.FAN_SID0:             # v5: fan songs (sid >= 100000) have their own bar-table directory
        p = os.path.join(FAN_BARS_ROOT, f"{int(sid):05d}_{int(slot)}.npy")
        return p if os.path.exists(p) else os.path.join(FAN_BARS_FALLBACK, f"{int(sid):05d}_{int(slot)}.npy")
    return os.path.join(root or bars_root(), f"{int(sid):05d}_{int(slot)}.npy")


def load_bar_table(sid, slot, root=None):
    """memory-mapped float16 [n_bars, BAR_COLS] or None when missing / malformed"""
    try:
        t = np.load(bars_path(sid, slot, root), mmap_mode="r")
    except Exception:
        return None
    return t if (t.ndim == 2 and t.shape[1] == BAR_COLS and t.shape[0] >= 1) else None


def n_bars_of(tempo, first, duration, last_tick):
    """number of bars of a chart: audio length in bars, at least the bars that hold notes"""
    sb = float(tempo.to_beat(max(float(duration) - float(first), 0.0)))
    return max(int(math.ceil(sb / 4.0)), int(last_tick) // BAR + 1, 1)


# ----------------------------------------------------------------------------------------------
# bar table
def compute_bar_table(mel, mel_mean, mel_std, mert, mert_fps, tempo, first, n_bars, mel_fps=200.0):
    """-> float16 [n_bars, BAR_COLS].  mel: float16/32 [T, 96] (200 fps) or None, mert: [T', 1024] (mmap ok) or None
    (missing parts stay 0, their validity column is 0).  Uses the chart's tempo map, so it can be called for a freshly generated
    beat grid at inference time with exactly the same code."""
    out = np.zeros((n_bars, BAR_COLS), np.float32)
    secs = float(first) + tempo.to_sec(4.0 * np.arange(n_bars + 1))
    if mel is not None and len(mel) >= 2:
        T = len(mel)
        mean = np.asarray(mel_mean, np.float32).reshape(1, -1)
        std = np.maximum(np.asarray(mel_std, np.float32), 1e-3).reshape(1, -1)
        x = (np.asarray(mel, np.float32) - mean) / std
        cs = np.zeros((T + 1, x.shape[1]), np.float64)
        np.cumsum(x, 0, dtype=np.float64, out=cs[1:])
        cs2 = np.zeros((T + 1, x.shape[1]), np.float64)
        np.cumsum(x.astype(np.float64) ** 2, 0, out=cs2[1:])
        fl = np.maximum(np.diff(x, axis=0), 0.0).sum(1)
        csf = np.zeros(T, np.float64)
        np.cumsum(fl, out=csf[1:])
        e_raw = np.round(secs * mel_fps).astype(np.int64)
        e0 = np.clip(e_raw[:-1], 0, T)
        e1 = np.clip(e_raw[1:], 0, T)
        n = e1 - e0
        ok = n > 0
        nn = np.maximum(n, 1)[:, None]
        m = (cs[e1] - cs[e0]) / nn
        v = np.maximum((cs2[e1] - cs2[e0]) / nn - m * m, 0.0)
        out[:, C_MEL0:C_MEL0 + D_MEL] = np.where(ok[:, None], m, 0.0)
        out[:, C_STD0:C_STD0 + D_MEL] = np.where(ok[:, None], np.sqrt(v), 0.0)
        nf = np.maximum(n - 1, 1)
        fsum = csf[np.clip(e1 - 1, 0, T - 1)] - csf[np.minimum(e0, T - 1)]
        out[:, C_FLUX] = np.where(n >= 2, fsum / nf, 0.0)
        span = np.maximum(e_raw[1:] - e_raw[:-1], 1)
        out[:, C_VF] = n / span
    if mert is not None and len(mert) >= 2:
        Tf = len(mert)
        f_raw = np.round(secs * float(mert_fps)).astype(np.int64)
        for b in range(n_bars):
            a, c = int(np.clip(f_raw[b], 0, Tf)), int(np.clip(f_raw[b + 1], 0, Tf))
            if c > a:
                out[b, :D_MERT] = np.asarray(mert[a:c], np.float32).mean(0)
    return out.astype(np.float16)


# ----------------------------------------------------------------------------------------------
# song arrays (inputs of the song encoder)
def pool_factor(nb, cap=SONG_CAP):
    return max(1, -(-int(nb) // int(cap)))


def pool_table(tab, g):
    """mean-pool consecutive groups of g bars: [nb, D] -> [ceil(nb/g), D] (float32)"""
    tab = np.asarray(tab, np.float32)
    if g <= 1:
        return tab
    nb = len(tab)
    S = -(-nb // g)
    out = np.empty((S, tab.shape[1]), np.float32)
    for i in range(S):
        out[i] = tab[i * g:(i + 1) * g].mean(0)
    return out


def rel_index(delta):
    """signed distance in bars (float array) -> embedding index 0 .. 2*REL_HALF: exact up to +-24 bars, then 8 steps per octave"""
    d = np.asarray(delta, np.float64)
    a = np.abs(d)
    r = np.where(a <= 24.0, np.round(a), 24.0 + np.round(8.0 * np.log2(np.maximum(a, 24.0) / 24.0)))
    r = np.minimum(r, REL_HALF)
    return (np.sign(d) * r).astype(np.int64) + REL_HALF


def bar_bpm(tempo, bars):
    """BPM at the start of each (float) bar position"""
    beats = 4.0 * np.asarray(bars, np.float64)
    i = np.clip(np.searchsorted(tempo.b, beats, side="right") - 1, 0, len(tempo.b) - 1)
    return tempo.p[i]


def song_arrays(tab, tempo, k, cap=SONG_CAP):
    """tab: bar table [nb, BAR_COLS] (any float dtype), tempo: NpTempo, k: current window start bar (can be fractional for windows that
    do not start on a bar line).  -> dict of numpy arrays for the song encoder (S = ceil(nb / g) <= cap):
       sfeat float16 [S, BAR_COLS] (mean-pooled), sscal float32 [S, 4] (log bpm/150, relative position, pool factor/4, nb/100),
       sidx int64 [S] (start bar of the token, clipped to IDX_MAX), srel int64 [S] (rel_index of the token centre - k)"""
    nb = len(tab)
    g = pool_factor(nb, cap)
    x = pool_table(tab, g)
    S = len(x)
    start = np.arange(S, dtype=np.float64) * g
    centre = start + (np.minimum(start + g, nb) - start - 1) / 2.0
    bpm = bar_bpm(tempo, centre)
    scal = np.stack([np.log(np.clip(bpm, 30.0, 600.0) / 150.0), centre / max(nb, 1), np.full(S, g / 4.0), np.full(S, nb / 100.0)],
                    1).astype(np.float32)
    return dict(sfeat=x.astype(np.float16), sscal=scal, sidx=np.minimum(start, IDX_MAX).astype(np.int64),
                srel=rel_index(centre - float(k)))


# ----------------------------------------------------------------------------------------------
# chart-derived per-bar statistics (aux targets and the ground-truth plan)
def bar_counts(V, ticks, offs, toks, nb):
    """-> float32 [nb, 4]: notes, hold-like (hold + touch hold), slide, touch-like (touch + touch hold) per bar.  Mirror invariant."""
    ticks = np.asarray(ticks)
    out = np.zeros((nb, 4), np.float32)
    if len(ticks) == 0:
        return out
    first = np.asarray(toks)[np.asarray(offs)[:-1]]
    bar = np.minimum(ticks // BAR, nb - 1).astype(np.int64)
    h, t = (first == V["N_HOLD"]) | (first == V["N_THOLD"]), (first == V["N_TOUCH"]) | (first == V["N_THOLD"])
    s = first == V["N_SLIDE"]
    out[:, 0] = np.bincount(bar, minlength=nb)[:nb]
    out[:, 1] = np.bincount(bar[h], minlength=nb)[:nb]
    out[:, 2] = np.bincount(bar[s], minlength=nb)[:nb]
    out[:, 3] = np.bincount(bar[t], minlength=nb)[:nb]
    return out


def aux_targets(cnt, nb, g):
    """per-bar counts [nb, 4] -> targets per memory token [S, AUX_COLS] float32 (see AUX_COLS)"""
    S = -(-nb // g)
    out = np.zeros((S, AUX_COLS), np.float32)
    for i in range(S):
        c = cnt[i * g:(i + 1) * g]
        n = float(c[:, 0].sum())
        out[i, 0] = math.log1p(n / len(c))
        if n > 0:
            out[i, 1:4] = c[:, 1:4].sum(0) / n
            out[i, 4] = 1.0
    return out


def window_ratio(cnt_bar):
    """notes of every 4-bar window [k, k+4) per bar, divided by the song mean notes per bar -> float array [nb] (index = k)"""
    c = np.asarray(cnt_bar, np.float64)
    nb = len(c)
    m = c.mean() if nb else 0.0
    if m <= 0:
        return np.ones(nb)
    s = np.convolve(np.concatenate([c, np.zeros(WIN_BARS - 1)]), np.ones(WIN_BARS), "valid")[:nb] / WIN_BARS
    return s / m


def plan_bucket(r, edges=PLAN_EDGES):
    """ratio -> 0 .. len(edges) (== N_PLAN - 1)"""
    return np.searchsorted(np.asarray(edges, np.float64), r, side="right")


def bd_bucket(n, edges=BD_EDGES):
    """notes of a bar (int / array) -> bucket 0 .. len(edges)"""
    return np.searchsorted(np.asarray(edges, np.float64), n, side="right")


def window_bd(cnt_bar, k, edges=BD_EDGES):
    """BD buckets of the 4 target bars k .. k+3 of a chart with per-bar note counts cnt_bar (bars past the end count 0) -> int64 [4]"""
    c = np.zeros(WIN_BARS, np.float64)
    seg = np.asarray(cnt_bar, np.float64)[max(int(k), 0):int(k) + WIN_BARS]
    c[:len(seg)] = seg
    return bd_bucket(c, edges).astype(np.int64)


def plan_from_pred(pred_log1p, g, nb, edges=PLAN_EDGES, quantiles=None, mode="rank"):
    """Inference-side plan: the song encoder's per-token prediction of log1p(mean notes per bar) -> PLAN bucket for EVERY window start
    bar k (int array [nb]).  mode "raw": ratio of the predicted window sum to the predicted song mean through the fixed edges.
    mode "rank": the predicted ratios are regression-to-the-mean compressed, so each window's rank within the song is mapped through the
    ground-truth ratio quantile function `quantiles` (101 values, seq_meta["v3"]["plan_q"]) before bucketing."""
    cnt = np.clip(np.expm1(np.asarray(pred_log1p, np.float64)), 0.0, None)
    per_bar = np.repeat(cnt, g)[:nb]
    r = window_ratio(per_bar)
    if mode == "rank" and quantiles is not None and nb > 1:
        rank = r.argsort(kind="stable").argsort(kind="stable")
        r = np.interp((rank + 0.5) / nb, np.linspace(0.0, 1.0, len(quantiles)), np.asarray(quantiles, np.float64))
    return plan_bucket(r, edges)


# ----------------------------------------------------------------------------------------------
# retrieval of an earlier similar section
def center_norm(mert):
    """[nb, 1024] -> per-song mean removed, rows L2-normalized (float32).  Cosine of these = similarity of two bars relative to the song."""
    x = np.asarray(mert, np.float32)
    x = x - x.mean(0, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-6)


def bar_embed(tab):
    """bar table [nb, BAR_COLS] -> center_norm of the MERT part, bars beyond the end of the audio (valid fraction ~0) get a zero vector"""
    t = np.asarray(tab, np.float32)
    ok = t[:, C_VF] > 0.5
    x = t[:, :D_MERT]
    x = x - (x[ok] if ok.any() else x).mean(0, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    x[~ok] = 0.0
    return x


def best_block(Bn, k, n_blk=WIN_BARS, gap=WIN_BARS):
    """Bn: center_norm of the bar MERT vectors (audio only).  For the window whose target starts at bar k, find the start bar e of the
    earlier block [e, e+n_blk) with the highest mean cosine to the bars [k, k+n_blk).  The block must end before the history range
    [k-gap, k): e + n_blk <= k - gap, so reference chart notes are strictly older than the history (and than the target).
    Returns (e, sim) or None when no candidate exists (k < gap + n_blk) or fewer than 2 bars of the window lie inside the song."""
    nb = len(Bn)
    k = int(k)
    hi = k - gap - n_blk
    nq = min(n_blk, nb - k)
    if hi < 0 or nq < 2:
        return None
    D = Bn[:hi + nq] @ Bn[k:k + nq].T                     # D[b, j] = cos(bar b, bar k + j)
    S = np.zeros(hi + 1, np.float32)
    for j in range(nq):
        S[:] += D[j:j + hi + 1, j]
    S /= nq
    e = int(np.argmax(S))
    return e, float(S[e])


def ref_bucket(sim, edges=REF_EDGES):
    """-> -1 (no usable reference) or 0 .. N_REFS-1"""
    if sim < edges[0]:
        return -1
    return int(np.searchsorted(np.asarray(edges[1:], np.float64), sim, side="right"))


class _Lite:
    def __init__(self, V):
        self.sh = frozenset(i for i, t in enumerate(V.toks) if t.startswith("SH_"))
        self.grandv = V["SH_GRANDV"]
        self.hold, self.thold = V["N_HOLD"], V["N_THOLD"]


_LITE = {}


def lite_body(V, body):
    """compact note body for the REF segment: kind + position (+ duration for holds, + first segment shape / via / end for slides)"""
    L = _LITE.get(id(V))
    if L is None:
        L = _LITE[id(V)] = _Lite(V)
    body = [int(x) for x in body]
    k0 = body[0]
    if k0 == L.hold or k0 == L.thold:
        return body[:2] + body[-2:]
    for j in range(2, len(body)):
        if body[j] in L.sh:
            return body[:2] + body[j:j + (3 if body[j] == L.grandv else 2)]
    return body[:2]


def ref_events(V, ticks, offs, toks, e, budget=REF_BUDGET, lite=True, rng=None, p_note=0.0, n_blk=WIN_BARS):
    """chart notes of the block [e, e+n_blk) as event tokens, times shifted so that the block start == the window start (t = tick - e*BAR,
    i.e. 0 .. 767, the same time tokens as the target).  Notes are taken in time order until `budget` tokens are used (a time-prefix of the
    block).  rng/p_note: training-time note dropout.  Returns (tokens int list, cur_time int list)."""
    a = int(np.searchsorted(ticks, e * BAR, "left"))
    b = int(np.searchsorted(ticks, (e + n_blk) * BAR, "left"))
    out_t, out_c = [], []
    last = None
    t0 = V.time0
    for i in range(a, b):
        if rng is not None and p_note > 0 and rng.random() < p_note:
            continue
        t = int(ticks[i]) - e * BAR
        body = toks[offs[i]:offs[i + 1]]
        body = lite_body(V, body) if lite else [int(x) for x in body]
        cost = len(body) + (1 if t != last else 0)
        if len(out_t) + cost > budget:
            break
        if t != last:
            out_t.append(t0 + t - T_MIN)
            out_c.append(t)
            last = t
        out_t.extend(body)
        out_c.extend([t] * len(body))
    return out_t, out_c


def retrieve_ref(V, Bn, ticks, offs, toks, k, edges=REF_EDGES, budget=REF_BUDGET, lite=True, rng=None, p_note=0.0):
    """full retrieval for the window starting at bar k.  ticks/offs/toks: the chart so far (ground truth in training, the already generated
    notes at inference; only notes with tick < (k - 4) * BAR are ever read).  Returns (marker_token, ev_tokens, ev_cur, info) where marker
    is REF_NONE or REF_0..3 (similarity bucket) and info = dict(e, sim)."""
    r = best_block(Bn, k)
    if r is None:
        return V["REF_NONE"], [], [], None
    e, sim = r
    bk = ref_bucket(sim, edges)
    if bk < 0:
        return V["REF_NONE"], [], [], dict(e=e, sim=sim)
    t, c = ref_events(V, ticks, offs, toks, e, budget, lite, rng, p_note)
    if not t:
        return V["REF_NONE"], [], [], dict(e=e, sim=sim)
    return V[f"REF_{bk}"], t, c, dict(e=e, sim=sim)


# ----------------------------------------------------------------------------------------------
def default_stats():
    return dict(plan_edges=PLAN_EDGES.tolist(), plan_q=None, ref_edges=list(REF_EDGES), ref_budget=REF_BUDGET, ref_lite=True,
                song_cap=SONG_CAP)


def load_v3_stats(path):
    d = default_stats()
    if path and os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            d.update(json.load(f))
    return d
