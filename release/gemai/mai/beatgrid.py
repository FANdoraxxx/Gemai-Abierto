"""Automatic beat-grid estimation for new songs (inference time).

    estimate_grid(audio_path, device="cuda:1", bpm_hint=None)
        -> dict(first=float seconds, beats=[...], bpms=[...], debug={...})

(beats, bpms) is a tempo map in the mai.simai.TempoMap convention: list of (beat, bpm) change points
(first change at beat 0, beats are quarter-note beats counted from beat 0).  `first` is the audio time (s)
of beat 0, which is always a bar line (4/4 assumed) at or after 0 s.

Method (v3):
  1. beat_this (CPJKU) framewise beat / downbeat logits (50 fps)           -> rough tempo (robust line through peaks)
  2. spectral-flux onset events (5 ms) folded modulo the beat period, BPM scanned on a fine grid around the
     tracker tempo (the fold sharpness peaks to ~0.005 BPM), then snapped to a clean value (integers are 98.7 % of
     all constant-tempo official charts);
  3. octave rule (double / half) from the tracker BPM + beat-subdivision energy of the onset fold;
  4. phase: beat-activation comb for the coarse phase, onset fold (+ calibrated bias) for the fine phase;
  5. optional tempo changes when params={"detect_changes": True}; disabled by default to reproduce sp0;
  6. bar phase (which beat is the downbeat) from the downbeat logits.

Tempo changes (detect_changes=True), v3.1 additions (scratch/gridbench/CHANGES.md):
  * stitch_chain: a tempo map need not be phase-continuous; every segment keeps its own tempo AND phase, junctions are
    clean when the phases agree (<= 15 ms) and otherwise a one-beat transition segment carries the phase jump;
  * pl_segments / pl_map: octave-normalised piecewise-linear segmentation of the tracker beats (tempo changes and phase
    breaks), per-segment onset-fold refit, octave chosen relative to the neighbour unless the octave features are
    decisive, neighbours merged unless two grids clearly beat one on the shorter segment;
  * bpm_hint with an octave mismatch: boundary of the hint-octave start section from the subdivision-density step.

Only numpy is needed for everything except `tracker_frames` / `estimate_grid` (torch + beat_this + ffmpeg).
"""
from __future__ import annotations
import math, subprocess, time
import numpy as np
from .ffmpeg_util import ffmpeg_exe

FPS = 50.0          # beat_this frame rate
SR = 22050          # beat_this sample rate

DEFAULT_PARAMS = dict(
    # --- tempo scan
    scan_rel=0.03,           # scan +-3 % around the tracker tempo
    scan_sigma_ms=5.0,       # coarse scan fold smoothing
    fine_sigma_ms=4.0,
    ev_thr=1.0,              # onset events: peaks above thr * median(env)
    snap_int_tol=0.03,       # snap to an integer BPM if closer than this
    # --- octave
    oct_always=95.0,         # tracker bpm below -> double
    oct_never=135.0,         # tracker bpm above -> keep
    oct_w=(-1.11, -0.98, 0.86, 0.76),   # logistic [bias, (b-115)/10, f_q, f_off] between oct_always and oct_never
    oct_halve=240.0,         # bpm above -> halve
    # --- phase
    onset_refine=True,
    onset_win_ms=55.0,
    onset_sigma_ms=4.0,
    onset_bias_ms=4.0,       # calibration: reported grid time minus onset-fold peak time
    bias_ms=4.0,             # same, when no onset envelope is available (tracker/comb phase)
    # --- downbeat
    down_mode="logit",
    first_prior=0.0,         # logit penalty per bar of delay of the first downbeat
    start_prior=0.0,         # reproduce the cached sp0 baseline; optional bonus for downbeats near audio start
    first_snap_ms=0.0,       # if the first downbeat is in [-snap, 0) ms clamp it to 0 instead of jumping one bar
    # --- tempo changes
    detect_changes=False,  # sp0 used a constant grid; opt in to the newer multi-tempo detector
    seg_lam=4.0e-3,          # DP penalty (log-ibi^2 units) per extra tempo segment
    seg_minlen=10,           # min intervals (beats) per segment
    seg_min_dur=3.5,         # merge very short tempo blips (seconds)
    seg_min_rel=0.015,       # ignore wobble below this relative tempo change
    seg_merge=0.04,          # adjacent segments closer than this (relative tempo) are merged
    seg_valid_ratio=1.08,    # a segment's fold sharpness at its own tempo must beat alternatives
    seg_glob_ratio=1.02,     # ... and the constant-tempo hypothesis over the same span
    seg_octave_ratio=1.12,   # keep a real 2x/0.5x switch only with this much extra sharpness
    seg_octave_keep_dur=6.0, # ... or when the tracker sustains the switch this long
    seg_scan_rel=0.02,
    seg_lo=60.0, seg_hi=300.0,    # local tempos outside are halved / doubled
    chain_tol_ms=25.0,       # max phase mismatch accepted at a tempo junction
    chain_win=8,             # junction search window (beats)
    chain_gain=1.025,        # chain must raise the onset-alignment score by this factor over the constant grid
    raw_seg_glob_ratio=1.04,
    raw_seg_octave_keep_dur=4.5,
    raw_seg_lo=105.0,
    raw_seg_hi=260.0,
    raw_chain_tol_ms=40.0,
    raw_chain_gain=1.035,
    pl_scan_rel=0.02, pl_min_dur=3.0, pl_half_tol=0.35, pl_phase_extra=0.15, pl_gain=1.04, pl_gain_k=0.5, pl_final_gain=1.02,
    pl_sig=0.010, pl_lam=25.0, pl_minlen=4,
    hint_nonoct_ignore=True, hint_osplit=True, osplit_win=8.0, osplit_hop=2.0, osplit_min=3, osplit_step=0.6, osplit_gain=0.15, osplit_tmin=8.0,
    pl=True, pl_anchor_hint=True, pl_oct="anchor", pl_oct_lo=60.0, pl_oct_hi=300.0, pl_oct_z=1.0, pl_oct_dur=8.0,
    stitch=True, stitch_clean_ms=15.0, stitch_win_s=5.0, stitch_trans_cost=0.3,
)


# ------------------------------------------------------------------------------------------ tracker
_TRACKERS = {}


def load_wave(path, sr=SR):
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _tracker(device):
    key = str(device)
    if key not in _TRACKERS:
        from beat_this.inference import Audio2Frames      # needs checkpoint beat_this-final0.ckpt in the torch hub cache
        _TRACKERS[key] = Audio2Frames(checkpoint_path="final0", device=device)
    return _TRACKERS[key]


def tracker_frames(audio_path, device="cuda:1", timing=None):
    """beat_this framewise logits (50 fps). returns (beat_logits, downbeat_logits, duration_s)"""
    t0 = time.time()
    wave = load_wave(audio_path)
    t1 = time.time()
    bl, dl = _tracker(device)(wave, SR)
    bl = bl.detach().cpu().numpy().astype(np.float32)
    dl = dl.detach().cpu().numpy().astype(np.float32)
    if timing is not None:
        timing["load"] = t1 - t0
        timing["model"] = time.time() - t1
    return bl, dl, len(wave) / SR


# ------------------------------------------------------------------------------------------ onset envelope
ONSET_HOP = 0.005
ONSET_LAG = 2
_MEL_FB = {}


def _mel_fb(sr, n_fft, n_mels, fmin=40.0, fmax=10000.0):
    key = (sr, n_fft, n_mels)
    if key not in _MEL_FB:
        h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        m2h = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        hz = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels + 2))
        bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
        fb = np.zeros((n_mels, len(bins)))
        for i in range(n_mels):
            l, c, r = hz[i], hz[i + 1], hz[i + 2]
            fb[i] = np.maximum(0.0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
        _MEL_FB[key] = fb.astype(np.float32)
    return _MEL_FB[key]


def onset_envelope(wave, sr=SR, hop_s=ONSET_HOP, n_fft=512, n_mels=48, lag=ONSET_LAG):
    """spectral-flux onset strength at 1/hop_s fps (frame j centred on j*hop_s, flux over `lag` frames)"""
    wave = np.asarray(wave, np.float32)
    n = len(wave)
    nfr = int(n / sr / hop_s) + 1
    centers = np.rint(np.arange(nfr) * hop_s * sr).astype(np.int64)
    x = np.pad(wave, (n_fft // 2, n_fft))
    win = np.hanning(n_fft).astype(np.float32)
    fb = _mel_fb(sr, n_fft, n_mels)
    ar = np.arange(n_fft)
    L = np.empty((nfr, n_mels), np.float32)
    for s in range(0, nfr, 2048):
        c = centers[s:s + 2048]
        fr = x[c[:, None] + ar[None, :]] * win
        sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        L[s:s + 2048] = 10.0 * np.log10(np.maximum(sp @ fb.T, 1e-10))
    L = np.maximum(L, L.max() - 80.0)
    d = np.maximum(L[lag:] - L[:-lag], 0.0).mean(axis=1)
    return np.concatenate([np.zeros(lag, np.float32), d.astype(np.float32)])


def _norm_env(env):
    e = np.asarray(env, np.float64)
    e = e / (np.median(e[e > 0]) + 1e-9) if (e > 0).any() else e
    return np.minimum(e, 20.0)


def onset_events(env, thr=1.0, hop_s=ONSET_HOP, lag=ONSET_LAG):
    """peak-picked onset events (sub-frame time, strength) of the normalised envelope"""
    e = _norm_env(env)
    t = (np.arange(len(e)) - lag / 2.0) * hop_s
    pk = np.nonzero((e[1:-1] >= e[:-2]) & (e[1:-1] > e[2:]) & (e[1:-1] > thr))[0] + 1
    y0, y1, y2 = e[pk - 1], e[pk], e[pk + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den < -1e-9, 0.5 * (y0 - y2) / np.where(den < -1e-9, den, -1.0), 0.0)
    return t[pk] + d * hop_s, e[pk]


# ------------------------------------------------------------------------------------------ folding
def _smooth_circ(h, sigma_bins):
    r = int(3 * sigma_bins) + 1
    xs = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (xs / sigma_bins) ** 2); g /= g.sum()
    hp = np.concatenate([h[-r:], h, h[:r]]) if r <= len(h) else np.tile(h, 2 * r // len(h) + 3)[:len(h) + 2 * r]
    return np.convolve(hp, g, mode="valid")


def fold_hist(t, w, P, nb):
    bins = ((t % P) / P * nb).astype(np.int64) % nb
    return np.bincount(bins, weights=w, minlength=nb)


def fold_score(t, w, b, sigma_ms):
    """sharpness of the smoothed fold of weighted events t at bpm b (higher = more periodic)"""
    P = 60.0 / b
    nb = max(16, int(round(P * 1000)))
    hs = _smooth_circ(fold_hist(t, w, P, nb), sigma_ms / (P * 1000.0 / nb))
    return float((hs * hs).sum() / (hs.sum() ** 2 + 1e-12) * nb)


def scan_bpm(t, w, centers, rel, sigma_ms, fine_sigma_ms, ntop=3):
    """best bpm (and score) of a fine scan around each centre.  returns (b, score, [(b, score), ...])"""
    allpk = []
    for c in centers:
        step = c * 1.5e-4
        bs = np.arange(c * (1 - rel), c * (1 + rel) + 1e-9, step)
        sc = np.array([fold_score(t, w, b, sigma_ms) for b in bs])
        loc = [i for i in range(1, len(bs) - 1) if sc[i] >= sc[i - 1] and sc[i] >= sc[i + 1]]
        loc = sorted(loc, key=lambda i: -sc[i])[:ntop]
        for i in loc:
            fb = np.arange(bs[i] - 1.5 * step, bs[i] + 1.5 * step + 1e-9, step / 6.0)
            fs = np.array([fold_score(t, w, b, fine_sigma_ms) for b in fb])
            j = int(np.argmax(fs))
            b_pk = float(fb[j])
            if 0 < j < len(fb) - 1:                                  # parabolic
                den = fs[j - 1] - 2 * fs[j] + fs[j + 1]
                if den < -1e-12:
                    b_pk += float(0.5 * (fs[j - 1] - fs[j + 1]) / den * (fb[1] - fb[0]))
            allpk.append((b_pk, float(fs[j])))
    if not allpk:
        return None
    allpk.sort(key=lambda x: -x[1])
    return allpk[0][0], allpk[0][1], allpk


def snap_bpm(b, prm):
    r = round(b)
    if abs(b - r) <= prm["snap_int_tol"]:
        return float(r)
    return float(round(b, 2))


# ------------------------------------------------------------------------------------------ onset fold (phase)
class _OnsetFold:
    """1 ms resolution fold of the onset envelope -> phase of the strongest folded onset energy"""

    def __init__(self, env, hop_s=ONSET_HOP, lag=ONSET_LAG):
        t_env = (np.arange(len(env)) - lag / 2.0) * hop_s
        self.t1k = np.arange(0.0, t_env[-1], 0.001)
        self.e1k = np.interp(self.t1k, t_env, _norm_env(env))

    def curve(self, P, sigma_ms=4.0, t0=0.0, t1=None):
        nb = max(8, int(round(P * 1000)))
        i0 = int(t0 * 1000); i1 = len(self.t1k) if t1 is None else min(len(self.t1k), int(t1 * 1000))
        tt = self.t1k[i0:i1]
        h = fold_hist(tt, self.e1k[i0:i1], P, nb)
        return _smooth_circ(h, sigma_ms / (P * 1000.0 / nb)), nb

    def refine(self, P, phi0, win_ms=35.0, sigma_ms=4.0, t0=0.0, t1=None):
        """phase (in [0,P)) of the strongest folded onset energy within +-win_ms of phi0 -> (phi, prominence, at_edge)"""
        hs, nb = self.curve(P, sigma_ms, t0, t1)
        bw = P * 1000.0 / nb
        c = int(round(phi0 / P * nb)) % nb
        w = int(round(win_ms / bw))
        idx = (np.arange(c - w, c + w + 1)) % nb
        seg = hs[idx]
        j = int(np.argmax(seg))
        jj = int(idx[j])
        y0, y1, y2 = hs[(jj - 1) % nb], hs[jj], hs[(jj + 1) % nb]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den < -1e-12 else 0.0
        phi = ((jj + d) / nb * P) % P
        prom = float((hs[jj] - np.median(hs)) / (hs.max() - np.median(hs) + 1e-12))
        return phi, prom, (j == 0 or j == len(seg) - 1)

    def window_max(self, hs, nb, P, centre_phase, win_ms=35.0):
        bw = P * 1000.0 / nb
        c = int(round(centre_phase / P * nb))
        w = int(round(win_ms / bw))
        return float(hs[(np.arange(c - w, c + w + 1)) % nb].max())


# ------------------------------------------------------------------------------------------ helpers
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def pick_peaks(logit, radius=3, thr=0.0, refine="parab"):
    """local maxima of a logit curve -> (times_s, logits)"""
    x = np.asarray(logit, np.float64)
    n = len(x)
    if n < 2 * radius + 2:
        return np.zeros(0), np.zeros(0)
    pad = np.pad(x, radius, constant_values=-1e9)
    mx = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1).max(axis=1)
    idx = np.nonzero((x >= mx) & (x > thr))[0]
    if len(idx) > 1:                                  # plateaus
        keep = [idx[0]]
        for i in idx[1:]:
            if i - keep[-1] <= 1 and x[i] == x[keep[-1]]:
                continue
            keep.append(i)
        idx = np.array(keep)
    pos = idx.astype(np.float64)
    if refine == "parab":
        ok = (idx > 0) & (idx < n - 1)
        i = idx[ok]
        y0, y1, y2 = x[i - 1], x[i], x[i + 1]
        den = y0 - 2 * y1 + y2
        d = np.zeros(len(i))
        nz = den < -1e-9
        d[nz] = np.clip(0.5 * (y0[nz] - y2[nz]) / den[nz], -0.5, 0.5)
        pos[ok] += d
    return pos / FPS, x[idx]


def _initial_period(t):
    d = np.diff(t)
    d = d[(d > 0.18) & (d < 2.0)]
    if len(d) < 4:
        return None
    lg = np.log(d)
    bins = np.arange(math.log(0.18), math.log(2.0) + 0.004, 0.004)
    h, _ = np.histogram(lg, bins)
    hs = np.convolve(h, np.ones(5), "same")
    j = int(np.argmax(hs))
    c = math.exp(0.5 * (bins[j] + bins[j + 1]))
    sel = d[(d > c * 0.97) & (d < c * 1.03)]
    return float(np.median(sel)) if len(sel) else c


def _clean_and_index(t, s, P0):
    """drop spurious extra peaks, assign integer beat indices. returns (t, s, k)"""
    t = np.asarray(t, np.float64); s = np.asarray(s, np.float64)
    for _ in range(200):
        if len(t) < 3:
            break
        d = np.diff(t)
        bad = np.nonzero(d < 0.55 * P0)[0]
        if len(bad) == 0:
            break
        i = bad[0]
        drop = i if s[i] < s[i + 1] else i + 1
        t = np.delete(t, drop); s = np.delete(s, drop)
    n = len(t)
    d = np.diff(t)
    ok = (d > 0.7 * P0) & (d < 1.35 * P0)
    Pl = np.full(n - 1, P0)
    for i in range(n - 1):
        lo, hi = max(0, i - 5), min(n - 1, i + 6)
        v = d[lo:hi][ok[lo:hi]]
        if len(v) >= 2:
            Pl[i] = np.median(v)
    steps = np.maximum(1, np.round(d / Pl)).astype(int)
    k = np.concatenate([[0], np.cumsum(steps)])
    return t, s, k


def _tukey_irls(X, y, scale_floor=0.008, c=3.0, iters=20, beta0=None):
    """robust LS (Tukey biweight). X [n,m], y [n] in seconds"""
    n, m = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0] if beta0 is None else np.asarray(beta0, np.float64)
    w = np.ones(n)
    for it in range(iters):
        r = y - X @ beta
        sc = max(scale_floor, 1.4826 * np.median(np.abs(r)))
        u = r / (c * sc)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if (w > 0).sum() < m + 2:
            w = np.ones(n)
        sw = np.sqrt(w)
        nb = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        done = np.max(np.abs(nb - beta)) < 1e-10
        beta = nb
        if done:
            break
    r = y - X @ beta
    return beta, r, w


def tracker_tempo(bl):
    """rough tempo from the tracker: dict(b_reg, b_hist, tt, ss) or None"""
    tt, ss = pick_peaks(bl)
    if len(tt) < 8:
        return None
    P0 = _initial_period(tt)
    if P0 is None:
        return None
    t, s, k = _clean_and_index(tt, ss, P0)
    k = k.astype(np.float64)
    X = np.stack([np.ones_like(k), k], axis=1)
    a0 = np.median(t - P0 * k)
    beta, r, w = _tukey_irls(X, t, beta0=np.array([a0, P0]))
    b_reg = 60.0 / beta[1] if beta[1] > 0.1 else 60.0 / P0
    return dict(b_reg=float(b_reg), b_hist=float(60.0 / P0), tt=tt, ss=ss, t=t, k=k, a=float(beta[0]), P=float(beta[1]),
                inlier=float((w > 0).mean()))


# ------------------------------------------------------------------------------------------ comb (beat activation)
class _Comb:
    """fold the beat activation modulo a period (1 ms resolution) -> coarse phase"""

    def __init__(self, bl):
        a = _sig(np.asarray(bl, np.float64))
        self.t1k = np.arange(0, (len(a) - 1) / FPS, 0.001)
        self.a1k = np.interp(self.t1k, np.arange(len(a)) / FPS, a)

    def fold(self, P, t0=0.0, t1=None, sigma_ms=5.0):
        nb = max(8, int(round(P * 1000)))
        i0 = int(t0 * 1000); i1 = len(self.t1k) if t1 is None else min(len(self.t1k), int(t1 * 1000))
        h = fold_hist(self.t1k[i0:i1], self.a1k[i0:i1], P, nb)
        return _smooth_circ(h, sigma_ms / (P * 1000.0 / nb)), nb

    @staticmethod
    def peak(hs, nb, P, j=None):
        if j is None:
            j = int(np.argmax(hs))
        y0, y1, y2 = hs[(j - 1) % nb], hs[j], hs[(j + 1) % nb]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den < -1e-12 else 0.0
        return ((j + d) / nb * P) % P, float(hs[j])

    def best_phase_for(self, P, t0=0.0, t1=None):
        hs, nb = self.fold(P, t0, t1)
        return self.peak(hs, nb, P)


# ------------------------------------------------------------------------------------------ tempo / downbeat
def _tempo_sec(beats, bpms, g):
    """seconds (relative to beat 0) of beat g under a tempo map; extends backwards with the first bpm"""
    b = np.asarray(beats, np.float64); p = np.asarray(bpms, np.float64)
    secs = np.concatenate([[0.0], np.cumsum(np.diff(b) * 60.0 / p[:-1])])
    g = np.asarray(g, np.float64)
    i = np.clip(np.searchsorted(b, g, side="right") - 1, 0, len(b) - 1)
    return secs[i] + (g - b[i]) * 60.0 / p[i]


def _down_curve(dl, mode):
    x = np.asarray(dl, np.float64)
    x = _sig(x) if mode == "prob" else np.clip(x, -10, 10)
    pad = np.pad(x, 2, constant_values=-1e9)
    return np.lib.stride_tricks.sliding_window_view(pad, 5).max(axis=1)


def _sample(curve, tt):
    i = np.rint(np.asarray(tt) * FPS).astype(int)
    ok = (i >= 0) & (i < len(curve))
    out = np.zeros(len(i))
    out[ok] = curve[i[ok]]
    return out, ok


def _down_phase(dcurve, tfun, dur, shifts, gs_range, first_prior=0.0, start_prior=0.0, start_win=0.05):
    """best (score, bar phase p, shift) : downbeats are grid beats g with g % 4 == p.
    score = mean downbeat logit at the hypothesised downbeats + start_prior if a downbeat lies within start_win of 0 s."""
    best = None
    for s in shifts:
        gs = np.arange(*gs_range)
        taus = tfun(gs, s)
        ok = (taus >= 0) & (taus <= dur)
        if ok.sum() < 4:
            continue
        v, _ = _sample(dcurve, taus[ok])
        gg = gs[ok]; tt = taus[ok]
        bar = 4.0 * (tt[-1] - tt[0]) / max(gg[-1] - gg[0], 1)
        for p in range(4):
            sel = (gg % 4) == p
            if sel.sum() == 0:
                continue
            sc = float(v[sel].mean())
            if first_prior:
                sc -= first_prior * float(tt[sel][0]) / bar          # earlier first downbeat preferred
            if start_prior:
                allp = taus[(gs % 4) == p]
                if len(allp) and np.min(np.abs(allp)) <= start_win:
                    sc += start_prior
            if best is None or sc > best[0]:
                best = (sc, p, s)
    return best


# ------------------------------------------------------------------------------------------ octave
def _octave_features(ev, b1, phi_c, t0=0.0, t1=None):
    """beat-subdivision energy of the onset fold at tempo b1 (relative to the beat peak)"""
    if ev.fold is None:
        return 0.0, 0.0
    P = 60.0 / b1
    hs, nb = ev.fold.curve(P, 4.0, t0, t1)
    bw = P * 1000.0 / nb
    c = int(round(phi_c / P * nb)) % nb
    w = int(round(35.0 / bw))
    idx = np.arange(c - w, c + w + 1) % nb
    jm = int(idx[np.argmax(hs[idx])])
    med = float(np.median(hs))
    h0 = float(hs[jm]); den = h0 - med + 1e-9
    win = lambda cc: float(hs[(np.arange(int(round(cc)) - w, int(round(cc)) + w + 1)) % nb].max())
    f_off = np.clip((win(jm + nb / 2.0) - med) / den, 0.0, 1.5)
    f_q = np.clip(((win(jm + nb / 4.0) + win(jm + 3 * nb / 4.0)) / 2.0 - med) / den, 0.0, 1.5)
    return float(f_q), float(f_off)


def octave_factor(b1, f_q, f_off, prm):
    """multiplier (0.5 / 1 / 2) to turn the tracker-octave bpm b1 into the maimai chart bpm"""
    if b1 > prm["oct_halve"]:
        return 0.5
    if b1 < prm["oct_always"]:
        return 2.0
    if b1 >= prm["oct_never"]:
        return 1.0
    w = prm["oct_w"]
    z = w[0] + w[1] * (b1 - 115.0) / 10.0 + w[2] * f_q + w[3] * f_off
    return 2.0 if z > 0 else 1.0


class _Evidence:
    def __init__(self, bl, onset, prm):
        self.bl = bl
        self.comb = _Comb(bl)
        self.onset = onset
        if onset is not None:
            self.te, self.we = onset_events(onset, prm["ev_thr"])
            self.fold = _OnsetFold(onset)
            sig = 5
            xs = np.arange(-3 * sig, 3 * sig + 1)
            g = np.exp(-0.5 * (xs / sig) ** 2); g /= g.sum()
            self.e_s = np.convolve(self.fold.e1k, g, mode="same")
        else:
            self.te = self.we = self.fold = self.e_s = None

    def align(self, times):
        """mean smoothed onset strength at the given beat times (higher = grid sits on the audio events)"""
        if self.e_s is None:
            return 0.0
        i = np.rint(np.asarray(times) * 1000).astype(int)
        i = i[(i >= 0) & (i < len(self.e_s))]
        return float(self.e_s[i].mean()) if len(i) else 0.0

    def events_in(self, t0, t1):
        m = (self.te >= t0) & (self.te <= t1)
        return self.te[m], self.we[m]

    def scan_events(self, trk):
        if self.te is not None:
            return self.te, self.we
        return trk["tt"], _sig(trk["ss"])


# ------------------------------------------------------------------------------------------ phase
def _fine_phase(ev, P, prm, dbg, t0=0.0, t1=None):
    """phase (s, in [0,P)) of the beat grid with period P: activation comb -> onset fold refinement -> bias"""
    phi_c, _ = ev.comb.best_phase_for(P, t0, t1)
    if ev.fold is None or not prm["onset_refine"]:
        return (phi_c + prm["bias_ms"] / 1000.0) % P
    phi, prom, edge = ev.fold.refine(P, phi_c, prm["onset_win_ms"], prm["onset_sigma_ms"], t0, t1)
    dbg["onset_prom"] = round(prom, 3); dbg["onset_edge"] = bool(edge)
    dbg["onset_shift_ms"] = round((((phi - phi_c) + P / 2) % P - P / 2) * 1000, 2)
    if edge:
        return (phi_c + prm["bias_ms"] / 1000.0) % P
    return (phi + prm["onset_bias_ms"] / 1000.0) % P



# ------------------------------------------------------------------------------------------ tempo changes
def _pc_dp(x, lam, minlen):
    """optimal piecewise-constant segmentation of x (SSE + lam per extra segment); returns segment start indices (excl. 0)"""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)]); cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    f = np.full(n + 1, np.inf); f[0] = -lam
    prev = np.zeros(n + 1, int)
    for j in range(minlen, n + 1):
        lo = np.arange(0, j - minlen + 1)
        m = (j - lo).astype(np.float64)
        sm = cs[j] - cs[lo]
        sse = (cs2[j] - cs2[lo]) - sm * sm / m
        cand = f[lo] + sse + lam
        cand = np.where(np.isfinite(f[lo]), cand, np.inf)
        b = int(np.argmin(cand))
        f[j] = cand[b]; prev[j] = lo[b]
    cuts = []
    j = n
    while j > 0:
        cuts.append(int(prev[j])); j = int(prev[j])
    return sorted(set(cuts) - {0})


def _medfilt(x, k=5):
    r = k // 2
    p = np.pad(x, r, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)


def tempo_segments(trk, prm, use_raw=False):
    """segments of (roughly) constant tracker tempo from the inter-beat interval series.
    returns list of dict(a, b, t0, t1, per, raw_bpm, n) or None"""
    if use_raw and "tt" in trk:
        # Segment raw tracker peaks first.  Applying the global period's peak cleaning before this
        # point drops genuine beats around large tempo changes (notably at 2x and half time).
        t = np.asarray(trk["tt"], np.float64)
        s = np.asarray(trk["ss"], np.float64)
        if len(t) < 2 * prm["seg_minlen"] + 4:
            return None
        ibi = np.diff(t)
        good = (ibi > 0.15) & (ibi < 1.5)
    else:
        # Backward-compatible path for synthetic callers that only provide indexed tracker beats.
        t = trk["t"]; k = trk["k"]
        if len(t) < 2 * prm["seg_minlen"] + 4:
            return None
        dk = np.diff(k); dt = np.diff(t)
        ibi = dt / np.maximum(dk, 1)
        P0 = trk["P"]
        good = (dk >= 1) & (dk <= 2) & (ibi > 0.45 * P0) & (ibi < 2.2 * P0)
    idx = np.nonzero(good)[0]
    if len(idx) < 12:
        return None
    x = np.interp(np.arange(len(ibi)), idx, np.log(ibi[idx]))
    xm = _medfilt(x, 5)
    cuts = _pc_dp(xm, prm["seg_lam"], prm["seg_minlen"])
    edges = [0] + cuts + [len(ibi)]
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        v = np.log(ibi[a:b][good[a:b]]) if good[a:b].any() else x[a:b]
        segs.append(dict(a=a, b=b, mu=float(np.median(v))))

    def _join(i, j):
        a, b = min(segs[i]["a"], segs[j]["a"]), max(segs[i]["b"], segs[j]["b"])
        v = np.log(ibi[a:b][good[a:b]]) if good[a:b].any() else x[a:b]
        segs[min(i, j):max(i, j) + 1] = [dict(a=a, b=b, mu=float(np.median(v)))]

    # Very short blips are tracker wobble around a boundary, not useful tempo changes.
    while len(segs) > 1:
        short = [i for i, sg in enumerate(segs)
                 if t[sg["b"]] - t[sg["a"]] < prm["seg_min_dur"]]
        if not short:
            break
        i = min(short, key=lambda j: t[segs[j]["b"]] - t[segs[j]["a"]])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = max(nb, key=lambda k: segs[k]["b"] - segs[k]["a"])
        _join(i, j)
    # merge close neighbours
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i in range(len(segs) - 1):
            if abs(math.exp(segs[i]["mu"] - segs[i + 1]["mu"]) - 1) < prm["seg_merge"]:
                _join(i, i + 1)
                changed = True
                break
    out = []
    for sg in segs:
        a, b = sg["a"], sg["b"]                 # intervals a..b-1 -> beats a..b
        P_seg = math.exp(sg["mu"])
        if use_raw and "tt" in trk:
            tt_seg = t[a:b + 1]
            ss_seg = s[a:b + 1]
            t_c, s_c, k_c = _clean_and_index(tt_seg, ss_seg, P_seg)
            k_c = k_c.astype(np.float64)
            if len(k_c) >= 4:
                X = np.stack([np.ones_like(k_c), k_c - k_c[0]], axis=1)
                beta, r, w = _tukey_irls(X, t_c, beta0=np.array([t_c[0], P_seg]))
                per = float(beta[1]) if beta[1] > 0 else P_seg
            else:
                per = P_seg
            out.append(dict(a=a, b=b, t0=float(t_c[0]), t1=float(t_c[-1]), per=per,
                            raw_bpm=float(60.0 / P_seg), n=int(k_c[-1] - k_c[0])))
        else:
            kk = k[a:b + 1].astype(np.float64); tt = t[a:b + 1]
            if len(kk) >= 4:
                X = np.stack([np.ones_like(kk), kk - kk[0]], axis=1)
                beta, r, w = _tukey_irls(X, tt, beta0=np.array([tt[0], P_seg]))
                per = float(beta[1]) if beta[1] > 0 else P_seg
            else:
                per = P_seg
            out.append(dict(a=a, b=b, t0=float(tt[0]), t1=float(tt[-1]), per=per,
                            raw_bpm=float(60.0 / P_seg), n=int(kk[-1] - kk[0])))
    return out


# ------------------------------------------------------------------------------------------ piecewise-linear beat segmentation
def _local_period(d, k=9):
    r = k // 2
    p = np.pad(d, r, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)


def index_local(tt, ss):
    """integer beat index of raw tracker peaks with a running-median local period (robust to tempo changes)"""
    t = np.asarray(tt, np.float64); s = np.asarray(ss, np.float64)
    for _ in range(400):
        if len(t) < 3:
            break
        d = np.diff(t)
        bad = np.nonzero(d < 0.55 * _local_period(d))[0]
        if len(bad) == 0:
            break
        i = bad[0]
        drop = i if s[i] < s[i + 1] else i + 1
        t = np.delete(t, drop); s = np.delete(s, drop)
    d = np.diff(t)
    steps = np.maximum(1, np.round(d / _local_period(d))).astype(int)
    return t, s, np.concatenate([[0], np.cumsum(steps)])


def _pl_dp(t, k, w, sig, lam, minlen):
    """optimal piecewise-linear (t = a + P k) segmentation, cost = weighted SSE / sig^2 + lam per extra segment.
    returns list of (start, end) index pairs"""
    n = len(t)
    tc = t - t.mean(); kc = (k - k.mean()).astype(np.float64)
    cs = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    cw, ck, ct, ckk, ckt, ctt = cs(w), cs(w * kc), cs(w * tc), cs(w * kc * kc), cs(w * kc * tc), cs(w * tc * tc)
    f = np.full(n + 1, np.inf); f[0] = -lam
    prev = np.zeros(n + 1, int)
    inv = 1.0 / (sig * sig)
    for j in range(minlen, n + 1):
        lo = np.arange(0, j - minlen + 1)
        m = np.maximum(cw[j] - cw[lo], 1e-9)
        sk = ck[j] - ck[lo]; st = ct[j] - ct[lo]
        vk = (ckk[j] - ckk[lo]) - sk * sk / m
        cv = (ckt[j] - ckt[lo]) - sk * st / m
        sse = (ctt[j] - ctt[lo]) - st * st / m - np.where(vk > 1e-9, cv * cv / np.maximum(vk, 1e-9), 0.0)
        cand = f[lo] + np.maximum(sse, 0.0) * inv + lam
        b = int(np.argmin(cand))
        f[j] = cand[b]; prev[j] = lo[b]
    out = []
    j = n
    while j > 0:
        out.append((int(prev[j]), j)); j = int(prev[j])
    return out[::-1]


def pl_segments(trk, prm):
    """tempo/phase segments from piecewise-linear fits of the raw tracker beats, octave-normalised: the tracker's own
    beat level may flip between 1x / 2x / 0.5x of the global pulse; such flips are not tempo changes.
    returns list of dict(t0, t1, per, raw_bpm, n, a) (per = octave-normalised pulse period at the global pulse level)"""
    if "tt" not in trk or len(trk["tt"]) < 2 * prm["pl_minlen"] + 2:
        return None
    t = np.asarray(trk["tt"], np.float64)
    Pg = 60.0 / trk["b_reg"]
    d = np.diff(t)
    r = d / Pg
    cand = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    o = np.argmin(np.abs(np.log(r[:, None] / cand[None, :])), axis=1)
    u = cand[o]
    k = np.concatenate([[0.0], np.cumsum(u)])
    if len(t) < 2 * prm["pl_minlen"] + 2:
        return None
    w = np.ones(len(t))
    segs = None
    for it in range(3):
        segs = _pl_dp(t, k, w, prm["pl_sig"], prm["pl_lam"], prm["pl_minlen"])
        res = np.zeros(len(t))
        for a, b in segs:
            X = np.stack([np.ones(b - a), k[a:b] - k[a]], axis=1)
            beta = np.linalg.lstsq(X * np.sqrt(w[a:b, None]), t[a:b] * np.sqrt(w[a:b]), rcond=None)[0]
            res[a:b] = t[a:b] - X @ beta
        sc = max(0.006, 1.4826 * np.median(np.abs(res)))
        w = np.where(np.abs(res) < 3 * sc, 1.0, 0.05)
    out = []
    for a, b in segs:
        X = np.stack([np.ones(b - a), (k[a:b] - k[a]).astype(np.float64)], axis=1)
        sw = np.sqrt(w[a:b])
        beta = np.linalg.lstsq(X * sw[:, None], t[a:b] * sw, rcond=None)[0]
        per = float(beta[1])
        if per <= 0.05:
            per = Pg
        out.append(dict(a=a, b=b, t0=float(t[a]), t1=float(t[b - 1]), per=per, raw_bpm=60.0 / per,
                        n=int(round(k[b - 1] - k[a])), u=float(np.median(u[a:b - 1])) if b - a > 1 else 1.0))
    return out


def _seg_phase(ev, P, prm, t0, t1):
    dbg = {}
    return _fine_phase(ev, P, prm, dbg, max(0.0, t0 - P), t1 + P)


def refine_segments(ev, segs, f, b_glob, prm, dbg=None):
    """local onset-fold refinement of every segment's tempo, validation against the other tempos, merging of invalid ones.
    returns list of dict(t0, t1, b, phi, P, n) (final-octave bpm) ; len==1 -> constant tempo"""
    res = []
    for sg in segs:
        span = max(sg["t1"] - sg["t0"], 1.0)
        te, we = ev.events_in(sg["t0"] - 0.3, sg["t1"] + 0.3)
        b_guess = 60.0 / sg["per"] * f
        if len(te) >= 12:
            r = scan_bpm(te, we, [b_guess], prm["seg_scan_rel"], 6.0, 4.0, ntop=2)
            b = r[0] if r else b_guess
        else:
            b = b_guess
        tol = min(0.25, max(0.03, 0.03 * 140.0 / span))
        r_ = round(b)
        b = float(r_) if abs(b - r_) <= tol else float(round(b, 2))
        # octave handling: chart tempos live in ~[100, 250]; half/double of the global tempo is the same pulse
        for _ in range(2):
            if b > prm["seg_hi"]:
                b /= 2.0
            elif b < prm["seg_lo"]:
                b *= 2.0
        res.append(dict(t0=sg["t0"], t1=sg["t1"], b=float(b), n=sg["n"]))
    # merge equal-tempo neighbours
    def merge_equal():
        k = 0
        while k < len(res) - 1:
            if abs(res[k]["b"] / res[k + 1]["b"] - 1) < 0.004:
                keep = res[k] if res[k]["n"] >= res[k + 1]["n"] else res[k + 1]
                res[k:k + 2] = [dict(t0=res[k]["t0"], t1=res[k + 1]["t1"],
                                     b=keep["b"], n=res[k]["n"] + res[k + 1]["n"])]
            else:
                k += 1

    merge_equal()

    # Validation is relative to the constant-tempo hypothesis over the same span.  The longest
    # segment is the anchor: never drop the dominant tempo merely because a short alternation
    # folds slightly better locally.
    def sharp(sg, b):
        te, we = ev.events_in(sg["t0"] - 0.3, sg["t1"] + 0.3)
        if len(te) < 12:
            return 0.0
        return fold_score(te, we, b, 4.0)

    def octave_rel(a, b):
        r = a / b
        return abs(r - 2) < 0.06 or abs(r - 0.5) < 0.03 or abs(r - 1) < 0.02

    scores = {}

    def score(i, b):
        key = (i, round(float(b), 5))
        if key not in scores:
            scores[key] = sharp(res[i], b)
        return scores[key]

    # Resolve same-pulse octave ambiguity with local evidence instead of a hard 2x/0.5x ban.
    for i, sg in enumerate(res):
        own = score(i, sg["b"])
        const = score(i, b_glob)
        if dbg is not None:
            dbg.setdefault("refine_scores", []).append(
                (round(sg["t0"], 1), round(sg["t1"], 1), round(sg["b"], 2),
                 round(own, 3), round(const, 3)))
        rel = sg["b"] / b_glob
        if abs(rel - 1) < prm["seg_min_rel"]:
            sg["b"] = float(b_glob)
            sg["ratio"] = own / max(const, 1e-9)
        elif octave_rel(sg["b"], b_glob) and own < prm["seg_octave_ratio"] * const:
            if sg["t1"] - sg["t0"] < prm["seg_octave_keep_dur"]:
                sg["b"] = float(b_glob)
            sg["ratio"] = own / max(const, 1e-9)

    for _ in range(len(res)):
        if len(res) <= 1:
            break
        anchor = max(range(len(res)), key=lambda i: res[i]["n"])
        worst = None
        for i, sg in enumerate(res):
            if i == anchor:
                sg["ratio"] = score(i, sg["b"]) / max(score(i, b_glob), 1e-9)
                continue
            if abs(sg["b"] / b_glob - 1) < prm["seg_min_rel"]:
                sg["b"] = float(b_glob)
                sg["ratio"] = score(i, sg["b"]) / max(score(i, b_glob), 1e-9)
                continue
            own = score(i, sg["b"])
            const = score(i, b_glob)
            ratio = own / max(const, 1e-9)
            sg["ratio"] = ratio
            octave_sustained = (octave_rel(sg["b"], b_glob)
                                and sg["t1"] - sg["t0"] >= prm["seg_octave_keep_dur"])
            bad = (own < prm["seg_glob_ratio"] * max(const, 1e-9)
                   and not octave_sustained)
            if bad and (worst is None or ratio < worst[0]):
                worst = (ratio, i)
        if worst is None:
            break
        i = worst[1]
        if dbg is not None:
            dbg.setdefault("refine_merge", []).append(
                (round(res[i]["t0"], 1), round(res[i]["t1"], 1), round(res[i]["b"], 2),
                 round(worst[0], 3)))
        # Merge a weak/short transition into the longer neighbour rather than reverting the
        # entire song to one BPM (a single bad junction must stay local).
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(res)]
        keep_j = max(nb, key=lambda j: res[j]["n"])
        a, c = min(i, keep_j), max(i, keep_j)
        keep = res[keep_j]
        res[a:c + 1] = [dict(
            t0=min(res[a]["t0"], res[c]["t0"]),
            t1=max(res[a]["t1"], res[c]["t1"]),
            b=keep["b"], n=res[a]["n"] + res[c]["n"])]
        scores = {k: v for k, v in scores.items() if (k[0] < a or k[0] > c)}
        merge_equal()
    for sg in res:
        sg["P"] = 60.0 / sg["b"]
        sg["phi"] = _seg_phase(ev, sg["P"], prm, sg["t0"], sg["t1"])
    return res


# ------------------------------------------------------------------------------------------ independent per-segment map
def _grid(b, phi, t0, t1):
    P = 60.0 / b
    n0 = int(math.ceil((t0 - phi) / P)); n1 = int(math.floor((t1 - phi) / P))
    return phi + np.arange(n0, n1 + 1) * P


def _fit_span(ev, b_guess, t0, t1, prm, rel=0.02, snap_scale=True):
    """tempo + phase of the span [t0, t1] from the onset events: fold scan around b_guess, snap, comb+fold phase"""
    span = max(t1 - t0, 1.0)
    te, we = ev.events_in(t0 - 0.3, t1 + 0.3)
    b = float(b_guess)
    if len(te) >= 12:
        r = scan_bpm(te, we, [b_guess], rel, 6.0, 4.0, ntop=2)
        if r:
            b = r[0]
    tol = min(0.25, max(0.03, 0.03 * 140.0 / span))
    r_ = round(b)
    b = float(r_) if abs(b - r_) <= tol else float(round(b, 2))
    P = 60.0 / b
    phi = _seg_phase(ev, P, prm, t0, t1)
    return b, float(phi)


def _span_align(ev, b, phi, t0, t1):
    g = _grid(b, phi, t0, t1)
    return ev.align(g) if len(g) else 0.0


def pl_map(ev, trk, f, b_glob, prm, dbg=None):
    """tempo map from the octave-normalised piecewise-linear tracker segmentation; every segment keeps its own tempo and
    phase, neighbours are merged unless two grids clearly beat one.  returns (segs, raw_segs) or (None, None)"""
    raw = pl_segments(trk, prm)
    if not raw or len(raw) <= 1:
        return None, raw
    segs = []
    anchor = max(range(len(raw)), key=lambda i: raw[i]["t1"] - raw[i]["t0"])
    bref = [None] * len(raw)
    order = [anchor] + [i for d in range(1, len(raw)) for i in (anchor - d, anchor + d) if 0 <= i < len(raw)]
    fin = {}
    for i in order:
        sg = raw[i]
        b1 = 60.0 / (sg["per"] * max(sg.get("u", 1.0), 0.25))            # the tracker's own beat level in this segment
        t0, t1 = sg["t0"], sg["t1"]
        if i != anchor and prm["pl_oct"] == "hyb":
            P1 = 60.0 / b1
            phi_c, _ = ev.comb.best_phase_for(P1, t0, t1)
            f_q, f_off = _octave_features(ev, b1, phi_c, max(0.0, t0 - 0.3), t1 + 0.3)
            fs_f = octave_factor(b1, f_q, f_off, prm)
            nb = fin[i - 1] if i > anchor else fin[i + 1]
            b_near = b1 * 2.0 ** round(math.log2(nb / b1))
            w = prm["oct_w"]
            z = w[0] + w[1] * (b1 - 115.0) / 10.0 + w[2] * f_q + w[3] * f_off
            det = b1 > prm["oct_halve"] or b1 < prm["oct_always"] or b1 >= prm["oct_never"]
            strong = det or abs(z) > prm["pl_oct_z"]
            if strong and (t1 - t0) >= prm["pl_oct_dur"]:
                b_g = b1 * fs_f
            else:
                b_g = b_near
            fs = b_g / b1
        elif i != anchor and prm["pl_oct"] == "anchor":
            cands = [b1 * 2.0 ** j for j in range(-3, 4) if prm["pl_oct_lo"] <= b1 * 2.0 ** j <= prm["pl_oct_hi"]]
            if not cands:
                cands = [b1]
            b_g = min(cands, key=lambda c: abs(math.log(c / b_glob)))
            fs = b_g / b1
        elif prm["pl_oct"] == "feat" or i == anchor:
            P1 = 60.0 / b1
            phi_c, _ = ev.comb.best_phase_for(P1, t0, t1)
            f_q, f_off = _octave_features(ev, b1, phi_c, max(0.0, t0 - 0.3), t1 + 0.3)
            fs = octave_factor(b1, f_q, f_off, prm)
            b_g = b1 * fs
            if i == anchor and prm["pl_oct"] != "feat":
                b_g = b1 * 2.0 ** round(math.log2(b_glob / b1))          # the anchor follows the global tempo octave
        else:
            nb = fin[i - 1] if i > anchor else fin[i + 1]
            b_g = b1 * 2.0 ** round(math.log2(nb / b1))
            fs = b_g / b1
        b, phi = _fit_span(ev, b_g, t0, t1, prm, prm["pl_scan_rel"])
        fin[i] = b
        segs.append((i, dict(t0=t0, t1=t1, b=b, phi=phi, P=60.0 / b, n=sg["n"], b1=b1, fs=fs)))
    segs = [d for _, d in sorted(segs, key=lambda x: x[0])]
    # tiny segments are tracker wobble: absorb them into the longer neighbour before any test
    while len(segs) > 1:
        dur = [x["t1"] - x["t0"] for x in segs]
        i = int(np.argmin(dur))
        if dur[i] >= prm["pl_min_dur"]:
            break
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = max(nb, key=lambda q: dur[q])
        a, c = (segs[min(i, j)], segs[max(i, j)])
        t0, t1 = a["t0"], c["t1"]
        keep = segs[j]
        b, phi = _fit_span(ev, keep["b"], t0, t1, prm, prm["pl_scan_rel"])
        segs[min(i, j):max(i, j) + 1] = [dict(t0=t0, t1=t1, b=b, phi=phi, P=60.0 / b, n=a["n"] + c["n"], b1=keep["b1"], fs=keep["fs"])]
    if dbg is not None:
        dbg["pl_raw"] = [(round(x["t0"], 1), round(x["t1"], 1), round(x["b"], 2), round(x["b1"], 1), x["fs"]) for x in segs]

    def union(a, c):
        t0, t1 = a["t0"], c["t1"]
        wa = max(a["t1"] - a["t0"], 1e-3); wc = max(c["t1"] - c["t0"], 1e-3)
        bg = (a["b"] * wa + c["b"] * wc) / (wa + wc)
        b, phi = _fit_span(ev, bg, t0, t1, prm, prm["pl_scan_rel"])
        return dict(t0=t0, t1=t1, b=b, phi=phi, P=60.0 / b, n=a["n"] + c["n"], b1=a["b1"], fs=a["fs"])

    def octrel(x, y):
        r = x / y
        return abs(r - 2) < 0.08 or abs(r - 0.5) < 0.04

    cache = {}

    def gain(i):
        a, c = segs[i], segs[i + 1]
        key = (round(a["t0"], 3), round(a["t1"], 3), round(a["b"], 3), round(c["t1"], 3), round(c["b"], 3), round(c["phi"], 4), round(a["phi"], 4))
        if key in cache:
            return cache[key]
        if octrel(a["b"], c["b"]):
            cache[key] = (99.0, None)
            return cache[key]
        u = union(a, c)
        sh = a if (a["t1"] - a["t0"]) <= (c["t1"] - c["t0"]) else c
        g_own = _grid(sh["b"], sh["phi"], sh["t0"], sh["t1"])
        g_un = _grid(u["b"], u["phi"], sh["t0"], sh["t1"])
        nbt = max(len(g_own), 1)
        a_own = ev.align(g_own) if len(g_own) else 0.0
        a_un = ev.align(g_un) if len(g_un) else 0.0
        r = a_own / max(a_un, 1e-9)
        sc_ = r - prm["pl_gain_k"] / math.sqrt(nbt)
        if abs(a["b"] / c["b"] - 1) < 0.004:                    # phase-only break: demand more, never a half-beat shift
            d = ((c["phi"] - a["phi"] + a["P"] / 2) % a["P"]) - a["P"] / 2
            if abs(d) > prm["pl_half_tol"] * a["P"]:
                sc_ = -1.0
            else:
                sc_ -= prm["pl_phase_extra"]
        cache[key] = (sc_, u)
        return cache[key]

    log = []
    while len(segs) > 1:
        gs = [gain(i) for i in range(len(segs) - 1)]
        i = int(np.argmin([g[0] for g in gs]))
        if gs[i][0] >= prm["pl_gain"]:
            break
        log.append((round(segs[i]["t0"], 1), round(segs[i + 1]["t1"], 1), round(gs[i][0], 3)))
        segs[i:i + 2] = [gs[i][1]]
    if dbg is not None:
        dbg["pl_merged"] = log
        dbg["pl_gains"] = [round(gain(i)[0], 3) for i in range(len(segs) - 1)]
    return segs, raw


def hint_first(segs, ev, prm, hint):
    """a user tempo hint fixes the tempo of the first segment exactly (changing it later would shift every later beat);
    when the first segment is an octave away from the hint only that segment is rescaled"""
    if hint is None or not segs:
        return segs
    s0 = segs[0]
    r = s0["b"] / hint
    k = math.log2(r)
    if s0["b"] != hint and (abs(r - 1.0) < 0.015 or (len(segs) > 1 and abs(k - round(k)) < 0.03 and abs(round(k)) <= 2)):
        n0 = dict(s0, b=float(hint), P=60.0 / hint)
        phi = _seg_phase(ev, n0["P"], prm, n0["t0"], n0["t1"])
        if abs(r - 1.0) >= 0.015 and r < 1.0:
            # the hint doubled the segment: the old grid's phase is a candidate for the on-beat phase
            def _al(ph):
                g = _grid(hint, ph, n0["t0"], n0["t1"])
                return ev.align(g) if len(g) else 0.0
            if _al(s0["phi"]) > _al(phi):
                phi = s0["phi"]
        n0["phi"] = phi
        return [n0] + list(segs[1:])
    return segs


def octave_split(ev, hint, b_auto, dur, prm):
    """The hint is an octave away from the auto tempo: the start section is notated in the hint octave, the rest in the
    auto octave.  Locate the boundary as the single step of the subdivision-density index of the onset fold (charts are
    notated at the faster octave when the music is dense).  returns (t_split, step) or None"""
    if ev.fold is None:
        return None
    L = min(hint, b_auto)
    P = 60.0 / L
    ws, hop = prm["osplit_win"], prm["osplit_hop"]
    cs = np.arange(0.0, dur - ws + 1e-9, hop)
    if len(cs) < 2 * prm["osplit_min"] + 2:
        return None
    w = prm["oct_w"]
    u = np.zeros(len(cs))
    for i, t0 in enumerate(cs):
        phi_c, _ = ev.comb.best_phase_for(P, t0, t0 + ws)
        fq, fo = _octave_features(ev, L, phi_c, t0, t0 + ws)
        u[i] = w[2] * fq + w[3] * fo
    n = len(u); mn = prm["osplit_min"]
    sse0 = float(((u - u.mean()) ** 2).sum()) + 1e-9
    best = None
    for m in range(mn, n - mn + 1):
        a, c = u[:m], u[m:]
        sse = float(((a - a.mean()) ** 2).sum() + ((c - c.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, m, float(c.mean() - a.mean()))
    sse, m, step = best
    if hint <= b_auto + 1e-6 and step <= 0:
        return None
    if hint > b_auto and step >= 0:
        return None
    if abs(step) < prm["osplit_step"] or (sse0 - sse) / sse0 < prm["osplit_gain"] or cs[m] < prm["osplit_tmin"]:
        return None
    return float(cs[m]), step


def apply_octave_split(segs, t_split, k, hint, ev, prm):
    """segs (auto octave) -> segments before t_split rescaled by k (octave of the hint), later ones kept"""
    out = []
    for sg in segs:
        if sg["t1"] <= t_split:
            parts = [(sg, True)]
        elif sg["t0"] >= t_split:
            parts = [(sg, False)]
        else:
            parts = [(dict(sg, t1=t_split), True), (dict(sg, t0=t_split), False)]
        for q, scaled in parts:
            if not scaled:
                out.append(q)
                continue
            b = q["b"] * k
            P = 60.0 / b
            phi = _seg_phase(ev, P, prm, q["t0"], q["t1"])
            if k > 1.0:
                al = lambda ph: (ev.align(_grid(b, ph, q["t0"], q["t1"])) if len(_grid(b, ph, q["t0"], q["t1"])) else 0.0)
                if al(q["phi"]) > al(phi):
                    phi = q["phi"]
            out.append(dict(q, b=b, P=P, phi=phi))
    if out and abs(out[0]["b"] / hint - 1.0) < 0.015 and out[0]["b"] != hint:
        out[0] = dict(out[0], b=float(hint), P=60.0 / hint)
        out[0]["phi"] = _seg_phase(ev, out[0]["P"], prm, out[0]["t0"], out[0]["t1"])
    return out


def merge_short_segment(segs, ev, prm):
    """Merge the segment pair with the worst junction phase mismatch (or the shortest segment), preserving the longer neighbour's tempo."""
    if len(segs) <= 1:
        return segs
    W = prm["chain_win"]
    errs = []
    for j in range(len(segs) - 1):
        sj, sk = segs[j], segs[j + 1]
        Tb = 0.5 * (sj["t1"] + sk["t0"])
        Pj, Pk = sj["P"], sk["P"]
        mc = int(round((Tb - sj["phi"]) / Pj))
        best = None
        for m in range(mc - W, mc + W + 1):
            tau = sj["phi"] + m * Pj
            n = int(round((tau - sk["phi"]) / Pk))
            d = tau - (sk["phi"] + n * Pk)
            cost = abs(d) * 1000.0 + 1.5 * abs(tau - Tb) / Pj
            if best is None or cost < best[0]:
                best = (cost, abs(d) * 1000.0)
        errs.append(best[1])
    bad_j = int(np.argmax(errs))
    if errs[bad_j] > prm["chain_tol_ms"]:
        a, c = bad_j, bad_j + 1
    else:
        i = min(range(len(segs)), key=lambda j: segs[j]["t1"] - segs[j]["t0"])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        keep_j = max(nb, key=lambda j: segs[j]["n"])
        a, c = min(i, keep_j), max(i, keep_j)
    keep = segs[a] if segs[a]["n"] >= segs[c]["n"] else segs[c]
    merged = dict(
        t0=min(segs[a]["t0"], segs[c]["t0"]),
        t1=max(segs[a]["t1"], segs[c]["t1"]),
        b=keep["b"],
        n=segs[a]["n"] + segs[c]["n"],
        ratio=keep.get("ratio", 0.0),
    )
    merged["P"] = 60.0 / merged["b"]
    merged["phi"] = _seg_phase(ev, merged["P"], prm, merged["t0"], merged["t1"])
    out = segs[:a] + [merged] + segs[c + 1:]
    k = 0
    while k < len(out) - 1:
        if abs(out[k]["b"] / out[k + 1]["b"] - 1) < 0.004:
            kp = out[k] if out[k]["n"] >= out[k + 1]["n"] else out[k + 1]
            m2 = dict(t0=out[k]["t0"], t1=out[k + 1]["t1"], b=kp["b"], n=out[k]["n"] + out[k + 1]["n"], ratio=kp.get("ratio", 0.0))
            m2["P"] = 60.0 / m2["b"]
            m2["phi"] = _seg_phase(ev, m2["P"], prm, m2["t0"], m2["t1"])
            out[k:k + 2] = [m2]
        else:
            k += 1
    return out


def build_chain(segs, prm):
    """join constant-tempo segments into one phase-continuous tempo chain.
    returns dict(phi0, P0, gj=[chain beat index of every junction], bpms=[...], errs=[ms]) or None"""
    off = [0]
    gj = []
    errs = []
    W = prm["chain_win"]
    for j in range(len(segs) - 1):
        sj, sk = segs[j], segs[j + 1]
        Tb = 0.5 * (sj["t1"] + sk["t0"])
        Pj, Pk = sj["P"], sk["P"]
        mc = int(round((Tb - sj["phi"]) / Pj))
        best = None
        for m in range(mc - W, mc + W + 1):
            tau = sj["phi"] + m * Pj
            n = int(round((tau - sk["phi"]) / Pk))
            d = tau - (sk["phi"] + n * Pk)
            g = m + off[j]
            # Prefer a junction on a bar line, then on a whole beat.  The temporal
            # midpoint remains a soft prior rather than a hard boundary.
            snap = 0.0 if g % 4 == 0 else (0.6 if g % 1 == 0 else 2.0)
            cost = abs(d) * 1000.0 + 1.5 * abs(tau - Tb) / Pj + snap
            if best is None or cost < best[0]:
                best = (cost, m, n, d, g)
        _, m, n, d, g = best
        if abs(d) * 1000.0 > prm["chain_tol_ms"]:
            return None
        gj.append(g); errs.append(d * 1000.0)
        off.append(g - n)
    if any(b <= a for a, b in zip(gj[:-1], gj[1:])):
        return None
    return dict(phi0=segs[0]["phi"], P0=segs[0]["P"], gj=gj, bpms=[s["b"] for s in segs], errs=errs)


def stitch_chain(segs, ev, prm):
    """Like build_chain, but never fails: every segment keeps its own tempo AND phase.  At a junction the new grid is
    continued from the last old beat when the two phases agree (|d| <= stitch_clean_ms); otherwise a one-beat
    transition segment (bpm = 60 / interval) jumps from the last old-grid beat to the first beat of the new grid."""
    gj = []
    bpms = [segs[0]["b"]]
    errs = []
    off = 0
    W = prm["chain_win"]
    clean_s = prm["stitch_clean_ms"] / 1000.0
    for j in range(len(segs) - 1):
        sj, sk = segs[j], segs[j + 1]
        Tb = 0.5 * (sj["t1"] + sk["t0"])
        Pj, Pk = sj["P"], sk["P"]
        Pa = 0.5 * (Pj + Pk)
        mc = int(round((Tb - sj["phi"]) / Pj))
        tw = prm["stitch_win_s"]
        best = None
        for m in range(mc - W, mc + W + 1):
            tau = sj["phi"] + m * Pj
            n = int(round((tau - sk["phi"]) / Pk))
            d = tau - (sk["phi"] + n * Pk)
            snap = 0.0 if (m + off) % 4 == 0 else (0.15 if (m + off) % 2 == 0 else 0.3)
            bj = tau - Pj * np.arange(0, int(tw / Pj) + 1)
            if abs(d) <= clean_s:
                bk = tau + Pk * np.arange(1, int(tw / Pk) + 1)
                trans = 0.0
                n_after = 1
            else:
                n2 = n + 1
                while sk["phi"] + n2 * Pk - tau < 0.7 * Pa:
                    n2 += 1
                while n2 > n + 1 and sk["phi"] + (n2 - 1) * Pk - tau >= 0.7 * Pa:
                    n2 -= 1
                t2 = sk["phi"] + n2 * Pk
                bk = t2 + Pk * np.arange(0, int(tw / Pk) + 1)
                trans = prm["stitch_trans_cost"]
            sc = 0.5 * (ev.align(bj) + ev.align(bk))
            cost = -sc + trans + snap + 0.05 * abs(tau - Tb) / Pj
            if best is None or cost < best[0]:
                best = (cost, m, n, d, tau)
        _, m, n, d, tau = best
        g = m + off
        if abs(d) <= clean_s:
            gj.append(g); bpms.append(sk["b"])
            off = g - n
            errs.append(d * 1000.0)
        else:
            n2 = n + 1
            Pa_ = Pa
            while sk["phi"] + n2 * Pk - tau < 0.7 * Pa_:
                n2 += 1
            while n2 > n + 1 and sk["phi"] + (n2 - 1) * Pk - tau >= 0.7 * Pa_:
                n2 -= 1
            t2 = sk["phi"] + n2 * Pk
            bt = 60.0 / (t2 - tau)
            gj.append(g); bpms.append(float(bt))
            gj.append(g + 1); bpms.append(sk["b"])
            off = g + 1 - n2
            errs.append(0.0)
    return dict(phi0=segs[0]["phi"], P0=segs[0]["P"], gj=gj, bpms=bpms, errs=errs, stitched=True)


def chain_times(chain, g):
    """time (s) of chain beat g (integer array) ; chain beats before the first junction use the first tempo"""
    beats = np.array([-1000.0] + [float(x) for x in chain["gj"]])
    bpms = np.array(chain["bpms"], np.float64)
    t_ref = chain["phi0"] - 1000.0 * chain["P0"]
    return t_ref + _tempo_sec(beats, bpms, np.asarray(g, np.float64) + 0.0)


# ------------------------------------------------------------------------------------------ main
def grid_from_frames(bl, dl, dur, bpm_hint=None, params=None, onset=None):
    """numpy part of the estimator. bl/dl: tracker logits (50 fps); onset: optional onset envelope (5 ms hop)."""
    bl = np.asarray(bl)
    dl = np.asarray(dl)
    if (bl.ndim != 1 or dl.ndim != 1 or not len(bl) or bl.shape != dl.shape
            or not np.isfinite(bl).all() or not np.isfinite(dl).all()):
        raise ValueError("beat/downbeat logits must be nonempty finite matching one-dimensional arrays")
    try:
        dur = float(dur)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration must be finite and positive") from exc
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError("duration must be finite and positive")
    if bpm_hint is not None:
        try:
            bpm_hint = float(bpm_hint)
        except (TypeError, ValueError) as exc:
            raise ValueError("bpm_hint must be finite and positive") from exc
        if not math.isfinite(bpm_hint) or bpm_hint <= 0:
            raise ValueError("bpm_hint must be finite and positive")
    prm = dict(DEFAULT_PARAMS, **(params or {}))
    dbg = {}
    ev = _Evidence(bl, onset, prm)
    dcurve = _down_curve(dl, prm["down_mode"])
    shifts = [0.0]
    f = 1.0
    trk = None
    # bpm_hint + detect_changes: the hint fixes the tempo OCTAVE of the start of the song (e.g. auto said 130, the
    # song is 260); tempo changes are still detected, relative to the hinted octave.  Without detect_changes, or
    # when the tracker finds no beats, the hint is used as a constant tempo (the old behaviour).
    hint_changes = bpm_hint is not None and bool(prm.get("detect_changes"))
    b_auto = None
    osplit = None
    if hint_changes:
        trk = tracker_tempo(bl)
    if bpm_hint is not None and trk is None:
        b = float(bpm_hint)
        dbg["mode"] = "hint"
    else:
        if trk is None:
            trk = tracker_tempo(bl)
        if trk is None:
            dbg["fail"] = "no beats"
            return dict(first=0.0, beats=[0.0], bpms=[120.0], debug=dbg)
        dbg["b_reg"] = round(trk["b_reg"], 3); dbg["b_hist"] = round(trk["b_hist"], 3)
        te, we = ev.scan_events(trk)
        centers = [trk["b_reg"]]
        if abs(trk["b_hist"] / trk["b_reg"] - 1) > 0.01:
            centers.append(trk["b_hist"])
        r = scan_bpm(te, we, centers, prm["scan_rel"], prm["scan_sigma_ms"], prm["fine_sigma_ms"])
        if r is None:
            b1 = trk["b_reg"]; dbg["scan"] = "failed"
        else:
            b1 = r[0]; dbg["scan_score"] = round(r[1], 3)
        dbg["b_scan"] = round(b1, 4)
        P1 = 60.0 / b1
        phi1, _ = ev.comb.best_phase_for(P1)
        f_q, f_off = _octave_features(ev, b1, phi1)
        f = octave_factor(b1, f_q, f_off, prm)
        dbg["octave"] = f; dbg["f_q"] = round(f_q, 3); dbg["f_off"] = round(f_off, 3)
        b = b1 * f
        if f != 1.0 and r is not None:                       # re-scan at the final octave for the final precision
            r2 = scan_bpm(te, we, [b], 0.002, prm["scan_sigma_ms"], prm["fine_sigma_ms"], ntop=2)
            if r2 is not None:
                b = r2[0]
        b = snap_bpm(b, prm)
        if f == 0.5:
            shifts = [0.0, 0.5]
        if hint_changes and prm.get("hint_nonoct_ignore") and abs(math.log2(float(bpm_hint) / b) - round(math.log2(float(bpm_hint) / b))) > 0.03:
            # not an octave of the auto tempo: the hint is the tempo of an early section only, the song is tracked freely
            hint_changes = False
            dbg["hint_ignored"] = True
        osplit = None
        if hint_changes and prm.get("hint_osplit") and abs(math.log2(float(bpm_hint) / b)) > 0.5:
            osplit = octave_split(ev, float(bpm_hint), b, dur, prm)
            if osplit is not None:
                osplit_k = 2.0 ** round(math.log2(float(bpm_hint) / b))
                dbg["osplit"] = (round(osplit[0], 1), round(osplit[1], 2), osplit_k)
        if hint_changes and osplit is None:
            k = 2.0 ** round(math.log2(float(bpm_hint) / b))
            f *= k
            b_auto = b
            b = float(bpm_hint)
            shifts = [0.0, 0.5] if f < 1.0 else [0.0]
            # local-tempo octave ranges must contain the hinted tempo (only widened, never narrowed)
            prm["seg_lo"] = min(prm["seg_lo"], b / 1.25); prm["seg_hi"] = max(prm["seg_hi"], b * 1.3)
            prm["raw_seg_lo"] = min(prm.get("raw_seg_lo", 105.0), b / 1.25); prm["raw_seg_hi"] = max(prm.get("raw_seg_hi", 260.0), b * 1.3)
            dbg["mode"] = "hint+changes"; dbg["hint_octave_k"] = k
    P = 60.0 / b

    # ---- constant-tempo phase
    phi = _fine_phase(ev, P, prm, dbg)
    if hint_changes and dbg.get("hint_octave_k", 1.0) > 1.0:
        # the hint doubled the auto tempo: the auto beats are every k-th true beat, so their phase is a
        # candidate for the true phase (the fold at the doubled period cannot tell the on- from the off-beat)
        phi_a = float(_fine_phase(ev, 60.0 / b_auto, prm, {})) % P

        def _al(ph):
            tk = ph + np.arange(-8, int(dur / P) + 16) * P
            return ev.align(tk[(tk >= 0) & (tk <= dur)])

        if _al(phi_a) > _al(phi):
            phi = phi_a
            dbg["phase_from_auto"] = True
    dbg["phi"] = float(phi); dbg["b"] = float(b); dbg["octave_f"] = float(f)

    # ---- tempo changes (only with an onset envelope and the tracker; with a bpm hint only in "hint+changes" mode)
    chain = None
    best_segs = None
    if prm["detect_changes"] and trk is not None and ev.fold is not None:
        tk = phi + np.arange(-8, int(dur / P) + 16) * P
        tk = tk[(tk >= 0) & (tk <= dur)]
        a_k = ev.align(tk)
        dbg["align_const"] = round(a_k, 3)

        def _try_mode(use_raw):
            if use_raw:
                m_prm = dict(
                    prm,
                    seg_glob_ratio=prm.get("raw_seg_glob_ratio", 1.04),
                    seg_octave_keep_dur=prm.get("raw_seg_octave_keep_dur", 4.5),
                    seg_lo=prm.get("raw_seg_lo", 105.0),
                    seg_hi=prm.get("raw_seg_hi", 260.0),
                    chain_tol_ms=prm.get("raw_chain_tol_ms", 40.0),
                    chain_gain=prm.get("raw_chain_gain", 1.035),
                )
            else:
                m_prm = prm
            segs0 = tempo_segments(trk, m_prm, use_raw=use_raw)
            if not segs0 or len(segs0) <= 1:
                return None, None, 0, None
            segs = refine_segments(ev, segs0, f, b, m_prm)
            best_cand = None
            cur = segs
            fails = 0
            for _attempt in range(4):
                if len(cur) <= 1:
                    break
                chs = []
                if hint_changes and osplit is None:
                    cur = hint_first(cur, ev, m_prm, bpm_hint)
                ch0 = build_chain(cur, m_prm)
                if ch0 is not None:
                    chs.append(ch0)
                if prm.get("stitch"):
                    chs.append(stitch_chain(cur, ev, m_prm))
                if not chs:
                    fails += 1
                for ch in chs:
                    gg = np.arange(-8, int(dur / (60.0 / max(ch["bpms"]))) + 16)
                    tc = chain_times(ch, gg)
                    tc = tc[(tc >= 0) & (tc <= dur)]
                    a_ch = ev.align(tc)
                    key = (a_ch / max(a_k, 1e-9), -len(ch["bpms"]))
                    if best_cand is None or key > best_cand[0]:
                        best_cand = (key, ch, a_ch, cur, m_prm["chain_gain"])
                    if use_raw and key[0] >= m_prm["chain_gain"]:
                        break
                if len(cur) > 1:
                    cur = merge_short_segment(cur, ev, m_prm)
            return segs0, segs, fails, best_cand

        segs0_idx, segs_idx, fails_idx, cand_idx = _try_mode(False)
        segs0_raw, segs_raw, fails_raw, cand_raw = (_try_mode(True) if "tt" in trk else (None, None, 0, None))

        cand_pl = None
        segs_pl = raw_pl = None
        if prm.get("pl"):
            segs_pl, raw_pl = pl_map(ev, trk, f, (b_auto if (hint_changes and b_auto and not prm.get("pl_anchor_hint")) else b), prm, dbg)
            if segs_pl is not None and hint_changes and osplit is None:
                segs_pl = hint_first(segs_pl, ev, prm, bpm_hint)
            if segs_pl is not None and len(segs_pl) > 1:
                ch = stitch_chain(segs_pl, ev, prm)
                gg = np.arange(-8, int(dur / (60.0 / max(ch["bpms"]))) + 16)
                tc = chain_times(ch, gg); tc = tc[(tc >= 0) & (tc <= dur)]
                a_ch = ev.align(tc)
                cand_pl = ((a_ch / max(a_k, 1e-9), -len(ch["bpms"])), ch, a_ch, segs_pl, prm["pl_final_gain"])
        if prm.get("dbg_cands"):
            dbg["cands"] = {}
            for nm_, cn_ in (("idx", cand_idx), ("raw", cand_raw), ("pl", cand_pl)):
                if cn_ is not None:
                    dbg["cands"][nm_] = (cn_[1], float(cn_[0][0]))
            dbg["cands"]["const"] = (dict(phi0=float(phi), P0=float(P), gj=[], bpms=[float(b)], errs=[]), 1.0)
        valid_cands = []
        for mode_name, s0, s_ref, fls, cnd in (("idx", segs0_idx, segs_idx, fails_idx, cand_idx),
                                                ("raw", segs0_raw, segs_raw, fails_raw, cand_raw),
                                                ("pl", raw_pl, segs_pl, 0, cand_pl)):
            if cnd is not None and cnd[0][0] >= cnd[4]:
                valid_cands.append((cnd[0], mode_name, s0, s_ref, fls, cnd))

        if valid_cands:
            valid_cands.sort(key=lambda x: x[0], reverse=True)
            if prm.get("pl_first"):
                pls = [c for c in valid_cands if c[1] == "pl"]
                if pls:
                    valid_cands = pls + [c for c in valid_cands if c[1] != "pl"]
            _, mode_name, s0, s_ref, fls, best_candidate = valid_cands[0]
            key, ch, a_ch, best_segs, _ = best_candidate
            dbg["seg_mode"] = mode_name
            dbg["n_segs_raw"] = len(s0)
            dbg["segs"] = [(round(sg["t0"], 1), round(sg["t1"], 1), sg["b"], round(sg.get("ratio", 0), 2)) for sg in s_ref]
            dbg["align_chain"] = round(a_ch, 3)
            dbg["chain_ratio"] = round(key[0], 3)
            chain = ch
            dbg["chain_errs_ms"] = [round(x, 1) for x in ch["errs"]]
            dbg["segs_final"] = [(round(sg["t0"], 1), round(sg["t1"], 1), sg["b"]) for sg in best_segs]
        else:
            # Record diagnostics from whichever mode came closer
            pick = (segs0_raw, segs_raw, fails_raw, cand_raw) if (cand_raw is not None and (cand_idx is None or cand_raw[0] > cand_idx[0])) else (segs0_idx, segs_idx, fails_idx, cand_idx)
            s0, s_ref, fls, cnd = pick
            if s0 is not None:
                dbg["n_segs_raw"] = len(s0)
                dbg["segs"] = [(round(sg["t0"], 1), round(sg["t1"], 1), sg["b"], round(sg.get("ratio", 0), 2)) for sg in s_ref]
            if cnd is not None:
                key, ch, a_ch, best_segs, _ = cnd
                dbg["align_chain"] = round(a_ch, 3)
                dbg["chain_ratio"] = round(key[0], 3)
            if s0 is not None and len(s0) > 1:
                dbg["chain"] = "failed" if fls else "weak"

    # ---- hint octave section: the start section is notated in the hint octave
    if osplit is not None:
        segs_o = best_segs if chain is not None else [dict(t0=0.0, t1=dur, b=float(b), phi=float(phi), P=float(P), n=1)]
        segs_o = apply_octave_split(segs_o, osplit[0], osplit_k, float(bpm_hint), ev, prm)
        if len(segs_o) > 1:
            chain = stitch_chain(segs_o, ev, prm)
        else:
            b = float(segs_o[0]["b"]); P = 60.0 / b; phi = float(segs_o[0]["phi"]); chain = None
        dbg["mode"] = "hint+changes+osplit"

    # ---- bar phase
    if chain is None:
        tfun = lambda g, s=0.0: phi + (np.asarray(g, np.float64) + s) * P
        gmax = int(dur / P) + 8
    else:
        tfun = lambda g, s=0.0: chain_times(chain, np.asarray(g, np.float64) + s)
        gmax = int(dur / (60.0 / max(chain["bpms"]))) + 8
        shifts = [0.0]
    best = _down_phase(dcurve, tfun, dur, shifts, (-8, gmax), prm["first_prior"], prm["start_prior"])
    if best is None:
        best = (0.0, 0, 0.0)
    _, p, s = best
    dbg["down_phase"] = p
    gs = np.arange(-8, gmax)
    taus = tfun(gs, s)
    cand = np.nonzero((gs % 4 == p) & (taus >= -prm["first_snap_ms"] / 1000.0))[0]
    g0 = int(gs[cand[0]]) if len(cand) else int(gs[0])
    first = float(tfun(g0, s))
    dbg["first_raw"] = round(first, 4)
    if first < 0:
        first = 0.0
    if chain is None:
        beats, bpms = [0.0], [float(b)]
    else:
        j0 = int(np.searchsorted(np.array(chain["gj"]), g0, side="right"))
        beats = [0.0] + [float(x - g0) for x in chain["gj"][j0:]]
        bpms = [float(x) for x in chain["bpms"][j0:]]
        dbg["chain"] = True
    if hint_changes and bpms and abs(bpms[0] / float(bpm_hint) - 1.0) < 0.015:
        bpms[0] = float(bpm_hint)                       # the user's tempo for the start, exactly
    dbg["snapped"] = [round(x, 3) for x in bpms]
    return dict(first=float(first), beats=beats, bpms=bpms, debug=dbg)


def estimate_grid(audio_path, device="cuda:1", bpm_hint=None, params=None):
    """beat grid of a song. returns dict(first, beats, bpms, debug); see module docstring"""
    if bpm_hint is not None:
        try:
            bpm_hint = float(bpm_hint)
        except (TypeError, ValueError) as exc:
            raise ValueError("bpm_hint must be finite and positive") from exc
        if not math.isfinite(bpm_hint) or bpm_hint <= 0:
            raise ValueError("bpm_hint must be finite and positive")
    prm = dict(DEFAULT_PARAMS, **(params or {}))
    tm = {}
    t0 = time.time()
    wave = load_wave(audio_path)
    tm["load"] = time.time() - t0
    t0 = time.time()
    bl, dl = _tracker(device)(wave, SR)
    bl = bl.detach().cpu().numpy().astype(np.float32)
    dl = dl.detach().cpu().numpy().astype(np.float32)
    tm["model"] = time.time() - t0
    t0 = time.time()
    onset = onset_envelope(wave)
    tm["onset"] = time.time() - t0
    t0 = time.time()
    dur = len(wave) / SR
    res = grid_from_frames(bl, dl, dur, bpm_hint=bpm_hint, params=prm, onset=onset)
    tm["fit"] = time.time() - t0
    res["debug"]["timing"] = tm
    res["debug"]["duration"] = dur
    return res
