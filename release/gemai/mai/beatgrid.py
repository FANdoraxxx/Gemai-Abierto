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
def _octave_features(ev, b1, phi_c):
    """beat-subdivision energy of the onset fold at tempo b1 (relative to the beat peak)"""
    if ev.fold is None:
        return 0.0, 0.0
    P = 60.0 / b1
    hs, nb = ev.fold.curve(P, 4.0)
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
        if hint_changes:
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
                ch = build_chain(cur, m_prm)
                if ch is not None:
                    gg = np.arange(-8, int(dur / (60.0 / max(ch["bpms"]))) + 16)
                    tc = chain_times(ch, gg)
                    tc = tc[(tc >= 0) & (tc <= dur)]
                    a_ch = ev.align(tc)
                    key = (a_ch / max(a_k, 1e-9), -len(ch["bpms"]))
                    if best_cand is None or key > best_cand[0]:
                        best_cand = (key, ch, a_ch, cur, m_prm["chain_gain"])
                    if use_raw and key[0] >= m_prm["chain_gain"]:
                        break
                else:
                    fails += 1
                if len(cur) > 1:
                    cur = merge_short_segment(cur, ev, m_prm)
            return segs0, segs, fails, best_cand

        segs0_idx, segs_idx, fails_idx, cand_idx = _try_mode(False)
        segs0_raw, segs_raw, fails_raw, cand_raw = (_try_mode(True) if "tt" in trk else (None, None, 0, None))

        valid_cands = []
        for mode_name, s0, s_ref, fls, cnd in (("idx", segs0_idx, segs_idx, fails_idx, cand_idx),
                                                ("raw", segs0_raw, segs_raw, fails_raw, cand_raw)):
            if cnd is not None and cnd[0][0] >= cnd[4]:
                valid_cands.append((cnd[0], mode_name, s0, s_ref, fls, cnd))

        if valid_cands:
            valid_cands.sort(key=lambda x: x[0], reverse=True)
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
