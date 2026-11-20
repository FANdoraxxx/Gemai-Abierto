"""Per-bar audio features for the section predictor (numpy only, shared by training and inference).

Input = the v3 *bar table* of the song (mai/v3.compute_bar_table, float16 [nb, 1218], first 1024 cols = mean MERT-330M L8 of the bar,
cols 1024..1120 = mean normalised log-mel, 1120..1216 = its temporal std, 1216 = spectral flux, 1217 = fraction of the bar inside the audio).
The table depends on AUDIO + BAR GRID only (never on chart notes), so the same function is used for official songs (precomputed
E:\\maiAIV3_feats\\bars_v3) and for generated/fresh audio (computed at inference from mel + MERT).

Feature groups (see FEATURE_NAMES):
  * pca  : MERT bar vector, song-centred + L2-normalised per bar, projected on a PCA basis fitted on the training songs (structure, not timbre)
  * sim  : self-similarity of that centred vector: checkerboard novelty, recurrence vs bars >= 8 back (choruses repeat), neighbour sims
  * aud  : loudness z (2-bar smoothed, raw, local mean, slope), brightness, spectral flux, mel-std, audio-validity
  * pos  : position in the song, song length, log bpm
"""
from __future__ import annotations

import numpy as np

D_MERT = 1024
C_MEL0 = 1024
C_STD0 = 1120
C_FLUX = 1216
C_VF = 1217
N_PCA = 64


def _sm(x, k=2):
    x = np.asarray(x, np.float64)
    if k <= 1 or len(x) == 0:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def centred_mert(tab):
    """-> (Cn [nb,1024] float32 song-centred + per-bar L2-normalised MERT vectors, ok [nb] bool rows with a real feature)"""
    M = np.asarray(tab[:, :D_MERT], np.float32)
    ok = np.linalg.norm(M, axis=1) > 1e-3
    Mu = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-6)
    if ok.sum() >= 2:
        C = Mu - Mu[ok].mean(0, keepdims=True)
    else:
        C = Mu
    C[~ok] = 0.0
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-6)
    Cn[~ok] = 0.0
    return Cn.astype(np.float32), ok


def novelty(Cn, L=4):
    """checkerboard novelty at every bar (0 = same music on both sides); same kernel as mai/ann.bar_novelty, on the centred vectors"""
    nb = len(Cn)
    S = Cn @ Cn.T
    nov = np.zeros(nb)
    for b in range(2, nb - 1):
        a0, a1, b0, b1 = max(0, b - L), b, b, min(nb, b + L)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        nov[b] = S[a0:a1, a0:a1].mean() + S[b0:b1, b0:b1].mean() - 2.0 * S[a0:a1, b0:b1].mean()
    nov = np.maximum(nov, 0.0)
    m = nov.max()
    return (nov / m if m > 1e-9 else nov).astype(np.float32)


def recurrence(Cn, gap=8, k=4):
    """recurrence of every bar against the earlier bars: max cosine similarity to bars <= b-gap, and the mean of its top-k such sims.
    A chorus bar typically re-states an earlier block -> high value; a through-composed verse -> low."""
    nb = len(Cn)
    S = Cn @ Cn.T
    mx = np.zeros(nb, np.float32)
    top = np.zeros(nb, np.float32)
    for b in range(gap, nb):
        if not np.any(Cn[b]):
            continue
        s = S[b, : b - gap + 1]
        if len(s) == 0:
            continue
        mx[b] = float(s.max())
        top[b] = float(np.sort(s)[-min(k, len(s)):].mean())
    return mx, top


def fit_pca(Cn, n=N_PCA, max_bars=60000, seed=0):
    """PCA basis (mean, components [n,1024]) of the centred MERT bars; subsamples rows for speed"""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(np.linalg.norm(Cn, axis=1) > 0)
    if len(idx) > max_bars:
        idx = idx[rng.choice(len(idx), max_bars, replace=False)]
    X = np.asarray(Cn[idx], np.float32)
    mean = X.mean(0)
    Xc = X - mean
    # randomised SVD
    Q = np.linalg.qr(Xc @ rng.normal(size=(D_MERT, n + 12)).astype(np.float32))[0]
    B = Q.T @ Xc
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mean.astype(np.float32), Vt[:n].astype(np.float32)


def bar_feats(tab, pca_mean, pca_comp, bpm=None):
    """bar table [nb,1218] -> float32 [nb, D_FEAT] (see FEATURE_NAMES)"""
    tab = np.asarray(tab, np.float32)
    nb = len(tab)
    Cn, ok = centred_mert(tab)
    proj = Cn @ np.asarray(pca_comp, np.float32).T - np.asarray(pca_mean, np.float32) @ np.asarray(pca_comp, np.float32).T
    nov = novelty(Cn)
    rec_mx, rec_top = recurrence(Cn)
    S = Cn @ Cn.T
    prev = np.zeros(nb, np.float32)
    nxt = np.zeros(nb, np.float32)
    if nb > 1:
        prev[1:] = np.sum(Cn[1:] * Cn[:-1], 1)
        nxt[:-1] = np.sum(Cn[:-1] * Cn[1:], 1)
    mel = tab[:, C_MEL0:C_MEL0 + 96]
    std = tab[:, C_STD0:C_STD0 + 96]
    loud = mel.mean(1)
    loud_s = _sm(loud, 2)
    med = float(np.median(loud)) if nb else 0.0
    scale = float(np.std(loud)) + 1e-6
    lz = (loud - med) / scale
    lz_s = (loud_s - med) / scale
    bright = mel[:, 60:].mean(1) - mel[:, :30].mean(1)
    lo = mel[:, :24].mean(1)
    mid = mel[:, 24:60].mean(1)
    hi = mel[:, 60:].mean(1)
    flux = tab[:, C_FLUX]
    flux_z = (flux - float(np.median(flux))) / (float(np.std(flux)) + 1e-6)
    stdz = std.mean(1)
    stdz = (stdz - float(np.median(stdz))) / (float(np.std(stdz)) + 1e-6)
    vf = tab[:, C_VF]
    local = _sm(lz, 9)
    slope = np.zeros(nb, np.float32)
    for b in range(nb):
        a, c = max(0, b - 4), min(nb, b + 5)
        if c - a >= 2:
            slope[b] = (lz_s[min(c - 1, b + 4)] - lz_s[max(a, b - 4)])
    b = np.arange(nb, dtype=np.float32)
    pos = b / max(nb, 1)
    L = np.full(nb, np.log1p(nb) / 5.0, np.float32)
    Lb = np.full(nb, np.log(max(bpm or 120.0, 1.0)), np.float32) / 5.0
    cols = [proj,
            nov[:, None], rec_mx[:, None], rec_top[:, None], prev[:, None], nxt[:, None],
            lz[:, None], lz_s[:, None], (lz_s - local)[:, None], slope[:, None],
            bright[:, None], flux_z[:, None], stdz[:, None], flux[:, None],
            lo[:, None], mid[:, None], hi[:, None], vf[:, None],
            pos[:, None], (1.0 - pos)[:, None], L[:, None], Lb[:, None],
            ok.astype(np.float32)[:, None]]
    F = np.concatenate(cols, 1).astype(np.float32)
    return F


def feature_names():
    names = [f"pca{i}" for i in range(N_PCA)]
    names += ["nov", "rec_max", "rec_top", "sim_prev", "sim_next", "loud_z", "loud_zs", "loud_rel", "loud_slope",
              "bright", "flux_z", "std_z", "flux", "mel_lo", "mel_mid", "mel_hi", "vf", "pos", "pos_rev", "len", "bpm", "ok"]
    return names


def clip_feats(F):
    """robust clip of the numeric columns (inference safety for unusual audio)"""
    F = np.array(F, np.float32, copy=True)
    return np.clip(F, -20.0, 20.0)


def intensity(tab):
    """per-bar audio intensity of a bar table -> dict(loud_z, loud_zs, flux_z, bright, ok); loud_z = (mel dB - song median) / song std"""
    tab = np.asarray(tab, np.float32)
    mel = tab[:, C_MEL0:C_MEL0 + 96]
    loud = mel.mean(1)
    med, sc = float(np.median(loud)), float(np.std(loud)) + 1e-6
    lz = (loud - med) / sc
    flux = tab[:, C_FLUX]
    fz = (flux - float(np.median(flux))) / (float(np.std(flux)) + 1e-6)
    return dict(loud_z=lz.astype(np.float32), loud_zs=_sm(lz, 2).astype(np.float32), flux_z=fz.astype(np.float32),
                bright=(mel[:, 60:].mean(1) - mel[:, :30].mean(1)).astype(np.float32),
                ok=(np.linalg.norm(tab[:, :D_MERT], axis=1) > 1e-3))


N_FEAT = len(feature_names())
