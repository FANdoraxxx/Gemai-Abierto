"""Shared training/inference window construction.

Notes supplied to ``pack_notes`` are already quantized to integer ticks and in
canonical order.  Token windows and audio windows use the same relative origin:
history [-768, 0), target [0, 768), audio [-768, 960).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .audio import FPS
from .tokens import TPB, T_MIN, T_MAX, L_BEATS, encode_note

BAR = 4 * TPB
HIST = -T_MIN
TGT = T_MAX
FUT_TICKS = 8 * TPB          # v6 infill: future span = 2 bars (8 beats) after the target window
LOOK = L_BEATS * TPB
N_FRAMES = HIST + TGT + LOOK
N_MEL = 96
MAX_LEN = 1024
HEADER = 12
INVALID = -4.0

_TICKS0 = np.arange(N_FRAMES, dtype=np.float64)
_BAND = np.arange(N_MEL, dtype=np.float32) / 95.0 - 0.5


def _rnd(x):
    return int(math.floor(x + 0.5))


def feature_token(V, name, state):
    """seq_version 2 feature token.  name in TOUCH / EXN / FEST.
    state: None or "auto" -> *_UNK (model decides); False / "off" / 0 -> *_N (none); True / "on" -> *_Y (present, amount
    unspecified); 1 / "low" -> *_L, 2 / "mid" -> *_M, 3 / "high" -> *_H (amount tertiles among Master charts that use it)."""
    if state is None or state == "auto":
        return V[f"{name}_UNK"]
    if state is True or state == "on":
        return V[f"{name}_Y"]
    if state is False or state == "off":
        return V[f"{name}_N"]
    if isinstance(state, str):
        state = {"none": 0, "low": 1, "mid": 2, "high": 3}[state]
    return V[f"{name}_{'NLMH'[int(state)]}"]


def density_token(V, bucket):
    """seq_version 2 chart-density token; bucket None -> DENS_UNK, else 0 .. N_DENS-1 (see mai.v2.density_bucket)."""
    return V["DENS_UNK"] if bucket is None else V[f"DENS_{int(bucket)}"]


def plan_token(V, bucket):
    """seq_version 3 planned-density token of the current window; bucket None -> PLAN_UNK, else 0 .. N_PLAN-1 (see mai.v3.plan_bucket)."""
    return V["PLAN_UNK"] if bucket is None else V[f"PLAN_{int(bucket)}"]


def bd_tokens(V, buckets):
    """seq_version 5: the 4 BD tokens; buckets None -> BD1..4_UNK, else 4 ints 0 .. N_BD-1 (mai.v3.window_bd)"""
    if buckets is None:
        return np.asarray([V[f"BD{j}_UNK"] for j in range(1, 5)], np.int32)
    return np.asarray([V[f"BD{j}_{int(b)}"] for j, b in zip(range(1, 5), buckets)], np.int32)


def bdi_ids(V, buckets):
    """seq_version 6: the 4 interleaved BDI token ids; buckets None -> BDI_UNK for every bar"""
    if buckets is None:
        return [V.bdi0] * 4
    return [V.bdi0 + 1 + int(b) for b in buckets]


def sec_token(V, label):
    """seq_version 4 section-label condition of the window; label None -> SEC_UNK, else 0 .. N_SEC-1 (mai.ann.LABELS)"""
    return V["SEC_UNK"] if label is None else V[f"SEC_{int(label)}"]


def secn_token(V, flag):
    """seq_version 4 "new section starts inside the window" condition; flag None -> SECN_UNK, else 0 / 1"""
    return V["SECN_UNK"] if flag is None else V[f"SECN_{int(flag)}"]


def unk_condition_row(V):
    """all-UNK condition row (CFG unconditional row); length V.n_cond"""
    names = ("LV", "SLOT", "TYPE", "TOUCH", "EXN", "FEST", "DES", "VER") + (("DENS",) if V.seq_version >= 2 else ()) + \
            (("PLAN",) if V.seq_version >= 3 else ()) + (("SEC", "SECN") if V.seq_version >= 4 else ()) + \
            (("BD1", "BD2", "BD3", "BD4") if V.seq_version == 5 else ())
    return np.asarray([V[f"{n}_UNK"] for n in names], np.int32)


def condition_tokens(V, chart):
    """Return the condition tokens (8 for seq_version 1, 9 for seq_version 2), with the training metadata semantics.

    seq_version 2 reads the optional chart keys touch_b / ex_b / fest_b (0 none, 1..3 low/mid/high; fall back to the boolean
    touch / ex / fest -> N / Y) and dens_b (density bucket; absent -> DENS_UNK)."""
    row = _condition_tokens_v1(V, chart)
    if getattr(V, "seq_version", 1) < 2:
        return row
    for i, (name, flag) in enumerate((("TOUCH", "touch"), ("EXN", "ex"), ("FEST", "fest")), 3):
        b = chart.get(flag + "_b")
        if b is not None:
            row[i] = feature_token(V, name, int(b))
    extra = [density_token(V, chart.get("dens_b"))]
    if V.seq_version >= 3:
        extra.append(plan_token(V, chart.get("plan_b")))
    if V.seq_version >= 4:
        extra.append(sec_token(V, chart.get("sec_b")))
        extra.append(secn_token(V, chart.get("secn_b")))
    if V.seq_version == 5:
        extra.extend(bd_tokens(V, chart.get("bd_b")).tolist())
    return np.concatenate([row, np.asarray(extra, np.int32)])


def _condition_tokens_v1(V, chart):
    lv = chart.get("lv")
    if lv is None or (isinstance(lv, float) and math.isnan(lv)):
        lvt = V["LV_UNK"]
    else:
        lvt = V[f"LV_{min(max(_rnd(lv * 10) - 10, 0), 140)}"]
    slot = int(chart["slot"]) - 2
    slt = V[f"SLOT_{slot}"] if 0 <= slot < 5 else V["SLOT_UNK"]
    typ = V["TYPE_DX"] if chart.get("dx") else V["TYPE_STD"]
    yn = lambda name, flag: V[f"{name}_{'Y' if flag else 'N'}"]
    des = int(chart.get("des_id", -2))
    if des >= 0 and f"DES_{des}" in V.idx:
        dt = V[f"DES_{des}"]
    elif des == -1 or des >= 0:
        dt = V["DES_OTHER"]
    else:
        dt = V["DES_UNK"]
    ver = int(chart.get("versionid", -1))
    vt = V[f"VER_{ver}"] if 0 <= ver < 32 else V["VER_UNK"]
    return np.array([lvt, slt, typ, yn("TOUCH", chart.get("touch")),
                     yn("EXN", chart.get("ex")), yn("FEST", chart.get("fest")),
                     dt, vt], dtype=np.int32)


def bpm_progress_tokens(V, tempo, start_tick, song_beats):
    """Return the BPM bucket at the window start and the song-progress bucket."""
    sb = start_tick / TPB
    bpm = tempo.bpm_at(sb)
    bt = V[f"BPM_{min(max(_rnd((bpm - 40) / 10), 0), 39)}"]
    pr = min(max(int(math.floor(10 * sb / song_beats)), 0), 9)
    return bt, V[f"PROG_{pr}"]


def pack_notes(V, notes):
    """Pack canonical tick notes into the same arrays written by tools/prep.py.

    Input order and notes are preserved. ``offs`` addresses the encoded note
    bodies; time tokens are added only when a window is built.
    """
    notes = list(notes)
    bodies = [encode_note(V, note) for note in notes]
    offs = np.zeros(len(notes) + 1, dtype=np.int32)
    if bodies:
        offs[1:] = np.cumsum([len(body) for body in bodies])
    toks = np.fromiter(itertools.chain.from_iterable(bodies), dtype=np.int16,
                       count=int(offs[-1]))
    ticks = np.fromiter((int(note.beat) for note in notes), dtype=np.int32,
                        count=len(notes))
    return ticks, offs, toks


def _events(V, ticks, offs, toks, a, b, start_tick, out_t, out_c):
    if b <= a:
        return
    o0 = int(offs[a])
    tk = toks[o0:int(offs[b])].tolist()
    of = (offs[a:b + 1] - o0).tolist()
    tl = (ticks[a:b] - start_tick).tolist()
    time0 = V.time0
    last = None
    for i in range(b - a):
        t = tl[i]
        if t != last:
            out_t.append(time0 + t - T_MIN)
            out_c.append(t)
            last = t
        seg = tk[of[i]:of[i + 1]]
        out_t.extend(seg)
        out_c.extend([t] * len(seg))


def _events_fut(V, ticks, offs, toks, a, b, fut_start, cur_base, out_t, out_c):
    """Emit future events: time tokens relative to ``fut_start`` (T_0 .. T_383), cur_time = cur_base + relative
    (cur_base == TGT for the true future -> 768 .. 1151)."""
    if b <= a:
        return
    o0 = int(offs[a])
    tk = toks[o0:int(offs[b])].tolist()
    of = (offs[a:b + 1] - o0).tolist()
    tl = (ticks[a:b] - int(fut_start)).tolist()
    time0 = V.time0
    last = None
    for i in range(b - a):
        t = tl[i]
        if t != last:
            out_t.append(time0 + t - T_MIN)
            out_c.append(int(cur_base) + t)
            last = t
        seg = tk[of[i]:of[i + 1]]
        out_t.extend(seg)
        out_c.extend([int(cur_base) + t] * len(seg))


def _costs(ticks, offs, a, b):
    """Token cost per note, counting a time token whenever the tick changes."""
    if b <= a:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    lens = np.diff(offs[a:b + 1]).astype(np.int64)
    tl = ticks[a:b]
    new = np.empty(b - a, bool)
    new[0] = True
    new[1:] = tl[1:] != tl[:-1]
    return lens + new, new


def build_window_tokens(V, ticks, offs, toks, start_tick, conditions, bpm_token,
                        prog_token, *, max_len=MAX_LEN, include_target=True,
                        reserve_target_tokens=0, ref=None, with_fut=False,
                        fut=None, fut_start=None):
    """Return int32 tokens/cur_time and uint8 loss_mask, or None if too long.

    Training includes target events and EOS. Inference stops at TGT and uses an
    all-zero loss mask. Its history is cropped at note boundaries to leave room
    for ``reserve_target_tokens`` target tokens and one EOS. Reserving the actual
    target length produces exactly the training prefix through TGT.

    ``ref`` (seq_version 3 only): (marker_token, event_tokens, event_cur_time) -> the header becomes
    BOS conds BPM PROG <marker> <ref events> CTX.  With ref=None the output is exactly the v1/v2 layout.

    ``with_fut`` (v6 infill): append ``V.FUT`` (cur_time = TGT) + the notes of the future 8 beats
    ``[fut_start, fut_start + FUT_TICKS)`` (time tokens T_0..T_383, cur_time = TGT + relative) + ``V.TGT``.
    ``fut`` = (ticks, offs, toks) arrays the future notes are read from (default: the same arrays); this lets the
    caller pass the UNcorrupted chart while ``ticks/offs/toks`` hold a corrupted history.
    """
    s = int(start_tick)
    a = int(np.searchsorted(ticks, s - HIST, "left"))
    m = int(np.searchsorted(ticks, s, "left"))
    if include_target:
        b = int(np.searchsorted(ticks, s + TGT, "left"))
        cost_t, _ = _costs(ticks, offs, m, b)
        tgt_len = int(cost_t.sum())
    else:
        tgt_len = max(0, int(reserve_target_tokens))
    hdr = len(conditions) + 4                 # BOS + conditions + BPM + PROG + CTX (== HEADER for the 8 v1 conditions)
    if ref is not None:
        hdr += 1 + len(ref[1])                # + marker + REF events
    f_ticks, f_offs, f_toks = (ticks, offs, toks) if fut is None else fut
    if fut_start is None:
        fut_start = s + TGT
    fut_start = int(fut_start)
    f0 = int(np.searchsorted(f_ticks, fut_start, "left"))
    f1 = int(np.searchsorted(f_ticks, fut_start + FUT_TICKS, "left")) if with_fut else f0
    cost_f = _costs(f_ticks, f_offs, f0, f1)[0] if with_fut else np.zeros(0, np.int64)
    cost_h, new_h = _costs(ticks, offs, a, m)
    nh = m - a
    # The target window is NEVER cropped; crop future notes from the farthest end first, then the oldest history notes.
    fixed = hdr + 1 + tgt_len + 1 + (1 if with_fut else 0)          # header + TGT + target + EOS (+ FUT marker)
    fut_cost = int(cost_f.sum())
    hist_cost = int(cost_h.sum())
    kf = len(cost_f)
    if with_fut:
        while kf > 0 and fixed + hist_cost + int(cost_f[:kf].sum()) > max_len:
            kf -= 1
        fut_cost = int(cost_f[:kf].sum())
    budget = max_len - fixed - fut_cost
    if budget < 0:
        return None
    j = 0
    if nh:
        if hist_cost <= budget:
            j = 0
        else:
            suffix = np.cumsum(cost_h[::-1])[::-1]
            ok = suffix <= budget
            j = int(np.argmax(ok)) if ok.any() else nh
            while j < nh and suffix[j] + (0 if new_h[j] else 1) > budget:
                j += 1
    if ref is None:
        T = [V.BOS] + conditions.tolist() + [bpm_token, prog_token, V.CTX]
        C = [T_MIN] * hdr
    else:
        T = [V.BOS] + conditions.tolist() + [bpm_token, prog_token, int(ref[0])] + [int(x) for x in ref[1]] + [V.CTX]
        C = [T_MIN] * (len(conditions) + 4) + [int(x) for x in ref[2]] + [T_MIN]
    _events(V, ticks, offs, toks, a + j, m, s, T, C)
    if with_fut:
        T.append(V.FUT)
        C.append(TGT)
        _events_fut(V, f_ticks, f_offs, f_toks, f0, f0 + kf, fut_start, TGT, T, C)
    tgt_pos = len(T)
    T.append(V.TGT)
    C.append(0)
    if include_target:
        _events(V, ticks, offs, toks, m, b, s, T, C)
        T.append(V.EOS)
        C.append(C[-1])
    n = len(T)
    mask = np.zeros(n, np.uint8)
    if include_target:
        mask[tgt_pos:n - 1] = 1
    return {"tokens": np.asarray(T, np.int32),
            "cur_time": np.asarray(C, np.int32), "loss_mask": mask}


def window_times(tempo, first, start_tick):
    """audio times (s) of the 1728 window ticks (-768 .. +959 relative to start_tick), no augmentation"""
    ticks = _TICKS0 + float(start_tick - HIST)
    return first + tempo.to_sec(ticks / TPB)


def window_mel(mel, tempo, first, start_tick, mel_mean, mel_std, *, rng=None,
               train=False, inv_std=None, neg_mean_std=None, out_times=None):
    """Return normalized float16 [1728,96] audio; invalid frames are -4.

    Cached ``inv_std`` and ``neg_mean_std`` keep the fused training fast path.
    Augmentation and early-return RNG consumption match the optimized loader.
    """
    ticks = _TICKS0 + float(start_tick - HIST)
    times = first + tempo.to_sec(ticks / TPB)
    if train and rng is not None:
        center = times.mean()
        times = center + (times - center) * (1.0 + rng.normal(0.0, 0.0015)) + rng.uniform(-0.012, 0.012)
    if out_times is not None:                    # pretrained-feature path reuses exactly the same (augmented) time grid
        out_times.append(times)
    T = mel.shape[0]
    if T < 2:
        return np.full((N_FRAMES, N_MEL), INVALID, np.float16)
    f = times * FPS
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)
    valid = (i0 >= 0) & (i0 + 1 < T)
    nvalid = int(valid.sum())
    if nvalid == 0:
        return np.full((N_FRAMES, N_MEL), INVALID, np.float16)
    np.clip(i0, 0, T - 2, out=i0)
    a = np.take(mel, i0, axis=0)
    i0 += 1
    b = np.take(mel, i0, axis=0)
    if inv_std is None:
        std = np.maximum(np.asarray(mel_std, np.float32), 1e-3).reshape(1, -1)
        inv = (1.0 / std).astype(np.float32)
    else:
        inv = inv_std
    wb = w[:, None] * inv
    wa = inv - wb
    if neg_mean_std is None:
        mean = np.asarray(mel_mean, np.float32).reshape(1, -1)
        cvec = (-mean * inv).astype(np.float32)
    else:
        cvec = neg_mean_std
    if train and rng is not None:
        cvec = cvec + (rng.uniform(-0.3, 0.3) * _BAND)[None, :]
    x = a * wa
    x += b * wb
    x += cvec
    if train and rng is not None and rng.random() < 0.3:
        wd = int(rng.integers(1, 9))
        st = int(rng.integers(0, N_MEL - wd + 1))
        x[:, st:st + wd] = 0.0
    if nvalid < N_FRAMES:
        x[~valid] = INVALID
    return x.astype(np.float16)
