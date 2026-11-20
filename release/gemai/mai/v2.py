"""seq_version 2 helpers: feature-amount / chart-density conditioning, history corruption, extra audio augmentation.

Nothing here is used by seq_version 1 (v1 checkpoints / v1 training are byte-for-byte unchanged).

Chart-level statistics (computed from the token arrays of charts.pkl by tools/make_v2_stats.py -> work/v2_stats.json and stored
in every v2 checkpoint as ckpt["seq_meta"]):
  touch amount = touch+touch-hold notes / notes      -> N (none) / LOW / MID / HIGH  (tertiles among Master+Re:Master train charts using it)
  ex    amount = notes with EX / notes               -> same
  fest  amount = number of multi-segment slide arms  -> same
  density      = notes per SECOND of the whole chart (notes / (audio duration - first))  -> DENS_0 .. DENS_{N_DENS-1}
                 (12 log-spaced buckets fitted on Master+Re:Master train charts, edges in seq_meta["dens_edges"])
"""
from __future__ import annotations
import json
import math
import numpy as np

from .tokens import TPB, N_DENS
from .simai import SENSORS

# bucket b = number of edges <= density (notes / s), 0 .. N_DENS-1.  Fallback only: the fitted edges live in seq_meta["dens_edges"]
DENS_EDGES = np.geomspace(2.5, 7.57, N_DENS - 1)
HIST_TICKS = 16 * TPB
TGT_TICKS = 16 * TPB
INVALID = -4.0
N_MEL = 96


# ----------------------------------------------------------------------------------------------
# chart statistics
def density_bucket(d, edges=DENS_EDGES):
    return int(np.searchsorted(np.asarray(edges), float(d), side="right"))


def density_center(b, edges=DENS_EDGES):
    """geometric centre (notes / beat) of a density bucket"""
    e = np.asarray(edges)
    lo = e[0] / (e[1] / e[0]) if b <= 0 else e[b - 1]
    hi = e[-1] * (e[-1] / e[-2]) if b >= len(e) else e[b]
    return float(math.sqrt(lo * hi))


def amount_bucket(x, thr):
    """0 none (x <= 0), 1 low (x <= thr[0]), 2 mid (x <= thr[1]), 3 high"""
    if x <= 0:
        return 0
    return 1 if x <= thr[0] else (2 if x <= thr[1] else 3)


def chart_amounts(V, ticks, offs, toks, seconds):
    """-> dict(nn, nt, ne, nf, dens): notes, touch(+hold) notes, EX notes, multi-segment slide arms, notes per second
    (seconds = audio duration - first offset, the "whole chart" length)"""
    toks = np.asarray(toks)
    offs = np.asarray(offs)
    nn = int(len(ticks))
    nt = int((toks == V["N_TOUCH"]).sum() + (toks == V["N_THOLD"]).sum())
    ne = int((toks == V["EX"]).sum())
    nf = 0
    ap = np.flatnonzero(toks == V["ARM"])
    if len(ap):
        sh = np.array([i for i, t in enumerate(V.toks) if t.startswith("SH_")])
        sp = np.flatnonzero(np.isin(toks, sh))
        nxt = np.append(ap[1:], len(toks))
        nend = offs[np.searchsorted(offs, ap, "right")]
        end = np.minimum(nxt, nend)
        cnt = np.searchsorted(sp, end) - np.searchsorted(sp, ap)
        nf = int((cnt > 1).sum())
    return dict(nn=nn, nt=nt, ne=ne, nf=nf, dens=nn / max(float(seconds), 1.0))


def annotate_chart(chart, amounts, meta):
    """add touch_b / ex_b / fest_b / dens_b (and lv_dens helper keys) to a charts.pkl dict, in place"""
    thr = meta["thr"]
    nn = max(amounts["nn"], 1)
    chart["touch_b"] = amount_bucket(amounts["nt"] / nn, thr["touch"])
    chart["ex_b"] = amount_bucket(amounts["ne"] / nn, thr["ex"])
    chart["fest_b"] = amount_bucket(amounts["nf"], thr["fest"])
    chart["dens_b"] = density_bucket(amounts["dens"], meta.get("dens_edges", DENS_EDGES))
    return chart


def target_density(level, meta):
    """typical notes / second of a Master/Re:Master chart of this level: median per 0.5 level (meta["level_bucket"]),
    linearly interpolated in log space between the level nodes (clamped at both ends)."""
    t = meta["level_bucket"]
    lv = np.asarray(t["levels"], np.float64)
    return float(math.exp(np.interp(float(level), lv, np.log(np.asarray(t["median_dens"], np.float64)))))


def density_bucket_for_level(level, meta, shift=0.0):
    """DENS bucket the inference code should request for `level` (shift = extra buckets, positive = denser)."""
    b = density_bucket(target_density(level, meta), meta.get("dens_edges", DENS_EDGES)) + int(round(shift))
    return int(min(max(b, 0), N_DENS - 1))


def load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seq_meta_of(stats):
    """the part of v2_stats.json that is stored in checkpoints (everything except the per-chart table)"""
    return {k: v for k, v in stats.items() if k != "charts"}


# ----------------------------------------------------------------------------------------------
# history corruption (training only)
_SH_CACHE = {}


def _sh_ids(V):
    r = _SH_CACHE.get(id(V))
    if r is None:
        r = _SH_CACHE[id(V)] = frozenset(i for i, t in enumerate(V.toks) if t.startswith("SH_"))
    return r


def _multi_segment(V, body):
    sh, arm, n = _sh_ids(V), V["ARM"], 0
    for t in body:
        if t == arm:
            n = 0
        elif t in sh:
            n += 1
            if n > 1:
                return True
    return False


def corrupt_window_arrays(V, ticks, offs, toks, s, rng, *, p_drop_all=0.12, p_thin=0.22, p_strip=0.16, p_note=0.10,
                          p_move=0.5, thin_range=(0.25, 0.7)):
    """Return (ticks, offs, toks) restricted to [s-768, s+768) whose HISTORY part is corrupted (the target part is untouched).
    The history normally gives away the chart's density / feature use, which makes the chart-level conditions (level, density,
    touch / ex / fest amount) redundant in teacher forcing and the model sticky to its own recent output; one mode per window:
      p_drop_all  the whole history is removed
      p_thin      density thinning: each history note is kept with probability r ~ U(thin_range)
      p_strip     feature stripping: remove the history's touch notes / EX flags / multi-segment slides (each with p .6 / .5 / .5,
                  at least one)
      otherwise   light noise: every note with probability p_note is dropped (or, with p_move, moved to a random button / sensor)
    Returns None when the window has no history (nothing to corrupt)."""
    a = int(np.searchsorted(ticks, s - HIST_TICKS, "left"))
    m = int(np.searchsorted(ticks, s, "left"))
    b = int(np.searchsorted(ticks, s + TGT_TICKS, "left"))
    if m == a:
        return None
    nh = m - a
    n_touch, n_thold, ex_id = V["N_TOUCH"], V["N_THOLD"], V["EX"]
    btn0 = V["BTN_1"]
    sen0 = V["SEN_" + SENSORS[0]]
    u = rng.random()
    keep = np.ones(nh, bool)
    move = np.zeros(nh, bool)
    strip = None
    if u < p_drop_all:
        keep[:] = False
    elif u < p_drop_all + p_thin:
        keep = rng.random(nh) < rng.uniform(*thin_range)
    elif u < p_drop_all + p_thin + p_strip:
        strip = rng.random(3) < (0.6, 0.5, 0.5)
        if not strip.any():
            strip[int(rng.integers(3))] = True
    else:
        hit = rng.random(nh) < p_note
        mv = rng.random(nh) < p_move
        keep = ~hit | mv
        move = hit & mv
    bodies, kept_idx = [], []
    for j in np.flatnonzero(keep):
        i = a + int(j)
        body = toks[offs[i]:offs[i + 1]]
        if strip is not None:
            if strip[0] and body[0] in (n_touch, n_thold):
                continue
            if strip[2] and _multi_segment(V, body):
                continue
            if strip[1] and ex_id in body:
                body = body[body != ex_id]
        if move[j] and len(body) > 1:
            body = body.copy()
            if body[0] in (n_touch, n_thold):
                body[1] = sen0 + int(rng.integers(len(SENSORS)))
            else:
                body[1] = btn0 + int(rng.integers(8))
        bodies.append(body)
        kept_idx.append(i)
    h_len = np.array([len(x) for x in bodies], np.int64)
    h_tot = int(h_len.sum())
    h_toks = np.concatenate(bodies).astype(toks.dtype) if bodies else toks[:0]
    t_toks = toks[offs[m]:offs[b]]
    new_toks = np.concatenate([h_toks, t_toks])
    new_offs = np.concatenate([[0], np.cumsum(h_len), (offs[m + 1:b + 1] - offs[m]) + h_tot]).astype(offs.dtype)
    new_ticks = np.concatenate([ticks[kept_idx] if kept_idx else ticks[:0], ticks[m:b]]).astype(ticks.dtype)
    return new_ticks, new_offs, new_toks


# ----------------------------------------------------------------------------------------------
# extra audio augmentation (v2 only; applied after mai.sequence.window_mel)
_BAND = np.arange(N_MEL, dtype=np.float32)


def augment_mel_v2(x, rng):
    """x: float16 [1728, 96] normalized window (invalid frames == -4).  In-place-ish; returns float16.
    per-window gain offset N(0, .2); gaussian noise (p .5, std .03-.12); up to 2 extra band masks (p .4 each, width <= 10);
    one short time mask (p .25, 4-14 frames = up to 0.3 beat) set to 0."""
    y = x.astype(np.float32)
    valid = (y[:, 0] != INVALID) | (y[:, N_MEL // 2] != INVALID)
    y[valid] += np.float32(rng.normal(0.0, 0.2))
    if rng.random() < 0.5:
        y[valid] += rng.normal(0.0, rng.uniform(0.03, 0.12), (int(valid.sum()), N_MEL)).astype(np.float32)
    for _ in range(2):
        if rng.random() < 0.4:
            wd = int(rng.integers(1, 11))
            st = int(rng.integers(0, N_MEL - wd + 1))
            y[valid, st:st + wd] = 0.0
    if rng.random() < 0.25:
        wd = int(rng.integers(4, 15))
        st = int(rng.integers(0, y.shape[0] - wd + 1))
        seg = valid[st:st + wd]
        y[st:st + wd][seg] = 0.0
    return y.astype(np.float16)


# ----------------------------------------------------------------------------------------------
# inference helper: user options -> v2 condition row
def v2_condition_row(V, meta, *, level=None, slot=5, chart_type="auto", touch="auto", ex="auto", fest="auto",
                     des_id=-2, versionid=-1, dens=None, dens_shift=0, plan=None, sec=None, secn=None):
    """Condition row (int32[9]) for generation with a seq_version 2 checkpoint.

    level       float 1.0..15.0 (x.7 style decimal level) or None -> LV_UNK.  The model also gets the smooth level embedding
                (inside the model, automatically) so any 0.1 step is a valid, ordered level.
    slot        5 = Master, 6 = Re:Master.   chart_type "std" | "dx" | "auto"(TYPE_UNK)
    touch/ex/fest   "auto" (UNK: the model decides) | "off" (none) | "on" (present, amount unspecified = *_Y)
                    | "low" | "mid" | "high" (amount tertiles among Master charts that use the feature)
    des_id      designer id (index into the designers list) | -1 other | -2 unknown (DES_UNK)
    plan        (seq_version 3 only) PLAN bucket 0..8 of the window, mai.v3.plan_from_pred(...)[k]; None -> PLAN_UNK
    sec, secn   (seq_version 4 only) section label id 0..9 of the window's first bar / 1 when a new section starts inside the window
                (mai.ann.window_tokens(...)); None -> SEC_UNK / SECN_UNK (no annotation)
    dens        None -> DENS_UNK (level conditioning only) | "auto" -> typical density bucket of `level` (median per 0.5 level
                on Master/Re:Master, seq_meta["level_bucket"]) | int bucket | float notes per SECOND of the whole chart.
                dens_shift moves the bucket (positive = denser).
    """
    from .sequence import condition_tokens, feature_token, density_token
    chart = dict(lv=level, slot=slot, dx=(chart_type == "dx"), touch=False, ex=False, fest=False,
                 des_id=des_id, versionid=versionid)
    row = condition_tokens(V, chart)
    if chart_type == "auto":
        row[2] = V["TYPE_UNK"]
    for i, (name, st) in enumerate((("TOUCH", touch), ("EXN", ex), ("FEST", fest)), 3):
        row[i] = feature_token(V, name, st)
    if dens is None:
        b = None
    elif dens == "auto":
        b = None if level is None else density_bucket_for_level(level, meta, dens_shift)
    elif isinstance(dens, float):
        b = min(max(density_bucket(dens, meta.get("dens_edges", DENS_EDGES)) + int(dens_shift), 0), N_DENS - 1)
    else:
        b = min(max(int(dens) + int(dens_shift), 0), N_DENS - 1)
    row[8] = density_token(V, b)
    if getattr(V, "seq_version", 2) >= 3:             # seq_version 3: 10th condition = planned density of the window (None -> PLAN_UNK)
        from .sequence import plan_token
        row[9] = plan_token(V, plan)
    if getattr(V, "seq_version", 2) >= 4:             # seq_version 4: 11th/12th condition = section label / new-section flag of the window
        from .sequence import sec_token, secn_token
        row[10] = sec_token(V, sec)
        row[11] = secn_token(V, secn)
    return row
