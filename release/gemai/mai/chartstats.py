"""Feature extraction for maimai charts (canonical tick notes + NpTempo) and distribution distances.

extract(notes, tempo) -> flat dict: scalar features are python floats, histogram features are np.ndarray with
keys starting with 'h_'.  Helpers at the bottom compare feature sets (Jensen-Shannon distance of histograms,
standardised mean shifts / 1-D Wasserstein distance of scalars).
"""
from __future__ import annotations
import math
import numpy as np
from .simai import TPB, SHAPES
from . import playcheck

BEAT4 = 12   # 1/4 beat in ticks (a 16th note)
HOLD_BINS = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9])          # beats
SPEED_BINS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 1e9])  # ring steps per second
RHYTHM_NAMES = ["1/1", "1/2", "1/4", "1/3", "1/6", "1/8", "1/12", "1/16", "other"]
STEP = 2 * playcheck.R_A * math.sin(math.radians(22.5))  # distance between adjacent buttons

def _rhythm_class(r):
    r %= TPB
    if r == 0: return 0
    if r == 24: return 1
    if r in (12, 36): return 2
    if r in (16, 32): return 3
    if r in (8, 40): return 4
    if r in (6, 18, 30, 42): return 5
    if r in (4, 20, 28, 44): return 6
    if r % 3 == 0: return 7      # odd multiples of 3 ticks (1/16 of a beat)
    return 8

_RH = np.array([_rhythm_class(r) for r in range(TPB)], np.int8)

def ring_dist(a, b):
    d = abs(a - b) % 8
    return min(d, 8 - d)

def _hist(vals, bins):
    h, _ = np.histogram(vals, bins=bins)
    return h.astype(np.float64)

def _window_counts(times, win, step=0.25):
    """counts of events per window of `win` seconds sliding by `step` over the active span"""
    if len(times) == 0:
        return np.zeros(1)
    t0, t1 = times[0], times[-1]
    if t1 - t0 <= win:
        return np.array([len(times)], dtype=np.float64)
    starts = np.arange(t0, t1 - win + 1e-9, step)
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, starts + win, side="left")
    return (hi - lo).astype(np.float64)

def _window_sums(times, weights, win, step=0.25):
    if len(times) == 0:
        return np.zeros(1)
    t0, t1 = times[0], times[-1]
    cw = np.concatenate([[0.0], np.cumsum(weights)])
    if t1 - t0 <= win:
        return np.array([cw[-1]])
    starts = np.arange(t0, t1 - win + 1e-9, step)
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, starts + win, side="left")
    return cw[hi] - cw[lo]

NOTE_W = {"tap": 1.0, "hold": 1.2, "slide": 1.0, "touch": 0.6, "thold": 0.8}

def _stat3(prefix, w, win):
    r = w / win
    return {prefix + "_mean": float(r.mean()), prefix + "_p90": float(np.percentile(r, 90)), prefix + "_max": float(r.max())}

def extract(notes, tempo, with_playcheck=True, pc=None):
    """feature dict for one chart. `pc` may be a precomputed playcheck.check result (needs pairs, see below)."""
    f = {}
    n = len(notes)
    f["n_notes"] = float(n)
    if n == 0:
        return f
    kinds = [x.kind for x in notes]
    cnt = {k: kinds.count(k) for k in ("tap", "hold", "slide", "touch", "thold")}
    for k, v in cnt.items():
        f["n_" + k] = float(v); f["r_" + k] = v / n
    n_arms = sum(len(x.arms) for x in notes if x.kind == "slide")
    f["n_arms"] = float(n_arms)
    f["n_brk"] = float(sum(1 for x in notes if x.brk or any(a.brk for a in x.arms)))
    f["n_ex"] = float(sum(1 for x in notes if x.ex or any(a.ex for a in x.arms)))
    f["n_fw"] = float(sum(1 for x in notes if x.fw))
    f["n_headless"] = float(sum(1 for x in notes if x.headless))
    for k in ("brk", "ex", "fw", "headless"):
        f["r_" + k] = f["n_" + k] / n
    f["r_multiarm"] = float(sum(1 for x in notes if x.kind == "slide" and len(x.arms) > 1)) / max(1, cnt["slide"])

    ticks = np.array([int(x.beat) for x in notes], np.int64)
    ends = np.array([int(x.end_beat()) for x in notes], np.int64)
    ts = tempo.to_sec(ticks / TPB)
    te = tempo.to_sec(ends / TPB)
    order = np.argsort(ticks, kind="stable")
    ts_sorted = ts[order]
    t_first, t_last = float(ts_sorted[0]), float(te.max())
    f["t_first"] = t_first
    f["dur_sec"] = float(ts_sorted[-1] - t_first)      # first to last onset
    f["len_sec"] = t_last - t_first                      # first onset to last release
    f["notes_per_sec"] = n / max(1.0, f["dur_sec"])
    f["log_n_notes"] = math.log(n)

    # --- nps over sliding windows (2 s), plus 1 s / 4 s maxima
    w2 = _window_counts(ts_sorted, 2.0)
    f.update(_stat3("nps", w2, 2.0))
    f["nps1_max"] = float(_window_counts(ts_sorted, 1.0).max())
    w4 = _window_counts(ts_sorted, 4.0); f["nps4_max"] = float(w4.max() / 4.0)
    f["nps4_p90"] = float(np.percentile(w4, 90) / 4.0)
    btn_t = np.sort(np.array([ts[i] for i in range(n) if kinds[i] in ("tap", "hold", "slide")]))
    if len(btn_t):
        f.update(_stat3("nps_btn", _window_counts(btn_t, 2.0), 2.0))
    else:
        f.update({"nps_btn_mean": 0.0, "nps_btn_p90": 0.0, "nps_btn_max": 0.0})
    for k in ("slide", "touch"):
        tt = np.sort(np.array([ts[i] for i in range(n) if kinds[i] in ((k,) if k == "slide" else ("touch", "thold"))]))
        f[k + "_nps_max"] = float(_window_counts(tt, 2.0).max() / 2.0) if len(tt) else 0.0

    # weighted density: slide arms add weight, touches count less
    wts = np.array([NOTE_W[k] + (0.6 * len(x.arms) if k == "slide" else 0.0) for k, x in zip(kinds, notes)])
    ws = _window_sums(ts_sorted, wts[order], 2.0)
    f.update(_stat3("wnps", ws, 2.0))
    top = np.sort(ws)[::-1]
    f["wnps_top5"] = float(top[:max(1, len(top) // 20)].mean() / 2.0)
    f["wnps_top20"] = float(top[:max(1, len(top) // 5)].mean() / 2.0)

    # --- onsets, simultaneity
    onsets = {}
    for x in notes:
        onsets.setdefault(int(x.beat), []).append(x)
    otick = np.array(sorted(onsets), np.int64)
    osec = tempo.to_sec(otick / TPB)
    f["n_onsets"] = float(len(otick))
    gaps = np.diff(osec) if len(osec) > 1 else np.array([1.0])
    f["gap_median"] = float(np.median(gaps)); f["gap_p10"] = float(np.percentile(gaps, 10))
    for th in (0.08, 0.125, 0.17, 0.25):
        f["frac_gap_lt_%d" % int(th * 1000)] = float((gaps < th).mean())
    btn_sets = []   # (tick, sorted tuple of button positions) for onsets that need a finger on a button
    simul_hist = np.zeros(5)
    n_each = 0
    for t in otick:
        g = onsets[int(t)]
        bs = sorted(x.pos for x in g if x.kind in ("tap", "hold") or (x.kind == "slide" and not x.headless))
        if bs:
            btn_sets.append((int(t), bs))
        if len(bs) >= 2:
            n_each += 1
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    simul_hist[ring_dist(bs[i], bs[j])] += 1
    f["each_rate"] = n_each / max(1, len(btn_sets))
    f["simul_note_rate"] = float(sum(len(g) for g in onsets.values() if len(g) >= 2)) / n
    f["h_simul_dist"] = simul_hist
    f["max_simul"] = float(max(len(g) for g in onsets.values()))

    # --- consecutive-contact distance for fast (<= 1/4 beat) neighbours, jacks
    cons = np.zeros(5); n_jack = 0; n_fast = 0
    for i in range(1, len(btn_sets)):
        (t0, a), (t1, b) = btn_sets[i - 1], btn_sets[i]
        if t1 - t0 <= BEAT4:
            d = min(ring_dist(p, q) for p in a for q in b)
            cons[d] += 1; n_fast += 1
            if d == 0: n_jack += 1
    f["h_cons_dist"] = cons
    f["n_fast_pairs"] = float(n_fast)
    f["fast_pair_rate"] = n_fast / max(1, len(btn_sets))
    f["jack_rate"] = n_jack / max(1, len(btn_sets))
    f["jack_per_sec"] = n_jack / max(1.0, f["dur_sec"])
    f["far_fast_rate"] = float(cons[3:].sum()) / max(1, len(btn_sets))   # distance 3-4 within a 16th

    # --- slides
    shp = np.zeros(len(SHAPES)); speeds = []; durs_s = []; nseg = []
    for x in notes:
        if x.kind != "slide": continue
        for a in x.arms:
            for sg in a.segs:
                shp[SHAPES.index(sg.shape)] += 1
            nseg.append(len(a.segs))
            ds = float(tempo.to_sec((int(x.beat) + int(a.wait) + int(a.dur)) / TPB) - tempo.to_sec((int(x.beat) + int(a.wait)) / TPB))
            durs_s.append(ds)
            pts = playcheck.arm_waypoints(x.pos, a.segs)
            L = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) / STEP
            speeds.append(L / max(ds, 1e-3))
    f["h_slide_shape"] = shp
    if speeds:
        sp = np.array(speeds)
        f["slide_speed_mean"] = float(sp.mean()); f["slide_speed_p90"] = float(np.percentile(sp, 90)); f["slide_speed_max"] = float(sp.max())
        f["fest_rate"] = float(np.mean(np.array(nseg) > 1))
        f["slide_seg_mean"] = float(np.mean(nseg))
        f["slide_dur_mean_s"] = float(np.mean(durs_s))
    else:
        for k in ("slide_speed_mean", "slide_speed_p90", "slide_speed_max", "fest_rate", "slide_seg_mean", "slide_dur_mean_s"):
            f[k] = 0.0
    f["h_slide_speed"] = _hist(speeds, SPEED_BINS)

    # --- hold / slide durations (beats)
    hd = [x.dur / TPB for x in notes if x.kind in ("hold", "thold")]
    sd = [a.dur / TPB for x in notes if x.kind == "slide" for a in x.arms]
    f["h_hold_dur"] = _hist(hd, HOLD_BINS)
    f["h_slide_dur"] = _hist(sd, HOLD_BINS)
    f["hold_dur_mean_beats"] = float(np.mean(hd)) if hd else 0.0
    # fraction of chart time covered by at least one sustained contact
    f["sustain_cover"] = 0.0
    sus = sorted([(float(ts[i]), float(te[i])) for i in range(n) if kinds[i] in ("hold", "thold")]
                 + [(float(tempo.to_sec((int(x.beat) + int(a.wait)) / TPB)), float(tempo.to_sec((int(x.beat) + int(a.wait) + int(a.dur)) / TPB)))
                    for x in notes if x.kind == "slide" for a in x.arms])
    if sus:
        cov = 0.0; cs, ce = sus[0]
        for s, e in sus[1:]:
            if s > ce: cov += ce - cs; cs, ce = s, e
            else: ce = max(ce, e)
        cov += ce - cs
        f["sustain_cover"] = cov / max(1.0, f["len_sec"])

    # --- rhythm grid
    f["h_rhythm"] = np.bincount(_RH[ticks % TPB], minlength=9).astype(np.float64)
    rh = f["h_rhythm"] / n
    for i, nm in enumerate(RHYTHM_NAMES):
        f["rh_" + nm.replace("/", "_")] = float(rh[i])

    # --- bpm
    bpms = np.asarray(tempo.p); bt = np.asarray(tempo.b)
    seg_end = np.append(bt[1:], max(float(ends.max()) / TPB, bt[-1] + 1e-6))
    seg_sec = (seg_end - bt) * 60.0 / bpms
    f["bpm_mean"] = float((bpms * seg_sec).sum() / seg_sec.sum())
    f["bpm_main"] = float(bpms[int(np.argmax(seg_sec))])
    f["bpm_min"] = float(bpms.min()); f["bpm_max"] = float(bpms.max()); f["n_bpm_changes"] = float(len(bpms) - 1)
    # beat-density: notes per beat and "16th-rate": notes per sec normalised by bpm_main
    f["notes_per_beat"] = n / max(1.0, (ticks.max() - ticks.min()) / TPB)

    # --- hand demand (playcheck)
    if with_playcheck:
        r = pc if pc is not None else playcheck.check(notes, tempo)
        ne = max(1, r["n_events"])
        f["pc_viol"] = float(r["n_viol"]); f["pc_viol_rate"] = r["n_viol"] / ne
        f["pc_hands_viol"] = float(r["n_hands_viol"]); f["pc_jump_viol"] = float(r["n_jump_viol"])
        f["pc_anchored_frac"] = float(r["anchored"].mean())
        f["pc_two_hand_frac"] = float((r["hands_needed"] >= 2).mean())
        f["pc_nactive_mean"] = float(r["n_active"].mean()); f["pc_nactive_max"] = float(r["n_active"].max())
        f["pc_sus2_frac"] = float((r["n_sus"] >= 2).mean())
        f["pc_forced_rate"] = r["n_forced"] / ne
        f["pc_dyn_edge_rate"] = r["n_dyn_edges"] / ne
        f["pc_static_edge_rate"] = r["n_static_edges"] / ne
        fh = r["fh"]
        if len(fh):
            dt, dd = fh[:, 0], fh[:, 1]
            f["pc_fh_pairs"] = float(len(fh)); f["pc_fh_dt_p10"] = float(np.percentile(dt, 10))
            f["pc_fh_fast"] = float((dt < 0.14).sum())
            f["pc_fh_fast_far"] = float(((dt < 0.2) & (dd > 1.0)).sum())
            f["pc_fh_speed_p90"] = float(np.percentile(dd / np.maximum(dt, 1e-3), 90))
        else:
            f["pc_fh_pairs"] = 0.0; f["pc_fh_fast"] = 0.0; f["pc_fh_fast_far"] = 0.0
            f["pc_fh_dt_p10"] = 5.0; f["pc_fh_speed_p90"] = 0.0
    return f

def scalar_keys(f):
    return [k for k, v in f.items() if not isinstance(v, np.ndarray)]

def hist_keys(f):
    return [k for k, v in f.items() if isinstance(v, np.ndarray)]

SCALAR_KEYS = None

# ------------------------------------------------------------------------------------ comparisons
def js_distance(p, q, eps=1e-12):
    """Jensen-Shannon distance (sqrt of JS divergence, base 2; in [0,1]) between two non-negative arrays"""
    p = np.asarray(p, np.float64); q = np.asarray(q, np.float64)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 and qs <= 0:
        return 0.0
    if ps <= 0 or qs <= 0:
        return 1.0
    p = p / ps; q = q / qs
    m = 0.5 * (p + q)
    def kl(a, b):
        mk = a > 0
        return float((a[mk] * np.log2(a[mk] / np.maximum(b[mk], eps))).sum())
    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))

def aggregate_hists(feats):
    """sum histograms over a list of feature dicts -> dict key -> array"""
    out = {}
    for f in feats:
        for k, v in f.items():
            if isinstance(v, np.ndarray):
                out[k] = out.get(k, 0) + v
    return out

def hist_distances(ref_feats, cand_feats):
    """JS distance per histogram between the aggregated histograms of two chart sets"""
    A, B = aggregate_hists(ref_feats), aggregate_hists(cand_feats)
    keys = list(A) + [k for k in B if k not in A]
    out = {}
    for k in keys:
        a = A.get(k)
        b = B.get(k)
        if a is None:
            a = np.zeros_like(b)
        if b is None:
            b = np.zeros_like(a)
        out[k] = js_distance(a, b)
    return out

def scalar_shifts(ref_feats, cand_feats, keys=None):
    """per scalar feature: standardised mean shift (cand - ref)/std_ref and normalised 1-D Wasserstein distance"""
    if not ref_feats or not cand_feats:
        return {}
    if keys is None:
        keys = list(dict.fromkeys(k for f in ref_feats for k in scalar_keys(f)))
    out = {}
    for k in keys:
        # Empty charts contribute zero note-derived features; missing fields of
        # nonempty/foreign feature dictionaries remain undefined.
        a = np.array([f.get(k, 0.0) for f in ref_feats if k in f or f.get("n_notes") == 0], np.float64)
        b = np.array([f.get(k, 0.0) for f in cand_feats if k in f or f.get("n_notes") == 0], np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 1:
            continue
        sd = a.std() + 1e-9
        q = np.linspace(0, 1, 101)
        w = float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))) / sd)
        out[k] = dict(shift=float((b.mean() - a.mean()) / sd), wasserstein=w, ref_mean=float(a.mean()), cand_mean=float(b.mean()))
    return out

def feature_distance(ref_feats, cand_feats, keys=None):
    """one-number summaries: mean JS over histograms, mean |standardised shift| and mean Wasserstein over `keys`"""
    hd = hist_distances(ref_feats, cand_feats)
    sh = scalar_shifts(ref_feats, cand_feats, keys)
    return dict(hist_js=hd, mean_hist_js=float(np.mean(list(hd.values()))) if hd else 0.0,
                scalars=sh,
                mean_abs_shift=float(np.mean([abs(v["shift"]) for v in sh.values()])) if sh else 0.0,
                mean_wasserstein=float(np.mean([v["wasserstein"] for v in sh.values()])) if sh else 0.0)

# scalar features that matter most when comparing generated vs reference charts
KEY_SCALARS = ["nps_mean", "nps_p90", "nps_max", "r_tap", "r_hold", "r_slide", "r_touch", "r_thold", "r_brk", "r_ex", "r_headless",
               "each_rate", "jack_rate", "far_fast_rate", "fest_rate", "slide_speed_mean", "rh_1_1", "rh_1_2", "rh_1_4", "rh_1_3",
               "rh_1_6", "rh_1_8", "rh_1_12", "rh_1_16", "rh_other", "sustain_cover", "frac_gap_lt_125", "pc_anchored_frac",
               "pc_two_hand_frac", "pc_viol_rate"]
