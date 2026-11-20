"""Quality mode: spend inference time (no retraining) for better charts.

fast   = the shipped generator (K = 6 candidates per 4-bar window, rule rerank, muri gate).
fine   = K = 16 candidates per window + one-window lookahead + one self-refine pass.
max    = fine + two self-refine passes + best of two whole charts (by the chart cost below).

Lookahead (inside the window loop, after the normal candidate score)
  The top ``la_top`` clean candidates each get ``la_samples`` samples of the NEXT window, decoded conditioned on that candidate
  (full stack: history, REF/PLAN/song context of the next window, steering state, hand state; all M x S samples run in one batched
  decode with per-row cache positions).  Continuation value of a candidate = mean over its samples of
      score of the continuation window (activity / repeat / density / muri penalties of the usual scorer, with the muri gate forked
      from the state that contains the candidate)  -  w_hand * hand-travel excess  +  la_lp * mean log-prob of the sample
      -  penalties for a stuck decode (fallback / resample / many backtracks).
  The committed candidate maximises  score + la_weight * value.

Self-refine (after the whole chart)
  Every window gets a cost (rule penalties of the scorer, muri events of the whole-chart dx pass at full price, hand travel,
  fallback flags, model log-prob below the song average).  The worst ``refine_frac`` windows (in time order) are regenerated: K
  candidates conditioned on the preceding generated notes (state rebuilt by replaying the earlier windows); every candidate is scored
  as  rules(window + next ``refine_after`` windows)  +  muri(region)  +  hand(region)  -  refine_lp * sum log P(following windows | candidate)
  where the following windows are the existing next window and every later window whose REF block overlaps the regenerated one
  (teacher-forced under the CFG-mixed, tempered model distribution, REF rebuilt from the modified chart).  A candidate is accepted only
  when playcheck / overlap checks stay clean and its total cost is lower than the existing window's by ``refine_margin``.
  REF consistency: later windows are not regenerated; windows that retrieved the changed bars as their REF segment are part of the
  bidirectional log-prob term, so a replacement that makes their (fixed) content unlikely is not accepted.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import handtravel, murigate, playcheck, v3 as V3
from .model import DecodeCache
from .sequence import bpm_progress_tokens, pack_notes
from .tokens import T_MAX, T_MIN, TPB, canonicalize, decode_events

BAR = 4 * TPB
HAND_FREE = 0.10

PRESETS = {
    "fast": dict(),
    "fine": dict(window_candidates=16, la_top=4, la_samples=2, w_hand=2.0, refine_passes=1),
    "max": dict(window_candidates=16, la_top=4, la_samples=2, w_hand=2.0, refine_passes=2, candidates=2),
}


def _gen():
    from . import generate
    return generate


def resolve_quality(options):
    """Concrete values for the auto (-1) fields of the chosen preset; explicit user values win."""
    if options.quality not in PRESETS:
        raise ValueError("quality must be fast, fine or max")
    p = PRESETS[options.quality]
    upd = {}
    for k, default in (("la_top", 0), ("la_samples", 0), ("refine_passes", 0)):
        if getattr(options, k) < 0:
            upd[k] = p.get(k, default)
    if options.w_hand < 0:
        upd["w_hand"] = (1.0 if getattr(options, "hand_mode", "target") == "target" else p.get("w_hand", 0.0)) if options.quality != "fast" else 0.0
    if options.quality != "fast":
        upd["window_candidates"] = max(options.window_candidates, p["window_candidates"])
        upd["candidates"] = max(options.candidates, p.get("candidates", 1))
    return replace(options, **upd)


# ---------------------------------------------------------------------------------------------- state forks
def fork_scorer(sc):
    o = copy.copy(sc)
    o.sigs, o.shapes, o.rks = list(sc.sigs), list(sc.shapes), list(sc.rks)
    o.seen = Counter(sc.seen)
    return o


def fork_gate(g):
    o = copy.copy(g)
    o.hist = list(g.hist)
    o.used = Counter(g.used)
    return o


def fork_steer(s):
    o = copy.copy(s)
    o.features = copy.copy(s.features)
    o.tilt = copy.copy(s.tilt)
    o.knobs = {k: copy.copy(v) for k, v in s.knobs.items()}
    o.rhythm_knobs = {k: copy.copy(v) for k, v in s.rhythm_knobs.items()}
    o.cont_knob = copy.copy(s.cont_knob)
    o.shape_knobs = {k: copy.copy(v) for k, v in getattr(s, "shape_knobs", {}).items()}
    o.shape_vec = np.array(getattr(s, "shape_vec", np.zeros(0)))
    o.window_expected = dict(s.window_expected)
    o.log = list(s.log)
    o.counts = Counter(s.counts)
    o._sigc = dict(s._sigc)
    o._mcache = {}
    o.prior_bars = dict(s.prior_bars)
    return o


# ---------------------------------------------------------------------------------------------- batched decode of several prefixes
def concat_caches(caches):
    L = len(caches[0].k)
    cat = lambda ts: torch.cat(ts, 0)
    out = DecodeCache([cat([c.k[l] for c in caches]) for l in range(L)], [cat([c.v[l] for c in caches]) for l in range(L)],
                      [cat([c.ck[l] for c in caches]) for l in range(L)], [cat([c.cv[l] for c in caches]) for l in range(L)])
    out.len = max(c.len for c in caches)
    if caches[0].g is not None:
        out.g = cat([c.g for c in caches])
    if getattr(caches[0], "sk", None) is not None:
        out.sk = [None if caches[0].sk[l] is None else cat([c.sk[l] for c in caches]) for l in range(L)]
        out.sv = [None if caches[0].sv[l] is None else cat([c.sv[l] for c in caches]) for l in range(L)]
        out.smask = cat([c.smask for c in caches])
        out.skeep = cat([c.skeep for c in caches])
    return out


def begin_multi(backend, enc, groups, song):
    """Prefill every group's R prefix rows separately (different prefix lengths), tile each K_g times, concatenate along the batch."""
    G = _gen()
    model, device = backend.model, enc.device
    R = len(groups[0]["prefixes"])
    caches, logits0s, plens = [], [], []
    for grp in groups:
        pre = grp["prefixes"]
        tok = torch.as_tensor(np.stack([p["tokens"] for p in pre]), dtype=torch.long, device=device)
        ct = torch.as_tensor(np.stack([p["cur_time"] for p in pre]), dtype=torch.long, device=device)
        with backend.ac():
            cache = model.init_cache(enc.expand(R, -1, -1), song=song)
            logits0 = model.prefill(tok, ct, cache).float().cpu().numpy()
        K = len(grp["guards"])
        plens.append(cache.len)
        logits0s.append(logits0)
        caches.append(G.replicate_cache(cache, K) if K > 1 else cache)
    big = concat_caches(caches) if len(caches) > 1 else caches[0]
    k_tot = sum(len(g["guards"]) for g in groups)
    return G._Window(backend, device, None, max(plens), k_tot, R, cache=big), logits0s, plens


def decode_groups_co(model, enc, groups, options, V, max_tick, *, backend, song=None):
    """``decode_candidates_co`` for several groups that differ in their prefix (history), hand state and steering.
    groups: list of dict(prefixes=[R dicts], guards=[K_g HandGuard], bias=callable|None, rng_factory=callable(j, attempt)).
    Returns one ``[(tokens, log_probs, stats)]`` list per group.  All candidates share the audio window and the song memory."""
    G = _gen()
    R = len(groups[0]["prefixes"])
    plan = G.guidance_plan(options)
    n_designers = G.designer_rows(options, R)
    window, logits0s, plens = begin_multi(backend, enc, groups, song)
    try:
        cands = []
        for g, grp in enumerate(groups):
            limit = min(options.max_target_tokens + 1, model.cfg.max_len - plens[g])
            off = len(cands)
            for j, guard in enumerate(grp["guards"]):
                rf = (lambda index, attempt, _rf=grp["rng_factory"], _off=off: _rf(index - _off, attempt))
                c = G._Candidate(len(cands), guard, rf, V, options, max_tick, limit)
                c.logits0 = c.logits = logits0s[g]
                c.bias, c.group, c.plen = grp.get("bias"), g, plens[g]
                cands.append(c)
        K = len(cands)
        pos = [c.plen for c in cands]
        deadline = time.monotonic() + options.window_seconds
        max_iterations = 6 * max(c.limit for c in cands) + 8 * options.max_backtracks + 200
        pad = V.PAD
        timing = G.timing_mask(V)
        while True:
            feeds = {}
            timed_out = time.monotonic() > deadline
            for c in cands:
                if c.finished:
                    continue
                token = G._advance(c, V, options, plan, n_designers, c.bias, timing, timed_out, max_iterations)
                if c.finished:
                    continue
                pos[c.index] = c.plen + len(c.tokens) - (1 if token is not None else 0)
                if token is not None:
                    feeds[c.index] = token
            live = [c for c in cands if not c.finished]
            if not live:
                break
            if not feeds:
                continue
            positions = {pos[k] for k in feeds}
            tok_row = np.full(K * R, pad, np.int64)
            ct_row = np.zeros(K * R, np.int64)
            pos_row = np.empty(K * R, np.int64)
            for c in cands:
                base = c.index * R
                pos_row[base:base + R] = pos[c.index]
                if c.index in feeds:
                    tok_row[base:base + R] = feeds[c.index]
                    ct_row[base:base + R] = c.grammar.cur_time
            out = yield (window, tok_row, ct_row, pos_row, next(iter(positions)) if len(positions) == 1 and len(groups) == 1 else None)
            for k in feeds:
                cands[k].logits = out[k * R:(k + 1) * R]
                pos[k] += 1
        res = [[] for _ in groups]
        for c in cands:
            stats = dict(tokens=len(c.tokens), prefix_tokens=c.plen, backtracks=c.total_backtracks + c.backtracks,
                         token_budget_forced=c.forced, resamples=c.resamples, fallback=c.fallback)
            res[c.group].append((c.tokens, c.log_probs, stats))
        return res
    finally:
        backend.end_window(window)


# ---------------------------------------------------------------------------------------------- teacher-forced log-probability
def target_cur_times(V, tokens):
    """cur_time input of every target token (running time of the last time token, 0 before the first), as in training."""
    out, cur = [], 0
    for t in tokens:
        if V.is_time(int(t)):
            cur = V.tok_time(int(t))
        out.append(cur)
    return out


@torch.no_grad()
def tf_logprobs(q, enc, song, items, chunk_rows=24):
    """items: list of (prefixes [R dicts], target tokens incl. EOS).  Returns a list of summed log-probabilities of the targets under
    the sampling distribution without grammar mask / steering bias: CFG-combined rows, tempered (timing tokens ``temperature``,
    the rest ``content_temperature``), softmax over the whole vocabulary, designer rows mixed."""
    G = _gen()
    o, V, model = q.options, q.V, q.model
    device = enc.device
    R = len(items[0][0])
    n_des = G.designer_rows(o, R)
    plan = G.guidance_plan(o)
    timing = torch.as_tensor(G.timing_mask(V), device=device)
    t_time = o.temperature if o.temperature > 0 else 1.0
    div = torch.where(timing, torch.tensor(t_time, device=device), torch.tensor(max(o.content_temperature, 1e-6), device=device)).float()
    per = max(1, chunk_rows // R)
    out = []
    for s0 in range(0, len(items), per):
        chunk = items[s0:s0 + per]
        ns = len(chunk)
        seqs, cts, meta = [], [], []
        for prefixes, target in chunk:
            tct = target_cur_times(V, target)
            for p in prefixes:
                seqs.append(list(map(int, p["tokens"])) + [int(t) for t in target])
                cts.append(list(map(int, p["cur_time"])) + tct)
            meta.append((len(prefixes[0]["tokens"]), len(target)))
        L = max(len(s) for s in seqs)
        tok = torch.full((len(seqs), L), V.PAD, dtype=torch.long)
        ct = torch.zeros((len(seqs), L), dtype=torch.long)
        for b, (s, c) in enumerate(zip(seqs, cts)):
            tok[b, :len(s)] = torch.as_tensor(s)
            ct[b, :len(c)] = torch.as_tensor(c)
        tok, ct = tok.to(device), ct.to(device)
        song_c = None
        if song is not None:
            song_c = dict(mem=song["mem"].repeat(ns, 1, 1), mask=song["mask"].repeat(ns, 1, 1, 1), keep=song["keep"].repeat(ns))
        with q.backend.ac():
            h = model.decode_hidden(tok, ct, enc.expand(len(seqs), -1, -1), None, song_c)
        for si, (lp, tl) in enumerate(meta):
            hs = h[si * R:(si + 1) * R, lp - 1:lp - 1 + tl]                     # positions predicting the targets
            with q.backend.ac():
                vals = F.linear(hs, model.tok_emb.weight).float()                # [R, T, V]
            cond = vals[:n_des]
            comb = cond
            row = n_des
            for _, w, _, _ in plan:
                comb = comb + (w - 1.0) * (cond - vals[row:row + n_des])
                row += n_des
            if o.cfg > 1:
                comb = comb + (o.cfg - 1.0) * (cond - vals[row][None])
            logp = torch.log_softmax(comb / div, dim=-1)                         # [n_des, T, V]
            tgt = torch.as_tensor(chunk[si][1], device=device)
            lpt = logp.gather(-1, tgt[None, :, None].expand(n_des, -1, 1)).squeeze(-1)
            mix = torch.logsumexp(lpt, 0) - math.log(n_des)
            out.append(float(mix.sum()))
    return out


# ---------------------------------------------------------------------------------------------- helpers
def hold_carry(carry_end, notes):
    for n in notes:
        if n.kind in ("hold", "thold", "slide"):
            carry_end = max(carry_end, int(n.end_beat()))
    return carry_end


def hand_pen(pen):
    return max(0.0, pen - HAND_FREE)


# ---------------------------------------------------------------------------------------------- density / activity guard
GUARD_SPARSE = 0.55       # a window must hold at least this share of the notes expected for its level and audio (only when >= GUARD_MIN_E are expected)
GUARD_MIN_E = 8.0
GUARD_DENSE = 2.0         # ... and at most this ratio (n + 1) / (E + 1)
GUARD_ACT = 0.12          # ... and its activity penalty (empty half-bars while the music is active) may not exceed this


def guard_violation(parts):
    """0 for a window inside the guard, else the size of the violation (sparse share + dense ratio excess + activity excess).  parts = WindowScorer.score parts."""
    v = 0.0
    e = parts.get("expected")
    n = parts.get("n_notes", 0)
    if e is not None:
        if e >= GUARD_MIN_E:
            v += max(0.0, GUARD_SPARSE - n / e) / GUARD_SPARSE
        v += max(0.0, (n + 1.0) / (e + 1.0) - GUARD_DENSE)
    v += max(0.0, parts.get("p_activity", 0.0) - GUARD_ACT) / GUARD_ACT
    return float(v)


def guard_components(parts):
    """(sparse, dense, activity) parts of guard_violation"""
    e, n = parts.get("expected"), parts.get("n_notes", 0)
    sp = dn = 0.0
    if e is not None:
        if e >= GUARD_MIN_E:
            sp = max(0.0, GUARD_SPARSE - n / e) / GUARD_SPARSE
        dn = max(0.0, (n + 1.0) / (e + 1.0) - GUARD_DENSE)
    return sp, dn, max(0.0, parts.get("p_activity", 0.0) - GUARD_ACT) / GUARD_ACT


def guard_state(parts_list):
    """None for an empty pool, else dict(min = least guard violation, dir = +1 when the pool is too sparse / leaves active music empty, -1 when too dense)"""
    if not parts_list:
        return None
    viol = [guard_violation(p) for p in parts_list]
    comp = np.mean([guard_components(p) for p in parts_list], axis=0)
    return dict(min=float(min(viol)), dir=(1 if comp[0] + comp[2] > comp[1] else -1))


def guard_shift(tilt, direction, k):
    """temporary offset on the density tilt for guard resample round k (1-based): towards fewer notes (the tilt is a penalty per beat of gap, so negative = longer gaps)
    or more notes.  The knob lags changes of the expected density by a few windows, so it can sit far from where the new window needs it."""
    if direction < 0:
        return -(max(float(tilt), 0.0) + 0.3) - 0.5 * (k - 1)
    return 0.8 * k


def guard_filter(entries):
    """entries of the window loop (e[5] = parts): keep the ones inside the guard.  When none is (a genuine break / intro the audio model counts as
    active, or a window the model wants dense), keep the least violating ones: the guard removes outliers when an alternative exists, it never forces notes."""
    viol = [guard_violation(e[5]) for e in entries]
    for e, v in zip(entries, viol):
        e[5]["p_guard"] = v
    lo = min(viol)
    keep = [e for e, v in zip(entries, viol) if v <= lo + (1e-9 if lo <= 1e-9 else 0.05)]
    return keep, float(lo)


def muri_full(gate, counts):
    """full-price muri cost of a category count (no per-chart budget discount)"""
    hard = counts.get("overlap", 0) + (counts.get("outer", 0) + counts.get("tail", 0) if gate.hard_tail else 0)
    pen = gate.w_hard * (hard + counts.get("overlap", 0))
    for cat in ["inner", "multi"] + ([] if gate.hard_tail else ["outer", "tail"]):
        pen += gate.w_soft * murigate.SOFT_WEIGHT[cat] * counts.get(cat, 0)
    return float(pen), int(hard)


def pick_top(entries, m):
    """the m best distinct (by notes) candidate entries by score"""
    from .simai import canon_key
    out, seen = [], set()
    for e in sorted(entries, key=lambda e: (-e[1], e[2], e[3])):
        key = tuple(canon_key(n) for n in e[4])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= m:
            break
    return out


def stuck_penalty(stats):
    return (2.0 if stats.get("fallback") else 0.0) + (0.5 if stats.get("resamples") else 0.0) + min(0.3, 0.002 * stats.get("backtracks", 0))


# ---------------------------------------------------------------------------------------------- lookahead
def lookahead(q, st, top):
    """Continuation value of the candidate entries ``top`` of the current window (generator; see module doc).
    st: SimpleNamespace(index, start, window_end, notes, packed, scorer, gate, bias, hands, carry_end).
    entries: (hard, score, rnd, k, new_notes, parts, results, guards, assessed).  Returns (values, info)."""
    G = _gen()
    o, V = q.options, q.V
    t0 = time.perf_counter()
    nxt = st.index + 1
    start_n = q.starts[nxt]
    end_n = min(start_n + T_MAX, q.end_tick)
    bpm_n, prog_n = bpm_progress_tokens(V, q.tempo, start_n, q.song_beats)
    w3n, ref_n, rows_n = None, None, q.rows
    if V.seq_version >= 3:
        ref_n = (V["REF_NONE"], [], [])
        if q.v3ctx is not None:
            w3n = q.v3ctx.window(start_n // V3.BAR, *st.packed, batch=len(q.rows), plan_shift=o.plan_shift,
                                 use_plan=o.v3_plan, use_ref=o.v3_ref, use_song=o.v3_song)
            ref_n = w3n["ref"]
        rows_n = G.rows_with_plan(V, o, q.rows, w3n["plan_token"] if w3n else V["PLAN_UNK"])
    enc_n = q.encode(start_n)
    old_scale = st.scorer.scale
    scale_n = old_scale
    if w3n is not None and o.plan_density > 0 and w3n["plan"] is not None:
        scale_n = q.scale0 * float(q.gen.plan_ratio(w3n["plan"])) ** o.plan_density
    S = o.la_samples
    groups, ctx = [], []
    for m, e in enumerate(top):
        new_m, tokens_m, src = e[4], e[6][e[3]][0], e[7][e[3]]
        notes_m = st.notes + list(new_m)
        packed_m = pack_notes(V, notes_m)
        resv = min(o.max_target_tokens, max(256, len(tokens_m) + 96), q.model.cfg.max_len - V.header_len - 2)
        prefixes_m, _ = q.gen.build_prefixes(packed_m, start_n, rows_n, bpm_n, prog_n, resv, ref_n, log=q.log)
        if prefixes_m[0] is None:
            raise ValueError("checkpoint max_len is too small for an inference prefix")
        base = G.HandGuard(V, q.tempo, start_n, q.end_tick, src.tracker.copy(), o.prune_end)
        base.holds = list(src.holds)
        base.recent = [n for n in notes_m if n.beat >= start_n - 6 * 4 * TPB]
        bias_m = fork_steer(st.bias)
        st.scorer.scale = scale_n
        bias_m.begin_window(nxt, start_n, notes_m)
        st.scorer.scale = old_scale
        groups.append(dict(prefixes=prefixes_m, guards=[base.clone() for _ in range(S)], bias=bias_m,
                           rng_factory=lambda j, attempt, _n=nxt: np.random.default_rng([q.seed, _n, 1000 + j, attempt])))
        ctx.append((notes_m, bias_m))
    results = yield from decode_groups_co(q.model, enc_n, groups, o, V, min(T_MAX, q.end_tick - start_n), backend=q.backend,
                                          song=w3n["song"] if w3n is not None else None)
    values, details = [], []
    for m, e in enumerate(top):
        new_m = e[4]
        res = results[m]
        notes_next = [canonicalize(decode_events(V, toks, start_n)) for toks, _, _ in res]
        gate_m, assessed = None, [None] * S
        if st.gate is not None:
            gate_m = fork_gate(st.gate)
            sec0, calls0 = gate_m.seconds, gate_m.calls
            ass = e[8] if (e[8] is not None and e[8]["ok"]) else dict(events=[])
            gate_m.accept(new_m, ass, st.start, st.window_end)
            gate_m.begin_window(nxt)
            assessed = gate_m.assess(notes_next, start_n, end_n)
            st.gate.seconds += gate_m.seconds - sec0
            st.gate.calls += gate_m.calls - calls0
        sc_m = fork_scorer(st.scorer)
        sc_m.scale = scale_n
        sc_m.accept(e[5])
        carry_m = hold_carry(st.carry_end, new_m)
        hands_m = st.hands.copy() if st.hands is not None else None
        if hands_m is not None:
            hands_m.advance(new_m, st.start, st.window_end)
        vals = []
        for j in range(S):
            lp = float(np.mean(res[j][1])) if res[j][1] else 0.0
            score_j, parts_j = sc_m.score(notes_next[j], start_n, lp, carry_m, muri=assessed[j])
            ph = 0.0
            if hands_m is not None and o.w_hand > 0:
                ph = hands_m.term(hands_m.peek(notes_next[j], start_n, end_n)[0])
            gj = o.qguard_w * guard_violation(parts_j) if o.qguard else 0.0
            vals.append(score_j - o.w_hand * ph + o.la_lp * lp - stuck_penalty(res[j][2]) - gj)
        values.append(float(0.5 * np.max(vals) + 0.5 * np.mean(vals)))      # the next window picks the best of K: favour a good best case
        details.append(dict(value=round(values[-1], 4), vals=[round(v, 3) for v in vals], notes=[len(n) for n in notes_next]))
    return values, dict(seconds=round(time.perf_counter() - t0, 3), details=details)


# ---------------------------------------------------------------------------------------------- replay / badness
def replay(q, records, upto, collect=None, with_tracker=False):
    """Rebuild the generation state in front of window ``upto`` from the window records (scorer, steering, hand model, tracker).
    ``collect``: list that receives (score, parts, hand penalty) of every replayed window."""
    scorer = q.make_scorer()
    bias = q.make_bias(scorer)
    hands = q.make_hands()
    notes, carry_end = [], 0
    tracker = playcheck.HandTracker(q.tempo) if with_tracker else None
    for j in range(upto):
        r = records[j]
        scorer.scale = r["scale"]
        bias.begin_window(j, r["start"], notes)
        score, parts = scorer.score(r["notes"], r["start"], r["mean_lp"], carry_end)
        scorer.accept(parts)
        pen, hst = hands.advance(r["notes"], r["start"], r["end"])
        if collect is not None:
            collect.append((score, parts, pen))
        if tracker is not None:
            for n in r["notes"]:
                tracker.add(n)
        notes.extend(r["notes"])
        carry_end = hold_carry(carry_end, r["notes"])
    return SimpleNamespace(notes=notes, scorer=scorer, bias=bias, hands=hands, carry_end=carry_end, tracker=tracker)


def chart_notes(records):
    out = []
    for r in records:
        out.extend(r["notes"])
    return out


def whole_events(q, records):
    gate = q.refine_gate
    notes = canonicalize(chart_notes(records))
    ev = gate._events(notes, -1e9, 1e12)
    return ev


def window_badness(q, records):
    """Per-window cost components (higher = worse) of the current chart."""
    o = q.options
    n = len(records)
    col = []
    q_hands = replay(q, records, n, collect=col).hands
    ev = whole_events(q, records)
    gate = q.refine_gate
    sec_edges = [gate._sec(r["start"]) for r in records] + [gate._sec(records[-1]["end"]) + 1.0]
    counts = [Counter() for _ in range(n)]
    for e in ev or []:
        j = int(np.searchsorted(sec_edges, e["time"], side="right")) - 1
        if 0 <= j < n:
            counts[j][murigate.CAT_KEY[e["category"]]] += 1
    lps = np.asarray([r["mean_lp"] for r in records], float)
    z = (lps.mean() - lps) / max(float(lps.std()), 1e-6)
    comp = []
    for j in range(n):
        score, parts, pen = col[j]
        m_cost, hard = muri_full(gate, counts[j])
        c = dict(rules=-score, muri=m_cost, hard=hard, hand=o.w_hand * q_hands.bad(pen), stuck=stuck_penalty(records[j]["stats"]),
                 guard=(o.qguard_w * guard_violation(parts) if o.qguard else 0.0),
                 lpz=o.refine_lpz * max(0.0, float(z[j])), act=parts["p_activity"], rep=parts["p_repeat"], den=parts["p_density"],
                 inten=parts.get("p_intensity", 0.0))
        c["total"] = c["rules"] + c["muri"] + c["hand"] + c["stuck"] + c["lpz"] + c["guard"]
        comp.append(c)
    return comp, ev


# ---------------------------------------------------------------------------------------------- self-refine
def region_cost(q, records, st, i, cand):
    """Cheap (numpy) cost of a replacement ``cand`` = dict(notes, ...) of window i: rules + hand over the window and the next
    ``refine_after`` existing windows."""
    o = q.options
    n = len(records)
    sc = fork_scorer(st.scorer)
    hands = st.hands.copy()
    carry = st.carry_end
    rules = 0.0
    hand = 0.0
    parts0 = None
    for off in range(0, 1 + o.refine_after):
        j = i + off
        if j >= n:
            break
        r = records[j]
        nn = cand["notes"] if off == 0 else r["notes"]
        lp = cand["mean_lp"] if off == 0 else r["mean_lp"]
        sc.scale = r["scale"]
        score, parts = sc.score(nn, r["start"], lp, carry)
        if off == 0:
            parts0 = parts
        sc.accept(parts)
        carry = hold_carry(carry, nn)
        rules += -score
        pen, _ = hands.advance(nn, r["start"], r["end"])
        hand += o.w_hand * hands.term(pen)
    return rules, hand, parts0


def affected_windows(q, records, i):
    """Windows whose log-probability depends on window i: the next one, plus later windows whose REF block overlaps its bars."""
    n = len(records)
    aff = []
    if i + 1 < n:
        aff.append(i + 1)
    b0, b1 = records[i]["start"] // BAR, records[i]["start"] // BAR + 4
    for j in range(i + 2, n):
        e = records[j].get("ref_block")
        if e is not None and records[j].get("ref_tokens", 0) > 0 and e - 4 < b1 and e > b0:
            aff.append(j)
    return aff[:q.options.refine_max_affected]


def lp_terms(q, records, st, i, cand_list, aff):
    """Summed teacher-forced log-prob of the affected windows for each candidate in ``cand_list`` (dict with notes, tokens)."""
    G = _gen()
    o, V = q.options, q.V
    out = [0.0] * len(cand_list)
    for j in aff:
        r = records[j]
        start = r["start"]
        bpm, prog = bpm_progress_tokens(V, q.tempo, start, q.song_beats)
        items, song_j = [], None
        for ci, cand in enumerate(cand_list):
            notes_mod = list(st.notes) + list(cand["notes"])
            for jj in range(i + 1, j):
                notes_mod.extend(records[jj]["notes"])
            packed = pack_notes(V, notes_mod)
            w3, ref, rows_w = None, None, q.rows
            if V.seq_version >= 3:
                ref = (V["REF_NONE"], [], [])
                if q.v3ctx is not None:
                    w3 = q.v3ctx.window(start // V3.BAR, *packed, batch=len(q.rows), plan_shift=o.plan_shift,
                                        use_plan=o.v3_plan, use_ref=o.v3_ref, use_song=o.v3_song)
                    ref = w3["ref"]
                rows_w = G.rows_with_plan(V, o, q.rows, w3["plan_token"] if w3 else V["PLAN_UNK"])
            prev_tokens = len(cand["tokens"]) if j == i + 1 else len(records[j - 1]["tokens"])
            resv = min(o.max_target_tokens, max(256, prev_tokens + 96), q.model.cfg.max_len - V.header_len - 2)
            prefixes, _ = q.gen.build_prefixes(packed, start, rows_w, bpm, prog, resv, ref, log=q.log)
            if prefixes[0] is None:
                items.append(None)
                continue
            if w3 is not None:
                song_j = w3["song"]
            items.append((prefixes, [int(t) for t in r["tokens"]]))
        ok = [k for k, it in enumerate(items) if it is not None]
        if not ok:
            continue
        lps = tf_logprobs(q, q.encode(start), song_j, [items[k] for k in ok])
        for k, lp in zip(ok, lps):
            out[k] += lp
    return out


def refine_window(q, records, i, p, log):
    """Regenerate window i (K candidates), pick by the region cost; returns (accepted, info) (generator)."""
    G = _gen()
    o, V = q.options, q.V
    t0 = time.perf_counter()
    rec = records[i]
    start, end = rec["start"], rec["end"]
    st = replay(q, records, i, with_tracker=True)
    # ---- decode inputs of window i from the chart in front of it
    packed = pack_notes(V, st.notes)
    bpm, prog = bpm_progress_tokens(V, q.tempo, start, q.song_beats)
    w3, ref, rows_w = None, None, q.rows
    if V.seq_version >= 3:
        ref = (V["REF_NONE"], [], [])
        if q.v3ctx is not None:
            w3 = q.v3ctx.window(start // V3.BAR, *packed, batch=len(q.rows), plan_shift=o.plan_shift,
                                use_plan=o.v3_plan, use_ref=o.v3_ref, use_song=o.v3_song)
            ref = w3["ref"]
        rows_w = G.rows_with_plan(V, o, q.rows, w3["plan_token"] if w3 else V["PLAN_UNK"])
    prev_tokens = len(records[i - 1]["tokens"]) if i > 0 else 384
    resv = min(o.max_target_tokens, max(256, prev_tokens + 96), q.model.cfg.max_len - V.header_len - 2)
    prefixes, _ = q.gen.build_prefixes(packed, start, rows_w, bpm, prog, resv, ref, log=q.log)
    if prefixes[0] is None:
        return False, dict(window=i, skipped="prefix")
    holds = [n for n in st.notes if n.kind == "hold" and n.beat + n.dur > start]
    base = G.HandGuard(V, q.tempo, start, q.end_tick, st.tracker, o.prune_end)
    base.holds = holds
    base.recent = [n for n in st.notes if n.beat >= start - 6 * 4 * TPB]
    st.scorer.scale = rec["scale"]
    st.bias.begin_window(i, start, st.notes)
    enc = q.encode(start)
    K = o.refine_k if o.refine_k > 0 else o.window_candidates
    if os.environ.get("MAI_REFINE_SELFTEST") and not q.selftest_done and rec["stats"].get("chosen", -1) >= 0 \
            and rec["stats"].get("muri", {}).get("chosen_round", 0) == 0:
        # replay check: the state rebuilt from the records must reproduce the original decode of this window (same seeds)
        q.selftest_done = True
        guards_t = [base.clone() for _ in range(o.window_candidates)]
        res_t = yield from G.decode_candidates_co(q.model, enc, prefixes, o, V, guards_t,
                                                  lambda k, attempt: np.random.default_rng([q.seed, i, k, attempt]),
                                                  min(T_MAX, q.end_tick - start), bias=st.bias, backend=q.backend,
                                                  song=w3["song"] if w3 is not None else None)
        kc = rec["stats"]["chosen"]
        same = [list(map(int, r[0])) for r in res_t][kc] == list(map(int, rec["tokens"]))
        n_same = sum(list(map(int, r[0])) == list(map(int, rec["tokens"])) for r in res_t)
        q.log(f"SELFTEST window {i}: chosen candidate {kc} reproduced={same} (candidates with identical tokens: {n_same})", flush=True)
        # the replayed bias consumed state during that decode: rebuild it
        st = replay(q, records, i, with_tracker=True)
        st.scorer.scale = rec["scale"]
        st.bias.begin_window(i, start, st.notes)
        base = G.HandGuard(V, q.tempo, start, q.end_tick, st.tracker, o.prune_end)
        base.holds = holds
        base.recent = [n for n in st.notes if n.beat >= start - 6 * 4 * TPB]
    aff = affected_windows(q, records, i)
    existing = dict(notes=rec["notes"], tokens=rec["tokens"], mean_lp=rec["mean_lp"])
    ex_rules, ex_hand, ex_parts = region_cost(q, records, st, i, existing)
    ex_viol = guard_violation(ex_parts) if o.qguard else 0.0
    if o.qguard and ex_viol > o.qguard_resample:
        # the window violates the density / activity guard: regenerate it with the density tilt shifted the right way (as the guard resample rounds of the generation do)
        st.bias.tilt_shift = guard_shift(st.bias.tilt.value, guard_state([ex_parts])["dir"], 1)
    # ---- candidates (one round; a second one when nothing usable came out)
    pool = []
    seen = set()
    for rnd in range(2):
        guards = [base.clone() for _ in range(K)]

        def rng_factory(k, attempt, _rnd=rnd):
            return np.random.default_rng([q.seed, 7000 + 100 * p + _rnd, i, k, attempt])

        results = yield from G.decode_candidates_co(q.model, enc, prefixes, o, V, guards, rng_factory, min(T_MAX, q.end_tick - start),
                                                    bias=st.bias, backend=q.backend, song=w3["song"] if w3 is not None else None)
        for k, (tokens, probs, stats) in enumerate(results):
            nn = canonicalize(decode_events(V, tokens, start))
            key = tuple(simai_key(n) for n in nn)
            if key in seen:
                continue
            seen.add(key)
            pool.append(dict(notes=nn, tokens=tokens, probs=probs, stats=stats, guard=guards[k],
                             mean_lp=float(np.mean(probs)) if probs else 0.0))
        if pool or rnd == 1:
            break
    st.bias.tilt_shift = 0.0
    ex_key = tuple(simai_key(n) for n in existing["notes"])
    pool = [c for c in pool if tuple(simai_key(n) for n in c["notes"]) != ex_key and not c["stats"]["fallback"]]
    info = dict(window=i, pass_=p, candidates=len(pool), affected=aff)
    if not pool:
        info.update(accepted=False, reason="no candidate")
        return False, info
    # ---- cheap cost, then the expensive terms for the best few
    cheap = []
    for c in pool:
        c["rules"], c["hand"], c["parts0"] = region_cost(q, records, st, i, c)
        c["viol"] = guard_violation(c["parts0"]) if o.qguard else 0.0
    if o.qguard:
        # never accept a replacement that is worse than the existing window on the density / activity guard, nor one that thins the window out
        e_exp = ex_parts.get("expected")
        n_old = len(existing["notes"])
        pool = [c for c in pool if c["viol"] <= ex_viol + 0.02
                and (len(c["notes"]) >= o.refine_keep * n_old or (e_exp is not None and len(c["notes"]) >= 0.9 * e_exp))]
        if not pool:
            info.update(accepted=False, reason="guard")
            return False, info
    pool.sort(key=lambda c: c["rules"] + c["hand"])
    short = pool[:o.refine_eval]
    evalset = [existing] + short
    gate = q.refine_gate
    # muri over the region [window i, window i+refine_after] on the modified chart
    last = min(len(records) - 1, i + o.refine_after)
    t_lo, t_hi = gate._sec(start), gate._sec(records[last]["end"])
    after = []
    for jj in range(i + 1, len(records)):
        after.extend(records[jj]["notes"])
    for c in evalset:
        modified = canonicalize(list(st.notes) + list(c["notes"]) + after)
        c["modified"] = modified
        lo, hi = t_lo - gate.pre, t_hi + gate.post
        sub = [n for n in modified if gate._sec(n.end_beat()) >= lo and gate._sec(n.beat) <= hi]
        evs = gate._events(sub, t_lo, t_hi + gate.margin)
        if evs is None:
            c["muri"], c["hard"] = 0.0, 0
        else:
            cnt = Counter(murigate.CAT_KEY[e["category"]] for e in evs)
            c["muri"], c["hard"] = muri_full(gate, cnt)
            extra = murigate.static_overlaps(sub, start, records[last]["end"])
            if extra:
                c["hard"] += extra
                c["muri"] += gate.w_hard * 2 * extra
    lps = lp_terms(q, records, st, i, evalset, aff) if aff else [0.0] * len(evalset)
    for c, lp in zip(evalset, lps):
        c["lp"] = lp
    ex = evalset[0]
    ex["rules"], ex["hand"], ex["viol"] = ex_rules, ex_hand, ex_viol

    def total(c):
        return c["rules"] + c["hand"] + c["muri"] - o.refine_lp * c["lp"] + (o.qguard_w * c.get("viol", 0.0) if o.qguard else 0.0)
    for c in evalset:
        c["total"] = total(c)
    ranked = sorted(short, key=lambda c: c["total"])
    info.update(existing=dict(total=round(ex["total"], 3), rules=round(ex["rules"], 3), hand=round(ex["hand"], 3), muri=round(ex["muri"], 3),
                              lp=round(ex["lp"], 2)),
                best=[dict(total=round(c["total"], 3), rules=round(c["rules"], 3), hand=round(c["hand"], 3), muri=round(c["muri"], 3),
                           lp=round(c["lp"], 2)) for c in ranked[:3]])
    for c in ranked:
        if c["total"] >= ex["total"] - o.refine_margin:
            break
        if c["hard"] > ex["hard"]:
            continue
        chk = playcheck.check(c["modified"], q.tempo, stats=False)
        if chk["n_viol"] > 0:
            continue
        # accepted
        new_stats = dict(rec["stats"])
        new_stats.update(c["stats"])
        new_stats.update(resamples_all=c["stats"]["resamples"], fallback_all=int(bool(c["stats"]["fallback"])), refined=p + 1,
                         notes=len(c["notes"]), chosen=-1)
        rec_new = dict(rec)
        rec_new.update(notes=c["notes"], tokens=c["tokens"], probs=c["probs"], mean_lp=c["mean_lp"], stats=new_stats)
        records[i] = rec_new
        info.update(accepted=True, delta=round(ex["total"] - c["total"], 3), seconds=round(time.perf_counter() - t0, 2))
        return True, info
    info.update(accepted=False, reason="no improvement", seconds=round(time.perf_counter() - t0, 2))
    return False, info


def simai_key(n):
    from .simai import canon_key
    return canon_key(n)


def self_refine(q, records, log=print):
    """Run ``refine_passes`` passes (generator).  Returns the per-pass info list."""
    o = q.options
    out = []
    for p in range(o.refine_passes):
        t0 = time.perf_counter()
        comp, ev = window_badness(q, records)
        q.last_events = ev
        n = len(records)
        order = sorted(range(n), key=lambda j: -comp[j]["total"])
        k = max(1, int(math.ceil(o.refine_frac * n)))
        targets = sorted(j for j in order[:k] if comp[j]["total"] > o.refine_min_bad)
        if o.refine_tail_bars > 0:
            cut = q.end_tick - int(o.refine_tail_bars) * BAR
            targets = [j for j in targets if records[j]["start"] < cut]
        details, changed = [], 0
        for i in targets:
            ok, info = yield from refine_window(q, records, i, p, log)
            info["before"] = round(comp[i]["total"], 3)
            details.append(info)
            changed += int(ok)
            if ok:
                q.last_events = None            # events of the old chart: recomputed at the next pass; region terms fall back to none
            log(f"refine pass {p + 1} window {i + 1}/{n}: bad={comp[i]['total']:.2f} "
                + ("replaced (-%.2f)" % info["delta"] if ok else f"kept ({info.get('reason', '')})"), flush=True)
        out.append(dict(pass_=p + 1, targets=targets, changed=changed, seconds=round(time.perf_counter() - t0, 2),
                        total_before=round(sum(c["total"] for c in comp), 3), windows=details,
                        hard_before=int(sum(c["hard"] for c in comp))))
        if not changed:
            break
    return out


def chart_cost(q, records):
    comp, ev = window_badness(q, records)
    return float(sum(c["total"] for c in comp)), comp, ev
