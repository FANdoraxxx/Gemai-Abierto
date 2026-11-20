"""Cross-song batched decoding: several songs share one KV-cache pool and one decoder forward per step.

``generate.ChartGenerator.generate_co`` is a generator that yields a step request whenever it needs the decoder logits of
its current window.  ``run_pool`` keeps ``batch`` such generators alive, collects their requests, runs ONE decoder forward
over the union of their cache rows (``rows_step``: every row has its own position) and sends each song its logits.
The Python work of the songs (grammar, hand constraints, sampling, steering, reranking) is untouched, only the
per-token GPU launch overhead is shared.  Row ``k*R + r`` of a window's block is candidate ``k``, prefix row ``r`` --
the layout of the single-song path (``generate.LegacyBackend``).

Numerics: the pooled forward uses the masked ``rows_step`` kernel path for every step, so logits agree with the
single-song path to float rounding, not bit for bit; sampled charts are therefore equal in distribution, not identical.
"""
from __future__ import annotations

from collections import deque
import time

import numpy as np
import torch

from .model import DecodeCache
from . import generate as gen


@torch.no_grad()
def fast_rows_step(model, tok, cur_time, cache, pos, last):
    """``generate.rows_step`` with the attention written as batched matmuls + softmax.

    For one query per row the fused SDPA kernels are 2-3x slower than cuBLAS on the TITAN Xp (fp32); the maths is the same
    (differences ~4e-7), so charts are equal in distribution but not bit-identical to the single-song path.
    """
    cfg = model.cfg
    B = tok.shape[0]
    H, hd = cfg.heads, cfg.d // cfg.heads
    ct = cur_time.view(-1, 1)
    phase = torch.remainder(ct, cfg.n_phase)
    beat = (torch.div(ct, gen.TPB, rounding_mode="floor") + cfg.beat_offset).clamp(0, cfg.n_beat - 1)
    x = (model.tok_emb(tok.view(-1, 1)) + model.pos_emb(pos.view(-1, 1))
         + model.phase_emb(phase) + model.beat_emb(beat))
    if cfg.lv_cont:
        x = x + model.lv_term(tok.view(-1, 1))
    if cache.g is not None:
        x = x + cache.g[:, None]
    dpos, dneg = model.deltas(ct, x)
    visible = (torch.arange(last, device=tok.device)[None, :] <= pos[:, None])[:, None, None, :]
    rows = torch.arange(B, device=tok.device)
    scale = 1.0 / (hd ** 0.5)
    for l, blk in enumerate(model.dec_blocks):
        q, k, v = blk.sa._qkv(blk.ln1(x))
        cache.k[l][rows, :, pos] = k[:, :, 0].to(cache.k[l].dtype)
        cache.v[l][rows, :, pos] = v[:, :, 0].to(cache.v[l].dtype)
        s = torch.matmul(q * scale, cache.k[l][:, :, :last].to(q.dtype).transpose(-1, -2))
        s = s.masked_fill(~visible, float("-inf"))
        y = torch.matmul(torch.softmax(s, dim=-1), cache.v[l][:, :, :last].to(q.dtype))
        x = x + blk.sa.proj(y.transpose(1, 2).reshape(B, 1, -1))
        ca = blk.ca
        qc = ca.q(blk.ln2(x)).view(B, 1, ca.h, ca.hd).transpose(1, 2)
        sc = torch.matmul(qc * scale, cache.ck[l].to(qc.dtype).transpose(-1, -2)) + ca.bias(dpos, dneg, qc.dtype)
        yc = torch.matmul(torch.softmax(sc, dim=-1), cache.cv[l].to(qc.dtype))
        x = x + ca.proj(yc.transpose(1, 2).reshape(B, 1, -1))
        if blk.sca is not None and cache.sk is not None:
            sca = blk.sca
            qs = sca.q(blk.ln_s(x)).view(B, 1, sca.h, sca.hd).transpose(1, 2)
            ss = torch.matmul(qs * scale, cache.sk[l].to(qs.dtype).transpose(-1, -2)).masked_fill(~cache.smask, float("-inf"))
            ys = torch.matmul(torch.softmax(ss, dim=-1), cache.sv[l].to(qs.dtype))
            x = x + sca.proj(ys.transpose(1, 2).reshape(B, 1, -1)) * cache.skeep.to(qs.dtype).view(B, 1, 1)
        x = x + blk.ff(blk.ln3(x))
    return torch.nn.functional.linear(model.dec_ln(x[:, 0]), model.tok_emb.weight)


class _Graph:
    """CUDA graph of one decoder step over ALL pool rows with attention length ``last`` (static shapes)."""
    def __init__(self, pool, hi, last, packed_dev):
        self.hi, self.last = hi, last
        self.inp = packed_dev.clone()
        view = pool._view(hi)
        model, engine = pool.model, pool.engine
        run = lambda: pool.step_fn(model, self.inp[0], self.inp[1], view, self.inp[2], last=last)
        side = torch.cuda.Stream(device=pool.device)
        side.wait_stream(torch.cuda.current_stream(pool.device))
        with torch.cuda.stream(side), engine.autocast():
            for _ in range(2):          # warm-up with the real inputs of this step: its cache writes are idempotent
                run()
        torch.cuda.current_stream(pool.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool.graph_pool), engine.autocast():
            self.out = run()


class PoolBackend:
    def __init__(self, engine, rows, graph=False, fast=True):
        model = engine.model
        cfg = model.cfg
        self.engine, self.model, self.device, self.rows = engine, model, engine.device, rows
        self.dt = engine.dtype if engine.dtype is not None else torch.float32
        H, hd = cfg.heads, cfg.d // cfg.heads
        make = lambda length: [torch.zeros(rows, H, length, hd, dtype=self.dt, device=self.device)
                               for _ in model.dec_blocks]
        self.k, self.v = make(cfg.max_len), make(cfg.max_len)
        self.ck, self.cv = make(cfg.enc_len), make(cfg.enc_len)
        self.g = torch.zeros(rows, cfg.d, dtype=self.dt, device=self.device) if cfg.cond_inject else None
        # seq_version 3: song-memory K/V per decoder layer (padded to SONG_CAP tokens, masked), one slot per pool row
        self.sk = self.sv = self.smask = self.skeep = None
        if getattr(cfg, "song_dim", 0):
            from .v3 import SONG_CAP
            self.song_cap = SONG_CAP
            self.sk = [torch.zeros(rows, H, SONG_CAP, hd, dtype=self.dt, device=self.device) if blk.sca is not None else None
                       for blk in model.dec_blocks]
            self.sv = [torch.zeros(rows, H, SONG_CAP, hd, dtype=self.dt, device=self.device) if blk.sca is not None else None
                       for blk in model.dec_blocks]
            self.smask = torch.ones(rows, 1, 1, SONG_CAP, dtype=torch.bool, device=self.device)
            self.skeep = torch.zeros(rows, dtype=self.dt, device=self.device)
        self.free = [(0, rows)]
        self._views = {}
        self.steps = 0
        self.step_rows = 0
        self.graph = bool(graph) and self.device.type == "cuda"
        self.step_fn = fast_rows_step if fast else gen.rows_step
        self._graphs = {}
        self.graph_pool = torch.cuda.graph_pool_handle() if self.graph else None
        self.timing = dict(prep=0.0, forward=0.0, fetch=0.0)
        self._host = None
        self._pin_in = None

    # ------------------------------------------------------------------ row allocation (first fit)
    def alloc(self, n):
        for i, (start, size) in enumerate(self.free):
            if size >= n:
                if size == n:
                    del self.free[i]
                else:
                    self.free[i] = (start + n, size - n)
                return start
        return None

    def release(self, start, n):
        self.free.append((start, n))
        self.free.sort()
        merged = []
        for s, size in self.free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((s, size))
        self.free = merged

    # ------------------------------------------------------------------ stepping
    def _view(self, hi):
        view = self._views.get(hi)
        if view is None:
            view = DecodeCache([a[:hi] for a in self.k], [a[:hi] for a in self.v],
                               [a[:hi] for a in self.ck], [a[:hi] for a in self.cv])
            view.g = self.g[:hi] if self.g is not None else None
            if self.sk is not None:
                view.sk = [None if a is None else a[:hi] for a in self.sk]
                view.sv = [None if a is None else a[:hi] for a in self.sv]
                view.smask, view.skeep = self.smask[:hi], self.skeep[:hi]
            self._views[hi] = view
        return view

    def launch(self, requests):
        """Start one decoder step for ``requests`` (``(window, tok, cur_time, pos, uniform)`` tuples of live songs) and
        return a ticket for ``fetch``.  The GPU work and the logits download run asynchronously."""
        t0 = time.perf_counter()
        hi = max(w.row0 + w.rows for w, *_ in requests)
        if self.graph:
            hi = min(self.rows, -(-hi // 8) * 8)             # few distinct shapes -> few graphs
        if self.device.type == "cuda":
            if self._pin_in is None:                 # pinned staging buffer: the upload must not wait for queued kernels
                self._pin_in = torch.zeros((3, self.rows), dtype=torch.int64).pin_memory()
            packed = self._pin_in.numpy()[:, :hi]
        else:
            packed = np.zeros((3, hi), np.int64)
        packed[0].fill(requests[0][0].pad)
        packed[1].fill(0)
        packed[2].fill(0)
        last = 1
        for w, tok, ct, pos, _ in requests:
            sl = slice(w.row0, w.row0 + w.rows)
            packed[0, sl], packed[1, sl], packed[2, sl] = tok, ct, pos
            last = max(last, int(pos.max()) + 1)
        t = torch.from_numpy(packed) if self.device.type != "cuda" else self._pin_in[:, :hi]
        t = t.to(self.device, non_blocking=True)
        t1 = time.perf_counter()
        if self.graph:
            bucket = min(self.model.cfg.max_len, -(-last // 64) * 64)
            g = self._graphs.get((hi, bucket))
            if g is None:
                g = self._graphs[(hi, bucket)] = _Graph(self, hi, bucket, t)
            g.inp.copy_(t)
            g.graph.replay()
            out = g.out
        else:
            with self.engine.autocast():
                out = self.step_fn(self.model, t[0], t[1], self._view(hi), t[2], last=last)
        if self.device.type == "cuda":
            if self._host is None:
                self._host = torch.empty((self.rows, out.shape[-1]), dtype=torch.float32).pin_memory()
            host = self._host[:hi]
            host.copy_(out, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host, event = out.float().cpu(), None
        self.timing["prep"] += t1 - t0
        self.timing["forward"] += time.perf_counter() - t1
        self.steps += 1
        self.step_rows += hi
        return requests, host, event

    def fetch(self, ticket):
        """Wait for a ``launch`` and return the per-request logits arrays (private copies)."""
        requests, host, event = ticket
        t = time.perf_counter()
        if event is not None:
            event.synchronize()
        out = host.numpy().copy()
        self.timing["fetch"] += time.perf_counter() - t
        return [out[w.row0:w.row0 + w.rows] for w, *_ in requests]

    def step_many(self, requests):
        return self.fetch(self.launch(requests))


class LaneBackend:
    """The rows ``[row0, row0 + nrows)`` of a pool, owned by one song (same interface as ``generate.LegacyBackend``)."""
    def __init__(self, pool, row0, nrows, pad):
        self.pool, self.row0, self.nrows, self.pad = pool, row0, nrows, pad

    def begin_window(self, enc, prefixes, K, song=None):
        pool, model, device = self.pool, self.pool.model, self.pool.device
        R = len(prefixes)
        n = K * R
        if n > self.nrows:
            raise RuntimeError(f"window needs {n} rows, lane has {self.nrows}")
        r0 = self.row0
        tok = torch.as_tensor(np.stack([p["tokens"] for p in prefixes]), dtype=torch.long, device=device)
        ct = torch.as_tensor(np.stack([p["cur_time"] for p in prefixes]), dtype=torch.long, device=device)
        with pool.engine.autocast():
            # cross-attention K/V of this song's window, broadcast to all of its rows
            for l, blk in enumerate(model.dec_blocks):
                k, v = blk.ca.make_kv(enc)
                pool.ck[l][r0:r0 + n] = k.to(pool.dt)
                pool.cv[l][r0:r0 + n] = v.to(pool.dt)
            cache = DecodeCache([a[r0:r0 + R] for a in pool.k], [a[r0:r0 + R] for a in pool.v],
                                [a[r0:r0 + R] for a in pool.ck], [a[r0:r0 + R] for a in pool.cv])
            if pool.sk is not None:
                if song is not None:
                    S = int(song["mem"].shape[1])
                    if S > pool.song_cap:
                        raise RuntimeError(f"song memory has {S} tokens, pool slots hold {pool.song_cap}")
                    for l, blk in enumerate(model.dec_blocks):
                        if blk.sca is not None:
                            sk, sv = blk.sca.make_kv(song["mem"][:1])
                            pool.sk[l][r0:r0 + n] = 0
                            pool.sv[l][r0:r0 + n] = 0
                            pool.sk[l][r0:r0 + n, :, :S] = sk.to(pool.dt)
                            pool.sv[l][r0:r0 + n, :, :S] = sv.to(pool.dt)
                    pool.smask[r0:r0 + n] = False
                    pool.smask[r0:r0 + n, :, :, :S] = song["mask"][:1]
                    pool.skeep[r0:r0 + n] = song["keep"][:1].to(pool.dt)
                    cache.sk = [None if a is None else a[r0:r0 + R] for a in pool.sk]
                    cache.sv = [None if a is None else a[r0:r0 + R] for a in pool.sv]
                    cache.smask, cache.skeep = pool.smask[r0:r0 + R], pool.skeep[r0:r0 + R]
                else:                            # no song context for this window: keep 0 -> the song path contributes exactly 0
                    pool.smask[r0:r0 + n] = True
                    pool.skeep[r0:r0 + n] = 0
            logits0 = model.prefill(tok, ct, cache).float().cpu().numpy()
        L = cache.len
        if K > 1:
            for l in range(len(pool.k)):
                pool.k[l][r0 + R:r0 + n, :, :L] = pool.k[l][r0:r0 + R, :, :L].repeat(K - 1, 1, 1, 1)
                pool.v[l][r0 + R:r0 + n, :, :L] = pool.v[l][r0:r0 + R, :, :L].repeat(K - 1, 1, 1, 1)
        if cache.g is not None:
            pool.g[r0:r0 + n] = cache.g.to(pool.dt).repeat(K, 1)
        w = gen._Window(self, device, logits0, L, K, R, row0=r0)
        w.pad = self.pad
        return w

    def step(self, window, tok, ct, pos, uniform):          # single-song use of the pooled kernel path
        return self.pool.step_many([(window, tok, ct, pos, uniform)])[0]

    def end_window(self, window):
        pass


def run_pool(engine, jobs, *, batch=4, groups=1, log=None, on_result=None, graph=False, fast=True):
    """Generate ``jobs`` (dicts: audio, options, out, title, artist) with ``batch`` songs per group in flight.

    ``groups`` > 1 pipelines the GPU with the Python work: each group owns a KV pool and its own decoder launches, and
    while the GPU runs one group's step the CPU advances the songs of the other groups.
    Returns ``(results, stats)``; ``results`` is aligned with ``jobs``: ``{"info", "error", "seconds"}``.  A failing song
    never stops the others.  ``on_result(index, result)`` is called as soon as a song is done.
    """
    quiet = lambda *a, **k: None
    log = log or quiet
    needs = [engine.rows_needed(j["options"]) for j in jobs]
    biggest = sorted(needs, reverse=True)[:max(1, batch)]
    pools = [PoolBackend(engine, sum(biggest), graph=graph, fast=fast) for _ in range(max(1, groups))]
    pad = engine.V.PAD
    results = [None] * len(jobs)
    queue = deque(range(len(jobs)))
    active = {}                                   # job index -> [generator, lane backend, pending request, started, pool]
    members = [set() for _ in pools]              # job indices per group
    tickets = [None] * len(pools)                 # (live [(job, request)], ticket) per group

    def finish(i, info=None, error=None):
        entry = active.pop(i)
        entry[4].release(entry[1].row0, entry[1].nrows)
        members[pools.index(entry[4])].discard(i)
        results[i] = dict(info=info, error=error, seconds=time.perf_counter() - entry[3])
        if on_result:
            on_result(i, results[i])

    def advance(i, value):
        entry = active[i]
        try:
            entry[2] = entry[0].send(value) if value is not None else next(entry[0])
        except StopIteration as stop:
            entry[2] = None
            finish(i, info=stop.value)
        except Exception as exc:                  # isolate per-song failures
            entry[2] = None
            entry[0].close()
            finish(i, error=f"{type(exc).__name__}: {exc}")

    def start_jobs():
        while queue:
            order = sorted(range(len(pools)), key=lambda g: len(members[g]))
            for g in order:
                if len(members[g]) >= batch:
                    continue
                i = queue[0]
                row0 = pools[g].alloc(needs[i])
                if row0 is None:
                    continue
                queue.popleft()
                job = jobs[i]
                lane = LaneBackend(pools[g], row0, needs[i], pad)
                try:
                    generator = engine.generate_co(job["audio"], job["options"], out=job.get("out"),
                                                   title=job.get("title"), artist=job.get("artist", ""),
                                                   log=log, backend=lane)
                except Exception as exc:
                    pools[g].release(row0, needs[i])
                    results[i] = dict(info=None, error=f"{type(exc).__name__}: {exc}", seconds=0.0)
                    if on_result:
                        on_result(i, results[i])
                    break
                active[i] = [generator, lane, None, time.perf_counter(), pools[g]]
                members[g].add(i)
                advance(i, None)
                break
            else:
                return                           # no group can take the next job right now

    with torch.inference_mode():
        start_jobs()
        while queue or active:
            for g, pool in enumerate(pools):
                if tickets[g] is not None:
                    live, ticket = tickets[g]
                    tickets[g] = None
                    for (i, _), out in zip(live, pool.fetch(ticket)):
                        if i in active:
                            active[i][2] = None
                            advance(i, out)
                start_jobs()
                live = [(i, active[i][2]) for i in sorted(members[g]) if active[i][2] is not None]
                if live:
                    tickets[g] = (live, pool.launch([r for _, r in live]))
    timing = {k: round(sum(p.timing[k] for p in pools), 2) for k in pools[0].timing}
    return results, dict(steps=sum(p.steps for p in pools),
                         mean_rows=round(sum(p.step_rows for p in pools) / max(sum(p.steps for p in pools), 1), 1),
                         graphs=sum(len(p._graphs) for p in pools), **timing)
